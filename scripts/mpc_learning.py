from latentsafesets.policy import CEMSafeSetPolicy
import latentsafesets.utils as utils
import latentsafesets.utils.plot_utils as pu
from latentsafesets.utils.arg_parser import parse_args
from latentsafesets.rl_trainers import MPCTrainer

import os
import logging
from tqdm import trange
import numpy as np
import pprint
log = logging.getLogger("main")


if __name__ == '__main__':
    params = parse_args()
    # Misc preliminaries

    utils.seed(params['seed'])
    logdir = params['logdir']
    os.makedirs(logdir)
    utils.init_logging(logdir)
    log.info('Training safe set MPC with params...')
    log.info(pprint.pformat(params))
    logger = utils.EpochLogger(logdir)

    env = utils.make_env(params)

    # Setting up encoder

    modules = utils.make_modules(params, ss=True, val=True, dyn=True, gi=True, constr=True)

    encoder = modules['enc']
    safe_set = modules['ss']
    dynamics_model = modules['dyn']
    value_func = modules['val']
    constraint_function = modules['constr']
    goal_indicator = modules['gi']

    # Populate replay buffer

    replay_buffer = utils.load_replay_buffer(params, encoder)

    trainer = MPCTrainer(env, params, modules)

    trainer.initial_train(replay_buffer)

    log.info("Creating policy")
    policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                              constraint_function, goal_indicator, params)

    num_updates = params['num_updates']
    traj_per_update = params['traj_per_update']

    losses = {}
    avg_rewards = []
    std_rewards = []
    all_rewards = []
    constr_viols = []
    task_succ = []
    n_episodes = 0

    for i in range(num_updates):
        update_dir = os.path.join(logdir, "update_%d" % i)
        os.makedirs(update_dir)
        update_rewards = []

        # Collect Data
        for j in range(traj_per_update):
            log.info("Collecting trajectory %d for update %d" % (j, i))
            transitions = []

            obs = np.array(env.reset())
            policy.reset()
            done = False

            # Maintain ground truth info for plotting purposes
            movie_traj = [{'obs': obs.reshape((-1, 3, 64, 64))[0]}]
            traj_rews = []
            constr_viol = False
            succ = False
            for k in trange(params['horizon']):
                action = policy.act(obs / 255)
                next_obs, reward, done, info = env.step(action)
                next_obs = np.array(next_obs)
                movie_traj.append({'obs': next_obs.reshape((-1, 3, 64, 64))[0]})
                traj_rews.append(reward)

                constr = info['constraint']

                transition = {'obs': obs, 'action': action, 'reward': reward,
                              'next_obs': next_obs, 'done': done,
                              'constraint': constr, 'safe_set': 0, 'on_policy': 1}
                transitions.append(transition)
                obs = next_obs
                constr_viol = constr_viol or info['constraint']
                succ = succ or reward == 0

                if done:
                    break
            transitions[-1]['done'] = 1
            traj_reward = sum(traj_rews)

            logger.store(EpRet=traj_reward, EpLen=k+1, EpConstr=float(constr_viol))
            all_rewards.append(traj_rews)
            constr_viols.append(constr_viol)
            task_succ.append(succ)

            pu.make_movie(movie_traj, file=os.path.join(update_dir, 'trajectory%d.gif' % j))

            log.info('    Cost: %d' % traj_reward)

            in_ss = 0
            rtg = 0
            for transition in reversed(transitions):
                if transition['reward'] > -1:
                    in_ss = 1
                transition['safe_set'] = in_ss
                transition['rtg'] = rtg

                rtg = rtg + transition['reward']

            replay_buffer.store_transitions(transitions)
            update_rewards.append(traj_reward)

        mean_rew = float(np.mean(update_rewards))
        std_rew = float(np.std(update_rewards))
        avg_rewards.append(mean_rew)
        std_rewards.append(std_rew)

        log.info('Iteration %d average reward: %.4f' % (i, mean_rew))
        pu.simple_plot(avg_rewards, std=std_rewards, title='Average Rewards',
                       file=os.path.join(logdir, 'rewards.pdf'),
                       ylabel='Average Reward', xlabel='# Training updates')

        logger.log_tabular('Epoch', i)
        logger.log_tabular('TrainEpisodes', n_episodes)
        logger.log_tabular('TestEpisodes', traj_per_update)
        logger.log_tabular('EpRet')
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('EpConstr', average_only=True)
        logger.log_tabular('ConstrRate', np.mean(constr_viols))
        logger.log_tabular('SuccRate', np.mean(task_succ))
        logger.dump_tabular()
        n_episodes += traj_per_update

        # Update models

        trainer.update(replay_buffer, i)

        np.save(os.path.join(logdir, 'rewards.npy'), all_rewards)
        np.save(os.path.join(logdir, 'constr.npy'), constr_viols)
