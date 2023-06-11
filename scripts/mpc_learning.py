
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')

from latentsafesets.policy import CEMSafeSetPolicy#this is the class!
import latentsafesets.utils as utils
import latentsafesets.utils.plot_utils as pu
from latentsafesets.utils.arg_parser import parse_args
from latentsafesets.rl_trainers import MPCTrainer
#import latentsafesets.utils.pytorch_utils as ptu

import os
import logging
from tqdm import trange#mainly for showing the progress bar
import numpy as np
import pprint
#provides a capability to “pretty-print” arbitrary Python data structures in a form that can be used as input to the interpreter
log = logging.getLogger("main")#some logging stuff


if __name__ == '__main__':
    params = parse_args()#get the parameters from parse_args, see arg_parser.py
    # Misc preliminaries

    utils.seed(params['seed'])#around line 10, the default is -1, meaning random seed
    logdir = params['logdir']#around line 35
    os.makedirs(logdir)#e.g.: 'outputs/2022-07-15/17-41-16'
    utils.init_logging(logdir)#record started!
    log.info('Training safe set MPC with params...')#at the very very start
    log.info(pprint.pformat(params))#just to pretty print all the parameters!
    logger = utils.EpochLogger(logdir)#a kind of dynamic logger?

    env = utils.make_env(params)#spb, reacher, etc.#around line 148 in utils
    #The result is to have env=SimplePointBot in spb
    # Setting up encoder, around line 172 in utils, get all the parts equipped!

    modules = utils.make_modules(params, ss=True, val=True, dyn=True, gi=True, constr=True)
    #modules = utils.make_modulessafety(params, ss=True, val=True, dyn=True, gi=True, constr=True, cbfd=True)
    #the result is to set up the encoder, etc.
    encoder = modules['enc']#it is a value in a dictionary, uh?
    safe_set = modules['ss']
    dynamics_model = modules['dyn']
    value_func = modules['val']
    constraint_function = modules['constr']
    goal_indicator = modules['gi']
    #cbfdot_function = modules['cbfd']

    # Populate replay buffer
    #the following is loading replay buffer, rather than loading trajectories
    replay_buffer = utils.load_replay_buffer(params, encoder)#around line 123 in utils.py

    trainer = MPCTrainer(env, params, modules)#so that we can train MPC!

    trainer.initial_train(replay_buffer)#initialize all the parts!

    log.info("Creating policy")
    policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                              constraint_function, goal_indicator, params)
    #policy = CEMSafeSetPolicy(env, encoder, safe_set, value_func, dynamics_model,
                              #constraint_function, goal_indicator, cbfdot_function, params)

    num_updates = params['num_updates']#default 25
    traj_per_update = params['traj_per_update']#default 10

    losses = {}
    avg_rewards = []
    std_rewards = []
    all_rewards = []
    constr_viols = []
    task_succ = []
    n_episodes = 0

    for i in range(num_updates):#default 25 in spb
        update_dir = os.path.join(logdir, "update_%d" % i)#create the corresponding folder!
        os.makedirs(update_dir)#mkdir!
        update_rewards = []

        # Collect Data
        for j in range(traj_per_update):#default 10 in spb
            log.info("Collecting trajectory %d for update %d" % (j, i))
            transitions = []

            obs = np.array(env.reset())#the obs seems to be the observation as image rather than obstacle
            policy.reset()#self.mean, self.std = None, None
            done = False

            # Maintain ground truth info for plotting purposes
            movie_traj = [{'obs': obs.reshape((-1, 3, 64, 64))[0]}]#a dict
            traj_rews = []#rews: reward
            constr_viol = False
            succ = False
            for k in trange(params['horizon']):#default 100 in spb#This is MPC
                #print('obs.shape',obs.shape)(3,64,64)
                #print('env.state',env.state)#env.state [35.44344669 54.30340498]
                action = policy.act(obs / 255)#the CEM (candidates, elites, etc.) is in here
                #storch=ptu.torchify(env.state)#state torch
                #action = policy.actcbfd(obs/255,env.state)  # the CEM (candidates, elites, etc.) is in here
                next_obs, reward, done, info = env.step(action)#saRSa
                #next_obs, reward, done, info = env.stepsafety(action)  # 63 in simple_point_bot.py
                next_obs = np.array(next_obs)#to make this image a numpy array
                movie_traj.append({'obs': next_obs.reshape((-1, 3, 64, 64))[0]})#add this image
                traj_rews.append(reward)

                constr = info['constraint']#its use is seen a few lines later

                transition = {'obs': obs, 'action': action, 'reward': reward,#sARSa
                              'next_obs': next_obs, 'done': done,
                              'constraint': constr, 'safe_set': 0, 'on_policy': 1}
                '''
                transition = {'obs': obs, 'action': action, 'reward': reward,
                              'next_obs': next_obs, 'done': done,  # this is a dictionary
                              'constraint': constr, 'safe_set': 0,
                              'on_policy': 1,
                              'rdo': info['rdo'].tolist(),#rdo for relative distance old
                              'rdn': info['rdn'].tolist(),#rdn for relative distance new
                              'hvo': info['hvo'],#hvo for h value old
                              'hvn': info['hvn'],#hvn for h value new
                              'hvd': info['hvd'],#hvd for h value difference
                              'state': info['state'].tolist(),
                              'next_state': info['next_state'].tolist()
                              }  # add key and value into it!
                '''
                transitions.append(transition)
                obs = next_obs
                constr_viol = constr_viol or info['constraint']#a way to update constr_viol
                succ = succ or reward == 0#as said in the paper, reward=0 means success!

                if done:
                    break
            transitions[-1]['done'] = 1#change the last transition to success/done!
            traj_reward = sum(traj_rews)#total reward
            #EpRet is episode reward, EpLen=Episode Length, EpConstr=Episode constraints
            logger.store(EpRet=traj_reward, EpLen=k+1, EpConstr=float(constr_viol))
            all_rewards.append(traj_rews)#does it use any EpLen?
            constr_viols.append(constr_viol)#whether this 100-length traj violate any constraints
            task_succ.append(succ)
            #save the result in the gift form!
            pu.make_movie(movie_traj, file=os.path.join(update_dir, 'trajectory%d.gif' % j))

            log.info('    Cost: %d' % traj_reward)#see it in the terminal!

            in_ss = 0
            rtg = 0
            for transition in reversed(transitions):
                if transition['reward'] > -1:
                    in_ss = 1
                transition['safe_set'] = in_ss
                transition['rtg'] = rtg

                rtg = rtg + transition['reward']

            replay_buffer.store_transitions(transitions)#replay buffer online training
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
        n_episodes += traj_per_update#10 by default

        # Update models

        trainer.update(replay_buffer, i)#online training, right?

        np.save(os.path.join(logdir, 'rewards.npy'), all_rewards)
        np.save(os.path.join(logdir, 'constr.npy'), constr_viols)
