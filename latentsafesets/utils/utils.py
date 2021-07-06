import torch
import torch.nn as nn
import numpy as np

import logging
import os
import json
from datetime import datetime
import random
from tqdm import tqdm, trange

from latentsafesets.utils.replay_buffer_encoded import EncodedReplayBuffer
from latentsafesets.utils.replay_buffer import ReplayBuffer
from gym.wrappers import FrameStack

log = logging.getLogger("utils")


files = {
    'spb': [
        'SimplePointBot', 'SimplePointBotConstraints'
    ],
    'apb': [
        'AccelerationPointBot', 'AccelerationPointBotConstraint'
    ],
    'reacher': [
        'Reacher', 'ReacherConstraints', 'ReacherInteractions'
    ]
}


def seed(seed):
    # torch.set_deterministic(True)
    if seed == -1:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_file_prefix(exper_name=None, seed=-1):
    if exper_name is not None:
        folder = os.path.join('outputs', exper_name)
    else:
        now = datetime.now()
        date_string = now.strftime("%Y-%m-%d/%H-%M-%S")
        folder = os.path.join('outputs', date_string)
    if seed != -1:
        folder = os.path.join(folder, str(seed))
    return folder


def init_logging(folder, file_level=logging.INFO, console_level=logging.DEBUG):
    # set up logging to file
    logging.basicConfig(level=file_level,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        filename=os.path.join(folder, 'log.txt'),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(console_level)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def save_trajectories(trajectories, file):
    if not os.path.exists(file):
        os.makedirs(file)
    else:
        raise RuntimeError("Directory %s already exists." % file)

    for i, traj in enumerate(trajectories):
        save_trajectory(traj, file, i)


def save_trajectory(trajectory, file, n):
    im_fields = ('obs', 'next_obs')
    for field in im_fields:
        if field in trajectory[0]:
            dat = np.array([frame[field] for frame in trajectory], dtype=np.uint8)
            np.save(os.path.join(file, "%d_%s.npy" % (n, field)), dat)
    traj_no_ims = [{key: frame[key] for key in frame if key not in im_fields}
                   for frame in trajectory]
    with open(os.path.join(file, "%d.json" % n), "w") as f:
        json.dump(traj_no_ims, f)


def load_trajectories(num_traj, file):
    log.info('Loading trajectories from %s' % file)

    if not os.path.exists(file):
        raise RuntimeError("Could not find directory %s." % file)
    trajectories = []
    iterator = range(num_traj) if num_traj <= 200 else trange(num_traj)
    for i in iterator:
        if not os.path.exists(os.path.join(file, '%d.json' % i)):
            log.info('Could not find %d' % i)
            continue
        im_fields = ('obs', 'next_obs')
        with open(os.path.join(file, '%d.json' % i), 'r') as f:
            trajectory = json.load(f)
        im_dat = {}

        for field in im_fields:
            f = os.path.join(file, "%d_%s.npy" % (i, field))
            if os.path.exists(file):
                dat = np.load(f)
                im_dat[field] = dat.astype(np.uint8)

        for j, frame in list(enumerate(trajectory)):
            for key in im_dat:
                frame[key] = im_dat[key][j]
        trajectories.append(trajectory)

    return trajectories


def load_replay_buffer(params, encoder=None, first_only=False):
    log.info('Loading data')
    trajectories = []
    for directory, num in list(zip(params['data_dirs'], params['data_counts'])):
        real_dir = os.path.join('data', directory)
        trajectories += load_trajectories(num, file=real_dir)
        if first_only:
            print('wahoo')
            break

    log.info('Populating replay buffer')

    # Shuffle array so that when the replay fills up it doesn't remove one dataset before the other
    random.shuffle(trajectories)
    if encoder is not None:
        replay_buffer = EncodedReplayBuffer(encoder, params['buffer_size'])
    else:
        replay_buffer = ReplayBuffer(params['buffer_size'])

    for trajectory in tqdm(trajectories):
        replay_buffer.store_transitions(trajectory)

    return replay_buffer


def make_env(params, monitoring=False):
    from latentsafesets.envs import SimplePointBot, PushEnv, SimpleVideoSaver
    env_name = params['env']
    if env_name == 'spb':
        env = SimplePointBot(True)
    elif env_name == 'reacher':
        import dmc2gym

        env = dmc2gym.make(domain_name='reacher', task_name='hard', seed=params['seed'],
                           from_pixels=True, visualize_reward=False, channels_first=True)
    elif env_name == 'push':
        env = PushEnv()
    else:
        raise NotImplementedError

    if params['frame_stack'] > 1:
        env = FrameStack(env, params['frame_stack'])

    if monitoring:
        env = SimpleVideoSaver(env, os.path.join(params['logdir'], 'videos'))

    return env


def make_modules(params, ss=False, val=False, dyn=False,
                 gi=False, constr=False):
    from latentsafesets.modules import VanillaVAE, ValueEnsemble, \
        ValueFunction, PETSDynamics, GoalIndicator, ConstraintEstimator, BCSafeSet, \
        BellmanSafeSet
    import latentsafesets.utils.pytorch_utils as ptu

    modules = {}

    encoder = VanillaVAE(params)
    if params['enc_checkpoint']:
        encoder.load(params['enc_checkpoint'])
    modules['enc'] = encoder

    if ss:
        safe_set_type = params['safe_set_type']
        if safe_set_type == 'bc':
            safe_set = BCSafeSet(encoder, params)
        elif safe_set_type == 'bellman':
            safe_set = BellmanSafeSet(encoder, params)
        else:
            raise NotImplementedError
        if params['safe_set_checkpoint']:
            safe_set.load(params['safe_set_checkpoint'])
        modules['ss'] = safe_set

    if val:
        if params['val_ensemble']:
            value_func = ValueEnsemble(encoder, params).to(ptu.TORCH_DEVICE)
        else:
            value_func = ValueFunction(encoder, params).to(ptu.TORCH_DEVICE)
        if params['val_checkpoint']:
            value_func.load(params['val_checkpoint'])
        modules['val'] = value_func

    if dyn:
        dynamics = PETSDynamics(encoder, params)
        if params['dyn_checkpoint']:
            dynamics.load(params['dyn_checkpoint'])
        modules['dyn'] = dynamics

    if gi:
        goal_indicator = GoalIndicator(encoder, params).to(ptu.TORCH_DEVICE)
        if params['gi_checkpoint']:
            goal_indicator.load(params['gi_checkpoint'])
        modules['gi'] = goal_indicator

    if constr:
        constraint = ConstraintEstimator(encoder, params).to(ptu.TORCH_DEVICE)
        if params['constr_checkpoint']:
            constraint.load(params['constr_checkpoint'])
        modules['constr'] = constraint

    return modules


class RunningMeanStd(nn.Module):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        super(RunningMeanStd, self).__init__()

        from latentsafesets.utils.pytorch_utils import TORCH_DEVICE

        # We store these as parameters so they'll be stored in dynamic model state dicts
        self.mean = nn.Parameter(torch.zeros(shape, dtype=torch.float32, device=TORCH_DEVICE),
                                 requires_grad=False)
        self.var = nn.Parameter(torch.ones(shape, dtype=torch.float32, device=TORCH_DEVICE),
                                requires_grad=False)
        self.count = nn.Parameter(torch.tensor(epsilon))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + delta**2 * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = nn.Parameter(new_mean, requires_grad=False)
        self.var = nn.Parameter(new_var, requires_grad=False)
        self.count = nn.Parameter(new_count, requires_grad=False)

