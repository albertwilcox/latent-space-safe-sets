
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')

import latentsafesets.utils as utils

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='CEM Learning Args')
    parser.add_argument('--env', type=str, default='spb')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed')
    parser.add_argument('--log_freq', type=int, default=100,
                        help='How frequently to log updates')
    parser.add_argument('--plot_freq', type=int, default=500,
                        help='How frequently to produce plots')
    parser.add_argument('--checkpoint_freq', type=int, default=2000,
                        help='How frequently to save model checkpoints')
    parser.add_argument('--checkpoint_folder', type=str, default=None)
    parser.add_argument('--traj_per_update', default=10, type=int)
    parser.add_argument('--num_updates', type=int, default=25)#45)#40)#35)#5)#20)#15)#10)#30)#the default is 25#
    parser.add_argument('--exper_name', type=str, default=None)

    add_controller_args(parser)
    add_encoder_args(parser)
    add_ss_args(parser)
    add_dyn_args(parser)
    add_val_args(parser)
    add_constr_args(parser)
    add_gi_args(parser)
    add_cbfd_args(parser)

    params = vars(parser.parse_args())

    add_env_options(params)
    add_checkpoint_options(params)

    params['logdir'] = utils.get_file_prefix(exper_name=params['exper_name'], seed=params['seed'])

    return params


def add_controller_args(parser):
    # Controller params#JC: Doing CEM!
    parser.add_argument('--num_candidates', type=int, default=1000,#1000,#No. of candidate trajectories
                        help='Number of cantidates for CEM')
    parser.add_argument('--num_elites', type=int, default=100,#No. of elite trajectories
                        help='Number of elites for CEM')
    parser.add_argument('--n_particles', type=int, default=20)#sample one trajectory that much times!
    parser.add_argument('--plan_hor', type=int, default=5,#The plan horizon in MPC, right?
                        help='How many steps into the future to look when planning')
    parser.add_argument('--max_iters', type=int, default=5,#
                        help='How many CEM iterations')
    parser.add_argument('--random_percent', type=float, default=1.0,#1 is default#0,#you know this in the paper!
                        help='How many CEM candidates should be sampled from a new distribution')


def add_encoder_args(parser):
    # Latent embedding params
    parser.add_argument('--d_latent', type=int, default=32,#default 32
                        help='Size of latent space embedding')
    parser.add_argument('--enc_lr', type=float, default=1e-4,
                        help='Learning rate of encoder/decoder')
    parser.add_argument('--enc_kl_multiplier', type=float, default=1e-6,
                        help='Multiplier for KL loss in encoder optimization')
    parser.add_argument('--enc_batch_size', type=int, default=256,
                        help='How many states to sample for each embedding update')
    parser.add_argument('--enc_init_iters', type=int, default=100000,
                        help='Initial training iterations')
    parser.add_argument('--enc_checkpoint', type=str, default='/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets/outputs/2022-07-13/17-24-59/vae.pth',
                        #'outputs/2022-07-18/19-38-58/vae.pth',#'/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets/outputs/2022-07-13/17-24-59/vae.pth',#
                        help='File to load a CEM model from')#None,
    parser.add_argument('--enc_data_aug', action='store_true')


def add_ss_args(parser):
    # General safe set params
    parser.add_argument('--safe_set_thresh', type=float, default=0.8)
    parser.add_argument('--safe_set_thresh_mult', type=float, default=0.8)
    parser.add_argument('--safe_set_thresh_mult_iters', type=int, default=5)#8)#16)#16 is crazy, but fails#
    parser.add_argument('--safe_set_type', type=str, default='bellman')
    parser.add_argument('--safe_set_batch_size', type=int, default=256,
                        help='Batch size for safe set learning')
    parser.add_argument('--safe_set_bellman_coef', type=float, default=0.3)#0.8)#0.3 is original#0.9)#
    parser.add_argument('--safe_set_bellman_reduction', type=str, default='max',
                        choices=('add', 'max'))
    parser.add_argument('--safe_set_ensemble', action='store_true')
    parser.add_argument('--safe_set_n_models', type=int, default=5)
    parser.add_argument('--safe_set_init_iters', type=int, default=10000,
                        help='Initial training iterations')
    parser.add_argument('--safe_set_ignore', action='store_true')
    parser.add_argument('--safe_set_update_iters', type=int, default=512)
    parser.add_argument('--safe_set_checkpoint', type=str, default='outputs/2022-07-15/17-41-16/initial_train/ss.pth')#'outputs/2022-08-07/01-56-19/update_3/ss.pth')#'outputs/2022-08-07/01-36-19/update_4/ss.pth')#'outputs/2022-08-07/01-09-48/update_5/ss.pth')#
    # 'outputs/2022-08-06/22-42-02/update_13/ss.pth')#'outputs/2022-07-20/14-46-50/update_16/ss.pth')#None)#
    # #'outputs/2022-07-15/17-41-16/initial_train/ss.pth')#None)#'outputs/2022-07-18/22-58-04/initial_train/ss.pth')#
    # #'/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets/outputs/2022-07-15/17-41-16/initial_train/ss.pth')#

    # BC Safe set params
    parser.add_argument('--bc_lr', type=float, default=1e-4, help='Learning rate for safe set')
    parser.add_argument('--bc_hidden_size', type=int, default=200)
    parser.add_argument('--bc_n_hidden', type=int, default=3)


def add_dyn_args(parser):
    # Dynamics model params
    parser.add_argument('--dyn_lr', type=float, default=1e-3,
                        help='Learning rate of dynamics model')
    parser.add_argument('--dyn_batch_size', type=int, default=256)
    parser.add_argument('--dyn_n_models', type=int, default=5,
                        help='How many models in the dynamics ensemble')
    parser.add_argument('--dyn_n_layers', type=int, default=3)
    parser.add_argument('--dyn_size', type=int, default=128)
    parser.add_argument('--dyn_normalize_delta', action='store_true')
    parser.add_argument('--dyn_init_iters', type=int, default=10000,
                        help='Initial training iterations')
    parser.add_argument('--dyn_update_iters', type=int, default=512)
    parser.add_argument('--dyn_checkpoint', type=str, default='outputs/2022-07-15/17-41-16/initial_train/dyn.pth')#'outputs/2022-08-07/01-56-19/update_3/dyn.pth')#'outputs/2022-08-07/01-36-19/update_4/dyn.pth')#'outputs/2022-08-07/01-09-48/update_5/dyn.pth')#
    # 'outputs/2022-08-06/22-42-02/update_13/dyn.pth')#'outputs/2022-07-20/14-46-50/update_16/dyn.pth')#
    #'outputs/2022-07-18/22-58-04/initial_train/dyn.pth')#None)#'/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets/outputs/2022-07-15/17-41-16/initial_train/dyn.pth')#


def add_val_args(parser):
    # Value function params
    parser.add_argument('--val_lr', type=float, default=1e-4,
                        help='Learning rate for value network')
    parser.add_argument('--val_targ_update_freq', type=float, default=100,
                        help='How frequently to update the value function target network')
    parser.add_argument('--val_targ_update_rate', type=float, default=1.0,
                        help='How much to update the value target (value in (0,1])')
    parser.add_argument('--val_discount', type=float, default=0.99,
                        help='Discount in value approximation bellman eqtns')
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--val_hidden_size', type=int, default=200)
    parser.add_argument('--val_n_hidden', type=int, default=3)
    parser.add_argument('--val_ensemble', action='store_true')
    parser.add_argument('--val_n_models', type=int, default=5)
    parser.add_argument('--val_init_iters', type=int, default=10000,
                        help='Initial training iterations')
    parser.add_argument('--val_reduction', type=str, default='mean')
    parser.add_argument('--val_update_iters', type=int, default=2000)
    parser.add_argument('--val_checkpoint', type=str, default='outputs/2022-07-15/17-41-16/initial_train/val.pth')#'outputs/2022-08-07/01-56-19/update_3/val.pth')#'outputs/2022-08-07/01-36-19/update_4/val.pth')#'outputs/2022-08-07/01-09-48/update_5/val.pth')#
    # 'outputs/2022-08-06/22-42-02/update_13/val.pth')#'outputs/2022-07-20/14-46-50/update_16/val.pth')#
    #'outputs/2022-07-18/22-58-04/initial_train/val.pth')#None)#'/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets/outputs/2022-07-15/17-41-16/initial_train/val.pth')#


def add_constr_args(parser):
    # Constraint Estimator params
    parser.add_argument('--constr_lr', type=float, default=1e-4,
                        help='Learning rate for value network')
    parser.add_argument('--constr_hidden_size', type=int, default=200)
    parser.add_argument('--constr_n_hidden', type=int, default=3)
    parser.add_argument('--constr_batch_size', type=int, default=256)
    parser.add_argument('--constr_thresh', type=float, default=0.2,
                        help='Threshold for an obs to be considered in violation of constraints')
    parser.add_argument('--constr_init_iters', type=int, default=10000,
                        help='Initial training iterations')
    parser.add_argument('--constr_ignore', action='store_true')
    parser.add_argument('--constr_update_iters', type=int, default=512)
    parser.add_argument('--constr_checkpoint', type=str, default='outputs/2022-07-15/17-41-16/initial_train/constr.pth')#'outputs/2022-08-07/01-56-19/update_3/constr.pth')#'outputs/2022-08-07/01-36-19/update_4/constr.pth')#'outputs/2022-08-07/01-09-48/update_5/constr.pth')#
    # 'outputs/2022-08-06/22-42-02/update_13/constr.pth')#None)#'outputs/2022-07-20/14-46-50/update_16/constr.pth')#
    #'outputs/2022-07-18/22-58-04/initial_train/constr.pth')#'/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets/outputs/2022-07-15/17-41-16/initial_train/constr.pth')#


def add_gi_args(parser):
    # Goal Indicator params
    parser.add_argument('--gi_lr', type=float, default=1e-4,
                        help='Learning rate for value network')
    parser.add_argument('--gi_hidden_size', type=int, default=200)
    parser.add_argument('--gi_n_hidden', type=int, default=3)
    parser.add_argument('--gi_batch_size', type=int, default=256)
    parser.add_argument('--gi_thresh', type=float, default=0.5,
                        help='Threshold for an obs to be considered in violation of constraints')
    parser.add_argument('--gi_init_iters', type=int, default=10000,
                        help='Initial training iterations')
    parser.add_argument('--gi_update_iters', type=int, default=512)
    parser.add_argument('--gi_checkpoint', type=str, default='outputs/2022-07-15/17-41-16/initial_train/gi.pth')#'outputs/2022-08-07/01-56-19/update_3/gi.pth')#'outputs/2022-08-07/01-36-19/update_4/gi.pth')#'outputs/2022-08-07/01-09-48/update_5/gi.pth')#
    # 'outputs/2022-08-06/22-42-02/update_13/gi.pth')#'outputs/2022-07-20/14-46-50/update_16/gi.pth')#
    #'outputs/2022-07-18/22-58-04/initial_train/gi.pth')#None)#'/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets/outputs/2022-07-15/17-41-16/initial_train/gi.pth')#

def add_cbfd_args(parser):
    # Constraint Estimator params
    parser.add_argument('--cbfdot_thresh', type=float, default=3.2)#0.8)#
    parser.add_argument('--cbfdot_thresh_mult', type=float, default=1.25)#0.8)#
    parser.add_argument('--cbfd_lr', type=float, default=1e-4,
                        help='Learning rate for cbfd network')
    parser.add_argument('--cbfd_hidden_size', type=int, default=200)
    parser.add_argument('--cbfd_n_hidden', type=int, default=3)
    parser.add_argument('--cbfd_batch_size', type=int, default=256)
    parser.add_argument('--cbfd_thresh', type=float, default=0.2,
                        help='Threshold for an obs to be considered in violation of cbf dots')
    parser.add_argument('--cbfd_init_iters', type=int, default=320000,#160000,#80000,#40000,#20000,#30000,#10000,#
                        help='Initial training iterations')
    parser.add_argument('--cbfd_ignore', action='store_true')
    parser.add_argument('--cbfd_update_iters', type=int, default=512)
    parser.add_argument('--cbfd_checkpoint', type=str, default=None)#'outputs/2022-08-06/15-02-09/cbfd_180000.pth')#'outputs/2022-08-07/01-56-19/update_3/cbfd.pth')#'outputs/2022-08-07/01-36-19/update_4/cbfd.pth')#'outputs/2022-08-07/01-09-48/update_5/cbfd.pth')#
    # 'outputs/2022-08-06/22-42-02/update_13/cbfd.pth')#'outputs/2022-08-06/11-44-04/cbfd.pth')#'outputs/2022-08-06/12-29-56/cbfd.pth')#
    #'outputs/2022-08-06/11-44-04/cbfd.pth')#'outputs/2022-08-06/10-21-50/cbfd.pth')#'outputs/2022-08-03/01-06-16/cbfd.pth')#
    #'outputs/2022-07-15/17-41-16/initial_train/constr.pth')#'outputs/2022-07-20/14-46-50/update_16/constr.pth')#
    #'outputs/2022-07-18/22-58-04/initial_train/constr.pth')#'/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets/outputs/2022-07-15/17-41-16/initial_train/constr.pth')#

def add_env_options(params):
    params['horizon'] = 100
    params['d_act'] = 2
    params['frame_stack'] = 1
    if params['env'] == 'spb':
        params['buffer_size'] = 35000
        params['data_dirs'] = [
            'SimplePointBot',
            'SimplePointBotConstraints',
        ]
        params['data_counts'] = [
            50,
            50,
        ]
        params['frame_stack'] = 1
    elif params['env'] == 'reacher':
        params['buffer_size'] = 25000
        params['data_dirs'] = [
            'Reacher',
            'ReacherConstraints',
            'ReacherInteractions',
        ]
        params['data_counts'] = [
            50, 50, 100
        ]
        params['frame_stack'] = 3
    elif params['env'] == 'push':
        params['buffer_size'] = 300000
        params['data_dirs'] = [
            'Push',
            'PushOutbursts2'
        ]
        params['data_counts'] = [
            500,
            300,
        ]

        params['frame_stack'] = 1
        params['horizon'] = 150
    if params['frame_stack'] == 1:
        params['d_obs'] = (3, 64, 64)#3 channel images with size 64*64!
    else:
        params['d_obs'] = (params['frame_stack'], 3, 64, 64)#multiple images!


def add_checkpoint_options(params):
    if params['checkpoint_folder'] is not None:
        if params['enc_checkpoint'] is None:
            params['enc_checkpoint'] = os.path.join(params['checkpoint_folder'], 'vae.pth')
            if not os.path.exists(params['enc_checkpoint']):
                params['enc_checkpoint'] = None
        if params['val_checkpoint'] is None:
            params['val_checkpoint'] = os.path.join(params['checkpoint_folder'], 'val.pth')
            if not os.path.exists(params['val_checkpoint']):
                params['val_checkpoint'] = None
        elif params['val_checkpoint'] == 'ignore':
            params['val_checkpoint'] = None
        if params['dyn_checkpoint'] is None:
            params['dyn_checkpoint'] = os.path.join(params['checkpoint_folder'], 'dyn.pth')
            if not os.path.exists(params['dyn_checkpoint']):
                params['dyn_checkpoint'] = None
        elif params['dyn_checkpoint'] == 'ignore':
            params['dyn_checkpoint'] = None
        if params['safe_set_checkpoint'] is None:
            params['safe_set_checkpoint'] = os.path.join(params['checkpoint_folder'], 'ss.pth')
            if not os.path.exists(params['safe_set_checkpoint']):
                params['safe_set_checkpoint'] = None
        elif params['safe_set_checkpoint'] == 'ignore':
            params['safe_set_checkpoint'] = None
        if params['constr_checkpoint'] is None:
            params['constr_checkpoint'] = os.path.join(params['checkpoint_folder'], 'constr.pth')
            if not os.path.exists(params['constr_checkpoint']):
                params['constr_checkpoint'] = None
        elif params['constr_checkpoint'] == 'ignore':
            params['constr_checkpoint'] = None
        if params['gi_checkpoint'] is None:
            params['gi_checkpoint'] = os.path.join(params['checkpoint_folder'], 'gi.pth')
            if not os.path.exists(params['gi_checkpoint']):
                params['gi_checkpoint'] = None
        elif params['gi_checkpoint'] == 'ignore':
            params['gi_checkpoint'] = None
