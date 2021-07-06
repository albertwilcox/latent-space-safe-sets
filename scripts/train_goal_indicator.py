from latentsafesets.rl_trainers import GoalIndicatorTrainer
import latentsafesets.utils as utils
from latentsafesets.utils import LossPlotter
from latentsafesets.utils.arg_parser import parse_args

import os
import logging
import pprint
log = logging.getLogger("main")


if __name__ == '__main__':
    params = parse_args()

    logdir = params['logdir']
    os.makedirs(logdir)
    utils.init_logging(logdir)

    utils.seed(params['seed'])
    log.info('Training gi with params...')
    log.info(pprint.pformat(params))

    env = utils.make_env(params)

    modules = utils.make_modules(params, gi=True)
    encoder = modules['enc']
    goal_indicator = modules['gi']

    replay_buffer = utils.load_replay_buffer(params, encoder)

    loss_plotter = LossPlotter(logdir)

    trainer = GoalIndicatorTrainer(env, params, goal_indicator, loss_plotter)
    trainer.initial_train(replay_buffer, logdir)
