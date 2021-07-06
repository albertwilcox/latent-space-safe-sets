from latentsafesets.rl_trainers import ValueTrainer
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
    log.info('Training value with params...')
    log.info(pprint.pformat(params))

    env = utils.make_env(params)

    modules = utils.make_modules(params, val=True)
    encoder = modules['enc']
    value_func = modules['val']

    replay_buffer = utils.load_replay_buffer(params, encoder)

    loss_plotter = LossPlotter(logdir)

    value_trainer = ValueTrainer(env, params, value_func, loss_plotter)
    value_trainer.initial_train(replay_buffer, logdir)
