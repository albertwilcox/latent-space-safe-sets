from latentsafesets.rl_trainers.safe_set_trainer import SafeSetTrainer
import latentsafesets.utils as utils
from latentsafesets.utils import LossPlotter
from latentsafesets.utils.arg_parser import parse_args

import logging
import os
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

    modules = utils.make_modules(params, ss=True)
    encoder = modules['enc']
    safe_set = modules['ss']

    replay_buffer = utils.load_replay_buffer(params, encoder)

    loss_plotter = LossPlotter(logdir)

    trainer = SafeSetTrainer(env, params, safe_set, loss_plotter)
    trainer.initial_train(replay_buffer, logdir)
