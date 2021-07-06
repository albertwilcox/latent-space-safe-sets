import latentsafesets.utils as utils
from latentsafesets.utils import LossPlotter
from latentsafesets.utils.arg_parser import parse_args
from latentsafesets.rl_trainers import PETSDynamicsTrainer

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

    modules = utils.make_modules(params, dyn=True)
    encoder = modules['enc']
    dynamics = modules['dyn']

    replay_buffer = utils.load_replay_buffer(params, encoder)

    loss_plotter = LossPlotter(logdir)

    trainer = PETSDynamicsTrainer(params, dynamics, loss_plotter)
    trainer.initial_train(replay_buffer, logdir)
