
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')

from latentsafesets.rl_trainers import VAETrainer
import latentsafesets.utils as utils
from latentsafesets.utils import LossPlotter, EncoderDataLoader
from latentsafesets.utils.arg_parser import parse_args

import logging
import os
import pprint

os.environ['TORCH_CUDA_ARCH_LIST']="8.6"

log = logging.getLogger("main")


if __name__ == '__main__':

    params = parse_args()

    logdir = params['logdir']
    print(logdir)
    os.makedirs(logdir)
    utils.init_logging(logdir)

    utils.seed(params['seed'])
    log.info('Training value with params...')
    log.info(pprint.pformat(params))

    encoder_data_loader = EncoderDataLoader(params)

    modules = utils.make_modules(params)
    encoder = modules['enc']

    loss_plotter = LossPlotter(logdir)

    trainer = VAETrainer(params, encoder, loss_plotter)
    trainer.initial_train(encoder_data_loader, logdir, force_train=True)
