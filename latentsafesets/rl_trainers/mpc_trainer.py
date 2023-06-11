from latentsafesets.rl_trainers import VAETrainer, SafeSetTrainer, Trainer, ValueTrainer, ConstraintTrainer, GoalIndicatorTrainer, PETSDynamicsTrainer#, CBFdotTrainer

from latentsafesets.utils import LossPlotter, EncoderDataLoader

import os

import numpy as np

from .trainer import Trainer
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os

log = logging.getLogger("cbfd train")


class CBFdotTrainer(Trainer):
    def __init__(self, env, params, cbfd, loss_plotter):
        self.params = params
        self.cbfd =cbfd
        self.loss_plotter = loss_plotter
        self.env = env

        self.env_name = params['env']

    def initial_train(self, replay_buffer, update_dir):
        if self.cbfd.trained:
            self.plot(os.path.join(update_dir, "cbfd_start.pdf"), replay_buffer)
            return

        log.info('Beginning cbfdot initial optimization')

        for i in range(self.params['cbfd_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])#256
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']#0 or 1
            rda=np.concatenate((rdo,action),axis=1)
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            self.loss_plotter.add_data(info)#self.constr.update, not self.update!

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating cbfdot function heatmap')
                self.loss_plotter.plot()
                self.plot(os.path.join(update_dir, "cbfd%d.pdf" % i), replay_buffer)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.cbfd.save(os.path.join(update_dir, 'cbfd_%d.pth' % i))

        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def update(self, replay_buffer, update_dir):
        log.info('Beginning cbf dott update optimization')

        for _ in trange(self.params['cbfd_update_iters']):
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
            #next_obs, constr = out_dict['next_obs'], out_dict['constraint']
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']  # 0 or 1
            #print('rdo.shape',rdo.shape)#(256, 2)
            #print('action.shape',action.shape)#(256, 2)
            rda = np.concatenate((rdo, action),axis=1)
            #print('rda.shape',rda.shape)#(256, 4)
            #loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating cbf dot function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer)
        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']#rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)

class MPCTrainer(Trainer):

    def __init__(self, env, params, modules):

        self.params = params
        self.env = env

        self.logdir = params['logdir']

        loss_plotter = LossPlotter(os.path.join(params['logdir'], 'loss_plots'))
        self.encoder_data_loader = EncoderDataLoader(params)

        self.trainers = []#the following shows the sequence of training

        self.trainers.append(VAETrainer(params, modules['enc'], loss_plotter))
        self.trainers.append(PETSDynamicsTrainer(params, modules['dyn'], loss_plotter))
        self.trainers.append(ValueTrainer(env, params, modules['val'], loss_plotter))
        self.trainers.append(SafeSetTrainer(env, params, modules['ss'], loss_plotter))
        self.trainers.append(ConstraintTrainer(env, params, modules['constr'], loss_plotter))
        self.trainers.append(GoalIndicatorTrainer(env, params, modules['gi'], loss_plotter))
        #self.trainers.append(CBFdotTrainer(env, params, modules['cbfd'], loss_plotter))

    def initial_train(self, replay_buffer):#by default the replay buffer is the encoded version
        update_dir = os.path.join(self.logdir, 'initial_train')#create that folder!
        os.makedirs(update_dir, exist_ok=True)#mkdir is here!
        for trainer in self.trainers:#type() method returns class type of the argument(object) passed as parameter
            if type(trainer) == VAETrainer:#VAE is trained totally on images from that folder, no use of replay_buffer
                trainer.initial_train(self.encoder_data_loader, update_dir)
            else:#then it means that the VAE has been trained!
                trainer.initial_train(replay_buffer, update_dir)

    def update(self, replay_buffer, update_num):#the update folder!
        update_dir = os.path.join(self.logdir, 'update_%d' % update_num)
        os.makedirs(update_dir, exist_ok=True)
        for trainer in self.trainers:
            trainer.update(replay_buffer, update_dir)
