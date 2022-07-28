from .trainer import Trainer
import latentsafesets.utils.spb_utils as spbu
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os

log = logging.getLogger("dyn train")


class PETSDynamicsTrainer(Trainer):
    def __init__(self, params, dynamics, loss_plotter):
        self.params = params
        self.dynamics = dynamics
        self.loss_plotter = loss_plotter

        self.ensemble = params['dyn_n_models']#5 by default

        self.env_name = params['env']#spb/reacher/push

    def initial_train(self, replay_buffer, update_dir):
        if self.dynamics.trained:
            self.visualize(os.path.join(update_dir, "dyn_start.gif"), replay_buffer)
            return

        log.info('Beginning dynamics initial optimization')

        for i in range(self.params['dyn_init_iters']):#10000
            out_dict = replay_buffer.sample(self.params['dyn_batch_size'],#256
                                            ensemble=self.ensemble)
            obs, next_obs, act = out_dict['obs'], out_dict['next_obs'], out_dict['action']

            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)

            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:#100
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:#500
                log.info('Creating dynamics visualization')
                self.loss_plotter.plot()

                self.visualize(os.path.join(update_dir, "dyn%d.gif" % i), replay_buffer)

            if i % self.params['checkpoint_freq'] == 0 and i > 0:#2000
                self.dynamics.save(os.path.join(update_dir, 'dynamics_%d.pth' % i))

        self.dynamics.save(os.path.join(update_dir, 'dyn.pth'))

    def update(self, replay_buffer, update_dir):#this's for update0/1... after init train
        log.info('Beginning dynamics optimization')

        for _ in trange(self.params['dyn_update_iters']):#512
            out_dict = replay_buffer.sample(self.params['dyn_batch_size'],
                                            ensemble=self.ensemble)
            obs, next_obs, act = out_dict['obs'], out_dict['next_obs'], out_dict['action']

            loss, info = self.dynamics.update(obs, next_obs, act, already_embedded=True)
            self.loss_plotter.add_data(info)#the update is just the dynamics update

        log.info('Creating dynamics heatmap')
        self.loss_plotter.plot()
        self.visualize(os.path.join(update_dir, "dyn.gif"), replay_buffer)
        self.dynamics.save(os.path.join(update_dir, 'dyn.pth'))

    def visualize(self, file, replay_buffer):
        out_dict = replay_buffer.sample_chunk(8, 10)

        obs = out_dict['obs']
        act = out_dict['action']
        pu.visualize_dynamics(obs, act, self.dynamics, self.dynamics.encoder, file)
