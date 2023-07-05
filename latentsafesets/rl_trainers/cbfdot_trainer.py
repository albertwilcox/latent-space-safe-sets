
#import sys
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/jianning/PycharmProjects/pythonProject6/latent-space-safe-sets')


import numpy as np

from .trainer import Trainer
import latentsafesets.utils.plot_utils as pu

import logging
from tqdm import trange
import os

log = logging.getLogger("cbfd train")


class CBFdotTrainer(Trainer):#modified from contraint trainer
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
            #rdo for relative distance old, hvd for h value difference
            #print('rdo',rdo)#seems reasonable##print(rdo.shape)#(256,2)
            #print('action',action)#seems reasonable##print(action.shape)#(256,2)
            rda=np.concatenate((rdo,action),axis=1)#rda for relative distance+action
            #print(rda.shape)#(256,4)
            #print('hvd',hvd)#seems reasonable##print(hvd.shape)#(256,)
            #rdn,hvo,hvn=out_dict['rdn'], out_dict['hvo'], out_dict['hvn']#
            #print('rdn',rdn)#seems reasonable#
            #print('hvo',hvo)#seems reasonable#
            #print('hvn',hvn)#seems reasonable#
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)#not the update below
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

    def update(self, replay_buffer, update_dir):#this is the update process after initial train
        log.info('Beginning cbf dott update optimization')

        for _ in trange(self.params['cbfd_update_iters']):
            out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
            #next_obs, constr = out_dict['next_obs'], out_dict['constraint']
            rdo, action, hvd = out_dict['rdo'], out_dict['action'], out_dict['hvd']  # 0 or 1
            rda = np.concatenate((rdo, action),axis=1)

            #loss, info = self.constr.update(next_obs, constr, already_embedded=True)
            loss, info = self.cbfd.update(rda, hvd, already_embedded=True)
            self.loss_plotter.add_data(info)

        log.info('Creating cbf dot function heatmap')
        self.loss_plotter.plot()
        self.plot(os.path.join(update_dir, "cbfd.pdf"), replay_buffer)#a few lines later
        self.cbfd.save(os.path.join(update_dir, 'cbfd.pth'))

    def plot(self, file, replay_buffer):
        out_dict = replay_buffer.sample(self.params['cbfd_batch_size'])
        next_obs = out_dict['next_obs']
        #rdo = out_dict['rdo']
        pu.visualize_cbfdot(next_obs, self.cbfd,
                             file,
                             env=self.env)
