from .trainer import Trainer
import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.modules import VanillaVAE

import torch
from torchvision.utils import save_image
import numpy as np

import logging
import os

log = logging.getLogger("dyn train")


class VAETrainer(Trainer):
    def __init__(self, params, vae: VanillaVAE, loss_plotter):
        self.params = params
        self.vae = vae
        self.loss_plotter = loss_plotter

        self.frame_stack = params['frame_stack']
        self.d_latent = params['d_latent']

    def initial_train(self, enc_data_loader, update_dir, force_train=False):
        if self.vae.trained and not force_train:
            return

        log.info('Beginning vae initial optimization')

        for i in range(self.params['enc_init_iters']):
            obs = enc_data_loader.sample(self.params['enc_batch_size'])

            loss, info = self.vae.update(obs)
            self.loss_plotter.add_data(info)

            if i % self.params['log_freq'] == 0:
                self.loss_plotter.print(i)
            if i % self.params['plot_freq'] == 0:
                log.info('Creating vae visualizaton')
                self.loss_plotter.plot()
                self.plot_vae(obs, update_dir, i=i)
            if i % self.params['checkpoint_freq'] == 0 and i > 0:
                self.vae.save(os.path.join(update_dir, 'vae_%d.pth' % i))

        self.vae.save(os.path.join(update_dir, 'vae.pth'))

    def update(self, replay_buffer, update_dir):
        pass

    def plot_vae(self, obs, update_dir, i=0):
        if self.frame_stack == 1:
            obs = np.array([np.array(im).transpose((2, 0, 1)) for im in obs]) / 255
        else:
            obs = ptu.torchify(np.array(
                [[np.array(im).transpose((2, 0, 1)) for im in stack] for stack in obs]
            )) / 255

        with torch.no_grad():
            sample = torch.randn(64, self.d_latent).to(ptu.TORCH_DEVICE)
            sample = self.vae.decode(sample).cpu()
            if self.frame_stack > 1:
                # Sample n index randomely
                arange = torch.arange(64)
                ind = arange // 22
                sample = sample[arange, ind]
            save_image(sample, os.path.join(update_dir, 'sample_%d.png' % i))

        with torch.no_grad():
            data = ptu.torchify(obs[:8])
            recon = self.vae.decode(self.vae.encode(data))
            if self.frame_stack > 1:
                ls = []
                for j in range(self.frame_stack):
                    ls.append(data[:, j])
                    ls.append(recon[:, j].view(8, 3, 64, 64))
                comparison = torch.cat(ls)
            else:
                comparison = torch.cat([data,
                                        recon.view(8, 3, 64, 64)])
            save_image(comparison.cpu(), os.path.join(update_dir, 'recon_%d.png' % i))
