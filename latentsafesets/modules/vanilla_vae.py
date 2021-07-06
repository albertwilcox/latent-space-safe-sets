from latentsafesets.model import VAEEncoder, VAEDecoder
import latentsafesets.utils.pytorch_utils as ptu

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import numpy as np


class VanillaVAE(nn.Module):

    def __init__(self, params):
        super(VanillaVAE, self).__init__()

        self.d_obs = params['d_obs']
        self.d_latent = params['d_latent']
        self.kl_multiplier = params['enc_kl_multiplier']
        self.trained = False

        self.frame_stack = params['frame_stack']
        self.encoder = VAEEncoder(self.d_obs, self.d_latent).to(ptu.TORCH_DEVICE)
        self.decoder = VAEDecoder(self.d_obs, self.d_latent).to(ptu.TORCH_DEVICE)
        self.transform = transforms.RandomResizedCrop(64, scale=(0.8, 1.0)) \
            if params['enc_data_aug'] \
            else lambda x: x

        self.learning_rate = params['enc_lr']
        param_list = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(param_list, lr=self.learning_rate)

    def forward(self, inputs):
        mu, log_std = self.encoder(inputs)
        return mu, log_std

    def encode(self, inputs):
        mu, log_std = self(inputs)
        std = torch.exp(log_std)
        samples = torch.empty(mu.shape).normal_(mean=0, std=1).to(ptu.TORCH_DEVICE)
        encoding = mu + std * samples
        return encoding.detach()

    def decode(self, inputs):
        return torch.clamp(self.decoder(inputs), 0., 1.)

    def loss(self, obs: torch.Tensor, obs_in: torch.Tensor) -> (torch.Tensor, dict):
        mu, log_std = self(obs_in)
        std = torch.exp(log_std)
        samples = torch.empty(mu.shape).normal_(mean=0, std=1).to(ptu.TORCH_DEVICE)
        encoding = mu + std * samples
        kl_loss = 0.5 * torch.mean(mu ** 2 + std ** 2 - torch.log(std ** 2) - 1)
        reconstruction = self.decoder(encoding)
        targets = obs
        r_loss = F.mse_loss(reconstruction, targets, reduction='mean')

        loss = kl_loss * self.kl_multiplier + r_loss
        data = {
            'vae': loss.item(),
            'vae_kl': kl_loss.item(),
            'vae_recon': r_loss.item()}

        return loss, data

    def update(self, obs):

        # Augment it
        if self.frame_stack == 1:
            obs_in = np.array(
                [np.array(self.transform(im)).transpose((2, 0, 1)) for im in obs]) / 255
            obs_in = ptu.torchify(obs_in)

            obs = ptu.torchify(np.array([np.array(im).transpose((2, 0, 1)) for im in obs])) / 255
        else:
            obs_in = np.array([
                [np.array(self.transform(im)).transpose((2, 0, 1)) for im in stack]
                for stack in obs
            ]) / 255
            obs_in = ptu.torchify(obs_in)

            obs = ptu.torchify(np.array(
                [[np.array(im).transpose((2, 0, 1)) for im in stack] for stack in obs]
            )) / 255

        self.trained = True

        self.optimizer.zero_grad()
        loss, data = self.loss(obs, obs_in)
        loss.backward()
        self.optimizer.step()

        return loss.item(), data

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        self.load_state_dict(torch.load(file, map_location=ptu.TORCH_DEVICE))
        self.trained = True
