import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet
from .interfaces import EncodedModule

import torch
import torch.nn as nn


class ValueFunction(nn.Module, EncodedModule):

    def __init__(self, encoder, params: dict):
        """
        Initializes a value function Function
        """
        super(ValueFunction, self).__init__()
        EncodedModule.__init__(self, encoder)

        self.d_obs = params['d_obs']#(3, 64, 64)
        self.d_latent = params['d_latent']#32
        self.discount = params['val_discount']#gamma, 0.99
        self.targ_update_frequency = params['val_targ_update_freq']#100#should it be called frequency or period?
        self.targ_update_rate = params['val_targ_update_rate']#1.0
        self.targ_update_counter = 0#what is this?
        self.loss_func = torch.nn.SmoothL1Loss()#designate the loss function
        self.trained = False

        self.value_net = GenericNet(self.d_latent, 1, params['val_n_hidden'],#default 3
                                    params['val_hidden_size']).to(ptu.TORCH_DEVICE)
        self.value_net_target = GenericNet(self.d_latent, 1, params['val_n_hidden'],
                                           params['val_hidden_size']).to(ptu.TORCH_DEVICE)
        for param in self.value_net_target.parameters():
            param.requires_grad = False#indeed
        self.update_target(1)#see line 134

        lr = params['val_lr']#1e-4 default
        self.optimizer = torch.optim.Adam(self.value_net.parameters(), lr=lr)

    def forward(self, obs, already_embedded=False):
        if not already_embedded:
            embedding = self.encoder.encode(obs).detach()
        else:
            embedding = obs
        val = self.value_net(embedding)
        return val

    def forward_np(self, obs, already_embedded=False):
        obs = ptu.torchify(obs)
        values = self(obs, already_embedded)#is it using/calling forward?
        return ptu.to_numpy(values)

    def get_value(self, obs, already_embedded=False):
        return self.forward(obs, already_embedded)

    def get_value_np(self, obs, already_embedded=False):
        return self.forward_np(obs, already_embedded)

    def update(self, obs, rew, next_obs, dones, already_embedded=False):
        self.trained = True
        obs = ptu.torchify(obs)
        rew = ptu.torchify(rew)
        next_obs = ptu.torchify(next_obs)
        dones = ptu.torchify(dones)

        self.optimizer.zero_grad()
        loss = self.loss(obs, rew, next_obs, dones, already_embedded)
        loss.backward()
        self.step()#line 122

        return loss.item(), {'val': loss.item()}

    def update_init(self, obs, rtg, already_embedded=False):
        obs = ptu.torchify(obs)
        rtg = ptu.torchify(rtg)#reward to goal

        self.optimizer.zero_grad()
        loss = self.loss_init(obs, rtg, already_embedded)
        loss.backward()
        self.step()#line 122

        return loss.item(), {'val': loss.item()}

    def loss(self, obs, rews, next_obs, dones, already_embedded=False):
        if not already_embedded:
            emb = self.encoder.encode(obs).detach()#latent state
            next_emb = self.encoder.encode(next_obs).detach()
        else:
            emb = obs#latent state
            next_emb = next_obs

        val_out = self.value_net(emb).squeeze()
        target_out = self.value_net_target(next_emb).squeeze()#that is our target, value of next emb!
        targets = (rews + (1 - dones) * self.discount * target_out).detach()#see 650 notes!

        # Set states in the goal to have zero value
        zero_goal = True#???It may not be a bug, maybe just a way to write the code like English
        if zero_goal:
            targets = targets * rews.bool().float()

        loss = self.loss_func(val_out, targets)
        return loss

    def loss_init(self, obs, rtg, already_embedded=False):
        """

        :param obs: observations shape (batch, *d_obs or latent)
        :param rtg: rewards to go shape (batch,)
        :param already_embedded: whether already embedded
        :return: the loss
        """
        if not already_embedded:
            emb = self.encoder.encode(obs).detach()
        else:
            emb = obs

        targets = rtg

        val_out = self.value_net(emb).squeeze()

        loss = self.loss_func(val_out, targets)
        return loss

    def step(self):
        """
        This assumes you've already done backprop. Steps optimizers
        """
        self.optimizer.step()

        # Update target function periodically
        self.targ_update_counter += 1
        if self.targ_update_counter % self.targ_update_frequency == 0:#update the target network every 100 updates
            self.value_net_target.load_state_dict(self.value_net.state_dict())
            self.targ_update_counter = 0

    def update_target(self, rate=-1):
        if rate == -1:
            rate = self.targ_update_rate
        if rate == 1:
            self.value_net_target.load_state_dict(self.value_net.state_dict())#keep it!
        else:
            for param, target_param in list(zip(self.value_net.parameters(),
                                                self.value_net_target.parameters())):
                target_param.data = target_param * (1 - rate) + param * rate

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        from latentsafesets.utils.pytorch_utils import TORCH_DEVICE
        self.load_state_dict(torch.load(file, map_location=TORCH_DEVICE))
        self.trained = True
