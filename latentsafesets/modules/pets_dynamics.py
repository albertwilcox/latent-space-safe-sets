from .interfaces import EncodedModule
import latentsafesets.utils.pytorch_utils as ptu
import latentsafesets.utils as utils
from latentsafesets.model import GenericNet

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


class PETSDynamics(nn.Module, EncodedModule):
    """
    Implementation of PETS dynamics in a latent space
    """

    def __init__(self, encoder, params: dict):
        super(PETSDynamics, self).__init__()
        EncodedModule.__init__(self, encoder)

        self.d_obs = params['d_obs']
        self.d_latent = params['d_latent']
        self.d_act = params['d_act']

        self.plot_freq = params['plot_freq']
        self.checkpoint_freq = params['checkpoint_freq']
        self.normalize_delta = params['dyn_normalize_delta']
        self.n_particles = params['n_particles']
        self.trained = False

        # Dynamics args
        self.n_models = params['dyn_n_models']
        size = params['dyn_size']
        n_layers = params['dyn_n_layers']
        self.models = nn.ModuleList([
            ProbabilisticDynamicsModel(self.d_latent, self.d_act, size=size, n_layers=n_layers)
                .to(ptu.TORCH_DEVICE)
            for _ in range(self.n_models)
        ])
        self.delta_rms = utils.RunningMeanStd(shape=(self.d_latent,))

        self.logdir = params['logdir']

        self.learning_rate = params['dyn_lr']
        self.param_list = []
        for model in self.models:
            self.param_list += list(model.parameters())
        self.optimizer = optim.Adam(self.param_list, lr=self.learning_rate)

    def update(self, obs, next_obs, act, already_embedded=False):
        """
        Updates pets dynamics
        :param obs: shape (ensemble, batch, *d_obs or d_latent)
        :param next_obs: shape (ensemble, batch, *d_obs or d_latent)
        :param act: shape (ensemble, batch, d_act)
        :param already_embedded: Whether or not obs already embedded
        """
        self.trained = True
        obs = ptu.torchify(obs)
        next_obs = ptu.torchify(next_obs)
        act = ptu.torchify(act)

        if not already_embedded:
            emb = self.encoder.encode(obs)
            next_emb = self.encoder.encode(next_obs)
        else:
            emb = obs
            next_emb = next_obs

        if self.normalize_delta:
            delta = next_emb - emb
            self.delta_rms.update(delta.detach())
            for model in self.models:
                model.update_statistics(self.delta_rms.mean.detach(),
                                        torch.sqrt(self.delta_rms.var).detach())

        loss = 0
        for i, model in list(enumerate(self.models)):
            # Mini samples so the models are not identical
            emb_batch = emb[i]
            next_emb_batch = next_emb[i]
            delta_emb = next_emb_batch - emb_batch
            act_batch = act[i]

            loss_batch = model.loss(emb_batch, delta_emb, act_batch)
            loss = loss + loss_batch / self.n_models

        self.optimizer.zero_grad()
        loss.backward()
        self.step()

        return loss.item(), {'dyn': loss.item()}

    def predict(self, obs, act_seq, already_embedded=False):
        """
        Given the current obs, predict sequences of observations for each action sequence
        in act_seq for each model in self.dynamics_models

        This corresponds to the TS-1 in the PETS paper
        :param obs: Tensor, dimension (d_latent) if already embedded or (*d_obs)
        :param act_seq: Tensor, dimension (num_candidates, planning_hor, d_act)
        :param already_embedded: Whether or not obs is already embedded in the latent space
        :return: Final obs prediction, dimension (n_particles, num_candidates, planning_hor, d_latent)
        """
        if already_embedded:
            emb = obs
        else:
            emb = self.encoder.encode(obs).detach()

        (num_candidates, plan_hor, d_act) = act_seq.shape

        predicted_emb = torch.zeros((self.n_particles, num_candidates, plan_hor, self.d_latent))\
            .to(ptu.TORCH_DEVICE)
        running_emb = emb.repeat((num_candidates * self.n_particles, 1))
        for t in range(plan_hor):
            act = act_seq[:, t, :]
            act_tiled = act.repeat((self.n_particles, 1))
            model_ind = np.random.randint(0, self.n_models)
            model = self.models[model_ind]
            next_emb = model.get_next_emb(running_emb, act_tiled)
            predicted_emb[:, :, t, :] = next_emb.reshape((self.n_particles, num_candidates, self.d_latent))

            running_emb = next_emb
        return predicted_emb

    def step(self):
        self.optimizer.step()

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        from latentsafesets.utils.pytorch_utils import TORCH_DEVICE
        self.load_state_dict(torch.load(file, map_location=TORCH_DEVICE))
        self.trained = True

        for model in self.models:
            model.update_statistics(self.delta_rms.mean.detach(),
                                    torch.sqrt(self.delta_rms.var).detach())

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate


class ProbabilisticDynamicsModel(nn.Module):
    def __init__(self, d_latent, d_act, n_layers=3, size=128):
        super(ProbabilisticDynamicsModel, self).__init__()
        self.d_latent = d_latent
        self.d_act = d_act
        self.n_layers = n_layers
        self.size = size
        self.delta_network = GenericNet(
            d_in=self.d_latent + self.d_act,
            d_out=self.d_latent * 2,
            n_layers=self.n_layers,
            size=self.size,
        )

        self.delta_mean = None
        self.delta_std = None

    def update_statistics(self, mean, std):
        self.delta_mean = mean
        self.delta_std = std

    def forward(self, emb, acs):
        inp = emb
        concat = torch.cat((inp, acs), dim=1)
        delta_normalized_both = self.delta_network(concat)
        delta_normalized_mean = delta_normalized_both[:, :self.d_latent]
        delta_normalized_logstd = delta_normalized_both[:, self.d_latent:]
        delta_normalized_std = torch.exp(delta_normalized_logstd)
        dist = torch.distributions.normal.Normal(delta_normalized_mean, delta_normalized_std)
        return dist

    def loss(self, emb, delta_unnormalized, act):
        delta_normalized = self._normalize_delta(delta_unnormalized)
        dist = self(emb, act)
        log_prob = dist.log_prob(delta_normalized)
        loss = torch.mean(log_prob)
        return -loss

    def get_next_emb(self, emb, acs):
        dist = self(emb, acs)
        delta_normalized = dist.rsample()
        delta = self._unnormalize_delta(delta_normalized)
        return emb + delta

    def get_next_emb_and_loss(self, emb, delta_unnormalized, act):
        """
        Basically combines the last two functions so you can do both with one pass through the net
        """
        dist = self(emb, act)
        delta_normalized = self._normalize_delta(delta_unnormalized)
        log_prob = dist.log_prob(delta_normalized)
        loss = torch.mean(log_prob)
        loss = -loss

        pred_delta_normalized = dist.rsample()
        delta = self._unnormalize_delta(pred_delta_normalized)
        return emb + delta, loss

    def _normalize_delta(self, delta_unnormalized):
        if self.delta_mean is not None:
            return (delta_unnormalized - self.delta_mean) / (self.delta_std + 1e-8)
        else:
            return delta_unnormalized

    def _unnormalize_delta(self, delta_normalized):
        if self.delta_mean is not None:
            return delta_normalized * self.delta_std + self.delta_mean
        else:
            return delta_normalized
