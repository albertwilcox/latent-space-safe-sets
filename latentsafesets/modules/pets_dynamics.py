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

        self.d_obs = params['d_obs']#(3, 64, 64)
        self.d_latent = params['d_latent']#32
        self.d_act = params['d_act']#2

        self.plot_freq = params['plot_freq']#500
        self.checkpoint_freq = params['checkpoint_freq']#2000
        self.normalize_delta = params['dyn_normalize_delta']#False, but what is it?
        self.n_particles = params['n_particles']#20
        self.trained = False

        # Dynamics args
        self.n_models = params['dyn_n_models']#5
        size = params['dyn_size']#128 units as seen in the paper
        n_layers = params['dyn_n_layers']#3 thus there are 3-1=2 hidden layers
        self.models = nn.ModuleList([#see line 146 for definition
            ProbabilisticDynamicsModel(self.d_latent, self.d_act, size=size, n_layers=n_layers)
                .to(ptu.TORCH_DEVICE)
            for _ in range(self.n_models)
        ])
        self.delta_rms = utils.RunningMeanStd(shape=(self.d_latent,))

        self.logdir = params['logdir']#'outputs/2022-07-18/19-38-58'

        self.learning_rate = params['dyn_lr']#0.001 as seen in the paper
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
        in act_seq for each model in self.dynamics_models#(there are altogether 20 models, right?)

        This corresponds to the TS-1 in the PETS paper#ref 43 in the LS3 paper!
        :param obs: Tensor, dimension (d_latent) if already embedded or (*d_obs)
        :param act_seq: Tensor, dimension (num_candidates, planning_hor, d_act)#here it is (1000,5,2)
        :param already_embedded: Whether or not obs is already embedded in the latent space
        :return: Final obs prediction, dimension (n_particles, num_candidates, planning_hor, d_latent)
        """#(20,1000,5,32)
        if already_embedded:
            emb = obs
        else:
            emb = self.encoder.encode(obs).detach()

        (num_candidates, plan_hor, d_act) = act_seq.shape#(1000,5,2)

        predicted_emb = torch.zeros((self.n_particles, num_candidates, plan_hor, self.d_latent))\
            .to(ptu.TORCH_DEVICE)#GPU
        running_emb = emb.repeat((num_candidates * self.n_particles, 1))#20000!(20000,32)
        for t in range(plan_hor):#H=5
            act = act_seq[:, t, :]#shape (1000,2)
            #print('t',t,'act',act)#when t=0 act=nan! the problem is from outside!
            act_tiled = act.repeat((self.n_particles, 1))#([20000,32])?
            model_ind = np.random.randint(0, self.n_models)#5
            model = self.models[model_ind]#randomly choose models?
            #print('running_emb.shape',running_emb.shape)#torch.Size([20000, 32])
            #print('act_tiled.shape',act_tiled.shape)#torch.Size([20000, 2])
            next_emb = model.get_next_emb(running_emb, act_tiled)
            #print('next_emb.shape', next_emb.shape)#torch.Size([20000, 32])
            predicted_emb[:, :, t, :] = next_emb.reshape((self.n_particles, num_candidates, self.d_latent))
            #predicted_emb should have shape (20,1000,5,32)
            running_emb = next_emb#next time step!
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
        self.d_latent = d_latent#32
        self.d_act = d_act#2
        self.n_layers = n_layers#3
        self.size = size#128
        self.delta_network = GenericNet(
            d_in=self.d_latent + self.d_act,
            d_out=self.d_latent * 2,#*2 means including both mean and variance
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
        delta_normalized_logstd = delta_normalized_both[:, self.d_latent:]#pay close attention to the line below
        delta_normalized_std = torch.exp(delta_normalized_logstd)+1e-6##torch.exp(delta_normalized_logstd)
        #delta_normalized_std=torch.where(delta_normalized_std<1e-6,1e-6,delta_normalized_std)#why not working?
        dist = torch.distributions.normal.Normal(delta_normalized_mean, delta_normalized_std)
        return dist

    def loss(self, emb, delta_unnormalized, act):
        delta_normalized = self._normalize_delta(delta_unnormalized)#204
        dist = self(emb, act)
        log_prob = dist.log_prob(delta_normalized)
        loss = torch.mean(log_prob)
        return -loss#equation 8 in the paper!

    def get_next_emb(self, emb, acs):
        #if torch.min(emb)<=0:#it doesn't matter!
            #print('torch.min(emb)<=0!',torch.min(emb))
        #if torch.min(acs)<=0:#it doesn't matter!
            #print('torch.min(acs)<=0!',torch.min(acs))
        #if (torch.min(emb)-torch.min(acs))<=0:#it matters!!!
            #print('(torch.min(emb)-torch.min(acs))<=0!',torch.min(emb)-torch.min(acs))
        #if torch.min(self.delta_std)<=0:
            #print('torch.min(delta_std)<=0!',torch.min(self.delta_std))
        #print('embinner01', emb)#it isn't nan#the last thing I can print out before error occurs!
        #print('acs01', acs)#this becomes nan!
        #print('torch.min(emb)',torch.min(emb))
        #print('torch.min(acs)', torch.min(acs))
        dist = self(emb, acs)
        delta_normalized = dist.rsample()#applying reparameterization trick
        #print('torch.min(delta_normalized)',torch.min(delta_normalized))
        #print('torch.min(self.delta_mean)',torch.min(self.delta_mean))
        #print('torch.min(self.delta_std)', torch.min(self.delta_std))
        delta = self._unnormalize_delta(delta_normalized)#210
        #if torch.min(delta)<=0:#it's not a problem!#we know that big minus delta is a big problem!
        #print('torch.min(delta)!',torch.min(delta))#<=0#how can min(delta) not to be negative!
        #if torch.min(self.delta_std)==0:
            #print('torch.min(delta_std)==0!',torch.min(self.delta_std))
        #print('embinner02.shape',emb.shape)#torch.Size([20000, 32])
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
        if self.delta_mean is not None:#big self.delta_std lead to big unnormalized delta?
            return delta_normalized * self.delta_std + self.delta_mean
        else:
            return delta_normalized
