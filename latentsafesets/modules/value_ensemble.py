import latentsafesets.utils.pytorch_utils as ptu
from .interfaces import EncodedModule
from latentsafesets.modules import ValueFunction

import torch
import torch.nn as nn


class ValueEnsemble(nn.Module, EncodedModule):

    def __init__(self, encoder, params: dict):
        """
        Initializes a value function Function
        """
        super(ValueEnsemble, self).__init__()
        EncodedModule.__init__(self, encoder)
        self.trained = False

        self.n_models = params['val_n_models']
        self.reduction = params['val_reduction']

        self.models = nn.ModuleList([
            ValueFunction(encoder, params) for _ in range(self.n_models)
        ])

    def reduce(self, out):
        if self.reduction == 'mean':
            return torch.mean(out, dim=0)
        elif self.reduction == 'random':
            rand = torch.randint(out.shape[0], size=(out.shape[1],), device=ptu.TORCH_DEVICE)
            return out.squeeze()[rand].diag().reshape((out.shape[1], 1))
        elif self.reduction == 'median':
            return torch.median(out, dim=0)[0]


    def forward(self, obs, already_embedded=False):
        """
        Returns normalized output from
        """
        if already_embedded:
            emb = obs
        else:
            emb = self.encoder.encode(obs).detach()
        out = torch.cat([
            model(emb, True)[None] for model in self.models
        ])
        return self.reduce(out)

    def forward_np(self, obs, already_embedded=False):
        obs = ptu.torchify(obs)
        values = self(obs, already_embedded)
        return ptu.to_numpy(values)

    def get_value(self, obs, already_embedded=False):
        return self.forward(obs, already_embedded)

    def get_value_np(self, obs, already_embedded=False):
        return self.forward_np(obs, already_embedded)

    # TODO: update all this to be compatible with other stuff if necessary
    def update(self, obs, rew, next_obs, dones, already_embedded=False):
        """
        Updates ensemble of value networks
        :param obs: Shape (ensemble, batch, *d_obs or d_latent)
        :param rew: Shape (ensemble, batch)
        :param next_obs: Shape (ensemble, batch, *d_obs or d_latent)
        :param dones: Shape (ensemble, batch)
        :param already_embedded: Whether the obs is already embedded in latent space
        :return:
        """
        self.trained = True
        obs = ptu.torchify(obs)
        rew = ptu.torchify(rew)
        next_obs = ptu.torchify(next_obs)
        dones = ptu.torchify(dones)

        if already_embedded:
            emb = obs
            next_emb = next_obs
        else:
            emb = self.encoder.encode(obs).detach()
            next_emb = self.encoder.encode(next_obs).detach()

        loss = torch.tensor(0., device=ptu.TORCH_DEVICE)
        for i, model in list(enumerate(self.models)):
            emb_batch = emb[i]
            rew_batch = rew[i]
            next_emb_batch = next_emb[i]
            dones_batch = dones[i]
            loss_batch, _ = model.update(emb_batch, rew_batch, next_emb_batch, dones_batch, True)
            loss += loss_batch

        return loss.item(), {'val': loss.item()}

    def update_init(self, obs, rtg, already_embedded=False):
        obs = ptu.torchify(obs)
        rtg = ptu.torchify(rtg)

        if already_embedded:
            emb = obs
        else:
            emb = self.encoder.encode(obs).detach()

        loss = torch.tensor(0., device=ptu.TORCH_DEVICE)
        for i, model in list(enumerate(self.models)):
            emb_batch = emb[i]
            rtg_batch = rtg[i]
            loss_batch, _ = model.update_init(emb_batch, rtg_batch, already_embedded=True)
            loss += loss_batch

        return loss.item(), {'val': loss.item()}

    def loss(self, obs, rew, next_obs, dones, already_embedded=False):
        if already_embedded:
            emb = obs
            next_emb = next_obs
        else:
            emb = self.encoder.encode(obs).detach()
            next_emb = self.encoder.encode(next_obs).detach()

        loss = torch.tensor(0., device=ptu.TORCH_DEVICE)
        for i, model in list(enumerate(self.models)):
            emb_batch = emb[i]
            rew_batch = rew[i]
            next_emb_batch = next_emb[i]
            dones_batch = dones[i]
            loss_batch, _ = model.update(emb_batch, rew_batch, next_emb_batch, dones_batch, True)
            loss += loss_batch

        return loss, {'val': loss.item()}

    def loss_init(self, obs, rtg, already_embedded=False):
        pass

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        from latentsafesets.utils.pytorch_utils import TORCH_DEVICE
        self.load_state_dict(torch.load(file, map_location=TORCH_DEVICE))
        self.trained = True
