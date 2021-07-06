import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet
from latentsafesets.modules import EncodedModule

import torch
import torch.nn as nn


class BCSafeSet(nn.Module, EncodedModule):
    """
    Basic safe set that uses simple binary cross entropy
    """

    def __init__(self, encoder, params: dict):
        super(BCSafeSet, self).__init__()
        EncodedModule.__init__(self, encoder)

        self._encoder = [encoder]

        self.d_obs = params['d_obs']
        self.d_latent = params['d_latent']
        self.reg = params['bc_reg']
        self.targ_update_counter = 0
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.trained = False

        self.net = GenericNet(self.d_latent, 1, params['bc_n_hidden'],
                              params['bc_hidden_size']) \
            .to(ptu.TORCH_DEVICE)

        lr = params['bc_lr']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def safe_set_probability(self, state, already_embedded=False):
        logits = self(state, already_embedded)
        return torch.sigmoid(logits)

    def safe_set_probability_np(self, states, already_embedded=False):
        states = ptu.torchify(states)
        probabilities = self.safe_set_probability(states, already_embedded)
        return probabilities.detach().cpu().numpy()

    def forward(self, obs, already_embedded=False):
        """
        Returns inputs to sigmoid for probabilities
        """
        if not already_embedded:
            embedding = self.encoder.encode(obs).detach()
        else:
            embedding = obs
        logits = self.net(embedding)
        return logits

    def update(self, out_dict, already_embedded=False):
        self.trained = True

        obs = out_dict['obs']
        ss = out_dict['safe_set']

        obs = ptu.torchify(obs)
        ss = ptu.torchify(ss)

        self.optimizer.zero_grad()
        loss, info = self.loss(obs, ss, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), info

    def loss(self, obs, ss, already_embedded=False):
        logits_pos = self(obs, already_embedded).squeeze()
        loss = self.loss_func(logits_pos, ss)

        info = {
            'ss': loss.item(),
        }

        return loss, info

    def step(self):
        """
        This assumes you've already done backprop. Steps optimizers
        """
        self.optimizer.step()

    def save(self, file):
        torch.save(self.net.state_dict(), file)

    def load(self, file):
        self.net.load_state_dict(torch.load(file, map_location=ptu.TORCH_DEVICE))
        self.trained = True
