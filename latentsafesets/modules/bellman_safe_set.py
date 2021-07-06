import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet
from latentsafesets.modules import EncodedModule

import torch
import torch.nn as nn


class BellmanSafeSet(nn.Module, EncodedModule):
    """
    Safe set that uses the bellman backup addition from section 4.2 of the LS3 paper
    TODO: add link to arxiv
    """

    def __init__(self, encoder, params: dict):
        super(BellmanSafeSet, self).__init__()
        EncodedModule.__init__(self, encoder)

        self._encoder = [encoder]

        self.d_obs = params['d_obs']
        self.d_latent = params['d_latent']
        self.bellman_coef = params['safe_set_bellman_coef']
        self.reduction = params['safe_set_bellman_reduction']
        self.targ_update_counter = 0
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.trained = False

        self.net = GenericNet(self.d_latent, 1, params['bc_n_hidden'],
                              params['bc_hidden_size']) \
            .to(ptu.TORCH_DEVICE)
        self.target_net = GenericNet(self.d_latent, 1, params['bc_n_hidden'],
                                     params['bc_hidden_size']) \
            .to(ptu.TORCH_DEVICE)

        lr = params['bc_lr']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.t = 0

    def safe_set_probability(self, state, already_embedded=False):
        logits = self(state, already_embedded)
        return torch.sigmoid(logits)

    def safe_set_probability_np(self, states, already_embedded=False):
        states = ptu.torchify(states)
        probabilities = self.safe_set_probability(states, already_embedded)
        return probabilities.detach().cpu().numpy()

    def forward(self, obs, already_embedded=False, target=False):
        """
        Returns inputs to sigmoid for probabilities
        """
        if not already_embedded:
            embedding = self.encoder.encode(obs).detach()
        else:
            embedding = obs
        if target:
            logits = self.target_net(embedding)
        else:
            logits = self.net(embedding)
        return logits

    def update(self, out_dict, already_embedded=False):
        self.trained = True

        obs = out_dict['obs']
        next_obs = out_dict['next_obs']
        ss = out_dict['safe_set']

        obs = ptu.torchify(obs)
        next_obs = ptu.torchify(next_obs)
        ss = ptu.torchify(ss)

        self.optimizer.zero_grad()
        loss, info = self.loss(obs, next_obs, ss, already_embedded)
        loss.backward()
        self.step()

        self.t += 1
        if self.t % 100 == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        return loss.item(), info

    def loss(self, obs, next_obs, ss, already_embedded=False):
        logits = self(obs, already_embedded).squeeze()
        logits_next = self(next_obs, already_embedded, target=True).squeeze().detach()

        if self.reduction == 'add':
            targets = self.bellman_coef * torch.sigmoid(logits_next) + (1 - self.bellman_coef) * ss
        elif self.reduction == 'max':
            targets = torch.max(self.bellman_coef * torch.sigmoid(logits_next), ss)
        else:
            raise ValueError('%s invalid' % self.reduction)

        loss = self.loss_func(logits, targets)
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
