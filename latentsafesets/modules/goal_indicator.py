import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet
from .interfaces import EncodedModule

import torch
import torch.nn as nn


class GoalIndicator(nn.Module, EncodedModule):
    """
    Simple goal set predictor using binary cross entropy
    """

    def __init__(self, encoder, params: dict):
        """
        Initializes a goal indicator
        """
        super(GoalIndicator, self).__init__()
        EncodedModule.__init__(self, encoder)

        self.d_obs = params['d_obs']
        self.d_latent = params['d_latent']
        self.batch_size = params['gi_batch_size']
        self.targ_update_counter = 0
        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.trained = False

        self.net = GenericNet(self.d_latent, 1, params['gi_n_hidden'],
                              params['gi_hidden_size']) \
            .to(ptu.TORCH_DEVICE)

        lr = params['gi_lr']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, obs, already_embedded=False):
        """
        Returns inputs to sigmoid for probabilities
        """
        if not already_embedded:
            embedding = self.encoder.encode(obs).detach()
        else:
            embedding = obs
        log_probs = self.net(embedding)
        return log_probs

    def prob(self, obs, already_embedded=False):
        obs = ptu.torchify(obs)
        logits = self(obs, already_embedded)
        probs = torch.sigmoid(logits)
        return ptu.to_numpy(probs)

    def update(self, next_obs, rew, already_embedded=False):
        self.trained = True

        next_obs = ptu.torchify(next_obs)
        rew = ptu.torchify(rew)

        self.optimizer.zero_grad()
        loss = self.loss(next_obs, rew, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), {'gi': loss.item()}

    def loss(self, next_obs, rew, already_embedded=False):
        # Assuming that rew = {-1: not in goal, 0: in goal}
        logits = self(next_obs, already_embedded).squeeze()
        targets = torch.logical_not(rew).float()
        loss = self.loss_func(logits, targets)
        return loss

    def step(self):
        """
        This assumes you've already done backprop. Steps optimizers
        """
        self.optimizer.step()

    def save(self, file):
        torch.save(self.net.state_dict(), file)

    def load(self, file):
        from latentsafesets.utils.pytorch_utils import TORCH_DEVICE
        self.net.load_state_dict(torch.load(file, map_location=TORCH_DEVICE))
        self.trained = True
