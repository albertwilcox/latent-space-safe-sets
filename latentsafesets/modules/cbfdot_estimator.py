import latentsafesets.utils.pytorch_utils as ptu
from latentsafesets.model import GenericNet
from .interfaces import EncodedModule

import torch
import torch.nn as nn


class CBFdotEstimator(nn.Module, EncodedModule):#supervised learning very similar to gi
    """
    Simple constraint predictor using binary cross entropy
    """

    def __init__(self, encoder, params: dict):
        """
        Initializes a constraint estimator
        """
        super(CBFdotEstimator, self).__init__()
        EncodedModule.__init__(self, encoder)

        self.d_obs = params['d_obs']#(3,64,64)
        self.d_latent = 4#2+2#params['d_latent']#32
        self.batch_size = params['cbfd_batch_size']#256
        self.targ_update_counter = 0
        self.loss_func = torch.nn.SmoothL1Loss()#designate the loss function#torch.nn.BCEWithLogitsLoss()#
        self.trained = False

        self.net = GenericNet(self.d_latent, 1, params['cbfd_n_hidden'],
                              params['cbfd_hidden_size']) \
            .to(ptu.TORCH_DEVICE)
        #print(self.net)#input size 4, output size 1
        lr = params['cbfd_lr']
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, obs, already_embedded=False):
        """
        Returns inputs to sigmoid for probabilities
        """
        if not already_embedded:
            embedding = obs#self.encoder.encode(obs).detach()#workaround
        else:
            embedding = obs
        #print('embedding.shape',embedding.shape)#torch.Size([1000,5,4])#torch.Size([256,4])#torch.Size([180,4])#
        log_probs = self.net(embedding)#why 3 kinds of sizes?
        return log_probs

    def cbfdots(self, obs, already_embedded=False):
        obs = ptu.torchify(obs)
        logits = self(obs, already_embedded)
        probs = logits#torch.sigmoid(logits)#
        return ptu.to_numpy(probs)

    def update(self, next_obs, constr, already_embedded=False):
        self.trained = True
        next_obs = ptu.torchify(next_obs)#input
        constr = ptu.torchify(constr)#output

        self.optimizer.zero_grad()
        loss = self.loss(next_obs, constr, already_embedded)
        loss.backward()
        self.step()

        return loss.item(), {'cbfd': loss.item()}

    def loss(self, next_obs, constr, already_embedded=False):
        logits = self(next_obs, already_embedded).squeeze()#.forward!
        targets = constr
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
