import torch
import torch.nn as nn

import latentsafesets.utils.pytorch_utils as ptu


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 1, 1)


class GenericNet(nn.Module):

    def __init__(self, d_in, d_out, n_hidden=2, d_hidden=128, activation=nn.ReLU,
                 last_activation=None, **kwargs):
        super(GenericNet, self).__init__()

        assert n_hidden >= 1, "Must have at least 1 hidden layer"
        layers = [nn.Linear(d_in, d_hidden), activation()]
        for _ in range(n_hidden - 1):
            layers.extend([nn.Linear(d_hidden, d_hidden), activation()])
        layers.append(nn.Linear(d_hidden, d_out))
        if last_activation is not None:
            layers.append(last_activation())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
 #       print(x.shape)
        return self.model(x)


class VAEEncoder(nn.Module):
    def __init__(self, d_obs, d_latent=32, image_channels=3, h_dim=256):
        super(VAEEncoder, self).__init__()
        self.d_obs = d_obs
        self.d_in = d_obs if len(d_obs) == 3 else (d_obs[0] * d_obs[1], d_obs[2], d_obs[3])
        self.out_dim = d_latent
        in_channels = image_channels if len(d_obs) == 3 else image_channels * d_obs[0]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=3),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, h_dim, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc1 = nn.Linear(h_dim, d_latent)
        self.fc2 = nn.Linear(h_dim, d_latent)

    def forward(self, x):
#        print(x.shape)
        batch_dims = x.shape[:-len(self.d_obs)]
        if len(batch_dims) == 0:
            batch_dims = (1,)
        observation = x.reshape(-1, *self.d_in)

        z = self.encoder(observation)
        mu = self.fc1(z)
        log_std = self.fc2(z)
        mu = mu.view(*batch_dims, -1)
        log_std = log_std.view(*batch_dims, -1)
        return mu, log_std

    def encode(self, x):
        ptu.torchify(x)
        mu, log_std = self(x)
        std = log_std.mul(0.5).exp_()
        dist = torch.distributions.normal.Normal(mu, std)
        z = dist.rsample()
        return z


class VAEDecoder(nn.Module):
    def __init__(self, d_obs, d_latent=32, image_channels=3, h_dim=256):
        super(VAEDecoder, self).__init__()
        self.d_obs = d_obs
        self.d_out = d_obs if len(d_obs) == 3 else (d_obs[0] * d_obs[1], d_obs[2], d_obs[3])
        self.d_latent = d_latent
        out_channels = image_channels if len(d_obs) == 3 else image_channels * d_obs[0]

        self.decoder = nn.Sequential(
            nn.Linear(d_latent, h_dim),
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, out_channels, kernel_size=6, stride=2),
            # nn.Sigmoid(),
        )

    def forward(self, x):
        batch_dims = x.shape[:-1]
        if len(batch_dims) == 0:
            batch_dims = (1,)
        hidden = x.view(-1, self.d_latent)

        # trajlen, batchsize = hidden.size(0), hidden.size(1)
        # hidden = hidden.view(trajlen * batchsize, -1)
        z = self.decoder(hidden)

        z = z.view(*batch_dims, *self.d_obs)
        return z

    def decode(self, x):
        return self(x)

