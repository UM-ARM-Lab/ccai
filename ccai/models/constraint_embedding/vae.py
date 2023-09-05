import torch
from torch import nn
from torch.nn import functional as F


class VAE(nn.Module):
    def __init__(self, latent_dim, num_channels=[32, 64, 128, 256], voxels=False):
        super().__init__()
        if voxels:
            self.encoder = Conv3DStochasticEncoder(latent_dim, num_channels)
            self.decoder = Conv3DDecoder(latent_dim, num_channels[::-1])
        else:
            self.encoder = ConvEncoder(latent_dim, num_channels)
            self.decoder = ConvDecoder(latent_dim, num_channels[::-1])

        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)

    def encode(self, x):
        latent_mu, latent_sigma = self.encoder(x)
        latent = latent_mu + latent_sigma * torch.randn_like(latent_mu)
        return latent, latent_mu, latent_sigma

    def forward(self, x):
        latent, latent_mu, latent_sigma = self.encode(x)
        x_hat = self.decoder(latent)
        return x_hat, latent, latent_mu, latent_sigma

    def get_kl_divergence(self, latent, latent_mu, latent_sigma, use_samples=False):
        q = torch.distributions.normal.Normal(latent_mu, latent_sigma)
        return torch.distributions.kl.kl_divergence(q, self.prior).sum(dim=1)


class Conv3DStochasticEncoder(nn.Module):

    def __init__(self, latent_dim, num_channels=[32, 64, 128, 256]):
        super().__init__()
        self.act_fn = F.relu
        self.conv_net = nn.Sequential(
            nn.Conv3d(1, num_channels[0], (3, 3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(num_channels[0], num_channels[1], (3, 3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(num_channels[1], num_channels[2], (3, 3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(num_channels[2], num_channels[3], (3, 3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        self.h_dim = int(num_channels[3] * 4 * 4 * 4)
        self.fc1 = nn.Linear(self.h_dim, latent_dim * 2)

    def forward(self, x):
        h = self.conv_net(x)
        h = self.fc1(h.view(-1, self.h_dim))

        latent_mu, h_sigma = torch.chunk(h, dim=1, chunks=2)
        latent_sigma = F.softplus(h_sigma) + 1e-2
        return latent_mu, latent_sigma


class Conv3DEncoder(nn.Module):

    def __init__(self, latent_dim, num_channels=[32, 64, 128, 256]):
        super().__init__()
        self.act_fn = F.relu
        self.conv_net = nn.Sequential(
            nn.Conv3d(1, num_channels[0], (3, 3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(num_channels[0], num_channels[1], (3, 3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(num_channels[1], num_channels[2], (3, 3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(num_channels[2], num_channels[3], (3, 3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        #self.conv_net = torch.compile(self.conv_net, mode='max-autotune')
        self.h_dim = int(num_channels[3] * 4 * 4 * 4)
        self.fc1 = nn.Linear(self.h_dim, latent_dim)

    def forward(self, x):
        h = self.conv_net(x)
        h = h.view(-1, self.h_dim)
        h = self.fc1(h)
        return h


class Conv3DDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels=[256, 128, 64, 32]):
        super().__init__()
        self.act_fn = F.relu
        self.h_dim = int(num_channels[0] * 4 * 4 * 4)
        self.first_conv_dim = (num_channels[0], 4, 4, 4)
        self.fc1 = nn.Linear(latent_dim, self.h_dim)

        self.deconv_net = nn.Sequential(
            nn.ConvTranspose3d(num_channels[0], num_channels[1], (4, 4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(num_channels[1], num_channels[2], (4, 4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(num_channels[2], num_channels[3], (4, 4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose3d(num_channels[3], 1, (4, 4, 4), stride=2, padding=1)
        )

    def forward(self, latent):
        h = self.act_fn(self.fc1(latent)).view(-1, *self.first_conv_dim)
        return self.deconv_net(h)


class ConvEncoder(nn.Module):

    def __init__(self, latent_dim, num_channels=[32, 64, 128, 256]):
        super().__init__()
        self.act_fn = F.relu
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, num_channels[0], (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels[0], num_channels[1], (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels[1], num_channels[2], (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(num_channels[2], num_channels[3], (3, 3), stride=2, padding=1),
            nn.ReLU()
        )
        self.h_dim = int(num_channels[3] * 4 * 4)
        self.fc1 = nn.Linear(self.h_dim, latent_dim * 2)

    def forward(self, x):
        h = self.conv_net(x)
        h = self.fc1(h.view(-1, self.h_dim))

        latent_mu, h_sigma = torch.chunk(h, dim=1, chunks=2)
        latent_sigma = F.softplus(h_sigma) + 1e-2
        return latent_mu, latent_sigma


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim, num_channels=[256, 128, 64, 32]):
        super().__init__()
        self.act_fn = F.relu
        self.h_dim = int(num_channels[0] * 4 * 4)
        self.first_conv_dim = (num_channels[0], 4, 4)
        self.fc1 = nn.Linear(latent_dim, self.h_dim)
        self.deconv_net = nn.Sequential(
            nn.ConvTranspose2d(num_channels[0], num_channels[1], (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[1], num_channels[2], (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[2], num_channels[3], (4, 4), stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(num_channels[3], 1, (4, 4), stride=2, padding=1)
        )

    def forward(self, latent):
        h = self.act_fn(self.fc1(latent)).view(-1, *self.first_conv_dim)
        x = self.deconv_net(h)
        return x


if __name__ == "__main__":
    model = Conv3DEncoder(64).cuda()
    #model = torch.compile(model, mode='reduce-overhead')
    with torch.no_grad():
        x = torch.randn(256, 1, 64, 64, 64).cuda()
    y = model(x)
    print(y.shape)

    import time
    s = time.time()
    for _ in range(1000):
        with torch.no_grad():
            y = model(x)
    e = time.time()
    print(e - s)
