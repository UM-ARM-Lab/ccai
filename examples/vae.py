import torch

# MLP based VAE. Ignore constraints and controls. forward should sample in train mode and use the mean in eval mode

class VAE(torch.nn.Module):
    def __init__(self, nx, nz, nc, nhidden):
        super(VAE, self).__init__()
        self.nx = nx
        self.nz = nz
        self.nc = nc
        self.nhidden = nhidden
        
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(nx + nc, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nz*2),
            )
        
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(nz + nc, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nhidden),
            torch.nn.ReLU(),
            torch.nn.Linear(nhidden, nx),
            )
    
    def forward(self, x, c):
        # Encoder
        input = torch.cat([x, c], dim=-1)
        z, mu, logvar = self.encode(input)
        # Decoder
        return self.decoder(torch.cat((z, c)), dim=-1), mu, logvar
    
    def loss(self, x, x_recon, mu, logvar):
        # Reconstruction loss
        recon_loss = torch.nn.functional.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + 1e-2 * kl_loss
    
    def sample(self, mu, logvar):
        std = torch.exp(.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps*std
        return z
    
    def decode(self, z, c):
        input = torch.cat([z, c], dim=-1)
        return self.decoder(input)
    
    def encode(self, x, c):
        input = torch.cat([x, c], dim=-1)
        mu, logvar = torch.chunk(self.encoder(input), 2, dim=-1)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
        
    def reparameterize(self, mu, logvar):
        if self.training:
            # Reparameterization trick
            z = self.sample(mu, logvar)
        else:
            z = mu
        return z