import torch.nn as nn


class AutoEncoderDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim=512):
        print(f"AutoEncoderDecoder::init() input_dim: {input_dim} latent_dim: {latent_dim}")
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, input_dim),
        )


    def encode(self, x):
        return self.encoder(x).detach()

    def decode(self, z):
        return self.decoder(z).detach()

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
