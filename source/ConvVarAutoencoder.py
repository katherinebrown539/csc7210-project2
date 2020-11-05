import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, channel, height, width):
        super(Unflatten, self).__init__()
        self.channel = channel
        self.height = height
        self.width = width

    def forward(self, input):
        return input.view(input.size(0), self.channel, self.height, self.width)


class ConvVAE(nn.Module):

    def __init__(self, latent_size, device):
        super(ConvVAE, self).__init__()
        self.device = device
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(6272, 1024),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 6272),
            nn.ReLU(),
            Unflatten(128, 7, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=),
            nn.Sigmoid()
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        return mu, logvar

    def decode(self, z):
        z = self.decoder(z)
        return z

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # reconstruction loss
        BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def fit(self, n_epochs, train_loader):
        self.train()
        train_loss = 0
        for epoch in range(n_epochs):
            train_loss_ep = 0

            for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
                data = data.to(self.device)

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.forward(data)

                loss = self.loss_function(recon_batch, data, mu, logvar)
                train_loss += loss.item()

                loss.backward()
                self.optimizer.step()

            train_loss_ep /= len(train_loader.dataset)
            train_loss += train_loss_ep

        return train_loss