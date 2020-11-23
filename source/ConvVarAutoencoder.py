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

    def __init__(self, latent_size, device, task):
        super(ConvVAE, self).__init__()
        self.device = device
        self.latent_size = latent_size
        self.task = task
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(294912, 1024),
            nn.ReLU()
        )

        # hidden => mu
        self.fc1 = nn.Linear(1024, self.latent_size)

        # hidden => logvar
        self.fc2 = nn.Linear(1024, self.latent_size)

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 294912),
            nn.ReLU(),
            Unflatten(32, 96, 96),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
        h = self.encoder(x)
        print(x.size())
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
        BCE = F.binary_cross_entropy(recon_x.view(-1, 442368), x.view(-1, 442368), reduction='sum')

        # KL divergence loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def fit(self, n_epochs, train_loader, validation_loader=None):
        self.train()
        train_loss = 0
        val_loss = 0
        for epoch in range(n_epochs):
            train_loss_ep = 0
            val_loss_ep = 0
            for batch_idx, (data, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):
                data = data.to(self.device)

                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.forward(data)

                loss = self.loss_function(recon_batch, data, mu, logvar)
                train_loss_ep += loss.item()

                loss.backward()
                self.optimizer.step()
            if validation_loader is not None:
                for data in validation_loader:
                    images, _ = data
                    images = images.to(self.device)
                    # clear the gradients of all optimized variables
                    self.optimizer.zero_grad()
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = self.forward(images)
                    # calculate the loss
                    loss = self.criterion(outputs, images)
                    # backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    val_loss_ep += loss.item()*images.size(0)
                val_loss_ep /= len(validation_loader.dataset)
                val_loss += val_loss_ep
                history["validation_loss"].append(val_loss)
            train_loss_ep /= len(train_loader.dataset)
            train_loss += train_loss_ep
            print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                val_loss
                ))
        torch.save(self.state_dict(), "models/ConvAE_{0}_{1}.pth".format(self.task,epoch))

        return train_loss