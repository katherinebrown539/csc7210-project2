import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from DiabetesData import DiabeticData

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, device="cpu"):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        start_size = 3
        layer_sizes = [1024,4]
        encoder_layers = []
        for end_size in layer_sizes:
            conv = nn.Conv2d(start_size, end_size, 3, padding=1)
            relu = nn.ReLU()
            pool = nn.MaxPool2d(2,2)
            start_size = end_size
            encoder_layers.extend([conv,relu,pool])
        layer_sizes.reverse()
        
        decoder_layers = []
        for i in range(len(layer_sizes)-1):
            start_size = layer_sizes[i]
            end_size = layer_sizes[i+1]
            conv = nn.ConvTranspose2d(start_size, end_size, 2, stride=2)
            relu = nn.ReLU()
            decoder_layers.extend([conv,relu])
        
        decoder_layers.append(nn.ConvTranspose2d(layer_sizes[-1], 3, 2, stride=2))
        decoder_layers.append(nn.Sigmoid())
        # decoder_layers.append(nn.ReLU())

        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.decoder_layers = nn.ModuleList(decoder_layers)


        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.BCELoss() # nn.BCELoss()
        self.device = device
        

    def forward(self, x):

        for layer in self.encoder_layers:
            x = layer(x)

        for layer in self.decoder_layers:
            x = layer(x)

        return x

    def fit(self, n_epochs, train_loader):
        print("in fit function")
        for epoch in range(1, n_epochs+1):
            print("Epoch: {0}".format(epoch))
            # monitor training loss
            train_loss = 0.0
            
            ###################
            # train the model #
            ###################
            with tqdm(total=len(train_loader)) as pbar:
                for data in train_loader:
                    # _ stands in for labels, here
                    # no need to flatten images
                    images, _ = data
                    images = images.to(self.device)
                    # clear the gradients of all optimized variables
                    self.optimizer.zero_grad()
                    # forward pass: compute predicted outputs by passing inputs to the model
                    outputs = self.forward(images)
                    # calculate the loss
                    print(outputs.shape)
                    print(images.shape)
                    loss = self.criterion(outputs, images)
                    # backward pass: compute gradient of the loss with respect to model parameters
                    loss.backward()
                    # perform a single optimization step (parameter update)
                    self.optimizer.step()
                    # update running training loss
#                     print("Loss: " , loss.item())
#                     print("img size: ", images.size(0))
                    train_loss += loss.item()*images.size(0)
                    pbar.update(1)
            # print avg training statistics 
            train_loss = train_loss/len(train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, 
                train_loss
                ))
