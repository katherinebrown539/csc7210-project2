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

        #Encoder
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32,16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 4, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
       
        #Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(16, 32, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(32, 64, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(64, 3, 2, stride=2)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() # nn.BCELoss()
        self.device = device
        self.to(device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = F.relu(self.t_conv4(x))
                
        return x

    def fit(self, n_epochs, train_loader):
        print("in fit function")
        for epoch in range(1, n_epochs+1):
            print(epoch)
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
