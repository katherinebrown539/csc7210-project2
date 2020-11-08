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


# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, device="cpu", task='task'):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        self.encoder = nn.ModuleList([
            nn.Conv2d(3, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(512, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            Flatten(),
            nn.Linear(2304, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU()
            ])


        self.decoder = nn.ModuleList([
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2304),
            Unflatten(4, 96,96),
            nn.ConvTranspose2d(4, 512, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 3, 2, stride=2),
            nn.Sigmoid()
            ])

        # # conv layer (depth from 3 --> 16), 3x3 kernels
        # self.conv1 = nn.Conv2d(3, 512, 3, padding=1)  
        # # conv layer (depth from 16 --> 4), 3x3 kernels
        # self.conv2 = nn.Conv2d(512, 4, 3, padding=1)
        # # pooling layer to reduce x-y dims by two; kernel and stride of 2
        # self.pool = nn.MaxPool2d(2, 2)
        
        # self.fc_1 = nn.Linear()

        # ## decoder layers ##
        # ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        # self.t_conv1 = nn.ConvTranspose2d(4, 512, 2, stride=2)
        # self.t_conv2 = nn.ConvTranspose2d(512, 3, 2, stride=2)
        
        self.to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.criterion = nn.MSELoss() # nn.BCELoss()
        self.device = device
        self.task = task
        

    def forward(self, x):
        for layer in self.encoder:
            print(layer)
            x = layer(x)
        for layer in self.decoder:
            print(layer)
            x = layer(x)
        # ## encode ##
        # # add hidden layers with relu activation function
        # # and maxpooling after
        # x = F.relu(self.conv1(x))
        # x = self.pool(x)
        # # add second hidden layer
        # x = F.relu(self.conv2(x))
        # x = self.pool(x)  # compressed representation
        
        # ## decode ##
        # # add transpose conv layers, with relu activation function
        # x = F.relu(self.t_conv1(x))
        # # output layer (with sigmoid for scaling from 0 to 1)
        # x = F.sigmoid(self.t_conv2(x))
                
        return x

    def fit(self, n_epochs, train_loader, validation_loader=None):
        print("in fit function")
        history = {}
        history["training_loss"] = []
        history["validation_loss"] = []
        for epoch in range(1, n_epochs+1):
            print("Epoch: {0}".format(epoch))
            # monitor training loss
            train_loss = 0.0
            val_loss = 0.0
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
            history["training_loss"].append(train_loss)
            
            torch.save(self.state_dict(), "models/ConvAE_{0}_{1}.pth".format(self.task,epoch))

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
                    val_loss += loss.item()*images.size(0)
                history["validation_loss"].append(val_loss)
            print('Epoch: {} \tTraining Loss: {:.6f}\tValidation Loss: {:.6f}'.format(
                epoch, 
                train_loss,
                val_loss
                ))
        self.visualize(history)

    def visualize(self, history):
        plt.plot(history['training_loss'])
        plt.plot(history['validation_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("training_{0}.png".format(self.task))
        plt.clf()