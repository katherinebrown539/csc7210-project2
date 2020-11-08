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


VGG_type = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "VGG16": [64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,],
    "VGG19": [64,64,"M",28,128,"M",256,256,256,256,"M",512,512,512,512,"M",512,512,512,512]
}

class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super(Decoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.upconv2 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.upconv3 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2, padding=0
        )
        self.upconv4 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=3,
            kernel_size=2,
            stride=2,
            padding=0,
        )

        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(0.2)
        self.op = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.upconv1(x))
        x = self.bn1(x)
        x = self.relu(self.upconv2(x))
        x = self.bn2(x)
        x = self.relu(self.upconv3(x))
        x = self.bn3(x)
        x = self.relu(self.upconv4(x))
        output = self.op(x)

        return x


class VGGEncoder(nn.Module):
    def __init__(self, vgg_version="VGG16", in_channels=3):
        super(VGGEncoder, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self.create_conv(VGG_type[vgg_version])
        # after completing all the conv layer the final matrix will be [ bs , 512, 7 , 7]

    def forward(self, x):
        x = self.conv_layers(x)

        return x

    def create_conv(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    ),
                    nn.BatchNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x

            elif x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]


        return nn.Sequential(*layers)


class VGGAutoencoder(nn.Module):
    def __init__(self, channels=3):
        super(ConvAutoEncoder, self).__init__()
        self.enc = VGGEncoder("VGG16")
        self.dec = Decoder(channels)

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)

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
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, 
                train_loss
                ))
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
        self.visualize(history)