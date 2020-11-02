#!/usr/bin/env python
# coding: utf-8

# ## Define Imports and Determine Device
import os, sys
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
os.sys.path.insert(0, ".")
from DiabetesData import DiabeticData
from Autoencoder import ConvAutoencoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

task = ([0,1], [2,3,4])
batch_size=16
epochs = 1
root_dir = "data/diabetes"
# task = ([0,1,2], (3,4))

data = pd.read_csv("data/trainLabels.csv")
# data = data.sample(frac=0.25)
train, test = train_test_split(data, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)

train = train[train["level"] < 2]
print(train)
#filter out 1s from training set

train = train.reset_index()
test = test.reset_index()
val = val.reset_index()

data = {'train': DiabeticData(df = train, transform_key="train", root_dir=root_dir, task = task),
        'valid': DiabeticData(df = val, transform_key="valid", root_dir=root_dir, task = task),
        'test': DiabeticData(df = test, transform_key="test", root_dir=root_dir, task = task)
        }

dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'valid': DataLoader(data['valid'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=1, shuffle=True)
} 

print(train.shape)
print(val.shape)
print(test.shape)


model = ConvAutoencoder(device)
print(model)

model.fit(epochs, dataloaders["train"])

def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.255])
    inv_tensor = inv_normalize(tensor)
    img = inv_tensor.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))

classes = ['none', 'severe']
# obtain one batch of test images
dataiter = iter(dataloaders["valid"])
images, labels = dataiter.next()
images = images.to(device)
# get sample outputs
output = model(images)
# output = F.softmax(output)
# prep images for display
images = images.cpu().numpy()


# output is resized into a batch of iages
output = output.view(batch_size, 3, 224, 224)
# use detach when it's an output that requires_grad
output = output.cpu().detach().numpy()


# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in range(batch_size):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(output[idx])
    ax.set_title(classes[labels[idx]])
plt.savefig("autoencoded.png")    

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(24,4))
for idx in range(batch_size):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])

plt.savefig("original.png")
# Now, we loop through the training set and calculate reconstruction loss 



results = []
results_cols = ["Image Label", "Reconstruction Loss"]
for x, y in dataloaders['test']:
    X = x.to(device)
    output = model(X)
    output = output.cpu().detach().numpy()
    for i in range(y.shape[0]):
        ls = 0
        image = x[i].numpy()
        ouptut = output[i]
        label = y[i].numpy()
        ls = np.sum(np.square(image.ravel() - output.ravel()))
        results.append([label, ls])

results = pd.DataFrame(results, columns=results_cols)
results.to_csv("reconstruction_error.csv")


label_1 = results[results["Image Label"] == 1]
label_0 = results[results["Image Label"] == 0]

avg_1 = np.mean(label_1.values)
avg_0 = np.mean(label_0.values)

print("Average Reconstruction Error (Prediction = 0)", avg_0)
print("Average Reconstruction Error (Prediction = 1)", avg_1)

