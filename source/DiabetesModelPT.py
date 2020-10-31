# PyTorch Imports 
#from IPython.core.interactiveshell import InteractiveShell
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, sampler
import torch.nn as nn
from torchsummary import summary # Useful for examining network
from torch.nn import NLLLoss, CrossEntropyLoss
# Data Science | Image Tools | MatplotLib
import numpy as np
import pandas as pd
import os, sys, shutil, time, argparse
from datetime import date
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from tqdm import tqdm

# Image manipulations
from PIL import Image

# Visualizations
import matplotlib.pyplot as plt


class DiabetesModel(nn.Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(1024, 2)
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.to('cuda')
            self.device = 'cuda'
        print(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.0005, momentum=0.985, nesterov=True)
        summary(self.model, input_size=(3,224,224))
    def forward(self, x):
        x = self.model(x)
        # x = F.relu(x)
        return x

    def fit(self, train_generator, validation_generator=None, n_epochs=50):
        for epoch in range(n_epochs):
            self.train()
            with tqdm(total=len(train_generator)) as pbar:
                for local_batch, local_labels in train_generator:
                    self.optimizer.zero_grad() #test comment
                    local_batch, local_labels = local_batch.to(self.device), local_labels.to(self.device)
                    output = self.forward(local_batch)
                    loss = self.criterion(output, local_labels.squeeze())
                    loss.backward()
                    self.optimizer.step()
                    pbar.update(1)
                
    def predict(self, generator):
        self.eval()
        # criterion = nn.NLLLoss()
        test_loss = 0
        correct = 0
        probabilities = []
        predictions = []
        correct = []
        with torch.no_grad():
            for data, target in generator:
                data, target = data.to(self.device), target.to(self.device)
                output = self.forward(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss
                output = F.log_softmax(output, dim=1)
                
                probability = torch.exp(output)
                probability = probability.cpu().numpy()
                target = target.cpu().numpy()

                probabilities.extend(probability[:,1])
                correct.extend(target)

        return probabilities, correct