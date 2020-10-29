# PyTorch Imports
#from IPython.core.interactiveshell import InteractiveShell
from torchvision import transforms, datasets, models
import torch
import torch.nn.functional as F
from torch import optim, cuda
from torch.utils.data import Dataset, DataLoader, sampler
import torch.nn as nn
from torchsummary import summary # Useful for examining network
from collections import Counter
# Data Science | Image Tools | MatplotLib
import numpy as np
import pandas as pd
import matplotlib
import os, sys, shutil, time, argparse, collections
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from tqdm import tqdm
from datetime import date
# Image manipulations
from PIL import Image

# Visualizations
import matplotlib.pyplot as plt

from DiabetesModelPT as DiabetesModel

task = ([0,1], (2,3,4))
# task = ([0,1,2], (3,4))

data = pd.read_csv("data/trainLabels.csv")

train, test = train_test_split(data, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)

train = train.reset_index()
test = test.reset_index()
val = val.reset_index()

data = {'train': DiabeticData(df = train, transform_key="train", root_dir=self.root_dir, task = task),
        'valid': DiabeticData(df = val, transform_key="valid", root_dir=self.root_dir, task = task),
        'test': DiabeticData(df = test, transform_key="test", root_dir=self.root_dir, task = task),
        }

model = DiabetesModel()

model.fit(data['train'], n_epochs=5)
pred, true = model.predict(data['test'])

labels = [p >= 0.5 for p in pred]
print(accuracy_score(true, labels))
print(recall_score(true, labels))
print(precision_score(true, labels))
print(f1_score(true, labels))
print(roc_auc_score(true, pred))