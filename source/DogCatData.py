# PyTorch Imports
from torchvision import transforms
from torch.utils.data import Dataset
import torch
# Data Science | Image Tools | MatplotLib
import numpy as np
import pandas as pd
import os, sys, shutil, time, argparse

# Image manipulations
from PIL import Image

class DogCatData(Dataset):
    '''
        Class to process a custom image dataset. Particularly, this class is designed to handle the 
        Chest XRay data https://stanfordmlgroup.github.io/competitions/chexpert/

        Class Variables:
            * data_frame: this is a dataframe containing the relative file paths and labels 
            * test: boolean; True => this dataset does not have labels, only image paths
            * root_dir: string pointing to the root of the image directory, paths in the data_frame are to be
                        concatenated to this value to yield the correct path
            * transform: A pytorch transform pipeline of preprocessing operations to conduct to prepare
                         the image for the model
            * 
    '''
    def __init__(self,df,root_dir,transform_key=None, test = False, normalize=True):
        '''
            Constructor for Dataset class. This method assigns the class variables based on the parameters

            Parameters
                * df: this is a dataframe containing the relative file paths and labels 
                * root_dir: string pointing to the root of the image directory, paths in the data_frame are to be
                            concatenated to this value to yield the correct path
                * transform: Optional; A pytorch transform pipeline of preprocessing operations to conduct to prepare
                             the image for the model; if None/no argument provided, method define_image_transforms is called
                * task: defines whether binary classification of diabetic retinopathy severity or a binary classification
                            "multi" => each class is one-hot encoded as a vector
                            tuple => tuple[0] is negative class tuple[1] is positive class
                * test: OPTIONAL; boolean; True => this dataset does not have labels, only image paths
        '''
        self.data_frame = df #passing dataframe instead of csv file because of train/test split
        print(self.data_frame)
        self.test = test 
        self.root_dir = root_dir
        self.transform = self.define_image_transforms(transform_key, normalize)
        #image data 
       
    def define_image_transforms(self, key, normalize=False):
        '''
            This function defines the pipeline of preprocessing operations that is required for an image
            to be processed by a model

            No parameters

            Upon Completion:
                * A dictionary with pipelines for training, validation, and testing data are returned
                * The pipeline includes (1) resizing the image to 224x224
                                        (2) converting the image to a pytorch tensor
                                        (3) normalizing the image (this is required; idk why)
        '''
        if normalize:
            image_transforms = {
                "train":
                transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
                "valid":
                transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]),
                "test":
                transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            } 
        else:
            image_transforms = {
                "train":
                transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(96),
                transforms.ToTensor()
            ]),
                "valid":
                transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(96),
                transforms.ToTensor()
            ]),
                "test":
                transforms.Compose([
                transforms.Resize(128),
                transforms.CenterCrop(96),
                transforms.ToTensor()
            ])
            } 
        return image_transforms[key]

    def __len__(self):
        '''
            Method inherited from Pytorch Dataset class
            Returns the number of items in the datset
        '''
        return self.data_frame.shape[0]
    
    def __getitem__(self, idx):
        '''
            Method inherited from Pytorch Dataset class
            Returns a given image based on the index passed

            Parameters:
                * idx: integer corresponding to the location of a particular image in the dataframe

            Upon completion, this method will open the image based on the "Path" column and return the image and the label,
            if self.test == False
        '''

        #print(self.task)
        if torch.is_tensor(idx):
            idx = idx.tolist()


        image_name = os.path.join(self.root_dir,
                                self.data_frame.loc[idx, "filename"])
        
        with Image.open(image_name) as img:
            img = Image.open(image_name)
            img = img.convert('RGB')
        #print(img.shape)
            img_tensor = self.transform(img)
        if not self.test:
            image_label = self.data_frame.loc[idx, "label"]
            i=0
            #return one-hot encoded tensor
            label_tensor = torch.tensor(image_label, dtype=torch.long)
            image_label = label_tensor.clone()
            return(img_tensor, image_label)
        return (img_tensor)

