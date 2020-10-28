import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def getDataGenerators(data_file):
    #use sklearn to split data into training, test, val
    df = pd.read_csv(data_file)
    x = 'image'
    y = 'level'
    df.loc[df[y] >= 2, "level_new"] = 1
    df.loc[df[y] < 2, "level_new"] = 0
    y = "level_new"
    df[y] = df[y].astype(str)
    pth = 'data/diabetes/'
    img_size = (100,100)
    batch_size = 16

    train, test_df = train_test_split(df, test_size = 0.1, random_state=random.randint(1,100))
    train_df, val_df = train_test_split(train, test_size=0.1, random_state=random.randint(1,100))

    #create data generators

    # Base train/validation generator
    _datagen = ImageDataGenerator(
        rescale=1./255.,
        validation_split=0.25,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
        )
    # Train generator
    train_generator = _datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=pth,
        x_col=x,
        y_col=y,
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        target_size=img_size)
    print('Train generator created')
    # Validation generator
    val_generator = _datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=pth,
        x_col=x,
        y_col=y,
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        target_size=img_size)    
    print('Validation generator created')
    # Test generator
    _test_datagen=ImageDataGenerator(rescale=1./255.)
    test_generator = _test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=pth,
        x_col=x,
        y_col=y,
        batch_size=batch_size,
        shuffle=True,
        class_mode="categorical",
        target_size=img_size)     
    print('Test generator created')

    return train_generator, val_generator, test_generator

train_generator, val_generator, test_generator = getDataGenerators(data_file="data/trainLabels.csv")

x,y = train_generator.next()
for i in range(0,3):
    image = x[i]
    label = y[i]
    print (label)