




import pandas as pd
import numpy as np
import sys
import os
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def getDataGeneratorsKeras(data_file):
    #use sklearn to split data into training, test, val
    df = pd.read_csv(data_file)
    x = 'image'
    y = 'level'
    df.loc[df[y] >= 2, "level_new"] = 1
    df.loc[df[y] < 2, "level_new"] = 0
    y = "level_new"
    df[y] = df[y].astype(str)
    pth = 'data/diabetes/'
    img_size = (224,224)
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


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Dropout, MaxPooling2D, Dense, Activation
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import os
os.sys.path.insert(0, ".")
import DiabetesData

#get diabetes dataloaders
train_generator, val_generator, test_generator = DiabetesData.getDataGenerators(data_file="data/trainLabels.csv")
batch_size = 16
epochs = 1
input_shape_=(3,224,224)
model = Sequential()


model.add(Conv2D(32, (3, 3), input_shape=input_shape_))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])


#fit model
model.fit(x=train_generator, 
          batch_size=batch_size,
          epochs=epochs)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_generator)
print("test loss, test acc:", results)
