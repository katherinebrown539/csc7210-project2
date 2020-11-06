from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np

filename = "data/dogcat_ad.csv"
root_dir = "data/dogcat/train"
classes = ['cat', 'dog']
x='filename'
y='label'
# task = ([0,1,2], (3,4))

data = pd.read_csv(filename)
# data = data.sample(frac=0.25)
train, test = train_test_split(data, test_size=0.1)
train, val = train_test_split(train, test_size=0.1)

train = train[train["label"] == 1]
print(train)
#filter out 1s from training set

train = train.reset_index()
test = test.reset_index()
val = val.reset_index()


train_generator=datagen.flow_from_dataframe(
    dataframe=train,
    directory=root_dir,
    x_col=x,
    y_col=y,
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="binary",
    target_size=(100,100))

valid_generator=datagen.flow_from_dataframe(
    dataframe=val,
    directory=root_dir,
    x_col=x,
    y_col=y,
    subset="validation",
    batch_size=32,
    seed=42,
    shuffle=True,
    class_mode="binary",
    target_size=(100,100))

test_generator=test_datagen.flow_from_dataframe(
    dataframe=test,
    directory=root_dir
    x_col=x,
    y_col=y,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode='binary',
    target_size=(100,100))

input_img = keras.Input(shape=(100, 100, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(train_generator)