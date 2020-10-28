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
