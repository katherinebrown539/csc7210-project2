import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Conv2D, Dropout, MaxPooling2D, Dense
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
        
model.add(Conv2D(32, 7, strides=(2,2), activation="relu", input_shape=input_shape_))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(32, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(64, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(64, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(128, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(128, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(256, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(256, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(256, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(Conv2D(256, 3, strides=(1,1), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dense(1024, activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(1024, activation="relu"))
model.add(Dense(2, activation="softmax"))




model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])


#fit model
model.fit(x=train_generator, 
          batch_size=batch_size,
          epochs=epochs)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_generator)
print("test loss, test acc:", results)
