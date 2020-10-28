import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
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
#generate model
base_model = keras.applications.DenseNet121(weights='imagenet', include_top=False)
x = base_model.output
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional VGG16 layers
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])


#fit model
model.fit(x=train_generator, 
          batch_size=batch_size,
          epochs=epochs)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_generator)
print("test loss, test acc:", results)
