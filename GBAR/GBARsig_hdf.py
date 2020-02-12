#Python version : 2.7.5

from keras import models
from keras import layers
from keras.utils import to_categorical 
import uproot
import numpy as np
import random
from tqdm import tqdm
from keras.utils.io_utils import HDF5Matrix

filename = "SIGNALMCwaveform.hdf5"

X_train = HDF5Matrix(filename, 'X_train')
X_test  = HDF5Matrix(filename, 'X_test')
Y_train = HDF5Matrix(filename, 'Y_train')
Y_test  = HDF5Matrix(filename, 'Y_test')


model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(112 * 88,)))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(2, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=20,shuffle='batch', batch_size=128)
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('test_acc: ', test_acc)
