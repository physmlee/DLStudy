#########################################################
##                                                     ##
## MNIST_hdf5.py                                       ##
## Written by Seungmok Lee                             ##
## Seoul Nat'l Univ.                                   ##
## Department of Physics and Astronomy                 ##
## email: physmlee@gmail.com                           ##
## git : https://github.com/physmlee/DLStudy           ##
## Date: 2020.02.10                                    ##
##                                                     ##
## Tested Enviornment                                  ##
##   Python		2.7                                    ##
##   tensorflow	1.14.0                                 ##
##   keras		2.3.1                                  ##
##   h5py       2.7.1                                  ##
## In Ubuntu 18.04 LTS                                 ##
##                                                     ##
#########################################################
##                                                     ##
##                   INSTRUCTION                       ##
##                                                     ##
## In keras, to train with dataset larger than the     ##
## machine's memory capacity, HDF5 and DataGenerator   ##
## are considerable. Both of them read training data   ##
## from disk when requested, rather than loading all   ##
## the dataset on the memory first. Despite they seems ##
## similar, using HDF5 is much faster and easier than  ##
## using DataGenerator.                                ##
##                                                     ##
## This macro reads dataset from hdf5 file. HDF5Matrix ##
## class allows us to load data only if requested. We  ##
## can use the loaded dataset as if it was a numpy     ##
## array. If you monitor your system resource (eg.     ##
## use 'top' command in linux), you can observe that   ##
## this macro only occupies small part of memory. It   ##
## also takes not that longer than the original MNIST  ##
## code. It took about 23 times longer for me, which   ##
## is obvious considering that we have 20 times larger ##
## dataset to train.                                   ##
##                                                     ##
## Run this by typing                                  ##
##                                                     ##
##   >> python MNIST_hdf5.py                           ##
##                                                     ##
#########################################################
##                                                     ##
## Refernce                                            ##
## [1] https://colab.research.google.com/github/AviatorMoser/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb#scrollTo=nHLN8vcb6rws
##     MNIST in Keras Example Code                     ##
## [2] https://www.pythonforthelab.com/blog/how-to-use-hdf5-files-in-python/
##     How To Use HDF5 Files In Python                 ##
##                                                     ##
#########################################################

from subprocess import call
from os.path import isfile

import numpy as np
import matplotlib.pyplot as plt
import random

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

from keras.utils.io_utils import HDF5Matrix

filename = './data/MNIST.hdf5'

# Read dataset from hdf5 file
X_train = HDF5Matrix(filename, 'X_train')
X_test  = HDF5Matrix(filename, 'X_test')
Y_train = HDF5Matrix(filename, 'Y_train')
Y_test  = HDF5Matrix(filename, 'Y_test')

# Build model as usual
model = Sequential()

model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# You can give them just as numpy array.
model.fit(X_train, Y_train,
          batch_size=128, epochs=5, 
          shuffle = "batch", # You should pass shuffle = "batch" when using HDF5Matrix as input.
          verbose=1)

# Test the trained model
score = model.evaluate(X_test, Y_test)
print('Test score:', score[0])
print('Test accuracy:', score[1])

