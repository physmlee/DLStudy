#########################################################
##                                                     ##
## MNISTtoHDF.py                                       ##
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
## This macro converts MNIST dataset into hdf5 format. ##
## To test its memory management ability, this macro   ##
## will save the dataset 20 times in the output file.  ##
## To load all of the data on the cache memory, about  ##
## 17.6 GB must be consumed, which is costly (or even  ##
## impossible) for many computers.                     ##
##                                                     ##
## Run this by typing                                  ##
##                                                     ##
##   >> python MNISTtoHDF.py                           ##
##                                                     ##
## The output will be written in './data/MNIST.hdf5'   ##
## file.                                               ##
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

import uproot as ur
import numpy as np
import h5py

import os.path
from os import path

from keras.datasets import mnist
from keras.utils import np_utils

# Make Output Directory
if not path.isdir( './data' ):
    os.mkdir( 'data' )

filename = "./data/MNIST.hdf5"

# (Re-)Create output file
with h5py.File(filename, 'w') as f:
    # Download mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Process data as usual
    X_train = X_train.reshape( 60000, 784 )
    X_test  = X_test .reshape( 10000, 784 )

    X_train = X_train.astype( 'float16' ) # change integers to 16-bit floating point numbers
    X_test  = X_test .astype( 'float16' )

    X_train /= 255
    X_test  /= 255
    
    nb_classes = 10

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # Create an empty dataset able to hold up to 1000 elements.
    d_X_train = f.create_dataset('X_train', (60000 * 20, 784), dtype = 'float16')
    d_X_test  = f.create_dataset('X_test' , (10000 * 20, 784), dtype = 'float16')
    d_Y_train = f.create_dataset('Y_train', (60000 * 20,  10), dtype = 'float16')
    d_Y_test  = f.create_dataset('Y_test' , (10000 * 20,  10), dtype = 'float16')
    
    # Write mnist dataset 20 times. It will create a hdf5 file of 2.2 GB.
    for i in range(20):
        d_X_train[i * 60000 : (i+1) * 60000] = X_train
        d_X_test [i * 10000 : (i+1) * 10000] = X_test
        d_Y_train[i * 60000 : (i+1) * 60000] = Y_train
        d_Y_test [i * 10000 : (i+1) * 10000] = Y_test

# Code Ends Here
