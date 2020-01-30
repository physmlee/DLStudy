#########################################################
##                                                     ##
## MNISTtoROOT.py                                      ##
## Written by Seungmok Lee                             ##
## Seoul Nat'l Univ.                                   ##
## Department of Physics and Astronomy                 ##
## email: physmlee@gmail.com                           ##
## git : https://github.com/physmlee/DLStudy           ##
## Date: 2020.01.26                                    ##
##                                                     ##
## Tested Enviornment                                  ##
##   Python     2.7                                    ##
##   ROOT       6.18/04                                ##
##   tensorflow 1.14.0                                 ##
##   keras      2.3.1                                  ##
## In Ubuntu 18.04 LTS                                 ##
##                                                     ##
#########################################################
##                                                     ##
##                   INSTRUCTION                       ##
##                                                     ##
## This macro downloads MNIST dataset and converts it   #
## to root file.                                       ##
## The output is written at "./data/MNIST.root" file.  ##
##                                                     ##
## Run this by typing                                  ##
##                                                     ##
##   >> python MNIST.py                                ##
##                                                     ##
#########################################################
##                                                     ##
## Refernce                                            ##
## [1] https://colab.research.google.com/github/AviatorMoser/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb#scrollTo=nHLN8vcb6rws
##     MNIST in Keras Example Code                     ##
## [2] https://root.cern.ch/doc/master/pyroot002__TTreeAsMatrix_8py.html
##     PyRoot Writing Root File Example Code           ##
##                                                     ##
#########################################################

from ROOT import TMVA, TFile, TTree
from subprocess import call
import os.path
from os import path

import numpy as np
import matplotlib.pyplot as plt
import random
from array import array
from tqdm import tqdm

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# Make Output ROOT File
if not path.isdir( './data' ):
    os.mkdir( 'data' )
file = TFile( './data/MNIST.root', 'recreate' )

# Load MNIST Data Set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Flattening Data
X_train = X_train.reshape( 60000, 784 ) # reshape 60,000 28 x 28 matrices into 60,000 784-length vectors.
X_test  = X_test .reshape( 10000, 784 ) # reshape 10,000 28 x 28 matrices into 10,000 784-length vectors.

X_train = X_train.astype( 'float32' )   # change integers to 32-bit floating point numbers
X_test  = X_test .astype( 'float32' )

X_train /= 255                        # normalize each value for each pixel for the entire vector for each input
X_test  /= 255

# Make Tree
nb_classes = 10 # number of unique digits
treelist = []

for i in range(nb_classes):
    treelist.append( TTree( 'train%d' %(i), 'MNIST_train%d' %(i) ) )
    treelist.append( TTree( 'test%d'  %(i), 'MNIST_test%d'  %(i) ) )

# Set Branch
imagelist = []
for tree in treelist:
    for j in range( 28 * 28 ):
        imagelist.append( np.empty( (1), dtype="float32" ) )
        tree.Branch( 'image%d' %(j), imagelist[j], 'image%d/F' %(j) )

# Fill Tree
for i in tqdm( range( X_train.shape[0] ), desc='Writing Training Data' ): # Draw progress bar
    for j in range( 28 * 28 ):
        imagelist[j][0] = X_train[i][j] # Load data
    treelist[ 2*y_train[i] ].Fill() # Fill the tree

for i in tqdm( range( X_test.shape[0] ),  desc='Writing   Test   Data' ): # Draw progress bar
    for j in range( 28 * 28 ):
        imagelist[j][0] = X_test[i][j] # Load data
    treelist[ 2*y_test[i] + 1 ].Fill() # Fill the tree

# Write and Close
file.Write()
file.Close()

# Code Ends Here
