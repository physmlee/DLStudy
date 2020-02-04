#########################################################
##                                                     ##
## MNIST_PyMVA_CNN.py                                  ##
## Written by Seungmok Lee                             ##
## Seoul Nat'l Univ.                                   ##
## Department of Physics and Astronomy                 ##
## email: physmlee@gmail.com                           ##
## git : https://github.com/physmlee/DLStudy           ##
## Date: 2020.02.04                                    ##
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
## This macro is an example multiclass CNN for MNIST   ##
## dataset using TMVA PyKeras methods. It loads the    ##
## data file, and then carries out CNN. If the data    ##
## file is not found, generates it running             ##
## 'MNISTtoROOT.py' macro.                             ##
##                                                     ##
## Run this by typing                                  ##
##                                                     ##
##   >> python MNIST_PyMVA_CNN.py                      ##
##                                                     ##
## It will save the model architecture file in         ##
## 'MNIST_PyMVA_CNN_Model.h5'. Classification result   ##
## will be saved in 'dataset' directory. Please refer  ##
## to the TMVA Users Guide about how to read the       ##
## output file.                                        ##
##                                                     ##
#########################################################
##                                                     ##
## Refernce                                            ##
## [1] https://colab.research.google.com/github/AviatorMoser/keras-mnist-tutorial/blob/master/MNIST%20in%20Keras.ipynb#scrollTo=nHLN8vcb6rws
##     MNIST in Keras Example Code                     ##
## [2] https://root.cern/doc/master/MulticlassKeras_8py_source.html
##     TMVA Multiclass Keras Example Code              ##
## [3] https://github.com/root-project/root/blob/master/documentation/tmva/UsersGuide/TMVAUsersGuide.pdf
##     TMVA 4 Users Guide for ROOT >= 6.12 Version     ##
##                                                     ##
#########################################################

from ROOT import TMVA, TFile, TTree, TCut, gROOT
import os.path
from os import path
from os.path import isfile
import sys

import numpy as np
import matplotlib.pyplot as plt
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# import some additional tools
# from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Flatten, Reshape
from keras.layers.normalization import BatchNormalization

# Load MNIST dataset ROOT file
datafilename = './data/MNIST.root'
if not isfile( datafilename ): # If there is no dataset root file...
    print( '  ' + datafilename + ' file was not found.' )
    print( '  ' + 'You should run MNISTtoROOT.py first.' )
    print( '  ' + 'Trying to run MNISTtoROOT.py...' )

    if isfile( 'MNISTtoROOT.py' ):
        print( '  ' + 'Running MNISTtoROOT.py' ) # ... run 'MNISTtoROOT.py' to make the dataset file.
        os.system( 'python MNISTtoROOT.py' )

        if not isfile( datafilename ): # If dataset file was not created...
            print( '  ' + 'MNISTtoROOT.py failed. Exiting.' ) # ... exit.
            sys.exit()

    else: # If there isn't 'MNISTtoROOT.py' file...
        print( '  ' + 'MNISTtoROOT.py not found. Exiting.' ) # ... exit.
        sys.exit()

data = TFile.Open(datafilename)


# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
output = TFile.Open('./data/MNIST_PyMVA_CNN.root', 'RECREATE')
factory = TMVA.Factory( 'TMVAClassification', output,
                        '!V:Color:DrawProgressBar:'
                        '!Silent:' # You can use silent mode instead. Silent mode has fewer outputs.
                        'Transformations=:' # No preprocessing for input variable
                        'AnalysisType=multiclass' ) # It is a multiclass classification example

# Load data trees and variables
nb_classes = 10 # number of classes
weight = 1.0 # default weight
cut = TCut('') # no cut
traintree = []
testtree  = []

for i in range( nb_classes ):
    traintree.append( data.Get( 'train%d' %(i) ) ) # Load training trees
    testtree .append( data.Get( 'test%d'  %(i) ) ) # Load testing trees

dataloader = TMVA.DataLoader( 'dataset' )
for branch in traintree[0].GetListOfBranches():
    dataloader.AddVariable( branch.GetName() ) # register all the 784 pixels as variable

for i in range( nb_classes ):
    dataloader.AddTree( traintree[i], '%d' %(i), weight, cut, TMVA.Types.kTraining ) # Add trees specifying their purpose (Training) 
    dataloader.AddTree( testtree[i] , '%d' %(i), weight, cut, TMVA.Types.kTesting  ) # Add trees specifying their purpose (Testing)

dataloader.PrepareTrainingAndTestTree(cut,
                                     '!CalcCorrelations:' # Skip calculating decorrelation matrix
                                     'NormMode=None:' # Normalization makes the entry numbers of each class to be equal. It is not our business.
                                     '!V') # No verbose option

model = Sequential()                                 # Linear stacking of layers

# Convolution Layer 1
model.add(Reshape((28,28, 1), input_shape=(784,)))
model.add(Conv2D(32, (3, 3))) # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
convLayer01 = Activation('relu')                     # activation
model.add(convLayer01)

# Convolution Layer 2
model.add(Conv2D(32, (3, 3)))                        # 32 different 3x3 kernels -- so 32 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
model.add(Activation('relu'))                        # activation
convLayer02 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
model.add(convLayer02)

# Convolution Layer 3
model.add(Conv2D(64,(3, 3)))                         # 64 different 3x3 kernels -- so 64 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
convLayer03 = Activation('relu')                     # activation
model.add(convLayer03)

# Convolution Layer 4
model.add(Conv2D(64, (3, 3)))                        # 64 different 3x3 kernels -- so 64 feature maps
model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
model.add(Activation('relu'))                        # activation
convLayer04 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
model.add(convLayer04)
model.add(Flatten())                                 # Flatten final 4x4x64 output matrix into a 1024-length vector

# Fully Connected Layer 5
model.add(Dense(512))                                # 512 FCN nodes
model.add(BatchNormalization())                      # normalization
model.add(Activation('relu'))                        # activation

# Fully Connected Layer 6                       
model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes
model.add(Dense(10))                                 # final 10 FCN nodes
model.add(Activation('softmax'))                     # softmax activation

# we'll use the same optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.save('MNIST_PyMVA_CNN_Model.h5')
model.summary()

# Visualize model as graph
try:
    from keras.utils.vis_utils import plot_model
    plot_model(model, to_file='MNIST_PyMVA_CNN_Model.png', show_shapes=True)
except:
    print('[INFO] Failed to make model plot')

# Book method
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, "PyKerasMNIST",
                   '!H:!V:VarTransform=:'
                   'FilenameModel=MNIST_PyMVA_CNN_Model.h5:'
                   'ValidationSize=1:' # I don't want to split my training dataset to validation dataset, but atleast one data must be given to validation dataset.
                   '!SaveBestOnly:'  # Save the last result, not the best one.
                   'NumEpochs=5:'  # Train 5 times
                   'BatchSize=128')  # Calculate gradient descent using 128 samples

# Run TMVA
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

# Code Ends Here
