#########################################################
##                                                     ##
## MNISTTMVA.py                                        ##
## Written by Seungmok Lee                             ##
## Seoul Nat'l Univ.                                   ##
## Department of Physics and Astronomy                 ##
## email: physmlee@gmail.com                           ##
##                                                     ##
## 2020.01.26.                                         ##
## With Python 2.7.17 and ROOT 6.18/04                 ##
## In Ubuntu 18.04 LTS                                 ##
##                                                     ##
#########################################################
##                                                     ##
##                   INSTRUCTION                       ##
##                                                     ##
## This macro is an example multiclass classifier for  ##
## MNIST dataset using TMVA methods. It loads the data ##
## file, and then carries out the DNN. If the data     ##
## file is not found, generates it running             ##
## 'MNISTtoROOT.py' macro.                             ##
##                                                     ##
## Run it by typing                                    ##
##                                                     ##
##   >> python MNISTTMVA.py                            ##
##                                                     ##
## It will save the model architecture file in         ##
## 'PyKerasMNIST.h5'. Classification result will be    ##
## saved in 'dataset' directory. Please refer to the   ##
## TMVA Users Guide about how to read the output file. ##
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

# Load MNIST dataset ROOT file
datafilename = './data/MNIST.root'
if not isfile( datafilename ):
    print( '  ' + datafilename + ' file was not found.' )
    print( '  ' + 'You should run MNISTtoROOT.py first.' )
    print( '  ' + 'Trying to run MNISTtoROOT.py...' )

    if isfile( 'MNISTtoROOT.py' ):
        print( '  ' + 'Running MNISTtoROOT.py' )
        os.system( 'python MNISTtoROOT.py' )

        if not isfile( datafilename ):
            print( '  ' + 'MNISTtoROOT.py failed. Exiting.' )
            sys.exit()

    else:
        print( '  ' + 'MNISTtoROOT.py not found. Exiting.' )
        sys.exit()

data = TFile.Open(datafilename)


# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()
output = TFile.Open('./data/MNISTTMVA.root', 'RECREATE')
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

dataloader.PrepareTrainingAndTestTree(TCut(''),
                                     '!CalcCorrelations:' # Skip calculating decorrelation matrix
                                     'NormMode=None:' # Normalization makes the entry numbers of each class to be equal. It is not our business.
                                     '!V') # No verbose option

# Model generating start
# Stack layers linearly. It is very common option.
model = Sequential()

# First hidden layer
model.add(Dense(512, input_shape=(784,))) #(784,) is not a typo -- that represents a 784 length vector!
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Second hidden layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# Final output layer
model.add(Dense(10)) # MNIST has 10 output categories
model.add(Activation('softmax'))

# Compile and set loss and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save the compiled model
model.save('PyKerasMNIST.h5')

# Model generating end. Show the built model summary
model.summary()

# Book methods
factory.BookMethod( dataloader, TMVA.Types.kPyKeras, "PyKerasMNIST",
                    '!H:!V:VarTransform=:'
                    'FilenameModel=PyKerasMNIST.h5:'
                    'NumEpochs=5:' # Train 5 times
                    'BatchSize=128') # Calculate gradient descent using 128 samples

# Run TMVA
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

# Code Ends Here
