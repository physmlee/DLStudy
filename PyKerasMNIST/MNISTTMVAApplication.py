#########################################################
##                                                     ##
## MNISTTMVAApplication.py                             ##
## Written by Seungmok Lee                             ##
## Seoul Nat'l Univ.                                   ##
## Department of Physics and Astronomy                 ##
## email: physmlee@gmail.com                           ##
## git : https://github.com/physmlee/DLStudy           ##
## Date: 2020.01.30                                    ##
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
## This macro is an example multiclass classifier      ##
## application for MNIST dataset using TMVA methods.   ##
## This macro loads the trained weight file, which     ##
## must be generated by 'MNISTTMVA.py' macro before    ##
## running. This macro evaluates the accuracy,         ##
## applying all the MNIST dataset. If the data file is ##
## not found, getenates dataset running                ##
## 'MNISTtoROOT.py' python macro.                      ##
##                                                     ##
## Run this by typing                                  ##
##                                                     ##
##   >> python MNISTTMVAApplication.py                 ##
##                                                     ##
#########################################################
##                                                     ##
## Refernce                                            ##
## [1] https://root.cern/doc/master/ApplicationClassificationKeras_8py_source.html
##     TMVA Class. Application Keras Example Code      ##
## [2] https://github.com/root-project/root/blob/master/documentation/tmva/UsersGuide/TMVAUsersGuide.pdf
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
from tqdm import tqdm

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# Check if trained weight file exists
weightfilename = './dataset/weights/TMVAClassification_PyKerasMNIST.weights.xml'
if not isfile( weightfilename ):
    print( '[Error]  Could not open weight file.' )
    print( '[Errpr]  Please run MNISTTMVA.py first' )
    print( '[Error]  Exit.' )
    sys.exit()

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
reader = TMVA.Reader("Color:!Silent")
output = TFile.Open('./data/MNISTTMVA.root')

# Load data trees
nb_classes = 10 # number of classes
traintree = []
testtree  = []

for i in range( nb_classes ):
    traintree.append( data.Get( 'train%d' %(i) ) ) # Load training trees
    testtree .append( data.Get( 'test%d'  %(i) ) ) # Load testing trees

# Register variables
imagelist = []
for i in range( 28 * 28 ):
    imagelist.append( np.empty( (1), dtype="float32" ) )
    reader.AddVariable('image%d' %(i), imagelist[i]) # register all the 784 pixels as variable
    for tree in traintree:
        tree.SetBranchAddress('image%d' %(i), imagelist[i])
    for tree in testtree:
        tree.SetBranchAddress('image%d' %(i), imagelist[i])

# Book methods
methodname = 'PyKeras'
reader.BookMVA(methodname, weightfilename)

# Evaluate train/test accuracy
trainErr = 0
testErr = 0
for i in range(nb_classes):
    print('')
    print('<<Class %d>>' %(i))

    eventnum = traintree[i].GetEntries()
    for event in tqdm( range( eventnum ), desc='Applying to training data' ): # Draw progress bar
        traintree[i].GetEntry(event) # Get entry
        ans = 0
        maxprob = 0

        for cand in range(nb_classes): # For all the candidates,
            prob = reader.EvaluateMulticlass(methodname)[cand]
            if prob > maxprob:
                ans = cand # Which one has the largest probability?
                maxprob = prob # With what probability?
        if ans != i: # If the answer is wrong...
            trainErr += 1 # ... add 1 to trainErr
    
    eventnum = testtree[i].GetEntries()
    for event in tqdm( range( eventnum ), desc='Applying to testing  data' ): # Draw progress bar
        testtree[i].GetEntry(event) # Get entry
        ans = 0
        maxprob = 0

        for cand in range(nb_classes): # For all the candidates,
            prob = reader.EvaluateMulticlass(methodname)[cand]
            if prob > maxprob:
                ans = cand # Which one has the largest probability?
                maxprob = prob # With what probability?
        if ans != i: # If the answer is wrong...
            testErr += 1 # ... add 1 to testErr

# Print the result
print('Training accuracy: %f' %((60000.0 - trainErr) / 60000.0))
print('Testing  accuracy: %f' %((10000.0 - testErr ) / 10000.0))

# Code Ends Here