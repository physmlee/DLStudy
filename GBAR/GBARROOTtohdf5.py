#Python version : 2.7.5

from keras import models
from keras import layers
from keras.utils import to_categorical 
import uproot
import numpy as np
import random
from tqdm import tqdm
import h5py

filename = "SIGNALMCwaveform.hdf5"
trainsize = 200 # for a tree total train sample is 2*trainsize(up,dw)
testsize = trainsize/4
start = 0
startfortest = 0
startfortrain = 0
with h5py.File(filename,'w') as f:		
# In Python, we usually treat file thru 'with'.
# Outside the with block, the file becomes automatically closed.

	d_X_train = f.create_dataset('X_train', (160000,112*88), dtype='float32',compression='gzip',compression_opts=9)
	d_X_test = f.create_dataset('X_test', (40000,112*88), dtype='float32',compression='gzip',compression_opts=9)
	d_Y_train = f.create_dataset('Y_train', (160000, 2), dtype='float32',compression='gzip',compression_opts=9)
	d_Y_test = f.create_dataset('Y_test', (40000, 2), dtype='float32',compression='gzip',compression_opts=9)

	upfile = uproot.open("upSignalMC_evrec.root")
	dwfile = uproot.open("dwSignalMC_evrec.root")

	upTree = upfile["EvRec"]
	dwTree = dwfile["EvRec"]

	for k in range(0,400):
		start = k*(testsize+trainsize)
		startfortest = k*testsize
		startfortrain = k*trainsize
		uptr = upTree.array(branch='waveform',entrystart=start, entrystop=start+trainsize+testsize)
		dwtr = dwTree.array(branch='waveform',entrystart=start, entrystop=start+trainsize+testsize)

		X_train = np.array([])
		Y_train = np.array([])
		X_test = np.array([])
		Y_test = np.array([])

		for i in tqdm(range(trainsize), desc='Load up data'):
			arr = np.array([])
			for j in range(88):
				arr=np.concatenate((arr,uptr[i][j]),axis=None)
			X_train=np.append(X_train,arr,axis=0)
			Y_train=np.append(Y_train,0)


		for i in tqdm(range(trainsize), desc='Load dw data'):
			arr = np.array([])
			for j in range(88):
				arr=np.concatenate((arr,dwtr[i][j]),axis=None)
			X_train=np.concatenate((X_train,arr),axis=0)
			Y_train=np.append(Y_train,1)

		for i in tqdm(range(trainsize,trainsize+testsize), desc='Load up data'):
			arr = np.array([])
			for j in range(88):
				arr=np.concatenate((arr,uptr[i][j]),axis=None)
			X_test=np.concatenate((X_test,arr),axis=0)
			Y_test=np.append(Y_test,0)

		for i in tqdm(range(trainsize,trainsize+testsize), desc='Load dw data'):
			arr = np.array([])
			for j in range(88):
				arr=np.concatenate((arr,dwtr[i][j]),axis=None)
			X_test=np.concatenate((X_test,arr),axis=0)
			Y_test=np.append(Y_test,1)



		X_train=X_train.reshape(2*trainsize,112*88)
		X_test=X_test.reshape(2*testsize,112*88)

		X_train = X_train.astype('float32')
		X_test = X_test.astype('float32')
		X_train/=4095
		X_test/=4095

		s=np.arange(X_train.shape[0])
		np.random.shuffle(s)
		X_train = X_train[s]
		Y_train = Y_train[s]

		s2=np.arange(X_test.shape[0])
		np.random.shuffle(s2)
		X_test = X_test[s2]
		Y_test = Y_test[s2]

		Y_train = to_categorical(Y_train)
		Y_test = to_categorical(Y_test)

		d_X_train[2*startfortrain:2*(startfortrain+trainsize)] = X_train
		d_Y_train[2*startfortrain:2*(startfortrain+trainsize)] = Y_train
		d_X_test[2*startfortest:2*(startfortest+testsize)] = X_test
		d_Y_test[2*startfortest:2*(startfortest+testsize)] = Y_test

