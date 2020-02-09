#Python version : 2.7.5

from keras import models
from keras import layers
from keras.utils import to_categorical 
import uproot
import numpy as np
import random
from tqdm import tqdm

file = uproot.open("MNIST.root")

TrainTrlist = [] #train tree load
TestTrlist = [] #test tree load
TrainBrancheslist = [] #same with tree but key decoding is changed to unicode
TestBrancheslist =  []

for i in tqdm(range(10), desc='Load Tree'):
	TrainTrlist.append(file['train%d' %(i)])
	TestTrlist.append(file['test%d' %(i)])
	TrainBrancheslist.append(TrainTrlist[i].arrays(namedecode='utf-8'))
	TestBrancheslist.append(TestTrlist[i].arrays(namedecode='utf-8'))

BranchesName = TrainBrancheslist[0].keys() #Get Branch names

xtrainarr = []
xtestarr  = []
ytrainarr = []
ytestarr  = []

for j in tqdm(range(10), desc='Load train Data'):
	for i in range(len(TrainBrancheslist[j][BranchesName[0]])):
		arr = []
		for BranchName in BranchesName: #reading a row in the tree
			arr.append(TrainBrancheslist[j][BranchName][i]) 
		xtrainarr.append(arr)
		ytrainarr.append(j) #

for j in tqdm(range(10), desc='Load test Data'):
	for i in range(len(TestBrancheslist[j][BranchesName[0]])):
		arr = []
		for BranchName in BranchesName: #reading a row in the tree
			arr.append(TestBrancheslist[j][BranchName][i]) 
		xtestarr.append(arr)
		ytestarr.append(j)


combine = list(zip(xtrainarr,ytrainarr)) #shuffle data
random.shuffle(combine)
xtrainarr,ytrainarr = zip(*combine)

combine2 = list(zip(xtestarr,ytestarr))
random.shuffle(combine2)
xtestarr,ytestarr = zip(*combine2)


X_train = np.array(xtrainarr)
Y_train = np.array(ytrainarr)
X_test = np.array(xtestarr)
Y_test = np.array(ytestarr)

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('test_acc: ', test_acc)

