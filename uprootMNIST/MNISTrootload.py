from keras import models
from keras import layers
from keras.utils import to_categorical 
import uproot
import numpy as np
import awkward
import random
from tqdm import tqdm

file = uproot.open("MNIST.root")

trtreelist = []
trbrlist = []
tetreelist = []
tebrlist = []
for i in tqdm(range(10), desc='Load Tree'):
	trtreelist.append(file['train%d' %(i)])
	tetreelist.append(file['test%d' %(i)])
	trbrlist.append(trtreelist[i].arrays(namedecode ='utf-8'))
	tebrlist.append(tetreelist[i].arrays(namedecode ='utf-8'))

table = awkward.Table(trbrlist[0])

xtrainarr = []
xtestarr  = []
ytrainarr = []
ytestarr  = []
for j in tqdm(range(10), desc='Load train Data'):
	for i in range(len(trbrlist[j]['image0'])):
		arr = []
		for column_name in table[0]:
			arr.append(trbrlist[j][column_name][i])
		xtrainarr.append(arr)
		ytrainarr.append(j)

for j in tqdm(range(10), desc='Load test Data'):
	for i in range(len(tebrlist[j]['image0'])):
		arr = []
		for column_name in table[0]:
			arr.append(tebrlist[j][column_name][i])
		xtestarr.append(arr)
		ytestarr.append(j)

del tebrlist
del trbrlist
del trtreelist
del tetreelist

print(len(xtrainarr))
print(len(ytrainarr))

combine = list(zip(xtrainarr,ytrainarr))
random.shuffle(combine)
xtrainarr,ytrainarr = zip(*combine)

combine2 = list(zip(xtestarr,ytestarr))
random.shuffle(combine2)
xtestarr,ytestarr = zip(*combine2)

X_train = np.array(xtrainarr)
Y_train = np.array(ytrainarr)
X_test = np.array(xtestarr)
Y_test = np.array(ytestarr)
del xtrainarr
del ytrainarr
del xtestarr
del ytestarr

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5, batch_size=128)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('test_acc: ', test_acc)

