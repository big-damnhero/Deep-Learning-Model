#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import scipy.stats
import random
import math
import keras
from keras.models import Sequential
from keras import *
from keras.optimizers import *
from keras.callbacks import *
from keras.layers import *
from keras.utils import plot_model
from tabulate import tabulate
from scipy import ndimage,misc
#matplotlib.rcParams['figure.figsize']=(20.0,10.0)


# In[2]:


train_size=10
healthyPrefix = "../input/negglaucoma/healthy/"
glaucomaPrefix = "../input/posglaucoma/glaucoma/"
originalGlaucomaTrain = [None] * train_size
originalHealthyTrain = [None] * train_size
nRows,nCols =64, 64
trainX=np.zeros((train_size*2, nRows, nCols))
for i in range(train_size):
    if i < 9:
        picname=glaucomaPrefix+'0'+str(i+1)
    else:
        picname=glaucomaPrefix + str(i+1)
    eye=scipy.ndimage.imread(picname+'_g.jpg',flatten=True)
    eye = misc.imresize(eye, (nRows, nCols))
#     print(picname+'_g.jpg')
    originalGlaucomaTrain[i]=eye;
#     print(eye.shape[:])
    #eye=eye.flatten();
    trainX[i,:]=eye;#downSample(eye,100)/256.0;
for i in range(train_size):
    if i < 9:
        picname= healthyPrefix + '0'+str(i+1)
    else:
        picname= healthyPrefix + str(i+1)
    eye=scipy.ndimage.imread(picname+'_h.jpg',flatten=True)
    eye = misc.imresize(eye, (nRows, nCols))
#     print(picname+'_h.jpg')
    originalHealthyTrain[i]=eye;
    #eye=eye.flatten();
    #prin
    trainX[i+train_size,:]=eye#;=downSample(eye,100);#/256.0;


# In[3]:


train_size=10
test_size=5
originalGlaucomaTest = [None] * test_size
originalHealthyTest = [None] * test_size
testX=np.zeros((test_size*2, nRows, nCols))
for i in range(test_size):
    picname= glaucomaPrefix + str(train_size+i+1)
    eye=scipy.ndimage.imread(picname+'_g.jpg',flatten=True)
    eye = misc.imresize(eye, (nRows, nCols))
#     print(eye)
#     print(picname+'_g.jpg')
    originalGlaucomaTest[i]=eye;
    #eye=eye.flatten();
    testX[i,:]=eye#downSample(eye,100)/256.0;
for i in range(test_size):
    picname=healthyPrefix + str(train_size+i+1)
    eye=scipy.ndimage.imread(picname+'_h.jpg',flatten=True)
    eye = misc.imresize(eye, (nRows, nCols))
#     print(picname+'_h.jpg')
    originalHealthyTest[i]=eye;
   # eye=eye.flatten();
    testX[i+5,:]=eye#downSample(eye,100)/256.0;
    


# In[4]:


# Find the shape of input images and create the variable input_shape
nDims=1
train_data = trainX.reshape(trainX.shape[0], nRows, nCols, nDims)
test_data = testX.reshape(testX.shape[0], nRows, nCols, nDims)
input_shape = (nRows, nCols, 3)

# Change to float datatype
train_X = train_data.astype('float32')
test_X = test_data.astype('float32')

# Scale the data to lie between 0 to 1
train_X /= 255.0
test_X /= 255.0


# In[5]:


print(train_X.shape)
print(test_X.shape)


# In[6]:


train_Y=np.zeros((2*train_size,1))
test_Y=np.zeros((2*test_size,1))
for i in range(train_size):
    train_Y[i]=1;
for i in range(test_size):
    test_Y[i]=1
print(len(train_Y),len(test_Y),len(train_X),len(test_X))


# In[7]:


def augmentation(X, y):
    X1 = np.rot90(X, 1, (2, 1))
    X2 = np.rot90(X, 2, (2, 1))
    X3 = np.rot90(X, 3, (2, 1))
    X4 = np.array([np.fliplr(x) for x in X])
    X5 = np.array([np.flipud(x) for x in X])
    return np.concatenate((X, X1, X2, X3, X4)), np.concatenate((y, y, y, y, y))


# In[8]:


train_X, train_Y = augmentation(train_X, train_Y)
test_X, test_Y = augmentation(test_X, test_Y)
train_X = np.tile(train_X, (1, 1, 1, 3))
test_X = np.tile(test_X, (1, 1, 1, 3))
print(train_X.shape, train_Y.shape)
print(test_X.shape, test_Y.shape)


# In[17]:


def create_model(input_dim):
    model = Sequential()
    ##############
    model.add(Conv2D(200, (9, 9), padding = 'valid', input_shape=input_dim))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    ################
    ##############
    model.add(Conv2D(200, (7, 7), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    ################
    ##############
    model.add(Conv2D(250, (5, 5), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    ################
    ##############
    model.add(Conv2D(250, (2, 2), padding = 'valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    ################
    model.add(Flatten())
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.summary()
#     add_model(base_model.output).summary()
    return model


# In[18]:


model=create_model(input_shape)


# In[19]:


# for layer in model.layers[:-1]:
#     layer.trainable = False


# In[20]:


model.summary()


# In[ ]:


optimizer = optimizers.SGD(lr=1e-2, momentum=0.9, decay = 1e-6)
model.compile(optimizer= optimizer, loss='binary_crossentropy',metrics=['accuracy'])

checkpointer = ModelCheckpoint('/home/model_vgg.h5', monitor = 'val_acc', 
                              verbose = 1, save_best_only = True)

#not being used
reduce_lr = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.5, patience = 30, 
                             verbose = 1, min_lr = 0.0001)

history = model.fit(train_X, train_Y,epochs= 20, validation_data=(test_X, test_Y), 
         callbacks = [checkpointer])


# In[ ]:


plt.imshow(train_X[2, :, :, 0])
print(train_Y[2])


# In[ ]:


from keras.models import load_model
models = load_model("/home/model_vgg.h5")


# In[ ]:


trainScore, trainAcc = models.evaluate(train_X, train_Y)
print("Training Accuracy = ", trainAcc)
testScore, testAcc = models.evaluate(test_X,test_Y)
print("Validation Accuracy = ", testAcc)


# In[ ]:


fig, (ax_loss, ax_acc) = plt.subplots(2, 1, sharex = 'col', sharey = 'row')
fig.set_size_inches(15, 16)  #width, height

ax_loss.plot(history.history['loss'], label = 'train_loss')
ax_loss.plot(history.history['val_loss'], label = 'val_loss')
# ax_loss.plot(history.history['lr'], label = 'learning_rate')
ax_loss.legend()
# plt.xticks(lrepoch)
ax_loss.grid(True, which = 'both')
ax_loss.set_xlabel("num_epoch")
ax_loss.set_ylabel("loss/lr")
ax_loss.set_title("Loss")
ax_acc.plot(history.history['acc'], label = 'train_acc')
ax_acc.plot(history.history['val_acc'], label = "val_acc")
# ax_acc.plot(history.history['lr'], label = 'learning_rate')
ax_acc.legend()
ax_acc.grid(True, which = 'both')
# plt.xticks(lrepoch)
# ax_acc.set_xlabel("num_epoch")
ax_acc.set_ylabel("acc/lr")
ax_acc.set_title("Acc")


# In[ ]:


ans = model.predict(train_X)
print(np.hstack((ans, train_Y)))


# In[ ]:


print(ans.shape)


# In[ ]:


print(10,"\n")


# In[ ]:





# In[ ]:





# In[ ]:




