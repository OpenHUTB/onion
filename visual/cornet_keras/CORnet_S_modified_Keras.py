#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries, packages, etc.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs

import pandas as pd
import argparse
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


# In[ ]:


#Create 'toy' dataset to check whether model compiles

X = np.random.randint(0, 1, size=(50, 224, 224, 3))
Y = np.random.choice([0, 1], size=(50,))
#print(X.shape)
#print(Y.shape)

(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.25, random_state=42)

print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

#Make_blobs
#(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
#y = y.reshape((y.shape[0], 1))
#print(X.shape)
#print(y.shape)


# In[ ]:


#Block used for V2, V4 and IT

class CORblock_S:

    scale = 4  # Scale of the bottleneck convolution channels
    chanDim = -1 #If format is 'channels last'
    
    def __init__(self, out_channels, layer_name, times=1):
        #super().__init__()
        self.layer_name = layer_name
        
        
        self.out_channels = out_channels   
        self.inputs = inputs
        self.times = times
        
        self.conv_input = tf.keras.layers.Conv2D(self.out_channels, (1,1), padding='SAME', name=layer_name+'_convInp')
        self.skip = tf.keras.layers.Conv2D(self.out_channels, (1,1), strides=2, padding='SAME', use_bias=False, name=layer_name+'_skip')
        self.conv1 = tf.keras.layers.Conv2D(self.out_channels*self.scale, (1,1), padding='SAME', use_bias=False, name=layer_name+'_conv1')
        self.conv2 = tf.keras.layers.Conv2D(self.out_channels*self.scale, (3,3), padding='SAME', strides=1, use_bias=False, name=layer_name+'_conv2')
        self.conv3 = tf.keras.layers.Conv2D(self.out_channels,(1,1), padding='SAME', use_bias=False, name=layer_name+'_conv3')
             
    #@staticmethod 
    def CORblock_forward(self, input):
        
        x = self.conv_input(input)

        for t in range(self.times):
            
            if t == 0:
                skip = tf.keras.layers.BatchNormalization(axis=-1, name=self.layer_name+'_BNSkip')(self.skip(x))
            else:
                skip = x

            x = self.conv1(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, name=self.layer_name+'_'+str(t)+'_BN1')(x)
            x = tf.keras.layers.Activation("relu", name=self.layer_name+'_'+str(t)+'_A1')(x)

            x = self.conv2(x)
            if t == 0:  # adding a stride=2 for t=0
                x = tf.keras.layers.MaxPool2D(pool_size=(1, 1), strides=2, padding='SAME')(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, name=self.layer_name+'_'+str(t)+'_BN2')(x)
            x = tf.keras.layers.Activation("relu", name=self.layer_name+'_'+str(t)+'_A2')(x)

            x = self.conv3(x)
            x = tf.keras.layers.BatchNormalization(axis=-1, name=self.layer_name+'_'+str(t)+'_BN3')(x)

            x += skip
            x = tf.keras.layers.Activation("relu", name=self.layer_name+'_'+str(t)+'_AOutput')(x)

        return x


# In[ ]:


#Define & implement model 

# Functional API format
inputs = tf.keras.Input(shape=(224,224,3))

# V1
x = tf.keras.layers.Conv2D(64, (7,7), strides=2, padding='SAME', use_bias=False)(inputs)#Check
x = tf.keras.layers.BatchNormalization(axis=-1)(x)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='SAME')(x)
x = tf.keras.layers.Conv2D(64, (3,3), strides=1, padding='SAME', use_bias=False)(x)
x = tf.keras.layers.BatchNormalization(axis=-1)(x)
x = tf.keras.layers.Activation("relu")(x)

# weight sharing for V2-V4-IT
x = CORblock_S(128, layer_name="V2", times=2).CORblock_forward(x)
x = CORblock_S(256, layer_name="V4", times=4).CORblock_forward(x)
x = CORblock_S(512, layer_name="IT", times=2).CORblock_forward(x)
    
x = GlobalAveragePooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1000)(x)
    
model = tf.keras.Model(inputs, x, name='cornet_keras')

opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9) #Add correct learning rate
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])

print(model.summary())
    
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=2, batch_size=10) #Paper- 43 epochs training
score = model.evaluate(testX, testY, verbose=0)

model.summary()
