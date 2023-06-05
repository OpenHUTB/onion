#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import necessary packages
from CORnet_S import CORnetS
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator


# In[6]:


#Parse arguments-- use when training on AffectNet/other dataset

#ap = argparse.ArgumentParser()
#ap.add_argument("-d", "--dataset", required = True, help = "path to input dataset")
#ap.add_argument("-m","--model", required = True, help = "path to output model")
#args = vars(ap.parse_args())


# In[7]:
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="data",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="test", target_size=(224,224))

'''print("Loading images..")
imagePaths = list(paths.list_images(args["dataset"]))

#Include foll. preproc steps if you're using Adrian's preprocessing scripts, else modify:

sp = SimplePreprocessor(32,32)
iap = ImageToArrayPreprocessor()

#Load dataset and scale raw pixels to between 0 and 1
sdl = SimpleDatasetLoader(preprocessors = [sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float")/255.0'''


# In[8]:


#Initialize optimizer, model
print("Compiling model")
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9) #Add correct learning rate
model = CORnetS()

#Compile model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt, metrics=['accuracy'])

#Train model
model.fit(trainX, trainY, validation_data=(testX, testY), epochs= 3, batch_size= 50) #Paper- 43 epochs training

#Save network (weights) to disk in HDF5 format
print("Serializing network")
model.save(args["model"])

#Evaluate how well the model is doing
score = model.evaluate(testX, testY, verbose=0)

#Create another prediction script which can load weights (if stored after training) & make foll predns on test data
print("Evaluating network")
#predictions=model.predict(testX,testY,batch_size=32)
predictions=model.predict(testX,testY)

model.summary()
#print(classification_report(testY.argmax(axis=1),predictions.argmax(axis=1),target_names=["", ""]))


# In[ ]:




