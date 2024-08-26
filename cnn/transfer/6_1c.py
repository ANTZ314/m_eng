#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
[1] Load Pre-Trained Model, re-shape test input image & make un-modified prediction
[2] Load datasets and re-shape all images to match formatting
[3] Create 4 classes and 1-hot encode
[4] Modify Inception Model - from 1000 to 4 output classes
[5] Visualise training metrics
[6] Import single image and make prediction (always CAT)?
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## RESNET MODEL - 1000 Classes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import decode_predictions
import time

from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.layers import Input                           # TO CREATE A CUSTOM MODEL


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD DATA SHAPED DATA - All images                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_data = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_03d_data.npy')  # Load the data
labels  = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_03d_label.npy')  # Load the labels

print("[INFO] TRAINING SET SHAPE: {}".format(img_data.shape))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 6 CLASSES & DATA PREPARATION                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_classes = 6  												# define the number of classes
num_of_samples = img_data.shape[0]								# Get the number of samples

names = ['folding_marks', 'growth_marks', 'pinhole', 'grain_off', 'non_defective', 'loose_grains']

# One-Hot Encoding of labels #
Y=np_utils.to_categorical(labels,num_classes)
# Shuffle data              #
x,y = shuffle(img_data,Y,random_state=2)
# Split data - Train/Test   #
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print("[INFO] PRE-PROCESSING SUCCESSFULL --")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## MODIFY EXISTING RESNET-50 MODEL
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("-- MODIFYING RESNET-50 MODEL --")

# View ResNet's last-layer dimensions/parameters = (None,2048)
#model.summary()

## Custom Model - Define new input layer dimmensions
image_input = Input(shape = (224,224,3))

## Custom Model - Define input tensors
model = ResNet50(input_tensor = (image_input), include_top = 'True', weights = 'imagenet')

## Get output of the max pooling layer (last before resnet output layer)
last_layer = model.get_layer('avg_pool').output         # Has all ResNet layers before it
x = Flatten(name='flatten')(last_layer)                 # Added new flatten layer "x"

## Add new Dense layer with 4 Classes as Output
out = Dense(num_classes, activation=('softmax'), name='output_layer')(x)

## Custom Model - Create the Model
custom_resnet_model = Model(inputs=image_input, outputs=out)
#custom_resnet_model.summary()                           # Dimensions = 4 classes (None, 4)

## COMPILE THE MODEL
## Optimizer='adam'
custom_resnet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

no_batch = 32
no_epoch = 12

t=time.time()

## FIT THE MODEL
hist = custom_resnet_model.fit(X_train, y_train,
	batch_size=no_batch,
	epochs=no_epoch,
	verbose=1,
	validation_data=(X_test,y_test))

# METRICS:
# loss:         xx
# accuracy:     xx
# val_loss:     xx
# val_accuracy: xx

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## ~~ MODEL EVALUATION FUNCTION ~~ ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

# View Results
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

## [INFO] loss=0.6219, accuracy: 74.8264%

##======================================================================##
##                  PLOT THE OUTPUTS OF THE MODEL                       ##
##======================================================================##
import matplotlib.pyplot as plt

# visualizing losses and accuracy
train_loss=hist.history['loss']
train_acc=hist.history['accuracy']
val_loss=hist.history['val_loss']
val_acc=hist.history['val_accuracy']
xc=range(no_epoch)  #number of epochs

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('Loss_6_1c.png')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('Accuracy_6_1c.png')

print("[INFO] TRAINING DATA PLOTTED.")

##======================================================================##
##                  MAKE PREDICTION WITH MODIFIED MODEL                 ##
##======================================================================##
print("Order of loaded class list:")
print(names)
print("\n")

#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='/home/antz/Downloads/MEng/CNN/kaggle_07/validate/folding_marks447.jpg'
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
feature= custom_resnet_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: folding_marks")
print("Predicted Class " + names[prediction])           # 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Test Image: 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='/home/antz/Downloads/MEng/CNN/kaggle_07/validate/grain_off1.jpg'
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
feature= custom_resnet_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: grain_off")
print("Predicted Class " + names[prediction])           # 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Test Image: 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='/home/antz/Downloads/MEng/CNN/kaggle_07/validate/growth_marks193.jpg'
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
feature= custom_resnet_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: growth_marks")
print("Predicted Class " + names[prediction])           # 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Test Image: 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='/home/antz/Downloads/MEng/CNN/kaggle_07/validate/loose_grains63.jpg'
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
feature= custom_resnet_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: loose_grains")
print("Predicted Class " + names[prediction])           # 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Test Image: 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='/home/antz/Downloads/MEng/CNN/kaggle_07/validate/pinhole398.jpg'
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
feature= custom_resnet_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: pinhole")
print("Predicted Class: " + names[prediction])           # 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

