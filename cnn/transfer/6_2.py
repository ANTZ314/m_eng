#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Modifies InceptionV3 from 1000 output classes to 6 output classes
    Dataset: class_6: dim[224, 224, 3], Train(480), Test(120)

STEPS:
    [1] Load Pre-Trained Model, re-shape test input image & make un-modified prediction
    [2] Load datasets and re-shape all images to match formatting
    [3] Create 4 classes and 1-hot encode
    [4] Modify Inception Model - from 1000 to 4 output classes
    [5] RESULTS:
"""
##====================================================================##
## AIM:
## To remove the last (Dense) layer from the model (1000 classes)
## Replace using 6 classes in kaggle (600 images @ 227x227x3): 
## folding | grain_off | growth | loose_grain | pinhole | non-defective
##====================================================================##
import numpy as np
## Note: If model is not there it will auto-download
from keras.applications.inception_v3 import InceptionV3
## TO CREATE A CUSTOM MODEL ##
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.layers import Input                           
## Dataset manipulation ##
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
## Prediction ##
import matplotlib.pyplot as plt
import time
from tensorflow.keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import decode_predictions

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD DATA SHAPED DATA - All images                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_data = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_03c_data.npy')  # Load the data
labels  = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_03c_label.npy')  # Load the labels

print("[INFO] TRAINING SET SHAPE: {}".format(img_data.shape))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 6 CLASSES & DATA PREPARATION                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_classes = 6                                                 # define the number of classes
num_of_samples = img_data.shape[0]                              # Get the number of samples
labels=np.ones(num_of_samples,dtype='int64')                    # creates array of {size, shape} of 1's
#print(labels)

labels[0:599]    =0                                             # growth_marks
labels[599:1198] =1                                             # grain_off
labels[1198:1797]=2                                             # loose_grains
labels[1797:2396]=3                                             # folding_marks
labels[2396:2996]=4                                             # non_defective
labels[2996:3595]=5                                             # pinhole

names =['growth_marks', 'grain_off', 'loose_grains', 'folding_marks', 'non_defective', 'pinhole']

# One-Hot Encoding of labels #
Y=np_utils.to_categorical(labels,num_classes)
# Shuffle data              #
x,y = shuffle(img_data,Y,random_state=2)
# Split data - Train/Test   #
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print("[INFO] PRE-PROCESSING SUCCESSFULL --")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Download & Load Model: INCEPTION V3 - 1000 Classes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
model = InceptionV3()
# View Inception_V3's last-layer = avg_pool (None,2048)
#model.summary()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## MODIFY EXISTING INCEPTION_V3 MODEL - UNTESTED??
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
print("-- MODIFYING INCEPTION-V3 MODEL --")

## Custom Model - Define new input layer dimmensions
image_input = Input(shape=(224,224,3))      

## Custom Model - Define input tensors
model = InceptionV3(input_tensor=(image_input), include_top='True', weights='imagenet')

## Get output of the max pooling layer (last before resnet output layer)
last_layer = model.get_layer('avg_pool').output         # Has all ResNet layers before it
x = Flatten(name='flatten')(last_layer)                 # Added new flatten layer "x"

## Add new Dense layer with 4 Classes as Output
out = Dense(num_classes, activation=('softmax'), name='output_layer')(x)

## Custom Model - Create the Model
custom_incept_model = Model(inputs=image_input, outputs=out)
#custom_incept_model.summary()                           # Dimensions = 4 classes: Dense (None, 4)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Train the Cusom Model - FREEZE ALL LAYERS EXCEPT LAST LAYER
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#----------------------# From class_6_1.py #----------------------#
"""
# Make all layers untrainable ()
for layer in custom_incept_model.layers[:-1]:
    layer.trainable = False

# Except Last-Layer is trainable
custom_incept_model.layers[-1].trainable
"""
#----------------------# From class_6_1.py #----------------------#


## COMPILE THE MODEL - optimizer was 'rmsprop', changed to 'adam'
custom_incept_model.compile(loss='categorical_crossentropy',optimizer='adam' ,metrics=['accuracy'])

no_batch = 32
no_epoch = 12

t=time.time()
## FIT THE MODEL
hist = custom_incept_model.fit(X_train, y_train, batch_size=no_batch, epochs=no_epoch, verbose=1, validation_data=(X_test,y_test))
print('Training time: %s' % (t - time.time()))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## ~~ MODEL EVALUATION FUNCTION ~~ ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
(loss, accuracy) = custom_incept_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

# View Results
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

## [INFO] loss=0.6219, accuracy: 74.8264%


##======================================================================##
##                  PLOT THE OUTPUTS OF THE MODEL                       ##
##======================================================================##
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
plt.savefig('Loss_6_3.png')

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
plt.savefig('Accuracy_6_3.png')

print("-- TRAINING METRICS PLOTTED AND SAVED --")


##======================================================================##
##                  MAKE PREDICTION WITH MODIFIED MODEL                 ##
##======================================================================##
print("Loaded class list:")
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
print("\n")
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
feature= efficient_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: grain_off")
print("Predicted Class " + names[prediction])           # 
print("\n")
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
feature= efficient_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: growth_marks")
print("Predicted Class " + names[prediction])           # 
print("\n")
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
feature= efficient_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: loose_grains")
print("Predicted Class " + names[prediction])           # 
print("\n")
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
feature= efficient_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: pinhole")
print("Predicted Class: " + names[prediction])          # 
print("\n")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
