#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
[1] Load Pre-Trained Model, re-shape test input image & make un-modified prediction
[2] Load datasets and re-shape all images to match formatting
[3] Create 4 classes and 1-hot encode
[4] Modify Inception Model - from 1000 to 4 output classes
[5] RESULTS:
    # loss:         2.6806
    # accuracy:     0.3250
    # val_loss:     1.5890
    # val_accuracy: 0.3500
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## INCEPTION V3 - 1000 Classes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Note: If model is not there it will auto-download
from keras.applications.inception_v3 import InceptionV3
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
import numpy as np
import os

os.chdir("/home/antz/0_samples/M_Eng/dataset/transfer/test_img/")  # goto test images file path


# Load Model
model = InceptionV3()

#-------------------------------------------------------#
# DATA PREPROCESSING - Single test image                #
#-------------------------------------------------------#
## LOAD IMAGE AS ARRAY
img_path = "dog2.jpg"             # Load test image
img = image.load_img(img_path, target_size=(299,299)) # match input layer of VGG16
x = image.img_to_array(img)             # convert to array
#print(x)                               # view image = array of 3 channels btwn 0 & 224

## CONVERT TO PROPER DIMENSION
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)
# print(x)                              # view pixel data in 3 channels

## MAKE PREDICTION ON TEST IMAGE
feature = model.predict(x)              # Get features test image
#print(feature)                         # view 1000 model features

#-------------------------------------------------------#
# INCEPTION_V3 PREDICTION                               #
#-------------------------------------------------------#
# pred_img = decode_predictions(feature)  # All possible Predictions of test image
pred_img = decode_predictions(feature, 2) # Top 2 Predictions of test image
print('predicted class', pred_img)        # print preditions



##===============================================================##
## AIM:
## To remove the last (Dense) layer from the model (1000 classes)
## and replace using 4 classes: cat | dog | human | horse
## From Dataset of 4x class folders with 25 images of each class
##===============================================================##

#-------------------------------------------------------#
# DATA PREPROCESSING - All images                       #
#-------------------------------------------------------#
## Main Dataset Path:
data_path = '/home/antz/0_samples/M_Eng/dataset/transfer/4class'# 4 datasets
data_dir_list = os.listdir(data_path)                           # list directories
print(data_dir_list)                                            # show 4 directories

img_data_list = []                                              # create empty image data list

## Fetch & Process each image from 4 sub-folders (classes) ##
for dataset in data_dir_list:
    ## Get each image from the directories
    img_list = os.listdir(data_path + "/" + dataset)    
    #print(img_list)                                            # check file path
    
    for img in img_list:
        ## Create the list with full image file paths ##
        img_path = data_path + "/" + dataset + "/" + img        # linux version
        # img_path = data_path+"\\"+dataset+"\\"+img            # Windows version
        img = image.load_img(img_path,target_size=(299,299))    # load & resize
        x = image.img_to_array(img)                             # convert to array
        x = np.expand_dims(x, axis=0)                           # expand dimensions
        x = preprocess_input(x)                                 # ??
        img_data_list.append(x)                                 # add each processed image

## Reshape data size to match Model shape ##
img_data = np.array(img_data_list)          # convert image data into numpy array
#print(img_data)                            # view data
#print(img_data.shape)                      # (100, 1, 299, 299, 3)

img_data = np.rollaxis(img_data,1,0)        # (1, 100, 299, 299, 3)
img_data = img_data[0]                      # (100, 299, 299, 3)


#-------------------------------------------------------#
# Create 4 Classes : cat, dog, human, horse             #
#-------------------------------------------------------#
num_classes = 4
num_samples = 100

## Create 4 labels for each class
labels = np.ones(100,dtype=('int64'))       # creates array of 100x '1s'
#print(labels)
labels[0:25] = 0                            # cat
labels[25:50] = 1                           # dog
labels[50:75] = 2                           # human
labels[75:100] = 3                          # horse
names = ['cats','dogs','human','horse']  # Create the names


#----------------------------------------------------------#
# Encode labels, shuffle dataset and Split into Train/Test #
#----------------------------------------------------------#
## Convert labels into 1-hot encoding
from keras.utils import np_utils

Y=np_utils.to_categorical(labels, num_classes)
#print(Y)                                   # Show encoded array

## Shuffle data to remove imported image order (class grouping)
from sklearn.utils import shuffle
x,y = shuffle(img_data, Y, random_state=2)  # data is no longer in the loaded sequence

## Split the data into training & testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=(2))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## MODIFY EXISTING INCEPTION_V3 MODEL - UNTESTED??
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# View Inception_V3's last-layer = avg_pool (None,2048)
model.summary()

from keras.models import Model
from keras.layers import Flatten, Dense
from keras.layers import Input                           # TO CREATE A CUSTOM MODEL

## Custom Model - Define new input layer dimmensions
image_input = Input(shape=(299,299,3))      

## Custom Model - Define input tensors
model = InceptionV3(input_tensor=(image_input), include_top='True', weights='imagenet')

## Get output of the max pooling layer (last before resnet output layer)
last_layer = model.get_layer('avg_pool').output         # Has all ResNet layers before it
x = Flatten(name='flatten')(last_layer)                 # Added new flatten layer "x"

## Add new Dense layer with 4 Classes as Output
out = Dense(num_classes, activation=('softmax'), name='output_layer')(x)

## Custom Model - Create the Model
custom_incept_model = Model(inputs=image_input, outputs=out)
custom_incept_model.summary()                           # Dimensions = 4 classes: Dense (None, 4)


## COMPILE THE MODEL
custom_incept_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

## FIT THE MODEL
hist = custom_incept_model.fit(X_train, y_train, batch_size=20, epochs=40, verbose=1, validation_data=(X_test,y_test))

#------------------------------------------#
# loss:         5.9637e-04      0.1363
# accuracy:     1.0000          0.9625
# val_loss:     3.2450          179.4446
# val_accuracy: 0.2500          0.3000
#------------------------------------------#

##======================================================================##
##                  PLOT THE OUTPUTS OF THE MODEL                       ##
##======================================================================##
import matplotlib.pyplot as plt

# visualizing losses and accuracy
train_loss=hist.history['loss']
train_acc=hist.history['accuracy']
val_loss=hist.history['val_loss']
val_acc=hist.history['val_accuracy']
xc=range(40)  #number of epochs

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
0
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


##======================================================================##
##                  MAKE PREDICTION WITH MODIFIED MODEL                 ##
##======================================================================##
# Set test image path!
os.chdir("/home/antz/0_samples/M_Eng/dataset/transfer/test_img/")

#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='horse.jpg'

## Open the image - Causes Error: "load_img"##
#from PIL import Image  # to view the image
#image = Image.open(img_path)
#image.show()

img=image.load_img(img_path, target_size=(299,299))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
feature= custom_incept_model.predict(x)                 # Predict
print(feature)
# Convert prediction into label
prediction = np.argmax(feature)
print("Predicted Class " + names[prediction])           
# Dog   - Correct!
# Horse - Incorrect
# Cat   - Correct!
 