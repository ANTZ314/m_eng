# -*- coding: utf-8 -*-
"""
Description:
Modify 3x different Pre-Trained Models from keras (VGG16, ResNet50, InceptionV3)
Modifies ResNet50 from 1000 output classes to 4 output classes (cat, dog, horse, human)

STEPS:
[1] Load 3X Pre-Trained ModelS
[2] Re-shape loaded test input image & make prediction (with each)
[3] Load and re-shape FULL datset  images (four folders)
[4] Create 4 classes and 1-hot encode them
[5] Modify Inception (each?) Model - from 1000 to 4 output classes

SPYDER:
Set working directory to:
/home/antz/0_samples/M_Eng/dataset/transfer/4class/
Pre-Trained model test image "/test_img/cheetah.jpg", "/test_img/dandy.jpg"...
"""

#-------------------------------------------------------#
# MODEL1 - VGG16                                        #
#-------------------------------------------------------#
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image

import numpy as np
import os
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

# Set test image path!
os.chdir("/home/antz/0_samples/M_Eng/dataset/transfer/test_img/")

model=VGG16()
print(model.summary())

#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='cheetah.jpg'
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)                   # 'x' is now image data in 3 channels

#-----------------------------------------#
## VGG16 Prediction on single test image  #
#-----------------------------------------#
feature= model.predict(x)
print('Predicted class',decode_predictions(feature,5))

#-------------------------------------------------------#
# MODEL2 - ResNet50                                     #
#-------------------------------------------------------#
from keras.applications.resnet50 import ResNet50
model2 = ResNet50(include_top='True')
print(model2.summary())

#--------------------------------------------#
## ResNet50 Prediction on single test image  #
#--------------------------------------------#
feature = model2.predict(x)
print('Predicted class',decode_predictions(feature,5))

#-------------------------------------------------------#
# MODEL3 - InceptionV3                                  #
#-------------------------------------------------------#
from keras.applications.inception_v3 import InceptionV3
model3 = InceptionV3()

#----------------------------------------------#
## InceptionV3 Prediction on single test image #
#----------------------------------------------#
feature = model2.predict(x)
print('Predicted class',decode_predictions(feature,5))




##===============================================================##
## AIM:
## To remove the last (Dense) layer from the model (1000 classes)
## and replace using 4 classes: cat | dog | human | horse
## From Dataset of 4x class folders with 25 images of each class
##===============================================================##


#-------------------------------------------------------#
# DATA PREPROCESSING - All images                       #
#-------------------------------------------------------#
data_path='/home/antz/0_samples/M_Eng/dataset/transfer/4class/'
data_dir_list=os.listdir(data_path)
print(data_dir_list)

img_data_list=[]

for dataset in data_dir_list:
    img_list=os.listdir(data_path+"\\"+dataset)
    for img in img_list:
        img_path=data_path+"\\"+dataset+"\\"+img
        img=image.load_img(img_path,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        x=preprocess_input(x)
        img_data_list.append(x)

img_data=np.array(img_data_list)

print(img_data.shape)
img_data=np.rollaxis(img_data,1,0)
img_data=img_data[0]


#-------------------------------------------------------#
# Create 4 Classes : cat, dog, human, horse             #
#-------------------------------------------------------#
num_classes=4
num_of_samples=100
labels=np.ones(100,dtype='int64')
print(labels)

labels[0:25]=0
labels[25:50]=1
labels[50:75]=2
labels[75:100]=3
names =['cats','dogs','horses','human']

#----------------------------#
# One-Hot Encoding of labels #
#----------------------------#
from keras.utils import np_utils
Y=np_utils.to_categorical(labels,num_classes)
print(Y)

#---------------------------#
# Shuffle data              #
#---------------------------#
from sklearn.utils import shuffle
x,y = shuffle(img_data,Y,random_state=2)

#---------------------------#
# Split data - Train/Test   #
#---------------------------#
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)



#-------------------------------------------------------#
# MODEL2 - ResNet50 modified to 4 classes               #
#-------------------------------------------------------#

model2.summary()

from keras.models import Model
from keras.layers import Flatten, Dense
from keras.layers import Input

image_input=Input(shape=(224,224,3))

model2=ResNet50(input_tensor=image_input, include_top='True',weights='imagenet')
last_layer=model2.get_layer('avg_pool').output
x=Flatten(name='flattern')(last_layer)
out=Dense(num_classes, activation='softmax', name='output_layer')(x)


custom_resnet_model= Model(inputs=image_input, outputs=out)
custom_resnet_model.summary()
model2.summary()

# optimizer can also be 'Adam'
custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='rmsprop' ,metrics=['accuracy'])
hist=custom_resnet_model.fit(X_train,y_train, batch_size=20, epochs=40, verbose=1, validation_data=(X_test,y_test))


#-------------------------------------------------------#
# Visualise the Training Metrics                        #
#-------------------------------------------------------#
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

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_accuracy vs val_accuracy')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])


