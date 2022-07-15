# -*- coding: utf-8 -*-
"""
======================================================
Was: MonkeyClassifier.py

Description:
- Modified Pre-trained Model from 1000 outputs to 10 classes 
- Dataset = 10 types of monkeys in 10 seperate folders

Note:
Run in SPYDER
Working directory set to dataset directory in Spyder
Greatly editted according to Tutorial Video order - Check run?
Set Dataset Path:	../../Datasets/??
======================================================
"""

import numpy as np
import os
import time
from keras.applications import ResNet50
#from keras.applications import VGG16
#from keras.applications import InceptionV3
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten

from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Load Original Models (1000 Classes from Imagenet)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
model  = ResNet50()
#model = VGG16()
#model = InceptionV3()
print(model.summary())

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Image Preprocessing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_path = 'n312.jpg'
img = image.load_img(img_path, target_size=(224, 224))

## Reshape image dimensions to match model input ##
x = image.img_to_array(img)
print (x.shape)
x = np.expand_dims(x, axis=0)
print (x.shape)
x = preprocess_input(x)
print('Input image shape:', x.shape)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Make immediate prediction from PreTrained Models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
feature = model.predict(x)						# Make prediction with ResNet50's 1000 classes
dec_pred = decode_predictions(feature,5)		# Get top 5 predictions

print('Predicted class',dec_pred)
#plt.imshow(img)								# view the image


#=================================================================#
# Customise Models to 10 Classes (10 types of monkeys)
#=================================================================#
# Loading the training data
PATH = os.getcwd()								# gtcwd() = Get Current Working Directory
# Define data path
data_path = PATH + '/training'					# 
data_dir_list = os.listdir(data_path)			# list of 10 class sub-folders

img_data_list=[]
labels=[]
#names=[]

# Loop through each sub-folder
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))

	# Loop through each image in the sub-folders
	for img in img_list:
		# Get and shape each image
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		print('Input image shape:', x.shape)
		img_data_list.append(x)							# create list of data
		labels.append(int(dataset[1]))					# creating list of image labels
		#names.append(data)

img_data = np.array(img_data_list)						# convert data to numpy array

## STORE ##
np.save('train_data.npy', img_data)						# Store the data in numopy file
np.save('train_label.npy',labels)						# Store the labels in numopy file

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD - No need to re-do above data loading & conversion
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_data = np.load('train_data.npy')					# Load the data
labels = np.load('train_label.npy')						# Load the labels

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Not mentioned in this video???
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Define the number of classes
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_classes = 10  										# define the number of classes
num_of_samples = img_data.shape[0]						# Get the number of samples

names = ['n01','n02','n03','n04','n05','n06','n07','n08','n09']
# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)		# One-hot encoding

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset (80% X_Train + 20% X_Test)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


#=================================================================#
# Create a Custom ResNet Model
#=================================================================#
#from keras.layers import Input

#Training the classifier alone
image_input = Input(shape=(224, 224, 3))

# 'top=True'= Include the last layer
model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
model.summary()											# View original ResNet model (1000 Classes)

## Extract last pooling layer (before 1000 class Dense layer)
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)

## Create a new last-layer
out = Dense(num_classes, activation='softmax', name='output_layer')(x)

## Create Custom Model using new "last-layer" with 10 Classes
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()							# View the new model (10 Classes)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Train the Cusom Model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Make all layers untrainable
for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

# Except Last-Layer is trainable
custom_resnet_model.layers[-1].trainable

# Compile new Model
custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#=================================================================#
# Testing & Validating the Custom Model - Fitting
#=================================================================#
t=time.time()

# Fitting & Validating
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Evaluating "Custom Model"
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
data_path = PATH + '/validation'
data_dir_list = os.listdir(data_path)

img_data_list=[]
labels=[]
for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		print('Input image shape:', x.shape)
		img_data_list.append(x)
		labels.append(int(dataset[1]))

img_data = np.array(img_data_list)
# STORE
np.save('validation.npy', img_data)
np.save('validation_label.npy',labels)
# LOAD
img_data = np.load('validation.npy')
labels = np.load('validation_label.npy')

# Changes from Video?
Y=np_utils.to_categorical(labels, num_classes)	


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Not mentioned in this video???
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
"""
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)
"""

(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

# View Results
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

feature= custom_resnet_model.predict(x)
prediction = np.argmax(feature)
print("Predicted Class " + names[prediction])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# visualizing losses and accuracy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import matplotlib.pyplot as plt

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(12)

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

#=============================================================================#
#=============================================================================#
# Creating a Basic Mode1 from Scratch
#=============================================================================#
#=============================================================================#
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten,MaxPooling2D
#create model

model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', 
                 input_shape=(224,224,3)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))

model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(10, activation='relu'))
model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('softmax'))
model.compile(keras.optimizers.Adam(lr=1e-5), 'categorical_crossentropy', metrics=['accuracy'])

## Fitting the Model ##
hist = model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Fine tune the resnet 50
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#image_input = Input(shape=(224, 224, 3))
model = ResNet50(weights='imagenet',include_top=False)
model.summary()
last_layer = model.output
# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(last_layer)
# add fully-connected & dropout layers
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)
# a softmax layer for 4 classes
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

# this is the model we will train
custom_resnet_model2 = Model(inputs=model.input, outputs=out)

custom_resnet_model2.summary()

for layer in custom_resnet_model2.layers[:-6]:
	layer.trainable = False

custom_resnet_model2.layers[-1].trainable

custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()
hist = custom_resnet_model2.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Define the number of classes - Line 115??
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_classes = 10
num_of_samples = img_data.shape[0]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# convert class labels to on-hot encoding
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
names = ['n01','n02','n03','n04','n05','n06','n07','n08','n09']

Y_val = np_utils.to_categorical(labels, num_classes)
x_val=img_data
(loss, accuracy) = model.evaluate(x_val, Y_val, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
