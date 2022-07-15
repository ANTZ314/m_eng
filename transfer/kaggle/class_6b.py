# -*- coding: utf-8 -*-
"""
======================================================
Description:
	Load the already prepared train_data & train_label files
    Create custom model & fine-tune ResNet-50 model
    Make prediction on "test_files"

STEPS [UPDATE]:
	[1] Load and re-shape FULL datset  images (6 folders)
	[2] Image pre-processing - 6 classes, labels, 1-hot encode, shuffle & split
	[3] Create Basic model from scratch (CUSTOM MODEL)
	[4] Fine-tune RESNET-50 model on New dataset		<---- CHANGE TO CUSTOM MODEL CREATED!!
	[5] EVALUATE???
======================================================
"""
import os
import numpy as np
import time
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

##~REMOVED~##
#from keras.applications import VGG16
#from keras.applications import InceptionV3
#from keras.applications.imagenet_utils import decode_predictions
#from keras.layers import Input

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD - No need to re-do above data loading & conversion
# Data pre-processed and stored to '.npy' files
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_data = np.load('train_data.npy')					          # Load the data
labels = np.load('train_label.npy')						          # Load the labels
print(img_data.shape)                                             # Check: (2880, 224, 224, 3)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Data Pre-Processing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Define the number of classes
num_classes = 6  										          # define the number of classes
num_of_samples = img_data.shape[0]						          # Get the number of samples

## Unused??
#names =['growth_marks', 'grain_off', 'loose_grains', 'folding_marks', 'non_defective', 'pinhole']

# One-Hot Encoding of labels #
Y = np_utils.to_categorical(labels, num_classes)		# One-hot encoding
# Shuffle data #
x,y = shuffle(img_data,Y, random_state=2)
# Split data - Train/Test-80/20% #
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


#=============================================================================#
# Creating a Basic Mode1_1 from Scratch             <---- Unnecessary here?????
#=============================================================================#
"""
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten

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
"""

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Fine tune the ResNet-50 Model_2				(?? NOT ABOVE CUSTOM MODEL ??)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#image_input = Input(shape=(224, 224, 3))
model = ResNet50(weights='imagenet',include_top=False)
model.summary()
last_layer = model.output

## Add a global spatial average pooling layer ##
x = GlobalAveragePooling2D()(last_layer)

## Add fully-connected & dropout layers ##
x = Dense(512, activation='relu',name='fc-1')(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu',name='fc-2')(x)
x = Dropout(0.5)(x)

## Output = softmax layer for 6 classes ##
out = Dense(num_classes, activation='softmax',name='output_layer')(x)

## this is the model we will train ##
custom_resnet_model2 = Model(inputs=model.input, outputs=out)
custom_resnet_model2.summary()

## Only the last 6 layers are Trainable??
for layer in custom_resnet_model2.layers[:-6]:
	layer.trainable = False

## last layer trainable ? ? ?
custom_resnet_model2.layers[-1].trainable

## Compile the fine-tuned model
custom_resnet_model2.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

## Fitting the Model ##
t = time.time()
hist = custom_resnet_model2.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Save and/or Load the trained model as a pickle string:
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import pickle

# Save model as Pickle
pickle.dump(hist, open('model.pkl', 'wb'))

# If Stored - Load the pickle file back to model:
pickled_model = pickle.load(open('model.pkl', 'rb'))
pickled_model.predict(X_test)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Evaluate the Models ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Evaluate Custom_resnet_model2
(loss, accuracy) = custom_resnet_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

# convert class labels to on-hot encoding
names = ['growth_marks', 'grain_off', 'loose_grains', 'folding_marks', 'non_defective', 'pinhole']

## Evaluate "model" - removed above
Y_val = np_utils.to_categorical(labels, num_classes)
x_val = img_data

(loss, accuracy) = model.evaluate(x_val, Y_val, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


#=============================================================================#
# Add Metric visualisation??
#=============================================================================#



#=============================================================================#
##                  MAKE PREDICTION WITH MODIFIED MODEL                      ##
#=============================================================================#
# Set test image path!
os.chdir("/home/antz/0_samples/M_Eng/dataset/transfer/test_img/")
os.chdir("/home/antz/Desktop/models/kaggle/")

#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='test.jpg'                                     # TEST INPUT IMAGE (GRAIN_OFF)
img = image.load_img(img_path, target_size=(224,224))
x   = image.img_to_array(img)
x   = np.expand_dims(x,axis=0)
x   = preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
feature = custom_resnet_model2.predict(x)               # Predict
print(feature)
# Convert prediction into label
prediction = np.argmax(feature)
print("Predicted Class " + names[prediction])           # 

## PREDICTION RESULTS - CORRECT:
## [3.6756894e-01 6.1149842e-01 1.7556423e-02 7.7217189e-04 2.5417248e-03 6.2296822e-05]
## Predicted Class grain_off
