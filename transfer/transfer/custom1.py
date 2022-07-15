#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From TL_CNN:

Description:
- Dog/Cat Dataset Used
- Image Data Pre-Processing
- Image-Array Formatting
- Create Custom Model (simple)
- Plotting Model Accuracy & Loss
- Classify (Predict) Test Image
"""

import glob
import numpy as np
import os
import shutil

##====================================================================##
##        DATA PREPARARTION - Train, Test Validation                        ##
##====================================================================##
np.random.seed(42)

# Get entire cat/dog dataset (25000 images of cats & dogs)
files = glob.glob('train/*')
## Set to clean leather samples training set ##

# Create 2 lists from file names if contains "cat" or "dog" in the name
cat_files = [fn for fn in files if 'cat' in fn]
dog_files = [fn for fn in files if 'dog' in fn]

# From both lists - take 1500 unique images and create Training Set
cat_train = np.random.choice(cat_files, size=1500, replace=False)
dog_train = np.random.choice(dog_files, size=1500, replace=False)
# Remove those 1500 files from the original set
cat_files = list(set(cat_files) - set(cat_train))
dog_files = list(set(dog_files) - set(dog_train))

# Do the same to create Validation Set of 500 images
cat_val = np.random.choice(cat_files, size=500, replace=False)
dog_val = np.random.choice(dog_files, size=500, replace=False)
# Remove validation images from original dataset
cat_files = list(set(cat_files) - set(cat_val))
dog_files = list(set(dog_files) - set(dog_val))

# Create Test Set from main dataset (Not using Kaggle Test Set)
cat_test = np.random.choice(cat_files, size=500, replace=False)
dog_test = np.random.choice(dog_files, size=500, replace=False)

# View all datasets details
print('Cat datasets:', cat_train.shape, cat_val.shape, cat_test.shape)
print('Dog datasets:', dog_train.shape, dog_val.shape, dog_test.shape)


##-------------------------------------------------------------------##
##                 DATA FILES - Create and store                     ##
##-------------------------------------------------------------------##
# create each one's directory
train_dir = 'training_data'
val_dir = 'validation_data'
test_dir = 'test_data'

# Create Train, Test, Validate files
train_files = np.concatenate([cat_train, dog_train])
validate_files = np.concatenate([cat_val, dog_val])
test_files = np.concatenate([cat_test, dog_test])

# Create directories to store data files
os.mkdir(train_dir) if not os.path.isdir(train_dir) else None
os.mkdir(val_dir) if not os.path.isdir(val_dir) else None
os.mkdir(test_dir) if not os.path.isdir(test_dir) else None
# Copy all files into the folders
for fn in train_files:
    shutil.copy(fn, train_dir)          # 4000 files

for fn in validate_files:
    shutil.copy(fn, val_dir)            # 1000 files
    
for fn in test_files:
    shutil.copy(fn, test_dir)           # 1000 files


##====================================================================##
##                      IMAGE FORMAT CONVERSION                       ##
##====================================================================##
##---------------------------------------------------------------------##
## Image to Numpy Array
##---------------------------------------------------------------------##
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

##-----------------------? ? ? ? ? ? ? ? ?-----------------------------##
# JupyterNote - sets the backend of matplotlib to the 'inline' backend
#%matplotlib inline     # For Jupyter Notebook plot
%matplotlib auto        # For Cammand line plot
##---------------------------------------------------------------------##

##---------------------------------------------------------------------##
# Re-Shape Image Arrays
##---------------------------------------------------------------------##
# Define Image Dimensions [150,150]
IMG_DIM = (150, 150)

# 4000 cat dog images (4000, 150, 150, 3)
train_files = glob.glob('training_data/*')          
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in train_files]
train_imgs = np.array(train_imgs)
#train_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in train_files]     # Error - Windows
train_labels = [fn.split('/')[1].split('.')[0].strip() for fn in train_files]

validation_files = glob.glob('validation_data/*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in validation_files]
validation_imgs = np.array(validation_imgs)
#validation_labels = [fn.split('\\')[1].split('.')[0].strip() for fn in validation_files]     # Error - Windows
validation_labels = [fn.split('/')[1].split('.')[0].strip() for fn in validation_files]

print('Train dataset shape:', train_imgs.shape, 
      '\tValidation dataset shape:', validation_imgs.shape)


##---------------------------------------------------------------------##
## Scaling value 0 to 1 for images
##---------------------------------------------------------------------##
train_imgs_scaled = train_imgs.astype('float32')
validation_imgs_scaled  = validation_imgs.astype('float32')
train_imgs_scaled /= 255
validation_imgs_scaled /= 255

# View original image                     (cat = 0-2000)(dog = 2001-4000)
print(train_imgs[0].shape)
array_to_img(train_imgs[0])
# View scaled image
print(train_imgs_scaled[0].shape)
array_to_img(train_imgs_scaled[0]) 


##---------------------------------------------------------------------##
## Setting up parameter from training model
##---------------------------------------------------------------------##
batch_size = 30
num_classes = 2
epochs = 30
input_shape = (150, 150, 3)

##---------------------------------------------------------------------##
# 1-Hot Encode text category labels
##---------------------------------------------------------------------##
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(train_labels)
train_labels_enc = le.transform(train_labels)
validation_labels_enc = le.transform(validation_labels)

print(train_labels[1495:1505], train_labels_enc[1495:1505])


##====================================================================##
##                 CREATING CUSTOM CNN MODEL                          ##
##====================================================================##
from tensorflow import keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import optimizers


model = Sequential()

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Tensorflow Error -> "Optimzer.RMSprop()"
# model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(), metrics=['accuracy'])

## Fixed above TensorFlow Error:
optimizer1=keras.optimizers.RMSprop(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer1, metrics=['accuracy'])

model.summary()

##---------------------------------------------------------------------##
## Fitting the Custom Model
##---------------------------------------------------------------------##
history = model.fit(x=train_imgs_scaled, y=train_labels_enc,
                    validation_data=(validation_imgs_scaled, validation_labels_enc),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

## ~ RESULTS ~
## Epoch 30/30 - loss: 0.0357 - accuracy: 0.9940 - val_loss: 7.5669 - val_accuracy: 0.5250



##======================================================================##
##                  PLOT THE OUTPUTS OF THE MODEL                       ##
##======================================================================##
## NOTE: Must all be run at once - not line by line

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

t = f.suptitle('Basic CNN Performance', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

print(history.history.keys())
epoch_list = list(range(1,31))
#ax1.plot(epoch_list, history.history['acc'], label='Train Accuracy')           # KeyError: 'acc'
#ax1.plot(epoch_list, history.history['val_acc'], label='Validation Accuracy')  # KeyError: 'val_acc'
ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')           # solve
ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')  # solve
ax1.set_xticks(np.arange(0, 31, 5))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
ax2.set_xticks(np.arange(0, 31, 5))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")

##---------------------------------------------------------------------##