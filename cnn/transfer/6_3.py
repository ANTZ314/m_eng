# -*- coding: utf-8 -*-
"""
======================================================
Description:
    Modifies VGG-16 model from 1000 output classes to 6 output classes
    Dataset: Kaggle_03 - dim[224, 224, 3], Dataset(3600), split(0.2)

STEPS:
    [1] Load and re-shape FULL datset  images (6 folders)
    [2] Image pre-processing - 6 classes, labels, 1-hot encode, shuffle & split
    [3] Load and train VGG-16 pre-trained model
    [4] Fine-tune VGG-16 model on New dataset
    [5] Visualise Training data 
    [6] EVALUATE with test images or Validation set ? ? ?

SPYDER - Set working directory to:
/home/antz/Desktop/models/dataset/kaggle/
======================================================
"""
## LOAD DEPENDENCIES & ENTIRE DATASET (ALL 6 FOLDERS) ##
from keras.preprocessing import image
import numpy as np
import time
from keras.applications.imagenet_utils import preprocess_input
## DATA PRE-PROCESSING ##
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
## IMPORT AND MODIFIY DL MODEL ##
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.layers import Input
## EVALUATION & VISUALISATION ##
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD DATA SHAPED DATA - All images                            #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_data = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_03d_data.npy')  # Load the data
labels  = np.load('/home/antz/Downloads/MEng/CNN/pickles/kaggle_03d_label.npy')  # Load the labels

print("[INFO] TRAINING SET SHAPE: {}".format(img_data.shape))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# 6 CLASSES & DATA PREPARATION                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
num_classes = 6                                                 # define the number of classes
num_of_samples = img_data.shape[0]                              # Get the number of samples

names = ['folding_marks', 'growth_marks', 'pinhole', 'grain_off', 'non_defective', 'loose_grains']

# One-Hot Encoding of labels #
Y=np_utils.to_categorical(labels,num_classes)
# Shuffle data              #
x,y = shuffle(img_data,Y,random_state=2)
# Split data - Train/Test   #
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print("[INFO] PRE-PROCESSING SUCCESSFULL --")


#===============================================================================#
# MODEL2 - VGG-16 modified to 6 classes                                         #
#===============================================================================#
no_batch = 32
no_epoch = 12

# View VGG16's last-layer dimensions/parameters = (None,2048)
#model.summary()                                                                # Output(1000)

## Custom Model - Define new input layer dimmensions
image_input = Input(shape=(224,224,3))      

## Custom Model - Define input tensors
model = VGG16(input_tensor=(image_input), include_top='True', weights='imagenet')

## Get output of the max pooling layer (last before VGG-16 output layer)
last_layer = model.get_layer('fc2').output                                      # Has all VGG16 layers before it
x = Flatten(name='flatten2')(last_layer)                                        # Added new flatten layer "x"

## Add new Dense layer with 4 Classes as Output
out = Dense(num_classes, activation=('softmax'), name='output_layer')(x)

## Custom Model - Create the Model
custom_vgg16_model = Model(inputs=image_input, outputs=out)
#custom_vgg16_model.summary()                                                   # Ddimensions = 4 classes (None, 4)


## COMPILE THE MODEL ## - optimizer 'rmsprop' / 'adam'
custom_vgg16_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

t=time.time()

## FIT THE MODEL ##
hist = custom_vgg16_model.fit(X_train, y_train, batch_size=no_batch, epochs=no_epoch, verbose=1, validation_data=(X_test,y_test))
print('Training time: %s' % (t - time.time()))

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Different method of modifying VGG-16
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
from keras import models
from keras import layers

new_model = models.Sequential()
new_model.add(model)
new_model.add(Input(shape=(224,224,3)))
new_model.add(layers.Flatten())
new_model.add(layers.Dense(256, activation="relu"))
## If output binary(2) use sigmoid ##
new_model.add(layers.Dense(num_classes, activation="softmax"))
new_model.summary()

## COMPILE THE MODEL ## - optimizer 'rmsprop' / 'adam'
new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
## FIT THE MODEL ##
hist = new_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test,y_test))
"""

print("[INFO] MODEL CREATED, COMPILED & TRAINED SUCCESSFULLY")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## ~~ MODEL EVALUATION FUNCTION ~~ ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
(loss, accuracy) = custom_vgg16_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

# View Results
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

## [INFO] loss=0.6219, accuracy: 74.8264%


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Visualise the Training Metrics                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# visualizing losses and accuracy
train_loss=hist.history['loss']
train_acc=hist.history['accuracy']
val_loss=hist.history['val_loss']
val_acc=hist.history['val_accuracy']
xc=range(no_epoch)                                              # number of epochs

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available                                      # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('Loss_6_4.png')

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_accuracy vs val_accuracy')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available                                      # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.savefig('Accuracy_6_4.png')

print("[INFO] TRAINING METRICS PLOTTED")


##======================================================================##
##                  MAKE PREDICTION WITH MODIFIED MODEL                 ##
##======================================================================##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Test Image: 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
feature= custom_vgg16_model.predict(x)
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
feature= custom_vgg16_model.predict(x)
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
feature= custom_vgg16_model.predict(x)
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
feature= custom_vgg16_model.predict(x)
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
feature= custom_vgg16_model.predict(x)
print(feature)                                          # 
prediction = np.argmax(feature)
print("[INFO] Input Image: pinhole")
print("Predicted Class " + names[prediction])           # 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
