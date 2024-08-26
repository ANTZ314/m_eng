# -*- coding: utf-8 -*-
"""
======================================================
Description:
    Same as "class_6_1b.py" but with modified Model structure
    Modifies ResNet50 from 1000 output classes to 6 output classes
    Dataset: kaggle_03: dim[224, 224, 3], Train(3600)
STEPS:
    [1] Load the re-shaped datset from ".npy" files (6 folders)
    [2] Image pre-processing - 6 classes, labels, 1-hot encode, shuffle & split
    [3] Load and train ResNet-50 pre-trained model
    [4] Fine-tune RESNET-50 model on New dataset
    [5] Visualise Training data 
    [6] EVALUATE with test images or Validation set ? ? ?

SPYDER - Set working directory to:
/home/antz/Desktop/models/dataset/kaggle/
======================================================
"""

#=============================================================================#
# LOAD DEPENDENCIES & ENTIRE DATASET (ALL 6 FOLDERS)                          #
#=============================================================================#
## Import and Shape Dataset ##
import time
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet import ResNet50
from keras.applications.imagenet_utils import preprocess_input
## Dataset PreProcess/Re-Shape ##
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
## Modify ResNet-50 Model ##
from keras.models import Model
#from keras.layers import Flatten
#from keras.layers import Input
from keras.layers import GlobalAveragePooling2D, Dropout, Dense

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

names = ['folding_marks', 'growth_marks', 'pinhole', 'grain_off', 'non_defective', 'loose_grains']

# One-Hot Encoding of labels #
Y=np_utils.to_categorical(labels,num_classes)
# Shuffle data              #
x,y = shuffle(img_data,Y,random_state=2)
# Split data - Train/Test   #
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

print("[INFO] PRE-PROCESSING SUCCESSFULL --")


#=============================================================================#
# MODEL2 - ResNet50 modified to 6 classes                                     #
#=============================================================================#
print("-- MODIFYING RESNET-50 MODEL --")

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Fine tune the ResNet-50 Model_2
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
custom_resnet_model = Model(inputs=model.input, outputs=out)
custom_resnet_model.summary()

## Only the last 6 layers are Trainable??
for layer in custom_resnet_model.layers[:-6]:
    layer.trainable = False

## last layer trainable ? ? ?
custom_resnet_model.layers[-1].trainable

## Compile the fine-tuned model
custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

no_batch = 32
no_epoch = 12

## Fitting the Model ##
t = time.time()
hist = custom_resnet_model.fit(X_train, y_train,
    batch_size=no_batch,
    epochs=no_epoch,
    verbose=1,
    validation_data=(X_test, y_test))

print('Training time: %s' % (t - time.time()))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## ~~ MODEL EVALUATION FUNCTION ~~ ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

# View Results
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Visualise the Training Metrics                                  #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
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
plt.savefig('Loss_6_1d.png')

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
plt.savefig('Accuracy_6_1d.png')

print("-- TRAINING DATA PLOTTED & STORED --")


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
print("Predicted Class " + names[prediction])           # 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
