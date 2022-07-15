# -*- coding: utf-8 -*-
"""
======================================================
Description:
    Modifies ResNet50 from 1000 output classes to 6 output classes
    Fine-Tune on Kaggle dataset
    Evaluation Metrics & Test inference (test image?)

    [.] ADAPT TO 6 CLASSES OF "kaggle" LEATHER DATASET
    [.] ATTEMPT NEWER MODEL - RESNET-201 / DENSENET / EFFICIENTNET
    [.] [Move back to "transfer" folder]

STEPS:
    [1] Load and re-shape FULL datset  images (6 folders)
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
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
import numpy as np
import os
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import decode_predictions

#=============================================================================#
## AIM:
## To remove the last (Dense) layer from the model (1000 classes)
## Replace using 6 classes in kaggle (600 images @ 227x227x3): 
## folding | grain_off | growth | loose_grain | pinhole | non-defective
#=============================================================================#

#-------------------------------------------------------#
# LOAD DATA & PRE-PROCESS - All images                  #
#-------------------------------------------------------#
data_path='/home/antz/Desktop/models/dataset/kaggle/'
data_dir_list=os.listdir(data_path)
print(data_dir_list)                                            # Matches "names = []" below

img_data_list=[]

## Fetch & Process each image from 6 sub-folders (classes) ##
for dataset in data_dir_list:
    ## Get each image from the directories
    img_list = os.listdir(data_path+"/"+dataset)
    for img in img_list:
        ## Create the list with full image file paths ##
        img_path = data_path + "/" + dataset + "/" + img        # linux version
        img = image.load_img(img_path,target_size=(224,224))    # load & resize
        x = image.img_to_array(img)                             # convert to array
        x = np.expand_dims(x, axis=0)                           # 
        x = preprocess_input(x)                                 # 
        img_data_list.append(x)                                 # add each processed image

img_data=np.array(img_data_list)

print(img_data.shape)                                           # 
img_data=np.rollaxis(img_data,1,0)                              # 
img_data=img_data[0]                                            # (3600, 1, 224, 224, 3)

#------------------------------------------------------------------------------------------#
# Create 6 Classes :                                                                       #
# 'growth_marks', 'grain_off', 'loose_grains', 'folding_marks', 'non_defective', 'pinhole' #
#------------------------------------------------------------------------------------------#
num_classes=6                                                   #
num_of_samples=3600                                             #
labels=np.ones(3600,dtype='int64')                              #
#print(labels)

labels[0:600]    =0
labels[600:1200] =1
labels[1200:1800]=2
labels[1800:2400]=3
labels[2400:3000]=4
labels[3000:3600]=5
names =['growth_marks', 'grain_off', 'loose_grains', 'folding_marks', 'non_defective', 'pinhole']

"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## STORE RESHAPED DATA AS .npy FILE ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
np.save('train_data.npy', img_data)                     # Store the data in numopy file
np.save('train_label.npy',labels)                       # Store the labels in numopy file

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD - No need to re-do above data loading & conversion
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_data = np.load('train_data.npy')                    # Load the data
labels = np.load('train_label.npy')                     # Load the labels
print(img_data.shape)                                   # Check: (2880, 224, 224, 3)
"""

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
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)



#=============================================================================#
# MODEL2 - ResNet50 modified to 6 classes                                     #
#=============================================================================#
model = ResNet50(include_top='True')

#--------------------------------------------#
## ResNet50 Prediction on single test image  #
#--------------------------------------------#
#feature = model2.predict(x)
#print('Predicted class',decode_predictions(feature,5))

model.summary()

from keras.models import Model
from keras.layers import Flatten, Dense
from keras.layers import Input

image_input=Input(shape=(224,224,3))

model=ResNet50(input_tensor=image_input, include_top='True',weights='imagenet')
last_layer=model.get_layer('avg_pool').output
x=Flatten(name='flattern')(last_layer)
out=Dense(num_classes, activation='softmax', name='output_layer')(x)


custom_resnet_model= Model(inputs=image_input, outputs=out)
custom_resnet_model.summary()

# optimizer can also be 'Adam'
custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='rmsprop' ,metrics=['accuracy'])

# Fit the model (Run the Model)
hist=custom_resnet_model.fit(X_train,y_train, batch_size=20, epochs=40, verbose=1, validation_data=(X_test,y_test))

##RESULTS:
# loss:         0.1343 
# accuracy:     0.9525 
# val_loss:     7.0823 
# val_accuracy: 0.4800

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



##======================================================================##
##                  MAKE PREDICTION WITH MODIFIED MODEL                 ##
##======================================================================##
# Set test image path!
os.chdir("/home/antz/0_samples/M_Eng/dataset/transfer/test_img/")

#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='sunflower.jpg'                                # TEST INPUT IMAGE
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
feature= custom_resnet_model.predict(x)                 # Predict
print(feature)
# Convert prediction into label
prediction = np.argmax(feature)
print("Predicted Class " + names[prediction])           # 


