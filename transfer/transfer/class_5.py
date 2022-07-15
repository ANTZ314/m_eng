# -*- coding: utf-8 -*-
"""
Description:
Modifies ResNet50 from 1000 output classes to 5 output classes 
Dataset = (daisy, dandelion, rose, sunflower, tulip)

STEPS:
[1] 
[2] Load and re-shape FULL datset  images (four folders)
[3] Create 5 classes and 1-hot encode them
[4] Shuffle the data and Split into Train-Test (80/20)
[5] Import ResNet Model & Modify from 1000 to 5 output classes

SPYDER - Set working directory to:
/home/antz/0_samples/M_Eng/dataset/transfer/5class/
"""

#=======================================================#
# MODEL1 - VGG16                                        #
#=======================================================#
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
import numpy as np
import os
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.imagenet_utils import decode_predictions

##===============================================================##
## AIM:
## To remove the last (Dense) layer from the model (1000 classes) and
## replace using 5 classes: daisy | dandelion | roses | sunflower | tulip
## From Dataset of 5x class folders with 100 images of each class
##===============================================================##

#-------------------------------------------------------#
# DATA PREPROCESSING - All images                       #
#-------------------------------------------------------#
data_path='/home/antz/0_samples/M_Eng/dataset/transfer/5flower/'
data_dir_list=os.listdir(data_path)
print(data_dir_list)

img_data_list=[]

## Fetch & Process each image from 5 sub-folders (classes) ##
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
img_data=img_data[0]                                            # (500, 1, 224, 224, 3)


#----------------------------------------------------------------------#
# Create 5 Classes : 'daisy','dandelion','roses','sunflowers','tulips' #
#----------------------------------------------------------------------#
num_classes=5                                           #
num_of_samples=500                                      #
labels=np.ones(500,dtype='int64')                       #
#print(labels)

labels[0:100]  =0
labels[100:200]=1
labels[200:300]=2
labels[300:400]=3
names =['daisy','dandelion','roses','sunflower','tulips']

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



#-------------------------------------------------------#
# MODEL2 - ResNet50 modified to 4 classes               #
#-------------------------------------------------------#
#=======================================================#
# MODEL2 - ResNet50                                     #
#=======================================================#
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
#custom_resnet_model.summary() #duplicate?

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
img_path='sunflower.jpg'
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


