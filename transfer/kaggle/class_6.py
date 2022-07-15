# -*- coding: utf-8 -*-
"""
======================================================
Description:
	Modifies ResNet50 from 1000 output classes to 6 output classes 
	Dataset = (folding, grain_off, growth, loose_grain, pinhole, non-defective)

	[x] Modify Kaggle dataset for validation set:
	    -> (600 each to 480/120-test/valid = 2880 training & 720 validation
	[.] Adapt 10 class to 6 classes - "kaggle" leather dataset
	[.] Attempt newer model - RESNET-201 / DENSENET / EFFICIENTNET
	[.] Move beck to "transfer" folder when testing complete


STEPS:
	[1] Load and re-shape FULL datset  images (6 folders)
	[2] Image pre-processing - 6 classes, labels, 1-hot encode, shuffle & split
	[3] 
	[4] 
	[5] 
	[6] TEST MODEL STORAGE AND RETREIVAL?
	[7] 


SPYDER - Set working directory to:
/home/antz/Desktop/models/kaggle/
/home/antz/Desktop/models/dataset/kaggle/
======================================================
"""

#from keras.applications import VGG16
#from keras.applications import InceptionV3
#from keras.applications.imagenet_utils import decode_predictions
import numpy as np
import os
import time
from keras.applications.resnet import ResNet50
from keras.preprocessing import image
import pickle

from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten


#========================================================================#
## AIM:
## Customise Models to 6 Classes (6 types of defects)
## To remove the last (Dense) layer from the model (1000 classes)
## Replace using 6 classes in kaggle (600 images @ [227x227x3] each): 
## folding | grain_off | growth | loose_grain | pinhole | non-defective
#========================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD Training Data
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#PATH = os.getcwd()								         # gtcwd() = Get Current Working Directory
#data_path = PATH + '/training'				  	         # 

data_path='/home/antz/Desktop/models/dataset/kaggle/train'
data_dir_list=os.listdir(data_path)
print(data_dir_list)                                     # Output: "names = []" below

img_data_list=[]
labels=[]

## Fetch & Process each image from 6 sub-folders (classes) ##
for dataset in data_dir_list:
    ## Get each image from the directories ##
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
    ## Create the list with full image file paths ##
	for img in img_list:
		# Get and shape each image
		img_path = data_path + '/'+ dataset + '/'+ img 
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		img_data_list.append(x)							    # create list of data

img_data = np.array(img_data_list)						    # convert data to numpy array
print(img_data.shape)                                       # (2880, 1, 224, 224, 3)
img_data=np.rollaxis(img_data,1,0)                          # swap first 2 elements
print(img_data.shape)                                       # (1, 3601, 224, 224, 3)
img_data=img_data[0]                                        # Remove first element
print(img_data.shape)                                       # (2880, 224, 224, 3)

#------------------------------------------------------------------------------------------#
# Create 6 Class Labels :                                                                  #
# 'growth_marks', 'grain_off', 'loose_grains', 'folding_marks', 'non_defective', 'pinhole' #
#------------------------------------------------------------------------------------------#
labels=np.ones(2880,dtype='int64')                      # creates array of {size, shape} of 1's
#print(labels)

labels[0:600]    =0
labels[600:1200] =1
labels[1200:1800]=2
labels[1800:2400]=3
labels[2400:3000]=4
labels[3000:3600]=5

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## STORE RESHAPED DATA AS .npy FILE ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
np.save('train_data.npy', img_data)						# Store the data in numopy file
np.save('train_label.npy',labels)						# Store the labels in numopy file

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD - No need to re-do above data loading & conversion
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
img_data = np.load('train_data.npy')					# Load the data
labels = np.load('train_label.npy')						# Load the labels
print(img_data.shape)                                   # Check: (2880, 224, 224, 3)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Data Pre-Processing
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#----------------------------#
# Define the number of classes
#-----------------------------#
num_classes = 6  										# define the number of classes
num_of_samples = img_data.shape[0]						# Get the number of samples

names =['growth_marks', 'grain_off', 'loose_grains', 'folding_marks', 'non_defective', 'pinhole']

#----------------------------#
# One-Hot Encoding of labels #
#----------------------------#
Y = np_utils.to_categorical(labels, num_classes)		# One-hot encoding

#---------------------------#
# Shuffle data              #
#---------------------------#
x,y = shuffle(img_data,Y, random_state=2)

#----------------------------------#
# Split data - Train/Test-80/20%   #
#----------------------------------#
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)


#=================================================================#
# Create a Custom ResNet Model - 1000 Classes to 6 Classes        #
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

## Create Custom Model using new "last-layer" with 6 Classes
custom_resnet_model = Model(inputs=image_input,outputs= out)
custom_resnet_model.summary()							# View the new model (6 Classes)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Train the Cusom Model - FREEZE ALL LAYERS EXCEPT LAST LAYER
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Make all layers untrainable ()
for layer in custom_resnet_model.layers[:-1]:
	layer.trainable = False

# Except Last-Layer is trainable
custom_resnet_model.layers[-1].trainable

# Compile new Model
custom_resnet_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

t=time.time()

# Fitting & Validating
hist = custom_resnet_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))

# ~~~~ RESULTS: ~~~~
# loss: 		0.4726
# accuracy: 	0.8229
# val_loss: 	0.6219 
# val_accuracy: 0.7483


#=================================================================#
# SAVE THE TRAINED MODEL TO DISK - [UNTESTED]
#=================================================================#
# [1] Save pickle file
filename = 'finalized_model.sav'
pickle.dump(hist, open(filename, 'wb'))


# [2] Save pickle file
Pkl_Filename = "Pickle_RL_Model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(LR_Model, file)


# [3] Save the trained model as a pickle string.
saved_model = pickle.dumps(hist)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# LOAD THE TRAINED MODEL FROM DISK - [UNTESTED]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [1] Load the Pickle file
loaded_model = pickle.load(open(filename, 'rb'))
score = loaded_model.score(X_test, Y_test)			# calculate loaded model score
print("Test score: {0:.2f} %".format(100 * score))	# view score
print(result)									


# [2] Load the Pickle file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)
Pickled_LR_Model									# loaded model
Ypredict = Pickled_LR_Model.predict(Xtest) 			# make prediction with loaded model

# [3] Load the Pickle String
model_from_pickle = pickle.loads(saved_model)



#============================================================================================#
#============================================================================================#
# Testing & Evaluating the Custom Model - Fitting
#============================================================================================#
#============================================================================================#
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
## VALIDATION SET ----------> (480-TRAIN)/(120-VALID) 
# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
#data_path = PATH + '/validation'                   # if working dir in dataset folder
#data_dir_list = os.listdir(data_path)
data_path='/home/antz/Desktop/models/dataset/kaggle/validate'
data_dir_list=os.listdir(data_path)
print(data_dir_list)

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
		#print('Input image shape:', x.shape)
		img_data_list.append(x)

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
# Reshaping the input data: (not in video)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

## ~~ MODEL EVALUATION FUNCTION ~~ ##
(loss, accuracy) = custom_resnet_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

# View Results
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

## [INFO] loss=0.6219, accuracy: 74.8264%

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-------------------------------------------------------#
## Convert single image data to correct format          #
#-------------------------------------------------------#
img_path='/home/antz/Desktop/models/kaggle/test1.jpg'
img=image.load_img(img_path, target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

#-------------------------------------------------------#
## Modified Model Prediction on single test image       #
#-------------------------------------------------------#
## ~~~~=======~~~ WHAT IS BEING PREDICTED HERE?? ~~~~========~~~ ##
feature= custom_resnet_model.predict(x)
print(feature)											# 
prediction = np.argmax(feature)
print("Predicted Class " + names[prediction])           # 
## ~~~~=======~~~ WHAT IS BEING PREDICTED HERE?? ~~~~========~~~ ##
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# visualizing losses and accuracy
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import matplotlib.pyplot as plt
# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
xc=range(12)                                # No. of Epochs

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#