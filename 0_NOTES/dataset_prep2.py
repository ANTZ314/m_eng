"""
Description:
	Fetch dataset made up of TRAIN & VALIDATION sets from Kaggle
	Convert to to format & shape utilised in ViT applications

Method 1:
	-> [1] Combine Test & Validation sets into single class subfolders (remove validation)
	-> [2] Convert each class directory into [x_train, x_test, y_train, y_test] as required
Method 2:
	-> [1] Fetch entire training dataset & convert to correct shape (without train labels set)
	-> [2] Fetch entire validation dataset & convert to correct shape (without test labels set)
"""


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Break larger images into smaller tiles
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
from PIL import Image
from itertools import product
import os

def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')

        img.crop(box).save(out)

#--------------------------------
dir1 = "/content/leather/pinhole"

# For each file in directory "test" to "tiles":
d = 32
dir_in  = dir1                          # first directory
dir_out = "/content/tiled"              # hopefully not all images are put in same directory

# List files in that directory
for filename in os.listdir(dir_in):
    f = os.path.join(dir_in, filename)
    # checking if it is a file
    if os.path.isfile(f):
        #print(f)                       # view each file created
        tile(f, dir_in, dir_out, d)     # split into blocks
        os.remove(f)                    # delete original


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [1] Fetch entire training dataset & convert to correct shape (without train labels set)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Load G_Drive & Fetch Kaggle dataset 
from google.colab import drive
drive.mount('/content/gdrive')

# Copy Kaggle Leather Dataset to current directory
!cp -r /content/gdrive/MyDrive/2021_M_Eng/models/dataset/kaggle .

!ls /content/kaggle/train/ -l

#========================================================================================#
# 										METHOD 1
#========================================================================================#
"""
/leather/
|--folding_marks (600 images)
|--growth_marks (600 images)
|--loose_grains ((600 images)
|--non_defective (600 images)
|--pinhole (600 images)
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [3] Combine Test & Validation sets into single class subfolders (remove validation)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
!cp -r /content/kaggle/train/* /content/leather/

!ls /content/leather -l

!cp -r /content/kaggle/validate/folding_marks/* /content/leather/folding_marks/
!cp -r /content/kaggle/validate/grain_off/* /content/leather/grain_off/
!cp -r /content/kaggle/validate/growth_marks/* /content/leather/growth_marks/
!cp -r /content/kaggle/validate/loose_grains/* /content/leather/loose_grains/
!cp -r /content/kaggle/validate/non_defective/* /content/leather/non_defective/
!cp -r /content/kaggle/validate/pinhole/* /content/leather/pinhole/

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [4] Convert each class directory into [x_train, x_test, y_train, y_test] as required
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Kaggle Class Labels
class_types = ['folding_marks', 'grain_off', 'growth_marks', 'loose_grains', 'non_defective', 'pinhole']

import glob
folding_marks = glob.glob('/content/leather/folding_marks/*.*')
grain_off = glob.glob('/content/leather/grain_off/*.*')
growth_marks = glob.glob('/content/leather/growth_marks/*.*')
loose_grains = glob.glob('/content/leather/loose_grains/*.*')
non_defective = glob.glob('/content/leather/non_defective/*.*')
pinhole = glob.glob('/content/leather/pinhole/*.*')

data = []
labels = []

for i in folding_marks:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (224,224))
    image=np.array(image)
    data.append(image)
    labels.append(0)
for i in grain_off:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (224,224))
    image=np.array(image)
    data.append(image)
    labels.append(1)
for i in growth_marks:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (224,224))
    image=np.array(image)
    data.append(image)
    labels.append(2)
for i in loose_grains:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (224,224))
    image=np.array(image)
    data.append(image)
    labels.append(3)
for i in non_defective:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (224,224))
    image=np.array(image)
    data.append(image)
    labels.append(3)
for i in pinhole:   
    image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
    target_size= (224,224))
    image=np.array(image)
    data.append(image)
    labels.append(3)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2,
                                                random_state=42)

print(f"x_train shape: {X_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {X_test.shape} - y_test shape: {y_test.shape}")

"""
x_train shape: (2880, 224, 224, 3) - y_train shape: (2880,)
x_test  shape: (720, 224, 224, 3)  - y_test shape:  (720,)
"""
#========================================================================================#
# 										METHOD 2
#========================================================================================#
import os
from keras.preprocessing import image
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

#-------------------------------------------------------#
# DATA PREPROCESSING - All images                       #
#-------------------------------------------------------#
## Main Dataset Path:
data_path = '/content/kaggle/train'                             # 6 folders with 2880 images
data_dir_list = os.listdir(data_path)                           # list directories
print(data_dir_list)                                            # show 6 directories (480 images each)

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
        img = image.load_img(img_path,target_size=(224,224))    # load & resize
        x = image.img_to_array(img)                             # convert to array
        x = np.expand_dims(x, axis=0)                           # expand dimensions
        x = preprocess_input(x)                                 # ??
        img_data_list.append(x)                                 # add each processed image


## Reshape data size to match Model shape ##
x_train = np.array(img_data_list)                               # convert image data into numpy array
#print(img_data)                                                # view data
print(x_train.shape)                                            # (2880, 1, 224, 224, 3)
x_train = np.rollaxis(x_train,1,0)                              # (1, 2880, 224, 224, 3)
x_train = x_train[0]                                            # (2880, 224, 224, 3)
print(x_train.shape)                                            # (2880, 224, 224, 3)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# [2] Fetch entire validation dataset & convert to correct shape (without test labels set)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#-------------------------------------------------------#
# DATA PREPROCESSING - All images                       #
#-------------------------------------------------------#
## Main Dataset Path:
data_path = '/content/kaggle/validate'                          # 6 folders with 720 images
data_dir_list = os.listdir(data_path)                           # list directories
print(data_dir_list)                                            # show 6 directories (120 images each)

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
        img = image.load_img(img_path,target_size=(224,224))    # load & resize
        x = image.img_to_array(img)                             # convert to array
        x = np.expand_dims(x, axis=0)                           # expand dimensions
        x = preprocess_input(x)                                 # ??
        img_data_list.append(x)                                 # add each processed image


## Reshape data size to match Model shape ##
x_test = np.array(img_data_list)                              # convert image data into numpy array
#print(img_data)                                              # view data
#print(x_test.shape)                                          # (720, 1, 224, 224, 3)
x_test = np.rollaxis(x_test,1,0)                              # (1, 720, 224, 224, 3)
x_test = x_test[0]                                            # (720, 224, 224, 3)
print(x_test.shape)                                           # (720, 224, 224, 3)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

