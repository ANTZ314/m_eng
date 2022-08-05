
#=================================================================#
# Check Colab hardware specifications:
#=================================================================#
## Check hardware specs
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


#=================================================================#
# Show directory & list files within:
#=================================================================#
!pwd    # Check currect directory

!ls /content -l


#=================================================================#
# Install dependencies & view packages:
#=================================================================#
!pip install package-name

!pip freeze | grep package-name

#=================================================================#
# Import Project from GITHUB
#=================================================================#
!git clone https://github.com/xxx.git


#=================================================================#
# Import Project from G-Drive
#=================================================================#
from google.colab import drive
drive.mount('/content/gdrive')


#=================================================================#
# Copy Files from G-Drive to '/content/'
#=================================================================#
# Find directory and copy contents to current directory
!cp -r /content/gdrive/MyDrive/PATH/TO/PROJECT/* .

# Move folder contents to current directory
!mv /content/leather .

# Check success?
!ls /content -l


#=================================================================#
# Loop through All or 'n' files in a directory:
#=================================================================#
import os
import shutil

source = r'/path/source/'                 # files location
destination = r'/path/dest/'              # where to move to 
folder = os.listdir(source)               # returns a list with all the files in source

files = os.listdir(source)

# All files
for f in files:
    print(source + f)

print("\n")

# 'n' files
for f in range(4):
    print(source + files[f])

#=================================================================#
# Copy Test Image and View the image:
#=================================================================#
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# View image from dataset
img_path1 = '/content/001.png'

img = mpimg.imread(img_path1)
imgplot = plt.imshow(img)
plt.show()


#=================================================================#
# Make prediction & view:
#=================================================================#
feature = ViT_model.predict(x_test)                  # Get features test image
print(feature)                                       # view 1000 model features

prediction = np.argmax(feature)
print('predicted class', prediction)                    # print preditions
print("Predicted Class " + class_types[prediction])     # class_types = class labels list


#=================================================================#
# Prepare & Make Prediction 01:							[ERROR]
#=================================================================#
import cv2
import tensorflow as tf
import numpy as np

CATEGORIES = ["Tree", "Stump", "Ground"]
IMG_SIZE = 224


def prepare(filepath):
    IMG_SIZE = 150 # This value must be the same as the value in Part1
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Able to load a .model, .h3, .chibai and even .dog
model = tf.keras.models.load_model("models/test.model")

prediction = model.predict([prepare('image.jpg')])
print("Predictions:")
print(prediction)
print(np.argmax(prediction))


#=================================================================#
# Prepare & Make Prediction 02:							[ERROR]
#=================================================================#
from PIL import Image
import numpy as np
from skimage import transform

def load(filename):
   np_image = Image.open(filename)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (256, 256, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image

 image = load('my_file.jpg')
 model.predict(image)