# -*- coding: utf-8 -*-
"""
# ViT Model with Custom Leather Dataset

**NOTES**
* ViT Code taken from **ViT_3** & **ViT_4a**
* Kaggle dataset - 6 folders of 600 images each @ [224, 224, 3]
* Kaggle new     - 6 folders of 29,400 imgs each @ [32, 32, 3]

**STEPS:**
* Connect to G_Drive & Copy compressed dataset file (~174Mb)
* Extract the file (.tar / .zip) to new directory
* Begin ViT processing...
"""

"""
## Using dataset: kaggle_0X
```
/content/leather/
|--folding_marks (600 -> 29,400 images)
|--growth_marks (600 -> 29,400  images)
|--loose_grains ((600 -> 29,400  images)
|--non_defective (600 -> 29,400  images)
|--pinhole (600 -> 29,400  images)
```
"""

"""
### Load the dataset images into a numpy array format
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Kaggle Class Labels
class_types = ['folding_marks', 'grain_off', 'growth_marks', 'loose_grains', 'non_defective', 'pinhole']

import glob
folding_marks = glob.glob('/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/kaggle_02/folding_marks/*.*')
grain_off = glob.glob('/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/kaggle_02/grain_off/*.*')
growth_marks = glob.glob('/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/kaggle_02/growth_marks/*.*')
loose_grains = glob.glob('/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/kaggle_02/loose_grains/*.*')
non_defective = glob.glob('/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/kaggle_02/non_defective/*.*')
pinhole = glob.glob('/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/kaggle_02/pinhole/*.*')

def load_data():

  data = []
  labels = []

  for i in folding_marks:   
      image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
      target_size= (32,32))
      image=np.array(image)
      data.append(image)
      labels.append(0)
  for i in grain_off:   
      image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
      target_size= (32,32))
      image=np.array(image)
      data.append(image)
      labels.append(1)
  for i in growth_marks:   
      image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
      target_size= (32,32))
      image=np.array(image)
      data.append(image)
      labels.append(2)
  for i in loose_grains:   
      image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
      target_size= (32,32))
      image=np.array(image)
      data.append(image)
      labels.append(3)
  for i in non_defective:   
      image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
      target_size= (32,32))
      image=np.array(image)
      data.append(image)
      labels.append(4)
  for i in pinhole:   
      image=tf.keras.preprocessing.image.load_img(i, color_mode='rgb', 
      target_size= (32,32))
      image=np.array(image)
      data.append(image)
      labels.append(5)

  data = np.array(data)
  labels = np.array(labels)

  return data, labels

data, labels = load_data()

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""## ViT Model Begin:

Attempt **ViT_3**

Also utilise pretrained model...
"""

print(class_types)
print('check shapes: ', x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# train_im, test_im = X_train/255.0 , x_test/255.0
train_lab_categorical = tf.keras.utils.to_categorical(y_train, num_classes=6, dtype='uint8')
test_lab_categorical = tf.keras.utils.to_categorical(y_test, num_classes=6, dtype='uint8')


train_im, valid_im, train_lab, valid_lab = train_test_split(x_train, train_lab_categorical, test_size=0.20, 
                                                            stratify=train_lab_categorical, 
                                                            random_state=42, shuffle = True) # stratify is unncessary 

print("train data shape after the split: ", train_im.shape)
print('new validation data shape: ', valid_im.shape)
print("validation labels shape: ", valid_lab.shape)

#print('train im and label types: ', type(train_im), type(train_lab))

training_data = tf.data.Dataset.from_tensor_slices((train_im, train_lab))
validation_data = tf.data.Dataset.from_tensor_slices((valid_im, valid_lab))
test_data = tf.data.Dataset.from_tensor_slices((x_test, test_lab_categorical))

#print("test data shape: ", x_test.shape)
#print('check types; ', type(training_data), type(validation_data))

autotune = tf.data.AUTOTUNE 

# Set buffer sizes to match shape outputs:
train_data_batches = training_data.shuffle(buffer_size=2304).batch(128).prefetch(buffer_size=autotune)
valid_data_batches = validation_data.shuffle(buffer_size=576).batch(32).prefetch(buffer_size=autotune)
test_data_batches = test_data.shuffle(buffer_size=720).batch(32).prefetch(buffer_size=autotune)

"""## Patch Generation

Divides images into patches of given patch size
"""

import tensorflow as tf
from tensorflow.keras import layers

#==========================#
### generate patches 
#==========================#
class generate_patch(layers.Layer):
  def __init__(self, patch_size):
    super(generate_patch, self).__init__()
    self.patch_size = patch_size
    
  def call(self, images):
    batch_size = tf.shape(images)[0]
    patches = tf.image.extract_patches(images=images, 
                                       sizes=[1, self.patch_size, self.patch_size, 1], 
                                       strides=[1, self.patch_size, self.patch_size, 1], rates=[1, 1, 1, 1], padding="VALID")
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, -1, patch_dims]) #here shape is (batch_size, num_patches, patch_h*patch_w*c) 
    return patches

"""## Helper function to visualize the patches (SKIP)"""

import matplotlib.pyplot as plt
from itertools import islice #, count

train_iter_7im, train_iter_7label = next(islice(training_data, 7, None)) # access the 7th element from the iterator


train_iter_7im = tf.expand_dims(train_iter_7im, 0)
train_iter_7label = train_iter_7label.numpy() 

print('check shapes: ', train_iter_7im.shape) 

patch_size=4 

######################
# num patches (W * H) /P^2 where W, H are from original image, P is patch dim. 
# Original image (H * W * C), patch N * P*P *C, N num patches
######################
generate_patch_layer = generate_patch(patch_size=patch_size)
patches = generate_patch_layer(train_iter_7im)

print ('patch per image and patches shape: ', patches.shape[1], '\n', patches.shape)


def render_image_and_patches(image, patches):
    plt.figure(figsize=(6, 6))
    plt.imshow(tf.cast(image[0], tf.uint8))
    plt.xlabel(class_types [np.argmax(train_iter_7label)], fontsize=13)
    n = int(np.sqrt(patches.shape[1]))
    plt.figure(figsize=(6, 6))
    #plt.suptitle(f"Image Patches", size=13)
    for i, patch in enumerate(patches[0]):
        ax = plt.subplot(n, n, i+1)
        patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
        ax.imshow(patch_img.numpy().astype("uint8"))
        ax.axis('off')    


render_image_and_patches(train_iter_7im, patches)

"""## Positonal Encoding Layer"""

class PatchEncode_Embed(layers.Layer):
  """
  2 steps happen here:
  	1. flatten the patches
  	2. Map to dim D; patch embeddings
  """

  def __init__(self, num_patches, projection_dim):
    super(PatchEncode_Embed, self).__init__()
    self.num_patches = num_patches
    self.projection = layers.Dense(units=projection_dim)
    self.position_embedding = layers.Embedding(
    input_dim=num_patches, output_dim=projection_dim)


  def call(self, patch):
    positions = tf.range(start=0, limit=self.num_patches, delta=1)
    encoded = self.projection(patch) +               self.position_embedding(positions)
    return encoded

"""## Patch Generation & Positional Encoding:

This part:
* takes images as inputs,  
* Conv layer filter matches query dim of multi-head attention layer
* Add embeddings by randomly initializing the weights
"""

def generate_patch_conv_orgPaper_f(patch_size, hidden_size, inputs):
  patches = layers.Conv2D(filters=hidden_size, kernel_size=patch_size, strides=patch_size, padding='valid')(inputs)
  row_axis, col_axis = (1, 2) # channels last images
  seq_len = (inputs.shape[row_axis] // patch_size) * (inputs.shape[col_axis] // patch_size)
  x = tf.reshape(patches, [-1, seq_len, hidden_size])
  return x

"""## Positonal Encoding Layer"""

class AddPositionEmbs(layers.Layer):
  """inputs are image patches 
  Custom layer to add positional embeddings to the inputs."""

  def __init__(self, posemb_init=None, **kwargs):
    super().__init__(**kwargs)
    self.posemb_init = posemb_init
    #posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input') # used in original code

  def build(self, inputs_shape):
    pos_emb_shape = (1, inputs_shape[1], inputs_shape[2])
    self.pos_embedding = self.add_weight('pos_embedding', pos_emb_shape, initializer=self.posemb_init)

  def call(self, inputs, inputs_positions=None):
    # inputs.shape is (batch_size, seq_len, emb_dim).
    pos_embedding = tf.cast(self.pos_embedding, inputs.dtype)

    return inputs + pos_embedding

pos_embed_layer = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02))

"""## Transformer Encoder Block:

part of ViT Implementation:
This block implements the Transformer Encoder Block

Contains 3 parts--
1. LayerNorm 
2. Multi-Layer Perceptron 
3. Multi-Head Attention

For repeating the Transformer Encoder Block we use Encoder_f function. 
"""

def mlp_block_f(mlp_dim, inputs):
  x = layers.Dense(units=mlp_dim, activation=tf.nn.gelu)(inputs)
  x = layers.Dropout(rate=0.1)(x) # dropout rate is from original paper,
  x = layers.Dense(units=inputs.shape[-1], activation=tf.nn.gelu)(x) # check GELU paper
  x = layers.Dropout(rate=0.1)(x)
  return x

def Encoder1Dblock_f(num_heads, mlp_dim, inputs):
  x = layers.LayerNormalization(dtype=inputs.dtype)(inputs)
  x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1], dropout=0.1)(x, x) 
  # self attention multi-head, dropout_rate is from original implementation
  x = layers.Add()([x, inputs]) # 1st residual part 
  
  y = layers.LayerNormalization(dtype=x.dtype)(x)
  y = mlp_block_f(mlp_dim, y)
  y_1 = layers.Add()([y, x]) #2nd residual part 
  return y_1

def Encoder_f(num_layers, mlp_dim, num_heads, inputs):
  x = AddPositionEmbs(posemb_init=tf.keras.initializers.RandomNormal(stddev=0.02), name='posembed_input')(inputs)
  x = layers.Dropout(rate=0.2)(x)
  for _ in range(num_layers):
    x = Encoder1Dblock_f(num_heads, mlp_dim, x)

  encoded = layers.LayerNormalization(name='encoder_norm')(x)
  return encoded

"""# VISION TRANSFORMER MAIN:

Building blocks of ViT:

Check other gists or the complete notebook []

* Patches (generate_patch_conv_orgPaper_f) + embeddings (within Encoder_f)
* Transformer Encoder Block (Encoder_f)
* Final Classification

### Hyperparameters
"""

transformer_layers = 6
patch_size = 4
hidden_size = 64
num_heads = 4
mlp_dim = 128

rescale_layer = tf.keras.Sequential([layers.experimental.preprocessing.Rescaling(1./255)])

"""## Build the ViT Model:"""

def build_ViT():
  inputs = layers.Input(shape=train_im.shape[1:])
  # rescaling (normalizing pixel val between 0 and 1)
  rescale = rescale_layer(inputs)
  # generate patches with conv layer
  patches = generate_patch_conv_orgPaper_f(patch_size, hidden_size, rescale)

  #===================================#
  # ready for the transformer blocks
  #===================================#
  encoder_out = Encoder_f(transformer_layers, mlp_dim, num_heads, patches)  

  #===================================#
  #  final part (mlp to classification)
  #===================================#
  #encoder_out_rank = int(tf.experimental.numpy.ndim(encoder_out))
  im_representation = tf.reduce_mean(encoder_out, axis=1)  # (1,) or (1,2)
  # similar to the GAP, this is from original Google GitHub

  logits = layers.Dense(units=len(class_types), name='head', kernel_initializer=tf.keras.initializers.zeros)(im_representation) # !!! important !!! activation is linear 

  final_model = tf.keras.Model(inputs = inputs, outputs = logits)
  return final_model

ViT_model = build_ViT()
#ViT_model.summary()

"""# Using ViT_4a's Model Analysis:"""

#------------------------------------------
## Required Add-Ons:
#!pip install tensorflow-addons
#------------------------------------------

"""## Model - Compile, Reduce & FIT

1h33 to Train
Test accuracy: 91.77%
Test top 5 accuracy: 100.0%
"""

#from tensorflow import keras
#from tensorflow.keras import layers
import tensorflow_addons as tfa

# Define parameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "/tmp/checkpoint"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history

# Test accuracy: 91.98%
# Test top 5 accuracy: 100.0%
history = run_experiment(ViT_model)     # with GUP ~2Hrs

"""
=============================================================
## Store Trained Model - SKIP
=============================================================

import pickle

# [2] Save "Training History Data" as a pickle file
Pkl_Filename = "vit_c03_hist.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(history, file)

=============================================================
## Load Model
Requires TensorFlow-AddOns
=============================================================

from tensorflow import keras
#from tensorflow.keras import layers
import tensorflow_addons as tfa
import pickle

Pkl_Filename = "vit_c03_hist.pkl"  

# [2] Load the "Training History Data" Pickle file
with open(Pkl_Filename, 'rb') as file:  
    vit_c03_history = pickle.load(file)
vit_c03_history									# loaded model
"""


"""
=============================================================
## Plot Training Data
Replace all "history" with loaded pickle model "vit_c03_history"
================================================================
"""
# list all data in history
print(history.history.keys())

# V04 PLOTTING:
loss = history.history['loss']
v_loss = history.history['val_loss']

acc = history.history['accuracy'] 
v_acc = history.history['val_accuracy']

top5_acc = history.history['top-5-accuracy']
val_top5_acc = history.history['val_top-5-accuracy']
epochs = range(len(loss))

fig = plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.yscale('log')
plt.plot(epochs, loss, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Loss')
plt.plot(epochs, v_loss, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Loss')
# plt.ylim(0.3, 100)
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(1, 3, 2)
plt.plot(epochs, acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Acc')
plt.plot(epochs, v_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.subplot(1, 3, 3)
plt.plot(epochs, top5_acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Top 5 Acc')
plt.plot(epochs, val_top5_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Top5 Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Top5 Accuracy', fontsize=12)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("/home/antz/Downloads/MEng/ViT_Results/acc_loss.png", dpi=250)
plt.show()


"""----------------------------------------------------
## Load Keras Stored Model
----------------------------------------------------
from tensorflow.keras.models import load_model
 
# load model
Load_ViT_model = load_model('model.h5')
"""

"""=================================================
## Create Confusion Matrix + Heatmap
================================================="""

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def conf_matrix(predictions): 
    ''' Plots conf. matrix and classification report '''
    cm=confusion_matrix(y_test, np.argmax(np.round(predictions), axis=1))
    print("Classification Report:\n")
    cr=classification_report(y_test,
                                np.argmax(np.round(predictions), axis=1), 
                                target_names=[class_types[i] for i in range(len(class_types))])
    print(cr)
    plt.figure(figsize=(12,12))
    sns_hmp = sns.heatmap(cm, annot=True, xticklabels = [class_types[i] for i in range(len(class_types))], 
                yticklabels = [class_types[i] for i in range(len(class_types))], fmt="d")
    fig = sns_hmp.get_figure()
    fig.savefig('/home/antz/Downloads/MEng/ViT_Results/matrix.png', dpi=250)
    plt.show()

# From Built Model
pred_class_cust03 = ViT_model.predict(x_test)
"""
# From Loadded Trained Model
pred_class_cust03 = Load_ViT_model.predict(x_test)
pred_class_cust03 = ViT_Cust03_Model.predict(x_test)
"""

conf_matrix(pred_class_cust03)

"""----------------------------------------------------
## View Test Image
----------------------------------------------------"""

# [SKIP] View test image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Drag'n'Drop Test Image  -  Dim[224, 224, 3]
new_test = '/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/test_im/grain_off_224.jpg'

img = mpimg.imread(new_test)
imgplot = plt.imshow(img)
#plt.savefig("/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/output.png", dpi=250)
plt.show()


"""# CONTINUE HERE...

Need to recompile +/-2hours
## Break large image to patches
## -OR- Reshape image??

(1x) input_img[224, 224, 3] --> (x) output_img[32, 32, 3] 
"""

from PIL import Image
from itertools import product
import os

#------------------------------------------
## Test image to correct position
#!mkdir /content/test
#!cp /content/03.jpg /content/test
#!ls /content/test/
#------------------------------------------

img = "/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/test_im/patch/"
# For each file in directory "dirX" "test" to "tiles":
d = 32          # patch size
dir_out = img   # output directory
dir_in  = img   # input directory
f = "grain_off_224.jpg"    # Filename [227, 227, 3]

# Split original image + Remane & store new patch images
def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{name}_{i}_{j}{ext}')

        img.crop(box).save(out)

# Convert test image to [32, 32, 3] patches
tile(f, dir_in, dir_out, d)     # split into blocks

# Check Patches created (???)
_, _, files = next(os.walk(dir_in))
file_count = len(files)
print("File Count: {}".format(file_count))

#------------------------------------------
## Remove larger image
#!rm /home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/test_im/patch/grain_off_224.jpg
#------------------------------------------

# Convert all patches into numpy array
X_data = []

files = glob.glob ("/home/antz/Documents/0_models/ViT/ViT_B/vit_custom03/test_im/patch/*.jpg")
for img_file in files:
    image = Image.open(img_file).convert('RGB')
    image = np.array(image)
    if image is None or image.shape != (d, d, 3):
        print(f'This image is bad: {img_file} {image.shape if image is not None else "None"}')
    else:
        X_data.append(image)

print('X_data shape:', np.array(X_data).shape)

##==================================================================================
## ValueError: Layer "model_1" expects 1 input(s), but it received 49 input tensors.
##==================================================================================

# Generate predictions for samples
prediction = ViT_model.predict(X_data)    # Error: 49 tensors - Expected 1??
print(prediction)                         # in Softmax format

# Generate arg maxes for predictions
classes = np.argmax(prediction, axis = 1)
print(classes)
