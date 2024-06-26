# -*- coding: utf-8 -*-
"""vit_custom02.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ydmzqoUPDSqNX4EpwtLjmvIT5QwsVqbg

# Vision Transformer (ViT)


#### Based on ViT_02
* Title: Image classification with Vision Transformer
* Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
* [Image classification with Vision Transformer](https://keras.io/examples/vision/image_classification_with_vision_transformer/)
* [GITHUB](https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_with_vision_transformer.py)

[Requires - TENSORFLOW: TensorFlow 2.4 or higher]

## Description
The ViT model applies the Transformer architecture with self-attention to sequences of image patches, without using convolution layers.
This example requires TensorFlow 2.4 or higher, as well as [TensorFlow Addons](https://www.tensorflow.org/addons/overview)
"""

# Required Add-Ons:
!pip install tensorflow-addons

"""## SETUP"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

"""## Get and prepare Dataset

REPLACE CIFAR-10 with Modified Kaggle
"""

# Define training parameters
num_classes = 100
input_shape = (32, 32, 3)
#(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

from google.colab import drive
drive.mount('/content/gdrive')

# Check dataset avaiabe
!ls /content/gdrive/MyDrive/dataset/

# Copy Kaggle Leather Dataset to current directory
!cp /content/gdrive/MyDrive/dataset/new_kaggle.tar.xz .

# Create dataset destination
!mkdir /content/leather

# Extract to new direcotry
#!tar -xvf  'new_kaggle.tar.xz' -C '/content/leather/'
!tar -xf  'new_kaggle.tar.xz' -C '/content/leather/'

!ls /content/leather/new_kaggle/ -l
# Move extracted sub-folder up one (moved manually)
!mv /content/leather/new_kaggle/* /content/leather/
# Remove left-overs
!rm -r /content/leather/new_kaggle
!rm -r /content/leather/temp
# Check extracted
!ls /content/leather/ -l
# check number of files..?

"""## Convert to Train/Test Numpy Arrays
```
/content/leather/
|--folding_marks (29,400 images)
|--growth_marks (29,400  images)
|--loose_grains (29,400  images)
|--non_defective (29,400  images)
|--pinhole (29,400  images)
```
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import glob
folding_marks = glob.glob('/content/leather/folding_marks/*.*')
grain_off = glob.glob('/content/leather/grain_off/*.*')
growth_marks = glob.glob('/content/leather/growth_marks/*.*')
loose_grains = glob.glob('/content/leather/loose_grains/*.*')
non_defective = glob.glob('/content/leather/non_defective/*.*')
pinhole = glob.glob('/content/leather/pinhole/*.*')

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

#-------- ViT_03
# [176,400] = [141,120] / [35,280] = [0.8 / 0.2]
data, labels = load_data()
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
#-------- ViT_03
print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

"""## Configure the hyperparameters"""

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 72   # We'll resize input images to this size
patch_size = 6    # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier

"""## Use data augmentation"""

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization
data_augmentation.layers[0].adapt(x_train)

"""## Implement multilayer perceptron (MLP)"""

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

"""## Implement patch creation as a layer"""

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

"""## Let's display patches for a sample image """

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")

"""## Implement the patch encoding layer:
The `PatchEncoder` layer will linearly transform a patch by projecting it into a vector of size `projection_dim`. 
In addition, it adds a learnable position embedding to the projected vector.
"""

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

"""## Build the ViT model
The ViT model consists of multiple Transformer blocks,
which use the `layers.MultiHeadAttention` layer as a self-attention mechanism
applied to the sequence of patches. The Transformer blocks produce a
`[batch_size, num_patches, projection_dim]` tensor, which is processed via an
classifier head with softmax to produce the final class probabilities output.
Unlike the technique described in the [paper](https://arxiv.org/abs/2010.11929),
which prepends a learnable embedding to the sequence of encoded patches to serve
as the image representation, all the outputs of the final Transformer block are
reshaped with `layers.Flatten()` and used as the image
representation input to the classifier head.

**Note** that the `layers.GlobalAveragePooling1D` layer
could also be used instead to aggregate the outputs of the Transformer block,
especially when the number of patches and the projection dimensions are large.
"""

def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

"""## Compile, train, and evaluate the mode"""

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

vit_classifier = create_vit_classifier()
history = run_experiment(vit_classifier)

"""After 100 epochs, the ViT model achieves around 55% accuracy and 82% top-5 accuracy on the test data. These are not competitive results on the CIFAR-100 dataset, as a ResNet50V2 trained from scratch on the same data can achieve 67% accuracy.

Note that the state of the art results reported in the [paper](https://arxiv.org/abs/2010.11929) are achieved by pre-training the ViT model using the JFT-300M dataset, then fine-tuning it on the target dataset. 

To improve the model quality without pre-training, you can try to train the model for more epochs, use a larger number of
Transformer layers, resize the input images, change the patch size, or increase the projection dimensions. 

Besides, as mentioned in the paper, the quality of the model is affected not only by architecture choices, but also by parameters such as the learning rate schedule, optimizer, weight decay, etc.

In practice, it's recommended to fine-tune a ViT model that was pre-trained using a large, high-resolution dataset.

## Store the Trained Model
"""

import pickle

# [2] Save pickle file
Pkl_Filename = "ViT_2_Model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(history, file)

# Copy across to GDrive?
!ls -l

"""## Load the saved model"""

Xtest = "test.png"  # Insert test image here

# [2] Load the Pickle file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LR_Model = pickle.load(file)
Pickled_LR_Model									              # loaded model
Ypredict = Pickled_LR_Model.predict(Xtest) 			# make prediction with loaded model

"""## Plot train and validation curves"""

loss = history.history['loss']
v_loss = history.history['val_loss']

acc = history.history['accuracy'] 
v_acc = history.history['val_accuracy']

#top5_acc = history.history['top5 acc']
#val_top5_acc = history.history['val_top5 acc']
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
"""
plt.plot(epochs, top5_acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Top 5 Acc')
plt.plot(epochs, val_top5_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Top5 Acc') 
plt.xlabel('Epochs', fontsize=11)
plt.ylabel('Top5 Accuracy', fontsize=12)
plt.legend(fontsize=12)
"""
plt.tight_layout()
# plt.savefig('/content/gdrive/My Drive/Colab Notebooks/resnet/train_acc.png', dpi=250)
plt.show()

"""## Create Confusion Matrix + Heatmap"""

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Kaggle Class Labels
class_types = ['folding_marks', 'grain_off', 'growth_marks', 'loose_grains', 'non_defective', 'pinhole']

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

pred_class = vit_classifier.predict(x_test)

conf_matrix(pred_class)