#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description:
    Unofficial COLAB Walkthrough of Vision Transformer:
    https://colab.research.google.com/github/hirotomusiker/schwert_colab_data_storage/blob/master/notebook/Vision_Transformer_Tutorial.ipynb

    Removed all test and visual confirmations - Bare functional code
    [UNTESTED LOCALLY] Must get label data (!wget links below)
    Test image must be resized to [224 x 224]
    
Inference Pipeline:
	[1] Split Image into Patches:
		The input image is split into 14 x 14 vectors with dimension of 768 by Conv2d (k=16x16) with stride=(16, 16).
	[2] Add Position Embeddings:
		Learnable position embedding vectors are added to the patch embedding vectors and fed to the transformer encoder.
	[3] Transformer Encoder:
		The embedding vectors are encoded by the transformer encoder. The dimension of input and output vectors are the same.
	[4] MLP (Classification) Head:
		The 0th output from the encoder is fed to the MLP head for classification to output the final classification results.

Requires timm (pytorch image models):
!pip install timm           # colab?
"""

## Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import PIL

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from timm import create_model

## ADDED ##
from keras.layers import Conv2d
## REMOVED ##
#import os
#import torchvision


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Prepare Model and Data -> CPU
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
model_name = "vit_base_patch16_224"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device = ", device)

# create a ViT model: 
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
model = create_model(model_name, pretrained=True).to(device)

# Define transforms for test
IMG_SIZE = (224, 224)
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)
transforms = [
              T.Resize(IMG_SIZE),
              T.ToTensor(),
              T.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
              ]

transforms = T.Compose(transforms)

#%%capture				                          # <--- Colab function: Remove?
# Get ImageNet Labels:
#!wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt
imagenet_labels = dict(enumerate(open('ilsvrc2012_wordnet_lemmas.txt')))

# Put test image into the working folder
img = PIL.Image.open('test.png')
img_tensor = transforms(img).unsqueeze(0).to(device)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Simple Inference
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# end-to-end inference
output = model(img_tensor)

"""
print("Inference Result:")
print(imagenet_labels[int(torch.argmax(output))])       # Test immediate inference
plt.imshow(img)                                         # View the test image
"""

#================================================================#
## VISION TRANSFORMER PIPELINE: [z_pic7a.png]
#================================================================#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## [1] Split Image into Patches
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# The input image is split into N patches (N = 14 x 14 for ViT-Base)
# and converted to D=768 embedding vectors by learnable 2D convolution

#Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))          # not run in Colab?

patches = model.patch_embed(img_tensor)                         # patch embedding convolution
print("Image tensor: ", img_tensor.shape)                       # Image tensor:     torch.Size([1, 3, 224, 224])
print("Patch embeddings: ", patches.shape)                      # Patch embeddings: torch.Size([1, 196, 768])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## [2] Add Position Embeddings
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#Visualization of position embeddings
pos_embed = model.pos_embed
print(pos_embed.shape)                                          # torch.Size([1, 197, 768])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## Make Transformer Input
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# A learnable class token is prepended to the patch embedding vectors as the 0th vector.
# 197 (1 + 14 x 14) learnable position embedding vectors are added to the patch embedding vectors.

transformer_input = torch.cat((model.cls_token, patches), dim=1) + pos_embed
print("Transformer input: ", transformer_input.shape)           # torch.Size([1, 197, 768])


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## [3] Transformer Encoder: [z_pic7b.png]
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
""" 
NOTE: Transformer Encoder Implementation Reference links in main file (tensoflow & PyTorch)

Transformer Encoder:
    [1] N (=197) embedded vectors are fed to the L (=12) series encoders.
    [2] The vectors are divided into query, key and value after expanded by an fc layer.
    [3] q, k and v are further divided into H (=12) and fed to the parallel attention heads.
    [4] Outputs from attention heads are concatenated to form the vectors whose shape is the same as the encoder input.
    [5] The vectors go through an fc, a layer norm and an MLP block that has two fc layers.
"""

# Series Transformer Encoders
print("Input tensor to Transformer (z0): ", transformer_input.shape)
x = transformer_input.clone()
for i, blk in enumerate(model.blocks):
    print("Entering the Transformer Encoder {}".format(i))
    x = blk(x)
x = model.norm(x)
transformer_output = x[:, 0]
print("Output vector from Transformer (z12-0):", transformer_output.shape)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## How Attention Works
## See what the actual attention looks like.
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
attention = model.blocks[0].attn                                # torch.Size([1, 197, 768])

# fc layer to expand the dimension
transformer_input_expanded = attention.qkv(transformer_input)[0] # torch.Size([197, 2304])

# Split qkv into mulitple q, k, and v vectors for multi-head attantion
qkv = transformer_input_expanded.reshape(197, 3, 12, 64)        # (N=197, (qkv), H=12, D/H=64)
print("split qkv : ", qkv.shape)                                # torch.Size([197, 3, 12, 64])
q = qkv[:, 0].permute(1, 0, 2)                                  # (H=12, N=197, D/H=64)
k = qkv[:, 1].permute(1, 0, 2)                                  # (H=12, N=197, D/H=64)
kT = k.permute(0, 2, 1)                                         # (H=12, D/H=64, N=197)
print("transposed ks: ", kT.shape)                              # torch.Size([12, 64, 197])


## REMOVE: View Attention Matrix
attention_matrix = q @ kT
print("attention matrix: ", attention_matrix.shape)
plt.imshow(attention_matrix[3].detach().cpu().numpy())
##[outputs image]


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## [4] MLP (Classification) Head
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# The 0-th output vector from the transformer output vectors (corresponding to the class token input) 
# is fed to the MLP head. The 1000-dimension classification result is the output of the whole pipeline.

print("Classification head: ", model.head)
result = model.head(transformer_output)
result_label_id = int(torch.argmax(result))

plt.plot(result.detach().cpu().numpy()[0])
plt.title("Classification result")
plt.xlabel("class id")

print("Inference result : id = {}, label name = {}".format(
    result_label_id, imagenet_labels[result_label_id]))

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
## 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
