# -*- coding: utf-8 -*-
"""
Input layer consists of (1, 8, 28) values.

First layer, Conv2D consists of 32 filters and ‘relu’ activation function with kernel size, (3,3).

Second layer, Conv2D consists of 64 filters and ‘relu’ activation function with kernel size, (3,3).

Thrid layer, MaxPooling has pool size of (2, 2).

Fifth layer, Flatten is used to flatten all its input into single dimension.

Sixth layer, Dense consists of 128 neurons and ‘relu’ activation function.

Seventh layer, Dropout has 0.5 as its value.

Eighth and final layer consists of 10 neurons and ‘softmax’ activation function.

Use categorical_crossentropy as loss function.

Use Adadelta() as Optimizer.

Use accuracy as metrics.

Use 128 as batch size.

Use 20 as epochs
"""
import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28 

if K.image_data_format() == 'channels_first': 
   x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols) 
   x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
   input_shape = (1, img_rows, img_cols) 
else: 
   x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
   x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
   input_shape = (img_rows, img_cols, 1) 
   
x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255 
x_test /= 255 

y_train = keras.utils.to_categorical(y_train, 10) 
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential() 
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape)) 
model.add(Conv2D(64, (3, 3), activation = 'relu')) 
model.add(MaxPooling2D(pool_size = (2, 2))) 
model.add(Dropout(0.25)) model.add(Flatten()) 
model.add(Dense(128, activation = 'relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(10, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 128, epochs = 12, verbose = 1, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 0) 

print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

pred = model.predict(x_test) 
pred = np.argmax(pred, axis = 1)[:5] 
label = np.argmax(y_test,axis = 1)[:5] 

print(pred) 
print(label)
