
"""
ChatGPT
"""
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from efficientnet.keras import preprocess_input

x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)


from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from efficientnet.keras import EfficientNetB0

base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
