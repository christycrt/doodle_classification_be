import tensorflow as tf

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import train_test_split as tts

data_dir = './data/doodle_data'

import os

quickdraw_classes = []
for root, dirs, files in os.walk(data_dir):
    for filename in files:
        quickdraw_classes.append(filename[18:-4])

print("class size: ", len(quickdraw_classes))
print("class: ", quickdraw_classes)

quickdraw = []
labels = []
i = 0
for quickdraw_class in quickdraw_classes:
  i += 1
  print(i,quickdraw_class, quickdraw_classes.index(quickdraw_class))
  data = np.load(data_dir + "/full_numpy_bitmap_"+quickdraw_class+".npy")
  data = data[:60000]
  for d in data:
    quickdraw.append(d.reshape(28, 28, 1)/255.0)
    labels.append(quickdraw_classes.index(quickdraw_class))
quickdraw = np.array(quickdraw)
labels = np.array(labels)

print(quickdraw.shape)
print(len(labels))

x_train, x_test, y_train, y_test = tts(quickdraw, labels, test_size=0.2, random_state=5)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32, (5, 5),input_shape=(28, 28, 1), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
model.add(tf.keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(quickdraw_classes), activation='softmax'))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_split=0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)

print('\nTest accuracy:', test_acc)

model.save("doodle_model_v1.h5")

