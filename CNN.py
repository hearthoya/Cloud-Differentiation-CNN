import tensorflow as tf
import os
import cv2
import imghdr
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from keras.metrics import Precision, Recall, BinaryAccuracy

# directory for data
data_dir = 'data'
# array of img extension types
image_exts = ['jpeg', 'jpg', 'bmp', 'png']
# remove bad images
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if tip not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print('Issue with image {}'.format(image_path))
# load data
data = tf.keras.utils.image_dataset_from_directory('data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
# scale data
data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()
# define sizes of each  data partition
train_size = int(len(data) * .7)
val_size = int(len(data) * .2) + 1
test_size = int(len(data) * .1) + 1
# actually split data up into training, validation and testing
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)
# make model
model = tf.keras.Sequential()
# layer 1 and 2
model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# layer 3 and 4
model.add(tf.keras.layers.Conv2D(32, (3, 3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# layer 5 and 6
model.add(tf.keras.layers.Conv2D(16, (3, 3), 1, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))
# layer 7
model.add(tf.keras.layers.Flatten())
# layer 8 and 9
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
# compile
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
# train model
logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=30, validation_data=val, callbacks=[tensorboard_callback])
# # plot data
# fig = plt.figure()
# plt.plot(hist.history['loss'], color='teal', label='loss')
# plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
# fig.suptitle('Loss', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()
# test
testImg = cv2.imread("cirrustest.jpg")
testOut = model.predict(np.expand_dims((tf.image.resize(testImg, (256, 256))) / 255, 0))
if testOut > 0.5:
    print(f'Test image is Cumulonimbus!')
else:
    print(f'Test image is Cirrus!')
# save model
model.save(os.path.join('models', 'cloudmodel.keras'))

# load saved model
# from keras.models import load_model
# new_model = load_model(os.path.join('models', 'cloudmodel.keras'))
