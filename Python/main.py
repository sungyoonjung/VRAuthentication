
# This is a script to work with VR Auth app
# Written by: Sung Yoon Jung
# Date: 3/20/22

import os
import numpy as np
from PIL import Image as im
import matplotlib.pyplot as plt
import PIL
import pathlib
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# convert the goodsigs text files into images
for x in range(100):
  textfileaddy = "C:/Users/SYJ/Documents/Code/Unity/VRAuth/textfiles/goodsigs/goodsig (" + str(x+1) + ").txt"
  rawdata = np.loadtxt(textfileaddy, delimiter=",")
  # Creates PIL image
  img = im.fromarray(np.uint8(rawdata) , 'L')
  #display(img)
  img.save(r"C:/Users/SYJ/Documents/Code/Unity/VRAuth/images/goodsigs/goodsig" + str(x+1) + ".jpg")

# convert the goodsigs text files into images
# for x in range(114):
#   textfileaddy = "C:/Users/SYJ/Documents/Code/Unity/VRAuth/textfiles/othersigs/othersig (" + str(x+1) + ").txt"
#   rawdata = np.loadtxt(textfileaddy, delimiter=",")
#   # Creates PIL image
#   img = im.fromarray(np.uint8(rawdata) , 'L')
#   #display(img)
#   img.save(r"C:/Users/SYJ/Documents/Code/Unity/VRAuth/images/othersigs/othersig" + str(x+1) + ".jpg")

filectr = 1
# convert a single text file into an image
textfileaddy = "C:/Users/SYJ/Documents/Code/Unity/VRAuth/textfiles/testsig"+str(filectr)+".txt"
rawdata = np.loadtxt(textfileaddy, delimiter=",")
img = im.fromarray(np.uint8(rawdata), 'L')
img.save(r"C:/Users/SYJ/Documents/Code/Unity/VRAuth/images/testsig"+str(filectr)+".jpg")



# -------------------------------------------------------------------
# ---------------- CREATING THE MODEL     ---------------------------
# -------------------------------------------------------------------


# CREATE A DATASET
batch_size = 16
img_height = 50
img_width = 100

data_dir = "C:/Users/SYJ/Documents/Code/Unity/VRAuth/images/"
class_names = ['goodsigs', 'othersigs']

# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
#
# AUTOTUNE = tf.data.AUTOTUNE
#
# train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
#
# normalization_layer = layers.Rescaling(1./255)
# num_classes = len(class_names)
#
# model1 = Sequential([
#   layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])
#
# model1.compile(optimizer='adam',
#                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                metrics=['accuracy'])
#
# model1.summary()
#
# # train the model!
# epochs=10
# history = model1.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# save model:
# model1.save("C:/Users/SYJ/Documents/Code/Unity/VRAuth/model1/");
# It can be used to reconstruct the model identically.
# reconstructed_model = keras.models.load_model("C:/Users/SYJ/Documents/Code/Unity/VRAuth/model1/")

# visualize training results
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
#
# data_augmentation = keras.Sequential(
#   [
#     layers.RandomFlip("horizontal",
#                       input_shape=(img_height,
#                                   img_width,
#                                   3)),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#   ]
# )
#
# # make a new model with augmentation
#
# model2 = Sequential([
#   data_augmentation,
#   layers.Rescaling(1./255),
#   layers.Conv2D(16, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding='same', activation='relu'),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.2),
#   layers.Flatten(),
#   layers.Dense(128, activation='relu'),
#   layers.Dense(num_classes)
# ])
#
# # compile the new model
# model2.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])
#
# model2.summary()
#
# # train the new model!
# epochs = 15
# history = model2.fit(
#   train_ds,
#   validation_data=val_ds,
#   epochs=epochs
# )

# save the new model
# model2.save("C:/Users/SYJ/Documents/Code/Unity/VRAuth/model2/");
#
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs_range = range(epochs)
#
# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()


# *****************************************************************
# *****************************************************************
# *****************************************************************
# PREDICTION TIME BABY

reconstructed_model = keras.models.load_model("C:/Users/SYJ/Documents/Code/Unity/VRAuth/model2/")

filectr = 1
testimagepath = "C:/Users/SYJ/Documents/Code/Unity/VRAuth/Images/testsig"+str(filectr)+".jpg"

testimage = tf.keras.utils.load_img(
    testimagepath, target_size=(img_height, img_width)
)

img_array = tf.keras.utils.img_to_array(testimage)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = reconstructed_model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)

if class_names[np.argmax(score)] == "goodsigs":
  with open("C:/Users/SYJ/Documents/Code/Unity/VRAuth/textfiles/results/results.txt", 'w') as writefile:
    writefile.write("Signature Confirmed! {:.2f}% confidence."
    .format(100 * np.max(score)))
else:
  with open("C:/Users/SYJ/Documents/Code/Unity/VRAuth/textfiles/results/results.txt", 'w') as writefile:
    writefile.write("Authentication FAILED! {:.2f}% confidence."
    .format(100 * np.max(score)))

filectr += 1