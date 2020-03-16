import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import image_data_manager as loader
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import *
import cv2
import os
from sklearn.model_selection import train_test_split
#from convautoencoder import ConvAutoencoder
from autoencoder_model import ConvAutoencoder

EPOCHS = 5
BS = 32

current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_script_path, os.pardir, "dataset_test")

xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
img_dataset_size = (320, 240)

smooth = 1.

def dice_coef(y_true, y_pred):
    # should be used due to unbalanced labels
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


# TODO use data-augmentation to create more samples
img_loader = loader.ImageDataManager(current_script_path, dataset_dir, "depth", img_dataset_size, xyz_ranges)
    
X_imgs, y_imgs = img_loader.get_encoder_data()
X_imgs = X_imgs.astype("float32") / 255.0
y_imgs = y_imgs.astype("float32") / 255.0
X_train, X_test, y_train, y_test = train_test_split(X_imgs, y_imgs, test_size=0.2, random_state=42)

autoencoder = ConvAutoencoder.build(*img_dataset_size, 1)

autoencoder.compile(loss=dice_loss, optimizer=Adadelta(), metrics=[dice_coef, "binary_crossentropy", 'accuracy'])

try:
    history = autoencoder.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = EPOCHS, batch_size = BS)
except KeyboardInterrupt:
    print("Interrupted")
finally:
    autoencoder.save("autoencoder_model.h5")

