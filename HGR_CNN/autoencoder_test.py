import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
import image_data_manager as loader
import autoencoder_train

current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_script_path, "dataset")

img_camera_size = (640, 480) 
img_dataset_size = (320, 240)

depth_max = float(1000) 
img_loader = loader.ImageDataManager(current_script_path, "", "depth", img_dataset_size, img_camera_size, depth_max)

grayscale_img,_ = img_loader.load_single_img("depth_22780_X-158.6_Y58.3_Z459.0_hand1_gest1_date03-11-2020_21#45#50n0.jpg")

test = [grayscale_img.astype("float32") / 255.0]
test = np.array(test)

autoencoder = tf.keras.models.load_model("autoencoder_model.h5", custom_objects={'dice_loss': autoencoder_train.dice_loss, "dice_coef": dice_coef})

decoded = np.squeeze(autoencoder.predict(test) * 255)

cv2.imwrite("denoised.png", decoded.astype("uint8"))
