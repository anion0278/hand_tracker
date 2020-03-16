import matplotlib

# import the necessary packages
from convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import tensorflow.keras.backend as K
import cv2
import os
import image_data_manager as loader
import autoencoder_model

current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_script_path, "dataset")

img_camera_size = (640, 480) 
img_dataset_size = (320, 240)

depth_max = float(1000) 
img_loader = loader.ImageDataManager(current_script_path, "", "rgbd", img_dataset_size, img_camera_size, depth_max)

grayscale_img,_ = img_loader.load_single_img("rgbd_97_X0_Y0_Z0_hand0_gest0_date02-25-2020_16#57#02.png")

test = [grayscale_img.astype("float32") / 255.0]
test = np.array(test)

autoencoder = tf.keras.models.load_model("autoencoder_model.h5", custom_objects={'custom_loss': custom_loss})

decoded = np.squeeze(autoencoder.predict(test) * 255)

cv2.imwrite("denoised.png", decoded.astype("uint8"))
