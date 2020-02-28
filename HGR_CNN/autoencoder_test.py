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

# TODO REMOVE!!!!
current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_script_path, "dataset")


def custom_loss(y_true, y_pred, coeff = 5):
    y_true1, y_pred1 = (255 - y_true) / 255, (255 - y_pred) / 255
    dif = y_true1 - y_pred1
    temp1 = K.l2_normalize(K.cast(K.greater(dif, 0),"float32")*y_true1,axis=-1) * coeff
    temp2 = K.cast(K.less(dif, 0),"float32")
    weight = temp1 + temp2  
    loss = K.abs(weight*dif)
    return K.mean(loss)   ##you need to return this when you use in your code
    #return K.eval(loss)


img_camera_size = (640, 480) 
img_dataset_size = (160, 120)

depth_max = float(1000) # millimeters
img_loader = loader.ImageDataManager(current_script_path, "", "rgbd", img_dataset_size, img_camera_size, depth_max)

grayscale_img,_ = img_loader.load_single_img("rgbd_97_X0_Y0_Z0_hand0_gest0_date02-25-2020_16#57#02.png")

test = [grayscale_img.astype("float32") / 255.0]
test = np.array(test)

#grayscale_img = cv2.imread("rgbd_34_X551_Y476_Z636_hand1_gest1_date02-12-2020_14#49#51.png", cv2.IMREAD_GRAYSCALE)
#resized = cv2.resize(grayscale_img, img_dataset_size).astype(np.float32)

#test = [np.expand_dims(resized.astype("float32") / 255.0, axis=-1)]
#test = np.array(test)

autoencoder = tf.keras.models.load_model("autoencoder_model.h5", custom_objects={'custom_loss': custom_loss})

decoded = np.squeeze(autoencoder.predict(test) * 255)

cv2.imwrite("denoised.png", decoded.astype("uint8"))
