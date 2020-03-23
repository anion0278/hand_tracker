import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import image_data_manager as loader


current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_script_path, os.pardir, "dataset")
smooth = 1.

def dice_coef(y_true, y_pred):
    # should be used due to unbalanced labels
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

img_camera_size = (640, 480) 
img_dataset_size = (320, 240)
xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]

img_loader = loader.ImageDataManager(current_script_path,dataset_dir,"depth",img_dataset_size,xyz_ranges)

grayscale_img,_ = img_loader.load_single_img("depth_22780_X-158.6_Y58.3_Z459.0_hand1_gest1_date03-11-2020_21#45#50.jpg")

test = [grayscale_img.astype("float32") / 255.0]
test = np.array(test)

autoencoder = tf.keras.models.load_model(os.path.join(current_script_path,"models","autoencoder_model.h5"), custom_objects={'dice_loss': dice_loss, "dice_coef": dice_coef})

decoded = np.squeeze(autoencoder.predict(test) * 255)

cv2.imwrite("denoised.png", decoded.astype("uint8"))