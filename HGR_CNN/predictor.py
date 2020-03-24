import os
import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras.backend as K
import image_data_manager as loader

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

class Predictor:
    def __init__(self,model_name,):
        self.model_name = model_name
        self.autoencoder = tf.keras.models.load_model(self.model_name, custom_objects={'dice_loss': dice_loss, "dice_coef": dice_coef})

    def predict(self,img):
        predicted = np.squeeze(self.autoencoder.predict(img))
        return predicted


if __name__ == "__main__":
    current_script_path = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(current_script_path, os.pardir, "dataset")
    pr = Predictor(os.path.join(current_script_path,"models","autoencoder_model.h5"))