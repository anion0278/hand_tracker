import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

smooth = 1.

def dice_coef(y_true, y_pred):
    # should be used due to unbalanced labels
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)


class Predictor:
    def __init__(self,model_name):
        self.model_name = model_name
        self.autoencoder = tf.keras.models.load_model(self.model_name, custom_objects={'dice_loss': dice_loss, "dice_coef": dice_coef})

    def predict(self,img):
        predicted = np.squeeze(self.autoencoder.predict(img))
        return predicted
