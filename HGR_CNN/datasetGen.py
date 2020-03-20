import cv2
import numpy as np
import os
import image_data_manager as loader
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import *
from convautoencoder import ConvAutoencoder
from random import shuffle

EPOCHS = 5
BS = 32

smooth = 1.

current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_script_path, os.pardir, "dataset")

xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
img_dataset_size = (320, 240)

def train_generator(img_dir, batch_size, input_size):
    list_images = os.listdir(img_dir)
    shuffle(list_images) #Randomize the choice of batches
    ids_train_split = range(len(list_images))
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                img,mask = ld.load_image_pair(list_images[id])
                x_batch.append(img)
                y_batch.append(mask)

            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32)

            yield x_batch, y_batch

def dice_coef(y_true, y_pred):
    # should be used due to unbalanced labels
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

ld = loader.ImageDataManager(current_script_path,dataset_dir,"depth",img_dataset_size,xyz_ranges)

autoencoder = ConvAutoencoder.build(*img_dataset_size, 1)
autoencoder.compile(loss=dice_loss, optimizer=Adadelta(), metrics=[dice_coef, "binary_crossentropy", 'accuracy'])

try:
    history = autoencoder.fit_generator(train_generator(os.path.join(dataset_dir,"mask"),BS,img_dataset_size),steps_per_epoch=2,epochs=EPOCHS,verbose=1)
except KeyboardInterrupt:
    print("Interrupted")
finally:
    autoencoder.save("autoencoder_model.h5")


