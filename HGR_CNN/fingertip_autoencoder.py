
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from convautoencoder import ConvAutoencoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import argparse
import image_data_manager as loader
import tensorflow.keras.backend as K
import cv2
import os
from sklearn.model_selection import train_test_split

def ssim_loss(y_true, y_pred):
	return -tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

def custom_loss(y_true, y_pred, coeff = 5):
    y_true1, y_pred1 = (255 - y_true) / 255, (255 - y_pred) / 255
    dif = y_true1 - y_pred1
    temp1 = K.l2_normalize(K.cast(K.greater(dif, 0),"float32")*y_true1,axis=-1) * coeff
    temp2 = K.cast(K.less(dif, 0),"float32")
    weight = temp1 + temp2  
    loss = K.abs(weight*dif)
    return K.mean(loss)   ##you need to return this when you use in your code
    #return K.eval(loss)

# initialize the number of epochs to train for and batch size
EPOCHS = 5
BS = 32

# TODO REMOVE!!!!
current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_script_path, "dataset")

img_camera_size = (640, 480) 
img_dataset_size = (160, 120)

depth_max = float(1000) # millimeters
img_loader = loader.ImageDataManager(current_script_path, dataset_dir, "rgbd", img_dataset_size, img_camera_size, depth_max)
    
# load the MNIST dataset
X_imgs, y_imgs = img_loader.get_encoder_data()

# add a channel dimension to every image in the dataset, then scale
# the pixel intensities to the range [0, 1]
X_imgs = X_imgs.astype("float32") / 255.0
y_imgs = y_imgs.astype("float32") / 255.0

X_train, X_test, y_train, y_test = train_test_split(X_imgs, y_imgs, test_size=0.2, random_state=42)

# construct our convolutional autoencoder
(encoder, decoder, autoencoder) = ConvAutoencoder.build(160, 120, 1)
opt = Adam(lr = 1e-3)
autoencoder.compile(loss="binary_crossentropy", optimizer=opt, metrics=[custom_loss, 'mse', 'accuracy'])

try:
    # train the convolutional autoencoder
    H = autoencoder.fit(X_train, y_train,
	    validation_data = (X_test, y_test),
	    epochs = EPOCHS,
	    batch_size = BS)
except KeyboardInterrupt:
    print("Interrupted")
finally:
    autoencoder.save("autoencoder_model.h5")

# construct a plot that plots and saves the training history
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label = "train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("history_plot.png")

# use the convolutional autoencoder to make predictions on the
# testing images, then initialize our list of output images
#decoded = autoencoder.predict(testX)
#outputs = None

## save the outputs image to disk
#cv2.imwrite(args["output"], outputs)
