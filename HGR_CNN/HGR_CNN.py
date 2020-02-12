import cv2
import os
import sys
import pyrealsense2 as rs
import numpy as np
import datatypes
import image_data_loader as loader
import dataset_generator as gen
import cnn_model 
from time import time

record_command = "record"
train_command = "train"
prediction_command = "online_prection"

current_script_path = os.path.dirname(os.path.realpath(__file__))
logs_dir = os.path.join(current_script_path, "logs", format(time()))
dataset_dir = os.path.join(current_script_path, "dataset")

img_camera_size = (640, 480) 
img_dataset_size = (160, 120)

filters_count = 32
learning_rate = 0.0001
batch_size = 64
epochs_count = 50
test_data_ratio = 0.2

if __name__ == "__main__":

    #sys.argv = [sys.argv[0], record_command]
    sys.argv = [sys.argv[0], train_command]
    #sys.argv = [sys.argv[0], prediction_command]
    print(sys.argv) 

    img_loader = loader.ImageDataLoader(current_script_path, dataset_dir, "rgbd", img_dataset_size)

    if (len(sys.argv) == 1):
        print("No arguments provided. See help (-h).")
        sys.exit(0)

    if (sys.argv[1] == record_command):
        print("Dataset recording...")
        recorder = gen.DatasetGenerator(dataset_dir, img_camera_size, img_dataset_size)
        recorder.record_data(datatypes.Gesture.POINTING)
        sys.exit(0)

    if (sys.argv[1] == train_command):
        print("Training...")
        X_data, y_data = img_loader.get_train_data()
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, None)
        model.train(X_data, y_data, epochs_count, batch_size, logs_dir, test_data_ratio)
        model.save(os.path.join(current_script_path, "new_model.h5"))
        sys.exit(0)