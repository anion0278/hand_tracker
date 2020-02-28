import cv2
import os
import sys
import pyrealsense2 as rs
import numpy as np
import datatypes
import image_data_manager as loader
import dataset_generator as gen
import cnn_model 
import predictor_facade as predictor 
from time import time

record_command = "record"
train_command = "train"
predict_command = "predict"
online_command = "online_prection"

model_name = "current_model.h5"

current_script_path = os.path.dirname(os.path.realpath(__file__))
logs_dir = os.path.join(current_script_path, "logs", format(time()))
dataset_dir = os.path.join(current_script_path, "dataset")

record_when_no_hand = False
recorded_gesture = datatypes.Gesture.POINTING

img_camera_size = (640, 480) 
img_dataset_size = (160, 120)

depth_max = float(1000) # millimeters

filters_count = 32
learning_rate = 0.0005
batch_size = 100
epochs_count = 40
test_data_ratio = 0.05

if __name__ == "__main__":

    sys.argv = [sys.argv[0], record_command]
    #sys.argv = [sys.argv[0], train_command]
    #sys.argv = [sys.argv[0], predict_command, os.path.join(dataset_dir, "rgbd_7638_X0_Y0_Z0_hand0_gest0_date02-12-2020_14#55#42.png")]
    #sys.argv = [sys.argv[0], online_command]
    print(sys.argv) 

    img_loader = loader.ImageDataManager(current_script_path, dataset_dir, "rgbd", img_dataset_size, img_camera_size, depth_max)
    
    if (len(sys.argv) == 1):
        print("No arguments provided. See help (-h).")
        sys.exit(0)

    if (sys.argv[1] == record_command):
        print("Dataset recording...")
        recorder = gen.DatasetGenerator(record_when_no_hand, dataset_dir, img_camera_size, img_dataset_size, depth_max)
        recorder.record_data(recorded_gesture)
        sys.exit(0)

    if (sys.argv[1] == train_command):
        print("Training...")
        X_data, y_data = img_loader.get_train_data()
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, None)
        model.train(X_data, y_data, epochs_count, batch_size, logs_dir, test_data_ratio)
        model.save(os.path.join(current_script_path, "new_model.h5"))
        sys.exit(0)

    if (sys.argv[1] == online_command):
        print("Online prediction...")
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(current_script_path, model_name))
        predict = predictor.OnlinePredictor(model, img_camera_size, img_dataset_size, depth_max)
        predict.predict_online()
        sys.exit(0)

    if (sys.argv[1] == predict_command and not(sys.argv[2].isspace())):
        print("Predicting: %s" % sys.argv[2])
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(current_script_path, model_name))
        X_predict, y_predict = img_loader.load_single_img(sys.argv[2])
        #cv2.imwrite(os.path.join(current_script_path, "rgbd_6791_X261_Y175_Z356_hand1_gest1_date02-12-2020_14#41#54 TTT.png"), X_predict1)
        #X_predict, y_predict = img_loader.load_single_img(os.path.join(current_script_path, "rgbd_6791_X261_Y175_Z356_hand1_gest1_date02-12-2020_14#41#54 TTT.png"))
        result = model.predict_single_image(X_predict, y_predict)
        
        print("Expected: %s - predicted: %s" % (y_predict, result))
        result[0] *= img_camera_size[0]
        result[1] *= img_camera_size[1]
        result[2] *= depth_max
        result = np.round(result).astype("int")
        print("[X:%s; Y:%s; Z:%s; Hand:%s; Gesture:%s;]" % (result[0],result[1],result[2], result[3] == 1, result[4]))
        sys.exit(0)