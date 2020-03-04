import cv2
import os
import sys
import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import datatypes
import image_data_manager as loader
import dataset_generator as gen
import cnn_model 
import predictor_facade as predictor
import SimulationPredictor as spredictor
from time import time
import time as tm

record_command = "record"
train_command = "train"
predict_command = "predict"
continue_train = "continue_train"
online_command = "online_prection"
simulation_command = "simulation_prection"

model_name = "current_model.h5"

current_script_path = os.path.dirname(os.path.realpath(__file__))
logs_dir = os.path.join(current_script_path, "logs", format(time()))
dataset_dir = os.path.join(current_script_path, os.pardir, "dataset") # pardir - parent dir (one lvl up)
models_dir = os.path.join(current_script_path, "models") 

record_when_no_hand = False
recorded_gesture = datatypes.Gesture.POINTING

img_camera_size = (640, 480) 
img_dataset_size = (160, 120)

xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
depth_max = float(1000) # millimeters
depth_min = float(0)
x_min = float(-700)
x_max = float(700)
y_min = float(-600)
y_max = float(500)

filters_count = 32
learning_rate = 0.0008 # use high values because we have BatchNorm and Dropout
batch_size = 100
epochs_count = 50
test_data_ratio = 0.2

if __name__ == "__main__":

    #sys.argv = [sys.argv[0], record_command]
    #sys.argv = [sys.argv[0], train_command] 
    #sys.argv = [sys.argv[0], continue_train] 
    #sys.argv = [sys.argv[0], predict_command, os.path.join(dataset_dir, "depth_77_X356.4_Y-342.1_Z414.0_hand1_gest1_date03-02-2020_15-33-02.png")]
    #sys.argv = [sys.argv[0], predict_command, "depth_77_X356.4_Y-342.1_Z414.0_hand1_gest1_date03-02-2020_15-33-02.jpg"]
    #sys.argv = [sys.argv[0], online_command]
    sys.argv = [sys.argv[0], simulation_command]
    print(sys.argv) 

    img_loader = loader.ImageDataManager(current_script_path, dataset_dir, "depth", img_dataset_size, xyz_ranges)
    
    if (len(sys.argv) == 1):
        print("No arguments provided. See help (-h).")
        sys.exit(0)

    if (sys.argv[1] == record_command):
        print("Dataset recording...")
        recorder = gen.DatasetGenerator(record_when_no_hand, dataset_dir, img_camera_size, img_dataset_size, xyz_ranges)
        recorder.record_data(recorded_gesture)
        sys.exit(0)

    if (sys.argv[1] == train_command):
        print("Training...")
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, None)
        X_data, y_data = img_loader.get_train_data()
        model.train(X_data, y_data, epochs_count, batch_size, logs_dir, test_data_ratio)
        model.save(os.path.join(models_dir, "new_model.h5"))
        sys.exit(0)

    if (sys.argv[1] == continue_train): # FIXME  increases error!!! 
        prev_model_name = "new_model.h5" # input from params
        print(f"Continue training of {prev_model_name}")
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(models_dir, prev_model_name))
        X_data, y_data = img_loader.get_train_data()
        model.train(X_data, y_data, epochs_count, batch_size, logs_dir, test_data_ratio)
        model.save(os.path.join(models_dir, "new_model.h5"))
        sys.exit(0)

    if (sys.argv[1] == online_command):
        print("Online prediction...")
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(models_dir, model_name))
        predict = predictor.OnlinePredictor(model, img_camera_size, img_dataset_size, xyz_ranges)
        predict.predict_online()
        sys.exit(0)

    if (sys.argv[1] == predict_command and not(sys.argv[2].isspace())):
        print("Predicting: %s" % sys.argv[2])
        tf.config.set_visible_devices([], 'GPU') #  faster for prediction (2x), or loading?
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(models_dir, model_name))
        X_predict, y_predict = img_loader.load_single_img(sys.argv[2])
        #cv2.imwrite(os.path.join(current_script_path, "rgbd_6791_X261_Y175_Z356_hand1_gest1_date02-12-2020_14#41#54 TTT.png"), X_predict1)
        #X_predict, y_predict = img_loader.load_single_img(os.path.join(current_script_path, "rgbd_6791_X261_Y175_Z356_hand1_gest1_date02-12-2020_14#41#54 TTT.png"))
        result = model.predict_single_image(X_predict, y_predict)
        
        print("Expected: %s - predicted: %s" % (y_predict, result))
        for i in range(0,3):  
            result[i] = result[i] *  (abs(xyz_ranges[i][0]) +  xyz_ranges[i][1]) - abs(xyz_ranges[i][0])
        result = np.round(result).astype("int")
        print("[X:%s; Y:%s; Z:%s; Hand:%s; Gesture:%s;]" % (result[0],result[1],result[2], result[3] == 1, result[4]))
        sys.exit(0)

    if (sys.argv[1] == simulation_command):
        print("Simulation prediction...")
        model = cnn_model.CnnModel(filters_count, learning_rate, img_dataset_size, os.path.join(models_dir, model_name))
        spredict = spredictor.SimulationPredictor(model, img_camera_size, img_dataset_size, depth_max,depth_min,x_min,x_max,y_min,y_max, xyz_ranges)
        spredict.predict_online()
        sys.exit(0)