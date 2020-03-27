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
import tensorboard_starter
import predictor_facade as p
import simulation_predictor as sp
from time import time, strftime, gmtime
import time as tm
import autoencoder_wrapper as m
import autoencoder_unet_model as current_model
import config as c
from datetime import datetime

record_command = "record"
train_command = "train"
predict_command = "predict"
continue_train = "continue_train"
camera_command = "camera_prediction"
simulation_command = "simulation_prediction"

config = c.Configuration(version_name = "autoencoder", debug_mode=True, latest_model_name="autoencoder_model.h5")

new_model_path = os.path.join(config.models_dir, "new_model.h5")

if __name__ == "__main__":
    #sys.argv = [sys.argv[0], record_command]
    #sys.argv = [sys.argv[0], train_command] 
    sys.argv = [sys.argv[0], continue_train, "c_model.h5"] 
    #sys.argv = [sys.argv[0], predict_command, os.path.join(c.current_dir_path, "testdata", "test5.jpg")]
    #sys.argv = [sys.argv[0], camera_command]
    #sys.argv = [sys.argv[0], simulation_command]
    c.msg(sys.argv) 

    m.check_gpu()

    img_manager = loader.ImageDataManager(config)
    
    if (len(sys.argv) == 1):
        c.msg("No arguments provided.")
        sys.exit(0)

    if (sys.argv[1] == record_command):
        c.msg("Dataset recording...")
        recorder = gen.DatasetGenerator(config, img_manager)
        recorder.record_data(current_gesture = datatypes.Gesture.POINTING)
        sys.exit(0)

    if (sys.argv[1] == "tb"): #start tensorboard
        c.msg("Starting tensorboard...")
        tensorboard_starter.start_and_open()
        sys.exit(0)

    if (sys.argv[1] == train_command):
        c.msg("Training...")
        model = m.ModelWrapper(current_model.build(config.img_dataset_size), config)
        model.recompile()
        model.train(*img_manager.get_autoencoder_datagens())
        model.save(new_model_path)
        sys.exit(0)

    if (sys.argv[1] == continue_train and not(sys.argv[2].isspace())): 
        prev_model_name = sys.argv[2]
        c.msg(f"Continue training of {prev_model_name}...")
        model = m.load_model(os.path.join(config.models_dir, prev_model_name), config)
        model.train(*img_manager.get_autoencoder_datagens())
        model.save(new_model_path)
        sys.exit(0)

    if (sys.argv[1] == camera_command):
        c.msg("Online prediction from camera...")
        model = m.load_model(config.latest_model_path, config)
        predictor = p.OnlinePredictor(model, config, img_manager)
        predictor.predict_online()
        sys.exit(0)

    if (sys.argv[1] == predict_command and not(sys.argv[2].isspace())):
        c.msg("Predicting: %s" % sys.argv[2])
        model = m.load_model(config.latest_model_path, config)
        img = img_manager.load_single_img(sys.argv[2])
        mask = model.predict_single(img)
        result = img_manager.img_mask_alongside(img, mask)
        #img_manager.save_image(result, os.path.join(c.current_dir_path, "testdata", "mask.png"))
        img_manager.show_image(result)
        # TODO test method for each model wrapper type
        sys.exit(0)

    if (sys.argv[1] == simulation_command):
        c.msg("Online prediction from simulation...")
        model = m.load_model(config.latest_model_path, config)
        simulation = sp.SimulationPredictor(model, config, img_manager)
        simulation.predict_online()
        sys.exit(0)

    c.msg("Unrecognized input args. Check spelling.")