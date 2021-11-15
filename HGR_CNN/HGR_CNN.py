import os
import cv2
import sys
import tensorflow as tf
import pyrealsense2 as rs
import numpy as np
import datatypes
import image_data_manager as idm
import dataset_generator as gen
import tensorboard_starter
import predictor as p
import autoencoder_wrapper as m
import autoencoder_unet_model as current_model
#import autoencoder_simple_model as current_model
import config as c
import video_catcher as vc
import simulation_catcher as sc
import dataset_manager as dm

record_command = "record"
train_command = "train"
predict_command = "predict"
continue_train = "continue_train"
camera_command = "camera_prediction"
simulation_command = "simulation_prediction"
evaluate_command = "evaluate model"
tb_command = "tb"
show_command = "show"

config = c.Configuration(version_name = "autoencoder", debug_mode=False, latest_model_name="CD_270k.h5")
#config = c.Configuration(version_name = "autoencoder", debug_mode=False, latest_model_name="UR3_fullhand_300k.h5")


new_model_path = os.path.join(config.models_dir, "new_model.h5")

if __name__ == "__main__":
    #sys.argv = [sys.argv[0], tb_command]
    #sys.argv = [sys.argv[0], record_command]
    #sys.argv = [sys.argv[0], train_command] 
    #sys.argv = [sys.argv[0], continue_train, "hand_only_and_bgr.h5"] 
    #sys.argv = [sys.argv[0], predict_command, os.path.join(c.current_dir_path, "testdata", "41.png")]
    sys.argv = [sys.argv[0], camera_command]
    #sys.argv = [sys.argv[0], evaluate_command]
    #sys.argv = [sys.argv[0], simulation_command]
    #sys.argv = [sys.argv[0], show_command]
    c.msg(sys.argv) 

    m.check_gpu()

    img_manager = idm.ImageDataManager(config)
    
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
        dataset_manager = dm.DatasetManager(config)
        model = m.ModelWrapper(current_model.build(config.img_dataset_size), config)
        model.recompile()
        model.save_model_graph_img() # possibly visualize
        model.train(*dataset_manager.get_autoencoder_datagens())
        model.save(new_model_path)
        sys.exit(0)

    if (sys.argv[1] == continue_train and not(sys.argv[2].isspace())): 
        prev_model_name = sys.argv[2]
        dataset_manager = dm.DatasetManager(config)
        c.msg(f"Continue training of {prev_model_name}...")
        model = m.load_model(os.path.join(config.models_dir, prev_model_name), config)
        model.train(*dataset_manager.get_autoencoder_datagens())
        model.save(new_model_path)
        sys.exit(0)

    if (sys.argv[1] == camera_command):
        c.msg("Online prediction from camera...")
        catcher = vc.VideoImageCatcher(config)
        model = m.load_model(config.latest_model_path, config)
        predictor = p.Predictor(model,config,img_manager,catcher)
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
        catcher = sc.SimulationCatcher(config)
        model = m.load_model(config.latest_model_path, config)
        simulation = p.Predictor(model, config, img_manager,catcher)
        simulation.predict_online()
        sys.exit(0)

    if (sys.argv[1] == evaluate_command):
        dataset_manager = dm.DatasetManager(config)
        c.msg("Evaluation of selected model...")
        model = m.load_model(config.latest_model_path, config)
        model.evaluate(*dataset_manager.get_eval_datagens())
        sys.exit(0)

    if (sys.argv[1] == show_command):
        dataset_manager = dm.DatasetManager(config)
        c.msg("Showing first batch...")
        model = m.load_model(config.latest_model_path, config)
        #model.show(*dataset_manager.get_eval_datagens())
        model.show2(*dataset_manager.get_autoencoder_datagens())
        sys.exit(0)
    c.msg("Unrecognized input args. Check spelling.")

#import predictor_facade as p
#import simulation_predictor as sp
#import video_fetcher as vf
#import simulation_fetcher as sf
#import visualizer as vis
#import data_logger as dl
#import results_evaluator as res

        #if (sys.argv[1] == online_command):
        #print("Online prediction...")
        #model_name = os.path.join(models_dir,"autoencoder_model.h5")
        #predictor = pr.Predictor(model_name)
        #video_fetcher = vf.VideoImageFetcher(img_camera_size,camera_rate)
        #video_fetcher.init_stream()
        #key = 1

        #while key != 27:
        #    source = image_manager.encode_camera_image(video_fetcher.get_depth_raw())
        #    predicted = predictor.predict(source)
        #    visualizer.display_video("predicted",image_manager.decode_predicted(predicted),1)
        #    visualizer.display_video("source",video_fetcher.get_depth_img(),1)
        #    visualizer.display_video("camera image",video_fetcher.get_color(),1)

        #video_fetcher.close_stream()

        # if (sys.argv[1] == simulation_command):
        #print("Simulation prediction...")
        #model_name = os.path.join(models_dir,"autoencoder_model.h5")
        #predictor = pr.Predictor(model_name)
        #logger = dl.DataLogger(os.path.join(current_script_path,"log.txt"))
        #evaluator = res.ResultsEvaluator()
        #sim_fetcher = sf.SimulationFetcher()
        #sim_fetcher.init_stream()
        #try:

        #    while True:
        #        source = image_manager.encode_sim_image(sim_fetcher.get_depth_img())
        #        mask = image_manager.resize_to_dataset(sim_fetcher.get_mask())
        #        predicted = predictor.predict(source)
        #        visualizer.display_video("predicted",image_manager.decode_predicted(predicted),1)
        #        fault = evaluator.compare_two_masks(mask,predicted)
        #        print(fault)
        #        logger.log_data(fault)
        #        visualizer.display_video("camera image",mask,1)
        #except KeyboardInterrupt:
        #    print("Program stopped by user")
        #finally:
        #    logger.save_data()
        #    sim_fetcher.close_stream()
        #    sys.exit(0)