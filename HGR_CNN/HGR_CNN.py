import os
import sys
import datatypes
import image_data_manager as idm
import dataset_generator as gen
import tensorboard_starter
import predictor as p
import model_wrapper as m
import config as c
import video_catcher as vc
import simulation_catcher as sc
import dataset_manager as dm

# ========================== SELECT MODEL HERE ========================== 
import predefined_seg_model as current_model
#import autoencoder_unet_model as current_model
#import autoencoder_simple_model as current_model

record_command = "record"
train_command = "train"
predict_command = "predict"
continue_train = "continue_train"
camera_command = "camera_prediction"
simulation_command = "simulation_prediction"
evaluate_command = "evaluate model"
tb_command = "tb"
show_command = "show"

config = c.Configuration(version_name = "seg_model_unet_34k_obst", debug_mode=False, latest_model_name="CD_270k.h5") # version_name helps with TensorBoard logs!
#config = c.Configuration(version_name = "autoencoder", debug_mode=False, latest_model_name="UR3_fullhand_300k.h5")


new_model_path = os.path.join(config.models_dir, "sm_unet-4ch.h5")

if __name__ == "__main__":
    #sys.argv = [sys.argv[0], tb_command]
    #sys.argv = [sys.argv[0], record_command]
    sys.argv = [sys.argv[0], train_command] 
    #sys.argv = [sys.argv[0], continue_train, "checkpoints/latest_checkpoint.h5"] 
    #sys.argv = [sys.argv[0], predict_command, os.path.join(c.current_dir_path, "testdata", "41.png")]
    #sys.argv = [sys.argv[0], camera_command]
    #sys.argv = [sys.argv[0], segmentation_models_command]
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
        config.learning_rate = config.learning_rate / 5
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
