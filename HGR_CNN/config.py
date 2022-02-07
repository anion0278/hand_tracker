import os
import sys
from datetime import datetime

current_dir_path = os.path.dirname(os.path.realpath(__file__))
date_time_str = datetime.now().strftime("%d-%b-%Y-%H-%M-%S")

msg_format = "[USER] {}"

def msg(message):
    print(msg_format.format(message))

class Configuration():
    def __init__(self, version_name, debug_mode, latest_model_name):
        self.version_name = version_name
        self.DEBUG = debug_mode
        self.datatype = "float32"
        self.filters_count = 16
        self.batch_size = 40
        self.epochs_count = 8
        self.test_data_ratio = 0.2
        self.learning_rate = 1.0
        self.img_camera_size = (424, 240)
        self.camera_rate = 30
        self.img_dataset_size = (320, 240)
        self.xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
        self.record_when_no_hand = False
        self.use_gpu = True
        #self.imgs_dir = "dataset-orig"
        #self.masks_dir = "mask-orig"
        self.imgs_dir = "img"
        self.masks_dir = "mask2"
        self.val_imgs_dir = "img-valid"
        self.val_masks_dir = "mask2-valid"

        self.benchmark = False
        self.benchmark_start = (-0.05,0.04,0.738)
        self.benchmark_file = os.path.join(current_dir_path,"logs","benchmark.csv")

        if self.DEBUG:
            self.imgs_dir = "dataset-test"
            self.masks_dir = "mask-test"  

        self.dataset_dir_path = os.path.join(current_dir_path, os.pardir, "dataset") # pardir - parent dir (one lvl up)
        self.models_dir = os.path.join(current_dir_path, "models") 
        self.logs_dir = os.path.join(current_dir_path, "logs", self.version_name+f"_LR{self.learning_rate}_{date_time_str}")
        self.latest_model_path = os.path.join(self.models_dir, latest_model_name)

        self.camera_image_dir = os.path.join(current_dir_path, os.pardir, "camera_image")
        self.camera_RGB_dir = "RGB"
        self.camera_depth_dir = "Depth"
        self.camera_label_dir = "Labeled"
        self.camera_predicted_dir = "Predicted"
        self.camera_RGB_path = os.path.join(self.camera_image_dir,self.camera_RGB_dir)
        self.camera_depth_path = os.path.join(self.camera_image_dir,self.camera_depth_dir)
        self.camera_predicted_path = os.path.join(self.camera_image_dir,self.camera_predicted_dir)
        self.status_print()

    def status_print(self):
        msg(f"Version: {self.version_name}") 
        msg(f"Debug: {self.DEBUG}") 

    def msg(self, message):
        msg(message = message)