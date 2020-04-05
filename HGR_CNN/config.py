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
        self.batch_size = 50
        self.epochs_count = 20
        self.test_data_ratio = 0.2
        self.learning_rate = 1.1
        self.img_camera_size = (640, 480)
        self.camera_rate = 30
        self.img_dataset_size = (320, 240)
        self.xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
        self.record_when_no_hand = False
        self.use_gpu = False
        self.imgs_dir = "dataset-aug"
        self.masks_dir = "mask-aug"

        if self.DEBUG:
            self.imgs_dir = "dataset-test"
            self.masks_dir = "mask-test"  

        self.dataset_dir_path = os.path.join(current_dir_path, os.pardir, "dataset") # pardir - parent dir (one lvl up)
        self.models_dir = os.path.join(current_dir_path, "models") 
        self.logs_dir = os.path.join(current_dir_path, "logs", self.version_name+f"_LR{self.learning_rate}_{date_time_str}")
        self.latest_model_path = os.path.join(self.models_dir, latest_model_name)
        self.status_print()

    def status_print(self):
        msg(f"Version: {self.version_name}") 
        msg(f"Debug: {self.DEBUG}") 

    def msg(self, message):
        msg(message = message)