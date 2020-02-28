import numpy as np
import os
import cv2
import re
import datatypes
from datetime import datetime

def get_img_name(img_counter, tip_pos, is_hand_detected, gesture):
    if not is_hand_detected: 
        tip_pos = (0,0,0)
        gesture = datatypes.Gesture.UNDEFINED
    timestamp = datetime.now().strftime("%m-%d-%Y_%H#%M#%S")
    is_hand_detected_binary = int(is_hand_detected * 1)
    # TODO put counter of image to the end of name
    return "rgbd_{}_X{}_Y{}_Z{}_hand{}_gest{}_date{}.png".format(img_counter, tip_pos[0], tip_pos[1], tip_pos[2], is_hand_detected_binary, gesture.value, timestamp)

class ImageDataManager:
    def __init__(self, main_script_path, dataset_dir, image_state_base, image_target_size, image_camera_size, depth_max):
        self.image_state_base = image_state_base
        self.main_script_path = main_script_path
        self.dataset_dir = dataset_dir
        self.image_target_size = image_target_size
        self.image_camera_size = image_camera_size
        self.depth_max = depth_max

    def load_single_img(self, img_relative_path):
        img_path = os.path.join(self.main_script_path, img_relative_path)
        print("Loading image from %s ..." % img_path)
        X_img_data = self.__load_resized(img_path)
        y_expected = self.parse_expected_value(img_relative_path)
        print("Loaded: " + img_relative_path + " -> " + str(y_expected))
        return X_img_data, y_expected

    def get_train_data(self):
        X_train = []
        y_train = []
        for train_img_path in self.__get_train_images_names_from_folder():
            X_img_data, y_expected = self.load_single_img(os.path.join(self.dataset_dir, train_img_path))
            X_train.append(X_img_data)
            y_train.append(y_expected)
        return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    def get_encoder_data(self):
        X_train = []
        y_train = []
        for train_img_path in self.__get_train_images_names_from_folder():
            X_img_data, y_expected = self.load_image_pair(os.path.join(self.dataset_dir, train_img_path))
            X_train.append(X_img_data)
            y_train.append(y_expected)
        return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    def __get_train_images_names_from_folder(self):
        return list(filter(lambda x: x.startswith(self.image_state_base) and len(x) > 15, os.listdir(self.dataset_dir)))

    def parse_expected_value(self, img_name):
        result = []
        # TODO counter of image should be in the end of name
        regex_name_match = re.search('.*' + self.image_state_base + '_\d+_X(\d+)_Y(\d+)_Z(\d+)_hand(.+)_gest(\d+)_date', img_name)
        for fingerIndex in range(0,5):
            y_value = int(regex_name_match.group(fingerIndex + 1))
            result.append(y_value)
        result[0] /= self.image_camera_size[0]
        result[1] /= self.image_camera_size[1]
        result[2] /= self.depth_max
        return result

    def load_image_pair(self, img_path):
        depth_image = self.__load_resized(img_path)
        params = self.parse_expected_value(os.path.basename(img_path))
        mask_image =  np.zeros((self.image_target_size[1], self.image_target_size[0],1), np.float32)
        params[0] *= self.image_target_size[0]
        params[1] *= self.image_target_size[1]
        if int(params[3]) is 1:
            cv2.circle(mask_image, center = (int(params[0]), int(params[1])), radius = 1,  color = 255, thickness=3, lineType=8, shift=0) 
        #cv2.imwrite("mask.png", mask_image)
        #cv2.imwrite("orig.png", depth_image)
        print("Loaded: " + os.path.basename(img_path))
        return depth_image, mask_image

    def __load_resized(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, self.image_target_size)[:,:,3].astype(np.float32)  # TODO check if cast is required
        return resized[..., np.newaxis]


# TODO unit test
if __name__ == "__main__":
    test_instance = ImageDataManager("", "", "rgbd", (3,3))

    result = test_instance.parse_expected_value("rgbd_1_X100_Y200_Z300_hand1_gest2_date02-12-2020_10#34#14")
    expected_result = [100, 200, 300, 1, 2]
    assert result == expected_result