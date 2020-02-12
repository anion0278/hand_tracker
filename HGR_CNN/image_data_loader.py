import numpy as np
import os
import cv2
import re

class ImageDataLoader:
    def __init__(self, main_script_path, dataset_dir, image_state_base, image_target_size):
        self.image_state_base = image_state_base
        self.main_script_path = main_script_path
        self.dataset_dir = dataset_dir
        self.image_target_size = image_target_size

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

    def __get_train_images_names_from_folder(self):
        return list(filter(lambda x: x.startswith(self.image_state_base) and len(x) > 15, os.listdir(self.dataset_dir)))

    def parse_expected_value(self, img_name):
        result = []
        # rgbd_1_X100_Y200_Z300_handTrue_gest1_date02-12-2020_10#34#14
        regex_name_match = re.search('.*' + self.image_state_base + '_\d+_X(\d+)_Y(\d+)_Z(\d+)_hand(.+)_gest(\d+)_date', img_name)
        for fingerIndex in range(0,5):
            y_value = int(regex_name_match.group(fingerIndex + 1))
            result.append(y_value)
        return result

    def __load_resized(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, self.image_target_size).astype(np.float32)
        return resized #[..., np.newaxis] # add new dimension


# TODO unit test
if __name__ == "__main__":
    test_instance = ImageDataLoader("", "", "rgbd", (3,3))

    result = test_instance.parse_expected_value("rgbd_1_X100_Y200_Z300_hand1_gest2_date02-12-2020_10#34#14")
    expected_result = [100, 200, 300, 1, 2]
    assert result == expected_result