import numpy as np
import os
import cv2
import re


def get_img_name(img_counter, tip_pos, is_hand_detected, gesture):
    if not is_hand_detected: 
        tip_pos = (0,0,0)
        gesture = datatypes.Gesture.UNDEFINED
    timestamp = datetime.now().strftime("%m-%d-%Y_%H#%M#%S")
    is_hand_detected_binary = int(is_hand_detected * 1)
    # TODO put counter of image to the end of name
    return "rgbd_{}_X{}_Y{}_Z{}_hand{}_gest{}_date{}.png".format(img_counter, tip_pos[0], tip_pos[1], tip_pos[2], is_hand_detected_binary, gesture.value, timestamp)

class ImageDataManager:
    _FingerTipPos = [0,0,0]
    _FingerTipPix = [0,0]
    _hand = 0
    _gest = 0

    def __init__(self, main_script_path, dataset_dir, image_state_base, image_target_size, xyz_ranges):
        self.image_state_base = image_state_base
        self.main_script_path = main_script_path
        self.dataset_dir = dataset_dir
        self.dataset_img_size = image_target_size
        self.xyz_ranges = xyz_ranges

    def load_single_img(self, img_relative_path):
        img_path = os.path.join(self.main_script_path, img_relative_path)
        print("Loading image from %s ..." % img_path)
        X_img_data = self.__load_resized(img_path)
        y_expected = self.parse_expected_value(img_relative_path)
        print("Loaded: " + img_relative_path + " -> " + str(y_expected))
        return X_img_data, y_expected
    
    def clip_depth(self,img):
        depth_image = np.clip(img,self.xyz_ranges[2][0],self.xyz_ranges[2][1])/self.xyz_ranges[2][1]
        depth_image_filtered = (255 - 255.0 * depth_image).astype('uint8')
        resized_img = cv2.resize(depth_image_filtered, self.dataset_img_size).astype('uint8')
        return resized_img
    

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
        #result = []
        # TODO counter of image should be in the end of name
        float_val_group = "(-?\d+(?:.\d+)?)"
        regex_name_match = re.search(f'_X{float_val_group}_Y{float_val_group}_Z{float_val_group}_hand(\d)_gest(\d)', img_name)
      
        for i in range(1,4):
            self._FingerTipPos[i-1] = float(regex_name_match.group(i))
        self._hand = int(regex_name_match.group(4))
        self._gest = int(regex_name_match.group(5))
        return self.__posNormalizedToWorkspace

    def parse_expected_mask(self, img_name):
        regex_name_match = re.search(f'_x_(\d+)_y_(\d+)', img_name)
        for i in range(1,3):
            self._FingerTipPix[i-1] = int(regex_name_match.group(i))
        return self._FingerTipPix
            #result.append(y_value)

    def __posNormalizedToWorkspace(self):
        result = []
        for i in range(0,3):  
            result[i] =  (self._FingerTipPos[i] + abs(self.xyz_ranges[i][0]))  / (abs(self.xyz_ranges[i][0]) +  self.xyz_ranges[i][1])
        return result

    def __load_resized(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # TODO check if cast is required
        img = cv2.resize(img,self.dataset_img_size).astype(np.float32) # TODO put all type transformations into single place
        #TODO allow to define lower size of float values -> for instance, float16
        #cv2.imwrite("test-rot.png", img) 
        return img[..., np.newaxis]

    def get_encoder_data(self,img_name):
        X_train = []
        y_train = []
        all_image_names = self.__get_train_images_names_from_folder()
        for img_name in all_image_names:
            if re.match(f"{self.image_state_base}_\d+_X.+_date.+x_\d+_y_\d+", img_name):
                continue
            X_img_data, y_expected = self.load_image_pair(img_name, all_image_names)
            X_train.append(X_img_data)
            y_train.append(y_expected)
        return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)

    def load_image_pair(self, label_img_name):#, all_img_names):
        img_name = re.split("_x_",label_img_name)[0]+".jpg"
        
        depth_image = self.__load_resized(os.path.join(self.dataset_dir,"img", img_name))
        #regex_basename_match = re.search("(" + self.image_state_base + "_\d+_X.+_date.+#\d+)(?:n\d+)?\.", img_name)
        #r = regex_basename_match.group(1)
        #label_img_name = list(filter(lambda x: (r+"_x_") in x, all_img_names))[0]
        mask_image = self.__load_resized(os.path.join(self.dataset_dir,"mask",label_img_name))
        #print("Loaded: " + img_name)
        #cv2.imwrite("mask.png", mask_image)
        #cv2.imwrite("orig.png", depth_image)
        return depth_image, mask_image
   
    def decode_predicted(self,img):
        return 255*img

    def encode_camera_image(self,source):
        image = self.clip_depth(source)
        image_aug = image[...,np.newaxis]
        image_norm = [image_aug.astype("float32") / 255.0]
        image_out = np.array(image_norm)
        return image_out

    def encode_sim_image(self,source):
        image = self.resize_to_dataset(source)
        image_aug = image[...,np.newaxis]
        image_norm = [image_aug.astype("float32") / 255.0]
        image_out = np.array(image_norm)
        return image_out

    def resize_to_dataset(self,img):
        image_out = cv2.resize(img, self.dataset_img_size)
        return image_out



# TODO unit test
if __name__ == "__main__":
    #test_instance = ImageDataManager("", "", "rgbd", (1,1), xyz_ranges)
    #result = test_instance.parse_expected_value("rgbd_1_X-100.2_Y-200.12_Z300.12_hand1_gest2_date02-12-2020_10#34#14")
    #expected_result = [-100.2, -200.12, 300.12, 1, 2]
    #assert result == expected_result

    xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
    test_instance = ImageDataManager("", "", "rgbd", (1,1),  [(-700, 700), (-600, 500), (0, 1000)])
    result = test_instance.parse_expected_value("depth_0_X376.3_Y341.0_Z415.6_hand1_gest1_date03-11-2020_16#02#08_x_536_y_42")
    expected_result = [0.5, 0.5, 0.5, 1, 2]
    assert result == expected_result