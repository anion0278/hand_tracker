import numpy as np
import os
import cv2
import re
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import blob_recognizer as b

def binarization_norm(image):
    image = np.array(image) / 255.0
    image[image >= .5] = 1.
    image[image < .5] = 0.
    return image

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

    def __init__(self, config):
        self.img_dataset_size = config.img_dataset_size
        self.config = config

    def load_single_img(self, img_path):
        self.config.msg("Loading image from %s ..." % img_path)
        img_data = self.__load_resized(img_path)
        self.config.msg("Loaded: " + img_path)
        return img_data

    def prepare_image(self, image):
        return cv2.resize(image,self.img_dataset_size).astype(self.config.datatype)

    def __load_resized(self, img_path):
        # if img_path.endswith(".png"):
            # load from png 4-channel image
            # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:,:,3]
        # else:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return self.prepare_image(img)

    def save_image(self, image, img_path):
        cv2.imwrite(img_path, image.astype("uint8"))

    def show_image(self, image, title="Image", wait=True):
        cv2.imshow(title, image)
        return cv2.waitKey(-1 if wait else 1)

    def overlay_text_on_img(self, image, text, y_pos, color=(255,255,0)):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (10, y_pos), font, fontScale = 0.7, color = color, lineType = 2)

    def overlay_circle_on_img(self, image, pos, color=(255,0,0)):
        cv2.circle(image, center = (int(pos[0]),int(pos[1])), radius = 2,  color = color, thickness=2, lineType=8, shift=0) 

    def stack_images(self, img1, img2):
        return np.hstack((img1, img2))

    def img_mask_alongside(self, img, mask):
        center = b.find_blob(mask)
        if center is not None:
            self.overlay_circle_on_img(img, center) 
        return self.stack_images(img.astype("uint8"), mask)

    # TODO - move all dataset stuff to dataset_manager
    def get_autoencoder_datagens(self, jit=True):
        # jit -> load images in batches from folders, not all together
        if jit:
            return self.__get_jit_autoencoder_datagens(self.config.dataset_dir_path, self.config.imgs_dir, self.config.masks_dir, self.config.batch_size)
        else:
            raise ModuleNotFoundError("Not implemented yet!")

    def __get_jit_autoencoder_datagens(self, dataset_path, imgs_dir, masks_dir, batch_size):
        augmentations = dict(#featurewise_center=True,
                            #featurewise_std_normalization=True,
                            #dtype = datatype
                            rescale=1. / 255.0,
                            horizontal_flip = True,
                            vertical_flip = True)

        seed = 1
        image_generator = self.__get_datagen(augmentations, seed, dataset_path, imgs_dir, batch_size)

        #TODO add preprocessing_function = binarization_norm
        mask_generator = self.__get_datagen(augmentations, seed, dataset_path, masks_dir, batch_size)

        train_steps = np.floor(len(os.listdir(os.path.join(dataset_path, imgs_dir))) / batch_size)
        train_gen = (pair for pair in zip(image_generator, mask_generator))
        # TODO separate train from val:
        # https://github.com/keras-team/keras/issues/5862
        return train_gen, train_steps, None, 0

    def __get_datagen(self, augmentations, seed, dataset_dir, class_dir, batch_size):
        datagen = ImageDataGenerator(**augmentations,) 
        #preprocessing_function = binarization_norm
        generator = datagen.flow_from_directory(dataset_dir,
                                                        class_mode=None,
                                                        classes=[class_dir],
                                                        color_mode="grayscale",
                                                        target_size=reversed(self.img_dataset_size), # h, w !
                                                        seed=seed, batch_size = batch_size)
        generator.next()
        return generator

    # Regression model compatibility
    def load_single_dataset_img(self, img_path):
        img_path = os.path.join(self.main_script_path, img_relative_path)
        self.config.msg("Loading image from %s ..." % img_path)
        X_img_data = self.__load_resized(img_path)
        y_expected = self.parse_expected_value(img_relative_path)
        self.config.msg("Loaded: " + img_relative_path + " -> " + str(y_expected))
        return X_img_data, y_expected
    
    def clip_depth(self,img):
        depth_image = np.clip(img,self.xyz_ranges[2][0],self.xyz_ranges[2][1]) / self.xyz_ranges[2][1]
        depth_image_filtered = (255 - 255.0 * depth_image).astype('uint8')
        resized_img = cv2.resize(depth_image_filtered, self.dataset_img_size).astype('uint8')
        return resized_img
    

    def get_train_data(self):
        X_train = []
        y_train = []
        for train_img_path in self.__get_train_images_names_from_folder():
            X_img_data, y_expected = self.load_single_dataset_img(os.path.join(self.dataset_dir, train_img_path))
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
            self._FingerTipPos[i - 1] = float(regex_name_match.group(i))
        self._hand = int(regex_name_match.group(4))
        self._gest = int(regex_name_match.group(5))
        return self.__posNormalizedToWorkspace

    def __posNormalizedToWorkspace(self):
        result = []
        for i in range(0,3):  
            result[i] = (self._FingerTipPos[i] + abs(self.xyz_ranges[i][0])) / (abs(self.xyz_ranges[i][0]) + self.xyz_ranges[i][1])
        return result


       
    def decode_predicted(self,img):
        return 255 * img

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
    pass
    #test_instance = ImageDataManager("", "", "rgbd", (1,1), xyz_ranges)
    #result = test_instance.parse_expected_value("rgbd_1_X-100.2_Y-200.12_Z300.12_hand1_gest2_date02-12-2020_10#34#14")
    #expected_result = [-100.2, -200.12, 300.12, 1, 2]
    #assert result == expected_result

    #xyz_ranges = [(-700, 700), (-600, 500), (0, 1000)]
    #test_instance = ImageDataManager("", "", "rgbd", (1,1),  [(-700, 700), (-600, 500), (0, 1000)])
    #result = test_instance.parse_expected_value("depth_0_X376.3_Y341.0_Z415.6_hand1_gest1_date03-11-2020_16#02#08_x_536_y_42")
    #expected_result = [0.5, 0.5, 0.5, 1, 2]
    #assert result == expected_result