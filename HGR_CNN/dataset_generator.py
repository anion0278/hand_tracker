import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datatypes
import simple_recognizer
import sys
from datetime import datetime


class DatasetGenerator:
    def __init__(self, record_when_no_hand, dataset_path, camera_img_size, dataset_img_size, depth_max):
        self.dataset_path = dataset_path
        self.camera_img_size = camera_img_size
        self.dataset_img_size = dataset_img_size
        self.depth_max_calibration = depth_max
        self.record_when_no_hand = record_when_no_hand
        self.img_counter = 0 

    def get_img_name(self, img_counter, tip_pos, is_hand_detected, gesture):
        if not is_hand_detected: 
            tip_pos = (0,0,0)
            gesture = datatypes.Gesture.UNDEFINED
        timestamp = datetime.now().strftime("%m-%d-%Y_%H#%M#%S")
        is_hand_detected_binary = int(is_hand_detected * 1)
        # TODO put counter of image to the end of name
        return "rgbd_{}_X{}_Y{}_Z{}_hand{}_gest{}_date{}.png".format(img_counter, tip_pos[0], tip_pos[1], tip_pos[2], is_hand_detected_binary, gesture.value, timestamp)

    def overlay_text_on_img(self, image, text, y_pos):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (10, y_pos), font, fontScale = 0.7, color = (255,255,0), lineType = 2)

    def overlay_circle_on_img(self, image, pos):
        cv2.circle(image, center = (pos[0], pos[1]), radius = 4,  color = (255,0,0), thickness=6, lineType=8, shift=0) 

    def create_rgbd_img(self, color_image, depth_image):
        depth_image_filtered = np.clip(depth_image, 0, self.depth_max_calibration) / self.depth_max_calibration
        depth_image_filtered = (255 - 255.0 * depth_image_filtered).astype('uint8') 
        return cv2.merge((*cv2.split(color_image), depth_image_filtered))

    def create_and_save_dataset_img(self, color_image, depth_image, current_gesture):
        full_data_img = self.create_rgbd_img(color_image, depth_image)
        resized_img = cv2.resize(full_data_img, self.dataset_img_size).astype(np.float32)
        index_tip_pos, is_hand_detected = simple_recognizer.recognize_finger_tip(color_image, depth_image)
        
        if self.record_when_no_hand or is_hand_detected:
            img_name = self.get_img_name(self.img_counter, index_tip_pos, is_hand_detected, current_gesture)
            img_path = os.path.join(self.dataset_path, img_name)
            cv2.imwrite(img_path, resized_img)
            self.img_counter += 1
        return index_tip_pos, is_hand_detected
        
    def record_data(self, current_gesture):
        window_name = "Dataset Recorder"
        pipeline = rs.pipeline()
        config = rs.config()
        image_rate = 30
        config.enable_stream(rs.stream.depth, self.camera_img_size[0], self.camera_img_size[1], rs.format.z16, image_rate)
        config.enable_stream(rs.stream.color, self.camera_img_size[0], self.camera_img_size[1], rs.format.bgr8, image_rate)

        profile = pipeline.start(config)

        color_sensor = profile.get_device().query_sensors()[1]
        color_sensor.set_option(rs.option.enable_auto_exposure, False)
        color_sensor.set_option(rs.option.enable_auto_white_balance, False)

        
        align_to = rs.stream.color
        align = rs.align(align_to)

        key = -1

        try:
            while key != 27:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames() # original frames
                aligned_frames = align.process(frames) # aligned frames
           
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame: # check correctness of the frames, or skip
                    continue 

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to
                # 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                user_img = np.hstack((color_image, depth_colormap))

                index_tip_pos, is_hand_detected = self.create_and_save_dataset_img(color_image, depth_image, current_gesture)
                if is_hand_detected: 
                    self.overlay_circle_on_img(user_img, index_tip_pos)

                self.overlay_text_on_img(user_img, "Tip position: %s" % str(index_tip_pos), y_pos = 50)       
                self.overlay_text_on_img(user_img, "Recorder gesture: %s" % current_gesture.name, y_pos = 80)       
                self.overlay_text_on_img(user_img, "Press ESC to close...", y_pos = 450)   
                cv2.imshow(window_name, user_img)

                key = cv2.waitKey(1)

        except KeyError:
            print("ESC pressed")

        finally:
            pipeline.stop()
            cv2.destroyWindow(window_name)
            cv2.destroyWindow("Mask")


#if __name__ == "__main__":

#    #sys.argv = [sys.argv[0], "record"]
#    print(sys.argv) 
    
#    recorder = DatasetGenerator(dataset_dir, img_size)
#    recorder.record_data(datatypes.Gesture.POINTING)
