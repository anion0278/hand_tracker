import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datatypes
import simple_recognizer
import sys
from datetime import datetime

current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(current_script_path, "dataset")

img_size = (640, 480)

class DatasetGenerator:
    def __init__(self, dataset_path, img_size):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.depth_max_calibration = float(1000) # millimeters
        self.img_counter = 0 

    def get_img_name(self, img_counter, tip_pos, is_hand_detected, gesture):
        if not is_hand_detected: 
            tip_pos = (0,0,0)
            gesture = datatypes.Gesture.UNDEFINED
        timestamp = datetime.now().strftime("%m-%d-%Y_%H#%M#%S")
        return f"rgbd_{img_counter}_X{tip_pos[0]}_Y{tip_pos[1]}_Z{tip_pos[2]}_hand{is_hand_detected}_gest{gesture.value}_date{timestamp}.png"

    def overlay_text_on_img(self, image, text, y_pos):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (10, y_pos), font, fontScale = 0.7, color = (255,255,0), lineType = 2)

    def create_rgbd_img(self, color_image, depth_image):
        depth_image_filtered = np.clip(depth_image, 0, self.depth_max_calibration) / self.depth_max_calibration
        depth_image_filtered = (255 - 255.0 * depth_image_filtered).astype('uint8') 
        return cv2.merge((*cv2.split(color_image), depth_image_filtered))

    def create_and_save_dataset_img(self, color_image, depth_image, current_gesture):
        full_data_img = self.create_rgbd_img(color_image, depth_image)
        index_tip_pos, is_hand_detected = simple_recognizer.recognize_finger_tip(color_image)
        img_name = self.get_img_name(self.img_counter, index_tip_pos, is_hand_detected, current_gesture)
        img_path = os.path.join(self.dataset_path, img_name)
        cv2.imwrite(img_path, full_data_img)
        self.img_counter += 1
        
    def record_data(self, current_gesture):
        window_name = "Dataset Recorder"
        pipeline = rs.pipeline()
        config = rs.config()
        image_rate = 30
        config.enable_stream(rs.stream.depth, img_size[0], img_size[1], rs.format.z16, image_rate)
        config.enable_stream(rs.stream.color, img_size[0], img_size[1], rs.format.bgr8, image_rate)

        pipeline.start(config)

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

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

                # Stack both images horizontally
                user_img = np.hstack((color_image, depth_colormap))
                self.overlay_text_on_img(user_img, "Depth median: %s" % np.median(depth_image), y_pos = 50)       
                self.overlay_text_on_img(user_img, "Recorder gesture: %s" % current_gesture.name, y_pos = 80)       
                self.overlay_text_on_img(user_img, "Press ESC to close...", y_pos = 450)       
                cv2.imshow('RealSense', user_img)

                self.create_and_save_dataset_img(color_image, depth_image, current_gesture)

                key = cv2.waitKey(1)
        finally:
            cv2.destroyWindow(window_name)
            pipeline.stop()


if __name__ == "__main__":

    #sys.argv = [sys.argv[0], "record"]
    print(sys.argv) 
    
    recorder = DatasetGenerator(dataset_path, img_size)
    recorder.record_data(datatypes.Gesture.POINTING)