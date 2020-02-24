import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datatypes
import simple_recognizer
import sys

# TODO REMOVE!!!!
current_script_path = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(current_script_path, "dataset")

class OnlinePredictor:
    def __init__(self, model, camera_img_size, dataset_img_size, depth_max):
        self.camera_img_size = camera_img_size
        self.dataset_img_size = dataset_img_size
        self.depth_max_calibration = depth_max
        self.model = model

    def overlay_text_on_img(self, image, text, y_pos):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, text, (10, y_pos), font, fontScale = 0.7, color = (255,255,0), lineType = 2)

    def overlay_circle_on_img(self, image, pos, color = (255,0,0)):
        cv2.circle(image, center = (pos[0], pos[1]), radius = 4,  color = color, thickness=6, lineType=8, shift=0) 

    # TODO extract common methods
    def create_rgbd_img(self, color_image, depth_image):
        depth_image_filtered = np.clip(depth_image, 0, self.depth_max_calibration) / self.depth_max_calibration
        depth_image_filtered = (255 - 255.0 * depth_image_filtered).astype('uint8') 
        return cv2.merge((*cv2.split(color_image), depth_image_filtered))
        
    def recognize_online(self, color_image, depth_image):
        depth_channel = self.create_rgbd_img(color_image, depth_image)[:,:,3]
        resized_img = cv2.resize(depth_channel, self.dataset_img_size).astype(np.float32)
        resized_img = resized_img[..., np.newaxis]
        index_tip_pos, is_hand_detected = simple_recognizer.recognize_finger_tip(color_image, depth_image)
        
        result = self.model.predict_single_image(resized_img, [0,0,0,0,0])
        #result = np.clip(result, 0, 1)
        result[0] *= self.camera_img_size[0]
        result[1] *= self.camera_img_size[1]
        result[2] *= self.depth_max_calibration
        result = np.round(result).astype("int")
        print("[X:%s; Y:%s; Z:%s; Hand:%s; Gesture:%s;]" % (result[0],result[1],result[2], result[3]== 1, result[4]))
        return result

    def predict_online(self):
        window_name = "Online predictor"
        pipeline = rs.pipeline()
        config = rs.config()
        image_rate = 30
        config.enable_stream(rs.stream.depth, self.camera_img_size[0], self.camera_img_size[1], rs.format.z16, image_rate)
        config.enable_stream(rs.stream.color, self.camera_img_size[0], self.camera_img_size[1], rs.format.bgr8, image_rate)

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

                # Apply colormap on depth image (image must be converted to
                # 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                # Stack both images horizontally
                user_img = np.hstack((color_image, depth_colormap))

                result = self.recognize_online(color_image, depth_image)

                if is_hand_detected: 
                    self.overlay_circle_on_img(user_img, index_tip_pos)

                if result[3]== 1: 
                    self.overlay_circle_on_img(user_img, result[0:3], color = (0,255,255))

                self.overlay_text_on_img(user_img, "Tip position (Ground truth): %s" % str(index_tip_pos), y_pos = 50)       
                self.overlay_text_on_img(user_img, "Predicted tip position: %s" % str(result[0:3]) , y_pos = 80)       
                
                gesture_name = datatypes.Gesture(result[4]).name
                self.overlay_text_on_img(user_img, "Predicted gesture (+status): %s (%s)" % (gesture_name, result[3]== 1), y_pos = 110)       
                self.overlay_text_on_img(user_img, "Press ESC to close...", y_pos = 450)   
                cv2.imshow(window_name, user_img)

                key = cv2.waitKey(1)

        except KeyError:
            print("ESC pressed")

        finally:
            pipeline.stop()
            cv2.destroyWindow(window_name)
            cv2.destroyWindow("Mask")
