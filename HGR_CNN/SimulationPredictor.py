import CoppeliaAPI
import numpy as np
import cv2

class SimulationPredictor:
    def __init__(self,model, camera_img_size, dataset_img_size, depth_max,depth_min,x_min,x_max,y_min,y_max):
        self.copsim = CoppeliaAPI.CoppeliaAPI()
        self.copsim.initSimulation()
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.depth_min = depth_min
        self.camera_img_size = camera_img_size
        self.dataset_img_size = dataset_img_size
        self.depth_max_calibration = depth_max
        self.model = model
        self.x_range = abs(self.x_min)+x_max
        self.y_range = abs(self.y_min)+y_max
        self.depth_range = abs(self.depth_min) + self.depth_max_calibration

    def recognize_online(self):
        depth = self.copsim.GetImage()
        depth =  np.clip(depth, 0, self.depth_max_calibration) / self.depth_max_calibration
        depth = (255 - 255.0 * depth).astype('uint8')
        resized_img = cv2.resize(depth, self.dataset_img_size).astype(np.float32)
        resized_img = resized_img[..., np.newaxis]

        result = self.model.predict_single_image(resized_img, [0,0,0,0,0])
        #result = np.clip(result, 0, 1)
        result[0] = result[0]*self.x_range - self.x_min
        result[1] = result[1]*self.y_range - self.y_min
        result[2] = result[2]*self.depth_range - self.depth_min
        result[3] = np.clip(result[3], 0, 1)    #TODO
        result[4] = np.clip(result[4], 0, 3)

        result = np.round(result).astype("int")
        print("[X:%s; Y:%s; Z:%s; Hand:%s; Gesture:%s;]" % (result[0],result[1],result[2], result[3]== 1, result[4]))
        return result

    def predict_online(self):

        predicted_pos = self.recognize_online()

        pos = [predicted_pos[0]/1000,predicted_pos[1]/1000,predicted_pos[2]/1000]
        self.copsim.setObjectPos(copsim.hand,pos)

