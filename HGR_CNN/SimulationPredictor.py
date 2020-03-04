import CoppeliaAPI
import numpy as np
import cv2

class SimulationPredictor:
    def __init__(self,model, camera_img_size, dataset_img_size, depth_max,depth_min,x_min,x_max,y_min,y_max, xyz_ranges):
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
        self.xyz_ranges = xyz_ranges

    def recognize_online(self):
        depth = self.copsim.GetImage()
        cv2.imwrite(str(int(1)) + ".jpg", depth)
        #depth =  np.clip(depth, 0, self.depth_max_calibration) / self.depth_max_calibration
        #cv2.imwrite(str(int(2)) + ".jpg", depth)
        #depth = (255 - 255.0 * depth).astype('uint8')
        #cv2.imwrite(str(int(3)) + ".jpg", depth)
        img = cv2.rotate(depth, 0)
        resized_img = cv2.resize(img, self.dataset_img_size).astype(np.float32)
        cv2.imwrite(str(int(4)) + ".jpg", resized_img)
        resized_img = resized_img[..., np.newaxis]
        cv2.imwrite(str(int(5)) + ".jpg", resized_img)
        result = self.model.predict_single_image(resized_img, [0,0,0,0,0])

        #result = np.clip(result, 0, 1)
        for i in range(0,3):  
            result[i] = result[i] *  (abs(self.xyz_ranges[i][0]) +  self.xyz_ranges[i][1]) - abs(self.xyz_ranges[i][0])
        #result[0] = result[0]*self.x_range - self.x_min
        #result[1] = result[1]*self.y_range - self.y_min
        #result[2] = result[2]*self.depth_range - self.depth_min
        result[3] = np.clip(result[3], 0, 1)    #TODO
        result[4] = np.clip(result[4], 0, 3)
        
        
        result = np.round(result).astype("int")
        print("[X:%s; Y:%s; Z:%s; Hand:%s; Gesture:%s;]" % (result[0],result[1],result[2], result[3]== 1, result[4]))
        return result

    def predict_online(self):
        
        key = -1

        try:
            while key != 27:
                predicted_pos = self.recognize_online()

                pos = [predicted_pos[0]/1000,predicted_pos[1]/1000,predicted_pos[2]/1000]
                self.copsim.SetObjectPos(self.copsim.sphere,pos)
        except KeyError:
            print("ESC pressed")

