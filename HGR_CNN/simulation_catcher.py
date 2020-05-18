import coppelia_wrapper as sim_wrapper
import numpy as np

class SimulationCatcher:
    """class for capturing image from CoppeliaSim simulation"""
    record_flag = False
    fingertip_pos = [0,0,0]
    hand_pos = [0,0,0]

    def __init__(self,config):
        self.__streaming = False
        self.config = config

    def init_stream(self):
        self.__pipeline = sim_wrapper.CoppeliaAPI()
        self.__pipeline.init_simulation()
        self.__streaming = True

    def close_stream(self):
        self.__streaming = False
        self.__pipeline.stopSimulation()

    def __fetch_data(self):
        if self.__streaming:
            try:
                self.__pipeline.move_sim()
                image = self.__pipeline.get_cam_image()
                mask = self.__pipeline.get_mask()
                self.fingertip_pos = self.__pipeline.get_fingertip_pos()
                self.hand_pos = self.__pipeline.get_hand_pos()
                return image,mask

            except Exception as ex:
               print('Exception during streaming.. %' % ex)
               return None
        else:
            print("Streaming not initialized")
            return None
    
    def get_depth_img(self):
        dm,_ = self.__fetch_data()
        return dm

    def get_mask(self):
        _,mask = self.__fetch_data()
        return mask
    def get_data(self):
        img,mask = self.__fetch_data()
        return img,mask

    def get_fingertip_pos(self):
        if self.__streaming:
            return self.fingertip_pos
