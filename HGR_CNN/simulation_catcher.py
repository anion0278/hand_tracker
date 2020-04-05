import coppelia_wrapper as sim_wrapper
import numpy as np

class SimulationCatcher:
    """class for capturing image from CoppeliaSim simulation"""

    def __init__(self,config):
        self.__streaming = False

    def init_stream(self):
        self.__pipeline = sim_wrapper.CoppeliaAPI()
        self.__pipeline.init_simulation()
        self.__streaming = True

    def close_stream(self):
        self.__streaming = False
        self.__pipeline.stopSimulation()

    def __fetch_image(self):
        if self.__streaming:
            try:
                image = self.__pipeline.get_cam_image()
                return image

            except Exception as ex:
               print('Exception during streaming.. %' % ex)
               return None
        else:
            print("Streaming not initialized")
            return None
    
    def get_depth_img(self):
        dm = self.__fetch_image()
        return dm

    def __fetch_mask(self):
        if self.__streaming:
            try:
                image = self.__pipeline.GetMask()
                return image

            except Exception as ex:
               print('Exception during streaming.. %' % ex)
               return None
        else:
            print("Streaming not initialized")
            return None
   
    def get_mask(self):
        mask = self.__fetch_mask()
        return mask
