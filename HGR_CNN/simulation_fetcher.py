import CoppeliaAPI
import numpy as np

class SimulationFetcher:
    """class for capturing image from CoppeliaSim simulation"""

    def __init__(self):
        self.__streaming = False

    def init_stream(self):
        self.__pipeline = CoppeliaAPI.CoppeliaAPI()
        self.__pipeline.initSimulation()
        self.__streaming = True

    def close_stream(self):
        self.__streaming = False
        self.__pipeline.stopSimulation()

    def __fetch_image(self):
        if self.__streaming:
            try:
                image = self.__pipeline.GetImage()
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