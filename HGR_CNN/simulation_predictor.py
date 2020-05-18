import coppelia_wrapper as sim_wrapper
import numpy as np
import cv2


class SimulationPredictor:
    def __init__(self, model, config, image_manager):
        self.copsim = sim_wrapper.CoppeliaAPI()
        self.copsim.init_simulation()
        self.image_manager = image_manager
        self.config = config
        self.model = model

    def recognize(self, image):
        depth = self.image_manager.prepare_image(image)
        mask = self.model.predict_single(depth)
        combined = self.image_manager.img_mask_alongside(depth, mask)
        self.image_manager.show_image(combined, wait = False)

    def predict_online(self):
        try:
            while True:
                current_image = self.copsim.get_cam_image()
                predicted_pos = self.recognize(current_image)
        except KeyboardInterrupt: # CTRL + C
            self.config.msg("Closing...")