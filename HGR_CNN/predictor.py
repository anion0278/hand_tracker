class Predictor:
    def __init__(self, model, config, image_manager,catcher):
        self.catcher = catcher
        self.catcher.init_stream()
        self.image_manager = image_manager
        self.config = config
        self.model = model

    def predict(self):
        try:
            image = self.catcher.get_depth_img()
            depth = self.image_manager.prepare_image(image)
            mask = self.model.predict_single(depth)
            combined = self.image_manager.img_mask_alongside(depth, mask)
            self.image_manager.show_image(combined, wait = False) 
        except:
            print("Prediction failed")

    def predict_online(self):
        try:
            while True:
                self.predict()
        except KeyboardInterrupt: # CTRL + C
            self.config.msg("Closing...")
