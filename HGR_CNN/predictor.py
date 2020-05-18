import results_evaluator as re
import blob_recognizer as b
import data_logger as dl
import cv2

class Predictor:
    predicted_pos = (0,0)
    real_pos = (0,0)
    def __init__(self, model, config, image_manager,catcher):
        self.catcher = catcher
        self.catcher.init_stream()
        self.image_manager = image_manager
        self.config = config #not used
        self.model = model
        self.evaluator = re.ResultsEvaluator()

    def predict(self):
        try:
            image,sim_mask = self.catcher.get_data()
            depth = self.image_manager.prepare_image(image)
            mask = self.image_manager.prepare_image(sim_mask)
            self.real_pos = b.find_blob(mask)
            predicted_mask = self.model.predict_single(depth)
            self.predicted_pos = b.find_blob(predicted_mask)
            combined = self.image_manager.img_mask_alongside(depth, predicted_mask)
            self.image_manager.show_image(combined, wait = False) 

            if self.config.benchmark:
                if self.predicted_pos is not None:
                    pos = self.catcher.get_fingertip_pos()
                    if pos is not None:
                        return pos
            
        except KeyboardInterrupt:
            raise KeyboardInterrupt

        except:
            print("Prediction failed")
            return None
            

    def predict_online(self):
        try:
            logger = dl.DataLogger(self.config.benchmark_file)
            data = str("X,Y,Z,R1,R2,P1,P2,Fault")
            logger.log_data(data)
            
            while True:
               pos = self.predict()
               
               if pos is not None:
                   fault = self.evaluator.get_fault_pix(self.real_pos,self.predicted_pos)
                   data = str("{},{},{},{},{},{},{},{}".format(pos[0],pos[1],pos[2],self.real_pos[0],self.real_pos[1],self.predicted_pos[0],self.predicted_pos[1],fault))
                   logger.log_data(data)
              
               
        except KeyboardInterrupt: # CTRL + C
            self.config.msg("Closing...")
            logger.save_data()
