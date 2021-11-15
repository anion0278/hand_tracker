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
        self.iteration = 0

    def predict(self):
        try:
            image,sim_mask = self.catcher.get_data()
            depth = self.image_manager.prepare_image(sim_mask)
            mask = self.image_manager.prepare_image(sim_mask)
            rgb = self.image_manager.prepare_image(image)
 
            self.real_pos = b.find_blob(mask)
            predicted_mask = self.model.predict_single(depth)

            pos_hand = b.find_hand(self.config.img_dataset_size,predicted_mask)
            img_name = self.image_manager.get_iteration_name(self.iteration,pos_hand[2],self.config.camera_depth_path)
            self.image_manager.save_image(depth,img_name)

            
            img_name = self.image_manager.get_iteration_name(self.iteration,pos_hand[2],self.config.camera_RGB_path)
            self.image_manager.save_image(rgb,img_name)

            #img_name = self.image_manager.get_iteration_name(self.iteration,pos_hand[2],self.config.camera_predicted_path)
            #self.image_manager.save_image(predicted_mask,img_name)
            
            #self.predicted_pos = b.find_blob(predicted_mask)
            detected,x,y,w = b.find_hand(self.config.img_dataset_size,predicted_mask)
            combined = self.image_manager.img_mask_alongside(depth,detected)
            self.image_manager.show_image(combined, wait = False) 

            #img_name = self.image_manager.get_iteration_name(self.iteration,(x,y,w),self.config.camera_RGB_dir)
            #self.image_manager.save_image(rgb,img_name)

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
               self.iteration +=1
               
        except KeyboardInterrupt: # CTRL + C
            self.config.msg("Closing...")
            logger.save_data()
