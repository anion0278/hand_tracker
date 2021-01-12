import cv2
import numpy as np

class BlobRecognizer():
    
    min_blob_size = 5
    max_blob_size = 2500
    sensitivity = 20

    def __init__(self,color):
        self.color = color

    def __binarize(self,image):
        
        #green tape
        upper = (80+sensitivity,255,255)  
        lower = (80-sensitivity,70,50)
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        return mask


    def __find_blob_binary(self,image):
        try: # this can be handled by utils method: cnts = imutils.grab_contours(cnts), see: https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
            _, contours,_ = cv2.findContours(image.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        except:
            contours,_ = cv2.findContours(image.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            if not contours:
                print('Contour not detected')
                return (0,0)

            blob = max(contours, key=lambda el: cv2.contourArea(el))
        
            blobSize = cv2.contourArea(blob)        
            #print(blobSize)

            if (blobSize < self.min_blob_size)or(blobSize > self.max_blob_size):
                return (0,0)

            M = cv2.moments(blob)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            return (center[0],center[1])
    
        except Exception as ex:
            print('No blob detected')
            return (0,0)

    def get_blob_pos(self,image):
        return self.__find_blob_binary(image)

    #def recognize_finger_tip(color_image, depth_image):
    #    z_heights = []
    #    for point in blob:
    #        z_heights.append(depth_image[point[0][1], point[0][0]])
    #        z = int(np.median(z_heights))
    #    return z

       
    
