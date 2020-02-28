import cv2
import numpy as np

min_blob_size = 60
max_blob_size = 2500

def recognize_finger_tip(color_image, depth_image):
    
    #green tape
    sensitivity = 20
    upper = (80+sensitivity,255,255)  
    lower = (80-sensitivity,70,50)
    
    hsv = cv2.cvtColor(color_image,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    #cv2.imshow('Mask', mask)
    #key = cv2.waitKey(1)
    #if key == 27: 
    #    raise KeyError

    try:
        _, contours, hierarchy_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        if not contours:
            print('Contour not detected')
            return (0,0,0), False

        blob = max(contours, key=lambda el: cv2.contourArea(el))
        
        blobSize = cv2.contourArea(blob)        
        print(blobSize)

        if (blobSize < min_blob_size)or(blobSize > max_blob_size):
            return (0,0,0), False

        M = cv2.moments(blob)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        z_heights = []
        for point in blob:
            z_heights.append(depth_image[point[0][1], point[0][0]])
        z = int(np.median(z_heights))

        return (center[0],center[1],z), True
    
    except Exception as ex:
        print('No blob detected')
        return (0,0,0), False
    