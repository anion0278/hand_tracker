import cv2
import numpy as np

min_blob_size = 50

def recognize_finger_tip(color_image, depth_image):
    
    #green tape
    upper = (140,150,30)  
    lower = (70,74,0)

    mask = cv2.inRange(color_image, lower, upper)
    cv2.imshow('Mask', mask)
    key = cv2.waitKey(1)
    if key == 27: 
        raise KeyError

    try:
        _, contours, hierarchy_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except:
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        if not contours:
            print('Contour not detected')
            return (0,0,0), False

        else:
            blob = max(contours, key=lambda el: cv2.contourArea(el))
                
            #minRect = cv2.minAreaRect(blob) # TODO check block coverage area
            if len(blob) < min_blob_size:
                return (0,0,0), False

            M = cv2.moments(blob)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            z_heights = []
            for point in blob:
                point_a = point[0]
                z_heights.append(depth_image[point_a[0], point_a[1]])
            z = int(np.round(np.mean(z_heights),0))

            return (center[0],center[1],z), True
    
    except Exception as ex:
        print('No blob detected')
        return (0,0,0), False
    