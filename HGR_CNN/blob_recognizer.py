import cv2

def find_blob(image):
    center = None
    try:
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        contours,_ = cv2.findContours(image.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return center
    largest_blob = max(contours, key=lambda element: cv2.contourArea(element))
    M = cv2.moments(largest_blob)
    if M["m00"] != 0:
        center = (M["m10"] / M["m00"], M["m01"] / M["m00"])
    return center

def find_hand(imdim,image):
    try:
        _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    except:
        contours,_ = cv2.findContours(image.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_blob = max(contours, key=lambda element: cv2.contourArea(element))
    (x,y),radius = cv2.minEnclosingCircle(largest_blob)
    
    if (x + radius)>imdim[0]:
        pos_x = imdim[0]-2*radius
    elif (x - radius)<0:
        pos_x = 0
    else: 
        pos_x = x - radius

    if (y + radius)>imdim[1]:
        pos_y = imdim[1]-2*radius
    elif (y - radius)<0:
        pos_y = 0
    else: 
        pos_y = y - radius

    return "{}_{}_{}".format(pos_x,pos_y,2*radius)