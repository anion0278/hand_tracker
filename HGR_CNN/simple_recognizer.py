def recognize_finger_tip(color_image, depth_image):
    
    #green tape
    upper = (140,150,30)
    lower = (70,74,0)

    mask = cv2.inRange(color_image, lower, upper)
    #_,contours, hierarchy_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        if not contours:
            print('Contour not detected')
        else:
            blob = max(contours, key=lambda el: cv2.contourArea(el))
            M = cv2.moments(blob)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            
            minRect = cv2.minAreaRect(blob)
            minRectPoints = cv2.boxPoints(minRect)
            depths = depth_image[minRectPoints[0],minRectPoints[1],minRectPoints[2],minRectPoints[3]]

            Z = np.median(depths)
            return (center[0],center[1],Z), True
    
    finally:
        print('No blob detected')
        return False
    