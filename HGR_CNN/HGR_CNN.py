import cv2
import pyrealsense2 as rs
import numpy as np

#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

try:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
 
    align_to = rs.stream.color
    align = rs.align(align_to)
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth = aligned_frames.get_depth_frame()
        color = aligned_frames.get_color_frame()
        if not depth or not color: continue

        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)        

        #ARUCO
        #corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,dictionary)
        #color_image	= cv2.aruco.drawDetectedMarkers(color_image,corners)
        
        #circle
        gray = cv2.medianBlur(gray,5)
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1 = 100, param2 = 30, minRadius = 0, maxRadius = 0)

        detected_circles = np.uint16(np.around(circles))
        for(x,y,r) in detected_circles[0,:]:
            cv2.circle(color_image,(x,y),r,(0,255,0),3)

    
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)       
finally:

    # Stop streaming
    pipeline.stop()
