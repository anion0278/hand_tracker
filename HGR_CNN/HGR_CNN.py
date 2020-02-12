import cv2
import pyrealsense2 as rs
import numpy as np

img_size = (640,480)
img_rate = 30

upper = (140,150,30)  #100,130,50
lower = (70,74,0) #200,200,130

#dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

def __init__(self, img_rate, img_size):
    self.img_size = img_size
    self.img_rate = img_rate

def camera_matrix(intrinsics):
    return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                     [            0, intrinsics.fy, intrinsics.ppy],
                     [            0,             0,              1]])

try:
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, img_size[0], img_size[1], rs.format.z16, img_rate)
    config.enable_stream(rs.stream.color, img_size[0], img_size[1], rs.format.bgr8, img_rate)

    pipeline.start(config)
 
    align_to = rs.stream.color
    align = rs.align(align_to)
    while True:
        
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('mask', cv2.WINDOW_AUTOSIZE)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth = aligned_frames.get_depth_frame()
        color = aligned_frames.get_color_frame()
        if not depth or not color: continue

        depth_image = np.asanyarray(depth.get_data())
        color_image = np.asanyarray(color.get_data())

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)        
        
        #BLOB detection
      
        mask = cv2.inRange(color_image, lower, upper)
        cv2.imshow('mask', mask)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours)>0:
            blob = max(contours, key=lambda el: cv2.contourArea(el))
            M = cv2.moments(blob)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            cv2.circle(color_image, center, 2, (0,0,255), -1)

        
        #ARUCO
        #color_intrinsic = color.profile.as_video_stream_profile().intrinsics

        #corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray,dictionary)
       
        #rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners,30, camera_matrix(color_intrinsic), np.zeros(5))
 
        #color_image	= cv2.aruco.drawDetectedMarkers(color_image,corners)
        #color_image	= cv2.aruco.drawAxis(color_image, camera_matrix(color_intrinsic), np.zeros(5), rvecs, tvecs,20)

        
        #circle
        #gray = cv2.medianBlur(gray,5)
        #circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1 = 100, param2 = 30, minRadius = 0, maxRadius = 0)

        #detected_circles = np.uint16(np.around(circles))
        #for(x,y,r) in detected_circles[0,:]:
        #    cv2.circle(color_image,(x,y),r,(0,255,0),3)

    

        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)       
finally:

    # Stop streaming
    pipeline.stop()

if __name__ == "__main__":

    #sys.argv = [sys.argv[0], "record"]
    print(sys.argv) 