import cv2

class Visualizer:
    """OpenCV visualizer"""
    
    def display_image(self,name,img):
        cv2.imshow(name,img)
        cv2.waitKey()

    def display_video(self,name,img,delay):
        cv2.imshow(name,img)
        cv2.waitKey(delay)


