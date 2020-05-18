import cv2

class Visualizer:
    """OpenCV visualizer"""
    
    def display_image(self,name,img):
        cv2.imshow(name,img.astype("uint8"))
        cv2.waitKey()

    def display_video(self,name,img,delay):
        cv2.imshow(name,img.astype("uint8"))
        cv2.waitKey(delay)


