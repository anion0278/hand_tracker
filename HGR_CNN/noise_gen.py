import numpy as np
import cv2
import noise
import random

class Noise_gen:
    def __init__(self,shape):                #init with noise image generation
        self.shape = (shape[0]*2,shape[1]*2)
        scale = 50.0
        octaves = 4
        persistence = 0.5
        lacunarity = 2.0

        self.world = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.world[i][j] = int(255*(noise.pnoise2(i/scale, 
                                    j/scale, 
                                    octaves=octaves, 
                                    persistence=persistence, 
                                    lacunarity=lacunarity, 
                                    repeatx=shape[0], 
                                    repeaty=shape[1], 
                                    base=0)))

    def addNoise(self,img):
        xrand = int(random.random() * img.shape[0])
        yrand = int(random.random() * img.shape[1])
        return cv2.equalizeHist(np.clip(self.world[xrand:xrand+img.shape[0],yrand:yrand+img.shape[1]] + img,0,255).astype(np.uint8))

