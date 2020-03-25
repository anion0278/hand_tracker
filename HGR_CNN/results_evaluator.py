import numpy as np
import simple_recognizer as sr

class ResultsEvaluator:
    """class for results evaluation"""
    def __init__(self):
        self.__recognizer = sr.BlobRecognizer([255,255,255])

    def get_fault_pix(self,original,predicted):
        return np.sqrt(np.square(predicted[1]-original[1])+np.square(predicted[0]-original[0]))

    def compare_two_masks(self,mask1,mask2):
        original = self.__recognizer.get_blob_pos(mask1)
        predicted = self.__recognizer.get_blob_pos(mask2)
        return self.get_fault_pix(original,predicted)