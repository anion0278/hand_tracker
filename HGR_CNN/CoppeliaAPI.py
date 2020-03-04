import sim
import cv2 as cv
import numpy as np
import time

class CoppeliaAPI:
    def __init__(self):
        sim.simxFinish(-1)
        self.clientID = sim.simxStart('127.0.0.1',19999,True,True,5000,5)
        if self.clientID!=-1:
            print ('Connected to remote API server')
        else:
            exit()

    def initSimulation(self):
        self.hand = self.GetObjectHandle('Hand')
        self.vision = self.GetObjectHandle('Vision_sensor')
        self.sphere = self.GetObjectHandle('Sphere')
        err, resolution, image = sim.simxGetVisionSensorImage(self.clientID,self.vision, 0,sim.simx_opmode_blocking)
        time.sleep(0.05)

    def GetImage(self):
        err, resolution, image = sim.simxGetVisionSensorImage(self.clientID,self.vision, 0,sim.simx_opmode_blocking)
        if err == sim.simx_return_ok:
            img = np.array(image,dtype=np.uint8)
            img.resize([resolution[1],resolution[0],3])
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            imgFlip = cv.flip(img,0)
            return imgFlip
        else:
            print(err)
        return None

    def GetObjectHandle(self,name):
        res,handle=sim.simxGetObjectHandle(self.clientID,name,sim.simx_opmode_oneshot_wait)
        if res==sim.simx_return_ok:
            print ('Handle '+name+' loaded')
            return handle
        else:
            print ('Remote API function call returned with error code: ',res)
            return None
        
    def GetObjectPos(self,handle,mode):
        res,posp=sim.simxGetObjectPosition(self.clientID,handle,-1,mode)
        if res==sim.simx_return_ok:
            #print ('Position loaded:',posp)
            return np.round(posp,4)
        else:
            print ('Remote API function call returned with error code: ',res)
            return [0,0,0]

    def SetObjectPos(self,handle,sposp):
        res=sim.simxSetObjectPosition(self.clientID,handle,-1,sposp,sim.simx_opmode_blocking)
        if not res==sim.simx_return_ok:
            print ('SOP_Remote API function call returned with error code: ',res)


        



