import sim
import cv2 as cv
import numpy as np
import time

class CoppeliaAPI:
    def __init__(self):
        sim.simxFinish(-1)
        self.__clientID = sim.simxStart('127.0.0.1',19999,True,True,5000,5)
        if self.__clientID!=-1:
            print ('Connected to remote API server')
        else:
            exit()

    def init_simulation(self):
        self.__hand = self.__GetObjectHandle('Hand')
        self.__fingertip = self.__GetObjectHandle('Fingertip_pos')
        self.__vision = self.__GetObjectHandle('Vision_sensor')
        self.__mask = self.__GetObjectHandle('Vision_sensor_mask')
        self.__sphere = self.__GetObjectHandle('Sphere')
        err, resolution, image = sim.simxGetVisionSensorImage(self.__clientID,self.__vision, 0,sim.simx_opmode_streaming)
        err, resolution, image = sim.simxGetVisionSensorImage(self.__clientID,self.__mask, 0,sim.simx_opmode_streaming)
        time.sleep(0.05)

    def stopSimulation(self):
        self.__clientID.simxFinish(-1)

    def get_cam_image(self):
        err, resolution, image = sim.simxGetVisionSensorImage(self.__clientID,self.__vision, 0,sim.simx_opmode_blocking)
        if err == sim.simx_return_ok:
            img = np.array(image,dtype=np.uint8)
            img.resize([resolution[1],resolution[0],3])
            img = img[:,:,1]
            return img
        else:
            print(err)
        return None

    def get_mask(self):
        err, resolution, image = sim.simxGetVisionSensorImage(self.__clientID,self.__mask, 0,sim.simx_opmode_blocking)
        if err == sim.simx_return_ok:
            img = np.array(image,dtype=np.uint8)
            img.resize([resolution[1],resolution[0],3])
            img = img[:,:,1]
            _,img_out = cv.threshold(img,10,255,cv.THRESH_BINARY)
            return img_out
        else:
            print(err)
        return None

    def move_sim(self):
        
        emptyBuff = bytearray()
        err,_,_,_,_=sim.simxCallScriptFunction(self.__clientID,"Hand",sim.sim_scripttype_childscript,"getData",[],[],[],emptyBuff,sim.simx_opmode_blocking)
        if err != sim.simx_return_ok:
           print(err)

    def __GetObjectHandle(self,name):
        res,handle=sim.simxGetObjectHandle(self.__clientID,name,sim.simx_opmode_oneshot_wait)
        if res==sim.simx_return_ok:
            print ('Handle '+name+' loaded')
            return handle
        else:
            print ('Remote API function call returned with error code: ',res)
            return None
        
    def __GetObjectPos(self,handle):
        res,posp=sim.simxGetObjectPosition(self.__clientID,handle,-1,sim.simx_opmode_blocking)
        if res==sim.simx_return_ok:
            #print ('Position loaded:',posp)
            return np.round(posp,4)
        else:
            print ('Remote API function call returned with error code: ',res)
            return [0,0,0]

    def __SetObjectPos(self,handle,sposp):
        res=sim.simxSetObjectPosition(self.__clientID,handle,-1,sposp,sim.simx_opmode_blocking)
        if not res==sim.simx_return_ok:
            print ('SOP_Remote API function call returned with error code: ',res)

    def get_fingertip_pos(self):
        return self.__GetObjectPos(self.__fingertip)

    def get_hand_pos(self):
        return self.__GetObjectPos(self.__hand)



        



