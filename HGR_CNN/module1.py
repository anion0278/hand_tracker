import math
import time
import cv2
import numpy as np
import pyrealsense2 as rs

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.decimate = 1
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

state = AppState()

realsense_ctx = rs.context()
connected_devices = []
configs = []
pipelines = []
for i in range(len(realsense_ctx.devices)):
    detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
    connected_devices.append(detected_camera)
    configs.append(rs.config())
    pipelines.append(rs.pipeline())

print(connected_devices)

for i in range(len(configs)):
    configs[i].enable_device(connected_devices[i])
    configs[i].enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    configs[i].enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
    pipelines[i].start(configs[i])
    


for pipeline in pipelines:
    pipeline.stop()