import pyrealsense2 as rs
import numpy as np

class VideoImageFetcher:
    """class for capturing image from Intel RealSense D435i camera"""

    def __init__(self,camera_img_size,image_rate):
        self.__camera_img_size = camera_img_size
        self.__image_rate = image_rate
        self.__streaming = False

    def init_stream(self):
        self.__pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, self.__camera_img_size[0], self.__camera_img_size[1], rs.format.z16, self.__image_rate)
        config.enable_stream(rs.stream.color, self.__camera_img_size[0], self.__camera_img_size[1], rs.format.bgr8, self.__image_rate)
        
        self.__colorizer = rs.colorizer()
        self.__colorizer.set_option(rs.option.color_scheme, 2)
        self.__filter = rs.hole_filling_filter()

        try:
            profile = self.__pipeline.start(config)

            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            print("Depth Scale is: " , depth_scale)

            clipping_distance_in_meters = 1 #1 meter
            self.__clipping_distance = clipping_distance_in_meters / depth_scale

            align_to = rs.stream.color
            self.__align = rs.align(align_to)
            self.__streaming = True
            return True
        except Exception as ex:
            print('Camera not connected.. %' % ex)
            return False

    def close_stream(self):
        self.__streaming = False
        self.__pipeline.stop()

    def __fetch_image(self):
        if self.__streaming:
            try:
                # Wait for a coherent pair of frames: depth and color
                frames = self.__pipeline.wait_for_frames() # original frames
                aligned_frames = self.__align.process(frames) # aligned frames
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame: # check correctness of the frames, or skip
                   return None

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                depth_colorized = np.asanyarray(self.__colorizer.colorize(self.__filter.process(depth_frame)).get_data())

                return depth_image,color_image,depth_colorized

            except Exception as ex:
               print('Exception during streaming.. %' % ex)
               return None
        else:
            print("Streaming not initialized")
            return None

    def get_color(self):
        _,ci,_ = self.__fetch_image()
        return ci

    def get_depth_img(self):
        _,_,dm = self.__fetch_image()
        return dm

    def get_depth_raw(self):
        di,_,_ = self.__fetch_image()
        return di

