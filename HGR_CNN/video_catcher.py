import pyrealsense2 as rs
import numpy as np

class VideoImageCatcher:
    """class for capturing image from Intel RealSense D435i camera"""

    def __init__(self,config):
        self.config = config
        self.__streaming = False

    def init_stream(self):
        self.__pipeline = rs.pipeline()
        pipeline_config = rs.config()

        pipeline_config.enable_stream(rs.stream.depth, self.config.img_camera_size[0], self.config.img_camera_size[1], rs.format.z16, self.config.camera_rate)
        pipeline_config.enable_stream(rs.stream.color, self.config.img_camera_size[0], self.config.img_camera_size[1], rs.format.bgr8, self.config.camera_rate)
   
        

        self.__colorizer = rs.colorizer()
        self.__colorizer.set_option(rs.option.visual_preset,1)
        self.__colorizer.set_option(rs.option.min_distance,0.2)        
        self.__colorizer.set_option(rs.option.max_distance,1.05)
        self.__colorizer.set_option(rs.option.color_scheme,2)
        self.__colorizer.set_option(rs.option.histogram_equalization_enabled,0)
        #self.__filter_HF = rs.hole_filling_filter()
        #self.__filter_HF.set_option(rs.option.holes_fill, 3)


        try:
            profile = self.__pipeline.start(pipeline_config)

            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            depth_sensor.set_option(rs.option.visual_preset,5)
         
            self.intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
            print("Depth Scale is: " , self.depth_scale)

            clipping_distance_in_meters = 1 #1 meter
            self.__clipping_distance = clipping_distance_in_meters / self.depth_scale

            align_to = rs.stream.color
            self.__align = rs.align(align_to)
            self.__streaming = True
            return True
        except Exception as ex:
            print('Camera not connected.. ')
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
                depth_frame = frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame: # check correctness of the frames, or skip
                  return None

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                #filtered = self.__filter_HF.process(depth_frame)
                filtered = self.__colorizer.colorize(depth_frame)
                #filtered = self.__colorizer.colorize(filtered)
                rawdepth = np.asanyarray(depth_frame.get_data())
                min = 0.2
                max = 1.05
                colored = np.clip(rawdepth*self.depth_scale,min,max)
                norm = lambda n: n/max
                colored = 255-norm(colored)*255

                depth_colorized = np.asanyarray(filtered.get_data())
                depth_colorized = depth_colorized[:,:,0]
                return depth_image,color_image,colored#depth_colorized
                #return depth_image,depth_image,depth_colorized

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

    def get_depth_raw_color(self):
        di,ci,_ = self.__fetch_image()
        return ci,di
    def get_data(self): #todo
        _,ci,dm = self.__fetch_image()
        return ci,dm

    def get_fingertip_pos(self): #todo
        if self.__streaming:
            return [0,0,0]

