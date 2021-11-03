import cv2
import mediapipe as mp
import video_catcher as vc
import pyrealsense2 as rs
import config as c
import numpy as np
import socket
import struct

HOST = "127.0.0.1"
PORT = 4023

addr = (HOST,PORT)

extr = rs.extrinsics()
extr.rotation = [-0.9966652989387512,-0.07396058738231659,-0.03447021543979645,-0.07489585131406784,0.996834397315979,0.026679327711462975,0.032387875020504,0.029172031208872795,-0.9990496039390564]
extr.translation = [0.06177416816353798,-0.3707936704158783,1.0009115934371948]

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

conf = c.Configuration(version_name = "autoencoder", debug_mode=False, latest_model_name="prekazky_blured_and_prekazky_34k.h5")
cap = vc.VideoImageCatcher(conf)
cap.init_stream()

with mp_hands.Hands(
    model_complexity = 0,
    max_num_hands = 1,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5) as hands:
    while True:
        image,depth = cap.get_depth_raw_color()
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            #for hand_landmarks in results.multi_hand_landmarks:
            #    mp_drawing.draw_landmarks(
            #        image,
            #        hand_landmarks,
            #        mp_hands.HAND_CONNECTIONS,
            #        mp_drawing_styles.get_default_hand_landmarks_style(),
            #        mp_drawing_styles.get_default_hand_connections_style())

            rows, cols, _ = image.shape
            for landmarks in results.multi_hand_landmarks:
                x = [landmark.x * cols for landmark in landmarks.landmark]
                y = [landmark.y * rows for landmark in landmarks.landmark]
                hand_coordinates = np.column_stack((x, y))
            cv2.circle(image,(int(hand_coordinates[8][0]),int(hand_coordinates[8][1])),10,(255,255,255),-1)

            hand_coordinates[8][0] = np.clip(hand_coordinates[8][0],0,cols-1)
            hand_coordinates[8][1] = np.clip(hand_coordinates[8][1],0,rows-1)
            u = rs.rs2_deproject_pixel_to_point(cap.intrinsics,[int(hand_coordinates[8][1]),int(hand_coordinates[8][0])],depth[int(hand_coordinates[8][1]),int(hand_coordinates[8][0])])
            v = rs.rs2_transform_point_to_point(extr,u)
            a = v[0]
            v[0] = -(v[1]/1000)
            v[1] = -(a/1000)
            v[2] = (v[2]/1000)+1

            buf = struct.pack('%sf' % len(v), *v)
            s.sendto(buf,addr)
        
        cv2.imshow('',cv2.flip(image,1))
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.close_stream()
s.detach()
s.close()

