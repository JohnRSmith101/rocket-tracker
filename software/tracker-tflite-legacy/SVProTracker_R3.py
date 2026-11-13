# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 John R. Smith
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Portions of this file are adapted from the TensorFlow Lite object
# detection example scripts by Evan Juras (EJ Technology Consultants /
# EdjeElectronics), "TensorFlow-Lite-Object-Detection-on-Android-and-
# Raspberry-Pi" (Apache-2.0), which are in turn based on TensorFlow Lite
# examples (Copyright The TensorFlow Authors, Apache-2.0).
#
# Developed with reference to "PiMediapipeObjectDetection" by Ramaditya
# Kotha (RamadityaK), MIT License, Copyright (c) 2024 RamadityaK.
# The MIT License text is in LICENSES/MIT-RamadityaK.txt.
# See the NOTICE file.

###############################################################
# Raspberry Pi Canon EOS 70D camera Object Detection + Arduino Tracking
# Version _R3: frame-rate-aware feedforward and prediction
# Runs a TensorFlow Lite (SSD-style) detection model: detect.tflite + labelmap.txt
###############################################################

import os
import argparse
import cv2
import numpy as np
import sys
import importlib.util
import serial
import time
import faulthandler
import ctypes

from array import array
from time import sleep
from datetime import datetime

import pygame
import pygame.camera
from pygame import mixer

# Serial Port Setup
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1.0)

#pygame and pygame camera initialization
pygame.init()
pygame.camera.init()


# ===== Predictor + Feedforward configuration (INSERTED) =====
# Telescope/camera field of view (degrees)
FOV_X_DEG = 3.20  # Was 3.88.  Theoretical:  3.20 deg
FOV_Y_DEG = 1.81  # Was 2.60.  Theoretical:  1.81 deg

# Measured-latency prediction horizon clamps (seconds)
TAU_MIN = 0.05
TAU_MAX = 1.0

MeasuredLag = 0.413  # was .413 seconds, fixed measured end-to-end latency

# Velocity low-pass filter (0..1). Higher = smoother but more lag.
VEL_ALPHA = 1.0  #was 0.88

# Position smoothing (0..1). Higher = more weight on current measurement.
POS_ALPHA = 1.0  # Was 0.6

# Feedforward taper and deadbands near screen center
FF_TAPER_AZ_PX = 40  # Was 20
FF_TAPER_EL_PX = 40  # Was 16
CENTER_HOLD_AZ_PX = 3
CENTER_HOLD_EL_PX = 3
VEL_DEADBAND_DEG_S = 0.002  # Was 0.015

# Frame-rate-aware feedforward scaling
FF_FULL_FPS = 7.0
FF_REDUCED_FPS = 4.0
FF_MIN_SCALE = 0.2
LOW_FPS_DISABLE_FF = 3.0
TAU_LOW_FPS_SCALE = 0.85

# Rate limit transmission to Arduino (seconds).
MIN_SEND_INTERVAL = 0.05  # 20 Hz max update rate
last_send = 0.0

# Predictor state
prev_meas_x_raw = None
prev_meas_y_raw = None
prev_meas_t = None
vx_deg_s_f = 0.0
vy_deg_s_f = 0.0
# ===== END Predictor + Feedforward configuration =====
time.sleep(.01) # was 3
ser.flushInput() #Clear the input buffer

ArduinoMessage = " "

# Timing controls
MIN_SEND_INTERVAL = 0.05   # 20 Hz max update rate
last_send = 0.0

# EMA smoothing for transmitted position only
alpha = POS_ALPHA
prev_x = 480.0
prev_y = 270.0

###############################################################
# Pygame bell
###############################################################
mixer.init()
alert = mixer.Sound('alert.wav')

###############################################################
# Display size & center
###############################################################
imgW = 960  #Was 800
imgH = 540  #was 600

CenterX = imgW // 2
CenterY = imgH // 2

###############################################################
# Tensorflow Lite setup (unchanged from original)
###############################################################
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', required=True)
parser.add_argument('--graph', default='detect.tflite')
parser.add_argument('--labels', default='labelmap.txt')
parser.add_argument('--threshold', default=0.3)
parser.add_argument('--resolution', default='1440x1080')
parser.add_argument('--edgetpu', action='store_true')

args = parser.parse_args()
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

if use_TPU and GRAPH_NAME == 'detect.tflite':
    GRAPH_NAME = 'edgetpu.tflite'

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
if labels[0] == '???':
    del(labels[0])

if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
        experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
floating_model = (input_details[0]['dtype'] == np.float32)

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

outname = output_details[0]['name']
if ('StatefulPartitionedCall' in outname):
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else:
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

#Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()



#List available cameras
cameras = pygame.camera.list_cameras()

if not cameras:
    print("No cameras found.  Exiting.")
    sys.exit()
    
#Select the first camera
camera_id = cameras[0]
print(f"Using camera: {camera_id}")

#Load the camera (default resolution is often 640x480)
cam = pygame.camera.Camera(camera_id, (imgW, imgH))

#Start capturing
cam.start()

#Get dimensions of pygame-captured image data
img = cam.get_image()
WIDTH, HEIGHT = img.get_width(), img.get_height()

print(WIDTH)
print(HEIGHT)
   
# Setup dislay for pygame-formatted imagery
#screen = pygame.display.set_mode((WIDTH, HEIGHT))
#pygame.display.set_caption("Pygame to OpenCV Conversion")


###############################################################
# Video Recording Setup
###############################################################
recording = False
video_writer = None
RecordingState = "Not Recording"

def start_recording():
    global recording,video_writer
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"recording_{timestamp}.mp4"
    
    #MP4 with H264 on Raspberry Pi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    video_writer = cv2.VideoWriter(filename,fourcc,30,(imgW, imgH))
    
    recording=True
    print(f"### Recording started: {filename}")
    
def stop_recording():
    global recording, video_writer
    if video_writer:
        video_writer.release()
        video_writer = None
    recording=False
    print("### Recording stopped")

running = True        
time.sleep(2) # let startup settle for entering Main Loop            

###############################################################
# Capture & display image using pygame, convert it to cv2 format
###############################################################
        
time.sleep(5) # let startup settle for entering Main Loop            
###############################################################
# Main Loop
###############################################################
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            #Stop the cam and quit
            cam.stop()
            running = False
            sys.exit()
#    ArduinoMessage = " "
    if ser.in_waiting > 0:
        ArduinoMessage = ser.readline().decode('utf-8').strip()
    
    time.sleep(0.001) #was 0.1
    t1 = cv2.getTickCount()

    # Acquire frame and resize to expected shape [1xHxWx3] 
    #Grab the current frame from the camera
    pygame_surface = cam.get_image() # pygame surface
    frame_timestamp = time.time() #used to calculate target angular velocity

    
    #Display the frame in the pygame window
#    screen.blit(pygame_surface, (0,0))
#    pygame.display.flip()
      
    #Convert the pygame Surface to a numpy array (RGB format)
    view = pygame.surfarray.array3d(pygame_surface)
    
    #Transpose the dimensions to (height, width, channels) for OpenCV
    opencv_frame_rgb = view.transpose([1,0,2])
    
    #Convert from RGB (Pygame's format) to BGR (OpenCV's format)
    opencv_frame_bgr = cv2.cvtColor(opencv_frame_rgb, cv2.COLOR_RGB2BGR)
    
    #Display the frame using OpenCV (Optional)
    #This will open a separate OpenCV window
#    cv2.imshow("OpenCV Frame",opencv_frame_bgr)
    cv2.imshow('Object Detector', opencv_frame_bgr)

#Begin adding object detection lines of code from rocket_tracker_R4.py    
    frame1=opencv_frame_bgr
    frame_timestamp = time.time()

    frame = frame1.copy() #2D Array
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #3D array (height, width, 3channels-RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height)) #3D array (height, width, 3channels-RGB)
    input_data = np.expand_dims(frame_resized, axis=0) #4D array(1,height, width, 3channels-RGB)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5


    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]


    for i in range(len(scores)):
        if scores[i] > min_conf_threshold:

            ymin = int(max(1, boxes[i][0] * imgH))
            xmin = int(max(1, boxes[i][1] * imgW))
            ymax = int(min(imgH, boxes[i][2] * imgH))
            xmax = int(min(imgW, boxes[i][3] * imgW))

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

            TgtX = (xmin + xmax) // 2
            TgtY = (ymin + ymax) // 2

            # clamp for safety
            TgtX_raw = max(0, min(imgW - 1, TgtX))
            TgtY_raw = max(0, min(imgH - 1, TgtY))

            # smoothing for transmitted position only; keep raw measurement for velocity
            TgtX_s = alpha * TgtX_raw + (1.0 - alpha) * prev_x
            TgtY_s = alpha * TgtY_raw + (1.0 - alpha) * prev_y
            prev_x, prev_y = TgtX_s, TgtY_s

# ===== Predictor + Feedforward DROP-IN (R1) =====
            now = time.time()
            tau = max(TAU_MIN, min(TAU_MAX, MeasuredLag))

            # velocity estimate from raw target motion to avoid EMA phase lag
            fps_est = 0.0
            if prev_meas_t is None:
                dtm = MIN_SEND_INTERVAL
                vx_px_s = 0.0
                vy_px_s = 0.0
            else:
                dtm = now - prev_meas_t
                if dtm < 1e-3:
                    dtm = MIN_SEND_INTERVAL

                fps_est = 1.0 / dtm
                vx_px_s = (TgtX_raw - prev_meas_x_raw) / dtm
                vy_px_s = (TgtY_raw - prev_meas_y_raw) / dtm

            # frame geometry (from actual frame)
            imgH, imgW = frame.shape[:2]

            # px -> deg/sec
            deg_per_px_x = FOV_X_DEG / float(imgW)
            deg_per_px_y = FOV_Y_DEG / float(imgH)
            vx_deg_s = vx_px_s * deg_per_px_x
            vy_deg_s = vy_px_s * deg_per_px_y

            # filter velocity
            vx_deg_s_f = VEL_ALPHA * vx_deg_s_f + (1.0 - VEL_ALPHA) * vx_deg_s
            vy_deg_s_f = VEL_ALPHA * vy_deg_s_f + (1.0 - VEL_ALPHA) * vy_deg_s

            # adapt prediction/feedforward aggressiveness to measured update rate
            if fps_est >= FF_FULL_FPS:
                ff_rate_scale = 1.0
                tau_use = tau
            elif fps_est >= FF_REDUCED_FPS:
                ff_rate_scale = 0.5
                tau_use = tau
            elif fps_est >= LOW_FPS_DISABLE_FF:
                ff_rate_scale = FF_MIN_SCALE
                tau_use = tau * TAU_LOW_FPS_SCALE
            else:
                ff_rate_scale = 0.0
                tau_use = tau * TAU_LOW_FPS_SCALE

            # predict true position using smoothed position and frame-rate-aware lag
            pred_x = TgtX_s + vx_px_s * tau_use
            pred_y = TgtY_s + vy_px_s * tau_use

            pred_x = max(0.0, min(imgW - 1.0, pred_x))
            pred_y = max(0.0, min(imgH - 1.0, pred_y))

            # taper feedforward near center to avoid center hunting
            err_az_px = abs(TgtX_s - CenterX)
            err_el_px = abs(TgtY_s - CenterY)
            ff_scale_az = min(1.0, err_az_px / float(FF_TAPER_AZ_PX))
            ff_scale_el = min(1.0, err_el_px / float(FF_TAPER_EL_PX))
            vx_send = vx_deg_s_f * ff_scale_az * ff_rate_scale
            vy_send = vy_deg_s_f * ff_scale_el * ff_rate_scale

            # velocity deadband to suppress detector jitter near center
            if abs(vx_send) < VEL_DEADBAND_DEG_S:
                vx_send = 0.0
            if abs(vy_send) < VEL_DEADBAND_DEG_S:
                vy_send = 0.0

            # center hold to avoid dithering at lock
            if err_az_px < CENTER_HOLD_AZ_PX:
                pred_x = float(CenterX)
                vx_send = 0.0
            if err_el_px < CENTER_HOLD_EL_PX:
                pred_y = float(CenterY)
                vy_send = 0.0

            # send predicted position and tapered feedforward
            if now - last_send >= MIN_SEND_INTERVAL:
                last_send = now
                packet = f"{TgtX_raw:.2f},{TgtY_raw:.2f},{vx_deg_s:.6f},{vy_deg_s:.6f}\n".encode("utf-8")
                ser.write(packet)

            prev_meas_x_raw = TgtX_raw
            prev_meas_y_raw = TgtY_raw
            prev_meas_t = now

# ===== END Predictor + Feedforward DROP-IN =====

            TgtX = int(round(TgtX_s))
            TgtY = int(round(TgtY_s))
            alert.play()


            cv2.putText(frame, f"Target: {TgtX},{TgtY}", (30,110),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2)

            object_name = labels[int(classes[i])]
            label = f'{object_name}: {int(scores[i]*100)}%'
            cv2.putText(frame, label, (xmin, ymin-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    #Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2 - t1)/freq
    frame_rate_calc = 1/time1

    #Draw ArduinoMessage in corner of frame
    cv2.putText(frame,ArduinoMessage,(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    #Draw a "+" sign in the screen center
    cv2.putText(frame,"+",(480,270),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    
    #Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    #Draw RecordingState in corner of frame
    cv2.putText(frame,RecordingState,(30,150),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)


###################################################################
# Write video frame if recording
###################################################################
    if recording and video_writer is not None:
        video_writer.write(frame)
    
###################################################################
#Display
###################################################################

    cv2.imshow('Object Detector', frame)
    key= cv2.waitKey(10) # units are msec.  was 1
    if key == ord('q'):
        break


    if key == ord('r'):
        if not recording:
            start_recording() 
            RecordingState = "Recording"
        else:
            stop_recording()
            RecordingState = "Not Recording"
######################################################
# Cleanup
######################################################
if recording:
    stop_recording()
                    
    #Check for a key press in the OpenCV window to quit
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        running = False
#        break
#Release resources
cam.stop()
pygame.quit
cv2.destroyAllWindows()   
