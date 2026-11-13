# TensorFlow Lite tracker (legacy)

`SVProTracker_R3.py` is the first tracking program. It runs a TensorFlow Lite SSD-style object detection
model (`detect.tflite` plus `labelmap.txt`) and sends target position and velocity to the Arduino gimbal controller.
It is kept for reference; the YOLO tracker in `../tracker-yolo/` supersedes it.

It was adapted from the TensorFlow Lite example scripts by Evan Juras (EJ Technology Consultants / EdjeElectronics) and developed with reference to
Ramaditya Kotha's PiMediapipeObjectDetection. See `../../NOTICE`.

## Run

```
python SVProTracker_R3.py --modeldir=../../models/tflite-ssd-oct2024
```

The model folder must contain `detect.tflite` and `labelmap.txt`. This program needs a model with the TensorFlow Lite
detection post-processing outputs (like `models/tflite-ssd-oct2024`). It cannot run the MediaPipe Model Maker model in
`models/mediapipe-dec2024`.

`alert.wav` must be in the same folder as the script.

License: Apache-2.0.
