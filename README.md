# Rocket Tracker

A system that tracks and video-records rockets as they climb into space. A
motorized telescope-and-camera gimbal is steered by an operator until a small
tracking camera sees the rocket. A neural-network detector on a Raspberry Pi 5
then takes over and keeps the rocket centered while the telescope camera
records.

The complete story of why it is built this way, and how to build one yourself,
is in the **Build Book** (see below). This repository has the software,
trained models, hardware design files and training dataset.

## How it works

1. An operator slews the system by joystick, looking through a finder scope, to
   bring the rising rocket into the tracking camera's field of view.
2. A Raspberry Pi 5 runs an object detector (YOLO11n, exported to NCNN) on the
   tracking camera's video. Each time it detects the rocket, it plays an alert
   sound. This tells the operator, who cannot look at a screen, to release the
   joystick trigger.
3. The Pi sends the rocket's position and velocity to an Arduino Due over USB
   serial. The Arduino runs closed-loop PID control of the azimuth and
   elevation stepper motors to keep the rocket centered. The telescope's own
   camera (a Canon EOS 70D on a Celestron C5) records the launch.
4. When detections stop, the alert stops, and the operator can take over.

## What is where

| Folder | Contents |
|---|---|
| `software/gimbal-controller/` | Arduino Due sketches: gimbal controller, and a version that logs PID data |
| `software/tracker-yolo/` | Raspberry Pi 5 tracking program using the YOLO11n model (current) |
| `software/tracker-tflite-legacy/` | Earlier tracking program using a TensorFlow Lite model (kept for reference) |
| `models/yolo/` | The trained YOLO model that Rocket Tracker uses, and training results |
| `models/tflite-ssd-oct2024/` | The earlier TensorFlow Lite model, used by the legacy tracker |
| `models/mediapipe-dec2024/` | A MediaPipe Model Maker model from December 2024, with training records |
| `datasets/` | The 633-image training dataset (documentation here, files in a Release) |
| `hardware/kicad/` | KiCad files and Gerbers for the Gimbal Controller Box and Lid circuit boards |
| `hardware/3d-printing/` | STL files for the 3D-printed parts (print in PETG) |
| `docs/` | Original drawings (the Build Book itself is on the Releases page) |
| `LICENSES/`, `LICENSE`, `NOTICE` | Licensing and credits |

## Downloads (large files)

GitHub limits file sizes in the repository, so these are on the
[Releases page](../../releases):

- **Build Book** (PDF and Word): the full design document
- `rocket-tracker-dataset-633-images.zip`: all images, with YOLO, Pascal VOC and COCO-JSON annotations
- `data.zip`: the same dataset packaged for the YOLO training notebook in Google Colab
- `yolo-training-runs-full.zip`: all three YOLO training runs with weights and plots

## Quick start (YOLO tracker)

1. Build the hardware following the Build Book.
2. On the Raspberry Pi 5, follow the Build Book section "Deploying YOLO models to Raspberry Pi 5", or see `software/tracker-yolo/README.md`.
3. Upload `software/gimbal-controller/Rocket_Tracker_2026_R14` to the Arduino Due.

## Status and limits

- The YOLO model was trained on only 633 images, many of them near-duplicates, and some classes
  have very few examples. Expect it to work best in conditions similar to the training footage.
  See `datasets/README.md`.
- The author built and tested this one system. Other hardware may need changes.
- The TensorFlow Lite and MediaPipe material is kept as a record; the YOLO tracker supersedes it.

## Safety

Motorized mounts can pinch fingers, and optics can cause permanent eye damage if pointed at the
sun. Build and operate at your own risk. See `LICENSE`.

## License

Open source, with different licenses for different parts: Apache-2.0 (Arduino and TFLite
software), AGPL-3.0 (YOLO tracker and models), CC BY-SA 4.0 (Build Book, hardware, datasets).
Details in `LICENSE`. Credits to the projects this builds on are in `NOTICE`.

Copyright 2026 John R. Smith.
