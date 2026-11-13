# YOLO tracker (current)

`SVProTracker_YOLO_NCNN_R2_Video_Data_Log.py` runs the YOLO11n model on the tracking camera's
video, picks the highest-confidence detection in each frame, and sends position and velocity to the
Arduino gimbal controller over serial (`/dev/ttyACM0`, 115200 baud). It plays `alert.wav` on every frame that
has a detection, and records video and a data log (press `r` to start and stop).

License: AGPL-3.0 (it uses Ultralytics YOLO). See `../../LICENSE`.

## Set up

1. Follow the Build Book section "Deploying YOLO models to Raspberry Pi 5" for the operating system and packages, or install the packages listed in `requirements.txt`.
2. Export the model to NCNN format (the exported folder is not included; you create it):
   ```
   yolo export model=../../models/yolo/my_model.pt format=ncnn
   ```
   This creates a folder named `my_model_ncnn_model` next to `my_model.pt`.
3. Run the tracker, pointing it at that folder:
   ```
   python SVProTracker_YOLO_NCNN_R2_Video_Data_Log.py --modeldir=../../models/yolo/my_model_ncnn_model
   ```

The option is `--modeldir` (not `--model`).

`alert.wav` must be in the same folder as the script. Replace it with any sound you prefer.
