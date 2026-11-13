# TensorFlow Lite model, October 2024

`detect.tflite` and `labelmap.txt` are the detection model used by the legacy tracker
(`../../software/tracker-tflite-legacy/`). License: Apache-2.0.

- It is an SSD-style TensorFlow Lite detector (it contains the standard TFLite detection post-processing block).
- The eight classes in `labelmap.txt` are: staging, blob, plume, backlit_plume, jellyfish, contrail, engines, point.
- The file is named for October 20, 2024 (`20241020`).

**Provenance:** a complete training record for this model was not kept. It was made with a Google Colab TensorFlow Lite training notebook of the kind provided by EdjeElectronics, on an earlier version of the rocket dataset. The model is provided as-is, as the file the author actually ran. It cannot be rebuilt from the records in this repository.

Run it with `python SVProTracker_R3.py --modeldir=../../models/tflite-ssd-oct2024` from the legacy tracker folder.
