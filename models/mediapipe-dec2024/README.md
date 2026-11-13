# MediaPipe Model Maker model, December 2024

`model.tflite` is an object detection model made with **MediaPipe Model Maker** on Google Colab in December 2024, trained on the
633-image dataset in `../../datasets/` (fixed split: 568 training, 65 validation; see `../../datasets/splits/`).
License: Apache-2.0.

- Base: a pre-trained MobileNetV2 model (MediaPipe Model Maker object detector).
- `labels_embedded_in_model.txt` lists the labels stored inside the model. It includes four unused leftovers (`rocket`, `backlit` twice, `1`) after the eight real classes. They are artifacts of annotating with RectLabel Pro and do not affect detection. The cleaned dataset annotations omit them.
- **This model is NOT used by `software/tracker-tflite-legacy/`**, which needs the SSD-style model in `../tflite-ssd-oct2024/`. This one is meant for the MediaPipe Tasks object detector.
- The MediaPipe Model Maker Colab procedure no longer works (Colab dropped support), so this model cannot easily be rebuilt. The training records are kept here.

## Training records (`training-records/`)

| File | What it is |
|---|---|
| `colab_log_2024-12-07_run.pdf` | Complete Colab log of a run on 2024-12-07 |
| `colab_log_2024-12-16_run.pdf` | Complete Colab log of a run on 2024-12-16 |
| `model_build_performance_by_epoch.xlsx` | Loss by epoch for the December 2024 build |
| `colab_log_earlier_small_dataset_rocket8_166_images.pdf` | Log of an earlier, smaller run (139 training and 27 validation images, 50 epochs) |

The notebook is Google's "Object detection model customization guide" (Apache-2.0, Copyright 2023 The MediaPipe Authors). The validation set is small (65 images, only 2 to 4 each for engines, jellyfish and staging), so performance numbers for those classes are not reliable.
