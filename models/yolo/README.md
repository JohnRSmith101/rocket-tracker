# YOLO model

The YOLO11n detector used by `software/tracker-yolo/`. License: AGPL-3.0 (Ultralytics YOLO). See `../../LICENSE`.

- `my_model.pt`: **the model Rocket Tracker uses**. It is the output of the 90-epoch yolo11n training run below.
- `classes.txt`: the eight classes: staging, blob, plume, backlit_plume, jellyfish, contrail, engines, point.
- `training-runs/`: results (plots, `results.csv`, `args.yaml`) for each training run. The full runs with weights are in `yolo-training-runs-full.zip` on the [Releases page](../../../../releases).

## Training runs

All runs used the Google Colab notebook "Train YOLO Models in Google Colab" from the EdjeElectronics `Train-and-Deploy-YOLO-Models` project, with Ultralytics 8.4.90, image size 640 and batch size 16, on the dataset in `../../datasets/`.

| Folder | Base model | Epochs | Best mAP50-95 (validation) |
|---|---|---|---|
| `yolo11n_60epochs` | yolo11n | 60 | 0.678 (epoch 52) |
| `yolo11n_90epochs_DEPLOYED` | yolo11n | 90 | 0.654 (epoch 83) |
| `yolo11s_60epochs` | yolo11s | 60 | 0.641 (epoch 48) |

**Do not compare the numbers across runs.** Each run used a different random split of the dataset, so they were not measured on the same validation images. The validation split was random and the images are near-duplicate video frames, so these scores are likely optimistic (see `../../datasets/README.md`). For a fair test, evaluate on footage from a launch that was not used in training.

## Use

See `../../software/tracker-yolo/README.md` for exporting the model to NCNN and running the tracker.
