# Training dataset

License: CC BY-SA 4.0. The files are large, so they are on the [Releases page](../../../releases), not in the repository.

## Files

| Download | Contents |
|---|---|
| `rocket-tracker-dataset-633-images.zip` | `images/` (633 JPEGs), `annotations_yolo/` (YOLO text), `annotations_voc_xml/` (Pascal VOC), `annotations_coco_json/` (train/validation), `splits/`, `classes.txt` |
| `data.zip` | The same images and YOLO labels packaged for the YOLO training notebook in Google Colab (`images/`, `labels/`, `classes.txt` at the top level) |

In this repository: `classes.txt`, `splits/` (the fixed 568/65 train/validation split used for MediaPipe) and `examples/`.

Mac users: `data.zip` must be created by selecting the items and choosing Compress, not by zipping a folder. See the Build Book section "YOLO Dataset Organization".

## Contents

- **633 images** with **633 annotation files** per format. YOLO format is one `.txt` per image: class number, then the box center and size as fractions of the image.
- **8 classes**: staging, blob, plume, backlit_plume, jellyfish, contrail, engines, point.
- All images are frames extracted from the author's own video clips of rocket launches, captured with cameras and optics similar to those used during automated tracking. Annotations were made with RectLabel Pro.

| Class | Labeled boxes |
|---|---|
| staging | 28 |
| blob | 136 |
| plume | 158 |
| backlit_plume | 102 |
| jellyfish | 15 |
| contrail | 35 |
| engines | 19 |
| point | 140 |

### Two kinds of frames

- **279 images at 1920x1080** are frames taken directly from video clips.
- **354 images at 1280x720** (`img280` onward) are frames from screen recordings of the tracker's display window. They include the on-screen "FPS" text and show narrow slices of the desktop window and terminal at the left and right edges. In live operation the model receives clean camera frames, because the overlay is drawn after detection. Because the labels mark only the rocket, the model treats these overlay areas as background, and the effect on detection is expected to be small. They are kept as they were used for training. An example frame is in `examples/`.

## Limitations

**1. Some classes have few examples.** The classes jellyfish (15 boxes), engines (19), staging (28) and contrail (35) have far fewer examples than the others. Expect weaker and less reliable detection of these classes. If you build your own system, collecting more examples of these cases is the most useful way to improve the model.

**2. Many frames are near-duplicates, and the YOLO validation split was random.** The images come from video clips, so consecutive frames look almost the same. The YOLO train/validation split was made randomly by the training notebook, so very similar frames can land on both sides of the split. The accuracy numbers in the YOLO training results (for example, mAP50 near 0.98) are therefore likely to be optimistic and will not necessarily hold on a new launch. That split is not fixed: running the notebook again produces a different split and slightly different results. For a fair test, evaluate the model on footage from a launch that was not used in training.

The MediaPipe version used a fixed split (568 train / 65 validation), listed in `splits/`. The same near-duplicate caution applies.

## Annotation formats

- **YOLO**: `annotations_yolo/imgN.txt`, class ids 0 to 7 in the order of `classes.txt`.
- **Pascal VOC**: `annotations_voc_xml/imgN.xml`.
- **COCO JSON**: `annotations_coco_json/train.json` and `validation.json`, category ids 1 to 8. The original files from the MediaPipe build also listed four unused leftover categories (`rocket`, `backlit` twice, `1`); they are removed here. No annotation used them.
