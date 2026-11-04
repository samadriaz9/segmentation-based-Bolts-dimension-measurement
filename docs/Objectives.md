# Objectives

Hardware for every number below: Intel Core i7-10870H, 16 GB RAM, NVIDIA GeForce RTX 3070 Laptop (8 GB), Windows, digital microscope on its stand. Capture size is 1280×720. Each reported number is raw output from this machine. The vernier readings are in `measurement/caliper.csv`: 6.0 mm, 7.0 mm, and 8.0 mm, one reading per bolt.

## MO-1

| Field | Entry |
|---|---|
| What is measured | Mean absolute error, in millimetres, of the reported flat-to-flat size. Width and height are both that distance, because the vernier was placed across two opposite sides of the hexagon. |
| How | A script compares the command output for each photo with a hand-measured caliper file. The table lists every instance, then the mean, the median, and the worst case. |
| Target | Mean absolute error at or below 0.25 mm, over at least 10 held-out photos. The three bolts are 6 mm, 7 mm, and 8 mm across the flats, so 0.25 mm keeps those sizes distinct. |
| Conditions | Locked microscope, 1280×720. Hexagon face toward the camera, with one 3.0 mm board square in the same photo. The same photos every run. |
| Not included | Frames where the bolt or the square is cut by the image border, and any frame taken after focus, zoom, or column height changed. |

### MO-1 result: missed

`python measurement/accuracy.py` measured the 15 validation photos and the 6 test photos. The command undistorts, then prints one flat-to-flat size. Raw output is `measurement/raw_outputs.json`.

| File | Split | Caliper | Raw mm | Raw error | Undistorted mm | Undistorted error | Cut by border |
|---|---|---:|---:|---:|---:|---:|---|
| 6mm_bolt_008.png | val | 6.0 | 5.582 | -0.418 | 5.535 | -0.466 | no |
| 6mm_bolt_010.png | val | 6.0 | 5.838 | -0.162 | 5.685 | -0.315 | no |
| 6mm_bolt_018.png | val | 6.0 | 5.589 | -0.411 | 5.599 | -0.401 | no |
| 6mm_bolt_019.png | val | 6.0 | 5.652 | -0.347 | 5.658 | -0.342 | no |
| 6mm_bolt_020.png | val | 6.0 | 5.643 | -0.357 | 5.692 | -0.308 | yes |
| 7mm_bolt_002.png | val | 7.0 | 7.268 | +0.268 | 7.189 | +0.189 | no |
| 7mm_bolt_005.png | val | 7.0 | 7.159 | +0.159 | 7.139 | +0.139 | yes |
| 7mm_bolt_007.png | val | 7.0 | 4.732 | -2.268 | 4.421 | -2.579 | yes |
| 7mm_bolt_009.png | val | 7.0 | 7.177 | +0.177 | 7.167 | +0.167 | no |
| 7mm_bolt_020.png | val | 7.0 | 7.299 | +0.299 | 7.214 | +0.214 | no |
| 8mm_bolt_008.png | val | 8.0 | 8.055 | +0.055 | 8.075 | +0.075 | no |
| 8mm_bolt_013.png | val | 8.0 | 7.904 | -0.096 | 7.958 | -0.042 | no |
| 8mm_bolt_016.png | val | 8.0 | 8.095 | +0.095 | 8.083 | +0.083 | no |
| 8mm_bolt_017.png | val | 8.0 | 8.029 | +0.029 | 8.082 | +0.082 | no |
| 8mm_bolt_023.png | val | 8.0 | 8.070 | +0.070 | 8.076 | +0.076 | yes |
| 6mm_bolt_016.png | test | 6.0 | 5.526 | -0.474 | 5.586 | -0.414 | no |
| 6mm_bolt_017.png | test | 6.0 | 5.428 | -0.572 | 5.412 | -0.588 | no |
| 7mm_bolt_010.png | test | 7.0 | 6.760 | -0.240 | 7.184 | +0.183 | yes |
| 7mm_bolt_019.png | test | 7.0 | 7.410 | +0.410 | 7.502 | +0.502 | no |
| 8mm_bolt_005.png | test | 8.0 | 7.054 | -0.946 | 6.943 | -1.057 | yes |
| 8mm_bolt_019.png | test | 8.0 | 7.659 | -0.341 | 7.586 | -0.414 | yes |

All 21, undistorted: mean absolute error 0.4111 mm, median 0.3077 mm, worst 2.5794 mm, mean absolute percentage error 6.0382%.

The 14 photos whose labelled outline does not touch the border: mean absolute error 0.2769 mm, median 0.2647 mm, worst 0.5875 mm (`6mm_bolt_017.png`, 5.412 mm), mean absolute percentage error 4.3494%. That is the set named in Conditions. It is still above 0.25 mm.

The 6 mm bolts read low on every fully visible photo, from 5.412 mm to 5.685 mm. The closest fully visible 8 mm reading is 7.958 mm, 0.042 mm under the caliper. None of the 6 mm photos read 6.029 mm.

Same 14 photos with undistortion skipped: mean absolute error 0.2724 mm, median 0.2835 mm, worst 0.5724 mm, mean absolute percentage error 4.2622%. Skipping undistortion changed the mean by 0.0045 mm. On these frames the correction did not bring the error under the target. The worst single change is `7mm_bolt_007.png`, which is cut by the frame: 4.732 mm raw, 4.421 mm after undistortion, against a 7.0 mm caliper.

## MO-2

| Field | Entry |
|---|---|
| What is measured | Mean intersection-over-union of the predicted bolt mask against the labelled bolt mask. |
| How | A script scores every image in the committed test split. The decision threshold is chosen on the validation split only. |
| Target | Mean bolt IoU at or above 0.90 on the full test split. A 0.25 mm measurement on a bolt a few millimetres across needs the outline on the true edge, including the thread. |
| Conditions | Undistorted images from the locked camera. Threshold fixed from validation before the test split is scored. |
| Not included | Images with the bolt cut by the frame, and any threshold tried after seeing test results. |

### MO-2 result: met

`models/test_metrics.json`, six test images, score threshold 0.95 chosen on validation: mask IoU 0.963, mAP@0.5 1.0, mAP@0.5:0.95 0.9693, precision 1.0, recall 1.0. The target was 0.90.

## Edge of the frame

The condition is the bolt near the edge. `measurement/edge_condition.py` orders the same 21 undistorted errors by the gap from the labelled outline to the border.

| Gap | Photos | Mean absolute error | Worst |
|---|---:|---:|---:|
| Touching the border | 7 | 0.6794 mm | 2.5794 mm |
| Within 5 percent of the border | 4 | 0.0967 mm | 0.1889 mm |
| Clear of the border | 10 | 0.3490 mm | 0.5875 mm |

Being close is not what breaks the measurement. The error jumps when the outline is cut by the frame. `7mm_bolt_007.png` is the worst, 4.421 mm against a 7.0 mm caliper. Photos that are clear of the border still miss 0.25 mm, because the 6 mm bolts read low even when the whole hexagon is inside the frame.

## Refusing a clipped outline

The command refuses when the predicted mask comes within 2 pixels of the frame. On these 21 photos that rule refuses 5: `7mm_bolt_005.png`, `7mm_bolt_007.png`, `7mm_bolt_010.png`, `8mm_bolt_005.png`, and `8mm_bolt_019.png`. The other 16 are still answered. Their mean absolute error is 0.2663 mm, median 0.2610 mm, worst 0.5875 mm. The cost is 5 refused photos. The rule removes the 2.5794 mm error and does not bring the mean under 0.25 mm.

`dataset/images/7mm_bolt_007.png` prints `object clipped by frame`, prints no millimetres, and exits 4.
