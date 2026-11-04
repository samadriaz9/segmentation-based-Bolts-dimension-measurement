# Plan

Measure It. The camera is a digital microscope on its stand. Column height, focus, zoom, and capture resolution stay locked after they are set. The computer is an Intel Core i7-10870H, 16 GB RAM, NVIDIA GeForce RTX 3070 Laptop (8 GB), Windows. Ground truth is a vernier caliper.

## Object

Three hexagonal bolts. The microscope looks straight down at the hexagon face. The vernier measurement is the distance between two opposite flat sides: 6 mm, 7 mm, and 8 mm. Corner-to-corner is larger, about 6.93 mm, 8.08 mm, and 9.24 mm, and that distance is not the ground truth. The reported width and the reported height are both the flat-to-flat distance, because every pair of opposite sides on a regular hexagon has the same length.

## Reference

There is no separate washer. The scale is one square of the 3.0 mm ChArUco board, already measured with the vernier. Every measurement photo shows the bolt beside the board, with at least one full square visible. That square is 3.0 mm, so it converts the bolt outline from pixels to millimetres. The caliper reading of the bolt is the ground truth.

## Order and time

Active work is 8 hours. Training time and installs do not count.

1. Print the board, lock the camera, capture 20 or more board images. 1.5 h.
2. Calibrate in OpenCV, save the camera matrix and distortion, report the reprojection error. 1.0 h.
3. Capture 70 or more images of a bolt beside the board. Undistort them. 1.5 h.
4. Label the bolt masks. Commit the 70/20/10 file list before training. 1.5 h.
5. Fine-tune the segmentation model and score the untouched test split. 0.5 h active.
6. One command from a new image to millimetres. 1.0 h.
7. At least 10 caliper comparisons, then the same instances on the raw frames. 1.0 h.

Items 8 and 9 wait until this list is done. If the time is gone, stop and list what is unfinished.

## Calibration board

The field of view is too small for a full 7×9 checkerboard in every frame, so the board is a ChArUco pattern with small squares. Partial views still calibrate. Two files are in `calibration/boards/`: 2.0 mm squares and 3.0 mm squares, drawn with OpenCV 5.0.0 and dictionary `DICT_4X4_250`. Print at 100% scale, tape the sheet to a rigid flat plate, and measure one square with the caliper. Keep the board on which at least six markers are sharp and fully inside the locked frame. Calibration uses the measured square size, not the nominal size.

## Model

Torchvision Mask R-CNN with a ResNet-50 FPN backbone, fine-tuned on these images, one class: bolt. The board square is found with the ChArUco detector, not by the segmentation model. Ultralytics YOLO is out because the assessment forbids it.

## Already decided to skip

No downloaded images. No second architecture search. Items 8 and 9 unless the accuracy table exists. The test-split file list is committed before training, chosen with seed 42, and never used to train, tune, or pick a checkpoint.

## Same-plane limit

The board and the bolt both rest on the base. The diameter in the image is the width of a cylinder, so that silhouette sits about one radius above the base. The accuracy table is where that shows up, and the write-up will use those numbers.

## Setup note, 4 November

The USB device is "Digital Microscope" and its only mode is 1280×720. Windows Camera had the device open, so capture has to happen with that app closed. A frame at 20:10 decodes seven markers, IDs 0, 14, 21, 22, 28, 35 and 36, and six chessboard corners. The first 20 frames are archived in `calibration/images_first_pass/`. Their reprojection error was 3.88 px. The second capture saved 27 frames in `calibration/images/`. Each frame's own grid fit was under 1 px, and the combined error finished at 1.14 px. The solved focal length was unstable, so those parameters were not kept. One plain square measured 3.0 mm by 3.0 mm, with a 2.1 mm marker. Twelve tilted frames, `calib_029.png` through `calib_040.png`, cover left, right, top, and bottom. Calibration on all 40 frames has a reprojection error of 1.23 px. The principal point lies inside the image. Parameters are in `calibration/camera_params.npz`. The first dataset pass was deleted because the counter changed size on its own and the bolts overlapped. The replacement set is complete: 24 photos each of the 6 mm, 7 mm, and 8 mm bolts, in `dataset/raw/6mm`, `dataset/raw/7mm`, and `dataset/raw/8mm`. Capture stopped after each 24 until the next bolt was confirmed. The board stays on a stiff sheet so it can be lifted and tilted for the calibration photos.

## Decision record

The scale is one measured 3.0 mm square of the ChArUco board, found by the OpenCV detector. The bolt outline is a fine-tuned torchvision Mask R-CNN. Width and height are both the short side of the minimum-area rectangle around that outline. Ultralytics YOLO was rejected because the assessment forbids it. A separate washer was rejected because the board is already in every measurement photo. The cost is in `docs/Objectives.md`: the 6 mm bolt reads about 0.3 mm to 0.6 mm low, and a hexagon cut by the frame can be reported as 4.421 mm against a 7.0 mm caliper. The command refuses only when the bolt or the square is missing. It does not yet refuse a clipped outline.
