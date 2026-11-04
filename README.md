# Measure It

Photograph a small bolt with a digital microscope and report its length and diameter in millimetres.

Recording: https://youtu.be/huN9wxi7zYc

Status: calibration, 72 labelled images, a fine-tuned Mask R-CNN, and a measure command are in the repository. The caliper comparison is in [Objectives](docs/Objectives.md). Mean absolute error on the fully visible held-out photos is 0.2769 mm, above the 0.25 mm target.

## Hardware

- Camera: digital microscope on a stand. Column height, focus, zoom, and resolution stay locked after calibration.
- Ground truth: vernier caliper.
- Computer: Intel Core i7-10870H, 16 GB RAM, NVIDIA GeForce RTX 3070 Laptop (8 GB), Windows.

## Documents

- [Plan](docs/Plan.md)
- [Objectives](docs/Objectives.md)
- [Definition of Done](docs/Definition_of_Done.md)

## Dataset

72 microscope photos, 1280×720, 24 each of the 6 mm, 7 mm, and 8 mm bolts. Labels are polygon masks of the bolt. The split ratio is 70.8% / 20.8% / 8.3%: 51 train, 15 validation, 6 test.

## Run

```text
python calibration/calibrate.py
python models/train_maskrcnn.py
python inference/measure.py path\to\photo.png
python measurement/accuracy.py
```

Library versions are pinned in `requirements.txt`. The measure command prints no millimetres when the bolt is missing, the square is missing, or the outline touches the frame.

The measure command undistorts the photo, finds the bolt, and scales it from the 3.0 mm square in that photo. Put the weights at `models/maskrcnn_bolt.pth`.

Download: https://drive.google.com/file/d/1-PsKevRXKI-9Jyr2yQLZASatWYW5CLN8/view?usp=sharing

SHA-256: `f89d6a8a3e9249c23e95732cf942faefc1a87faf09ef7b06f2fe72c38f5a8ee8`

## Calibration board

Print instructions are in [calibration/boards/PRINT.txt](calibration/boards/PRINT.txt). Print at 100% scale. Measure one square with the caliper before any calibration photo.
