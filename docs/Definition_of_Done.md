# Definition of Done

Written before implementation. A line is done only when the evidence beside it exists. A tick with no number, path, or commit link stays open.

- [x] Reprojection error reported, with the calibration image count. Evidence: 1.2318 px on 40 frames, `calibration/camera_params.npz`.
- [x] Camera matrix and distortion coefficients load from a saved file. Evidence: `inference/measure.py` `load_camera` reads `calibration/camera_params.npz`.
- [ ] A straight edge in a calibration image stays straight after undistortion. Evidence:
- [ ] 70 or more images taken with this microscope, undistorted, with bolt and washer masks. Evidence: 72 images and 72 bolt masks in `dataset/images` and `dataset/labels`. There is no washer. The reference is the 3.0 mm square. A separate undistorted copy of the dataset was not saved.
- [ ] Train / val / test file list committed before training. Split 70 / 20 / 10, seed 42. Evidence: the lists are 51 / 15 / 6. Git history does not show them committed before the training code.
- [ ] Test split unused for training, tuning, and checkpoint selection. Evidence: `models/train_maskrcnn.py` scores `dataset/test.txt` only after the checkpoint and the threshold are chosen on validation. The commit history does not prove that order.
- [x] Test metrics reported: mask IoU, mAP@0.5, mAP@0.5:0.95, precision, recall. Evidence: `models/test_metrics.json`, mask IoU 0.963, mAP@0.5 1.0, mAP@0.5:0.95 0.9693, precision 1.0, recall 1.0.
- [x] One command takes a new image and prints width (mm), height (mm), and confidence, and saves a mask overlay. Evidence: `python inference/measure.py path\to\photo.png`.
- [x] Object not found, and reference not found, each end with no measurement. Evidence: `calibration/images/calib_001.png` prints `object not found` and exits 2. `measurement/cases/bolt_only.png`, a crop of `6mm_bolt_016.png` with no full square, prints `reference not found` and exits 3. Neither prints millimetres.
- [x] Accuracy table for at least 10 instances. Every instance, then mean, median, and worst case. Raw outputs saved. Evidence: 21 photos in `docs/Objectives.md`. Raw JSON is `measurement/raw_outputs.json`.
- [x] The same instances measured on raw images and on undistorted images, errors side by side. Evidence: the same table. In-frame mean absolute error 0.2724 mm raw, 0.2769 mm undistorted.
- [x] MO-1 met or missed, with the raw table. Evidence: missed. All 21, mean absolute error 0.4111 mm. The 14 fully inside the frame, 0.2769 mm. Target 0.25 mm.
- [x] MO-2 met or missed, with the raw test score. Evidence: met. Test mask IoU 0.963.
- [x] Everything unfinished is listed here. Evidence: the list below.

Unfinished:

- A saved before-and-after image showing a straight edge after undistortion.
- Washer masks. The plan uses the ChArUco square instead.
- Proof in git that the split lists were committed before training. The lists are 51/15/6, not a strict 70/20/10.
- Item 8. No measured series for tilt, distance, lighting, or the object moving toward the edge.
- Item 9. The command refuses a missing bolt or a missing square. It does not refuse a low-trust outline, and there is no count of refused photos.
- `requirements.txt` is missing. The weights file is 176 MB, so GitHub will not store it. Download it from https://drive.google.com/file/d/1-PsKevRXKI-9Jyr2yQLZASatWYW5CLN8/view?usp=sharing into `models/maskrcnn_bolt.pth`. SHA-256 is in `models/config.json`.
