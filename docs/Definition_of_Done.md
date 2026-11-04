# Definition of Done

Written before implementation. A line is done only when the evidence beside it exists. A tick with no number, path, or commit link stays open.

- [x] Reprojection error reported, with the calibration image count. Evidence: 1.2318 px on 40 frames, `calibration/camera_params.npz`.
- [x] Camera matrix and distortion coefficients load from a saved file. Evidence: `inference/measure.py` `load_camera` reads `calibration/camera_params.npz`.
- [x] A straight edge in a calibration image stays straight after undistortion. Evidence: `calibration/check_straight.py`. Mean distance of a board row from a straight line, 20 views: 1.0376 px raw, 0.9417 px undistorted. `calib_005.png` falls from 2.4956 px to 0.4414 px. Image: `calibration/undistort_check/calib_005_raw_and_undistorted.png`.
- [ ] 70 or more images taken with this microscope, undistorted, with bolt and washer masks. Evidence: 72 images and 72 bolt masks in `dataset/images` and `dataset/labels`. There is no washer. The reference is the 3.0 mm square. A separate undistorted copy of the dataset was not saved.
- [ ] Train / val / test file list committed before training. Split 70 / 20 / 10, seed 42. Evidence: 51 / 15 / 6 images, which is 70.8% / 20.8% / 8.3% of the 72 photos. The 70/20/10 figure is that ratio, not a count of 70, 20, and 10 images. Git history does not show the lists committed before the training code.
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

- Washer masks. The plan uses the ChArUco square instead.
- Proof in git that the split lists were committed before the training code. The ratio of the 72 photos is 70.8% / 20.8% / 8.3%.
- The branch is `main`. The assessment asks for `dev-test01` and a pull request into `main`. The existing history cannot be reordered without rewriting commits that are already on GitHub.
- MO-1 is still missed after the clipped-outline rule. The 16 photos the command answers have a mean absolute error of 0.2663 mm.
- The weights file is 176 MB, so GitHub will not store it. Download it from https://drive.google.com/file/d/1-PsKevRXKI-9Jyr2yQLZASatWYW5CLN8/view?usp=sharing into `models/maskrcnn_bolt.pth`. SHA-256 is in `models/config.json`.
