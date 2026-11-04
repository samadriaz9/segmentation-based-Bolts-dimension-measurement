# Definition of Done

Written before implementation. A line is done only when the evidence beside it exists. A tick with no number, path, or commit link stays open.

- [ ] Reprojection error reported, with the calibration image count. Evidence:
- [ ] Camera matrix and distortion coefficients load from a saved file. Evidence:
- [ ] A straight edge in a calibration image stays straight after undistortion. Evidence:
- [ ] 70 or more images taken with this microscope, undistorted, with bolt and washer masks. Evidence:
- [ ] Train / val / test file list committed before training. Split 70 / 20 / 10, seed 42. Evidence:
- [ ] Test split unused for training, tuning, and checkpoint selection. Evidence:
- [ ] Test metrics reported: mask IoU, mAP@0.5, mAP@0.5:0.95, precision, recall. Evidence:
- [ ] One command takes a new image and prints width (mm), height (mm), and confidence, and saves a mask overlay. Evidence:
- [ ] Object not found, and reference not found, each end with no measurement. Evidence:
- [ ] Accuracy table for at least 10 instances. Every instance, then mean, median, and worst case. Raw outputs saved. Evidence:
- [ ] The same instances measured on raw images and on undistorted images, errors side by side. Evidence:
- [ ] MO-1 met or missed, with the raw table. Evidence:
- [ ] MO-2 met or missed, with the raw test score. Evidence:
- [ ] Everything unfinished is listed here. Evidence:

Items 8 and 9 of the assessment stay off this list until the accuracy table exists.
