# Objectives

Hardware for every number below: Intel Core i7-10870H, 16 GB RAM, NVIDIA GeForce RTX 3070 Laptop (8 GB), Windows, digital microscope on its stand. Image resolution is written here once the capture size is locked. Each reported number is raw output from this machine.

## MO-1

| Field | Entry |
|---|---|
| What is measured | Mean absolute error, in millimetres, of the reported flat-to-flat size. Width and height are both that distance, because the vernier was placed across two opposite sides of the hexagon. |
| How | A script compares the command output for each photo with a hand-measured caliper file. The table lists every instance, then the mean, the median, and the worst case. |
| Target | Mean absolute error at or below 0.25 mm, over at least 10 held-out photos. The three bolts are 6 mm, 7 mm, and 8 mm across the flats, so 0.25 mm keeps those sizes distinct. |
| Conditions | Locked microscope, 1280×720. Hexagon face toward the camera, with one 3.0 mm board square in the same photo. The same photos every run. |
| Not included | Frames where the bolt or the square is cut by the image border, and any frame taken after focus, zoom, or column height changed. |

## MO-2

| Field | Entry |
|---|---|
| What is measured | Mean intersection-over-union of the predicted bolt mask against the labelled bolt mask. |
| How | A script scores every image in the committed test split. The decision threshold is chosen on the validation split only. |
| Target | Mean bolt IoU at or above 0.90 on the full test split. A 0.25 mm measurement on a bolt a few millimetres across needs the outline on the true edge, including the thread. |
| Conditions | Undistorted images from the locked camera. Threshold fixed from validation before the test split is scored. |
| Not included | Images with the bolt cut by the frame, and any threshold tried after seeing test results. |
