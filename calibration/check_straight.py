"""Measure whether a board row is straighter after undistortion.

A ChArUco row is a straight line on the paper. This fits a line to the detected
corners of each row and reports the mean distance to that line, in pixels,
before and after undistortion. It also saves one side-by-side frame.

Run:
    python calibration/check_straight.py
"""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BOARD_PATH = ROOT / "boards" / "charuco_3.0mm.json"
IMAGE_DIR = ROOT / "images"
PARAMS_PATH = ROOT / "camera_params.npz"
OUTPUT_DIR = ROOT / "undistort_check"


def load_detector():
    meta = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        (meta["squares_x"], meta["squares_y"]),
        float(meta["square_length_mm_measured"]),
        float(meta["marker_length_mm_measured"]),
        dictionary,
    )
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 63
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.errorCorrectionRate = 0.8
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board)
    detector.setDetectorParameters(params)
    return board, detector


def undistort(image, camera_matrix, dist_coeffs):
    height, width = image.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 0)
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)


def row_errors(board, detector, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < 4:
        return []
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None or len(obj) < 4:
        return []
    board_xy = obj.reshape(-1, 3)[:, :2]
    pixels = img.reshape(-1, 2)
    errors = []
    drawn = []
    for y_value in np.unique(np.round(board_xy[:, 1], 3)):
        chosen = np.where(np.abs(board_xy[:, 1] - y_value) < 0.05)[0]
        if len(chosen) < 4:
            continue
        points = pixels[chosen].astype(np.float32)
        direction_x, direction_y, origin_x, origin_y = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        offset = points - np.array([origin_x, origin_y], dtype=np.float32)
        distance = np.abs(offset[:, 0] * direction_y - offset[:, 1] * direction_x)
        errors.append(float(distance.mean()))
        drawn.append(points)
    return errors, drawn


def main():
    board, detector = load_detector()
    data = np.load(PARAMS_PATH)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    raw_errors = []
    fixed_errors = []
    best = None
    for path in sorted(IMAGE_DIR.glob("calib_*.png")):
        image = cv2.imread(str(path))
        before = row_errors(board, detector, image)
        if not before or not before[0]:
            continue
        corrected = undistort(image, camera_matrix, dist_coeffs)
        after = row_errors(board, detector, corrected)
        if not after or not after[0]:
            continue
        raw_mean = float(np.mean(before[0]))
        fixed_mean = float(np.mean(after[0]))
        raw_errors.append(raw_mean)
        fixed_errors.append(fixed_mean)
        gain = raw_mean - fixed_mean
        if best is None or gain > best["gain"]:
            best = {
                "name": path.name,
                "gain": gain,
                "raw_mean": raw_mean,
                "fixed_mean": fixed_mean,
                "image": image,
                "corrected": corrected,
                "raw_rows": before[1],
                "fixed_rows": after[1],
            }
        print(f"{path.name} raw_px {raw_mean:.4f} undistorted_px {fixed_mean:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = np.hstack([best["image"], best["corrected"]])
    for points in best["raw_rows"]:
        for point in points:
            cv2.circle(panel, tuple(np.round(point).astype(int)), 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    shift = best["image"].shape[1]
    for points in best["fixed_rows"]:
        for point in points:
            center = (int(round(point[0])) + shift, int(round(point[1])))
            cv2.circle(panel, center, 4, (0, 255, 0), -1, lineType=cv2.LINE_AA)
    out_path = OUTPUT_DIR / f"{Path(best['name']).stem}_raw_and_undistorted.png"
    cv2.imwrite(str(out_path), panel)
    summary = {
        "views": len(raw_errors),
        "mean_row_error_raw_px": round(float(np.mean(raw_errors)), 4),
        "mean_row_error_undistorted_px": round(float(np.mean(fixed_errors)), 4),
        "example": best["name"],
        "example_raw_px": round(best["raw_mean"], 4),
        "example_undistorted_px": round(best["fixed_mean"], 4),
        "image": str(out_path.relative_to(ROOT.parent)).replace("\\", "/"),
    }
    (OUTPUT_DIR / "straight_edge.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
