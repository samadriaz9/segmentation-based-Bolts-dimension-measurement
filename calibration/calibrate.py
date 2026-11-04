"""Calibrate the microscope from calib_*.png and save the camera parameters."""

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
BOARD_PATH = ROOT / "boards" / "charuco_3.0mm.json"
IMAGE_DIR = ROOT / "images"
OUTPUT = ROOT / "camera_params.npz"
MIN_CORNERS = 6


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
    params.adaptiveThreshConstant = 5
    params.minMarkerPerimeterRate = 0.01
    params.errorCorrectionRate = 0.8
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board)
    detector.setDetectorParameters(params)
    return board, detector, meta


def main():
    board, detector, meta = load_detector()
    object_points = []
    image_points = []
    used = []
    image_size = None
    for path in sorted(IMAGE_DIR.glob("calib_*.png")):
        image = cv2.imread(str(path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        corners, ids, _, _ = detector.detectBoard(gray)
        if ids is None or len(ids) < MIN_CORNERS:
            print(f"skip {path.name}")
            continue
        obj, img = board.matchImagePoints(corners, ids)
        if obj is None or len(obj) < MIN_CORNERS:
            print(f"skip {path.name}")
            continue
        object_points.append(obj)
        image_points.append(img)
        used.append(path.name)

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    per_view = []
    for obj, img, rvec, tvec, name in zip(object_points, image_points, rvecs, tvecs, used):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        error = np.linalg.norm(img.reshape(-1, 2) - projected.reshape(-1, 2), axis=1).mean()
        per_view.append(float(error))
        print(f"{name} mean_error_px={error:.3f}")

    np.savez(
        OUTPUT,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error_px=float(rms),
        square_length_mm=float(meta["square_length_mm_measured"]),
        marker_length_mm=float(meta["marker_length_mm_measured"]),
        images=np.array(used),
    )
    print(f"views={len(used)} rms_px={float(rms):.4f} worst_px={max(per_view):.4f}")
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()
