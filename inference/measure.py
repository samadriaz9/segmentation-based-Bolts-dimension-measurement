"""Measure one microscope photo in millimetres.

The photo is raw. The command undistorts it, finds the bolt, and scales the
flat-to-flat size from the 3.0 mm ChArUco square in that same photo. Width and
height are both that distance, because the vernier was placed across opposite
flats of the hexagon.

The score threshold was chosen on the validation split and is not changed here.
If the bolt is missing, or the square is missing, the command prints no measurement.

Run:
    python inference/measure.py path\\to\\photo.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

ROOT = Path(__file__).resolve().parent.parent
PARAMS_PATH = ROOT / "calibration" / "camera_params.npz"
BOARD_PATH = ROOT / "calibration" / "boards" / "charuco_3.0mm.json"
WEIGHTS_PATH = ROOT / "models" / "maskrcnn_bolt.pth"
OUTPUT_DIR = ROOT / "inference" / "outputs"
SCORE_THRESHOLD = 0.95
MASK_THRESHOLD = 0.5
SQUARE_MM = 3.0
MARKER_MM = 2.1
NUM_CLASSES = 2


def load_camera():
    data = np.load(PARAMS_PATH)
    return data["camera_matrix"], data["dist_coeffs"]


def undistort(image, camera_matrix, dist_coeffs):
    height, width = image.shape[:2]
    new_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 0)
    return cv2.undistort(image, camera_matrix, dist_coeffs, None, new_matrix)


def load_board():
    import json

    meta = json.loads(BOARD_PATH.read_text(encoding="utf-8"))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        (int(meta["squares_x"]), int(meta["squares_y"])),
        float(meta["square_length_mm_measured"]),
        float(meta["marker_length_mm_measured"]),
        dictionary,
    )
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 63
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.008
    params.errorCorrectionRate = 0.8
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.CharucoDetector(board)
    detector.setDetectorParameters(params)
    return board, detector


def square_pixels(board, corners, ids):
    if ids is None or len(ids) < 2:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    if obj is None:
        return None
    board_xy = obj.reshape(-1, 3)[:, :2]
    pixels = img.reshape(-1, 2)
    lengths = []
    for i in range(len(board_xy)):
        for j in range(i + 1, len(board_xy)):
            if abs(float(np.linalg.norm(board_xy[i] - board_xy[j])) - SQUARE_MM) < 0.05:
                lengths.append(float(np.linalg.norm(pixels[i] - pixels[j])))
    if len(lengths) < 2:
        return None
    return float(np.median(lengths))


def marker_square_pixels(marker_corners):
    if marker_corners is None or len(marker_corners) == 0:
        return None
    lengths = []
    for corners in marker_corners:
        points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        for index in range(4):
            lengths.append(float(np.linalg.norm(points[index] - points[(index + 1) % 4])))
    if len(lengths) < 4:
        return None
    return float(np.median(lengths)) * SQUARE_MM / MARKER_MM


def find_square_px(gray, board, detector):
    corners, ids, marker_corners, _marker_ids = detector.detectBoard(gray)
    measured = square_pixels(board, corners, ids)
    if measured is not None:
        return measured
    return marker_square_pixels(marker_corners)


def load_model(device):
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, NUM_CLASSES)
    model.transform.min_size = (720,)
    model.transform.max_size = 1280
    state = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def find_bolt(model, image, device):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255.0)
    with torch.no_grad():
        output = model([tensor.to(device)])[0]
    scores = output["scores"].detach().cpu().numpy()
    labels = output["labels"].detach().cpu().numpy()
    masks = output["masks"].detach().cpu().numpy()
    candidates = [
        (float(score), mask[0] >= MASK_THRESHOLD)
        for score, label, mask in zip(scores, labels, masks)
        if int(label) == 1 and float(score) >= SCORE_THRESHOLD
    ]
    if not candidates:
        return None
    score, mask = max(candidates, key=lambda item: item[0])
    binary = mask.astype(np.uint8)
    if int(binary.sum()) < 50:
        return None
    return score, binary


def flat_to_flat_px(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None
    contour = max(contours, key=cv2.contourArea)
    _center, size, _angle = cv2.minAreaRect(contour)
    width, height = size
    if min(width, height) < 5:
        return None, None
    return float(min(width, height)), contour


def pixels_to_mm(flats_px, square_px):
    """Turn the flat-to-flat pixel length into millimetres.

    Valid only when this photo still shows the measured 3.0 mm square, the
    camera focus and zoom are the ones that were calibrated, and the square
    lies in the same plane as the size being measured.
    """
    return flats_px * SQUARE_MM / square_px


def draw_overlay(image, contour, width_mm, height_mm, confidence):
    overlay = image.copy()
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2, lineType=cv2.LINE_AA)
    text = f"W {width_mm:.2f} mm   H {height_mm:.2f} mm   {confidence:.2f}"
    cv2.putText(overlay, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(overlay, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return overlay


def fail(message, code):
    print(message)
    return code


class Measurer:
    """Loads the camera model and the bolt model once, then measures many photos."""

    def __init__(self):
        self.camera_matrix, self.dist_coeffs = load_camera()
        self.board, self.detector = load_board()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.device)

    def measure_bgr(self, raw):
        image = undistort(raw, self.camera_matrix, self.dist_coeffs)
        found = find_bolt(self.model, image, self.device)
        if found is None:
            return {"ok": False, "reason": "object not found", "undistorted": image}
        confidence, mask = found
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        square_px = find_square_px(gray, self.board, self.detector)
        if square_px is None:
            return {"ok": False, "reason": "reference not found", "undistorted": image, "confidence": confidence}
        flats_px, contour = flat_to_flat_px(mask)
        if flats_px is None:
            return {"ok": False, "reason": "object not found", "undistorted": image}
        flats_mm = pixels_to_mm(flats_px, square_px)
        overlay = draw_overlay(image, contour, flats_mm, flats_mm, confidence)
        return {
            "ok": True,
            "width_mm": flats_mm,
            "height_mm": flats_mm,
            "confidence": confidence,
            "square_px": square_px,
            "overlay": overlay,
            "undistorted": image,
        }


def measure(image_path, output_path, measurer=None):
    raw = cv2.imread(str(image_path))
    if raw is None:
        return fail(f"unreadable image {image_path}", 1)
    result = (measurer or Measurer()).measure_bgr(raw)
    if not result["ok"]:
        return fail(result["reason"], 2 if result["reason"] == "object not found" else 3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result["overlay"])
    print(f"width_mm {result['width_mm']:.3f}")
    print(f"height_mm {result['height_mm']:.3f}")
    print(f"confidence {result['confidence']:.4f}")
    print(f"overlay {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Undistort one photo and measure the bolt in millimetres.")
    parser.add_argument("image", type=Path, help="Raw microscope photo.")
    parser.add_argument("--output", type=Path, default=None, help="Where to write the mask overlay.")
    args = parser.parse_args()
    output = args.output
    if output is None:
        output = OUTPUT_DIR / f"{args.image.stem}_overlay.png"
    sys.exit(measure(args.image, output))


if __name__ == "__main__":
    main()
