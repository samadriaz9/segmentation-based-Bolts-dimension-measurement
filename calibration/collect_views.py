"""Live ChArUco capture. A frame is saved only when its corners fit the board."""

import json
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
import winsound
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "images"
BOARD_PATH = ROOT / "boards" / "charuco_3.0mm.json"
TARGET = 20
MIN_CORNERS = 6
MIN_LAPLACIAN = 25.0
MIN_PEAK = 180
MAX_FIT_MEAN = 1.00
MAX_FIT_WORST = 2.00
MIN_SHIFT_PX = 40.0
SETTLE_PX = 18.0
MIN_TILT_DEG = 15.0
PER_DIRECTION = 3
DIRECTIONS = ("left", "right", "top", "bottom")
PRIOR_K = np.array([[3100.0, 0.0, 640.0], [0.0, 3100.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_board():
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
    return board, detector


def saved_count():
    return len(list(IMAGE_DIR.glob("calib_*.png")))


def next_path():
    numbers = []
    for path in IMAGE_DIR.glob("calib_*.png"):
        try:
            numbers.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return IMAGE_DIR / f"calib_{max(numbers, default=0) + 1:03d}.png"


def open_microscope():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Digital Microscope did not open. Close any app using it.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = cap.read()
    if not ok or frame is None or frame.shape[1] != 1280:
        cap.release()
        raise RuntimeError("Microscope did not return a 1280x720 frame.")
    if cap.get(cv2.CAP_PROP_BRIGHTNESS) >= 0:
        cap.release()
        raise RuntimeError("Opened a different camera. Plug in the microscope and retry.")
    return cap


def centroids(marker_corners, marker_ids):
    if marker_ids is None or len(marker_ids) == 0:
        return {}
    points = {}
    for corners, marker_id in zip(marker_corners, marker_ids.flatten()):
        points[int(marker_id)] = corners.reshape(4, 2).mean(axis=0)
    return points


def mean_shift(previous, current):
    shared = [float(np.linalg.norm(previous[key] - current[key])) for key in previous.keys() & current.keys()]
    if not shared:
        return 1e9
    return float(np.mean(shared))


def homography_error(obj, img):
    board_xy = obj.reshape(-1, 3)[:, :2].astype(np.float32)
    pixels = img.reshape(-1, 2).astype(np.float32)
    if len(board_xy) < 4:
        return None
    homography, _ = cv2.findHomography(board_xy, pixels)
    if homography is None:
        return None
    projected = cv2.perspectiveTransform(board_xy.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error = np.linalg.norm(pixels - projected, axis=1)
    return float(error.mean()), float(error.max())


def tilt_pose(obj, img, camera_matrix, dist_coeffs):
    points = obj.reshape(-1, 3)
    if len(points) < 4:
        return None
    matrix = PRIOR_K if camera_matrix is None else camera_matrix
    distortion = np.zeros(5) if dist_coeffs is None else dist_coeffs
    flag = cv2.SOLVEPNP_ITERATIVE if len(points) >= 6 else cv2.SOLVEPNP_SQPNP
    ok, rvec, _ = cv2.solvePnP(
        points.reshape(-1, 1, 3),
        img.reshape(-1, 1, 2),
        matrix,
        distortion,
        flags=flag,
    )
    if not ok:
        return None
    rotation, _ = cv2.Rodrigues(np.asarray(rvec).reshape(3, 1))
    normal = rotation[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(abs(float(normal[2])), 0.0, 1.0))))
    if abs(float(normal[0])) >= abs(float(normal[1])):
        direction = "right" if float(normal[0]) > 0 else "left"
    else:
        direction = "bottom" if float(normal[1]) > 0 else "top"
    return tilt, direction


def calibration_rms(object_points, image_points, image_size):
    if len(object_points) < 6:
        return None, None, None
    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return float(rms), camera_matrix, dist_coeffs


def next_direction(counts):
    missing = [name for name in DIRECTIONS if counts[name] < PER_DIRECTION]
    return missing[0] if missing else None


def main():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    board, detector = load_board()
    cap = open_microscope()

    session_target = len(DIRECTIONS) * PER_DIRECTION
    root = tk.Tk()
    root.title(f"Tilted  0 / {session_target}")
    root.attributes("-topmost", True)
    root.geometry("900x820+30+20")
    root.configure(bg="black")
    count_label = tk.Label(root, text=f"0 / {session_target}", font=("Segoe UI", 64, "bold"), fg="white", bg="black")
    count_label.pack(pady=(6, 0))
    fit_label = tk.Label(root, text="", font=("Segoe UI", 22, "bold"), fg="white", bg="black")
    fit_label.pack()
    detail_label = tk.Label(root, text="", font=("Segoe UI", 16), fg="white", bg="black")
    detail_label.pack(pady=(0, 6))
    image_label = tk.Label(root, bg="black")
    image_label.pack()

    state = {
        "count": saved_count(),
        "last_saved": {},
        "pending": None,
        "alive": True,
        "object_points": [],
        "image_points": [],
        "image_size": None,
        "rms": None,
        "camera_matrix": None,
        "dist_coeffs": None,
        "directions": {name: 0 for name in DIRECTIONS},
    }

    def finish():
        state["alive"] = False
        cap.release()
        root.destroy()

    def show(frame, count_text, fit_text, detail, fg):
        count_label.configure(text=count_text, fg=fg)
        fit_label.configure(text=fit_text, fg=fg)
        detail_label.configure(text=detail, fg="white")
        root.title(f"Calibration  {count_text}")
        preview = cv2.resize(frame, (820, 460))
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        image_label.configure(image=photo)
        image_label.image = photo

    def tick():
        if not state["alive"]:
            return
        try:
            step()
        except Exception as exc:
            print(f"ERROR frame skipped: {exc}", flush=True)
            fit_label.configure(text="Frame skipped", fg="#ffd166")
        if state["alive"]:
            root.after(30, tick)

    def step():
        for _ in range(2):
            cap.grab()
        ok, frame = cap.read()
        if not ok or frame is None:
            print("ERROR camera frame lost", flush=True)
            finish()
            return

        clean = frame.copy()
        gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
        state["image_size"] = (gray.shape[1], gray.shape[0])
        laplacian = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        peak = int(gray.max())
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        n_corners = 0 if ids is None else len(ids)
        n_markers = 0 if marker_ids is None else len(marker_ids)
        centers = centroids(marker_corners, marker_ids)
        if n_markers:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        fit = None
        pose = None
        obj = img = None
        if n_corners >= 4:
            obj, img = board.matchImagePoints(corners, ids)
            if obj is not None and len(obj) >= 4:
                fit = homography_error(obj, img)
                pose = tilt_pose(obj, img, state["camera_matrix"], state["dist_coeffs"])

        sharp = laplacian >= MIN_LAPLACIAN and peak >= MIN_PEAK
        fit_ok = fit is not None and fit[0] <= MAX_FIT_MEAN and fit[1] <= MAX_FIT_WORST
        enough = n_corners >= MIN_CORNERS and n_markers >= 6
        moved = mean_shift(state["last_saved"], centers) >= MIN_SHIFT_PX
        tilt = None if pose is None else pose[0]
        direction = None if pose is None else pose[1]
        wanted = next_direction(state["directions"])
        session_count = sum(state["directions"].values())
        score = "  ".join(f"{name[0].upper()}{state['directions'][name]}" for name in DIRECTIONS)
        tilt_text = "tilt --" if tilt is None else f"tilt {tilt:.0f} deg"
        count_text = f"{session_count} / {session_target}"
        fg = "#ff6b6b"
        ask = "Lift one edge of the card." if wanted is None else f"Lift the {wanted} edge."
        detail = f"{score}    {tilt_text}"

        if wanted is None:
            fit_text = "DONE"
            detail = "All four edges are done. Close this window."
            fg = "#5dff8a"
        elif not sharp:
            fit_text = "Too dark or soft"
            detail = ask
        elif not enough or fit is None:
            fit_text = f"Need 6 corners    have {n_corners}"
            detail = ask
        elif not fit_ok:
            fit_text = f"fit {fit[0]:.2f} px  too high"
            detail = "Tip the card a little less, so the markers stay sharp. " + ask
        elif tilt is None or tilt < MIN_TILT_DEG:
            fit_text = f"fit {fit[0]:.2f} px  tilt more"
            detail = f"Now {0 if tilt is None else tilt:.0f} deg. Need {MIN_TILT_DEG:.0f}. {ask}"
        elif state["directions"][direction] >= PER_DIRECTION:
            fit_text = f"{direction} edge is done"
            detail = ask
            fg = "#ffd166"
        elif not moved:
            fit_text = f"fit {fit[0]:.2f} px  {direction}"
            detail = "Shift the card a little, still tilted. " + ask
            fg = "#ffd166"
        elif state["pending"] is None or mean_shift(state["pending"], centers) > SETTLE_PX:
            state["pending"] = centers
            fit_text = f"fit {fit[0]:.2f} px  {direction}"
            detail = "Hold still."
            fg = "#8ecae6"
        else:
            path = next_path()
            cv2.imwrite(str(path), clean)
            state["last_saved"] = centers
            state["pending"] = None
            state["directions"][direction] += 1
            session_count = sum(state["directions"].values())
            count_text = f"{session_count} / {session_target}"
            score = "  ".join(f"{name[0].upper()}{state['directions'][name]}" for name in DIRECTIONS)
            fit_text = f"Saved {direction}"
            detail = score
            fg = "#5dff8a"
            winsound.Beep(880, 120)
            print(
                f"SAVED {session_count}/{session_target} {path.name} "
                f"fit={fit[0]:.3f} tilt={tilt:.1f} direction={direction}",
                flush=True,
            )
            if next_direction(state["directions"]) is None:
                winsound.Beep(1200, 180)
                winsound.Beep(1400, 220)
                print(f"DONE {session_count}/{session_target}", flush=True)

        show(frame, count_text, fit_text, detail, fg)

    root.protocol("WM_DELETE_WINDOW", finish)
    print(f"READY {state['count']}/{TARGET}", flush=True)
    root.after(30, tick)
    root.mainloop()
    print(f"STOPPED {state['count']}/{TARGET}", flush=True)


if __name__ == "__main__":
    main()
