"""Reusable microscope window.

Take a picture, measure that picture in millimetres, then return to the live
view to show the vernier. Stay open and repeat.

Run:
    python inference/studio.py
"""

import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk

from measure import Measurer

ROOT = Path(__file__).resolve().parent.parent
CAPTURE_DIR = ROOT / "inference" / "captures"
LOG_PATH = CAPTURE_DIR / "measurements.jsonl"


def microscope_index():
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
        capture_output=True,
    )
    text = (proc.stderr + proc.stdout).decode("utf-8", errors="replace")
    names = re.findall(r'"([^"]+)" \(video\)', text)
    for index, name in enumerate(names):
        if "microscope" in name.lower():
            return index
    return None


def open_microscope():
    index = microscope_index()
    if index is None:
        raise RuntimeError("Plug in the Digital Microscope and close Windows Camera, then press Live camera.")
    # This OpenCV build only opens DirectShow when the backend is added to the index.
    cap = cv2.VideoCapture(index + cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Digital Microscope did not open. Close any app using it, then press Live camera.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError("Microscope did not return a frame. Press Live camera to try again.")
    return cap


class Studio:
    def __init__(self):
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        self.cap = None
        self.measurer = None
        self.live = True
        self.frozen = None
        self.capture_path = None
        self.alive = True
        self.ticking = False
        self.root = tk.Tk()
        self.root.title("Measure the bolt")
        self.root.configure(bg="#111111")
        self.root.geometry("980x860+40+20")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.prompt = tk.Label(
            self.root,
            text="Put the bolt and the 3 mm card in view, then press Take picture.",
            font=("Segoe UI", 16),
            fg="white",
            bg="#111111",
            wraplength=920,
        )
        self.prompt.pack(pady=(12, 4))
        self.result = tk.Label(
            self.root,
            text="No measurement yet",
            font=("Segoe UI", 28, "bold"),
            fg="#8d8d8d",
            bg="#111111",
        )
        self.result.pack(pady=(0, 8))
        self.image_label = tk.Label(self.root, bg="#111111")
        self.image_label.pack()
        buttons = tk.Frame(self.root, bg="#111111")
        buttons.pack(pady=10)
        self.take_button = tk.Button(buttons, text="Take picture", font=("Segoe UI", 16), width=16, command=self.take_picture)
        self.measure_button = tk.Button(buttons, text="Measure", font=("Segoe UI", 16), width=16, command=self.measure)
        self.live_button = tk.Button(buttons, text="Live camera", font=("Segoe UI", 16), width=16, command=self.show_live)
        self.take_button.pack(side="left", padx=8)
        self.measure_button.pack(side="left", padx=8)
        self.live_button.pack(side="left", padx=8)
        self.history = tk.Listbox(self.root, font=("Consolas", 12), height=6, bg="#1c1c1c", fg="white")
        self.history.pack(fill="x", padx=16, pady=(0, 12))
        self.root.after(100, self.start)

    def start(self):
        self.prompt.configure(text="Loading the bolt model…")
        self.root.update_idletasks()
        try:
            if self.measurer is None:
                self.measurer = Measurer()
            if self.cap is None:
                self.cap = open_microscope()
        except Exception as exc:
            self.prompt.configure(text=str(exc) + " Press Live camera to try again.")
            self.result.configure(text="Camera not open", fg="#ff6b6b")
            return
        self.prompt.configure(text="Put the bolt and the 3 mm card in view, then press Take picture.")
        self.result.configure(text="No measurement yet", fg="#8d8d8d")
        if not self.ticking:
            self.ticking = True
            self.tick()

    def tick(self):
        if not self.alive or self.cap is None:
            return
        if self.live:
            for _ in range(2):
                self.cap.grab()
            ok, frame = self.cap.read()
            if ok and frame is not None:
                self.show_frame(frame)
        self.root.after(30, self.tick)

    def show_frame(self, frame):
        preview = cv2.resize(frame, (900, 506))
        photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)))
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def take_picture(self):
        if self.cap is None:
            self.prompt.configure(text="The microscope is not open.")
            return
        for _ in range(3):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.prompt.configure(text="The camera did not return a frame. Press Take picture again.")
            return
        self.frozen = frame.copy()
        self.live = False
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.capture_path = CAPTURE_DIR / f"shot_{stamp}.png"
        cv2.imwrite(str(self.capture_path), self.frozen)
        self.show_frame(self.frozen)
        self.result.configure(text="Picture taken", fg="#ffd166")
        self.prompt.configure(text="Press Measure. Or press Live camera if you want to show the caliper instead.")

    def measure(self):
        if self.frozen is None or self.measurer is None:
            self.prompt.configure(text="Take a picture first. The bolt and the 3 mm card both need to be in it.")
            return
        self.prompt.configure(text="Measuring…")
        self.root.update_idletasks()
        result = self.measurer.measure_bgr(self.frozen)
        if not result["ok"]:
            self.result.configure(text=result["reason"], fg="#ff6b6b")
            self.prompt.configure(text="No measurement. Fix the view, press Live camera, then Take picture again.")
            self.history.insert(0, f"{self.capture_path.name}  {result['reason']}")
            return
        overlay_path = self.capture_path.with_name(self.capture_path.stem + "_overlay.png")
        cv2.imwrite(str(overlay_path), result["overlay"])
        self.show_frame(result["overlay"])
        width = result["width_mm"]
        height = result["height_mm"]
        confidence = result["confidence"]
        self.result.configure(text=f"{width:.2f} mm    {height:.2f} mm", fg="#7dffb3")
        self.prompt.configure(
            text=f"Confidence {confidence:.2f}. Press Live camera and hold the caliper under the microscope. Then Take picture for the next bolt."
        )
        record = {
            "file": str(self.capture_path.relative_to(ROOT)).replace("\\", "/"),
            "overlay": str(overlay_path.relative_to(ROOT)).replace("\\", "/"),
            "width_mm": round(width, 3),
            "height_mm": round(height, 3),
            "confidence": round(confidence, 4),
        }
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        self.history.insert(0, f"{self.capture_path.name}  {width:.2f} mm  conf {confidence:.2f}")

    def show_live(self):
        if self.cap is None:
            self.start()
            return
        self.live = True
        self.prompt.configure(text="Live camera. Hold the vernier under the microscope so the reading is on screen.")
        self.result.configure(text="Live", fg="white")

    def close(self):
        self.alive = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    Studio().run()
