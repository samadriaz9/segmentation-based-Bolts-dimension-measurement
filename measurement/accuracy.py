"""Compare the measure command with the vernier sizes.

The vernier was read once per bolt: 6 mm, 7 mm, and 8 mm across the flats.
Those three readings are measurement/caliper.csv. Each photo of a bolt uses
that bolt's reading. The photos are the validation and test lists, which were
not used to train the weights.

The same photo is measured twice: once through the command, which undistorts,
and once with undistortion skipped. A missing bolt or a missing square is a
refusal, not a zero.

Run from the repository root:
    python measurement/accuracy.py
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))

from measure import Measurer

DATASET = ROOT / "dataset"
CALIPER_PATH = Path(__file__).resolve().parent / "caliper.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "raw_outputs.json"
LISTS = ("val.txt", "test.txt")


def caliper_by_bolt():
    table = {}
    with CALIPER_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            table[row["bolt"]] = float(row["caliper_flat_to_flat_mm"])
    return table


def held_out_images():
    paths = []
    for name in LISTS:
        for line in (DATASET / name).read_text(encoding="utf-8").splitlines():
            relative = line.strip()
            if relative:
                paths.append(DATASET / relative)
    return paths


def bolt_name(path):
    return path.name.split("_")[0]


def cut_by_border(path):
    label = DATASET / "labels" / f"{path.stem}.txt"
    parts = label.read_text(encoding="utf-8").split()
    coords = [float(value) for value in parts[1:]]
    xs, ys = coords[0::2], coords[1::2]
    edge = min(min(xs), min(ys), 1 - max(xs), 1 - max(ys))
    return edge < 0.01


def record(measurer, path, truth, apply_undistort):
    image = cv2.imread(str(path))
    if image is None:
        return {"ok": False, "reason": "unreadable image"}
    result = measurer.measure_bgr(image, apply_undistort=apply_undistort, refuse_clipped=False)
    row = {
        "file": path.name,
        "caliper_mm": truth,
        "ok": bool(result["ok"]),
    }
    if not result["ok"]:
        row["reason"] = result["reason"]
        row["clipped"] = bool(result.get("clipped", False))
        return row
    width = float(result["width_mm"])
    error = width - truth
    row.update({
        "width_mm": round(width, 4),
        "height_mm": round(float(result["height_mm"]), 4),
        "confidence": round(float(result["confidence"]), 4),
        "square_px": round(float(result["square_px"]), 4),
        "error_mm": round(error, 4),
        "abs_error_mm": round(abs(error), 4),
        "abs_pct": round(abs(error) / truth * 100.0, 4),
        "clipped": bool(result.get("clipped", False)),
    })
    return row


def summarise(rows):
    errors = [row["abs_error_mm"] for row in rows if row["ok"]]
    percents = [row["abs_pct"] for row in rows if row["ok"]]
    if not errors:
        return {"measured": 0, "refused": len(rows)}
    ordered = np.sort(np.array(errors, dtype=np.float64))
    return {
        "measured": len(errors),
        "refused": len(rows) - len(errors),
        "mean_abs_mm": round(float(ordered.mean()), 4),
        "median_abs_mm": round(float(np.median(ordered)), 4),
        "worst_abs_mm": round(float(ordered.max()), 4),
        "mean_abs_pct": round(float(np.mean(percents)), 4),
    }


def main():
    truth = caliper_by_bolt()
    paths = held_out_images()
    test_names = {
        Path(line.strip()).name
        for line in (DATASET / "test.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    measurer = Measurer()
    undistorted = []
    raw = []
    for index, path in enumerate(paths, start=1):
        size = truth[bolt_name(path)]
        corrected = record(measurer, path, size, True)
        skipped = record(measurer, path, size, False)
        corrected["split"] = "test" if path.name in test_names else "val"
        skipped["split"] = corrected["split"]
        corrected["cut_by_border"] = cut_by_border(path)
        skipped["cut_by_border"] = corrected["cut_by_border"]
        undistorted.append(corrected)
        raw.append(skipped)
        if corrected["ok"]:
            print(
                f"{index}/{len(paths)} {path.name} caliper {size:.1f} "
                f"undistort {corrected['width_mm']:.3f} raw {skipped.get('width_mm', skipped.get('reason'))}",
                flush=True,
            )
        else:
            print(f"{index}/{len(paths)} {path.name} {corrected['reason']}", flush=True)
    payload = {
        "caliper_file": "measurement/caliper.csv",
        "photos": [path.name for path in paths],
        "undistorted": undistorted,
        "raw": raw,
        "summary_undistorted": summarise(undistorted),
        "summary_raw": summarise(raw),
        "summary_test_undistorted": summarise([row for row in undistorted if row["split"] == "test"]),
        "summary_undistorted_in_frame": summarise([row for row in undistorted if not row["cut_by_border"]]),
        "summary_raw_in_frame": summarise([row for row in raw if not row["cut_by_border"]]),
        "summary_unclipped": summarise([row for row in undistorted if not row.get("clipped")]),
        "clipped_count": sum(1 for row in undistorted if row.get("clipped")),
        "border_margin_px": 2,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "undistorted": payload["summary_undistorted"],
        "raw": payload["summary_raw"],
        "test_undistorted": payload["summary_test_undistorted"],
    }, indent=2))


if __name__ == "__main__":
    main()
