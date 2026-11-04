"""Error against how close the labelled bolt sits to the frame.

Item 8 of the assessment. The condition is the bolt near the edge. Distance is
the smallest gap from the labelled outline to the image border, as a fraction
of the frame. The errors are the undistorted command outputs already saved in
raw_outputs.json.

Run:
    python measurement/edge_condition.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "dataset" / "labels"
OUTPUTS = Path(__file__).resolve().parent / "raw_outputs.json"
REPORT = Path(__file__).resolve().parent / "edge_condition.json"


def border_gap(name):
    parts = (LABELS / f"{Path(name).stem}.txt").read_text(encoding="utf-8").split()
    coords = [float(value) for value in parts[1:]]
    xs, ys = coords[0::2], coords[1::2]
    return min(min(xs), min(ys), 1 - max(xs), 1 - max(ys))


def band(gap):
    if gap < 0.01:
        return "touching the border"
    if gap < 0.05:
        return "within 5 percent of the border"
    return "clear of the border"


def mean(values):
    return round(sum(values) / len(values), 4)


def main():
    data = json.loads(OUTPUTS.read_text(encoding="utf-8"))
    rows = []
    for row in data["undistorted"]:
        gap = border_gap(row["file"])
        rows.append({
            "file": row["file"],
            "border_gap": round(gap, 4),
            "band": band(gap),
            "abs_error_mm": row["abs_error_mm"],
            "clipped_by_model": bool(row.get("clipped")),
        })
    rows.sort(key=lambda item: item["border_gap"])
    bands = []
    for name in ("touching the border", "within 5 percent of the border", "clear of the border"):
        chosen = [item["abs_error_mm"] for item in rows if item["band"] == name]
        if not chosen:
            continue
        bands.append({
            "band": name,
            "photos": len(chosen),
            "mean_abs_mm": mean(chosen),
            "worst_abs_mm": round(max(chosen), 4),
        })
    report = {"rows": rows, "bands": bands}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for item in bands:
        print(f"{item['band']}: n={item['photos']} mean_abs_mm={item['mean_abs_mm']} worst={item['worst_abs_mm']}")


if __name__ == "__main__":
    main()
