"""
Team 2 — Computer Vision & Detection — standalone runner.

Runs only the stages Team 2 owns:
  T2.1 Preprocessing
  T2.2 Perspective correction (no-op unless corners are given)
  T2.3 Wall segmentation
  T2.4 Door/window detection
  T2.5 Column/fixture detection
  T2.6 OCR extraction
  T2.7 Text classification
  T2.8 Per-element confidence (already attached by each detector)

Output: a JSON file matching the "Architectural CV" handoff point in the
plan's pipeline (Section 2) — this is exactly what Team 3 (Geometry & BIM
Semantics) consumes next — plus a PNG overlay for a quick visual sanity check.

Usage:
    python3 run_detection.py path/to/floorplan.png
    python3 run_detection.py path/to/floorplan.png --out results/
"""
import argparse
import json
import os

import cv2
import numpy as np

from pipeline import preprocessing, wall_detection, opening_detection, ocr_text


def run(image_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # T2.1 / T2.2
    pre = preprocessing.load_and_preprocess(image_path)
    gray, binary = pre["gray"], pre["binary"]

    # T2.3
    walls, wall_mask = wall_detection.detect_walls(binary)

    # T2.4
    openings = opening_detection.detect_openings(walls, wall_mask)
    doors = [o for o in openings if o["type"] == "door"]
    windows = [o for o in openings if o["type"] == "window"]

    # T2.5
    columns = opening_detection.detect_columns_and_fixtures(binary)

    # T2.6 / T2.7
    raw_texts = ocr_text.extract_text(gray)
    texts = ocr_text.classify_texts(raw_texts)

    result = {
        "source_image": os.path.basename(image_path),
        "deskew_angle_deg": pre["deskew_angle_deg"],
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "columns": columns,
        "texts": texts,
    }

    json_path = os.path.join(out_dir, "detection.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    overlay_path = os.path.join(out_dir, "overlay.png")
    _save_overlay(pre["original_bgr"], walls, doors, windows, columns, overlay_path)

    print(f"Walls detected:   {len(walls)}")
    print(f"Doors detected:   {len(doors)}")
    print(f"Windows detected: {len(windows)}")
    print(f"Columns detected: {len(columns)}")
    print(f"Text elements:    {len(texts)}")
    print(f"\nJSON written to:    {json_path}")
    print(f"Overlay written to: {overlay_path}")

    return result


def _save_overlay(bgr_img, walls, doors, windows, columns, out_path):
    img = bgr_img.copy()
    for w in walls:
        (x1, y1), (x2, y2) = w["centerline"]
        color = (0, 200, 0) if w["confidence"] >= 0.5 else (0, 140, 255)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
    for o in doors:
        p = o["position_px"]
        cv2.circle(img, (int(p["x"]), int(p["y"])), 6, (19, 69, 139), -1)
    for o in windows:
        p = o["position_px"]
        cv2.circle(img, (int(p["x"]), int(p["y"])), 6, (255, 191, 0), -1)
    for c in columns:
        x, y, w_, h_ = c["bbox_px"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w_), int(y + h_)), (128, 0, 200), 2)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Team 2's CV detection stage on a floor-plan image.")
    parser.add_argument("image", help="Path to the floor-plan image")
    parser.add_argument("--out", default="results", help="Output directory (default: results/)")
    args = parser.parse_args()
    run(args.image, args.out)
