"""
Team 2 — Computer Vision & Detection — standalone runner.

Runs only the stages Team 2 owns:
  T2.1 Preprocessing
  T2.2 Perspective correction (automatic; falls back to no-op on clean plans)
  T2.3 Wall segmentation
  T2.4 Door/window detection
  T2.5 Column/fixture detection (with fixture_class typing)
  T2.5b Stair detection
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

from pipeline import preprocessing, wall_detection, opening_detection, ocr_text, stair_detection

# Confidence threshold for auto-applying perspective correction.
# Below this value the image is assumed to be already top-down (e.g. a
# clean CAD export) and no warp is applied, preserving existing behaviour.
PERSPECTIVE_CONFIDENCE_THRESHOLD = 0.5


def run(image_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # T2.1 — load, CLAHE, denoise, deskew, binarize
    pre = preprocessing.load_and_preprocess(image_path)
    gray, binary = pre["gray"], pre["binary"]

    # T2.2 — automatic perspective correction
    # Attempt to find the document boundary quad in the preprocessed gray image.
    corners, corner_conf = preprocessing.detect_document_corners(gray)

    if corners is not None and corner_conf >= PERSPECTIVE_CONFIDENCE_THRESHOLD:
        # Warp the original BGR image so the overlay is drawn on the same
        # image the detectors ran on.
        warped_bgr = preprocessing.correct_perspective(pre["original_bgr"], corners.tolist())
        # Re-derive gray from the warped BGR and run the full normalization
        # pipeline (CLAHE + denoise + binarize) again — photographed plans
        # need contrast/denoise at least as much as clean scans.
        warped_gray_raw = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        warped_gray, warped_binary = preprocessing._normalize_and_binarize(warped_gray_raw)
        # Deskew is intentionally skipped on the warped image: the homography
        # already corrects the dominant orientation, and re-running deskew on
        # the corrected image would chase residual noise rather than real skew.
        proc_bgr, proc_gray, proc_binary = warped_bgr, warped_gray, warped_binary
        persp_meta = {
            "applied": True,
            "confidence": round(float(corner_conf), 3),
            "corners_px": corners.tolist(),
        }
        print(f"Perspective correction: applied (confidence={persp_meta['confidence']:.3f})")
    else:
        proc_bgr = pre["original_bgr"]
        proc_gray, proc_binary = pre["gray"], pre["binary"]
        persp_meta = {"applied": False, "confidence": 0.0, "corners_px": None}
        print("Perspective correction: skipped (no confident quad found — top-down plan assumed)")

    # T2.3
    walls, wall_mask = wall_detection.detect_walls(proc_binary)

    # T2.4
    openings = opening_detection.detect_openings(walls, wall_mask)
    doors = [o for o in openings if o["type"] == "door"]
    windows = [o for o in openings if o["type"] == "window"]

    # T2.5 — columns/fixtures (now includes fixture_class field)
    columns = opening_detection.detect_columns_and_fixtures(proc_binary)

    # T2.5b — stair detection
    stairs = stair_detection.detect_stairs(proc_binary)

    # T2.6 / T2.7
    raw_texts = ocr_text.extract_text(proc_gray)
    texts = ocr_text.classify_texts(raw_texts)

    result = {
        "source_image": os.path.basename(image_path),
        "deskew_angle_deg": pre["deskew_angle_deg"],
        "perspective_correction": persp_meta,
        "walls": walls,
        "doors": doors,
        "windows": windows,
        "columns": columns,
        "stairs": stairs,
        "texts": texts,
    }

    json_path = os.path.join(out_dir, "detection.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    overlay_path = os.path.join(out_dir, "overlay.png")
    # Draw overlay on proc_bgr (warped image if correction was applied, or
    # the original — consistent with what the detectors actually ran on).
    _save_overlay(proc_bgr, walls, doors, windows, columns, stairs, overlay_path)

    print(f"Walls detected:   {len(walls)}")
    print(f"Doors detected:   {len(doors)}")
    print(f"Windows detected: {len(windows)}")
    print(f"Columns detected: {len(columns)}")
    print(f"Stairs detected:  {len(stairs)}")
    print(f"Text elements:    {len(texts)}")
    print(f"\nJSON written to:    {json_path}")
    print(f"Overlay written to: {overlay_path}")

    return result


def _save_overlay(bgr_img, walls, doors, windows, columns, stairs, out_path):
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
    for s in stairs:
        x, y, w_, h_ = s["bbox_px"]
        # BGR (0, 215, 255) → orange/gold — distinct from window blue (255,191,0)
        # and column purple (128,0,200).
        cv2.rectangle(img, (int(x), int(y)), (int(x + w_), int(y + h_)), (0, 215, 255), 2)
    cv2.imwrite(out_path, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Team 2's CV detection stage on a floor-plan image.")
    parser.add_argument("image", help="Path to the floor-plan image")
    parser.add_argument("--out", default="results", help="Output directory (default: results/)")
    args = parser.parse_args()
    run(args.image, args.out)
