# Computer Vision & Detection

Image Preprocessing → Architectural CV. This does **not** include geometry
cleanup, BIM semantics, 3D, or export. This package
runs independently and outputs exactly the JSON handoff.

## What's implemented

| Task | File | What it does |
|---|---|---|
| T2.1 Preprocessing | `pipeline/preprocessing.py` | Grayscale, CLAHE contrast normalization, denoise, deskew (via Hough line angle), adaptive binarization |
| T2.2 Perspective correction | `pipeline/preprocessing.py` | Automatic quadrilateral-boundary detection (`detect_document_corners`) + homography warp. Falls back to no-op on clean top-down plans (confidence < 0.5). Both paths run identical CLAHE/denoise processing. |
| T2.3 Wall segmentation | `pipeline/wall_detection.py` | Morphological line extraction + Hough line detection + collinear-segment merging → wall centerlines with confidence |
| T2.4 Door/window detection | `pipeline/opening_detection.py` | Finds gaps in wall runs, classifies by gap width into door/window |
| T2.5 Column/fixture detection | `pipeline/opening_detection.py` | Small roughly-square blobs outside wall runs; each carries a `fixture_class` label (`"column"` or `"fixture"`) based on size/aspect heuristic — low-confidence, flagged for human review |
| T2.5b Stair detection | `pipeline/stair_detection.py` | Hough-line-based grouping of closely-spaced parallel short segments (stair treads) → stair bounding boxes; both horizontal and vertical stair orientations supported; conservative confidence (≤ 0.55), flagged for human review |
| T2.6 OCR extraction | `pipeline/ocr_text.py` | `EasyOCR` text + bounding boxes — pure `pip install`, no separate system binary needed |
| T2.7 Text classification | `pipeline/ocr_text.py` | Labels each string as `room_name` / `dimension` / `identifier` / `unknown` |
| T2.8 Per-element confidence | (built into each detector) | Every wall/opening/text/stair carries a `confidence` field |

## Important, read this first

**These are classical computer-vision heuristics (OpenCV line/contour
detection), not trained ML models.** There was no way to train a real wall-
or door-segmentation model in this session. This code:

- Works well on clean, high-contrast CAD-exported floor plans (like the
  included sample).
- Will be noticeably less accurate on scanned or hand-drawn plans, and can
  false-positive on small OCR text as "columns" (see the sample output —
  the small purple boxes around some room-label letters are false column
  detections, a known limitation of the size/aspect-ratio heuristic).
- Stair detection in particular is prone to false positives on hatching,
  dense text blocks, and grid lines — all stair results are flagged for
  human review with deliberately low confidence scores.
- Is structured so a trained model is a **drop-in replacement**: keep the
  same input (`binary`/`gray` image) and output shape
  (`{id, centerline, thickness_px, confidence}` for walls;
  `{id, type, wall_id, position_px, width_px, confidence}` for openings)
  and nothing downstream needs to change.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> **Heads up:** `easyocr` pulls in PyTorch, so this install is a few
> hundred MB and can take a few minutes — that's the trade-off for not
> needing a separate system binary. If you'd rather have a much lighter
> install and don't mind the extra setup step, `pytesseract` +
> system-installed Tesseract is a smaller alternative; ping me if you'd
> like that version instead.

OCR uses **EasyOCR**, which is a pure `pip install` — no separate system
binary to download, install, or add to PATH. The first time it runs it
downloads its detection/recognition models (~100MB total) and caches them
in a local `ocr_models/` folder **inside this project** (not your user
home directory — this sidesteps Windows permission issues some setups hit
with the default `~/.EasyOCR` location). Every run after the first works
fully offline. If model download fails (no internet on first run, or the
`ocr_models/` folder can't be created) or EasyOCR isn't installed, OCR is
skipped gracefully with a one-line warning — wall/door/window detection is
unaffected either way.

## Run it

```bash
# Clean top-down CAD plan (perspective correction skipped automatically)
python3 run_detection.py sample/sample_floorplan.png --out results/

# Perspective-distorted photo (perspective correction applied)
python3 sample/generate_sample.py          # generates sample_floorplan_warped.png
python3 run_detection.py sample/sample_floorplan_warped.png --out results_warped/
```

This writes:
- `results/detection.json` — the CV output handoff (walls, doors, windows,
  columns, stairs, classified text, perspective-correction metadata)
- `results/overlay.png` — a visual overlay so you can sanity-check
  detections against the source image

**Overlay colour legend:**

| Colour | Element |
|---|---|
| Green line | High-confidence wall (≥ 0.5) |
| Orange line | Low-confidence wall (< 0.5) |
| Dark-brown dot | Door opening |
| Blue/cyan dot | Window opening |
| Purple box | Candidate column / fixture |
| Gold/amber box | Candidate stair region |

Run it on your own image the same way:

```bash
python3 run_detection.py /path/to/your_floorplan.png --out my_results/
```

Regenerate the synthetic test images any time with:

```bash
python3 sample/generate_sample.py
```

## Output shape (`detection.json`)

```json
{
  "source_image": "sample_floorplan.png",
  "deskew_angle_deg": 0.0,
  "perspective_correction": {
    "applied": false,
    "confidence": 0.0,
    "corners_px": null
  },
  "walls": [
    {"id": "wall_000", "centerline": [[x1,y1],[x2,y2]],
     "thickness_px": 8.0, "confidence": 0.91}
  ],
  "doors": [
    {"id": "door_001", "type": "door", "wall_id": "wall_003",
     "position_px": {"x":.., "y":..}, "width_px": 35.0, "confidence": 0.55}
  ],
  "windows": [ "... same shape as doors, type: \"window\" ..." ],
  "columns": [
    {"id": "column_004", "bbox_px": [x,y,w,h], "fixture_class": "column", "confidence": 0.35}
  ],
  "stairs": [
    {"id": "stair_000", "bbox_px": [x,y,w,h], "confidence": 0.4}
  ],
  "texts": [
    {"text": "BEDROOM 1", "bbox_px": [x,y,w,h],
     "kind": "room_name", "confidence": 0.6}
  ]
}
```

When `perspective_correction.applied` is `true`, all detections and the
overlay image are based on the homography-corrected (warped) image, not the
original photograph.

This is exactly the shape Team 3's geometry engine expects as input.
