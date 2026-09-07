"""
T2.4 Door/window detection, T2.5 Column/stair/fixture detection

Heuristic approach: openings show up as *gaps* in an otherwise-continuous
wall run, plus (for doors) a short arc/leaf stroke nearby. Without a trained
detector we can reliably flag "there is a gap here" (high confidence) but
door-vs-window classification is a lower-confidence heuristic based on gap
width, and gets explicitly flagged for human review at the confidence stage.
"""
import numpy as np
import cv2


DOOR_WIDTH_PX_RANGE = (18, 55)   # heuristic default before scale is known
WINDOW_WIDTH_PX_RANGE = (10, 40)


def detect_openings(walls: list, wall_mask: np.ndarray):
    """
    For each wall centerline, sample along it looking for gaps (runs of
    low mask density) which indicate an opening was cut into the wall.
    """
    openings = []
    opening_counter = 0

    for wall in walls:
        (x1, y1), (x2, y2) = wall["centerline"]
        length = np.hypot(x2 - x1, y2 - y1)
        if length < 20:
            continue

        n_samples = max(int(length), 10)
        xs = np.linspace(x1, x2, n_samples)
        ys = np.linspace(y1, y2, n_samples)

        density = []
        for x, y in zip(xs, ys):
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < wall_mask.shape[0] and 0 <= xi < wall_mask.shape[1]:
                # local window density around the point, perpendicular-agnostic
                y0, y1_ = max(0, yi - 2), min(wall_mask.shape[0], yi + 3)
                x0, x1_ = max(0, xi - 2), min(wall_mask.shape[1], xi + 3)
                patch = wall_mask[y0:y1_, x0:x1_]
                density.append(patch.mean() / 255.0)
            else:
                density.append(0.0)

        density = np.array(density)
        gap_mask = density < 0.15

        # find contiguous gap runs, ignoring the very ends (those are wall endpoints, not openings)
        in_gap = False
        start = 0
        margin = max(3, n_samples // 20)
        for i in range(margin, n_samples - margin):
            if gap_mask[i] and not in_gap:
                in_gap = True
                start = i
            elif not gap_mask[i] and in_gap:
                in_gap = False
                gap_len_samples = i - start
                gap_len_px = gap_len_samples * (length / n_samples)
                if gap_len_px >= 8:  # ignore tiny noise gaps
                    mid = (start + i) // 2
                    pos = (float(xs[mid]), float(ys[mid]))
                    opening_type, conf = _classify_gap(gap_len_px)
                    opening_counter += 1
                    openings.append({
                        "id": f"{opening_type}_{opening_counter:03d}",
                        "type": opening_type,
                        "wall_id": wall["id"],
                        "position_px": {"x": pos[0], "y": pos[1]},
                        "width_px": round(gap_len_px, 1),
                        "confidence": conf,
                    })

    return openings


def _classify_gap(width_px):
    """Very rough door-vs-window split by gap width. Flagged low-confidence
    since real classification needs door-swing-arc detection or a trained model."""
    if DOOR_WIDTH_PX_RANGE[0] <= width_px <= DOOR_WIDTH_PX_RANGE[1]:
        return "door", 0.55
    if WINDOW_WIDTH_PX_RANGE[0] <= width_px <= WINDOW_WIDTH_PX_RANGE[1]:
        return "window", 0.5
    # ambiguous width - lowest confidence, still flagged for human review
    return ("door" if width_px > 30 else "window"), 0.3


def detect_columns_and_fixtures(binary: np.ndarray, min_area=30, max_area=900):
    """
    T2.5 - small, roughly-square blobs that aren't part of long wall runs
    are flagged as candidate columns/fixtures. Coarse heuristic; a trained
    classifier should replace this for production-grade fixture typing.
    """
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h else 0
        if 0.6 <= aspect <= 1.6:  # roughly square -> more likely a column
            candidates.append({
                "id": f"column_{i:03d}",
                "bbox_px": [float(x), float(y), float(w), float(h)],
                "confidence": 0.35,
            })
    return candidates
