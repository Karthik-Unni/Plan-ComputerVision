"""
T2.5b Stair detection

Classical-CV heuristic — no trained stair detector is available in this
scaffold. Stairs appear on floor plans as a series of closely-spaced,
roughly-parallel short line segments (stair treads) packed within a
rectangular bounding region. We detect them with Hough line detection
filtered for near-horizontal or near-vertical short segments, then group
co-directional segments that are regularly spaced into candidate stair
bounding boxes.

Known limitations:
  - Dense text blocks, hatching, and grid lines can trigger false positives
    — all stair detections are flagged for human review (confidence ≤ 0.55).
  - Only axis-aligned stair runs are detected; diagonal staircases are not.
  - Confidence is deliberately conservative: it scales with the number of
    parallel treads found, capped at 0.55 so downstream consumers know this
    needs human validation.
"""
import cv2
import numpy as np

# Minimum number of parallel tread-like segments to declare a stair region
_MIN_TREADS = 4
# Max spread (px) across which grouped treads may span (perpendicular to treads)
_MAX_GROUP_SPAN = 150
# Coefficient of variation threshold for spacing regularity (lower = more regular)
_SPACING_CV_THRESHOLD = 0.45
# Hough parameters tuned for short tread segments
_HOUGH_THRESHOLD = 20
_MIN_LINE_LEN = 12
_MAX_LINE_GAP = 4
# Tolerance (degrees) for classifying a segment as near-horizontal or near-vertical
_AXIS_TOL_DEG = 12


def detect_stairs(binary: np.ndarray):
    """
    Detect candidate stair regions in a binarized floor-plan image.

    Parameters
    ----------
    binary : np.ndarray
        Binarized (uint8, 0/255) image from the preprocessing stage.

    Returns
    -------
    list of dict, each with keys:
        id          : str   e.g. "stair_000"
        bbox_px     : list  [x, y, w, h] bounding rect of the tread group
        confidence  : float conservative score in [0, 0.55]
    """
    lines = cv2.HoughLinesP(
        binary, 1, np.pi / 180,
        threshold=_HOUGH_THRESHOLD,
        minLineLength=_MIN_LINE_LEN,
        maxLineGap=_MAX_LINE_GAP,
    )
    if lines is None:
        return []

    horiz_segs = []  # segments near-horizontal → treads run left-right
    vert_segs = []   # segments near-vertical   → treads run top-bottom

    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = np.hypot(dx, dy)
        if length < _MIN_LINE_LEN:
            continue
        angle_deg = abs(np.degrees(np.arctan2(dy, dx)))  # 0–90 after abs
        # Fold reflex angles so we always have 0–90
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        if angle_deg <= _AXIS_TOL_DEG:
            # Near-horizontal segment: primary axis is X, grouping axis is Y
            mid_y = (y1 + y2) / 2.0
            horiz_segs.append((x1, y1, x2, y2, mid_y))
        elif angle_deg >= (90 - _AXIS_TOL_DEG):
            # Near-vertical segment: primary axis is Y, grouping axis is X
            mid_x = (x1 + x2) / 2.0
            vert_segs.append((x1, y1, x2, y2, mid_x))

    results = []
    counter = 0

    # Group horizontal treads by Y position, vertical treads by X position
    for segs, axis_label in [(horiz_segs, "h"), (vert_segs, "v")]:
        groups = _group_parallel_segments(segs)
        for group in groups:
            n = len(group)
            if n < _MIN_TREADS:
                continue

            # Bounding rect of all segments in the group
            all_x = [pt for seg in group for pt in (seg[0], seg[2])]
            all_y = [pt for seg in group for pt in (seg[1], seg[3])]
            bx = int(min(all_x))
            by = int(min(all_y))
            bw = int(max(all_x)) - bx
            bh = int(max(all_y)) - by

            # Conservative confidence: more treads → higher score, capped at 0.55
            confidence = round(min(0.1 * n, 0.55), 3)

            results.append({
                "id": f"stair_{counter:03d}",
                "bbox_px": [float(bx), float(by), float(bw), float(bh)],
                "confidence": confidence,
            })
            counter += 1

    return results


def _group_parallel_segments(segs):
    """
    Cluster co-directional segments into stair-region candidates.

    Two-pass approach:
      Pass 1 — Deduplicate: HoughLinesP often returns several slightly
        different line detections for the same physical tread (mid_y values
        within a few px of each other). These are binned together so a single
        tread counts as one, not as a cluster of 0-spacing pairs that would
        blow up the CV check.
      Pass 2 — Group bins whose overall span is ≤ _MAX_GROUP_SPAN and
        whose inter-bin spacing has CV < _SPACING_CV_THRESHOLD.

    Each seg tuple is (x1, y1, x2, y2, grouping_axis_midpoint).
    """
    if not segs:
        return []

    DEDUP_EPSILON = 5  # px — mid_y values within this distance = same tread

    segs_sorted = sorted(segs, key=lambda s: s[4])

    # Pass 1: bin segments into discrete trads by deduplicating close mid_y
    bins = []   # list of (representative_mid, [segs])
    for seg in segs_sorted:
        m = seg[4]
        if bins and abs(m - bins[-1][0]) <= DEDUP_EPSILON:
            bins[-1][1].append(seg)
        else:
            bins.append([m, [seg]])

    if len(bins) < _MIN_TREADS:
        return []

    # Pass 2: slide a window over tread-bins to find qualifying stair runs
    groups = []
    n = len(bins)
    for start in range(n):
        anchor_mid = bins[start][0]
        group_bins = [bins[start]]

        for end in range(start + 1, n):
            if bins[end][0] - anchor_mid > _MAX_GROUP_SPAN:
                break
            group_bins.append(bins[end])

        if len(group_bins) < _MIN_TREADS:
            continue

        # Check spacing regularity across the tread-bin centroids
        mids = [b[0] for b in group_bins]
        spacings = [mids[k + 1] - mids[k] for k in range(len(mids) - 1)]
        mean_sp = np.mean(spacings)
        if mean_sp == 0:
            continue
        cv = np.std(spacings) / mean_sp
        if cv < _SPACING_CV_THRESHOLD:
            # Flatten all segments from the qualifying bins
            all_segs = [s for b in group_bins for s in b[1]]
            groups.append(all_segs)
            break  # take the longest qualifying run from this start; move on

    return groups
