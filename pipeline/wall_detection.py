"""
T2.3 Wall segmentation
Preprocessed image -> wall pixel masks / centerlines with confidence

Classical-CV approach (no trained model available in this scaffold):
morphological extraction of long straight structures + Hough line
detection + merge of collinear/near-parallel segments into wall runs.
This works reasonably on clean CAD-exported floor plans; scanned/hand-drawn
plans will need a trained segmentation model (e.g. U-Net) swapped in here,
reading the same `binary` input and producing the same output contract.
"""
import cv2
import numpy as np


def detect_walls(binary: np.ndarray, min_len_px: int = 25):
    # Emphasize long straight structures (walls) over small noise/furniture
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    horiz = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
    vert = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    wall_mask = cv2.bitwise_or(horiz, vert)
    wall_mask = cv2.dilate(wall_mask, np.ones((3, 3), np.uint8), iterations=1)

    lines = cv2.HoughLinesP(
        wall_mask, 1, np.pi / 360, threshold=30,
        minLineLength=min_len_px, maxLineGap=8
    )

    segments = []
    if lines is not None:
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = map(float, l)
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < min_len_px:
                continue
            segments.append((x1, y1, x2, y2, length))

    merged = _merge_collinear(segments)

    walls = []
    for i, (x1, y1, x2, y2, length) in enumerate(merged):
        thickness_px = _estimate_thickness(wall_mask, (x1, y1), (x2, y2))
        # confidence: longer + thicker + well-supported by mask density = higher
        density = _line_support(wall_mask, (x1, y1), (x2, y2))
        confidence = float(np.clip(0.4 + 0.3 * density + min(length / 300, 0.3), 0, 0.99))
        walls.append({
            "id": f"wall_{i:03d}",
            "centerline": [[x1, y1], [x2, y2]],
            "thickness_px": thickness_px,
            "confidence": round(confidence, 3),
        })

    return walls, wall_mask


def _line_support(mask, p1, p2, samples=20):
    x1, y1 = p1
    x2, y2 = p2
    hits = 0
    for t in np.linspace(0, 1, samples):
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] > 0:
            hits += 1
    return hits / samples


def _estimate_thickness(mask, p1, p2, max_probe=30):
    """Probe perpendicular to the line to estimate wall thickness in px."""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy) or 1
    nx, ny = -dy / length, dx / length  # perpendicular unit vector

    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    thickness = 4  # fallback default
    for probe_len in range(1, max_probe):
        px1 = int(mx + nx * probe_len)
        py1 = int(my + ny * probe_len)
        px2 = int(mx - nx * probe_len)
        py2 = int(my - ny * probe_len)
        in_bounds = (0 <= py1 < mask.shape[0] and 0 <= px1 < mask.shape[1] and
                     0 <= py2 < mask.shape[0] and 0 <= px2 < mask.shape[1])
        if not in_bounds:
            break
        if mask[py1, px1] == 0 and mask[py2, px2] == 0:
            thickness = probe_len * 2
            break
    return max(thickness, 4)


def _merge_collinear(segments, angle_tol_deg=6, dist_tol=10):
    """Merge segments that are roughly collinear and close together into single wall runs."""
    if not segments:
        return []

    def angle(seg):
        x1, y1, x2, y2, _ = seg
        return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180

    used = [False] * len(segments)
    merged = []

    for i, seg in enumerate(segments):
        if used[i]:
            continue
        group = [seg]
        used[i] = True
        a_i = angle(seg)
        for j in range(i + 1, len(segments)):
            if used[j]:
                continue
            seg_j = segments[j]
            a_j = angle(seg_j)
            if min(abs(a_i - a_j), 180 - abs(a_i - a_j)) > angle_tol_deg:
                continue
            if _min_endpoint_dist(seg, seg_j) < dist_tol * 4:
                group.append(seg_j)
                used[j] = True

        merged.append(_fit_line_to_group(group))

    return merged


def _min_endpoint_dist(a, b):
    pa = [(a[0], a[1]), (a[2], a[3])]
    pb = [(b[0], b[1]), (b[2], b[3])]
    return min(np.hypot(x1 - x2, y1 - y2) for x1, y1 in pa for x2, y2 in pb)


def _fit_line_to_group(group):
    pts = []
    for x1, y1, x2, y2, _ in group:
        pts.append((x1, y1))
        pts.append((x2, y2))
    pts = np.array(pts)
    # Use extreme points along the principal axis as the merged endpoints
    mean = pts.mean(axis=0)
    centered = pts - mean
    _, _, vt = np.linalg.svd(centered)
    direction = vt[0]
    proj = centered @ direction
    p_min = pts[np.argmin(proj)]
    p_max = pts[np.argmax(proj)]
    length = float(np.hypot(*(p_max - p_min)))
    return (float(p_min[0]), float(p_min[1]), float(p_max[0]), float(p_max[1]), length)
