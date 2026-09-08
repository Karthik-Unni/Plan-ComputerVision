"""
T2.1 Preprocessing pipeline, T2.2 Perspective correction
Raw image -> deskewed, contrast-normalized, binarized image
"""
import cv2
import numpy as np


def _normalize_and_binarize(gray: np.ndarray):
    """
    Shared helper: CLAHE contrast normalization → denoise → adaptive binarize.

    Called by both the normal preprocessing path and the perspective-corrected
    path so photographed plans (the primary target of perspective correction)
    receive the same contrast/denoise treatment as clean scans — not less.
    """
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=7)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )
    return gray, binary


def load_and_preprocess(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast normalization + denoise + binarize (factored into helper so the
    # perspective-corrected path can reuse identical processing)
    gray, binary = _normalize_and_binarize(gray)

    # Deskew: find dominant line angle via Hough and rotate to correct it
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=120)
    angle_deg = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines.reshape(-1, 2)[:200]:
            deg = (theta * 180 / np.pi) - 90
            # fold into [-45, 45) so we correct toward nearest right angle
            deg = ((deg + 45) % 90) - 45
            angles.append(deg)
        if angles:
            angle_deg = float(np.median(angles))

    h, w = gray.shape
    if abs(angle_deg) > 0.3:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
        binary = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_REPLICATE)

    return {
        "original_bgr": img,
        "gray": gray,
        "binary": binary,
        "deskew_angle_deg": angle_deg,
    }


def detect_document_corners(gray: np.ndarray):
    """
    T2.2 - Automatically locate the four corners of a floor-plan document
    boundary in a perspective-distorted photograph.

    Strategy:
      1. Canny edge detection on the gray image.
      2. Find all external contours; pick the largest one whose
         cv2.approxPolyDP simplifies to exactly 4 points (a quadrilateral).
      3. Try a strict epsilon first (0.02 * perimeter); fall back to a looser
         one (0.05 * perimeter). Needing the looser epsilon is penalized in
         the confidence score.
      4. Order the 4 corners TL → TR → BR → BL via the sum/diff trick.

    Returns:
        (corners, confidence)
        - corners: float32 ndarray of shape (4, 2) ordered TL/TR/BR/BL,
          or None if no reasonable quadrilateral was found.
        - confidence: float in [0, 1]. Reflects:
            * how large the quad is relative to the image (area ratio),
            * penalized if a loose epsilon was needed.
          Returns 0.0 when corners is None.

    Known limitation: works best when the document boundary is the dominant
    dark rectangle in the image. Very cluttered backgrounds or images where
    the plan fills the entire frame (no margin) will return low confidence
    and fall back to the no-op path — which is the correct behaviour for
    clean top-down CAD exports.
    """
    h, w = gray.shape
    img_area = h * w

    edges = cv2.Canny(gray, 50, 150)
    # Dilate edges slightly to close small gaps in document borders
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0

    # Sort by area descending; test the top candidates (not just the largest,
    # in case the largest contour is a thin border artifact)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        if area < img_area * 0.05:
            # Contour is too small to be the document boundary
            break

        perimeter = cv2.arcLength(contour, True)

        # Try strict epsilon first
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        used_loose = False
        if len(approx) != 4:
            # Fall back to looser epsilon; penalize confidence
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
            used_loose = True

        if len(approx) != 4:
            continue  # Still not a quad; try next contour

        # Found a quad — compute confidence
        area_ratio = min(area / img_area, 1.0)
        epsilon_penalty = 0.2 if used_loose else 0.0
        confidence = float(area_ratio * (1.0 - epsilon_penalty))

        corners = _order_corners(approx.reshape(4, 2).astype(np.float32))
        return corners, confidence

    return None, 0.0


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order four corner points as: top-left, top-right, bottom-right, bottom-left.
    Uses the standard sum/diff trick:
      - TL has the smallest sum (x+y), BR has the largest sum.
      - TR has the smallest diff (x-y), BL has the largest diff.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]   # top-left
    ordered[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered


def correct_perspective(img: np.ndarray, corners: list | None = None):
    """
    T2.2 - homography correction for photographed (non top-down) plans.
    If explicit corners aren't supplied, this is a no-op passthrough:
    true perspective correction needs either 4 user-picked corners or a
    quadrilateral-boundary detector, which is out of scope for the
    classical-CV MVP here. Wired so Team 2 can drop in a real detector.
    """
    if not corners or len(corners) != 4:
        return img

    pts_src = np.array(corners, dtype=np.float32)
    w = int(max(
        np.linalg.norm(pts_src[0] - pts_src[1]),
        np.linalg.norm(pts_src[2] - pts_src[3]),
    ))
    h = int(max(
        np.linalg.norm(pts_src[1] - pts_src[2]),
        np.linalg.norm(pts_src[3] - pts_src[0]),
    ))
    pts_dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(pts_src, pts_dst)
    return cv2.warpPerspective(img, H, (w, h))
