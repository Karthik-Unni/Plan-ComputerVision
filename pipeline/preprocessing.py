"""
T2.1 Preprocessing pipeline, T2.2 Perspective correction
Raw image -> deskewed, contrast-normalized, binarized image
"""
import cv2
import numpy as np


def load_and_preprocess(path: str):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image at {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast normalization
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=7)

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

    # Binarize (adaptive, since scans/photos have uneven lighting)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    return {
        "original_bgr": img,
        "gray": gray,
        "binary": binary,
        "deskew_angle_deg": angle_deg,
    }


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
