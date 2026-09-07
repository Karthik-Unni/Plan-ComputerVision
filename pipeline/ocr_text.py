"""
T2.6 OCR extraction, T2.7 Text classification

Uses EasyOCR instead of pytesseract/Tesseract: EasyOCR is a pure `pip
install`, with no separate system binary to install and add to PATH. Its
detection/recognition models (~100MB) download automatically the first
time it runs and are cached locally after that, so the first call will be
slower and needs an internet connection once; every call after that runs
fully offline.
"""
import os
import re
import threading

_READER = None
_READER_LOCK = threading.Lock()
EASYOCR_AVAILABLE = True

try:
    import easyocr
except Exception:
    EASYOCR_AVAILABLE = False

# Store EasyOCR's downloaded models inside this project folder instead of the
# default ~/.EasyOCR in the user's home directory. Some Windows setups (
# restricted permissions, OneDrive-synced folders, antivirus) block writes
# to the home directory; a local folder avoids that entirely.
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ocr_models")
_MODEL_DIR = os.path.normpath(_MODEL_DIR)

DIMENSION_RE = re.compile(
    r"(\d+(\.\d+)?)\s*(mm|cm|m|ft|'|\")\s*(x|X|×)?\s*(\d+(\.\d+)?)?\s*(mm|cm|m|ft|'|\")?"
)
IDENTIFIER_RE = re.compile(r"^[A-Za-z]{1,3}-?\d{1,3}$")  # e.g. D1, W2, C-01


def _get_reader():
    """Lazily create (and cache) the EasyOCR reader, since it's a bit slow
    to initialize and only needs doing once per process. Models are stored
    in a local project folder (see _MODEL_DIR) rather than the user's home
    directory, to sidestep Windows permission issues."""
    global _READER
    if _READER is None:
        with _READER_LOCK:
            if _READER is None:
                os.makedirs(_MODEL_DIR, exist_ok=True)
                _READER = easyocr.Reader(
                    ["en"], gpu=False, verbose=False,
                    model_storage_directory=_MODEL_DIR,
                    user_network_directory=_MODEL_DIR,
                )
    return _READER


def extract_text(gray_image, min_confidence=0.4):
    """Run OCR and return raw text strings with bounding boxes.
    Degrades gracefully (returns []) if easyocr isn't installed or fails
    for any reason, so the rest of the pipeline still runs."""
    if not EASYOCR_AVAILABLE:
        return []

    try:
        reader = _get_reader()
        raw_results = reader.readtext(gray_image)
    except Exception as e:
        print(f"[ocr_text] OCR unavailable, skipping text extraction ({e}).")
        print(f"[ocr_text] Tip: models are stored in '{_MODEL_DIR}'. "
              f"If that folder can't be created/written to (permissions, "
              f"antivirus, synced-folder issues), try running as "
              f"administrator once, or moving this project out of a "
              f"cloud-synced folder (OneDrive/Dropbox) and retrying.")
        return []

    texts = []
    for bbox, text, conf in raw_results:
        text = text.strip()
        if not text or conf < min_confidence:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = float(min(xs)), float(min(ys))
        w, h = float(max(xs) - x), float(max(ys) - y)
        texts.append({
            "text": text,
            "bbox_px": [x, y, w, h],
            "ocr_confidence": float(conf),
        })
    return texts


def classify_texts(texts):
    """T2.7 - label each OCR string as room_name / dimension / identifier / unknown."""
    classified = []
    for t in texts:
        raw = t["text"]
        letters_only = re.sub(r"[^A-Za-z]", "", raw)
        digits_only = re.sub(r"[^0-9]", "", raw)

        if DIMENSION_RE.search(raw) and any(ch.isdigit() for ch in raw) and \
                any(unit in raw.lower() for unit in ("mm", "cm", " m", "ft", "'", '"')):
            kind = "dimension"
            conf = 0.8
        elif IDENTIFIER_RE.match(raw):
            kind = "identifier"
            conf = 0.7
        elif len(letters_only) >= 3 and len(digits_only) <= 2:
            # e.g. "BEDROOM 2", "LIVING ROOM", "BATH" - mostly letters,
            # at most a trailing room number
            kind = "room_name"
            conf = 0.6
        else:
            kind = "unknown"
            conf = 0.3
        classified.append({
            "text": raw,
            "bbox_px": t["bbox_px"],
            "kind": kind,
            "confidence": round(min(conf, t.get("ocr_confidence", 0.6)), 3)
                          if t.get("ocr_confidence") else conf,
        })
    return classified
