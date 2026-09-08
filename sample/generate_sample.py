"""
Generates synthetic floor-plan PNGs so the pipeline can be tested
end-to-end without needing a real scanned drawing.

Produces two files:
  sample_floorplan.png          — clean top-down CAD-style plan (original)
  sample_floorplan_warped.png   — perspective-distorted version of the same
                                  plan, for testing T2.2 auto-correction.

Run:
    python3 sample/generate_sample.py
"""
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np

# ---------------------------------------------------------------------------
# Shared drawing helpers
# ---------------------------------------------------------------------------

W, H = 900, 700
WALL = "black"
THICK = 8


def _make_floorplan(include_stairs: bool = False):
    """Draw a 2-bedroom apartment outline and return a PIL Image."""
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    def wall(x1, y1, x2, y2, gap=None):
        """Draw a wall line, optionally leaving a gap (door/window opening)."""
        if gap is None:
            d.line([(x1, y1), (x2, y2)], fill=WALL, width=THICK)
            return
        gx1, gx2 = gap
        if y1 == y2:  # horizontal wall
            d.line([(x1, y1), (gx1, y1)], fill=WALL, width=THICK)
            d.line([(gx2, y1), (x2, y1)], fill=WALL, width=THICK)
        else:  # vertical wall
            d.line([(x1, y1), (x1, gx1)], fill=WALL, width=THICK)
            d.line([(x1, gx2), (x1, y2)], fill=WALL, width=THICK)

    # Outer envelope
    wall(100, 100, 800, 100)                 # top
    wall(100, 100, 100, 600)                 # left
    wall(100, 600, 800, 600)                 # bottom
    wall(800, 100, 800, 600, gap=(300, 335))  # right, with front door gap

    # Interior partition walls creating 2 bedrooms + living room + bathroom
    wall(450, 100, 450, 350, gap=(150, 185))   # vertical divider, doorway gap
    wall(450, 350, 800, 350)                   # horizontal divider
    wall(100, 350, 450, 350, gap=(200, 235))   # left-bottom divider w/ doorway
    wall(650, 350, 650, 600, gap=(450, 480))   # bathroom divider w/ doorway

    # A couple of "windows" as small gaps in the outer wall
    d.line([(250, 100), (320, 100)], fill="white", width=THICK)
    d.line([(250, 97), (320, 97)], fill="deepskyblue", width=3)
    d.line([(600, 600), (670, 600)], fill="white", width=THICK)
    d.line([(600, 603), (670, 603)], fill="deepskyblue", width=3)

    # Stair block — only included in the warped/distorted sample so that
    # T2.5b stair detection has treads to find, without altering the original
    # clean baseline (sample_floorplan.png) that regression tests compare against.
    if include_stairs:
        stair_x0, stair_y0 = 115, 365
        stair_w = 60
        for step in range(8):
            sy = stair_y0 + step * 12
            d.line([(stair_x0, sy), (stair_x0 + stair_w, sy)], fill=WALL, width=2)

    # Room labels (OCR will read these)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    d.text((200, 200), "BEDROOM 1", fill="black", font=font)
    d.text((550, 180), "BEDROOM 2", fill="black", font=font)
    d.text((200, 450), "LIVING ROOM", fill="black", font=font)
    d.text((680, 450), "BATH", fill="black", font=font)

    # A dimension label near the top wall: 700 px == 8400mm -> 12 mm/px
    d.text((400, 75), "8400mm", fill="black", font=font)

    return img


# ---------------------------------------------------------------------------
# Generate top-down (clean) sample
# ---------------------------------------------------------------------------

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Plain baseline — no stair treads, matches the original regression baseline
plan_img = _make_floorplan(include_stairs=False)
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_floorplan.png")
plan_img.save(out_path)
print("Saved:", out_path)

# ---------------------------------------------------------------------------
# Generate perspective-distorted sample
# Simulates a photograph of a floor-plan document lying on a gray table:
#   1. Paste the white-background plan onto a larger gray canvas.
#   2. Apply a keystone perspective warp to the combined image.
# This gives detect_document_corners a clear white-doc-on-gray-background
# boundary to locate, matching what a real hand-held photograph would show.
# ---------------------------------------------------------------------------

try:
    import cv2

    # Step 1: place the plan on a larger gray canvas (simulates a table)
    PAD = 80          # padding around the plan on all sides
    CW = W + 2 * PAD  # canvas width
    CH = H + 2 * PAD  # canvas height
    canvas = np.full((CH, CW, 3), 160, dtype=np.uint8)  # mid-gray background

    # Warped sample includes stair treads so T2.5b has something to detect
    warped_plan_img = _make_floorplan(include_stairs=True)
    plan_bgr = np.array(warped_plan_img)[:, :, ::-1].copy()  # PIL RGB → numpy BGR
    canvas[PAD:PAD + H, PAD:PAD + W] = plan_bgr

    # Step 2: apply a keystone warp (top edge tilted inward ~15%)
    # Source corners: the four plan corners on the gray canvas
    src = np.float32([
        [PAD,       PAD],        # TL
        [PAD + W,   PAD],        # TR
        [PAD + W,   PAD + H],    # BR
        [PAD,       PAD + H],    # BL
    ])
    # Destination: tilt the top inward, keep the bottom roughly in place
    tilt = int(W * 0.13)
    dst = np.float32([
        [PAD + tilt,         PAD + int(H * 0.1)],   # TL shifted right+down
        [PAD + W - tilt,     PAD + int(H * 0.1)],   # TR shifted left+down
        [PAD + W + tilt // 2, PAD + H - 10],         # BR shifted right
        [PAD - tilt // 2,     PAD + H - 10],         # BL shifted left
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(canvas, M, (CW, CH), borderValue=(160, 160, 160))

    warped_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "sample_floorplan_warped.png"
    )
    cv2.imwrite(warped_path, warped)
    print("Saved:", warped_path)
    print()
    print("Run perspective-correction test with:")
    print("  python run_detection.py sample/sample_floorplan_warped.png --out results_warped/")

except ImportError:
    print("opencv-python-headless not installed — skipping warped sample generation.")
    print("Install it with:  pip install opencv-python-headless")

