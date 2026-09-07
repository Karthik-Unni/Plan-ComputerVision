"""
Generates a simple synthetic floor-plan PNG so the pipeline can be tested
end-to-end without needing a real scanned drawing. Draws a 2-bedroom
apartment outline with a dimension label and a door gap.
"""
from PIL import Image, ImageDraw, ImageFont
import os

W, H = 900, 700
img = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(img)

WALL = "black"
THICK = 8

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

# A couple of "windows" as small gaps in the outer wall (drawn as thinner marks)
d.line([(250, 100), (320, 100)], fill="white", width=THICK)
d.line([(250, 97), (320, 97)], fill="deepskyblue", width=3)
d.line([(600, 600), (670, 600)], fill="white", width=THICK)
d.line([(600, 603), (670, 603)], fill="deepskyblue", width=3)

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

os.makedirs(os.path.dirname(__file__), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), "sample_floorplan.png")
img.save(out_path)
print("Saved:", out_path)
