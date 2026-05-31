"""Debug the last captured frame."""
import cv2
import numpy as np
import os

if not os.path.exists('debug_last_frame.jpg'):
    print("No debug_last_frame.jpg found. Start scan first.")
    exit()

img = cv2.imread('debug_last_frame.jpg')
print(f"Frame: {img.shape[1]}x{img.shape[0]}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Brightness: mean={np.mean(gray):.0f}, std={np.std(gray):.0f}")

from backend.pipeline.preprocessing import PreprocessingPipeline
from backend.pipeline.dot_detector import DotDetector
from backend.pipeline.cell_segmenter import CellSegmenter
from backend.pipeline.braille_decoder import BrailleDecoder
from backend.api.main import run_pipeline

# Full pipeline
r = run_pipeline(img)
print(f"Decoded: '{r['text']}'")
print(f"Confidence: {r['confidence']:.1%}")

# Stage by stage
pp = PreprocessingPipeline()
dd = DotDetector()

pre = pp.process(img)
binary_white = np.sum(pre.binary_image == 255) / pre.binary_image.size
print(f"Binary white ratio: {binary_white:.3f}")

det = dd.detect(pre)
print(f"Dots: {len(det.dots)}")
for d in sorted(det.dots, key=lambda x: x.radius, reverse=True)[:10]:
    print(f"  x={d.x:.0f} y={d.y:.0f} r={d.radius:.1f} conf={d.confidence:.2f}")

# Also check what big contours exist
contours, _ = cv2.findContours(pre.binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
big_round = []
for c in contours:
    a = cv2.contourArea(c)
    if a < 50:
        continue
    p = cv2.arcLength(c, True)
    if p == 0:
        continue
    circ = 4 * 3.14159 * a / (p * p)
    if circ > 0.5:
        x, y, w, h = cv2.boundingRect(c)
        big_round.append((a, x, y, w, h, circ))

big_round.sort(reverse=True)
print(f"\nRound contours (area>50, circ>0.5): {len(big_round)}")
for a, x, y, w, h, c in big_round[:10]:
    print(f"  area={a:.0f} at ({x},{y}) size={w}x{h} circ={c:.2f}")
