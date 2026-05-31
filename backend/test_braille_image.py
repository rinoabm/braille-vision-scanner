"""Generate a synthetic Braille test image and test it via the upload API.

Run this script to verify the pipeline works end-to-end:
    python -m backend.test_braille_image

It creates a test image with the word "hello" in Braille and processes it.
"""

import cv2
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.api.main import run_pipeline


# Grade 1 Braille patterns for "hello"
# h = dots 1,2,5
# e = dots 1,5
# l = dots 1,2,3
# l = dots 1,2,3
# o = dots 1,3,5
BRAILLE_PATTERNS = {
    'h': [1, 2, 5],
    'e': [1, 5],
    'l': [1, 2, 3],
    'o': [1, 3, 5],
}

# Dot position grid (2 columns x 3 rows):
#   1  4
#   2  5
#   3  6
DOT_GRID = {
    1: (0, 0),  # col 0, row 0
    2: (0, 1),  # col 0, row 1
    3: (0, 2),  # col 0, row 2
    4: (1, 0),  # col 1, row 0
    5: (1, 1),  # col 1, row 1
    6: (1, 2),  # col 1, row 2
}


def create_braille_image(text: str, save_path: str = "test_braille.png") -> np.ndarray:
    """Create a synthetic Braille image for testing.
    
    Generates white dots on dark background simulating embossed Braille.
    """
    # Image dimensions
    width = 800
    height = 600
    
    # Braille cell dimensions (in pixels) - more realistic spacing
    dot_radius = 12
    col_spacing = 30      # horizontal distance between dot columns in a cell
    row_spacing = 30      # vertical distance between dot rows in a cell
    cell_spacing = 80     # horizontal distance between cells (center to center)
    
    # Create dark background (simulating paper in shadow)
    image = np.full((height, width), 40, dtype=np.uint8)
    
    # Starting position (centered)
    start_x = 100
    start_y = 200
    
    for char_idx, char in enumerate(text):
        if char == ' ':
            continue
            
        pattern = BRAILLE_PATTERNS.get(char, [])
        cell_x = start_x + char_idx * cell_spacing
        
        for dot_num in pattern:
            col, row = DOT_GRID[dot_num]
            dot_x = cell_x + col * col_spacing
            dot_y = start_y + row * row_spacing
            
            # Draw a bright dot (simulating raised dot catching light)
            cv2.circle(image, (dot_x, dot_y), dot_radius, 240, -1)
    
    # Convert to BGR for saving
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Save the image
    cv2.imwrite(save_path, bgr)
    print(f"Saved test image to: {save_path}")
    
    return bgr


def main():
    text = "hello"
    print(f"Creating Braille image for: '{text}'")
    print(f"Expected Braille patterns:")
    for char in text:
        dots = BRAILLE_PATTERNS.get(char, [])
        print(f"  '{char}' = dots {dots}")
    print()
    
    # Create the image
    image = create_braille_image(text, "test_braille_hello.png")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print()
    
    # Run through the pipeline
    print("Running pipeline...")
    try:
        # Debug: run each stage separately to see what's happening
        from backend.pipeline.preprocessing import PreprocessingPipeline
        from backend.pipeline.dot_detector import DotDetector
        from backend.pipeline.cell_segmenter import CellSegmenter
        from backend.pipeline.braille_decoder import BrailleDecoder

        preprocessing = PreprocessingPipeline()
        dot_detector = DotDetector()
        cell_segmenter = CellSegmenter()
        braille_decoder = BrailleDecoder()

        # Step 1: Preprocess
        preprocessed = preprocessing.process(image)
        print(f"  Preprocessed: binary image {preprocessed.binary_image.shape}")
        print(f"  Rotation corrected: {preprocessed.rotation_corrected:.2f} degrees")

        # Step 2: Detect dots
        detection = dot_detector.detect(preprocessed)
        print(f"  Dots detected: {len(detection.dots)}")
        for i, dot in enumerate(detection.dots[:20]):
            print(f"    Dot {i+1}: x={dot.x:.1f}, y={dot.y:.1f}, r={dot.radius:.1f}, conf={dot.confidence:.2f}")

        # Step 3: Segment into cells
        segmentation = cell_segmenter.segment(detection)
        print(f"  Cells segmented: {len(segmentation.cells)}")
        print(f"  Lines: {len(segmentation.lines)}")
        for line_idx, line in enumerate(segmentation.lines):
            print(f"    Line {line_idx}: {len(line)} cells")
            for cell in line:
                print(f"      Cell at {cell.grid_position}: dots={cell.dot_positions}, ambiguous={cell.is_ambiguous}")

        # Step 4: Decode
        decoded = braille_decoder.decode(segmentation)
        print(f"\n=== RESULT ===")
        print(f"Decoded text: '{decoded.text}'")
        print(f"Confidence: {sum(decoded.per_character_confidence)/max(1,len(decoded.per_character_confidence)):.2%}")
        print(f"Unrecognized: {len(decoded.unrecognized_indices)}")

    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
