"""Generate pipeline stage visualization images.

Creates a folder showing each processing stage applied to the Braille test image,
perfect for demo presentations and documentation.

Run: python -m backend.generate_pipeline_stages

Output: pipeline_stages/ folder with numbered images for each step.
"""

import cv2
import numpy as np
import os

from backend.pipeline.preprocessing import PreprocessingPipeline
from backend.pipeline.dot_detector import DotDetector
from backend.pipeline.cell_segmenter import CellSegmenter
from backend.pipeline.braille_decoder import BrailleDecoder
from backend.models.data_models import DotDetectionResult


def main():
    output_dir = "pipeline_stages"
    os.makedirs(output_dir, exist_ok=True)

    # Load test image
    img = cv2.imread("braille_test_sheet.png")
    if img is None:
        print("ERROR: braille_test_sheet.png not found. Run generate_printable_braille first.")
        return

    print("Generating pipeline stage images...")
    print(f"Output folder: {output_dir}/")
    print()

    # === Stage 0: Original Input ===
    cv2.imwrite(f"{output_dir}/0_original_input.png", img)
    print("Stage 0: Original input image")

    # === Stage 1: Grayscale Conversion ===
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/1_grayscale.png", gray)
    print("Stage 1: Grayscale conversion")

    # === Stage 2: Brightness Normalization ===
    pp = PreprocessingPipeline()
    normalized = pp.normalize_brightness(gray)
    cv2.imwrite(f"{output_dir}/2_brightness_normalized.png", normalized)
    print(f"Stage 2: Brightness normalized (mean={np.mean(normalized):.0f})")

    # === Stage 3: Rotation Detection & Correction ===
    corrected, angle = pp.correct_rotation(normalized)
    cv2.imwrite(f"{output_dir}/3_rotation_corrected.png", corrected)
    print(f"Stage 3: Rotation corrected ({angle:.1f} degrees)")

    # === Stage 4: Noise Reduction ===
    denoised = pp.apply_noise_reduction(corrected)
    cv2.imwrite(f"{output_dir}/4_noise_reduced.png", denoised)
    print("Stage 4: Noise reduction (Gaussian + Median filter)")

    # === Stage 5: Thresholding (Binary) ===
    pre = pp.process(img)
    cv2.imwrite(f"{output_dir}/5_binary_threshold.png", pre.binary_image)
    white_ratio = np.sum(pre.binary_image == 255) / pre.binary_image.size
    print(f"Stage 5: Binary thresholding (white ratio: {white_ratio:.1%})")

    # === Stage 6: Dot Detection ===
    dd = DotDetector()
    det = dd.detect(pre)
    
    # Draw detected dots on the original image
    dot_overlay = img.copy()
    for dot in det.dots:
        cx, cy, r = int(dot.x), int(dot.y), int(dot.radius)
        cv2.circle(dot_overlay, (cx, cy), r, (0, 255, 0), 2)  # Green circles
        cv2.circle(dot_overlay, (cx, cy), 2, (0, 0, 255), -1)  # Red center
    cv2.imwrite(f"{output_dir}/6_dots_detected.png", dot_overlay)
    print(f"Stage 6: Dot detection ({len(det.dots)} dots found)")

    # === Stage 7: Cell Segmentation ===
    cs = CellSegmenter()
    seg = cs.segment(det)
    
    # Draw cell boundaries on the image
    cell_overlay = img.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    for line_idx, line in enumerate(seg.lines):
        color = colors[line_idx % len(colors)]
        for cell in line:
            # Find dots belonging to this cell and draw bounding box
            cell_dots = []
            for dot in det.dots:
                # Simple proximity check
                for other_line in seg.lines:
                    for other_cell in other_line:
                        if other_cell.grid_position == cell.grid_position:
                            break
            
            # Draw cell label
            if cell.dot_positions:
                # Estimate cell position from grid_position
                line_i, cell_i = cell.grid_position
                label = str(cell.dot_positions)
                # Position text based on cell index
                tx = 50 + cell_i * 90
                ty = 130 + line_i * 170
                cv2.putText(cell_overlay, label, (tx, ty), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    cv2.imwrite(f"{output_dir}/7_cells_segmented.png", cell_overlay)
    print(f"Stage 7: Cell segmentation ({len(seg.cells)} cells, {len(seg.lines)} lines)")

    # === Stage 8: Braille Decoding ===
    bd = BrailleDecoder()
    decoded = bd.decode(seg)
    
    # Create a result visualization
    result_img = np.full((300, 800, 3), 30, dtype=np.uint8)
    
    # Title
    cv2.putText(result_img, "DECODED RESULT", (20, 40),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    
    # Decoded text
    lines = decoded.text.split('\n')
    for i, line in enumerate(lines):
        cv2.putText(result_img, f'Line {i+1}: "{line}"', (20, 90 + i * 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Confidence
    if decoded.per_character_confidence:
        avg_conf = sum(decoded.per_character_confidence) / len(decoded.per_character_confidence)
        cv2.putText(result_img, f"Confidence: {avg_conf:.1%}", (20, 270),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imwrite(f"{output_dir}/8_decoded_result.png", result_img)
    print(f"Stage 8: Braille decoded -> '{decoded.text}'")

    # === Summary image (all stages side by side) ===
    print()
    print("=" * 50)
    print(f"All stages saved to: {output_dir}/")
    print()
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(f"{output_dir}/{f}") // 1024
        print(f"  {f} ({size}KB)")
    print()
    print("Use these images in your demo video to show")
    print("the pipeline processing step by step!")


if __name__ == "__main__":
    main()
