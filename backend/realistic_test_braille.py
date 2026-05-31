"""Realistic Braille image testing with shadows, lighting variations, and paper textures.

Simulates real-world conditions:
- Paper texture (slight noise)
- Uneven lighting (gradient shadows)
- Dot shadows (3D raised dot effect)
- Partial shadows across the page
- Different paper colors (white, cream, gray)
- Camera-like perspective (slight blur)

Run: python -m backend.realistic_test_braille
"""

import cv2
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.api.main import run_pipeline


BRAILLE_PATTERNS = {
    'a': [1], 'b': [1,2], 'c': [1,4], 'd': [1,4,5], 'e': [1,5],
    'f': [1,2,4], 'g': [1,2,4,5], 'h': [1,2,5], 'i': [2,4], 'j': [2,4,5],
    'k': [1,3], 'l': [1,2,3], 'm': [1,3,4], 'n': [1,3,4,5], 'o': [1,3,5],
    'p': [1,2,3,4], 'q': [1,2,3,4,5], 'r': [1,2,3,5], 's': [2,3,4], 't': [2,3,4,5],
    'u': [1,3,6], 'v': [1,2,3,6], 'w': [2,4,5,6], 'x': [1,3,4,6],
    'y': [1,3,4,5,6], 'z': [1,3,5,6],
    ' ': [],
}

DOT_GRID = {1: (0,0), 2: (0,1), 3: (0,2), 4: (1,0), 5: (1,1), 6: (1,2)}


def create_realistic_braille(
    text: str,
    width: int = 1000,
    height: int = 700,
    dot_radius: int = 12,
    col_spacing: int = 30,
    row_spacing: int = 30,
    cell_spacing: int = 80,
    paper_color: int = 240,
    shadow_type: str = "none",  # "none", "left", "right", "top", "diagonal", "spot"
    paper_texture: bool = False,
    dot_3d: bool = False,
    blur_amount: int = 0,
    brightness_variation: int = 0,
) -> np.ndarray:
    """Create a realistic Braille image simulating real-world conditions."""
    
    # Create paper background
    image = np.full((height, width), paper_color, dtype=np.uint8)
    
    # Add paper texture (subtle noise)
    if paper_texture:
        texture = np.random.randint(-8, 8, (height, width), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + texture, 0, 255).astype(np.uint8)
    
    # Add shadow/lighting gradient
    if shadow_type == "left":
        # Shadow on left side (darker on left, brighter on right)
        gradient = np.linspace(0.6, 1.0, width).reshape(1, -1)
        image = (image.astype(np.float64) * gradient).astype(np.uint8)
    elif shadow_type == "right":
        gradient = np.linspace(1.0, 0.6, width).reshape(1, -1)
        image = (image.astype(np.float64) * gradient).astype(np.uint8)
    elif shadow_type == "top":
        gradient = np.linspace(0.6, 1.0, height).reshape(-1, 1)
        image = (image.astype(np.float64) * gradient).astype(np.uint8)
    elif shadow_type == "diagonal":
        # Diagonal shadow (top-left darker)
        y_grad = np.linspace(0.7, 1.0, height).reshape(-1, 1)
        x_grad = np.linspace(0.7, 1.0, width).reshape(1, -1)
        gradient = (y_grad + x_grad) / 2.0
        image = (image.astype(np.float64) * gradient).astype(np.uint8)
    elif shadow_type == "spot":
        # Circular shadow spot in the middle
        cy, cx = height // 2, width // 3
        Y, X = np.ogrid[:height, :width]
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
        shadow = np.clip(1.0 - 0.3 * np.exp(-dist**2 / (200**2)), 0.7, 1.0)
        image = (image.astype(np.float64) * shadow).astype(np.uint8)
    
    # Add brightness variation (simulating uneven room lighting)
    if brightness_variation > 0:
        var = np.random.randint(-brightness_variation, brightness_variation, 
                                 (height // 50 + 1, width // 50 + 1), dtype=np.int16)
        var_resized = cv2.resize(var.astype(np.float32), (width, height), 
                                  interpolation=cv2.INTER_LINEAR)
        image = np.clip(image.astype(np.float64) + var_resized, 0, 255).astype(np.uint8)
    
    # Draw Braille dots
    x_start = 150
    y_start = 250
    line_y = y_start
    cell_idx = 0
    
    for char in text:
        if char == '\n':
            line_y += row_spacing * 3 + 60
            cell_idx = 0
            continue
        if char == ' ':
            cell_idx += 1
            continue
        
        pattern = BRAILLE_PATTERNS.get(char, [])
        cell_x = x_start + cell_idx * cell_spacing
        
        if cell_x + col_spacing > width - 50:
            continue
        
        for dot_num in pattern:
            col, row = DOT_GRID[dot_num]
            dx = cell_x + col * col_spacing
            dy = line_y + row * row_spacing
            
            if dot_3d:
                # 3D raised dot effect: shadow below/right, highlight above/left
                # Real embossed Braille has strong shadows from side lighting
                # Use local background value for contrast calculation
                local_bg = int(image[min(dy, height-1), min(dx, width-1)])
                # Shadow (darkest part, offset to bottom-right)
                shadow_val = max(0, local_bg - 100)
                cv2.circle(image, (dx + 2, dy + 2), dot_radius, shadow_val, -1)
                # Main dot body (dark relative to local background)
                dot_val = max(0, local_bg - 80)
                cv2.circle(image, (dx, dy), dot_radius, dot_val, -1)
                # Highlight on top-left (slightly lighter than main dot)
                highlight_val = max(0, local_bg - 40)
                cv2.circle(image, (dx - 1, dy - 1), dot_radius - 3, highlight_val, -1)
            else:
                # Simple dark dot on light background
                cv2.circle(image, (dx, dy), dot_radius, 
                          max(0, paper_color - 180), -1)
        
        cell_idx += 1
    
    # Apply blur (simulating slightly out-of-focus camera)
    if blur_amount > 0:
        ksize = blur_amount * 2 + 1
        image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    
    # Convert to BGR
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return bgr


def run_test(name: str, text: str, expected: str, **kwargs) -> dict:
    """Run a single test and return results."""
    start = time.perf_counter()
    image = create_realistic_braille(text, **kwargs)
    
    # Save image for inspection
    safe_name = name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
    filename = f"test_images/{safe_name}.png"
    os.makedirs("test_images", exist_ok=True)
    cv2.imwrite(filename, image)
    
    result = run_pipeline(image)
    elapsed = (time.perf_counter() - start) * 1000
    
    decoded = result['text']
    passed = decoded == expected
    
    return {
        'name': name,
        'expected': expected,
        'got': decoded,
        'passed': passed,
        'confidence': result['confidence'],
        'time_ms': elapsed,
        'filename': filename,
    }


def main():
    print("=" * 70)
    print("REALISTIC BRAILLE IMAGE TESTING")
    print("Shadows, Textures, Lighting Variations, 3D Effects")
    print("=" * 70)
    print()
    
    results = []
    
    # --- Clean baseline ---
    print("--- BASELINE (clean images) ---")
    results.append(run_test("Clean 'hello'", "hello", "hello"))
    results.append(run_test("Clean 'world'", "world", "world"))
    results.append(run_test("Clean 'braille'", "braille", "braille"))
    
    # --- Paper texture ---
    print("--- PAPER TEXTURE ---")
    results.append(run_test("Textured paper 'hello'", "hello", "hello",
                           paper_texture=True))
    results.append(run_test("Textured paper 'braille'", "braille", "braille",
                           paper_texture=True))
    results.append(run_test("Textured paper 'vision'", "vision", "vision",
                           paper_texture=True))
    
    # --- Shadow variations ---
    print("--- SHADOWS ---")
    results.append(run_test("Left shadow 'hello'", "hello", "hello",
                           shadow_type="left"))
    results.append(run_test("Right shadow 'hello'", "hello", "hello",
                           shadow_type="right"))
    results.append(run_test("Top shadow 'hello'", "hello", "hello",
                           shadow_type="top"))
    results.append(run_test("Diagonal shadow 'hello'", "hello", "hello",
                           shadow_type="diagonal"))
    results.append(run_test("Spot shadow 'hello'", "hello", "hello",
                           shadow_type="spot"))
    results.append(run_test("Left shadow 'braille'", "braille", "braille",
                           shadow_type="left"))
    
    # --- 3D raised dots (realistic embossed Braille) ---
    print("--- 3D RAISED DOTS ---")
    results.append(run_test("3D dots 'hello'", "hello", "hello",
                           dot_3d=True))
    results.append(run_test("3D dots 'world'", "world", "world",
                           dot_3d=True))
    results.append(run_test("3D dots 'braille'", "braille", "braille",
                           dot_3d=True))
    results.append(run_test("3D dots + texture 'hello'", "hello", "hello",
                           dot_3d=True, paper_texture=True))
    results.append(run_test("3D dots + shadow 'hello'", "hello", "hello",
                           dot_3d=True, shadow_type="left"))
    
    # --- Different paper colors ---
    print("--- PAPER COLORS ---")
    results.append(run_test("Cream paper 'hello'", "hello", "hello",
                           paper_color=230))
    results.append(run_test("Gray paper 'hello'", "hello", "hello",
                           paper_color=180))
    results.append(run_test("Dark paper (white dots) 'hello'", "hello", "hello",
                           paper_color=40))
    
    # --- Slight blur (out of focus) ---
    print("--- CAMERA BLUR ---")
    results.append(run_test("Slight blur 'hello'", "hello", "hello",
                           blur_amount=1))
    results.append(run_test("Medium blur 'hello'", "hello", "hello",
                           blur_amount=2))
    
    # --- Combined conditions ---
    print("--- COMBINED CONDITIONS ---")
    results.append(run_test("Texture + shadow + blur 'hello'", "hello", "hello",
                           paper_texture=True, shadow_type="diagonal", blur_amount=1))
    results.append(run_test("3D + texture + shadow 'braille'", "braille", "braille",
                           dot_3d=True, paper_texture=True, shadow_type="left"))
    results.append(run_test("Gray paper + texture + shadow 'world'", "world", "world",
                           paper_color=190, paper_texture=True, shadow_type="right"))
    
    # --- Multi-word with realistic conditions ---
    print("--- REALISTIC SENTENCES ---")
    results.append(run_test("Sentence 'hi there'", "hi there", "hi there",
                           paper_texture=True, shadow_type="left"))
    results.append(run_test("Sentence 'good job'", "good job", "good job",
                           paper_texture=True, dot_3d=True))
    results.append(run_test("Multi-line realistic", "hello\nworld", "hello\nworld",
                           height=800, paper_texture=True, shadow_type="diagonal"))
    
    # --- Brightness variation ---
    print("--- UNEVEN LIGHTING ---")
    results.append(run_test("Uneven light 'hello'", "hello", "hello",
                           brightness_variation=15))
    results.append(run_test("Uneven light 'braille'", "braille", "braille",
                           brightness_variation=20))
    
    # ===== RESULTS =====
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    total_time = 0
    
    for r in results:
        status = "✅ PASS" if r['passed'] else "❌ FAIL"
        print(f"  {status} | {r['name']}")
        if not r['passed']:
            print(f"         Expected: {repr(r['expected'])}")
            print(f"         Got:      {repr(r['got'])}")
            print(f"         Conf: {r['confidence']:.1%} | File: {r['filename']}")
        if r['passed']:
            passed += 1
        else:
            failed += 1
        total_time += r['time_ms']
    
    print()
    print("-" * 70)
    print(f"  TOTAL: {passed}/{passed+failed} passed ({100*passed/(passed+failed):.0f}%)")
    print(f"  Average time: {total_time/(passed+failed):.0f}ms per image")
    print(f"  Total time: {total_time/1000:.1f}s")
    print(f"  Images saved to: test_images/")
    print("-" * 70)
    
    if failed == 0:
        print("\n  🎉 ALL REALISTIC TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {failed} test(s) failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
