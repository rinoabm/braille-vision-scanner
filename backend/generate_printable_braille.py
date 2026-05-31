"""Generate a printable Braille test sheet.

Print this image, then point your camera at the printed page.
The black dots on white background simulate how a camera sees embossed Braille
(dots appear darker due to shadows).

Run: python -m backend.generate_printable_braille
"""

import cv2
import numpy as np

# Braille patterns
PATTERNS = {
    'a': [1], 'b': [1,2], 'c': [1,4], 'd': [1,4,5], 'e': [1,5],
    'f': [1,2,4], 'g': [1,2,4,5], 'h': [1,2,5], 'i': [2,4], 'j': [2,4,5],
    'k': [1,3], 'l': [1,2,3], 'm': [1,3,4], 'n': [1,3,4,5], 'o': [1,3,5],
}

# Dot grid positions
DOT_GRID = {
    1: (0, 0), 2: (0, 1), 3: (0, 2),
    4: (1, 0), 5: (1, 1), 6: (1, 2),
}


def create_printable_sheet():
    """Create a high-contrast printable Braille test sheet."""
    width = 1200
    height = 900
    
    # White background (like paper)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    
    # Braille parameters (larger for printing)
    dot_radius = 14
    col_spacing = 36
    row_spacing = 36
    cell_spacing = 90
    
    # Title
    cv2.putText(image, "BRAILLE TEST SHEET",
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
    
    # Line 1: "hello"
    text1 = "hello"
    y_start = 150
    draw_braille_line(image, text1, 100, y_start, dot_radius, col_spacing, row_spacing, cell_spacing)
    
    # Line 2: "abc"
    text2 = "abc"
    y_start = 320
    draw_braille_line(image, text2, 100, y_start, dot_radius, col_spacing, row_spacing, cell_spacing)
    
    # Line 3: "no"
    text3 = "no"
    y_start = 490
    draw_braille_line(image, text3, 100, y_start, dot_radius, col_spacing, row_spacing, cell_spacing)

    # Line 4: "ok"
    text4 = "ok"
    y_start = 660
    draw_braille_line(image, text4, 100, y_start, dot_radius, col_spacing, row_spacing, cell_spacing)

    # Draw dot position reference
    cv2.putText(image, "Dot positions: 1=top-left, 2=mid-left, 3=bot-left, 4=top-right, 5=mid-right, 6=bot-right",
                (50, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    cv2.imwrite("braille_test_sheet.png", image)
    print("Saved: braille_test_sheet.png")
    print("Print this image and point your camera at it!")
    print()
    print("Expected results:")
    print(f"  Line 1: '{text1}'")
    print(f"  Line 2: '{text2}'")
    print(f"  Line 3: '{text3}'")
    print(f"  Line 4: '{text4}'")


def draw_braille_line(image, text, x_start, y_start, dot_radius, col_spacing, row_spacing, cell_spacing):
    """Draw a line of Braille text on the image."""
    for char_idx, char in enumerate(text):
        if char == ' ':
            continue
        pattern = PATTERNS.get(char, [])
        cell_x = x_start + char_idx * cell_spacing
        
        # Draw empty dot positions as light gray circles (guides)
        for pos in range(1, 7):
            col, row = DOT_GRID[pos]
            dx = cell_x + col * col_spacing
            dy = y_start + row * row_spacing
            cv2.circle(image, (dx, dy), dot_radius + 2, (220, 220, 220), 1)
        
        # Draw active dots as solid black circles
        for dot_num in pattern:
            col, row = DOT_GRID[dot_num]
            dx = cell_x + col * col_spacing
            dy = y_start + row * row_spacing
            cv2.circle(image, (dx, dy), dot_radius, (0, 0, 0), -1)


if __name__ == "__main__":
    create_printable_sheet()
