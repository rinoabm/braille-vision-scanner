"""Embossed Braille Dot Detector.

Detects raised Braille dots from camera images by looking for the
circular shadow/highlight patterns that embossed dots create under
ambient or side lighting.

Real embossed Braille appears as:
- A small circular shadow on one side of the dot (away from light)
- A small circular highlight on the other side (facing light)
- Or under diffuse lighting: a subtle circular bump with slight shadow ring

Detection strategy:
1. Use morphological operations to enhance circular features
2. Apply Difference of Gaussians (DoG) to detect blob-like structures
3. Use Hough Circle Transform as a secondary detector
4. Combine results for robust detection
"""

import cv2
import numpy as np
from typing import Optional

from backend.models.data_models import (
    DotDetectionResult,
    DotPosition,
    PreprocessedImage,
)


class EmbossedDotDetector:
    """Detects embossed (raised) Braille dots using shadow/highlight analysis.
    
    Works with real physical Braille where dots are the same color as paper
    but create subtle shadows and highlights visible to a camera.
    """

    def __init__(
        self,
        min_radius: int = 5,
        max_radius: int = 30,
        sensitivity: float = 0.7,
    ):
        """Initialize the embossed dot detector.
        
        Args:
            min_radius: Minimum expected dot radius in pixels.
            max_radius: Maximum expected dot radius in pixels.
            sensitivity: Detection sensitivity (0.0 to 1.0). Higher = more dots detected
                        but also more false positives.
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.sensitivity = sensitivity

    def detect(self, image: np.ndarray) -> DotDetectionResult:
        """Detect embossed Braille dots in a camera image.
        
        Uses multiple detection strategies and combines results:
        1. Difference of Gaussians (DoG) blob detection
        2. Hough Circle Transform
        3. Morphological top-hat filtering
        
        Args:
            image: Input image (BGR or grayscale) from camera.
            
        Returns:
            DotDetectionResult with detected dot positions.
        """
        import time
        start_time = time.perf_counter()

        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Enhance contrast locally (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Strategy 1: Difference of Gaussians (detects blob-like structures)
        dog_dots = self._detect_dog(enhanced)

        # Strategy 2: Hough Circle Transform
        hough_dots = self._detect_hough_circles(enhanced)

        # Strategy 3: Morphological top-hat (detects bright spots on dark bg or vice versa)
        tophat_dots = self._detect_tophat(enhanced)

        # Merge results: keep dots that appear in at least 2 of 3 methods
        # or have high confidence from any single method
        merged_dots = self._merge_detections(dog_dots, hough_dots, tophat_dots)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return DotDetectionResult(
            dots=merged_dots,
            low_confidence_regions=[],
            processing_time_ms=elapsed_ms,
        )

    def _detect_dog(self, gray: np.ndarray) -> list[DotPosition]:
        """Detect dots using Difference of Gaussians (blob detection).
        
        DoG highlights circular structures at a specific scale.
        """
        dots = []
        
        # Try multiple scales to handle different dot sizes
        for sigma in [2, 4, 6, 8]:
            # Two Gaussian blurs at different scales
            blur1 = cv2.GaussianBlur(gray, (0, 0), sigma)
            blur2 = cv2.GaussianBlur(gray, (0, 0), sigma * 1.6)
            
            # Difference of Gaussians
            dog = cv2.subtract(blur1, blur2)
            
            # Also check inverted (for dots that appear as dark spots)
            dog_inv = cv2.subtract(blur2, blur1)
            
            # Threshold to find strong responses
            threshold = int(15 * (1.0 - self.sensitivity * 0.5))
            
            for dog_img in [dog, dog_inv]:
                _, binary = cv2.threshold(dog_img, threshold, 255, cv2.THRESH_BINARY)
                
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    expected_min = np.pi * self.min_radius**2 * 0.5
                    expected_max = np.pi * self.max_radius**2 * 2.0
                    
                    if area < expected_min or area > expected_max:
                        continue
                    
                    # Check circularity
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if circularity < 0.3:  # Very lenient for embossed dots
                        continue
                    
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                    radius = np.sqrt(area / np.pi)
                    
                    confidence = min(1.0, circularity * 0.7 + 0.3)
                    dots.append(DotPosition(x=cx, y=cy, confidence=confidence, radius=radius))
        
        return dots

    def _detect_hough_circles(self, gray: np.ndarray) -> list[DotPosition]:
        """Detect dots using Hough Circle Transform."""
        dots = []
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
        
        # Hough Circle detection
        # param1: Canny edge detection threshold
        # param2: Accumulator threshold (lower = more circles detected)
        param2 = int(30 * (1.0 - self.sensitivity * 0.4))
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=self.min_radius * 2,
            param1=80,
            param2=max(10, param2),
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (x, y, r) in circles:
                # Verify the circle region has some contrast
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(mask, (x, y), r, 255, -1)
                region_mean = cv2.mean(gray, mask=mask)[0]
                
                # Check surrounding area
                outer_mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.circle(outer_mask, (x, y), int(r * 1.5), 255, -1)
                cv2.circle(outer_mask, (x, y), r, 0, -1)
                surround_mean = cv2.mean(gray, mask=outer_mask)[0]
                
                # There should be SOME contrast between dot and surroundings
                contrast = abs(region_mean - surround_mean)
                if contrast < 3:
                    continue
                
                confidence = min(1.0, contrast / 30.0)
                dots.append(DotPosition(x=float(x), y=float(y), confidence=confidence, radius=float(r)))
        
        return dots

    def _detect_tophat(self, gray: np.ndarray) -> list[DotPosition]:
        """Detect dots using morphological top-hat transform.
        
        Top-hat extracts bright spots smaller than the structuring element.
        Black-hat extracts dark spots.
        """
        dots = []
        
        # Structuring element sized for expected dots
        kernel_size = self.min_radius * 2 + 3
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # White top-hat (bright dots on darker background)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        
        # Black top-hat (dark dots/shadows on lighter background)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        
        threshold = int(20 * (1.0 - self.sensitivity * 0.5))
        
        for hat_img in [tophat, blackhat]:
            _, binary = cv2.threshold(hat_img, max(5, threshold), 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                expected_min = np.pi * self.min_radius**2 * 0.3
                expected_max = np.pi * self.max_radius**2 * 2.0
                
                if area < expected_min or area > expected_max:
                    continue
                
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                if circularity < 0.25:
                    continue
                
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                radius = np.sqrt(area / np.pi)
                
                confidence = min(1.0, circularity * 0.6 + 0.2)
                dots.append(DotPosition(x=cx, y=cy, confidence=confidence, radius=radius))
        
        return dots

    def _merge_detections(
        self,
        dog_dots: list[DotPosition],
        hough_dots: list[DotPosition],
        tophat_dots: list[DotPosition],
    ) -> list[DotPosition]:
        """Merge detections from multiple methods.
        
        Keeps dots that appear in at least 2 methods (within proximity threshold)
        or have high confidence from a single method.
        """
        all_dots = []
        
        # Tag each dot with its source
        tagged = []
        for d in dog_dots:
            tagged.append((d, 'dog'))
        for d in hough_dots:
            tagged.append((d, 'hough'))
        for d in tophat_dots:
            tagged.append((d, 'tophat'))
        
        if not tagged:
            return []
        
        # Cluster nearby dots (within min_radius distance)
        merge_dist = self.min_radius * 1.5
        used = [False] * len(tagged)
        
        for i in range(len(tagged)):
            if used[i]:
                continue
            
            cluster = [tagged[i]]
            used[i] = True
            sources = {tagged[i][1]}
            
            for j in range(i + 1, len(tagged)):
                if used[j]:
                    continue
                dist = np.sqrt((tagged[i][0].x - tagged[j][0].x)**2 + 
                              (tagged[i][0].y - tagged[j][0].y)**2)
                if dist < merge_dist:
                    cluster.append(tagged[j])
                    used[j] = True
                    sources.add(tagged[j][1])
            
            # Keep if detected by 2+ methods, or high confidence single detection
            max_conf = max(d.confidence for d, _ in cluster)
            
            if len(sources) >= 2 or max_conf >= 0.7:
                # Average position of cluster
                avg_x = sum(d.x for d, _ in cluster) / len(cluster)
                avg_y = sum(d.y for d, _ in cluster) / len(cluster)
                avg_r = sum(d.radius for d, _ in cluster) / len(cluster)
                
                # Boost confidence for multi-method detections
                confidence = min(1.0, max_conf + 0.1 * (len(sources) - 1))
                
                all_dots.append(DotPosition(
                    x=avg_x, y=avg_y, confidence=confidence, radius=avg_r
                ))
        
        return all_dots
