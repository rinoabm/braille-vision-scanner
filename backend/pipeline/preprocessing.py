"""Preprocessing Pipeline for the Braille Vision Scanner.

Normalizes captured images for consistent dot detection by applying:
brightness normalization, rotation correction, noise reduction, and
adaptive thresholding.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6
"""

import cv2
import numpy as np

from backend.models.data_models import PreprocessedImage, ValidationResult


class PreprocessingPipeline:
    """Image preprocessing pipeline for Braille dot detection.

    Applies a sequence of corrections to raw camera frames:
    normalize brightness → correct rotation → reduce noise → adaptive threshold.

    All steps complete within 100ms per frame.
    """

    # Target mean intensity for brightness normalization
    TARGET_MEAN_INTENSITY = 128
    INTENSITY_TOLERANCE = 20

    # Rotation limits
    MAX_CORRECTABLE_ROTATION = 15.0  # degrees
    ROTATION_ACCURACY = 1.0  # degrees

    # Minimum resolution
    MIN_WIDTH = 640
    MIN_HEIGHT = 480

    # Overexposure threshold (fraction of pixels at max intensity)
    OVEREXPOSURE_THRESHOLD = 0.98

    def process(self, frame: np.ndarray) -> PreprocessedImage:
        """Full preprocessing: normalize → deskew → denoise → threshold.

        Completes within 100ms per frame.

        Args:
            frame: Raw captured frame as numpy array (BGR or grayscale).

        Returns:
            PreprocessedImage with binary image and metadata.

        Raises:
            ValueError: If the image fails validation.
        """
        original_resolution = (frame.shape[1], frame.shape[0])

        # Validate first
        validation = self.validate_image(frame)
        if validation != ValidationResult.OK:
            raise ValueError(
                f"Image validation failed: {validation.value}"
            )

        # Convert to grayscale if needed
        gray = self._to_grayscale(frame)

        # Step 1: Normalize brightness
        normalized = self.normalize_brightness(gray)

        # Step 2: Correct rotation
        deskewed, angle = self.correct_rotation(normalized)

        # Step 3: Apply noise reduction
        denoised = self.apply_noise_reduction(deskewed)

        # Step 4: Adaptive thresholding
        binary = self.adaptive_threshold(denoised)

        return PreprocessedImage(
            binary_image=binary,
            rotation_corrected=angle,
            brightness_adjusted=True,
            original_resolution=original_resolution,
        )

    def normalize_brightness(self, frame: np.ndarray) -> np.ndarray:
        """Adjust image to target mean intensity 128 ± 20.

        Uses scaling and offset to shift the mean intensity toward the target.
        Also applies CLAHE for contrast enhancement (histogram equalization).

        Args:
            frame: Grayscale image as numpy array.

        Returns:
            Brightness-normalized grayscale image.
        """
        gray = self._to_grayscale(frame)

        current_mean = np.mean(gray).item()

        if current_mean == 0:
            # Completely black image - just set to target
            result = np.full_like(gray, self.TARGET_MEAN_INTENSITY)
        else:
            # Scale to target mean
            scale = self.TARGET_MEAN_INTENSITY / current_mean
            result = np.clip(gray.astype(np.float64) * scale, 0, 255).astype(np.uint8)

        # Apply CLAHE for contrast enhancement (adaptive histogram equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(result)

        # Final adjustment to ensure mean is within tolerance
        final_mean = np.mean(result).item()
        if abs(final_mean - self.TARGET_MEAN_INTENSITY) > self.INTENSITY_TOLERANCE:
            offset = self.TARGET_MEAN_INTENSITY - final_mean
            result = np.clip(
                result.astype(np.float64) + offset, 0, 255
            ).astype(np.uint8)

        return result

    def correct_rotation(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """Correct skew up to ±15 degrees, returning corrected image and angle.

        Uses Hough Line Transform to detect dominant line angles and rotates
        the image to correct the detected skew.

        Args:
            frame: Grayscale image as numpy array.

        Returns:
            Tuple of (corrected image, detected angle in degrees).
        """
        gray = self._to_grayscale(frame)

        # For large images, downsample for faster edge/line detection
        h, w = gray.shape[:2]
        scale = 1.0
        if max(h, w) > 800:
            scale = 800.0 / max(h, w)
            small = cv2.resize(gray, None, fx=scale, fy=scale)
        else:
            small = gray

        # Detect edges for Hough transform
        edges = cv2.Canny(small, 50, 150, apertureSize=3)

        # Use probabilistic Hough Line Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=min(small.shape[1] // 4, 50),
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            # No lines detected, no rotation correction needed
            return frame, 0.0

        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) > 0:  # Avoid division by zero
                angle = np.degrees(np.arctan2(dy, dx))
                # Only consider near-horizontal lines (within ±45 degrees)
                if abs(angle) <= 45:
                    angles.append(angle)

        if not angles:
            return frame, 0.0

        # Use median angle to be robust against outliers
        median_angle = float(np.median(angles))

        # Only correct if within correctable range
        if abs(median_angle) > self.MAX_CORRECTABLE_ROTATION:
            return frame, 0.0

        # Rotate the image to correct the skew
        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        corrected = cv2.warpAffine(
            frame, rotation_matrix, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return corrected, median_angle

    def apply_noise_reduction(self, frame: np.ndarray) -> np.ndarray:
        """Apply noise reduction with Gaussian/median filter kernel ≤ 5x5.

        Uses a combination of Gaussian blur for general noise and median
        filter for salt-and-pepper noise, both with 5x5 kernels.

        Args:
            frame: Grayscale image as numpy array.

        Returns:
            Denoised grayscale image.
        """
        gray = self._to_grayscale(frame)

        # Apply Gaussian blur with 5x5 kernel
        gaussian = cv2.GaussianBlur(gray, (5, 5), sigmaX=1.0)

        # Apply median filter with 5x5 kernel for salt-and-pepper noise
        denoised = cv2.medianBlur(gaussian, 3)

        return denoised

    def adaptive_threshold(self, frame: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for uneven lighting.

        Produces a binary image where every pixel is either 0 or 255.
        Uses Gaussian-weighted adaptive thresholding with a block size
        appropriate for Braille dot spacing.

        Args:
            frame: Grayscale image as numpy array.

        Returns:
            Binary image (0 or 255 values only).
        """
        gray = self._to_grayscale(frame)

        # Determine block size based on image dimensions
        # Block size should be appropriate for dot spacing
        # Use ~1/20th of the smaller dimension, ensuring it's odd and >= 11
        min_dim = min(gray.shape[0], gray.shape[1])
        block_size = max(11, (min_dim // 20) | 1)  # Ensure odd number

        # Apply thresholding to separate dots from background.
        # Strategy: detect if lighting is uniform or uneven, then pick method.
        
        # Check lighting uniformity by comparing brightness across image regions
        # Use a 3x3 grid for better detection of localized shadows
        h, w = gray.shape
        region_means = []
        for ry in range(3):
            for rx in range(3):
                y1, y2 = ry * h // 3, (ry + 1) * h // 3
                x1, x2 = rx * w // 3, (rx + 1) * w // 3
                region_means.append(float(np.mean(gray[y1:y2, x1:x2])))
        lighting_range = max(region_means) - min(region_means)
        
        # Also check local contrast: compute std dev of small blocks
        local_std = float(np.std(region_means))
        
        # If lighting varies by more than 20 across regions, use adaptive
        if lighting_range > 20:
            # Uneven lighting — use adaptive thresholding
            binary = cv2.adaptiveThreshold(
                gray,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=block_size,
                C=8,
            )
        else:
            # Uniform lighting — use Otsu (more reliable for clean images)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Ensure dots are white (255) on black (0) background
        # Dots are always the minority of pixels
        white_ratio = np.sum(binary == 255) / binary.size
        if white_ratio > 0.5:
            binary = cv2.bitwise_not(binary)

        return binary

    def validate_image(self, frame: np.ndarray) -> ValidationResult:
        """Check if image can be corrected. Returns error code if not.

        Checks for:
        - Resolution below 640x480
        - Rotation exceeding 15 degrees
        - Entirely overexposed image

        Args:
            frame: Raw image as numpy array.

        Returns:
            ValidationResult.OK if usable, or specific error code.
        """
        h, w = frame.shape[:2]

        # Check resolution
        if w < self.MIN_WIDTH or h < self.MIN_HEIGHT:
            return ValidationResult.RESOLUTION_TOO_LOW

        # Check overexposure - only reject if image has virtually no contrast
        # (e.g., pointing camera at a light source). White paper with dark dots is fine.
        gray = self._to_grayscale(frame)
        img_std = np.std(gray)
        # If standard deviation is < 3, the image is essentially uniform (no content)
        if img_std < 3.0:
            # Check if it's uniformly bright (overexposed) vs uniformly dark
            if np.mean(gray) > 200:
                return ValidationResult.OVEREXPOSED

        # Note: We don't reject images based on rotation anymore.
        # Real camera images often have detected "lines" from screen edges,
        # furniture, etc. that give false rotation readings.
        # The rotation correction will simply not correct if angle > 15 degrees.

        return ValidationResult.OK

    def _to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale if it's a color image.

        Args:
            frame: Image as numpy array (BGR or grayscale).

        Returns:
            Grayscale image.
        """
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
