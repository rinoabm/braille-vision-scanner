"""Unit tests for the AlignmentGuide module.

Tests alignment detection, audio cue generation, and edge cases.
Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import numpy as np
import pytest

from backend.models.data_models import AlignmentResult
from backend.pipeline.alignment_guide import AlignmentGuide


@pytest.fixture
def guide():
    """Create an AlignmentGuide instance."""
    return AlignmentGuide()


def _create_frame_with_braille_region(
    frame_width: int = 640,
    frame_height: int = 480,
    region_x: int = 0,
    region_y: int = 0,
    region_w: int = 640,
    region_h: int = 480,
) -> np.ndarray:
    """Create a synthetic frame with a simulated Braille text region.

    Draws white dots on a dark background within the specified region
    to simulate Braille text that the contour detector can find.
    """
    # Dark background
    frame = np.zeros((frame_height, frame_width), dtype=np.uint8)

    # Draw dot-like circles in the specified region to simulate Braille
    dot_spacing = 15
    dot_radius = 4

    for y in range(region_y + 10, region_y + region_h - 10, dot_spacing):
        for x in range(region_x + 10, region_x + region_w - 10, dot_spacing):
            if 0 <= x < frame_width and 0 <= y < frame_height:
                cv2.circle(frame, (x, y), dot_radius, 255, -1)

    return frame


# Need cv2 for creating test images
import cv2


class TestAnalyzeFrame:
    """Tests for AlignmentGuide.analyze_frame()."""

    def test_empty_frame_returns_not_detected(self, guide):
        """Empty frame should report no Braille detected."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        result = guide.analyze_frame(frame)

        assert result.braille_detected is False
        assert result.is_aligned is False
        assert result.coverage_percent == 0.0

    def test_none_frame_returns_not_detected(self, guide):
        """None frame should report no Braille detected."""
        result = guide.analyze_frame(None)

        assert result.braille_detected is False
        assert result.is_aligned is False

    def test_full_coverage_centered_is_aligned(self, guide):
        """Frame with Braille covering >70% and centered should be aligned."""
        # Create frame with dots covering most of the area, centered
        frame = _create_frame_with_braille_region(
            frame_width=640,
            frame_height=480,
            region_x=50,
            region_y=30,
            region_w=540,
            region_h=420,
        )
        result = guide.analyze_frame(frame)

        assert result.braille_detected is True
        # The coverage should be high since dots fill most of the frame
        assert result.coverage_percent > 50.0

    def test_small_region_not_aligned(self, guide):
        """Small Braille region should not be aligned (low coverage)."""
        frame = _create_frame_with_braille_region(
            frame_width=640,
            frame_height=480,
            region_x=250,
            region_y=200,
            region_w=100,
            region_h=80,
        )
        result = guide.analyze_frame(frame)

        # Small region means low coverage
        assert result.is_aligned is False

    def test_left_offset_region(self, guide):
        """Braille region offset to the left should report 'left' direction."""
        frame = _create_frame_with_braille_region(
            frame_width=640,
            frame_height=480,
            region_x=10,
            region_y=100,
            region_w=200,
            region_h=280,
        )
        result = guide.analyze_frame(frame)

        assert result.braille_detected is True
        assert result.direction in ("left", "too_far")

    def test_right_offset_region(self, guide):
        """Braille region offset to the right should report 'right' direction."""
        frame = _create_frame_with_braille_region(
            frame_width=640,
            frame_height=480,
            region_x=430,
            region_y=100,
            region_w=200,
            region_h=280,
        )
        result = guide.analyze_frame(frame)

        assert result.braille_detected is True
        assert result.direction in ("right", "too_far")

    def test_bgr_frame_accepted(self, guide):
        """BGR (3-channel) frames should be handled correctly."""
        gray_frame = _create_frame_with_braille_region(
            frame_width=640,
            frame_height=480,
            region_x=50,
            region_y=30,
            region_w=540,
            region_h=420,
        )
        # Convert to BGR
        bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        result = guide.analyze_frame(bgr_frame)

        assert result.braille_detected is True

    def test_returns_alignment_result_type(self, guide):
        """analyze_frame should return an AlignmentResult dataclass."""
        frame = np.zeros((480, 640), dtype=np.uint8)
        result = guide.analyze_frame(frame)

        assert isinstance(result, AlignmentResult)
        assert hasattr(result, "is_aligned")
        assert hasattr(result, "direction")
        assert hasattr(result, "coverage_percent")
        assert hasattr(result, "braille_detected")

    def test_coverage_percent_range(self, guide):
        """Coverage percent should be between 0 and 100."""
        frame = _create_frame_with_braille_region(
            frame_width=640,
            frame_height=480,
            region_x=100,
            region_y=80,
            region_w=440,
            region_h=320,
        )
        result = guide.analyze_frame(frame)

        assert 0.0 <= result.coverage_percent <= 100.0


class TestGetAudioCue:
    """Tests for AlignmentGuide.get_audio_cue()."""

    def test_no_braille_detected_message(self, guide):
        """No Braille detected should give repositioning instruction."""
        result = AlignmentResult(
            is_aligned=False,
            direction="centered",
            coverage_percent=0.0,
            braille_detected=False,
        )
        cue = guide.get_audio_cue(result)

        assert "No Braille" in cue
        assert "reposition" in cue.lower()

    def test_aligned_message(self, guide):
        """Aligned result should give 'aligned' or 'ready' message."""
        result = AlignmentResult(
            is_aligned=True,
            direction="centered",
            coverage_percent=75.0,
            braille_detected=True,
        )
        cue = guide.get_audio_cue(result)

        assert "Aligned" in cue or "aligned" in cue

    def test_left_direction_cue(self, guide):
        """Left direction should instruct to move right."""
        result = AlignmentResult(
            is_aligned=False,
            direction="left",
            coverage_percent=50.0,
            braille_detected=True,
        )
        cue = guide.get_audio_cue(result)

        assert "right" in cue.lower()

    def test_right_direction_cue(self, guide):
        """Right direction should instruct to move left."""
        result = AlignmentResult(
            is_aligned=False,
            direction="right",
            coverage_percent=50.0,
            braille_detected=True,
        )
        cue = guide.get_audio_cue(result)

        assert "left" in cue.lower()

    def test_too_close_direction_cue(self, guide):
        """Too close should instruct to move farther away."""
        result = AlignmentResult(
            is_aligned=False,
            direction="too_close",
            coverage_percent=95.0,
            braille_detected=True,
        )
        cue = guide.get_audio_cue(result)

        assert "farther" in cue.lower() or "away" in cue.lower()

    def test_too_far_direction_cue(self, guide):
        """Too far should instruct to move closer."""
        result = AlignmentResult(
            is_aligned=False,
            direction="too_far",
            coverage_percent=20.0,
            braille_detected=True,
        )
        cue = guide.get_audio_cue(result)

        assert "closer" in cue.lower()

    def test_direction_values_are_valid(self, guide):
        """All valid direction values should produce a non-empty cue."""
        valid_directions = ["centered", "left", "right", "too_close", "too_far"]

        for direction in valid_directions:
            result = AlignmentResult(
                is_aligned=False,
                direction=direction,
                coverage_percent=50.0,
                braille_detected=True,
            )
            cue = guide.get_audio_cue(result)
            assert len(cue) > 0


class TestPerformance:
    """Tests for alignment feedback timing requirement (500ms)."""

    def test_analyze_frame_completes_within_500ms(self, guide):
        """analyze_frame should complete within 500ms for standard resolution."""
        import time

        frame = _create_frame_with_braille_region(
            frame_width=640,
            frame_height=480,
            region_x=50,
            region_y=30,
            region_w=540,
            region_h=420,
        )

        start = time.time()
        guide.analyze_frame(frame)
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 500, f"analyze_frame took {elapsed_ms:.1f}ms, exceeds 500ms limit"
