"""Unit tests for the BrailleDecoder module.

Tests Grade 1 Braille decoding, number/capital indicators,
unknown pattern handling, and JSON serialization/deserialization.
"""

import json

import pytest

from backend.models.data_models import BrailleCell, DecodedText, SegmentationResult
from backend.pipeline.braille_decoder import (
    BRAILLE_DIGIT_TABLE,
    BRAILLE_LETTER_TABLE,
    BRAILLE_PUNCTUATION_TABLE,
    UNKNOWN_CHAR,
    BrailleDecoder,
)


@pytest.fixture
def decoder():
    """Create a BrailleDecoder instance."""
    return BrailleDecoder()


def make_cell(
    dots: list[int],
    confidence: float = 0.95,
    grid_pos: tuple[int, int] = (0, 0),
) -> BrailleCell:
    """Helper to create a BrailleCell with uniform confidence."""
    return BrailleCell(
        dot_positions=dots,
        confidence_scores=[confidence] * len(dots),
        grid_position=grid_pos,
    )


def make_segmentation(lines: list[list[BrailleCell]]) -> SegmentationResult:
    """Helper to create a SegmentationResult from lines."""
    all_cells = [cell for line in lines for cell in line]
    return SegmentationResult(
        cells=all_cells,
        lines=lines,
        ambiguous_regions=[],
        processing_time_ms=0.0,
    )


class TestDecodeCell:
    """Tests for decode_cell (stateless single-cell decoding)."""

    def test_decode_letter_a(self, decoder):
        """Letter 'a' is dot 1."""
        cell = make_cell([1])
        assert decoder.decode_cell(cell) == "a"

    def test_decode_letter_z(self, decoder):
        """Letter 'z' is dots 1,3,5,6."""
        cell = make_cell([1, 3, 5, 6])
        assert decoder.decode_cell(cell) == "z"

    def test_decode_all_letters(self, decoder):
        """All 26 letters should decode correctly."""
        expected = "abcdefghijklmnopqrstuvwxyz"
        patterns = [
            [1], [1, 2], [1, 4], [1, 4, 5], [1, 5],
            [1, 2, 4], [1, 2, 4, 5], [1, 2, 5], [2, 4], [2, 4, 5],
            [1, 3], [1, 2, 3], [1, 3, 4], [1, 3, 4, 5], [1, 3, 5],
            [1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 5], [2, 3, 4], [2, 3, 4, 5],
            [1, 3, 6], [1, 2, 3, 6], [2, 4, 5, 6], [1, 3, 4, 6],
            [1, 3, 4, 5, 6], [1, 3, 5, 6],
        ]
        for i, pattern in enumerate(patterns):
            cell = make_cell(pattern)
            assert decoder.decode_cell(cell) == expected[i], (
                f"Pattern {pattern} should decode to '{expected[i]}'"
            )

    def test_decode_space(self, decoder):
        """Empty cell is a space."""
        cell = make_cell([])
        assert decoder.decode_cell(cell) == " "

    def test_decode_period(self, decoder):
        """Period is dots 2,5,6."""
        cell = make_cell([2, 5, 6])
        assert decoder.decode_cell(cell) == "."

    def test_decode_comma(self, decoder):
        """Comma is dot 2."""
        cell = make_cell([2])
        assert decoder.decode_cell(cell) == ","

    def test_decode_question_mark(self, decoder):
        """Question mark is dots 2,3,6."""
        cell = make_cell([2, 3, 6])
        assert decoder.decode_cell(cell) == "?"

    def test_decode_exclamation(self, decoder):
        """Exclamation mark is dots 2,3,5."""
        cell = make_cell([2, 3, 5])
        assert decoder.decode_cell(cell) == "!"

    def test_decode_apostrophe(self, decoder):
        """Apostrophe is dot 3."""
        cell = make_cell([3])
        assert decoder.decode_cell(cell) == "'"

    def test_decode_hyphen(self, decoder):
        """Hyphen is dots 3,6."""
        cell = make_cell([3, 6])
        assert decoder.decode_cell(cell) == "-"

    def test_decode_unknown_pattern(self, decoder):
        """Unknown pattern should return U+FFFD."""
        cell = make_cell([1, 2, 3, 4, 5, 6])  # All dots - not a valid character
        assert decoder.decode_cell(cell) == UNKNOWN_CHAR

    def test_decode_number_indicator_returns_empty(self, decoder):
        """Number indicator (dots 3,4,5,6) returns empty string in stateless mode."""
        cell = make_cell([3, 4, 5, 6])
        assert decoder.decode_cell(cell) == ""

    def test_decode_capital_indicator_returns_empty(self, decoder):
        """Capital indicator (dot 6) returns empty string in stateless mode."""
        cell = make_cell([6])
        assert decoder.decode_cell(cell) == ""


class TestDecodeWithIndicators:
    """Tests for full decode() with number and capital indicators."""

    def test_number_indicator_activates_numeric_mode(self, decoder):
        """Number indicator followed by a-j patterns should produce digits."""
        # #abc -> 123
        line = [
            make_cell([3, 4, 5, 6]),  # number indicator
            make_cell([1]),            # 1 (a pattern)
            make_cell([1, 2]),         # 2 (b pattern)
            make_cell([1, 4]),         # 3 (c pattern)
        ]
        result = decoder.decode(make_segmentation([line]))
        assert result.text == "123"

    def test_numeric_mode_terminated_by_space(self, decoder):
        """Space terminates numeric mode, returning to letter mode."""
        # #ab <space> ab -> 12 ab
        line = [
            make_cell([3, 4, 5, 6]),  # number indicator
            make_cell([1]),            # 1
            make_cell([1, 2]),         # 2
            make_cell([]),             # space (terminates numeric)
            make_cell([1]),            # a
            make_cell([1, 2]),         # b
        ]
        result = decoder.decode(make_segmentation([line]))
        assert result.text == "12 ab"

    def test_numeric_mode_terminated_by_letter_indicator(self, decoder):
        """Letter indicator (dots 5,6) terminates numeric mode."""
        # #ab ;ab -> 12ab (letter indicator consumed, not output)
        line = [
            make_cell([3, 4, 5, 6]),  # number indicator
            make_cell([1]),            # 1
            make_cell([1, 2]),         # 2
            make_cell([5, 6]),         # letter indicator
            make_cell([1]),            # a
            make_cell([1, 2]),         # b
        ]
        result = decoder.decode(make_segmentation([line]))
        assert result.text == "12ab"

    def test_capital_indicator_capitalizes_next(self, decoder):
        """Capital indicator (dot 6) capitalizes the next letter."""
        # ^a -> A
        line = [
            make_cell([6]),   # capital indicator
            make_cell([1]),   # A (capitalized)
        ]
        result = decoder.decode(make_segmentation([line]))
        assert result.text == "A"

    def test_capital_indicator_only_affects_next(self, decoder):
        """Capital indicator only capitalizes the immediately following letter."""
        # ^ab -> Ab
        line = [
            make_cell([6]),    # capital indicator
            make_cell([1]),    # A
            make_cell([1, 2]), # b (not capitalized)
        ]
        result = decoder.decode(make_segmentation([line]))
        assert result.text == "Ab"

    def test_multiple_capital_indicators(self, decoder):
        """Multiple capital indicators each capitalize their next letter."""
        # ^a^b -> AB
        line = [
            make_cell([6]),    # capital indicator
            make_cell([1]),    # A
            make_cell([6]),    # capital indicator
            make_cell([1, 2]), # B
        ]
        result = decoder.decode(make_segmentation([line]))
        assert result.text == "AB"

    def test_all_digits(self, decoder):
        """All digits 0-9 should decode correctly with number indicator."""
        # #abcdefghij -> 1234567890
        line = [
            make_cell([3, 4, 5, 6]),  # number indicator
            make_cell([1]),            # 1
            make_cell([1, 2]),         # 2
            make_cell([1, 4]),         # 3
            make_cell([1, 4, 5]),      # 4
            make_cell([1, 5]),         # 5
            make_cell([1, 2, 4]),      # 6
            make_cell([1, 2, 4, 5]),   # 7
            make_cell([1, 2, 5]),      # 8
            make_cell([2, 4]),         # 9
            make_cell([2, 4, 5]),      # 0
        ]
        result = decoder.decode(make_segmentation([line]))
        assert result.text == "1234567890"

    def test_unknown_pattern_in_output(self, decoder):
        """Unknown patterns produce U+FFFD and are tracked in unrecognized_indices."""
        line = [
            make_cell([1]),                # a
            make_cell([1, 2, 3, 4, 5, 6]), # unknown
            make_cell([1, 2]),             # b
        ]
        result = decoder.decode(make_segmentation([line]))
        assert result.text == f"a{UNKNOWN_CHAR}b"
        assert 1 in result.unrecognized_indices

    def test_multiline_decode(self, decoder):
        """Multiple lines are decoded separately and joined with newlines."""
        line1 = [make_cell([1]), make_cell([1, 2])]  # ab
        line2 = [make_cell([1, 4]), make_cell([1, 4, 5])]  # cd
        result = decoder.decode(make_segmentation([line1, line2]))
        assert result.text == "ab\ncd"

    def test_empty_line(self, decoder):
        """Empty lines produce empty strings."""
        result = decoder.decode(make_segmentation([[]]))
        assert result.text == ""

    def test_confidence_scores_computed(self, decoder):
        """Per-character confidence should be computed from cell confidence scores."""
        line = [
            make_cell([1], confidence=0.9),
            make_cell([1, 2], confidence=0.8),
        ]
        result = decoder.decode(make_segmentation([line]))
        assert len(result.per_character_confidence) == 2
        assert abs(result.per_character_confidence[0] - 0.9) < 0.001
        assert abs(result.per_character_confidence[1] - 0.8) < 0.001

    def test_line_confidences_computed(self, decoder):
        """Line confidences should be the average of character confidences."""
        line = [
            make_cell([1], confidence=0.8),
            make_cell([1, 2], confidence=0.6),
        ]
        result = decoder.decode(make_segmentation([line]))
        assert len(result.line_confidences) == 1
        assert abs(result.line_confidences[0] - 0.7) < 0.001


class TestSerializeCell:
    """Tests for serialize_cell method."""

    def test_serialize_basic_cell(self, decoder):
        """Serialization should produce valid JSON with correct fields."""
        cell = BrailleCell(
            dot_positions=[1, 2, 4],
            confidence_scores=[0.95, 0.88, 0.92],
            grid_position=(0, 0),
        )
        result = decoder.serialize_cell(cell)
        data = json.loads(result)
        assert data["dot_positions"] == [1, 2, 4]
        assert data["confidence_scores"] == [0.95, 0.88, 0.92]

    def test_serialize_empty_cell(self, decoder):
        """Empty cell (space) should serialize to empty lists."""
        cell = BrailleCell(
            dot_positions=[],
            confidence_scores=[],
            grid_position=(0, 0),
        )
        result = decoder.serialize_cell(cell)
        data = json.loads(result)
        assert data["dot_positions"] == []
        assert data["confidence_scores"] == []

    def test_serialize_produces_valid_json(self, decoder):
        """Output should always be valid JSON."""
        cell = make_cell([1, 3, 5])
        result = decoder.serialize_cell(cell)
        # Should not raise
        json.loads(result)


class TestDeserializeCell:
    """Tests for deserialize_cell method."""

    def test_deserialize_valid_json(self, decoder):
        """Valid JSON should produce a correct BrailleCell."""
        json_str = '{"dot_positions": [1, 2, 4], "confidence_scores": [0.95, 0.88, 0.92]}'
        cell = decoder.deserialize_cell(json_str)
        assert cell.dot_positions == [1, 2, 4]
        assert len(cell.confidence_scores) == 3
        assert abs(cell.confidence_scores[0] - 0.95) < 0.0001
        assert abs(cell.confidence_scores[1] - 0.88) < 0.0001
        assert abs(cell.confidence_scores[2] - 0.92) < 0.0001

    def test_deserialize_empty_cell(self, decoder):
        """Empty lists should produce a blank cell."""
        json_str = '{"dot_positions": [], "confidence_scores": []}'
        cell = decoder.deserialize_cell(json_str)
        assert cell.dot_positions == []
        assert cell.confidence_scores == []

    def test_deserialize_invalid_json_syntax(self, decoder):
        """Invalid JSON should raise ValueError with syntax error message."""
        with pytest.raises(ValueError, match="Invalid JSON syntax"):
            decoder.deserialize_cell("not json at all")

    def test_deserialize_missing_dot_positions(self, decoder):
        """Missing dot_positions field should raise ValueError."""
        json_str = '{"confidence_scores": [0.9]}'
        with pytest.raises(ValueError, match="Missing required field.*dot_positions"):
            decoder.deserialize_cell(json_str)

    def test_deserialize_missing_confidence_scores(self, decoder):
        """Missing confidence_scores field should raise ValueError."""
        json_str = '{"dot_positions": [1]}'
        with pytest.raises(ValueError, match="Missing required field.*confidence_scores"):
            decoder.deserialize_cell(json_str)

    def test_deserialize_length_mismatch(self, decoder):
        """Mismatched list lengths should raise ValueError."""
        json_str = '{"dot_positions": [1, 2], "confidence_scores": [0.9]}'
        with pytest.raises(ValueError, match="List length mismatch"):
            decoder.deserialize_cell(json_str)

    def test_deserialize_dot_position_out_of_range(self, decoder):
        """Dot position outside 1-8 should raise ValueError."""
        json_str = '{"dot_positions": [0], "confidence_scores": [0.9]}'
        with pytest.raises(ValueError, match="outside valid range 1-8"):
            decoder.deserialize_cell(json_str)

        json_str = '{"dot_positions": [9], "confidence_scores": [0.9]}'
        with pytest.raises(ValueError, match="outside valid range 1-8"):
            decoder.deserialize_cell(json_str)

    def test_deserialize_confidence_out_of_range(self, decoder):
        """Confidence score outside 0.0-1.0 should raise ValueError."""
        json_str = '{"dot_positions": [1], "confidence_scores": [-0.1]}'
        with pytest.raises(ValueError, match="outside valid range 0.0-1.0"):
            decoder.deserialize_cell(json_str)

        json_str = '{"dot_positions": [1], "confidence_scores": [1.1]}'
        with pytest.raises(ValueError, match="outside valid range 0.0-1.0"):
            decoder.deserialize_cell(json_str)

    def test_deserialize_non_integer_dot_position(self, decoder):
        """Non-integer dot position should raise ValueError."""
        json_str = '{"dot_positions": [1.5], "confidence_scores": [0.9]}'
        with pytest.raises(ValueError, match="expected integer"):
            decoder.deserialize_cell(json_str)

    def test_deserialize_non_object_json(self, decoder):
        """JSON that's not an object should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid JSON structure"):
            decoder.deserialize_cell("[1, 2, 3]")

    def test_deserialize_dot_positions_not_list(self, decoder):
        """dot_positions that's not a list should raise ValueError."""
        json_str = '{"dot_positions": "not a list", "confidence_scores": []}'
        with pytest.raises(ValueError, match="must be a list"):
            decoder.deserialize_cell(json_str)

    def test_deserialize_confidence_scores_not_list(self, decoder):
        """confidence_scores that's not a list should raise ValueError."""
        json_str = '{"dot_positions": [], "confidence_scores": "not a list"}'
        with pytest.raises(ValueError, match="must be a list"):
            decoder.deserialize_cell(json_str)


class TestRoundTrip:
    """Tests for serialization round-trip property."""

    def test_round_trip_basic(self, decoder):
        """Serialize then deserialize should produce equivalent cell."""
        original = BrailleCell(
            dot_positions=[1, 2, 4],
            confidence_scores=[0.95, 0.88, 0.92],
            grid_position=(0, 0),
        )
        json_str = decoder.serialize_cell(original)
        restored = decoder.deserialize_cell(json_str)
        assert restored.dot_positions == original.dot_positions
        for orig, rest in zip(original.confidence_scores, restored.confidence_scores):
            assert abs(orig - rest) < 0.0001

    def test_round_trip_empty_cell(self, decoder):
        """Empty cell round-trip should work."""
        original = BrailleCell(
            dot_positions=[],
            confidence_scores=[],
            grid_position=(0, 0),
        )
        json_str = decoder.serialize_cell(original)
        restored = decoder.deserialize_cell(json_str)
        assert restored.dot_positions == []
        assert restored.confidence_scores == []

    def test_round_trip_all_dots(self, decoder):
        """Cell with all 6 dots should round-trip correctly."""
        original = BrailleCell(
            dot_positions=[1, 2, 3, 4, 5, 6],
            confidence_scores=[0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
            grid_position=(0, 0),
        )
        json_str = decoder.serialize_cell(original)
        restored = decoder.deserialize_cell(json_str)
        assert restored.dot_positions == original.dot_positions
        for orig, rest in zip(original.confidence_scores, restored.confidence_scores):
            assert abs(orig - rest) < 0.0001
