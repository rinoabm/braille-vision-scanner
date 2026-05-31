"""Braille Decoder module for the Braille Vision Scanner pipeline.

Maps segmented Braille cells to English characters using Grade 1 Braille alphabet.
Supports letters a-z, digits 0-9, common punctuation, number/capital indicators,
and JSON serialization/deserialization of cell patterns.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 11.1, 11.2, 11.3, 11.4, 11.5
"""

import json
from typing import Union

from backend.models.data_models import BrailleCell, DecodedText, SegmentationResult


# Grade 1 Braille lookup table: frozenset of dot positions -> character
# Dot positions use standard Braille numbering (1-6):
#   1 4
#   2 5
#   3 6
BRAILLE_LETTER_TABLE: dict[frozenset[int], str] = {
    frozenset([1]): "a",
    frozenset([1, 2]): "b",
    frozenset([1, 4]): "c",
    frozenset([1, 4, 5]): "d",
    frozenset([1, 5]): "e",
    frozenset([1, 2, 4]): "f",
    frozenset([1, 2, 4, 5]): "g",
    frozenset([1, 2, 5]): "h",
    frozenset([2, 4]): "i",
    frozenset([2, 4, 5]): "j",
    frozenset([1, 3]): "k",
    frozenset([1, 2, 3]): "l",
    frozenset([1, 3, 4]): "m",
    frozenset([1, 3, 4, 5]): "n",
    frozenset([1, 3, 5]): "o",
    frozenset([1, 2, 3, 4]): "p",
    frozenset([1, 2, 3, 4, 5]): "q",
    frozenset([1, 2, 3, 5]): "r",
    frozenset([2, 3, 4]): "s",
    frozenset([2, 3, 4, 5]): "t",
    frozenset([1, 3, 6]): "u",
    frozenset([1, 2, 3, 6]): "v",
    frozenset([2, 4, 5, 6]): "w",
    frozenset([1, 3, 4, 6]): "x",
    frozenset([1, 3, 4, 5, 6]): "y",
    frozenset([1, 3, 5, 6]): "z",
}

# Punctuation lookup table
BRAILLE_PUNCTUATION_TABLE: dict[frozenset[int], str] = {
    frozenset([2, 5, 6]): ".",
    frozenset([2]): ",",
    frozenset([2, 3, 6]): "?",
    frozenset([2, 3, 5]): "!",
    frozenset([3]): "'",
    frozenset([3, 6]): "-",
}

# Number indicator: dots 3-4-5-6
NUMBER_INDICATOR = frozenset([3, 4, 5, 6])

# Capital indicator: dot 6
CAPITAL_INDICATOR = frozenset([6])

# Letter indicator: dots 5-6 (terminates numeric mode)
LETTER_INDICATOR = frozenset([5, 6])

# Digit mapping: same patterns as letters a-j map to 1-9, 0
BRAILLE_DIGIT_TABLE: dict[frozenset[int], str] = {
    frozenset([1]): "1",         # a -> 1
    frozenset([1, 2]): "2",     # b -> 2
    frozenset([1, 4]): "3",     # c -> 3
    frozenset([1, 4, 5]): "4",  # d -> 4
    frozenset([1, 5]): "5",     # e -> 5
    frozenset([1, 2, 4]): "6",  # f -> 6
    frozenset([1, 2, 4, 5]): "7",  # g -> 7
    frozenset([1, 2, 5]): "8",  # h -> 8
    frozenset([2, 4]): "9",     # i -> 9
    frozenset([2, 4, 5]): "0",  # j -> 0
}

# Placeholder for unrecognized patterns
UNKNOWN_CHAR = "\ufffd"


class BrailleDecoder:
    """Decodes Braille cells to English text using Grade 1 Braille alphabet.

    Supports:
    - Letters a-z (with capital indicator for uppercase)
    - Digits 0-9 (with number indicator prefix)
    - Punctuation: period, comma, question mark, exclamation mark, apostrophe, hyphen
    - Space (empty cell)
    - Unknown patterns produce U+FFFD placeholder

    Also provides JSON serialization/deserialization for BrailleCell objects.
    """

    def __init__(self) -> None:
        """Initialize the decoder with lookup tables."""
        self._letter_table = BRAILLE_LETTER_TABLE
        self._punctuation_table = BRAILLE_PUNCTUATION_TABLE
        self._digit_table = BRAILLE_DIGIT_TABLE

    def decode(self, cells: SegmentationResult) -> DecodedText:
        """Decode all cells in a SegmentationResult to English text.

        Processes cells line by line, maintaining state for number and capital
        indicators within each line. State resets at line boundaries.

        Completes within 50ms per line of 40 cells.

        Args:
            cells: SegmentationResult containing lines of BrailleCell objects.

        Returns:
            DecodedText with decoded text, per-character confidence,
            unrecognized indices, and line confidences.
        """
        text_parts: list[str] = []
        per_character_confidence: list[float] = []
        unrecognized_indices: list[int] = []
        line_confidences: list[float] = []
        char_index = 0

        for line_idx, line in enumerate(cells.lines):
            line_chars, line_conf, line_unrecognized = self._decode_line(
                line, char_index
            )
            line_str = "".join(line_chars)
            text_parts.append(line_str)
            per_character_confidence.extend(line_conf)
            unrecognized_indices.extend(line_unrecognized)

            # Compute line confidence as average of character confidences
            if line_conf:
                line_confidences.append(sum(line_conf) / len(line_conf))
            else:
                line_confidences.append(0.0)

            char_index += len(line_str)

        full_text = "\n".join(text_parts)

        return DecodedText(
            text=full_text,
            per_character_confidence=per_character_confidence,
            unrecognized_indices=unrecognized_indices,
            line_confidences=line_confidences,
        )

    def _decode_line(
        self, line: list[BrailleCell], start_index: int
    ) -> tuple[list[str], list[float], list[int]]:
        """Decode a single line of Braille cells.

        Maintains number indicator and capital indicator state within the line.

        Args:
            line: List of BrailleCell objects in reading order.
            start_index: Starting character index for tracking unrecognized positions.

        Returns:
            Tuple of (decoded characters list, confidence scores, unrecognized indices).
        """
        chars: list[str] = []
        confidences: list[float] = []
        unrecognized: list[int] = []

        numeric_mode = False
        capitalize_next = False
        char_index = start_index

        for cell in line:
            dot_set = frozenset(cell.dot_positions)

            # Check for number indicator
            if dot_set == NUMBER_INDICATOR:
                numeric_mode = True
                continue

            # Check for capital indicator
            if dot_set == CAPITAL_INDICATOR:
                capitalize_next = True
                continue

            # Check for letter indicator (terminates numeric mode)
            if dot_set == LETTER_INDICATOR:
                numeric_mode = False
                continue

            # Space (empty cell) - terminates numeric mode
            if len(cell.dot_positions) == 0:
                chars.append(" ")
                confidences.append(1.0)
                numeric_mode = False
                capitalize_next = False
                char_index += 1
                continue

            # Decode the cell based on current mode
            decoded_char = self._decode_single_cell(
                dot_set, numeric_mode, capitalize_next
            )

            chars.append(decoded_char)

            # Compute confidence as average of cell's confidence scores
            if cell.confidence_scores:
                conf = sum(cell.confidence_scores) / len(cell.confidence_scores)
            else:
                conf = 0.0
            confidences.append(conf)

            # Track unrecognized characters
            if decoded_char == UNKNOWN_CHAR:
                unrecognized.append(char_index)

            # Reset capitalize flag after use
            capitalize_next = False
            char_index += 1

        return chars, confidences, unrecognized

    def _decode_single_cell(
        self, dot_set: frozenset[int], numeric_mode: bool, capitalize: bool
    ) -> str:
        """Decode a single cell's dot pattern to a character.

        Args:
            dot_set: Frozenset of active dot positions.
            numeric_mode: Whether number indicator is active.
            capitalize: Whether capital indicator was preceding.

        Returns:
            The decoded character, or U+FFFD if unrecognized.
        """
        if numeric_mode:
            # In numeric mode, try digit table first
            if dot_set in self._digit_table:
                return self._digit_table[dot_set]
            # Punctuation is still valid in numeric mode
            if dot_set in self._punctuation_table:
                return self._punctuation_table[dot_set]
            # Unknown pattern in numeric mode
            return UNKNOWN_CHAR

        # Try letter table
        if dot_set in self._letter_table:
            char = self._letter_table[dot_set]
            if capitalize:
                return char.upper()
            return char

        # Try punctuation table
        if dot_set in self._punctuation_table:
            return self._punctuation_table[dot_set]

        # Unknown pattern
        return UNKNOWN_CHAR

    def decode_cell(self, cell: BrailleCell) -> str:
        """Map a single cell to its English character (stateless).

        This method decodes a cell without considering context (no indicator state).
        For context-aware decoding, use decode() with a full SegmentationResult.

        Args:
            cell: A BrailleCell to decode.

        Returns:
            The decoded character, or U+FFFD if the pattern is unrecognized.
        """
        dot_set = frozenset(cell.dot_positions)

        # Space (empty cell)
        if len(cell.dot_positions) == 0:
            return " "

        # Number indicator
        if dot_set == NUMBER_INDICATOR:
            return ""

        # Capital indicator
        if dot_set == CAPITAL_INDICATOR:
            return ""

        # Letter indicator
        if dot_set == LETTER_INDICATOR:
            return ""

        # Try letter table
        if dot_set in self._letter_table:
            return self._letter_table[dot_set]

        # Try punctuation table
        if dot_set in self._punctuation_table:
            return self._punctuation_table[dot_set]

        # Unknown pattern
        return UNKNOWN_CHAR

    def serialize_cell(self, cell: BrailleCell) -> str:
        """Serialize a BrailleCell to JSON string.

        Produces JSON with dot_positions and confidence_scores fields.
        Format: {"dot_positions": [1, 2, 4], "confidence_scores": [0.95, 0.88, 0.92]}

        Args:
            cell: The BrailleCell to serialize.

        Returns:
            JSON string representation of the cell.
        """
        data = {
            "dot_positions": cell.dot_positions,
            "confidence_scores": cell.confidence_scores,
        }
        return json.dumps(data)

    def deserialize_cell(self, json_str: str) -> BrailleCell:
        """Deserialize a JSON string back to a BrailleCell.

        Validates the JSON structure and field values:
        - Must be valid JSON
        - Must have "dot_positions" (list of ints 1-8)
        - Must have "confidence_scores" (list of floats 0.0-1.0)
        - Lists must have the same length

        Args:
            json_str: JSON string to deserialize.

        Returns:
            A BrailleCell object reconstructed from the JSON.

        Raises:
            ValueError: With descriptive message identifying the specific
                malformation type if validation fails.
        """
        # Parse JSON
        try:
            data = json.loads(json_str)
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"Invalid JSON syntax: {e}")

        # Check it's a dict/object
        if not isinstance(data, dict):
            raise ValueError(
                "Invalid JSON structure: expected an object, "
                f"got {type(data).__name__}"
            )

        # Check required fields exist
        if "dot_positions" not in data:
            raise ValueError("Missing required field: 'dot_positions'")
        if "confidence_scores" not in data:
            raise ValueError("Missing required field: 'confidence_scores'")

        dot_positions = data["dot_positions"]
        confidence_scores = data["confidence_scores"]

        # Validate dot_positions is a list
        if not isinstance(dot_positions, list):
            raise ValueError(
                "Invalid field type: 'dot_positions' must be a list, "
                f"got {type(dot_positions).__name__}"
            )

        # Validate confidence_scores is a list
        if not isinstance(confidence_scores, list):
            raise ValueError(
                "Invalid field type: 'confidence_scores' must be a list, "
                f"got {type(confidence_scores).__name__}"
            )

        # Validate lists have same length
        if len(dot_positions) != len(confidence_scores):
            raise ValueError(
                "List length mismatch: 'dot_positions' has "
                f"{len(dot_positions)} elements but 'confidence_scores' has "
                f"{len(confidence_scores)} elements"
            )

        # Validate dot_positions values (integers in range 1-8)
        for i, pos in enumerate(dot_positions):
            if not isinstance(pos, int):
                raise ValueError(
                    f"Invalid dot_positions[{i}]: expected integer, "
                    f"got {type(pos).__name__}"
                )
            if pos < 1 or pos > 8:
                raise ValueError(
                    f"Invalid dot_positions[{i}]: value {pos} is outside "
                    "valid range 1-8"
                )

        # Validate confidence_scores values (floats in range 0.0-1.0)
        for i, score in enumerate(confidence_scores):
            if not isinstance(score, (int, float)):
                raise ValueError(
                    f"Invalid confidence_scores[{i}]: expected number, "
                    f"got {type(score).__name__}"
                )
            if score < 0.0 or score > 1.0:
                raise ValueError(
                    f"Invalid confidence_scores[{i}]: value {score} is outside "
                    "valid range 0.0-1.0"
                )

        # Construct BrailleCell with default grid_position
        return BrailleCell(
            dot_positions=dot_positions,
            confidence_scores=[float(s) for s in confidence_scores],
            grid_position=(0, 0),
            is_ambiguous=False,
        )
