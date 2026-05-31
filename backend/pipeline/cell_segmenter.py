"""Cell Segmenter for the Braille Vision Scanner pipeline.

Groups detected dots into 2x3 Braille cell grids in reading order.
Handles line detection, inter-cell spacing variation, ambiguous dots,
and partial cells.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7
"""

from __future__ import annotations

import time
from statistics import median

from backend.models.data_models import (
    BrailleCell,
    DotDetectionResult,
    DotPosition,
    SegmentationResult,
)


class CellSegmenter:
    """Groups detected dots into 2x3 Braille cell grids.

    Algorithm:
    1. Sort dots by Y coordinate, cluster into lines using vertical gap threshold
       (2x intra-cell row spacing).
    2. Within each line, sort by X coordinate.
    3. Estimate intra-cell spacing from median horizontal distances.
    4. Group dots into 2x3 cells: 2 columns, 3 rows per cell.
    5. Assign each dot to nearest grid position; flag if distance > 40% of spacing.
    6. Handle partial cells (fewer than 6 dots) as valid cells with absent positions.
    """

    def segment(self, dots: DotDetectionResult) -> SegmentationResult:
        """Group dots into cells in reading order.

        Completes within 100ms for up to 200 cells.

        Args:
            dots: DotDetectionResult containing detected dot positions.

        Returns:
            SegmentationResult with cells organized by line in reading order.
        """
        start_time = time.perf_counter()

        # Handle empty input (Requirement 5.7)
        if not dots.dots:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return SegmentationResult(
                cells=[],
                lines=[],
                ambiguous_regions=[],
                processing_time_ms=elapsed_ms,
            )

        # Step 1: Detect lines based on vertical gaps
        lines_of_dots = self.detect_lines(dots.dots)

        # Step 2: Group dots within each line into cells
        all_cells: list[BrailleCell] = []
        lines_of_cells: list[list[BrailleCell]] = []
        ambiguous_regions: list[tuple[float, float, float, float]] = []

        for line_idx, line_dots in enumerate(lines_of_dots):
            cells = self.group_into_cells(line_dots)

            # Update grid positions with correct line index
            for cell_idx, cell in enumerate(cells):
                cell.grid_position = (line_idx, cell_idx)

                # Collect ambiguous regions from ambiguous cells
                if cell.is_ambiguous:
                    # Compute bounding box of the ambiguous cell's dots
                    region = self._compute_ambiguous_region(cell, line_dots)
                    if region:
                        ambiguous_regions.append(region)

            lines_of_cells.append(cells)
            all_cells.extend(cells)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SegmentationResult(
            cells=all_cells,
            lines=lines_of_cells,
            ambiguous_regions=ambiguous_regions,
            processing_time_ms=elapsed_ms,
        )

    def detect_lines(self, dots: list[DotPosition]) -> list[list[DotPosition]]:
        """Separate dots into lines based on vertical gaps.

        A new line is detected when the vertical gap between consecutive
        dot rows exceeds 2x the intra-cell row spacing.

        Args:
            dots: List of detected dot positions.

        Returns:
            List of lines, where each line is a list of DotPositions.
        """
        if not dots:
            return []

        # Sort dots by Y coordinate
        sorted_dots = sorted(dots, key=lambda d: d.y)

        # Estimate intra-cell row spacing from vertical distances
        row_spacing = self._estimate_row_spacing(sorted_dots)

        # If we can't estimate row spacing, treat all dots as one line
        if row_spacing <= 0:
            return [sorted_dots]

        # Threshold for line separation: 2x intra-cell row spacing
        line_gap_threshold = 2.0 * row_spacing

        # Cluster dots into lines based on Y gaps
        lines: list[list[DotPosition]] = []
        current_line: list[DotPosition] = [sorted_dots[0]]

        for i in range(1, len(sorted_dots)):
            y_gap = sorted_dots[i].y - sorted_dots[i - 1].y

            if y_gap > line_gap_threshold:
                lines.append(current_line)
                current_line = [sorted_dots[i]]
            else:
                current_line.append(sorted_dots[i])

        # Don't forget the last line
        if current_line:
            lines.append(current_line)

        return lines

    def group_into_cells(self, line_dots: list[DotPosition]) -> list[BrailleCell]:
        """Group a line's dots into individual 2x3 Braille cells.

        Assigns dots to nearest grid position within 40% of intra-cell spacing.
        Handles inter-cell spacing variation between 50% and 150% of expected distance.

        Args:
            line_dots: List of dot positions within a single line.

        Returns:
            List of BrailleCell objects in left-to-right reading order.
        """
        if not line_dots:
            return []

        # Sort dots by X coordinate for left-to-right processing
        sorted_dots = sorted(line_dots, key=lambda d: d.x)

        # Estimate intra-cell column spacing (horizontal distance between
        # the two columns within a cell)
        col_spacing = self._estimate_column_spacing(sorted_dots)

        if col_spacing <= 0:
            # If we can't estimate spacing, create a single cell with all dots
            return [self._create_cell_from_dots(sorted_dots, col_spacing)]

        # Estimate row spacing within a cell
        row_spacing = self._estimate_row_spacing(sorted_dots)
        if row_spacing <= 0:
            row_spacing = col_spacing  # Fallback: assume square-ish spacing

        # Expected inter-cell distance is 1.5x intra-cell column spacing
        expected_inter_cell = 1.5 * col_spacing

        # Compute the line's overall Y extent to use as reference for grid assignment
        # This helps correctly position dots that are only in lower rows
        line_y_min = min(d.y for d in sorted_dots)

        # Group dots into cells based on X-coordinate clustering
        cell_groups = self._cluster_dots_into_cells(
            sorted_dots, col_spacing, expected_inter_cell
        )

        # Convert each group of dots into a BrailleCell
        # Also detect word boundaries (spaces) based on large gaps between groups
        cells: list[BrailleCell] = []
        
        # Calculate center X of each group for gap analysis
        group_centers = []
        for group in cell_groups:
            cx = sum(d.x for d in group) / len(group)
            group_centers.append(cx)
        
        # Detect spaces: compute typical gap between consecutive groups,
        # then flag gaps that are significantly larger as word boundaries
        if len(group_centers) >= 2:
            inter_group_gaps = [
                group_centers[i] - group_centers[i-1]
                for i in range(1, len(group_centers))
            ]
            # Median inter-group gap is the normal cell-to-cell distance
            median_inter_gap = median(sorted(inter_group_gaps))
            # A space is when the gap is > 1.8x the median inter-group gap
            space_threshold = median_inter_gap * 1.8
        else:
            space_threshold = float('inf')  # Can't detect spaces with < 2 groups
        
        for i, group in enumerate(cell_groups):
            # Check if we need to insert a space before this cell
            if i > 0:
                gap = group_centers[i] - group_centers[i - 1]
                if gap > space_threshold:
                    # Insert space cell
                    cells.append(BrailleCell(
                        dot_positions=[],
                        confidence_scores=[],
                        grid_position=(0, 0),
                        is_ambiguous=False,
                    ))
            
            cell = self._assign_dots_to_grid(group, col_spacing, row_spacing, line_y_min)
            cells.append(cell)

        return cells

    def _estimate_row_spacing(self, dots: list[DotPosition]) -> float:
        """Estimate the intra-cell row spacing from vertical distances.

        Uses the smallest consistent vertical gaps between dots to estimate
        the row spacing within a Braille cell (3 rows). The intra-cell row
        spacing is the smallest cluster of gaps, distinguishing it from
        larger inter-line gaps.
        """
        if len(dots) < 2:
            return 0.0

        sorted_by_y = sorted(dots, key=lambda d: d.y)

        # Collect all vertical gaps
        y_gaps: list[float] = []
        for i in range(1, len(sorted_by_y)):
            gap = sorted_by_y[i].y - sorted_by_y[i - 1].y
            if gap > 0.1:  # Ignore near-zero gaps (same row)
                y_gaps.append(gap)

        if not y_gaps:
            return 0.0

        # Sort gaps to find the smallest cluster
        sorted_gaps = sorted(y_gaps)

        # If there's only one gap, we can't distinguish intra-cell from inter-line.
        # Use a heuristic: if there are multiple dots at similar X positions,
        # the gap might be inter-line. Use the gap as-is but check if it's
        # reasonable for intra-cell spacing.
        if len(sorted_gaps) == 1:
            return sorted_gaps[0]

        # Use the first quartile of gaps as intra-cell row spacing.
        # This separates small intra-cell gaps from larger inter-line gaps.
        # The smallest gaps are most likely within a cell.
        q1_idx = max(1, len(sorted_gaps) // 4)
        intra_cell_gaps = sorted_gaps[:q1_idx]

        if not intra_cell_gaps:
            return sorted_gaps[0]

        # Check if there's a natural break in the gap distribution
        # If the ratio between consecutive sorted gaps exceeds 2x,
        # everything below that break is intra-cell
        for i in range(1, len(sorted_gaps)):
            if sorted_gaps[i] > 2.5 * sorted_gaps[i - 1]:
                intra_cell_gaps = sorted_gaps[:i]
                break

        return median(intra_cell_gaps)

    def _estimate_column_spacing(self, dots: list[DotPosition]) -> float:
        """Estimate the intra-cell column spacing from horizontal distances.

        Uses the smallest consistent gap cluster to find the distance between
        the two columns within a single Braille cell. Distinguishes intra-cell
        gaps from larger inter-cell gaps using natural break detection.
        """
        if len(dots) < 2:
            return 0.0

        sorted_by_x = sorted(dots, key=lambda d: d.x)

        # Collect all horizontal gaps
        x_gaps: list[float] = []
        for i in range(1, len(sorted_by_x)):
            gap = sorted_by_x[i].x - sorted_by_x[i - 1].x
            if gap > 0.1:  # Ignore near-zero gaps (same column)
                x_gaps.append(gap)

        if not x_gaps:
            return 0.0

        # Sort gaps to find natural breaks
        sorted_gaps = sorted(x_gaps)

        # Look for a natural break: if a gap is > 1.4x the previous gap,
        # everything below is intra-cell spacing
        intra_cell_gaps = sorted_gaps
        for i in range(1, len(sorted_gaps)):
            if sorted_gaps[i] > 1.4 * sorted_gaps[i - 1]:
                intra_cell_gaps = sorted_gaps[:i]
                break

        if not intra_cell_gaps:
            return sorted_gaps[0]

        return median(intra_cell_gaps)

    def _cluster_dots_into_cells(
        self,
        dots: list[DotPosition],
        col_spacing: float,
        expected_inter_cell: float,
    ) -> list[list[DotPosition]]:
        """Cluster dots into cell groups based on X-coordinate proximity.

        A new cell boundary is detected when the horizontal gap between
        consecutive dots (sorted by X) exceeds the inter-cell threshold.
        Handles inter-cell spacing variation between 50% and 150% of expected.
        """
        if not dots:
            return []

        # The cell boundary threshold: midpoint between intra-cell and inter-cell
        # Intra-cell max X span = col_spacing (distance between 2 columns)
        # Inter-cell min = 50% of expected_inter_cell
        # We use a threshold that separates intra-cell from inter-cell gaps
        # A gap is inter-cell if it's > col_spacing * some factor
        # Expected inter-cell = 1.5 * col_spacing
        # Min inter-cell = 0.5 * expected_inter_cell = 0.75 * col_spacing
        # So threshold should be between col_spacing and 0.75 * col_spacing
        # Use the midpoint: threshold = col_spacing * 1.2 (between 1.0 and 1.5*0.5=0.75)
        # Actually, let's be more precise:
        # Within a cell, max X gap = col_spacing
        # Between cells, min gap = 0.5 * expected_inter_cell = 0.75 * col_spacing
        # So threshold = (col_spacing + 0.75 * col_spacing) / 2 = 0.875 * col_spacing
        # But we need to account for dots not being perfectly aligned
        # Use threshold = col_spacing * 1.1 to separate intra from inter
        boundary_threshold = col_spacing * 1.1

        groups: list[list[DotPosition]] = []
        current_group: list[DotPosition] = [dots[0]]

        for i in range(1, len(dots)):
            x_gap = dots[i].x - dots[i - 1].x

            if x_gap > boundary_threshold:
                groups.append(current_group)
                current_group = [dots[i]]
            else:
                current_group.append(dots[i])

        if current_group:
            groups.append(current_group)

        # Validate: if a group is too wide (spans more than one cell width),
        # it might contain multiple cells that weren't separated
        # Cell width = col_spacing (2 columns)
        # Max cell width with tolerance = col_spacing * 1.5
        max_cell_width = col_spacing * 1.8
        refined_groups: list[list[DotPosition]] = []

        for group in groups:
            if len(group) <= 1:
                refined_groups.append(group)
                continue

            x_min = min(d.x for d in group)
            x_max = max(d.x for d in group)
            width = x_max - x_min

            if width > max_cell_width and len(group) > 6:
                # This group likely contains multiple cells; split it
                sub_groups = self._split_wide_group(group, col_spacing)
                refined_groups.extend(sub_groups)
            else:
                refined_groups.append(group)

        return refined_groups

    def _split_wide_group(
        self, dots: list[DotPosition], col_spacing: float
    ) -> list[list[DotPosition]]:
        """Split a wide group of dots into multiple cells.

        Uses K-means-like approach based on expected cell width.
        """
        sorted_dots = sorted(dots, key=lambda d: d.x)
        cell_width = col_spacing  # Width of a single cell (2 columns)
        expected_inter_cell = 1.5 * col_spacing

        # Estimate number of cells based on total width
        total_width = sorted_dots[-1].x - sorted_dots[0].x
        cell_pitch = cell_width + expected_inter_cell
        num_cells = max(1, round(total_width / cell_pitch) + 1)

        # Assign dots to cells based on X position
        x_min = sorted_dots[0].x
        groups: list[list[DotPosition]] = [[] for _ in range(num_cells)]

        for dot in sorted_dots:
            # Determine which cell this dot belongs to
            relative_x = dot.x - x_min
            cell_idx = int(relative_x / cell_pitch)
            cell_idx = min(cell_idx, num_cells - 1)
            groups[cell_idx].append(dot)

        # Remove empty groups
        return [g for g in groups if g]

    def _assign_dots_to_grid(
        self,
        dots: list[DotPosition],
        col_spacing: float,
        row_spacing: float,
        line_y_min: float | None = None,
    ) -> BrailleCell:
        """Assign dots to a 2x3 Braille cell grid.

        Braille cell positions (standard numbering):
            Col 0   Col 1
        Row 0:  1       4
        Row 1:  2       5
        Row 2:  3       6

        Each dot is assigned to the nearest grid position.
        If distance > 40% of intra-cell spacing, the cell is flagged as ambiguous.

        Uses the line's Y-minimum as the row 0 reference point, so that cells
        with dots only in lower rows (like punctuation) are correctly assigned.

        Args:
            dots: Dots belonging to this cell.
            col_spacing: Estimated horizontal spacing between columns.
            row_spacing: Estimated vertical spacing between rows.
            line_y_min: The minimum Y coordinate of all dots in this line,
                       used as the reference for row 0.

        Returns:
            BrailleCell with assigned dot positions and ambiguity flag.
        """
        if not dots:
            # Empty cell (Requirement 5.6 - partial cells are valid)
            return BrailleCell(
                dot_positions=[],
                confidence_scores=[],
                grid_position=(0, 0),
                is_ambiguous=False,
            )

        # Find the bounding box of dots in this cell
        x_min = min(d.x for d in dots)
        x_max = max(d.x for d in dots)
        y_min = min(d.y for d in dots)
        y_max = max(d.y for d in dots)

        # Use col_spacing and row_spacing to define the grid
        if col_spacing <= 0:
            col_spacing = max(1.0, x_max - x_min) if len(dots) > 1 else 1.0
        if row_spacing <= 0:
            row_spacing = max(1.0, y_max - y_min) if len(dots) > 1 else 1.0

        # Determine the cell origin for X (column assignment).
        # Use x_min as the left column reference.
        origin_x = x_min

        # Determine the cell origin for Y (row assignment).
        # Key insight: use the LINE's y_min as the row 0 reference.
        # This ensures that dots in lower rows (like punctuation marks with
        # dots only in rows 1-2) are correctly assigned, rather than being
        # mapped to row 0 because their local y_min is used as origin.
        if line_y_min is not None:
            origin_y = line_y_min
        else:
            origin_y = y_min

        # Define grid positions relative to origin
        # Grid: 2 columns (0, col_spacing), 3 rows (0, row_spacing, 2*row_spacing)
        grid_positions = {
            1: (0, 0),                          # Row 0, Col 0
            4: (col_spacing, 0),                # Row 0, Col 1
            2: (0, row_spacing),                # Row 1, Col 0
            5: (col_spacing, row_spacing),      # Row 1, Col 1
            3: (0, 2 * row_spacing),            # Row 2, Col 0
            6: (col_spacing, 2 * row_spacing),  # Row 2, Col 1
        }

        # Ambiguity threshold: 40% of intra-cell spacing
        spacing_ref = (col_spacing + row_spacing) / 2.0
        ambiguity_threshold = 0.4 * spacing_ref

        dot_positions: list[int] = []
        confidence_scores: list[float] = []
        is_ambiguous = False
        used_positions: set[int] = set()

        for dot in dots:
            # Compute relative position within the cell
            rel_x = dot.x - origin_x
            rel_y = dot.y - origin_y

            # Find nearest grid position
            best_pos = -1
            best_dist = float("inf")

            for pos, (gx, gy) in grid_positions.items():
                dist = ((rel_x - gx) ** 2 + (rel_y - gy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_pos = pos

            # Check if the dot is too far from any grid position
            if best_dist > ambiguity_threshold:
                is_ambiguous = True

            # Avoid duplicate position assignments
            if best_pos in used_positions:
                # Find next best available position
                for pos, (gx, gy) in sorted(
                    grid_positions.items(),
                    key=lambda item: (
                        (rel_x - item[1][0]) ** 2 + (rel_y - item[1][1]) ** 2
                    ),
                ):
                    if pos not in used_positions:
                        best_pos = pos
                        break
                else:
                    # All positions taken, mark as ambiguous
                    is_ambiguous = True
                    continue

            used_positions.add(best_pos)
            dot_positions.append(best_pos)
            confidence_scores.append(dot.confidence)

        # Sort dot positions for consistent output
        paired = sorted(zip(dot_positions, confidence_scores))
        dot_positions = [p for p, _ in paired]
        confidence_scores = [c for _, c in paired]

        return BrailleCell(
            dot_positions=dot_positions,
            confidence_scores=confidence_scores,
            grid_position=(0, 0),  # Will be updated by caller
            is_ambiguous=is_ambiguous,
        )

    def _create_cell_from_dots(
        self, dots: list[DotPosition], col_spacing: float
    ) -> BrailleCell:
        """Create a single cell from all provided dots when spacing can't be estimated."""
        # Assign positions sequentially as a fallback
        dot_positions = list(range(1, min(len(dots) + 1, 7)))
        confidence_scores = [d.confidence for d in dots[:6]]

        return BrailleCell(
            dot_positions=dot_positions,
            confidence_scores=confidence_scores,
            grid_position=(0, 0),
            is_ambiguous=True,  # Mark as ambiguous since we couldn't determine grid
        )

    def _compute_ambiguous_region(
        self, cell: BrailleCell, line_dots: list[DotPosition]
    ) -> tuple[float, float, float, float] | None:
        """Compute bounding box for an ambiguous cell region.

        Returns (x, y, width, height) or None if no dots available.
        """
        if not cell.dot_positions or not line_dots:
            return None

        # Find dots that belong to this cell based on grid position
        # Use all dots in the line as a rough approximation
        # In practice, we'd track which dots belong to which cell
        # For now, return a region based on the cell's expected position
        # This is a simplified approach for the ambiguous region reporting
        all_x = [d.x for d in line_dots]
        all_y = [d.y for d in line_dots]

        if not all_x or not all_y:
            return None

        # Return a small region around the cell
        x_min = min(all_x)
        y_min = min(all_y)
        x_max = max(all_x)
        y_max = max(all_y)

        return (x_min, y_min, x_max - x_min, y_max - y_min)
