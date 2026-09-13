"""Manga reading order algorithm for panels and speech bubbles.

Manga is read right-to-left and top-to-bottom. This module organizes detected panels
into columns and sorts text bubbles within each panel to ensure natural narrative flow
for translation.
"""

from __future__ import annotations

from src.translation.schemas import BoundingBox, PageDetection, PanelBox, TextBox


def box_contains_point(bbox: BoundingBox, x: float, y: float) -> bool:
    """Check if a normalized coordinate point (x, y) falls inside a bounding box."""
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


def compute_intersection_area(b1: BoundingBox, b2: BoundingBox) -> float:
    """Calculate the intersection area between two normalized bounding boxes."""
    ix1 = max(b1.x1, b2.x1)
    iy1 = max(b1.y1, b2.y1)
    ix2 = min(b1.x2, b2.x2)
    iy2 = min(b1.y2, b2.y2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    return (ix2 - ix1) * (iy2 - iy1)


def order_panels_manga(panels: list[PanelBox]) -> list[PanelBox]:
    """Sort manga panels in standard Right-to-Left, Top-to-Bottom reading order.

    Algorithm:
    1. Cluster panels into vertical columns based on horizontal (X) overlap.
    2. Sort columns from Right to Left (highest X center first).
    3. Sort panels within each column from Top to Bottom (lowest Y first).

    Args:
        panels: Unsorted list of PanelBox objects.

    Returns:
        Sorted list of PanelBox objects with reading_order_index assigned.
    """
    if not panels:
        return []

    if len(panels) == 1:
        single = panels[0].model_copy(update={"reading_order_index": 0})
        return [single]

    # Partition panels into columns
    columns: list[list[PanelBox]] = []

    # Sort candidates initially by rightmost edge descending
    sorted_by_x = sorted(panels, key=lambda p: (p.bbox.x2, p.bbox.x1), reverse=True)

    for panel in sorted_by_x:
        matched_col = False
        p_x_center = (panel.bbox.x1 + panel.bbox.x2) / 2.0

        for col in columns:
            # Check if panel horizontally overlaps with this column
            col_x1 = min(p.bbox.x1 for p in col)
            col_x2 = max(p.bbox.x2 for p in col)
            col_width = col_x2 - col_x1

            # Check overlap or proximity
            overlap_x1 = max(panel.bbox.x1, col_x1)
            overlap_x2 = min(panel.bbox.x2, col_x2)
            overlap_w = max(0.0, overlap_x2 - overlap_x1)

            # If there is meaningful horizontal overlap, assign to this column
            if col_width > 0 and (overlap_w / min(panel.bbox.width, col_width) > 0.3 or abs(p_x_center - (col_x1 + col_x2) / 2.0) < 0.15):
                col.append(panel)
                matched_col = True
                break

        if not matched_col:
            columns.append([panel])

    # Sort columns from Right to Left (highest average X first)
    def col_rightness(col: list[PanelBox]) -> float:
        return sum((p.bbox.x1 + p.bbox.x2) / 2.0 for p in col) / len(col)

    columns.sort(key=col_rightness, reverse=True)

    # Sort panels within each column from Top to Bottom
    ordered_panels: list[PanelBox] = []
    global_idx = 0
    for col in columns:
        col.sort(key=lambda p: p.bbox.y1)
        for p in col:
            ordered = p.model_copy(update={"reading_order_index": global_idx})
            ordered_panels.append(ordered)
            global_idx += 1

    return ordered_panels


def assign_text_boxes_to_panels(
    panels: list[PanelBox],
    text_boxes: list[TextBox],
) -> tuple[dict[int, list[TextBox]], list[TextBox]]:
    """Assign text boxes to containing panels based on spatial overlap.

    Args:
        panels: List of detected panels.
        text_boxes: List of detected text boxes.

    Returns:
        Tuple containing:
        - Dict mapping panel.id -> list of TextBoxes inside that panel.
        - List of unassociated TextBoxes (e.g. edge narration or outside gutters).
    """
    panel_map: dict[int, list[TextBox]] = {p.id: [] for p in panels}
    unassociated: list[TextBox] = []

    for tb in text_boxes:
        tb_center_x = (tb.bbox.x1 + tb.bbox.x2) / 2.0
        tb_center_y = (tb.bbox.y1 + tb.bbox.y2) / 2.0

        best_panel_id: int | None = None
        best_overlap: float = 0.0

        for p in panels:
            # Check center point containment first
            if box_contains_point(p.bbox, tb_center_x, tb_center_y):
                best_panel_id = p.id
                break

            # Fallback to maximum intersection area
            overlap = compute_intersection_area(tb.bbox, p.bbox)
            if overlap > best_overlap:
                best_overlap = overlap
                best_panel_id = p.id

        if best_panel_id is not None:
            updated_tb = tb.model_copy(update={"panel_id": best_panel_id})
            panel_map[best_panel_id].append(updated_tb)
        else:
            unassociated.append(tb)

    return panel_map, unassociated


def order_text_boxes_within_panel(text_boxes: list[TextBox]) -> list[TextBox]:
    """Order text boxes inside a single panel in Right-to-Left, Top-to-Bottom sequence.

    In manga dialogue, bubbles in the same panel are read from right to left,
    top to bottom.
    """
    if len(text_boxes) <= 1:
        return text_boxes

    # Cluster text boxes into rows if their vertical overlap is significant
    # Rows sorted Top to Bottom; within rows sorted Right to Left
    sorted_by_y = sorted(text_boxes, key=lambda t: t.bbox.y1)
    rows: list[list[TextBox]] = []

    for tb in sorted_by_y:
        matched_row = False
        tb_center_y = (tb.bbox.y1 + tb.bbox.y2) / 2.0

        for row in rows:
            row_y1 = min(t.bbox.y1 for t in row)
            row_y2 = max(t.bbox.y2 for t in row)
            row_h = row_y2 - row_y1

            overlap_y1 = max(tb.bbox.y1, row_y1)
            overlap_y2 = min(tb.bbox.y2, row_y2)
            overlap_h = max(0.0, overlap_y2 - overlap_y1)

            if row_h > 0 and (overlap_h / min(tb.bbox.height, row_h) > 0.4 or abs(tb_center_y - (row_y1 + row_y2) / 2.0) < 0.08):
                row.append(tb)
                matched_row = True
                break

        if not matched_row:
            rows.append([tb])

    # Sort rows Top to Bottom
    rows.sort(key=lambda r: min(t.bbox.y1 for t in r))

    # Sort text boxes within each row from Right to Left (highest x2 first)
    ordered: list[TextBox] = []
    for row in rows:
        row.sort(key=lambda t: t.bbox.x2, reverse=True)
        ordered.extend(row)

    return ordered


def sort_page_dialogue(page: PageDetection) -> list[TextBox]:
    """Establish reading order for all dialogue and SFX on a manga page.

    1. Orders panels Right-to-Left, Top-to-Bottom.
    2. Assigns each text box to its containing panel.
    3. Orders text boxes within each panel.
    4. Appends/prepends unassociated boundary narration.
    5. Sets `reading_order_index` sequentially on all TextBoxes.

    Args:
        page: PageDetection containing panels and text boxes.

    Returns:
        List of TextBoxes ordered by reading order.
    """
    if not page.text_boxes:
        return []

    # If no panels were detected, sort text boxes directly by page coordinates
    if not page.panels:
        ordered_direct = order_text_boxes_within_panel(page.text_boxes)
        return [
            tb.model_copy(update={"reading_order_index": i})
            for i, tb in enumerate(ordered_direct)
        ]

    # 1. Order panels
    ordered_panels = order_panels_manga(page.panels)

    # 2. Assign text boxes to panels
    panel_map, unassociated = assign_text_boxes_to_panels(ordered_panels, page.text_boxes)

    # 3. Assemble dialogue in panel reading order
    final_ordered_boxes: list[TextBox] = []

    # Unassociated boxes near the top of the page are prepended (top narration)
    top_narration = [t for t in unassociated if (t.bbox.y1 + t.bbox.y2) / 2.0 < 0.2]
    other_unassociated = [t for t in unassociated if t not in top_narration]

    for tb in order_text_boxes_within_panel(top_narration):
        final_ordered_boxes.append(tb)

    for p in ordered_panels:
        p_boxes = panel_map.get(p.id, [])
        if p_boxes:
            sorted_panel_boxes = order_text_boxes_within_panel(p_boxes)
            final_ordered_boxes.extend(sorted_panel_boxes)

    # Remaining unassociated boxes (e.g. bottom margin narration)
    for tb in order_text_boxes_within_panel(other_unassociated):
        final_ordered_boxes.append(tb)

    # Set sequential reading_order_index
    result: list[TextBox] = [
        tb.model_copy(update={"reading_order_index": idx})
        for idx, tb in enumerate(final_ordered_boxes)
    ]

    return result
