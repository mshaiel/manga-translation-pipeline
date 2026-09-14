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
    """Sort manga panels in standard Right-to-Left, Top-to-Bottom reading order using horizontal tier clustering.

    Algorithm:
    1. Sort panels by top edge (bbox.y1 ascending).
    2. Cluster panels into horizontal tiers (rows) if vertical center falls within Y range
       of an existing tier (or Y overlap > 30% of panel height).
    3. Sort tiers by minimum Y1 (top to bottom).
    4. Within each tier, sort panels from Right to Left (highest X2 first).

    Args:
        panels: Unsorted list of PanelBox objects.

    Returns:
        Sorted list of PanelBox objects with reading_order_index assigned.
    """
    if not panels:
        return []

    if len(panels) == 1:
        return [panels[0].model_copy(update={"reading_order_index": 0})]

    # 1. Sort panels initially by top edge ascending
    sorted_by_y = sorted(panels, key=lambda p: p.bbox.y1)
    tiers: list[list[PanelBox]] = []

    for panel in sorted_by_y:
        matched_tier = False
        p_center_y = (panel.bbox.y1 + panel.bbox.y2) / 2.0
        p_h = max(0.001, panel.bbox.height)

        for tier in tiers:
            tier_y1 = min(p.bbox.y1 for p in tier)
            tier_y2 = max(p.bbox.y2 for p in tier)

            overlap_y1 = max(panel.bbox.y1, tier_y1)
            overlap_y2 = min(panel.bbox.y2, tier_y2)
            overlap_h = max(0.0, overlap_y2 - overlap_y1)

            # Check if vertical center falls within tier or Y overlap > 30% of panel height
            if (tier_y1 <= p_center_y <= tier_y2) or (overlap_h / p_h > 0.30):
                tier.append(panel)
                matched_tier = True
                break

        if not matched_tier:
            tiers.append([panel])

    # 3. Sort tiers by min y1 (top to bottom)
    tiers.sort(key=lambda t: min(p.bbox.y1 for p in t))

    # 4. Within each tier, sort panels from Right to Left (highest x2 first)
    ordered_panels: list[PanelBox] = []
    global_idx = 0
    for tier in tiers:
        tier.sort(key=lambda p: p.bbox.x2, reverse=True)
        for p in tier:
            ordered_panels.append(p.model_copy(update={"reading_order_index": global_idx}))
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


def sort_page_dialogue(page: PageDetection, preserve_magi_order: bool = False) -> list[TextBox]:
    """Establish reading order for all dialogue and SFX on a manga page.

    1. If preserve_magi_order is True, respects Magi's native topological cut order.
    2. Otherwise, clusters panels into horizontal tiers and orders text boxes.
    3. Sets `reading_order_index` sequentially on all TextBoxes.

    Args:
        page: PageDetection containing panels and text boxes.
        preserve_magi_order: If True, preserves Magi's native topological order.

    Returns:
        List of TextBoxes ordered by reading order.
    """
    if not page.text_boxes:
        return []

    # If preserve_magi_order is True or no panels, preserve native Magi order
    if preserve_magi_order or not page.panels:
        if page.panels:
            ordered_panels = order_panels_manga(page.panels)
            panel_map, _ = assign_text_boxes_to_panels(ordered_panels, page.text_boxes)
            tb_to_panel = {}
            for pid, p_tbs in panel_map.items():
                for pt in p_tbs:
                    tb_to_panel[pt.id] = pid
            ordered_direct = [
                tb.model_copy(update={"panel_id": tb_to_panel.get(tb.id, tb.panel_id)})
                for tb in page.text_boxes
            ]
        else:
            ordered_direct = page.text_boxes if preserve_magi_order else order_text_boxes_within_panel(page.text_boxes)

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


class _DisjointSet:
    """Lightweight Union-Find structure for clustering text boxes."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def _should_group_boxes(
    a: TextBox,
    b: TextBox,
    text_char_map: dict[int, int] | None = None,
) -> bool:
    """Check whether two text boxes belong to the same multi-column speech bubble.

    Criteria:
    1. Same panel assignment.
    2. Neither box is a silence bubble (silence bubbles are never merged).
    3. Both are dialogue (is_essential=True, is_sfx=False).
    4. Neither box's OCR text is empty.
    5. Speaker compatibility: MUST NOT have explicitly conflicting speakers.
    6. Tight Proximity:
       - Multi-column vertical text (vertical overlap >= 50% and horizontal gap <= 1.8x width), OR
       - Split horizontal lines (horizontal overlap >= 50% and vertical gap <= 1.2x height).
    """
    # 1. Same panel: reject if panel IDs differ
    if a.panel_id != b.panel_id:
        return False

    # 2. Silence isolation: silence bubbles are never merged with dialogue or other bubbles
    if getattr(a, "is_silence", False) or getattr(b, "is_silence", False):
        return False
    from src.ocr.sfx_classifier import is_silence_bubble
    if is_silence_bubble(a.ocr_text) or is_silence_bubble(b.ocr_text):
        return False

    # 3. Both are dialogue (not SFX)
    if not (a.is_essential and not a.is_sfx and b.is_essential and not b.is_sfx):
        return False

    # 4. Neither box's OCR text is empty
    if not a.ocr_text or not a.ocr_text.strip() or not b.ocr_text or not b.ocr_text.strip():
        return False

    # 5. Speaker compatibility: Reject ONLY if there is an explicit mismatch between two identified speakers
    if text_char_map:
        ca = text_char_map.get(a.id)
        cb = text_char_map.get(b.id)
        if ca is not None and cb is not None and ca != cb:
            return False

    if a.speaker_cluster_id is not None and b.speaker_cluster_id is not None:
        if a.speaker_cluster_id != b.speaker_cluster_id:
            return False

    if a.speaker_name and b.speaker_name:
        if a.speaker_name != b.speaker_name:
            return False

    # 6. Proximity Check
    overlap_y1 = max(a.bbox.y1, b.bbox.y1)
    overlap_y2 = min(a.bbox.y2, b.bbox.y2)
    overlap_h = max(0.0, overlap_y2 - overlap_y1)
    min_h = min(a.bbox.height, b.bbox.height)

    overlap_x1 = max(a.bbox.x1, b.bbox.x1)
    overlap_x2 = min(a.bbox.x2, b.bbox.x2)
    overlap_w = max(0.0, overlap_x2 - overlap_x1)
    min_w = min(a.bbox.width, b.bbox.width)

    gap_x = max(0.0, max(a.bbox.x1, b.bbox.x1) - min(a.bbox.x2, b.bbox.x2))
    gap_y = max(0.0, max(a.bbox.y1, b.bbox.y1) - min(a.bbox.y2, b.bbox.y2))

    # Case A: Adjacent vertical columns (typical manga dialogue layout)
    is_multi_column = (
        min_h > 0.0
        and (overlap_h / min_h) >= 0.50
        and min_w > 0.0
        and (gap_x / min_w) <= 1.80
    )

    # Case B: Vertically stacked clauses in the same speech bubble
    is_stacked_clause = (
        min_w > 0.0
        and (overlap_w / min_w) >= 0.50
        and min_h > 0.0
        and (gap_y / min_h) <= 1.20
    )

    return is_multi_column or is_stacked_clause


def group_same_bubble_texts(
    text_boxes: list[TextBox],
    text_character_associations: list[tuple[int, int]] | None = None,
    enabled: bool = False,
) -> dict[int, list[TextBox]]:
    """Group adjacent text boxes forming a single multi-column speech bubble after OCR.

    When enabled=False (default), treats each TextBox as an independent speech bubble,
    ensuring each bubble is separately translated and typeset without cross-bubble merging.

    Args:
        text_boxes: Sequence of detected TextBoxes on a page with ocr_text populated.
        text_character_associations: Optional list of (text_box_id, character_id) associations.
        enabled: If False, keeps all speech bubbles separate and unmerged.

    Returns:
        Dict mapping group_id to the list of TextBoxes in that group.
    """
    if not text_boxes:
        return {}

    # Default: each speech bubble is translated and lettered independently
    if not enabled:
        return {tb.id: [tb] for tb in text_boxes}

    n = len(text_boxes)
    if n == 1:
        return {text_boxes[0].id: [text_boxes[0]]}

    text_char_map: dict[int, int] = {}
    if text_character_associations:
        for tb_id, char_id in text_character_associations:
            text_char_map[tb_id] = char_id

    dsu = _DisjointSet(n)
    for i in range(n):
        for j in range(i + 1, n):
            if _should_group_boxes(text_boxes[i], text_boxes[j], text_char_map=text_char_map):
                dsu.union(i, j)

    clusters: dict[int, list[TextBox]] = {}
    for i in range(n):
        root = dsu.find(i)
        clusters.setdefault(root, []).append(text_boxes[i])

    bubble_groups: dict[int, list[TextBox]] = {}
    for group in clusters.values():
        # Sort columns Right-to-Left (highest X2 first) for Japanese reading order
        group.sort(key=lambda tb: tb.bbox.x2, reverse=True)
        lowest_id = min(tb.id for tb in group)
        bubble_groups[lowest_id] = group

    return bubble_groups
