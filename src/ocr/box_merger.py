"""Multi-column text box clustering and merging.

Detects when Magi splits a single speech bubble containing multi-column vertical Japanese text
into multiple adjacent bounding boxes. Merges them via Union-Find into a single coherent
bounding box before OCR, preventing fragmented translations and English-on-English text collisions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from src.translation.schemas import BoundingBox, TextBox

logger = logging.getLogger(__name__)


class _DisjointSet:
    """Lightweight Union-Find data structure."""

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


def should_merge_boxes(
    a: TextBox,
    b: TextBox,
    vertical_overlap_threshold: float = 0.45,
    max_horizontal_gap_ratio: float = 1.5,
    max_normalized_gap: float = 0.045,
) -> bool:
    """Determine whether two text boxes belong to the same multi-column speech bubble.

    Criteria:
    1. Same panel assignment (or both unassigned).
    2. Same SFX classification (dialogue bubbles only merge with dialogue).
    3. Compatible speaker identity (no conflicting character names or clusters).
    4. Substantial vertical overlap (adjacent vertical columns in the same bubble).
    5. Close horizontal proximity (columns separated by a small margin).
    """
    # 1. Panel consistency
    if a.panel_id != b.panel_id:
        return False

    # 2. SFX consistency (don't merge dialogue with SFX)
    if a.is_sfx != b.is_sfx:
        return False

    # 3. Speaker consistency
    if (
        a.speaker_name is not None
        and b.speaker_name is not None
        and a.speaker_name != b.speaker_name
    ):
        return False
    if (
        a.speaker_cluster_id is not None
        and b.speaker_cluster_id is not None
        and a.speaker_cluster_id != b.speaker_cluster_id
    ):
        return False

    # 4. Vertical overlap
    overlap_y1 = max(a.bbox.y1, b.bbox.y1)
    overlap_y2 = min(a.bbox.y2, b.bbox.y2)
    overlap_h = max(0.0, overlap_y2 - overlap_y1)
    min_h = min(a.bbox.height, b.bbox.height)

    if min_h <= 0.0 or (overlap_h / min_h) < vertical_overlap_threshold:
        return False

    # 5. Horizontal gap between vertical columns
    gap_x = max(0.0, max(a.bbox.x1, b.bbox.x1) - min(a.bbox.x2, b.bbox.x2))
    max_col_w = max(a.bbox.width, b.bbox.width)

    gap_limit = max(max_col_w * max_horizontal_gap_ratio, max_normalized_gap)
    if gap_x > gap_limit:
        return False

    return True


def merge_adjacent_text_boxes(
    text_boxes: Sequence[TextBox],
    vertical_overlap_threshold: float = 0.45,
    max_horizontal_gap_ratio: float = 1.5,
    max_normalized_gap: float = 0.045,
) -> list[TextBox]:
    """Cluster adjacent text boxes that form a single speech bubble and merge them.

    Args:
        text_boxes: Sequence of detected TextBoxes on a page.
        vertical_overlap_threshold: Minimum vertical overlap ratio.
        max_horizontal_gap_ratio: Max horizontal gap as a multiple of column width.
        max_normalized_gap: Absolute fallback threshold for normalized horizontal gap.

    Returns:
        Consolidated list of TextBoxes with merged bounding boxes and attributes.
    """
    n = len(text_boxes)
    if n <= 1:
        return list(text_boxes)

    dsu = _DisjointSet(n)

    for i in range(n):
        for j in range(i + 1, n):
            if should_merge_boxes(
                text_boxes[i],
                text_boxes[j],
                vertical_overlap_threshold=vertical_overlap_threshold,
                max_horizontal_gap_ratio=max_horizontal_gap_ratio,
                max_normalized_gap=max_normalized_gap,
            ):
                dsu.union(i, j)

    # Group boxes by connected component
    clusters: dict[int, list[TextBox]] = {}
    for i in range(n):
        root = dsu.find(i)
        clusters.setdefault(root, []).append(text_boxes[i])

    merged_boxes: list[TextBox] = []

    for group in clusters.values():
        if len(group) == 1:
            merged_boxes.append(group[0])
            continue

        # Sort columns Right-to-Left (highest X2 first) for Japanese vertical reading order
        group.sort(key=lambda tb: tb.bbox.x2, reverse=True)

        # Unified bounding box covering all adjacent columns
        ux1 = min(tb.bbox.x1 for tb in group)
        uy1 = min(tb.bbox.y1 for tb in group)
        ux2 = max(tb.bbox.x2 for tb in group)
        uy2 = max(tb.bbox.y2 for tb in group)

        # Primary box attributes
        primary_id = min(tb.id for tb in group)
        has_tail = any(getattr(tb, "has_tail", False) for tb in group)
        is_essential = any(tb.is_essential for tb in group)
        is_sfx = all(tb.is_sfx for tb in group)

        speaker_name = next((tb.speaker_name for tb in group if tb.speaker_name), None)
        speaker_cluster = next((tb.speaker_cluster_id for tb in group if tb.speaker_cluster_id is not None), None)
        speaker_conf = max(tb.speaker_confidence for tb in group)

        merged_tb = TextBox(
            id=primary_id,
            bbox=BoundingBox(x1=ux1, y1=uy1, x2=ux2, y2=uy2),
            ocr_text="",
            is_essential=is_essential,
            is_sfx=is_sfx,
            has_tail=has_tail,
            speaker_name=speaker_name,
            speaker_cluster_id=speaker_cluster,
            speaker_confidence=speaker_conf,
            panel_id=group[0].panel_id,
        )
        logger.info(
            "Merged %d adjacent text columns into single bubble (id=%d, bbox=[%.3f, %.3f, %.3f, %.3f])",
            len(group),
            primary_id,
            ux1,
            uy1,
            ux2,
            uy2,
        )
        merged_boxes.append(merged_tb)

    # Preserve stable order by reading_order_index or original ID
    merged_boxes.sort(key=lambda tb: (tb.reading_order_index if tb.reading_order_index is not None else tb.id))
    return merged_boxes
