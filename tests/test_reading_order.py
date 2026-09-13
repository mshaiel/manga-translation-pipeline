"""Unit tests for manga reading order algorithm."""

from src.translation.schemas import PageDetection, PanelBox, TextBox
from src.typesetting.reading_order import (
    assign_text_boxes_to_panels,
    order_panels_manga,
    order_text_boxes_within_panel,
    sort_page_dialogue,
)


class TestReadingOrder:
    """Test suite for manga panel and dialogue reading order sorting."""

    def test_single_panel(self):
        panels = [PanelBox(id=1, bbox=[0.1, 0.1, 0.9, 0.9])]
        ordered = order_panels_manga(panels)
        assert len(ordered) == 1
        assert ordered[0].id == 1
        assert ordered[0].reading_order_index == 0

    def test_horizontal_panels_right_to_left(self):
        # Two panels side-by-side: Panel 0 on Left, Panel 1 on Right
        # In manga, Panel 1 (Right) must be read first!
        left_panel = PanelBox(id=0, bbox=[0.05, 0.1, 0.45, 0.9])
        right_panel = PanelBox(id=1, bbox=[0.55, 0.1, 0.95, 0.9])

        ordered = order_panels_manga([left_panel, right_panel])
        assert [p.id for p in ordered] == [1, 0]
        assert ordered[0].reading_order_index == 0
        assert ordered[1].reading_order_index == 1

    def test_vertical_panels_top_to_bottom(self):
        # Two stacked panels in the same column
        top_panel = PanelBox(id=0, bbox=[0.1, 0.05, 0.9, 0.45])
        bottom_panel = PanelBox(id=1, bbox=[0.1, 0.55, 0.9, 0.95])

        ordered = order_panels_manga([bottom_panel, top_panel])
        assert [p.id for p in ordered] == [0, 1]

    def test_shonen_page_layout_grid(self):
        # 4 panels:
        # Top-Right (0), Top-Left (1)
        # Bottom-Right (2), Bottom-Left (3)
        # Expected reading order: Top-Right (0) -> Bottom-Right (2) [Column 1]
        #                      -> Top-Left (1) -> Bottom-Left (3) [Column 2]
        p_tr = PanelBox(id=0, bbox=[0.55, 0.05, 0.95, 0.45])
        p_tl = PanelBox(id=1, bbox=[0.05, 0.05, 0.45, 0.45])
        p_br = PanelBox(id=2, bbox=[0.55, 0.55, 0.95, 0.95])
        p_bl = PanelBox(id=3, bbox=[0.05, 0.55, 0.45, 0.95])

        ordered = order_panels_manga([p_bl, p_tr, p_tl, p_br])
        ordered_ids = [p.id for p in ordered]

        # In column-based manga reading:
        # Right column panels come before Left column panels
        assert ordered_ids.index(0) < ordered_ids.index(1)
        assert ordered_ids.index(2) < ordered_ids.index(3)
        assert ordered_ids.index(0) < ordered_ids.index(2)

    def test_text_boxes_within_panel_right_to_left(self):
        # Two bubbles inside one panel: Bubble A (right), Bubble B (left)
        box_right = TextBox(id=0, bbox=[0.6, 0.2, 0.8, 0.4], ocr_text="First!")
        box_left = TextBox(id=1, bbox=[0.2, 0.2, 0.4, 0.4], ocr_text="Second!")

        ordered = order_text_boxes_within_panel([box_left, box_right])
        assert [b.id for b in ordered] == [0, 1]

    def test_assign_text_boxes_to_panels(self):
        panel_a = PanelBox(id=10, bbox=[0.5, 0.0, 1.0, 0.5])
        panel_b = PanelBox(id=20, bbox=[0.0, 0.0, 0.5, 0.5])

        box_in_a = TextBox(id=1, bbox=[0.6, 0.1, 0.8, 0.3])
        box_in_b = TextBox(id=2, bbox=[0.1, 0.1, 0.3, 0.3])
        box_unassociated = TextBox(id=3, bbox=[0.1, 0.8, 0.9, 0.95])

        panel_map, unassociated = assign_text_boxes_to_panels(
            [panel_a, panel_b],
            [box_in_a, box_in_b, box_unassociated],
        )

        assert len(panel_map[10]) == 1
        assert panel_map[10][0].id == 1
        assert len(panel_map[20]) == 1
        assert panel_map[20][0].id == 2
        assert len(unassociated) == 1
        assert unassociated[0].id == 3

    def test_sort_page_dialogue_end_to_end(self):
        p_right = PanelBox(id=1, bbox=[0.5, 0.0, 1.0, 1.0])
        p_left = PanelBox(id=2, bbox=[0.0, 0.0, 0.5, 1.0])

        t_right = TextBox(id=101, bbox=[0.6, 0.2, 0.8, 0.4])
        t_left = TextBox(id=102, bbox=[0.1, 0.2, 0.3, 0.4])

        page = PageDetection(
            page_index=0,
            panels=[p_left, p_right],
            text_boxes=[t_left, t_right],
        )

        sorted_boxes = sort_page_dialogue(page)
        assert len(sorted_boxes) == 2
        assert sorted_boxes[0].id == 101
        assert sorted_boxes[0].reading_order_index == 0
        assert sorted_boxes[1].id == 102
        assert sorted_boxes[1].reading_order_index == 1
