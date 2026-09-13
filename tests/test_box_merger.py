"""Unit tests for multi-column text box clustering and merging."""

from src.ocr.box_merger import merge_adjacent_text_boxes, should_merge_boxes
from src.translation.schemas import BoundingBox, TextBox


class TestBoxMerger:
    """Test suite for Union-Find adjacent text box merging."""

    def test_single_box_unaffected(self):
        tb = TextBox(id=0, bbox=[0.5, 0.2, 0.6, 0.5])
        merged = merge_adjacent_text_boxes([tb])
        assert len(merged) == 1
        assert merged[0].id == 0

    def test_merge_two_adjacent_vertical_columns(self):
        # Two vertical text columns in the same bubble
        # Column 1 (Right): X: [0.65, 0.70], Y: [0.20, 0.50]
        # Column 2 (Left):  X: [0.58, 0.63], Y: [0.22, 0.52]
        col_right = TextBox(
            id=0,
            bbox=BoundingBox(x1=0.65, y1=0.20, x2=0.70, y2=0.50),
            speaker_name="Luffy",
            has_tail=True,
            panel_id=1,
        )
        col_left = TextBox(
            id=1,
            bbox=BoundingBox(x1=0.58, y1=0.22, x2=0.63, y2=0.52),
            speaker_name="Luffy",
            has_tail=False,
            panel_id=1,
        )

        assert should_merge_boxes(col_right, col_left) is True

        merged = merge_adjacent_text_boxes([col_right, col_left])
        assert len(merged) == 1
        merged_tb = merged[0]
        assert merged_tb.id == 0
        assert merged_tb.has_tail is True
        assert merged_tb.speaker_name == "Luffy"
        assert merged_tb.bbox.x1 == 0.58
        assert merged_tb.bbox.y1 == 0.20
        assert merged_tb.bbox.x2 == 0.70
        assert merged_tb.bbox.y2 == 0.52

    def test_different_panels_not_merged(self):
        tb1 = TextBox(id=0, bbox=[0.5, 0.2, 0.6, 0.5], panel_id=1)
        tb2 = TextBox(id=1, bbox=[0.45, 0.2, 0.49, 0.5], panel_id=2)
        assert should_merge_boxes(tb1, tb2) is False
        merged = merge_adjacent_text_boxes([tb1, tb2])
        assert len(merged) == 2

    def test_conflicting_speakers_not_merged(self):
        tb1 = TextBox(id=0, bbox=[0.5, 0.2, 0.6, 0.5], speaker_name="Zoro")
        tb2 = TextBox(id=1, bbox=[0.45, 0.2, 0.49, 0.5], speaker_name="Sanji")
        assert should_merge_boxes(tb1, tb2) is False
        merged = merge_adjacent_text_boxes([tb1, tb2])
        assert len(merged) == 2

    def test_dialogue_and_sfx_not_merged(self):
        tb_dialogue = TextBox(id=0, bbox=[0.5, 0.2, 0.6, 0.5], is_sfx=False)
        tb_sfx = TextBox(id=1, bbox=[0.45, 0.2, 0.49, 0.5], is_sfx=True)
        assert should_merge_boxes(tb_dialogue, tb_sfx) is False
        merged = merge_adjacent_text_boxes([tb_dialogue, tb_sfx])
        assert len(merged) == 2

    def test_distant_boxes_not_merged(self):
        # Two boxes far apart horizontally
        tb1 = TextBox(id=0, bbox=[0.8, 0.2, 0.9, 0.5])
        tb2 = TextBox(id=1, bbox=[0.2, 0.2, 0.3, 0.5])
        assert should_merge_boxes(tb1, tb2) is False
        merged = merge_adjacent_text_boxes([tb1, tb2])
        assert len(merged) == 2
