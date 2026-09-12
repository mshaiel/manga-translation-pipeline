"""Unit tests for Magi v2 detector wrapper and parser."""

import numpy as np
import pytest

from src.detection.magi_detector import MagiDetector
from src.translation.schemas import PageDetection


class TestMagiDetector:
    """Test suite for Magi v2 model wrapper, parsing, and lifecycle management."""

    def test_magi_detector_init(self):
        detector = MagiDetector(mock_mode=True)
        assert detector.mock_mode is True
        assert detector.model is None

    def test_mock_inference(self):
        detector = MagiDetector(mock_mode=True)
        dummy_page = np.zeros((200, 100, 3), dtype=np.uint8)

        detections = detector.detect_chapter([dummy_page])
        assert len(detections) == 1
        page = detections[0]

        assert isinstance(page, PageDetection)
        assert page.page_index == 0
        assert page.image_width == 100
        assert page.image_height == 200
        assert len(page.panels) == 3
        assert len(page.text_boxes) == 4
        assert len(page.characters) == 2

        # Check speaker attribution linking
        # Text 0 linked to Char 0 ("Protagonist")
        assert page.text_boxes[0].speaker_name == "Protagonist"
        assert page.text_boxes[0].speaker_cluster_id == 1
        assert page.text_boxes[0].speaker_confidence == pytest.approx(0.95)
        assert page.text_boxes[0].is_essential is True
        assert page.text_boxes[0].is_sfx is False

        # Text 3 is SFX (is_essential_text = False)
        assert page.text_boxes[3].is_essential is False
        assert page.text_boxes[3].is_sfx is True

    def test_parse_magi_results_custom(self):
        raw_output = [
            {
                "panels": [[0.1, 0.1, 0.9, 0.5]],
                "texts": [[0.2, 0.2, 0.4, 0.3]],
                "characters": [[0.6, 0.2, 0.8, 0.4]],
                "text_character_associations": [(0, 0)],
                "character_names": ["Zoro"],
                "character_cluster_labels": [10],
                "is_essential_text": [True],
            }
        ]
        parsed = MagiDetector.parse_magi_results(raw_output, dimensions=[(500, 1000)])
        assert len(parsed) == 1
        p = parsed[0]
        assert p.image_width == 500
        assert p.image_height == 1000
        assert len(p.text_boxes) == 1
        assert p.text_boxes[0].speaker_name == "Zoro"
        assert p.text_boxes[0].speaker_cluster_id == 10
        assert p.text_character_associations == [(0, 0)]

    def test_context_manager_lifecycle(self):
        with MagiDetector(mock_mode=True) as detector:
            assert detector.mock_mode is True
        assert detector.model is None
