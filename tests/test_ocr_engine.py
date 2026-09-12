"""Unit tests for MangaOcrEngine wrapper."""

from PIL import Image

from src.ocr.manga_ocr_engine import MangaOcrEngine
from src.translation.schemas import PageDetection, TextBox


class TestMangaOcrEngine:
    """Test suite for manga OCR processing and text attribution."""

    def test_mock_ocr_crop(self):
        engine = MangaOcrEngine(mock_mode=True)
        img = Image.new("RGB", (50, 50), color=(255, 255, 255))
        res = engine.ocr_crop(img)
        assert isinstance(res, str)
        assert len(res) > 0

    def test_process_page(self):
        engine = MangaOcrEngine(mock_mode=True)
        page_img = Image.new("RGB", (200, 400), color=(255, 255, 255))

        tb1 = TextBox(id=0, bbox=[0.1, 0.1, 0.4, 0.3], is_essential=True)
        tb2 = TextBox(id=1, bbox=[0.5, 0.5, 0.8, 0.7], is_essential=False)

        detection = PageDetection(
            page_index=0,
            text_boxes=[tb1, tb2],
        )

        updated = engine.process_page(page_img, detection)
        assert updated.image_width == 200
        assert updated.image_height == 400
        assert len(updated.text_boxes) == 2

        # Dialogue text box
        assert updated.text_boxes[0].id == 0
        assert len(updated.text_boxes[0].ocr_text) > 0
        assert updated.text_boxes[0].is_sfx is False

        # SFX text box
        assert updated.text_boxes[1].id == 1
        assert len(updated.text_boxes[1].ocr_text) > 0
        assert updated.text_boxes[1].is_sfx is True

    def test_process_chapter(self):
        engine = MangaOcrEngine(mock_mode=True)
        img1 = Image.new("RGB", (100, 100), color=(255, 255, 255))
        img2 = Image.new("RGB", (100, 100), color=(255, 255, 255))

        det1 = PageDetection(page_index=0, text_boxes=[TextBox(id=0, bbox=[0.1, 0.1, 0.5, 0.5])])
        det2 = PageDetection(page_index=1, text_boxes=[TextBox(id=0, bbox=[0.2, 0.2, 0.6, 0.6])])

        results = engine.process_chapter([img1, img2], [det1, det2], use_tqdm=False)
        assert len(results) == 2
        assert results[0].page_index == 0
        assert results[1].page_index == 1
        assert len(results[0].text_boxes[0].ocr_text) > 0

    def test_context_manager_lifecycle(self):
        with MangaOcrEngine(mock_mode=True) as engine:
            assert engine.mock_mode is True
        assert engine.mocr is None
