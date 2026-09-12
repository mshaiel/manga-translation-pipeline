"""Unit tests for MangaTypesetter rendering and font-fitting engine."""

import numpy as np
from PIL import Image

from src.translation.schemas import (
    PageDetection,
    TextBox,
    TranslationItem,
    TranslationResponse,
)
from src.typesetting.renderer import MangaTypesetter


class TestMangaTypesetter:
    """Test suite for dialogue inpainting, font fitting, and SFX subtitles."""

    def test_fit_text_to_box_shrinks_and_wraps(self):
        typesetter = MangaTypesetter(max_font_size=24, min_font_size=8)
        text = "This is a long sentence of manga dialogue that requires word wrapping to fit."

        # Fit into a 100x80 box
        font, lines, text_w, text_h = typesetter.fit_text_to_box(text, box_width=100, box_height=80)
        assert len(lines) > 1
        assert text_w <= 100
        assert text_h <= 80

    def test_render_dialogue_inpaints_white(self):
        typesetter = MangaTypesetter()
        # Create a black image
        img = Image.new("RGB", (200, 200), color=(0, 0, 0))

        # Render dialogue inside box [50, 50, 150, 150]
        typesetter.render_dialogue(img, "Hello World!", (50, 50, 150, 150))

        # Check that the inpaint region has been filled with white (inpaint_bg_color)
        arr = np.array(img)
        # Point inside the bubble (e.g. at [55, 55]) should be white (255, 255, 255)
        assert np.all(arr[55, 55] == [255, 255, 255])
        # Point outside the bubble (e.g. at [10, 10]) should remain black (0, 0, 0)
        assert np.all(arr[10, 10] == [0, 0, 0])

    def test_render_sfx_subtitle_preserves_original_box(self):
        typesetter = MangaTypesetter()
        # Create a green image to simulate artwork
        img = Image.new("RGB", (200, 200), color=(0, 180, 0))

        # SFX box at [50, 50, 100, 100]
        typesetter.render_sfx_subtitle(img, "BOOM!", (50, 50, 100, 100))

        # The center of the original SFX box [75, 75] must NOT be inpainted white!
        arr = np.array(img)
        assert arr[75, 75, 1] > 100  # Still predominantly green artwork

    def test_typeset_page_end_to_end(self):
        typesetter = MangaTypesetter()
        img = Image.new("RGB", (400, 600), color=(200, 200, 200))

        tb_dialogue = TextBox(id=0, bbox=[0.1, 0.1, 0.4, 0.3], is_essential=True, is_sfx=False)
        tb_sfx = TextBox(id=1, bbox=[0.5, 0.5, 0.8, 0.7], is_essential=False, is_sfx=True)

        detection = PageDetection(
            page_index=0,
            image_width=400,
            image_height=600,
            text_boxes=[tb_dialogue, tb_sfx],
        )

        translation = TranslationResponse(
            translations=[
                TranslationItem(id=0, english="I will become King of the Pirates!"),
                TranslationItem(id=1, english="THUD!"),
            ],
            scene_summary_update="Scene update",
            confidence=0.98,
        )

        output_img = typesetter.typeset_page(img, detection, translation)
        assert isinstance(output_img, Image.Image)
        assert output_img.size == (400, 600)
