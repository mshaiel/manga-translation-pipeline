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

    def test_create_dilated_mask_panel_constrained(self):
        typesetter = MangaTypesetter()
        mask = typesetter.create_dilated_mask(
            image_shape=(500, 500),
            bbox_pixels=(100, 100, 150, 300),
            panel_pixels=(50, 50, 200, 400),
        )
        assert mask.shape == (500, 500)
        assert mask[100, 100] == 255
        # Pixels outside panel bounds must be strictly 0
        assert np.all(mask[:50, :] == 0)
        assert np.all(mask[400:, :] == 0)
        assert np.all(mask[:, :50] == 0)
        assert np.all(mask[:, 200:] == 0)

    def test_typeset_page_with_bubble_groups(self):
        typesetter = MangaTypesetter()
        img = Image.new("RGB", (400, 600), color=(200, 200, 200))

        tb1 = TextBox(id=1, bbox=[0.20, 0.10, 0.25, 0.30], ocr_text="お前は", is_essential=True)
        tb2 = TextBox(id=2, bbox=[0.15, 0.10, 0.19, 0.30], ocr_text="誰だ？", is_essential=True)
        tb_punct = TextBox(id=3, bbox=[0.50, 0.50, 0.55, 0.55], ocr_text="？", is_essential=True)

        detection = PageDetection(
            page_index=0,
            image_width=400,
            image_height=600,
            text_boxes=[tb1, tb2, tb_punct],
        )

        translation = TranslationResponse(
            translations=[
                TranslationItem(id=1, english="Who are you?"),
            ],
            scene_summary_update="Questioning scene",
            confidence=0.95,
        )

        bubble_groups = {
            1: [tb1, tb2],
            3: [tb_punct],
        }

        output_img = typesetter.typeset_page(img, detection, translation, bubble_groups=bubble_groups)
        assert isinstance(output_img, Image.Image)

    def test_segment_bubble_mask_stops_at_ink_contour(self):
        import cv2

        typesetter = MangaTypesetter()
        # Create a light gray canvas simulating manga screentone / background
        test_img = np.ones((600, 600), dtype=np.uint8) * 230

        # Draw a speech bubble oval with black ink border (thickness 4)
        cv2.ellipse(test_img, (300, 300), (90, 140), 0, 0, 360, 0, 4)
        cv2.ellipse(test_img, (300, 300), (86, 136), 0, 0, 360, 255, -1)
        # Add a text column inside the bubble
        cv2.rectangle(test_img, (290, 220), (310, 380), 0, -1)

        mask, (rx, ry, rw, rh) = typesetter.segment_bubble_mask(test_img, [(290, 220, 310, 380)])

        # 1. Inside bubble center must be masked (255)
        assert mask[300, 300] == 255
        # 2. Outside bubble must NOT be masked (0)
        assert mask[100, 300] == 0
        assert mask[500, 300] == 0
        assert mask[300, 100] == 0
        # 3. Outer black ink stroke must NOT be masked (preserved intact)
        assert mask[160, 300] == 0

    def test_resolve_non_overlapping_rects(self):
        # Two overlapping rectangles
        rects = {
            1: (100, 100, 80, 120),
            2: (150, 120, 80, 120),
        }
        resolved = MangaTypesetter.resolve_non_overlapping_rects(rects)
        x1, y1, w1, h1 = resolved[1]
        x2, y2, w2, h2 = resolved[2]

        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        overlap = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        assert overlap == 0
        assert w1 >= 10 and w2 >= 10

    def test_two_pass_connected_bubbles_no_text_overwrite(self):
        typesetter = MangaTypesetter()
        # Create a manga page with two connected/touching speech bubbles
        img = Image.new("RGB", (600, 600), color=(220, 220, 220))

        tb1 = TextBox(id=1, bbox=[0.20, 0.20, 0.40, 0.40], ocr_text="First bubble", is_essential=True)
        tb2 = TextBox(id=2, bbox=[0.35, 0.35, 0.55, 0.55], ocr_text="Second bubble", is_essential=True)

        detection = PageDetection(
            page_index=0,
            image_width=600,
            image_height=600,
            text_boxes=[tb1, tb2],
        )

        translation = TranslationResponse(
            translations=[
                TranslationItem(id=1, english="Bubble One"),
                TranslationItem(id=2, english="Bubble Two"),
            ],
            confidence=1.0,
        )

        out = typesetter.typeset_page(img, detection, translation)
        out_np = np.array(out)
        # Check that both dialogue texts exist and are rendered (not erased to pure white)
        # Inside the dialogue boxes, there should be black text pixels (0, 0, 0)
        box1_pixels = out_np[120:240, 120:240]
        box2_pixels = out_np[210:330, 210:330]
        assert np.any(box1_pixels == [0, 0, 0])
        assert np.any(box2_pixels == [0, 0, 0])

    def test_fit_text_to_box_never_breaks_words(self):
        typesetter = MangaTypesetter(max_font_size=48, min_font_size=8)
        text = "He was brutally killed by the dark swordsman."
        font, lines, text_w, text_h = typesetter.fit_text_to_box(text, box_width=120, box_height=140)
        all_words_in_lines = []
        for line in lines:
            all_words_in_lines.extend(line.split())
        assert all_words_in_lines == text.split()
        assert "killed" in all_words_in_lines
        assert "kille" not in all_words_in_lines

    def test_partition_conjoined_bubble_rects(self):
        import cv2

        mask = np.zeros((400, 600), dtype=np.uint8)
        # Lobe 1 (left)
        cv2.circle(mask, (200, 200), 100, 255, -1)
        # Lobe 2 (right)
        cv2.circle(mask, (380, 200), 90, 255, -1)

        bubble_centers = {
            1: (200, 200),
            2: (380, 200),
        }
        bubble_masks = {
            1: mask,
            2: mask,
        }

        partitioned = MangaTypesetter.partition_conjoined_bubble_rects(bubble_centers, bubble_masks)
        assert 1 in partitioned and 2 in partitioned
        r1 = partitioned[1]
        r2 = partitioned[2]
        # Partition divider is at (200 + 380) // 2 = 290
        assert r1[0] + r1[2] <= 290
        assert r2[0] >= 290

