"""Pillow-based typesetting engine for manga dialogue and sound effects.

Renders translated English text into speech bubbles using dynamic font fitting
and positions Viz Media style subtitles near original SFX art.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.ocr.sfx_classifier import is_punctuation_only
from src.translation.schemas import PageDetection, TextBox, TranslationResponse
from src.utils.image_utils import bbox_normalized_to_pixel, load_pil_image

logger = logging.getLogger(__name__)


class MangaTypesetter:
    """Renders dialogue and sound effects onto manga page images.

    Dialogue text boxes are inpainted with solid white and filled with center-aligned,
    dynamically fitted typography. Sound effects (SFX) are rendered as semi-transparent
    offset badges leaving the original manga artwork intact.
    """

    def __init__(
        self,
        font_path: str | Path = "assets/fonts/manga_font.ttf",
        max_font_size: int = 24,
        min_font_size: int = 8,
        dialogue_text_color: str | tuple[int, int, int] = (0, 0, 0),
        inpaint_bg_color: str | tuple[int, int, int] = (255, 255, 255),
        sfx_font_size_ratio: float = 0.70,
        sfx_text_color: str | tuple[int, int, int] = (50, 50, 50),
        sfx_badge_color: tuple[int, int, int, int] = (255, 255, 255, 210),
        mask_dilation_h_ratio: float = 0.80,
        mask_dilation_v_ratio: float = 0.15,
        min_dilation_h_px: int = 20,
        min_dilation_v_px: int = 8,
    ) -> None:
        """Initialize typesetting configuration.

        Args:
            font_path: Path to Truetype/Opentype font file.
            max_font_size: Upper bound for dialogue font scaling (pixels).
            min_font_size: Lower bound below which text will not be shrunk.
            dialogue_text_color: RGB color for translated dialogue.
            inpaint_bg_color: RGB fill color for speech bubbles (default pure white).
            sfx_font_size_ratio: Scale factor for SFX font size relative to dialogue.
            sfx_text_color: RGB color for SFX subtitles.
            sfx_badge_color: RGBA color for semi-transparent SFX backdrop.
            mask_dilation_h_ratio: Horizontal dilation as fraction of text box width.
            mask_dilation_v_ratio: Vertical dilation as fraction of text box height.
            min_dilation_h_px: Minimum horizontal dilation in pixels.
            min_dilation_v_px: Minimum vertical dilation in pixels.
        """
        self.font_path = Path(font_path)
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.dialogue_text_color = dialogue_text_color
        self.inpaint_bg_color = inpaint_bg_color
        self.sfx_font_size_ratio = sfx_font_size_ratio
        self.sfx_text_color = sfx_text_color
        self.sfx_badge_color = sfx_badge_color
        self.mask_dilation_h_ratio = mask_dilation_h_ratio
        self.mask_dilation_v_ratio = mask_dilation_v_ratio
        self.min_dilation_h_px = min_dilation_h_px
        self.min_dilation_v_px = min_dilation_v_px

        if not self.font_path.exists():
            logger.warning("Configured font not found at '%s'. Falling back to default PIL font.", self.font_path)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load the font at specified pixel size, falling back to system fonts or PIL default if missing."""
        font_p = self.font_path
        if not font_p.is_absolute() and not font_p.exists():
            repo_root = Path(__file__).resolve().parent.parent.parent
            candidate = repo_root / font_p
            if candidate.exists():
                font_p = candidate

        if font_p.exists():
            try:
                return ImageFont.truetype(str(font_p), size=size)
            except Exception as err:
                logger.warning("Failed to load truetype font '%s': %s", font_p, err)

        # System Truetype fallback (covers Ubuntu/Colab and Windows)
        system_candidates = [
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
        for sys_f in system_candidates:
            if Path(sys_f).exists():
                try:
                    return ImageFont.truetype(sys_f, size=size)
                except Exception:
                    pass

        try:
            return ImageFont.load_default(size=size)  # Pillow >= 10.1
        except TypeError:
            return ImageFont.load_default()

    def fit_text_to_box(
        self,
        text: str,
        box_width: int,
        box_height: int,
        padding_ratio: float = 0.08,
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int, int]:
        """Dynamically fit and word-wrap text to fit inside target bounding dimensions.

        Shrinks font size from `max_font_size` down to `min_font_size` in a loop
        until all wrapped lines fit completely within the padded box bounds.

        Args:
            text: Text string to render.
            box_width: Available pixel width.
            box_height: Available pixel height.
            padding_ratio: Margin fraction inside the box boundary (default 8%).

        Returns:
            Tuple of (fitted_font, wrapped_lines, total_text_width, total_text_height).
        """
        usable_w = max(10, int(box_width * (1.0 - padding_ratio * 2)))
        usable_h = max(10, int(box_height * (1.0 - padding_ratio * 2)))

        best_font = self._load_font(self.min_font_size)
        best_lines = [text]
        best_w, best_h = usable_w, usable_h

        # Dummy draw context for measuring textbbox
        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        # Shrink loop from max_font_size down to min_font_size
        for font_size in range(self.max_font_size, self.min_font_size - 1, -1):
            font = self._load_font(font_size)

            # Estimate approximate wrap character width based on average glyph width
            avg_char_w = max(4, font_size * 0.55)
            wrap_width = max(3, int(usable_w / avg_char_w))

            lines = textwrap.wrap(text, width=wrap_width, break_long_words=True)
            if not lines:
                lines = [text]

            # Measure total multi-line bounding box
            line_heights: list[int] = []
            line_widths: list[int] = []

            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
                line_widths.append(lw)
                line_heights.append(lh)

            spacing = max(2, int(font_size * 0.2))
            total_h = sum(line_heights) + spacing * (len(lines) - 1)
            total_w = max(line_widths) if line_widths else 0

            # Check if text fits inside available width and height
            if total_w <= usable_w and total_h <= usable_h:
                return font, lines, total_w, total_h

            best_font = font
            best_lines = lines
            best_w = total_w
            best_h = total_h

        return best_font, best_lines, best_w, best_h

    def create_dilated_mask(
        self,
        image_shape: tuple[int, int],
        bbox_pixels: tuple[int, int, int, int],
        panel_pixels: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Create an elliptical dilated mask around a text bounding box, constrained to panel bounds.

        Args:
            image_shape: (height, width) of the image.
            bbox_pixels: (px1, py1, px2, py2) of the text box.
            panel_pixels: Optional (px1, py1, px2, py2) of containing panel.

        Returns:
            Binary mask (uint8) of shape image_shape with 255 in dilated region.
        """
        img_h, img_w = image_shape
        px1, py1, px2, py2 = bbox_pixels
        bw = max(1, px2 - px1)
        bh = max(1, py2 - py1)

        seed_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.rectangle(seed_mask, (px1, py1), (px2, py2), 255, -1)

        # Elliptical dilation kernel proportional to box dimensions
        kw = max(int(bw * self.mask_dilation_h_ratio), self.min_dilation_h_px)
        kh = max(int(bh * self.mask_dilation_v_ratio), self.min_dilation_v_px)
        if kw % 2 == 0:
            kw += 1
        if kh % 2 == 0:
            kh += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
        dilated_mask = cv2.dilate(seed_mask, kernel, iterations=1)

        # Constrain dilated mask to panel boundaries if known
        if panel_pixels is not None:
            p_x1, p_y1, p_x2, p_y2 = panel_pixels
            p_x1, p_y1 = max(0, p_x1), max(0, p_y1)
            p_x2, p_y2 = min(img_w, p_x2), min(img_h, p_y2)
            if p_x1 > 0:
                dilated_mask[:, :p_x1] = 0
            if p_y1 > 0:
                dilated_mask[:p_y1, :] = 0
            if p_x2 < img_w:
                dilated_mask[:, p_x2:] = 0
            if p_y2 < img_h:
                dilated_mask[p_y2:, :] = 0

        return dilated_mask

    def render_dialogue_group(
        self,
        image: Image.Image,
        text: str,
        bboxes_pixels: list[tuple[int, int, int, int]],
        panel_pixels: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Erase text area(s) using morphological mask dilation and typeset translated dialogue.

        Args:
            image: PIL Image being modified in-place.
            text: Translated dialogue string.
            bboxes_pixels: List of (px1, py1, px2, py2) bounding boxes in the bubble.
            panel_pixels: Optional containing panel pixel bounds.
        """
        if not text.strip() or not bboxes_pixels:
            return

        img_w, img_h = image.size
        combined_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        for bbox in bboxes_pixels:
            m = self.create_dilated_mask((img_h, img_w), bbox, panel_pixels=panel_pixels)
            combined_mask = cv2.bitwise_or(combined_mask, m)

        # Apply inpaint mask to image
        img_np = np.array(image.convert("RGB"))
        bg = self.inpaint_bg_color
        fill_rgb = [bg[0], bg[1], bg[2]] if isinstance(bg, (tuple, list)) else [255, 255, 255]
        img_np[combined_mask == 255] = fill_rgb
        image.paste(Image.fromarray(img_np))

        # Compute bounding rect of the dilated mask for typesetting
        rx, ry, rw, rh = cv2.boundingRect(combined_mask)
        if rw <= 0 or rh <= 0:
            rx = min(b[0] for b in bboxes_pixels)
            ry = min(b[1] for b in bboxes_pixels)
            rw = max(b[2] for b in bboxes_pixels) - rx
            rh = max(b[3] for b in bboxes_pixels) - ry

        # Fit text into dilated bounding rect
        font, lines, text_w, text_h = self.fit_text_to_box(text, rw, rh)

        draw = ImageDraw.Draw(image)
        spacing = max(2, int(getattr(font, "size", 14) * 0.2))
        curr_y = max(ry, ry + (rh - text_h) // 2)

        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            lw = line_bbox[2] - line_bbox[0]
            lh = line_bbox[3] - line_bbox[1]
            curr_x = max(rx, rx + (rw - lw) // 2)
            draw.text((curr_x, curr_y), line, font=font, fill=self.dialogue_text_color)
            curr_y += lh + spacing

    def render_dialogue(
        self,
        image: Image.Image,
        text: str,
        bbox_pixels: tuple[int, int, int, int],
        panel_pixels: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Render single dialogue bubble using morphological mask dilation.

        Args:
            image: PIL Image being modified in-place.
            text: Translated dialogue string.
            bbox_pixels: (px1, py1, px2, py2) coordinates.
            panel_pixels: Optional containing panel pixel bounds.
        """
        self.render_dialogue_group(image, text, [bbox_pixels], panel_pixels=panel_pixels)

    def render_sfx_subtitle(
        self,
        image: Image.Image,
        sfx_text: str,
        bbox_pixels: tuple[int, int, int, int],
    ) -> None:
        """Render sound effect translation as an offset subtitle badge (Viz Media style).

        Leaves original artwork untouched and overlays a semi-transparent subtitle badge.

        Args:
            image: PIL Image being modified in-place.
            sfx_text: Translated SFX text (e.g., 'BOOM!', 'THUD').
            bbox_pixels: Original SFX bounding box in pixels.
        """
        if not sfx_text.strip():
            return

        px1, py1, px2, py2 = bbox_pixels
        sfx_font_size = max(self.min_font_size, int(self.max_font_size * self.sfx_font_size_ratio))
        font = self._load_font(sfx_font_size)

        # Measure text dimensions
        dummy_img = Image.new("RGBA", (1, 1))
        draw_dummy = ImageDraw.Draw(dummy_img)
        tbbox = draw_dummy.textbbox((0, 0), sfx_text, font=font)
        tw = tbbox[2] - tbbox[0]
        th = tbbox[3] - tbbox[1]

        # Position badge just below the original art box, offset by 4px
        pad_x, pad_y = 6, 4
        badge_w = tw + pad_x * 2
        badge_h = th + pad_y * 2

        badge_x1 = max(0, min(image.width - badge_w, px1))
        badge_y1 = min(image.height - badge_h, py2 + 2)

        # Render semi-transparent badge using an RGBA overlay
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw_overlay = ImageDraw.Draw(overlay)

        # Rounded background rectangle
        draw_overlay.rounded_rectangle(
            [badge_x1, badge_y1, badge_x1 + badge_w, badge_y1 + badge_h],
            radius=4,
            fill=self.sfx_badge_color,
            outline=(100, 100, 100, 180),
            width=1,
        )

        # Draw SFX text centered inside badge
        text_x = badge_x1 + pad_x
        text_y = badge_y1 + pad_y
        draw_overlay.text((text_x, text_y), sfx_text, font=font, fill=self.sfx_text_color)

        # Composite overlay back into base RGB image
        if image.mode != "RGBA":
            image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
        else:
            image.paste(Image.alpha_composite(image, overlay))

    def typeset_page(
        self,
        page_image: str | Path | Image.Image,
        detection: PageDetection,
        translation: TranslationResponse,
        bubble_groups: dict[int, list[TextBox]] | None = None,
    ) -> Image.Image:
        """Typeset a complete manga page with translated dialogue and SFX.

        Args:
            page_image: Original page image (PIL or path).
            detection: PageDetection containing text box coordinates and SFX tags.
            translation: TranslationResponse containing 1:1 mapped translations by id.
            bubble_groups: Optional mapping from group_id to list of TextBoxes in that bubble.

        Returns:
            New PIL Image with translated text typeset.
        """
        output_image = load_pil_image(page_image).copy()
        width, height = output_image.size

        # Create translation lookup by id
        trans_map = {t.id: t.english for t in translation.translations}

        # Build panel pixel coordinate lookup
        panel_pixels_map = {}
        for p in detection.panels:
            panel_pixels_map[p.id] = bbox_normalized_to_pixel(
                p.bbox.to_list(), width=width, height=height, clip=True
            )

        # If bubble_groups is not provided, treat each TextBox as its own group
        if bubble_groups is None:
            bubble_groups = {tb.id: [tb] for tb in detection.text_boxes}

        for group_id, group_tbs in bubble_groups.items():
            # Check if all boxes in this group are punctuation-only
            if all(is_punctuation_only(tb.ocr_text) for tb in group_tbs):
                logger.info("  Skipping punctuation-only group %d: '%s'", group_id, group_tbs[0].ocr_text)
                continue

            translated_en = trans_map.get(group_id, "")
            if not translated_en.strip():
                logger.warning("No translation found for group ID %d, skipping.", group_id)
                continue

            # Check if it's SFX (SFX boxes are rendered with subtitle badge)
            if any(tb.is_sfx for tb in group_tbs):
                primary_tb = group_tbs[0]
                px1, py1, px2, py2 = bbox_normalized_to_pixel(
                    primary_tb.bbox.to_list(), width=width, height=height, clip=True
                )
                logger.info("  Typeset SFX Box %d: '%s' -> '%s'", primary_tb.id, primary_tb.ocr_text, translated_en)
                self.render_sfx_subtitle(output_image, translated_en, (px1, py1, px2, py2))
            else:
                # Dialogue bubble (single or multi-column)
                bboxes_px = [
                    bbox_normalized_to_pixel(tb.bbox.to_list(), width=width, height=height, clip=True)
                    for tb in group_tbs
                ]
                panel_px = panel_pixels_map.get(group_tbs[0].panel_id) if group_tbs[0].panel_id is not None else None
                combined_ocr = "".join(tb.ocr_text for tb in group_tbs)
                logger.info(
                    "  Typeset Dialogue Group %d (%d boxes): '%s' -> '%s'",
                    group_id,
                    len(group_tbs),
                    combined_ocr,
                    translated_en,
                )
                self.render_dialogue_group(output_image, translated_en, bboxes_px, panel_pixels=panel_px)

        return output_image
