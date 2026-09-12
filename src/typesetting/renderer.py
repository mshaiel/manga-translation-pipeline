"""Pillow-based typesetting engine for manga dialogue and sound effects.

Renders translated English text into speech bubbles using dynamic font fitting
and positions Viz Media style subtitles near original SFX art.
"""

from __future__ import annotations

import logging
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from src.translation.schemas import PageDetection, TranslationResponse
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
        """
        self.font_path = Path(font_path)
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.dialogue_text_color = dialogue_text_color
        self.inpaint_bg_color = inpaint_bg_color
        self.sfx_font_size_ratio = sfx_font_size_ratio
        self.sfx_text_color = sfx_text_color
        self.sfx_badge_color = sfx_badge_color

        if not self.font_path.exists():
            logger.warning("Configured font not found at '%s'. Falling back to default PIL font.", self.font_path)

    def _load_font(self, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        """Load the font at specified pixel size, falling back to PIL default if missing."""
        if self.font_path.exists():
            try:
                return ImageFont.truetype(str(self.font_path), size=size)
            except Exception as err:
                logger.warning("Failed to load truetype font '%s': %s", self.font_path, err)
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

    def render_dialogue(
        self,
        image: Image.Image,
        text: str,
        bbox_pixels: tuple[int, int, int, int],
    ) -> None:
        """Inpaint original dialogue bubble with white and render wrapped English text.

        Args:
            image: PIL Image being modified in-place.
            text: Translated dialogue string.
            bbox_pixels: (px1, py1, px2, py2) coordinates.
        """
        px1, py1, px2, py2 = bbox_pixels
        bw = px2 - px1
        bh = py2 - py1

        if bw <= 0 or bh <= 0 or not text.strip():
            return

        draw = ImageDraw.Draw(image)

        # 1. Inpaint bubble region with white
        draw.rectangle([px1, py1, px2, py2], fill=self.inpaint_bg_color)

        # 2. Fit and word-wrap translated text
        font, lines, text_w, text_h = self.fit_text_to_box(text, bw, bh)

        # 3. Center vertically inside the bounding box
        spacing = max(2, int(getattr(font, "size", 14) * 0.2))
        curr_y = py1 + (bh - text_h) // 2

        for line in lines:
            line_bbox = draw.textbbox((0, 0), line, font=font)
            lw = line_bbox[2] - line_bbox[0]
            lh = line_bbox[3] - line_bbox[1]

            # Center horizontally
            curr_x = px1 + (bw - lw) // 2
            draw.text((curr_x, curr_y), line, font=font, fill=self.dialogue_text_color)
            curr_y += lh + spacing

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
    ) -> Image.Image:
        """Typeset a complete manga page with translated dialogue and SFX.

        Args:
            page_image: Original page image (PIL or path).
            detection: PageDetection containing text box coordinates and SFX tags.
            translation: TranslationResponse containing 1:1 mapped translations by id.

        Returns:
            New PIL Image with translated text typeset.
        """
        output_image = load_pil_image(page_image).copy()
        width, height = output_image.size

        # Create translation lookup by id
        trans_map = {t.id: t.english for t in translation.translations}

        for tb in detection.text_boxes:
            translated_en = trans_map.get(tb.id, "")
            if not translated_en.strip():
                continue

            # Convert normalized bounding box to pixel coordinates
            px1, py1, px2, py2 = bbox_normalized_to_pixel(
                tb.bbox.to_list(),
                width=width,
                height=height,
                clip=True,
            )

            if tb.is_sfx:
                # Viz Media style SFX offset badge
                self.render_sfx_subtitle(output_image, translated_en, (px1, py1, px2, py2))
            else:
                # White inpaint + dynamic centered text fit
                self.render_dialogue(output_image, translated_en, (px1, py1, px2, py2))

        return output_image
