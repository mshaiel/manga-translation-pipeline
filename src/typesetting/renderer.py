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

    def segment_bubble_mask(
        self,
        image_gray: np.ndarray,
        bboxes_pixels: list[tuple[int, int, int, int]],
        panel_pixels: tuple[int, int, int, int] | None = None,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """Segment speech bubble boundary using ink-barrier flood fill.

        Discovers the natural contour of speech bubbles by expanding from within the
        text boxes until halted by dark ink lines (bubble borders, panel edges, artwork).
        Falls back to a tight bounding rectangle if the region is borderless or open.

        Args:
            image_gray: Grayscale 2D numpy array of the page image (uint8).
            bboxes_pixels: List of (px1, py1, px2, py2) bounding boxes in the bubble.
            panel_pixels: Optional containing panel pixel bounds.

        Returns:
            Tuple of (binary_mask, (rx, ry, rw, rh)) where binary_mask has shape equal
            to image_gray, containing 255 within the detected bubble interior, and
            (rx, ry, rw, rh) is the bounding rectangle for typesetting.
        """
        img_h, img_w = image_gray.shape
        ux1 = min(b[0] for b in bboxes_pixels)
        uy1 = min(b[1] for b in bboxes_pixels)
        ux2 = max(b[2] for b in bboxes_pixels)
        uy2 = max(b[3] for b in bboxes_pixels)
        ubw = max(1, ux2 - ux1)
        ubh = max(1, uy2 - uy1)

        # 1. Build ink barrier: pixels darker than 140 represent black ink strokes
        ink_barrier = (image_gray < 140).astype(np.uint8) * 255
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        ink_closed = cv2.morphologyEx(ink_barrier, cv2.MORPH_CLOSE, kernel_close)
        free_space = (ink_closed == 0).astype(np.uint8) * 255

        # 2. Constrain free space to panel boundaries if provided
        if panel_pixels is not None:
            p_x1, p_y1, p_x2, p_y2 = panel_pixels
            p_x1, p_y1 = max(0, p_x1), max(0, p_y1)
            p_x2, p_y2 = min(img_w, p_x2), min(img_h, p_y2)
            if p_y1 > 0:
                free_space[:p_y1, :] = 0
            if p_y2 < img_h:
                free_space[p_y2:, :] = 0
            if p_x1 > 0:
                free_space[:, :p_x1] = 0
            if p_x2 < img_w:
                free_space[:, p_x2:] = 0

        accumulated_mask = np.zeros_like(image_gray)

        # 3. Flood-fill from brightest candidate seed points inside each text column
        for bx1, by1, bx2, by2 in bboxes_pixels:
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(img_w, bx2), min(img_h, by2)
            if bx2 <= bx1 or by2 <= by1:
                continue

            # Look for free space seeds inside or immediately bordering the text box
            search_pad = 12
            sx1 = max(0, bx1 - search_pad)
            sy1 = max(0, by1 - search_pad)
            sx2 = min(img_w, bx2 + search_pad)
            sy2 = min(img_h, by2 + search_pad)

            crop_free = free_space[sy1:sy2, sx1:sx2]
            valid_pts = np.argwhere(crop_free == 255)
            if len(valid_pts) == 0:
                continue

            # Prioritize points closest to the center of the text box
            bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2
            dists = (valid_pts[:, 0] + sy1 - bcy) ** 2 + (valid_pts[:, 1] + sx1 - bcx) ** 2
            best_idx = np.argmin(dists)
            seed_y = sy1 + int(valid_pts[best_idx][0])
            seed_x = sx1 + int(valid_pts[best_idx][1])

            ff_mask = np.zeros((img_h + 2, img_w + 2), dtype=np.uint8)
            ff_mask[0, :] = 1
            ff_mask[-1, :] = 1
            ff_mask[:, 0] = 1
            ff_mask[:, -1] = 1

            cv2.floodFill(
                free_space.copy(),
                ff_mask,
                (seed_x, seed_y),
                128,
                flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY,
            )

            raw_mask = (ff_mask[1:-1, 1:-1] == 255).astype(np.uint8) * 255
            contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(accumulated_mask, contours, -1, 255, -1)

        # 4. Erode slightly (1px) so the bubble's original black ink outline is 100% preserved
        if cv2.countNonZero(accumulated_mask) > 0:
            accumulated_mask = cv2.erode(
                accumulated_mask,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )

        rx, ry, rw, rh = cv2.boundingRect(accumulated_mask)
        mask_area = cv2.countNonZero(accumulated_mask)
        union_area = ubw * ubh

        # 5. Sanity Check: If flood fill leaked outside an open/unbordered bubble
        leaked = (
            mask_area == 0
            or rw > max(int(ubw * 3.5), 160)
            or rh > max(int(ubh * 3.0), 160)
            or mask_area > max(union_area * 6, 200000)
        )

        if leaked:
            # Fallback to tight text-fit mask with gentle 8px padding (never a giant rectangle)
            pad_x = 8
            pad_y = 6
            fallback_mask = np.zeros_like(image_gray)
            fx1 = max(0, ux1 - pad_x)
            fy1 = max(0, uy1 - pad_y)
            fx2 = min(img_w, ux2 + pad_x)
            fy2 = min(img_h, uy2 + pad_y)
            cv2.rectangle(fallback_mask, (fx1, fy1), (fx2, fy2), 255, -1)
            return fallback_mask, (fx1, fy1, fx2 - fx1, fy2 - fy1)

        return accumulated_mask, (rx, ry, rw, rh)

    @staticmethod
    def resolve_non_overlapping_rects(
        bubble_rects: dict[int, tuple[int, int, int, int]]
    ) -> dict[int, tuple[int, int, int, int]]:
        """Ensure that adjacent or connected bubbles' typesetting boxes do not collide.

        When two speech bubbles touch or overlap, splits their bounding rectangles
        along the dominant axis of separation so translated text remains distinct.
        """
        resolved = dict(bubble_rects)
        group_ids = list(resolved.keys())

        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                g1, g2 = group_ids[i], group_ids[j]
                x1, y1, w1, h1 = resolved[g1]
                x2, y2, w2, h2 = resolved[g2]

                # Check for intersection
                ix1 = max(x1, x2)
                iy1 = max(y1, y2)
                ix2 = min(x1 + w1, x2 + w2)
                iy2 = min(y1 + h1, y2 + h2)

                if ix2 > ix1 and iy2 > iy1:
                    cx1, cy1 = x1 + w1 / 2, y1 + h1 / 2
                    cx2, cy2 = x2 + w2 / 2, y2 + h2 / 2

                    dx = abs(cx1 - cx2)
                    dy = abs(cy1 - cy2)

                    if dx >= dy:
                        # Split horizontally
                        split_x = int((cx1 + cx2) / 2)
                        if cx1 < cx2:
                            new_w1 = max(10, split_x - x1)
                            new_x2 = max(split_x, x2)
                            new_w2 = max(10, (x2 + w2) - new_x2)
                            resolved[g1] = (x1, y1, new_w1, h1)
                            resolved[g2] = (new_x2, y2, new_w2, h2)
                        else:
                            new_w2 = max(10, split_x - x2)
                            new_x1 = max(split_x, x1)
                            new_w1 = max(10, (x1 + w1) - new_x1)
                            resolved[g2] = (x2, y2, new_w2, h2)
                            resolved[g1] = (new_x1, y1, new_w1, h1)
                    else:
                        # Split vertically
                        split_y = int((cy1 + cy2) / 2)
                        if cy1 < cy2:
                            new_h1 = max(10, split_y - y1)
                            new_y2 = max(split_y, y2)
                            new_h2 = max(10, (y2 + h2) - new_y2)
                            resolved[g1] = (x1, y1, w1, new_h1)
                            resolved[g2] = (x2, new_y2, w2, new_h2)
                        else:
                            new_h2 = max(10, split_y - y2)
                            new_y1 = max(split_y, y1)
                            new_h1 = max(10, (y1 + h1) - new_y1)
                            resolved[g2] = (x2, y2, w2, new_h2)
                            resolved[g1] = (x1, new_y1, w1, new_h1)

        return resolved

    def render_dialogue_group(
        self,
        image: Image.Image,
        text: str,
        bboxes_pixels: list[tuple[int, int, int, int]],
        panel_pixels: tuple[int, int, int, int] | None = None,
    ) -> None:
        """Erase text area(s) using ink-barrier bubble segmentation and typeset translated dialogue.

        Args:
            image: PIL Image being modified in-place.
            text: Translated dialogue string.
            bboxes_pixels: List of (px1, py1, px2, py2) bounding boxes in the bubble.
            panel_pixels: Optional containing panel pixel bounds.
        """
        if not text.strip() or not bboxes_pixels:
            return

        img_np = np.array(image.convert("RGB"))
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        mask, (rx, ry, rw, rh) = self.segment_bubble_mask(img_gray, bboxes_pixels, panel_pixels=panel_pixels)

        # Apply inpaint mask to image
        bg = self.inpaint_bg_color
        fill_rgb = [bg[0], bg[1], bg[2]] if isinstance(bg, (tuple, list)) else [255, 255, 255]
        img_np[mask == 255] = fill_rgb
        image.paste(Image.fromarray(img_np))

        # Fit text into segmented bounding rect
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
        """Render single dialogue bubble using ink-barrier bubble segmentation.

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

        Employs a Two-Pass Rendering Pipeline:
          - Pass 1 (Page-Wide Inpaint): Accurately contours all speech bubbles
            using ink-barrier segmentation and cleanly erases them simultaneously.
          - Pass 2 (Typesetting): Renders translated typography into resolved,
            non-overlapping bounding boxes and positions SFX subtitle badges.

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

        img_np = np.array(output_image.convert("RGB"))
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        # Prepare groups to render
        dialogue_items: dict[int, tuple[str, list[tuple[int, int, int, int]], tuple[int, int, int, int] | None]] = {}
        sfx_items: list[tuple[str, tuple[int, int, int, int]]] = []

        for group_id, group_tbs in bubble_groups.items():
            translated_en = trans_map.get(group_id, "")
            # Skip if punctuation only without explicit translation
            if not translated_en.strip():
                if all(is_punctuation_only(tb.ocr_text) for tb in group_tbs):
                    logger.info("  Skipping punctuation-only group %d: '%s'", group_id, group_tbs[0].ocr_text)
                continue

            if any(tb.is_sfx for tb in group_tbs):
                primary_tb = group_tbs[0]
                px1, py1, px2, py2 = bbox_normalized_to_pixel(
                    primary_tb.bbox.to_list(), width=width, height=height, clip=True
                )
                sfx_items.append((translated_en, (px1, py1, px2, py2)))
            else:
                bboxes_px = [
                    bbox_normalized_to_pixel(tb.bbox.to_list(), width=width, height=height, clip=True)
                    for tb in group_tbs
                ]
                panel_px = panel_pixels_map.get(group_tbs[0].panel_id) if group_tbs[0].panel_id is not None else None
                dialogue_items[group_id] = (translated_en, bboxes_px, panel_px)

        # ======================================================================
        # PASS 1: Page-Wide Inpaint (Clean all dialogue bubbles simultaneously)
        # ======================================================================
        composite_inpaint_mask = np.zeros((height, width), dtype=np.uint8)
        bubble_rects: dict[int, tuple[int, int, int, int]] = {}

        for group_id, (trans_text, bboxes_px, panel_px) in dialogue_items.items():
            b_mask, b_rect = self.segment_bubble_mask(img_gray, bboxes_px, panel_pixels=panel_px)
            composite_inpaint_mask = cv2.bitwise_or(composite_inpaint_mask, b_mask)
            bubble_rects[group_id] = b_rect

        # Apply composite inpaint mask to RGB canvas
        bg = self.inpaint_bg_color
        fill_rgb = [bg[0], bg[1], bg[2]] if isinstance(bg, (tuple, list)) else [255, 255, 255]
        img_np[composite_inpaint_mask == 255] = fill_rgb
        output_image = Image.fromarray(img_np)

        # ======================================================================
        # PASS 2: Typesetting Pass (Render text into clean, non-overlapping boxes)
        # ======================================================================
        # Resolve collisions between adjacent/connected bubbles
        resolved_rects = self.resolve_non_overlapping_rects(bubble_rects)

        draw = ImageDraw.Draw(output_image)

        for group_id, (trans_text, bboxes_px, _) in dialogue_items.items():
            rx, ry, rw, rh = resolved_rects.get(group_id, bubble_rects[group_id])
            if rw <= 0 or rh <= 0:
                rx = min(b[0] for b in bboxes_px)
                ry = min(b[1] for b in bboxes_px)
                rw = max(b[2] for b in bboxes_px) - rx
                rh = max(b[3] for b in bboxes_px) - ry

            font, lines, text_w, text_h = self.fit_text_to_box(trans_text, rw, rh)
            spacing = max(2, int(getattr(font, "size", 14) * 0.2))
            curr_y = max(ry, ry + (rh - text_h) // 2)

            for line in lines:
                line_bbox = draw.textbbox((0, 0), line, font=font)
                lw = line_bbox[2] - line_bbox[0]
                lh = line_bbox[3] - line_bbox[1]
                curr_x = max(rx, rx + (rw - lw) // 2)
                draw.text((curr_x, curr_y), line, font=font, fill=self.dialogue_text_color)
                curr_y += lh + spacing

        # Typeset SFX subtitle badges
        for sfx_text, bbox_px in sfx_items:
            self.render_sfx_subtitle(output_image, sfx_text, bbox_px)

        return output_image
