"""Pillow-based typesetting engine for manga dialogue and sound effects.

Renders translated English text into speech bubbles using dynamic font fitting
and positions Viz Media style subtitles near original SFX art.
"""

from __future__ import annotations

import logging
import re
import textwrap
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.ocr.sfx_classifier import is_punctuation_only, is_silence_bubble
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
        max_font_size: int = 48,
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
        padding_ratio: float = 0.10,
    ) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, list[str], int, int]:
        """Dynamically fit and word-wrap text into an elliptical speech bubble.

        Conforms to professional comic lettering standards:
        - Whole-word wrapping: NEVER slices words across lines (no 'kille-d').
        - Elliptical safe area: Computes allowed width for each line based on its
          vertical distance from the center of the oval (diamond/oval silhouette).
        - Dynamic scaling: Searches font sizes from max_font_size down to min_font_size
          to ensure dialogue comfortably fills the speech bubble without spillover.

        Args:
            text: Text string to render.
            box_width: Available pixel width.
            box_height: Available pixel height.
            padding_ratio: Margin fraction inside the box boundary (default 10%).

        Returns:
            Tuple of (fitted_font, wrapped_lines, total_text_width, total_text_height).
        """
        clean_text = text.strip()
        if not clean_text:
            return self._load_font(self.min_font_size), [""], 10, 10

        words = clean_text.split()
        safe_w = max(20, int(box_width * (1.0 - padding_ratio * 2)))
        safe_h = max(20, int(box_height * (1.0 - padding_ratio * 2)))

        dummy_img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(dummy_img)

        best_font = self._load_font(self.min_font_size)
        best_lines = [clean_text]
        best_w, best_h = safe_w, safe_h

        # Upper bound font size capped by safe height
        target_max_size = min(self.max_font_size, max(self.min_font_size, safe_h // 2))

        for font_size in range(target_max_size, self.min_font_size - 1, -1):
            font = self._load_font(font_size)
            spacing = max(2, int(font_size * 0.22))

            # Measure width of each word individually
            word_widths = []
            for w in words:
                bbox = draw.textbbox((0, 0), w, font=font)
                word_widths.append(bbox[2] - bbox[0])

            # If any single word exceeds safe_w, this font size is too large
            if max(word_widths) > safe_w:
                continue

            test_bbox = draw.textbbox((0, 0), "Ay", font=font)
            single_line_h = test_bbox[3] - test_bbox[1]
            line_step = single_line_h + spacing

            # Average width for elliptical lines (~85% of safe width)
            avg_chord = safe_w * 0.85
            candidate_lines: list[str] = []
            curr_line = ""

            # Greedy whole-word wrapping (zero word breaking)
            for word in words:
                candidate = f"{curr_line} {word}" if curr_line else word
                c_bbox = draw.textbbox((0, 0), candidate, font=font)
                cw = c_bbox[2] - c_bbox[0]
                if cw <= avg_chord or not curr_line:
                    curr_line = candidate
                else:
                    candidate_lines.append(curr_line)
                    curr_line = word
            if curr_line:
                candidate_lines.append(curr_line)

            num_lines = len(candidate_lines)
            total_text_h = num_lines * single_line_h + (num_lines - 1) * spacing
            if total_text_h > safe_h:
                continue

            # Verify that every line fits inside its specific elliptical chord:
            # At distance line_cy from vertical center, chord = safe_w * sqrt(1 - (2*line_cy/safe_h)^2)
            lines_fit = True
            line_widths = []
            for idx, line in enumerate(candidate_lines):
                lb = draw.textbbox((0, 0), line, font=font)
                lw = lb[2] - lb[0]
                line_widths.append(lw)

                line_cy = (idx + 0.5) * line_step - (total_text_h / 2.0)
                norm_y = abs(line_cy) / (safe_h / 2.0)
                chord_factor = np.sqrt(max(0.10, 1.0 - norm_y ** 2)) if norm_y < 1.0 else 0.30
                allowed_w = safe_w * chord_factor

                if lw > allowed_w:
                    lines_fit = False
                    break

            if lines_fit:
                return font, candidate_lines, max(line_widths), total_text_h

            best_font = font
            best_lines = candidate_lines
            best_w = max(line_widths) if line_widths else safe_w
            best_h = total_text_h

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

        # Mandatory Base Text Coverage (8px padding around union of text bounding boxes):
        # Guarantees that 100% of the Japanese text characters are covered with zero bleed-through!
        base_mask = np.zeros_like(image_gray)
        pad_base = 8
        fx1 = max(0, ux1 - pad_base)
        fy1 = max(0, uy1 - pad_base)
        fx2 = min(img_w, ux2 + pad_base)
        fy2 = min(img_h, uy2 + pad_base)
        cv2.rectangle(base_mask, (fx1, fy1), (fx2, fy2), 255, -1)
        base_rect = (fx1, fy1, fx2 - fx1, fy2 - fy1)

        # 2. Local search ROI: Constrain flood fill to a bounded search window around the text box
        # Speech bubbles in manga are centered around text; search window never needs to extend
        # into unrelated corners, gutters, or adjacent panels.
        pad_x = max(int(ubw * 1.5), 100)
        pad_y = max(int(ubh * 1.2), 100)
        roi_x1 = max(0, ux1 - pad_x)
        roi_y1 = max(0, uy1 - pad_y)
        roi_x2 = min(img_w, ux2 + pad_x)
        roi_y2 = min(img_h, uy2 + pad_y)

        # Constrain to panel boundaries if provided
        if panel_pixels is not None:
            p_x1, p_y1, p_x2, p_y2 = panel_pixels
            roi_x1 = max(roi_x1, max(0, p_x1))
            roi_y1 = max(roi_y1, max(0, p_y1))
            roi_x2 = min(roi_x2, min(img_w, p_x2))
            roi_y2 = min(roi_y2, min(img_h, p_y2))

        # Blank out free space outside the search ROI to prevent gutter/margin runaway
        roi_free_space = np.zeros_like(free_space)
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi_free_space[roi_y1:roi_y2, roi_x1:roi_x2] = free_space[roi_y1:roi_y2, roi_x1:roi_x2]

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

            crop_free = roi_free_space[sy1:sy2, sx1:sx2]
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
                roi_free_space.copy(),
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
        text_area = ubw * ubh

        # 5. Leak Detection:
        # A true leak occurs if:
        # a) Zero area (mask_area == 0)
        # b) Mask bleeds to the ROI edge without ink boundary (open bubble or margin runaway)
        # c) Mask area is disproportionately large compared to text box (> 6x text area)
        # d) Width or height blows out (> 4x text width or > 3.5x text height)
        edge_leak = False
        if mask_area > 0 and (roi_x2 > roi_x1 and roi_y2 > roi_y1):
            sub_mask = accumulated_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            if (
                np.any(sub_mask[0, :] == 255)
                or np.any(sub_mask[-1, :] == 255)
                or np.any(sub_mask[:, 0] == 255)
                or np.any(sub_mask[:, -1] == 255)
            ):
                edge_leak = True

        leaked = (
            mask_area == 0
            or edge_leak
            or (mask_area > max(int(text_area * 6.0), 30000))
            or (rw > max(int(ubw * 4.0), 300))
            or (rh > max(int(ubh * 3.5), 300))
        )

        if leaked:
            # Fallback to guaranteed rectangular base mask closely wrapping the text
            return base_mask, base_rect

        # Union bubble interior contour with base text coverage so 100% of text is guaranteed covered
        final_mask = cv2.bitwise_or(accumulated_mask, base_mask)
        rx, ry, rw, rh = cv2.boundingRect(final_mask)

        # Inset the typesetting box by 6% from the bubble contour so letters never collide with ink lines
        inset_x = max(2, int(rw * 0.06))
        inset_y = max(2, int(rh * 0.06))
        typeset_rect = (rx + inset_x, ry + inset_y, max(20, rw - 2 * inset_x), max(20, rh - 2 * inset_y))

        return final_mask, typeset_rect

    @staticmethod
    def partition_conjoined_bubble_rects(
        bubble_centers: dict[int, tuple[int, int]],
        bubble_masks: dict[int, np.ndarray],
    ) -> dict[int, tuple[int, int, int, int]]:
        """Partition bounding boxes of conjoined bubbles sharing a connected interior.

        When two distinct dialogue groups sit inside conjoined lobes of a compound bubble,
        splits the compound bounding box along the lobe divider so each dialogue group
        is lettered directly inside its own lobe.
        """
        result_rects = {}
        group_ids = list(bubble_centers.keys())

        for i, gid in enumerate(group_ids):
            mask = bubble_masks.get(gid)
            if mask is None:
                continue
            rx, ry, rw, rh = cv2.boundingRect(mask)
            if rw <= 0 or rh <= 0:
                continue

            my_cx, my_cy = bubble_centers[gid]

            # Check if any other dialogue group sits in the same connected mask
            for j, other_gid in enumerate(group_ids):
                if i == j:
                    continue
                other_cx, other_cy = bubble_centers[other_gid]
                if (
                    0 <= other_cx < mask.shape[1]
                    and 0 <= other_cy < mask.shape[0]
                    and mask[other_cy, other_cx] == 255
                ):
                    dx = abs(my_cx - other_cx)
                    dy = abs(my_cy - other_cy)
                    if dx >= dy:
                        # Split horizontally between lobe centers
                        split_x = (my_cx + other_cx) // 2
                        if my_cx < other_cx:
                            rw = min(rw, max(20, split_x - rx))
                        else:
                            new_rx = max(rx, split_x)
                            rw = min(rw, max(20, (rx + rw) - new_rx))
                            rx = new_rx
                    else:
                        # Split vertically between lobe centers
                        split_y = (my_cy + other_cy) // 2
                        if my_cy < other_cy:
                            rh = min(rh, max(20, split_y - ry))
                        else:
                            new_ry = max(ry, split_y)
                            rh = min(rh, max(20, (ry + rh) - new_ry))
                            ry = new_ry

            result_rects[gid] = (rx, ry, rw, rh)

        return result_rects

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

                    half_buffer = 4
                    if dx >= dy:
                        # Split horizontally with safety buffer
                        split_x = int((cx1 + cx2) / 2)
                        if cx1 < cx2:
                            new_w1 = max(10, split_x - x1 - half_buffer)
                            new_x2 = max(split_x + half_buffer, x2)
                            new_w2 = max(10, (x2 + w2) - new_x2)
                            resolved[g1] = (x1, y1, new_w1, h1)
                            resolved[g2] = (new_x2, y2, new_w2, h2)
                        else:
                            new_w2 = max(10, split_x - x2 - half_buffer)
                            new_x1 = max(split_x + half_buffer, x1)
                            new_w1 = max(10, (x1 + w1) - new_x1)
                            resolved[g2] = (x2, y2, new_w2, h2)
                            resolved[g1] = (new_x1, y1, new_w1, h1)
                    else:
                        # Split vertically with safety buffer
                        split_y = int((cy1 + cy2) / 2)
                        if cy1 < cy2:
                            new_h1 = max(10, split_y - y1 - half_buffer)
                            new_y2 = max(split_y + half_buffer, y2)
                            new_h2 = max(10, (y2 + h2) - new_y2)
                            resolved[g1] = (x1, y1, w1, new_h1)
                            resolved[g2] = (x2, new_y2, w2, new_h2)
                        else:
                            new_h2 = max(10, split_y - y2 - half_buffer)
                            new_y1 = max(split_y + half_buffer, y1)
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
            # If group is silence / vertical dots: leave completely untouched on the page!
            if (
                any(getattr(tb, "is_silence", False) for tb in group_tbs)
                or any(is_silence_bubble(tb.ocr_text) for tb in group_tbs)
            ):
                logger.info("  Preserving silence bubble %d untouched on page: '%s'", group_id, group_tbs[0].ocr_text)
                continue

            translated_en = trans_map.get(group_id, "")
            # Skip if punctuation only without explicit translation
            if not translated_en.strip():
                if all(is_punctuation_only(tb.ocr_text) for tb in group_tbs):
                    logger.info("  Skipping punctuation-only group %d: '%s'", group_id, group_tbs[0].ocr_text)
                continue

            # Anti-Japanese Leak Check: Never inpaint or typeset if translation contains Japanese characters
            if re.search(r"[\u3040-\u309F\u4E00-\u9FFF]", translated_en):
                logger.warning(
                    "  [Typesetter Safety] Refusing to render Japanese characters as English translation for group %d: '%s'",
                    group_id,
                    translated_en,
                )
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
        bubble_masks: dict[int, np.ndarray] = {}
        bubble_centers: dict[int, tuple[int, int]] = {}

        for group_id, (trans_text, bboxes_px, panel_px) in dialogue_items.items():
            b_mask, b_rect = self.segment_bubble_mask(img_gray, bboxes_px, panel_pixels=panel_px)
            composite_inpaint_mask = cv2.bitwise_or(composite_inpaint_mask, b_mask)
            bubble_rects[group_id] = b_rect
            bubble_masks[group_id] = b_mask
            cx = int(sum(b[0] + b[2] for b in bboxes_px) / (2 * len(bboxes_px)))
            cy = int(sum(b[1] + b[3] for b in bboxes_px) / (2 * len(bboxes_px)))
            bubble_centers[group_id] = (cx, cy)

        # Apply composite inpaint mask to RGB canvas
        bg = self.inpaint_bg_color
        fill_rgb = [bg[0], bg[1], bg[2]] if isinstance(bg, (tuple, list)) else [255, 255, 255]
        img_np[composite_inpaint_mask == 255] = fill_rgb
        output_image = Image.fromarray(img_np)

        # ======================================================================
        # PASS 2: Typesetting Pass (Render text into clean, non-overlapping boxes)
        # ======================================================================
        # Partition conjoined bubble masks if multiple dialogue groups share them
        partitioned_rects = self.partition_conjoined_bubble_rects(bubble_centers, bubble_masks)
        for gid, r in bubble_rects.items():
            if gid not in partitioned_rects:
                partitioned_rects[gid] = r

        # Resolve collisions between adjacent/connected bubbles
        resolved_rects = self.resolve_non_overlapping_rects(partitioned_rects)

        draw = ImageDraw.Draw(output_image)

        for group_id, (trans_text, bboxes_px, _) in dialogue_items.items():
            if not trans_text or not trans_text.strip():
                continue

            # Anti-Japanese Leak Check: Never typeset Japanese characters onto English translated page
            if re.search(r"[\u3040-\u309F\u4E00-\u9FFF]", trans_text):
                logger.warning(
                    "  [Typesetter Safety] Refusing to render Japanese characters as English translation for group %d: '%s'",
                    group_id,
                    trans_text,
                )
                continue

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
