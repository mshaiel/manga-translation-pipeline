"""Japanese Manga OCR engine wrapper using manga-ocr (kha-white).

Extracts Japanese text from bounding box crops produced by Magi.
Handles vertical/horizontal text, furigana, stylized manga fonts, and multi-line bubbles.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from src.ocr.box_merger import merge_adjacent_text_boxes
from src.ocr.sfx_classifier import is_sfx_candidate
from src.translation.schemas import PageDetection, TextBox
from src.utils.gpu_utils import clear_gpu_memory, get_device
from src.utils.image_utils import crop_image, load_pil_image

logger = logging.getLogger(__name__)


class MangaOcrEngine:
    """Wrapper for kha-white/manga-ocr-base optical character recognition.

    Processes text bounding box crops extracted from original page images.
    Guarantees strict VRAM unloading after chapter-wide processing.
    """

    def __init__(
        self,
        model_id: str = "kha-white/manga-ocr-base",
        device: str | None = None,
        mock_mode: bool = False,
        confidence_threshold: float = 0.40,
    ) -> None:
        """Initialize Manga OCR engine.

        Args:
            model_id: HuggingFace model repository ID.
            device: Target execution device ('cuda' or 'cpu').
            mock_mode: If True, uses deterministic synthetic OCR for local testing.
            confidence_threshold: Threshold below which OCR is flagged as low confidence.
        """
        self.model_id = model_id
        self.device = get_device(device or "cuda")
        self.mock_mode = mock_mode
        self.confidence_threshold = confidence_threshold
        self.mocr: Any = None

    def load_model(self) -> MangaOcrEngine:
        """Initialize and load the MangaOcr model onto GPU/CPU."""
        if self.mock_mode:
            logger.info("MangaOcrEngine initialized in mock mode (no weights loaded).")
            return self

        if self.mocr is not None:
            logger.debug("MangaOcr model is already loaded.")
            return self

        logger.info("Loading manga-ocr from '%s' onto device '%s'...", self.model_id, self.device)
        try:
            from manga_ocr import MangaOcr

            force_cpu = (self.device == "cpu")
            self.mocr = MangaOcr(pretrained_model_name_or_path=self.model_id, force_cpu=force_cpu)
            logger.info("manga-ocr loaded successfully on device '%s'.", self.device)
        except Exception as err:
            logger.error("Failed to load manga-ocr: %s", err)
            raise RuntimeError(
                f"Could not load manga-ocr model '{self.model_id}'. "
                "Ensure manga-ocr is installed via pip and model weights are accessible."
            ) from err

        return self

    def unload_model(self) -> None:
        """Unload manga-ocr instance, collect garbage, and flush CUDA VRAM cache."""
        if self.mocr is not None:
            logger.info("Unloading manga-ocr engine and releasing VRAM...")
            del self.mocr
            self.mocr = None

        clear_gpu_memory()

    def __enter__(self) -> MangaOcrEngine:
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit guaranteeing model unloading."""
        self.unload_model()

    def ocr_crop(self, crop: Image.Image) -> str:
        """Run OCR on an individual PIL image crop.

        Args:
            crop: PIL Image containing a single dialogue or SFX bounding box.

        Returns:
            Extracted Japanese text string.
        """
        if self.mock_mode or self.mocr is None:
            if not self.mock_mode:
                self.load_model()
                if not self.mock_mode:
                    return str(self.mocr(crop))
            return "これはテストです。"

        return str(self.mocr(crop))

    def process_page(
        self,
        page_image: str | Path | Image.Image,
        detection: PageDetection,
    ) -> PageDetection:
        """Extract OCR text for all text boxes on a single manga page.

        Preserves the exact list order and indexes of Magi's text boxes.

        Args:
            page_image: Full-resolution PIL Image or path to page image.
            detection: PageDetection model populated with Magi detection results.

        Returns:
            New PageDetection with updated TextBox objects (populated ocr_text, is_sfx, confidences).
        """
        if not self.mock_mode and self.mocr is None:
            self.load_model()

        pil_img = load_pil_image(page_image)
        width, height = pil_img.size

        # 1. Merge adjacent vertical text columns belonging to the same speech bubble
        candidate_boxes = merge_adjacent_text_boxes(detection.text_boxes)

        updated_text_boxes: list[TextBox] = []

        for tb in candidate_boxes:
            # Crop text region using normalized coordinates with 10% total expansion (5% margin each side)
            crop = crop_image(pil_img, tb.bbox.to_list(), normalized=True, expand_ratio=0.05)

            # Perform OCR on the PIL crop
            if self.mock_mode or self.mocr is None:
                ocr_text = self._generate_mock_ocr(tb.id, tb.is_essential)
                confidence = 0.95 if tb.is_essential else 0.85
            else:
                ocr_text = self.ocr_crop(crop)
                confidence = 0.95 if len(ocr_text.strip()) > 3 else 0.80

            # Evaluate SFX classification combining Magi signal + linguistic heuristics + tail cues
            sfx_flag = is_sfx_candidate(
                ocr_text=ocr_text,
                is_essential=tb.is_essential,
                ocr_confidence=confidence,
                confidence_threshold=self.confidence_threshold,
                has_tail=getattr(tb, "has_tail", False),
                speaker_name=tb.speaker_name,
                speaker_cluster_id=tb.speaker_cluster_id,
            )

            logger.info(
                "  [OCR Page %d Box %d] text='%s' | SFX=%s (essential=%s, tail=%s, speaker=%s)",
                detection.page_index + 1,
                tb.id,
                ocr_text,
                sfx_flag,
                tb.is_essential,
                getattr(tb, "has_tail", False),
                tb.speaker_name or (f"Cluster_{tb.speaker_cluster_id}" if tb.speaker_cluster_id is not None else None),
            )

            updated_box = tb.model_copy(
                update={
                    "ocr_text": ocr_text.strip(),
                    "ocr_confidence": confidence,
                    "is_sfx": sfx_flag,
                }
            )
            updated_text_boxes.append(updated_box)

        return detection.model_copy(
            update={
                "image_width": width,
                "image_height": height,
                "text_boxes": updated_text_boxes,
            }
        )

    def process_chapter(
        self,
        chapter_pages: Sequence[str | Path | Image.Image],
        detections: Sequence[PageDetection],
        use_tqdm: bool = True,
    ) -> list[PageDetection]:
        """Perform chapter-wide batch OCR across all pages.

        Args:
            chapter_pages: Sequence of page image paths or PIL images.
            detections: Sequence of PageDetection models from Stage 1.
            use_tqdm: Whether to show a progress bar.

        Returns:
            List of updated PageDetection models.
        """
        if len(chapter_pages) != len(detections):
            raise ValueError(
                f"Mismatch: got {len(chapter_pages)} pages but {len(detections)} detection records."
            )

        if not self.mock_mode and self.mocr is None:
            self.load_model()

        iterator = range(len(chapter_pages))
        if use_tqdm:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Running Manga OCR across Chapter")

        results: list[PageDetection] = []
        for i in iterator:
            page_det = self.process_page(chapter_pages[i], detections[i])
            results.append(page_det)

        return results

    @staticmethod
    def _generate_mock_ocr(text_id: int, is_essential: bool) -> str:
        """Deterministic mock text generator for unit testing."""
        if not is_essential:
            sfx_samples = ["ドン！", "ゴゴゴゴ", "ズズズ", "バーン！", "パチパチ"]
            return sfx_samples[text_id % len(sfx_samples)]

        dialogue_samples = [
            "おれは海賊王になる男だ！",
            "その日、少年の運命は大きく動き出した。",
            "待て！そこから先へは行かせない！",
            "フッ…やはりお前か。",
            "ありがとう、助かったよ。",
        ]
        return dialogue_samples[text_id % len(dialogue_samples)]
