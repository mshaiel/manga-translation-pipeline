"""Magi v2 wrapper for chapter-wide manga detection and speaker diarization.

References:
    Paper: "Tails Tell Tales: Chapter-wide Manga Transcriptions with Character Names"
    (ACCV 2024, arXiv:2408.00298)
    Model: ragavsachdeva/magiv2
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.translation.schemas import (
    BoundingBox,
    CharacterBox,
    PageDetection,
    PanelBox,
    TextBox,
)
from src.utils.gpu_utils import clear_gpu_memory, get_device
from src.utils.image_utils import load_image_for_magi

logger = logging.getLogger(__name__)


class MagiDetector:
    """Orchestrator for Magi v2 chapter-wide detection and character/speaker diarization.

    Detects panels, text bubbles, and characters across all pages of a chapter in a single pass.
    Clusters character identities across pages and associates speech bubbles with speakers
    via tail cues and spatial proximity.
    """

    def __init__(
        self,
        model_id: str = "ragavsachdeva/magiv2",
        device: str | None = None,
        trust_remote_code: bool = True,
        mock_mode: bool = False,
    ) -> None:
        """Initialize Magi detector configuration.

        Args:
            model_id: HuggingFace repository ID for Magi v2.
            device: Target device ('cuda' or 'cpu'). If None, resolved automatically.
            trust_remote_code: Required for Magi's custom architecture.
            mock_mode: If True, operates in synthetic test mode without loading heavy weights.
        """
        self.model_id = model_id
        self.device = get_device(device or "cuda")
        self.trust_remote_code = trust_remote_code
        self.mock_mode = mock_mode
        self.model: Any = None

    def load_model(self) -> MagiDetector:
        """Load Magi v2 weights into memory and move to target device."""
        if self.mock_mode:
            logger.info("MagiDetector initialized in mock mode (no weights downloaded).")
            return self

        if self.model is not None:
            logger.debug("Magi model is already loaded.")
            return self

        logger.info("Loading Magi v2 model from '%s' onto device '%s'...", self.model_id, self.device)
        try:
            from transformers import AutoModel

            # Compatibility shim for AutoBackbone across transformers versions
            try:
                from transformers.configuration_utils import PretrainedConfig
                from transformers.models.auto.modeling_auto import AutoBackbone
                from transformers.models.resnet.configuration_resnet import ResNetConfig

                if hasattr(AutoBackbone, "_model_mapping") and ResNetConfig in AutoBackbone._model_mapping:
                    AutoBackbone._model_mapping[PretrainedConfig] = AutoBackbone._model_mapping[ResNetConfig]
            except Exception:
                pass

            self.model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=self.trust_remote_code,
                disable_ocr=True,
            )
            self.model = self.model.to(self.device).eval()
            logger.info("Magi v2 loaded successfully.")
        except Exception as err:
            logger.error("Failed to load Magi model: %s", err)
            raise RuntimeError(
                f"Could not load Magi model '{self.model_id}'. "
                "Ensure torch and transformers are installed and you have an active internet connection."
            ) from err

        return self

    def unload_model(self) -> None:
        """Explicitly delete model instance, garbage collect, and release GPU VRAM."""
        if self.model is not None:
            logger.info("Unloading Magi v2 model and clearing GPU cache...")
            del self.model
            self.model = None

        clear_gpu_memory()

    def __enter__(self) -> MagiDetector:
        """Enter context manager by loading model."""
        self.load_model()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager by guaranteeing model unloading."""
        self.unload_model()

    def detect_chapter(
        self,
        chapter_pages: Sequence[str | Path | np.ndarray],
        character_bank: dict[str, Any] | None = None,
        use_tqdm: bool = True,
    ) -> list[PageDetection]:
        """Run chapter-wide prediction across all pages.

        Args:
            chapter_pages: Sequence of image file paths or numpy arrays (H, W, 3).
            character_bank: Optional dict with 'images' (list of crops) and 'names' (list of str).
            use_tqdm: Whether to display a progress bar.

        Returns:
            List of PageDetection objects, one per page.
        """
        if not chapter_pages:
            return []

        # Prepare images: ensure list of numpy arrays (H, W, 3) uint8 RGB
        formatted_pages: list[np.ndarray] = []
        dimensions: list[tuple[int, int]] = []  # (width, height)

        for page in chapter_pages:
            arr = load_image_for_magi(page)
            formatted_pages.append(arr)
            h, w, _ = arr.shape
            dimensions.append((w, h))

        bank = character_bank if character_bank is not None else {"images": [], "names": []}

        if self.mock_mode or self.model is None:
            if not self.mock_mode:
                # Auto-load if not already loaded
                self.load_model()
                if not self.mock_mode:
                    return self._run_inference(formatted_pages, bank, dimensions, use_tqdm=use_tqdm)
            return self._generate_mock_detections(formatted_pages, dimensions)

        return self._run_inference(formatted_pages, bank, dimensions, use_tqdm=use_tqdm)

    def _run_inference(
        self,
        formatted_pages: list[np.ndarray],
        character_bank: dict[str, Any],
        dimensions: list[tuple[int, int]],
        use_tqdm: bool,
    ) -> list[PageDetection]:
        """Execute model forward pass with torch.no_grad()."""
        import torch

        logger.info("Executing Magi v2 prediction for %d pages (do_ocr=False)...", len(formatted_pages))
        with torch.no_grad():
            raw_results = self.model.do_chapter_wide_prediction(
                formatted_pages,
                character_bank,
                use_tqdm=use_tqdm,
                do_ocr=False,
            )

        return self.parse_magi_results(raw_results, dimensions)

    @staticmethod
    def parse_magi_results(
        raw_results: Sequence[dict[str, Any]],
        dimensions: Sequence[tuple[int, int]] | None = None,
    ) -> list[PageDetection]:
        """Parse raw dictionary output from Magi v2 into typed PageDetection objects.

        Args:
            raw_results: List of dicts returned by Magi v2 `do_chapter_wide_prediction`.
            dimensions: Optional list of (width, height) tuples per page.

        Returns:
            List of PageDetection models.
        """
        parsed_pages: list[PageDetection] = []

        for p_idx, page_dict in enumerate(raw_results):
            w, h = dimensions[p_idx] if dimensions and p_idx < len(dimensions) else (None, None)

            # 1. Parse Panels
            raw_panels = page_dict.get("panels", [])
            panels = [
                PanelBox(id=i, bbox=BoundingBox.model_validate(p_box))
                for i, p_box in enumerate(raw_panels)
            ]

            # 2. Parse Characters
            raw_characters = page_dict.get("characters", [])
            char_cluster_labels = page_dict.get("character_cluster_labels", [])
            char_names = page_dict.get("character_names", [])

            characters: list[CharacterBox] = []
            for i, c_box in enumerate(raw_characters):
                cluster_id = char_cluster_labels[i] if i < len(char_cluster_labels) else None
                name = char_names[i] if i < len(char_names) and char_names[i] else None
                characters.append(
                    CharacterBox(
                        id=i,
                        bbox=BoundingBox.model_validate(c_box),
                        cluster_id=cluster_id,
                        name=name,
                    )
                )

            # 3. Parse Associations: mapping text_idx -> char_idx
            raw_associations = page_dict.get("text_character_associations", [])
            # Store as list of tuples (text_idx, char_idx)
            text_to_char_map: dict[int, int] = {}
            for assoc in raw_associations:
                if len(assoc) == 2:
                    text_to_char_map[int(assoc[0])] = int(assoc[1])

            # 4. Parse Text Boxes
            raw_texts = page_dict.get("texts", [])
            is_essential_flags = page_dict.get("is_essential_text", [True] * len(raw_texts))

            text_boxes: list[TextBox] = []
            for i, t_box in enumerate(raw_texts):
                is_ess = is_essential_flags[i] if i < len(is_essential_flags) else True
                char_idx = text_to_char_map.get(i)

                spk_name: str | None = None
                spk_cluster: int | None = None
                spk_conf = 0.5  # default confidence when unassociated

                if char_idx is not None and 0 <= char_idx < len(characters):
                    target_char = characters[char_idx]
                    spk_name = target_char.name
                    spk_cluster = target_char.cluster_id
                    spk_conf = 0.95

                text_boxes.append(
                    TextBox(
                        id=i,
                        bbox=BoundingBox.model_validate(t_box),
                        ocr_text="",
                        is_essential=bool(is_ess),
                        is_sfx=not bool(is_ess),
                        speaker_name=spk_name,
                        speaker_cluster_id=spk_cluster,
                        speaker_confidence=spk_conf,
                    )
                )

            parsed_pages.append(
                PageDetection(
                    page_index=p_idx,
                    image_width=w,
                    image_height=h,
                    panels=panels,
                    text_boxes=text_boxes,
                    characters=characters,
                    text_character_associations=[(int(a[0]), int(a[1])) for a in raw_associations if len(a) == 2],
                    character_names=[str(n) for n in char_names if n],
                    character_cluster_labels=[int(c) for c in char_cluster_labels],
                )
            )

        return parsed_pages

    def _generate_mock_detections(
        self,
        formatted_pages: list[np.ndarray],
        dimensions: list[tuple[int, int]],
    ) -> list[PageDetection]:
        """Generate deterministic synthetic detection data for tests and offline development."""
        mock_raw_results: list[dict[str, Any]] = []

        for _ in formatted_pages:
            page_data = {
                "panels": [
                    [0.55, 0.05, 0.95, 0.45],  # Top Right
                    [0.05, 0.05, 0.45, 0.45],  # Top Left
                    [0.05, 0.52, 0.95, 0.95],  # Bottom Full
                ],
                "texts": [
                    [0.75, 0.10, 0.90, 0.25],  # Text 0 (inside Panel 0)
                    [0.15, 0.15, 0.35, 0.30],  # Text 1 (inside Panel 1)
                    [0.60, 0.60, 0.85, 0.75],  # Text 2 (inside Panel 2)
                    [0.10, 0.80, 0.25, 0.90],  # Text 3 (SFX inside Panel 2)
                ],
                "characters": [
                    [0.60, 0.20, 0.85, 0.44],  # Char 0
                    [0.10, 0.20, 0.35, 0.44],  # Char 1
                ],
                "text_character_associations": [
                    (0, 0),
                    (1, 1),
                ],
                "character_names": ["Protagonist", "Rival"],
                "character_cluster_labels": [1, 2],
                "is_essential_text": [True, True, True, False],
            }
            mock_raw_results.append(page_data)

        return self.parse_magi_results(mock_raw_results, dimensions)
