"""Main orchestrator for the AI-Powered Manga Translation Pipeline.

Executes the stage-batched translation pipeline:
    1. Chapter-wide Detection & Speaker Diarization (Magi v2) -> Unload VRAM
    2. Chapter-wide Japanese OCR (manga-ocr) -> Unload VRAM
    3. Reading Order Analysis & SFX Classification
    4. Rolling Context Memory Assembly & LLM Translation (Gemini 2.5 Flash / Fallback)
    5. Pillow Typesetting (Dialogue Inpaint & Viz Media SFX Subtitles)
    6. Multi-format Export (PNG, PDF, CBZ)
    7. Quality Report Audit & Serialization (quality_report.json)

Includes checkpoint persistence for Google Colab session tolerance.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.context.memory_manager import RollingContextManager
from src.detection.magi_detector import MagiDetector
from src.export.exporter import ChapterExporter
from src.ocr.manga_ocr_engine import MangaOcrEngine
from src.translation.base_provider import TranslationProvider
from src.translation.gemini_provider import GeminiProvider
from src.translation.mock_provider import MockTranslationProvider
from src.translation.openai_provider import OpenAIProvider
from src.translation.schemas import (
    PageDetection,
    TranslationRequest,
    TranslationRequestItem,
    TranslationResponse,
)
from src.typesetting.reading_order import sort_page_dialogue
from src.typesetting.renderer import MangaTypesetter
from src.utils.gpu_utils import managed_gpu_memory
from src.utils.quality_report import generate_quality_report, save_quality_report

logger = logging.getLogger(__name__)


class MangaTranslationPipeline:
    """Stage-batched manga translation orchestrator."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        config_path: str | Path | None = None,
        mock_models: bool = False,
    ) -> None:
        """Initialize pipeline with configuration parameters.

        Args:
            config: Optional config dictionary.
            config_path: Optional path to YAML configuration file.
            mock_models: If True, uses mock detection, OCR, and translation for fast tests.
        """
        self.config = self._load_config(config, config_path)
        self.mock_models = mock_models or self.config.get("mock_models", False)

        # Base directories
        self.output_dir = Path(self.config.get("pipeline", {}).get("output_dir", "output"))
        self.checkpoints_dir = Path(self.config.get("pipeline", {}).get("checkpoints_dir", "checkpoints"))
        self.enable_checkpoints = self.config.get("pipeline", {}).get("enable_checkpoints", True)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.enable_checkpoints:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # Initialize sub-components
        self._init_components()

    def _load_config(
        self,
        config: dict[str, Any] | None,
        config_path: str | Path | None,
    ) -> dict[str, Any]:
        """Load default YAML config and merge any overrides."""
        base_cfg: dict[str, Any] = {}
        default_yaml = Path("config/default_config.yaml")

        if config_path and Path(config_path).exists():
            with open(config_path, encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f) or {}
        elif default_yaml.exists():
            with open(default_yaml, encoding="utf-8") as f:
                base_cfg = yaml.safe_load(f) or {}

        if config:
            base_cfg.update(config)

        return base_cfg

    def _init_components(self) -> None:
        """Initialize engine modules based on configuration."""
        det_cfg = self.config.get("detection", {})
        self.detector = MagiDetector(
            model_id=det_cfg.get("model_id", "ragavsachdeva/magiv2"),
            trust_remote_code=det_cfg.get("trust_remote_code", True),
            mock_mode=self.mock_models,
        )

        ocr_cfg = self.config.get("ocr", {})
        self.ocr_engine = MangaOcrEngine(
            model_id=ocr_cfg.get("model_id", "kha-white/manga-ocr-base"),
            mock_mode=self.mock_models,
            confidence_threshold=ocr_cfg.get("confidence_threshold", 0.40),
        )

        # Context manager
        ctx_cfg = self.config.get("context", {})
        self.context_manager = RollingContextManager(
            sliding_window_pages=ctx_cfg.get("sliding_window_pages", 6),
            scene_summary_interval=ctx_cfg.get("scene_summary_interval", 5),
            max_context_tokens=ctx_cfg.get("max_context_tokens", 2000),
        )

        # Translation provider
        self.translator = self._resolve_translation_provider()

        # Typesetter
        ts_cfg = self.config.get("typesetting", {})
        self.typesetter = MangaTypesetter(
            font_path=ts_cfg.get("font_path", "assets/fonts/manga_font.ttf"),
            max_font_size=ts_cfg.get("max_font_size", 24),
            min_font_size=ts_cfg.get("min_font_size", 8),
            dialogue_text_color=tuple(ts_cfg.get("dialogue_text_color", [0, 0, 0])),
            inpaint_bg_color=tuple(ts_cfg.get("inpaint_bg_color", [255, 255, 255])),
        )

        # Exporter
        self.exporter = ChapterExporter()

    def _resolve_translation_provider(self) -> TranslationProvider:
        """Resolve primary and fallback translation provider."""
        if self.mock_models:
            return MockTranslationProvider()

        trans_cfg = self.config.get("translation", {})
        primary = trans_cfg.get("primary_provider", "gemini")

        if primary == "gemini":
            gemini_cfg = trans_cfg.get("gemini", {})
            return GeminiProvider(
                model_name=gemini_cfg.get("model_name", "gemini-2.5-flash"),
                temperature=gemini_cfg.get("temperature", 0.2),
                max_retries=gemini_cfg.get("max_retries", 3),
            )
        elif primary == "openai":
            openai_cfg = trans_cfg.get("fallback", {})
            return OpenAIProvider(
                model_name=openai_cfg.get("model_name", "gpt-4o-mini"),
            )

        return MockTranslationProvider()

    def _save_checkpoint(self, filename: str, data: Any) -> None:
        """Persist intermediate stage results for Colab session tolerance."""
        if not self.enable_checkpoints:
            return
        dest = self.checkpoints_dir / filename
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved checkpoint: %s", dest)

    def _load_checkpoint(self, filename: str) -> Any | None:
        """Load stage results from existing checkpoint if available."""
        if not self.enable_checkpoints:
            return None
        target = self.checkpoints_dir / filename
        if target.exists():
            try:
                data = json.loads(target.read_text(encoding="utf-8"))
                logger.info("Found and resumed from checkpoint: %s", target)
                return data
            except Exception as err:
                logger.warning("Failed to read checkpoint %s: %s", target, err)
        return None

    def run(
        self,
        chapter_pages: Sequence[str | Path],
        chapter_name: str = "chapter_01",
        character_bank: dict[str, Any] | None = None,
        user_context: str | Path | None = None,
        resume_from_checkpoints: bool = True,
    ) -> dict[str, Any]:
        """Execute full stage-batched translation pipeline.

        Args:
            chapter_pages: Sequence of manga page image paths.
            chapter_name: Identifier for the chapter.
            character_bank: Optional dict with 'images' and 'names' for Magi diarization.
            user_context: Optional free-text or file path for story context injection.
            resume_from_checkpoints: Whether to skip completed stages if checkpoints exist.

        Returns:
            Dictionary containing:
                - 'translated_pages': List of PIL Images
                - 'quality_report': QualityReport model
                - 'png_dir': Path to saved PNGs
                - 'pdf_path': Path to generated PDF
                - 'cbz_path': Path to generated CBZ
        """
        start_time = time.time()
        logger.info("Starting Manga Translation Pipeline for [%s] (%d pages)...", chapter_name, len(chapter_pages))

        # Setup user context in rolling memory
        if user_context:
            if isinstance(user_context, Path) or (isinstance(user_context, str) and Path(user_context).exists()):
                self.context_manager.load_user_context_from_file(user_context)
            else:
                self.context_manager.set_chapter_info(title=chapter_name, user_context=str(user_context))

        def _is_valid_detections(det_list: Sequence[PageDetection]) -> bool:
            if not det_list:
                return False
            total_boxes = sum(len(p.text_boxes) for p in det_list)
            if total_boxes == 0:
                return False
            # Ensure bounding boxes are not collapsed zero-area points
            for page in det_list:
                for tb in page.text_boxes:
                    if tb.bbox.width > 0.005 and tb.bbox.height > 0.005:
                        return True
            return False

        # ======================================================================
        # STAGE 1: Chapter-Wide Detection & Diarization (Magi v2)
        # ======================================================================
        stage1_cp = self._load_checkpoint("stage1_detection.json") if resume_from_checkpoints else None
        detections: list[PageDetection] = []
        if stage1_cp:
            candidate_detections = [PageDetection.model_validate(p) for p in stage1_cp]
            if not _is_valid_detections(candidate_detections):
                logger.warning("Stage 1 checkpoint is empty or contains collapsed bounding boxes; re-running detection.")
                stage1_cp = None
            else:
                detections = candidate_detections
                logger.info("Loaded Stage 1 detections from checkpoint (%d text boxes).", sum(len(p.text_boxes) for p in detections))

        if not stage1_cp:
            with managed_gpu_memory("Stage 1: Magi v2 Detection & Diarization"):
                detections = self.detector.detect_chapter(
                    chapter_pages=chapter_pages,
                    character_bank=character_bank,
                    use_tqdm=True,
                )
                # Unload model from VRAM immediately after chapter inference
                self.detector.unload_model()

            self._save_checkpoint(
                "stage1_detection.json",
                [d.model_dump() for d in detections],
            )

        # ======================================================================
        # STAGE 2: Chapter-Wide Japanese OCR (manga-ocr)
        # ======================================================================
        stage2_cp = self._load_checkpoint("stage2_ocr.json") if resume_from_checkpoints else None
        if stage2_cp:
            candidate_ocr = [PageDetection.model_validate(p) for p in stage2_cp]
            has_text = any(tb.ocr_text.strip() for p in candidate_ocr for tb in p.text_boxes)
            if not _is_valid_detections(candidate_ocr) or not has_text:
                logger.warning("Stage 2 OCR checkpoint is empty or missing transcribed text; re-running OCR.")
                stage2_cp = None
            else:
                detections = candidate_ocr
                logger.info("Loaded Stage 2 OCR detections from checkpoint.")

        if not stage2_cp:
            with managed_gpu_memory("Stage 2: Manga OCR"):
                if not self.ocr_engine.mock_mode:
                    self.ocr_engine.load_model()
                detections = self.ocr_engine.process_chapter(
                    chapter_pages=chapter_pages,
                    detections=detections,
                    use_tqdm=True,
                )
                # Unload OCR from VRAM immediately
                self.ocr_engine.unload_model()

            self._save_checkpoint(
                "stage2_ocr.json",
                [d.model_dump() for d in detections],
            )

        # ======================================================================
        # STAGE 3 & 4: Reading Order, Rolling Context & Translation LLM
        # ======================================================================
        stage3_cp = self._load_checkpoint("stage3_translation.json") if resume_from_checkpoints else None
        translations: list[TranslationResponse] = []
        ordered_detections: list[PageDetection] = []

        if stage3_cp:
            candidate_trans = [TranslationResponse.model_validate(t) for t in stage3_cp.get("translations", [])]
            candidate_ord = [PageDetection.model_validate(d) for d in stage3_cp.get("ordered_detections", [])]
            has_trans = any(t.translations for t in candidate_trans)
            if not _is_valid_detections(candidate_ord) or not has_trans:
                logger.warning("Stage 3 translation checkpoint is empty or missing translations; re-running translation.")
                stage3_cp = None
            else:
                translations = candidate_trans
                ordered_detections = candidate_ord
                logger.info("Loaded Stage 3 Translations from checkpoint.")
        else:
            logger.info("Executing Stage 3/4: Reading Order and LLM Translation...")
            trans_cfg = self.config.get("translation", {})
            vm_cfg = trans_cfg.get("vision_mode", True)
            if isinstance(vm_cfg, dict):
                enable_vision = vm_cfg.get("enabled_by_default", True)
            else:
                enable_vision = bool(vm_cfg)
            preserve_magi = self.config.get("pipeline", {}).get("preserve_magi_order", False)

            for idx, det in enumerate(detections):
                # 1. Establish Right-to-Left, Top-to-Bottom reading order
                ordered_boxes = sort_page_dialogue(det, preserve_magi_order=preserve_magi)
                page_det = det.model_copy(update={"text_boxes": ordered_boxes})
                ordered_detections.append(page_det)

                # 2. Register characters on this page to cumulative registry
                self.context_manager.register_page_characters(idx, page_det)

                # 3. Assemble Translation Request
                trans_items = [
                    TranslationRequestItem(
                        id=tb.id,
                        speaker=tb.speaker_name or (f"Cluster_{tb.speaker_cluster_id}" if tb.speaker_cluster_id is not None else None),
                        japanese=tb.ocr_text,
                        is_sfx=tb.is_sfx,
                    )
                    for tb in ordered_boxes
                ]

                req = TranslationRequest(
                    page_index=idx,
                    text_boxes=trans_items,
                    context=self.context_manager.get_context(),
                    page_image_path=str(chapter_pages[idx]),
                    vision_mode=enable_vision,
                )

                # 4. Dispatch Translation API call
                trans_response = self.translator.translate(req)
                translations.append(trans_response)

                # 5. Append translated dialogue to rolling context (last 6 pages)
                self.context_manager.append_translated_page(
                    page_index=idx,
                    text_boxes=ordered_boxes,
                    translations=trans_response.translations,
                    scene_summary_update=trans_response.scene_summary_update,
                )

            self._save_checkpoint(
                "stage3_translation.json",
                {
                    "translations": [t.model_dump() for t in translations],
                    "ordered_detections": [d.model_dump() for d in ordered_detections],
                },
            )

        # ======================================================================
        # STAGE 5: Typesetting Engine (Pillow)
        # ======================================================================
        logger.info("Executing Stage 5: Pillow Typesetting for all pages...")
        translated_images: list[Image.Image] = []

        for idx, page_path in enumerate(chapter_pages):
            typeset_img = self.typesetter.typeset_page(
                page_image=page_path,
                detection=ordered_detections[idx],
                translation=translations[idx],
            )
            translated_images.append(typeset_img)

        # ======================================================================
        # STAGE 6: Multi-Format Document Export
        # ======================================================================
        logger.info("Executing Stage 6: Exporting PNGs, PDF, and CBZ...")
        pages_out_dir = self.output_dir / chapter_name / "pages"
        pages_out_dir.mkdir(parents=True, exist_ok=True)

        saved_png_paths: list[Path] = []
        for idx, t_img in enumerate(translated_images, start=1):
            png_dest = pages_out_dir / f"{idx:03d}.png"
            self.exporter.export_page_png(t_img, png_dest)
            saved_png_paths.append(png_dest)

        pdf_dest = self.output_dir / chapter_name / f"{chapter_name}.pdf"
        cbz_dest = self.output_dir / chapter_name / f"{chapter_name}.cbz"

        self.exporter.export_chapter_pdf(translated_images, pdf_dest)
        self.exporter.export_chapter_cbz(translated_images, cbz_dest)

        # ======================================================================
        # STAGE 7: Quality Report Compilation & Serialization
        # ======================================================================
        elapsed_seconds = time.time() - start_time
        logger.info("Stage 7: Compiling Quality Report (Elapsed: %.2fs)...", elapsed_seconds)

        quality_rep = generate_quality_report(
            chapter_name=chapter_name,
            detections=ordered_detections,
            translations=translations,
            processing_time_seconds=elapsed_seconds,
            llm_provider=self.translator.model_name,
            ocr_model="manga-ocr",
            magi_version="v2",
        )

        rep_path = self.output_dir / chapter_name / "quality_report.json"
        save_quality_report(quality_rep, rep_path)

        logger.info("Pipeline complete! Exported outputs to %s", self.output_dir / chapter_name)

        return {
            "translated_images": translated_images,
            "quality_report": quality_rep,
            "png_dir": pages_out_dir,
            "pdf_path": pdf_dest,
            "cbz_path": cbz_dest,
            "quality_report_path": rep_path,
        }
