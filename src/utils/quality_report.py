"""Quality audit and reporting module for the manga translation pipeline.

Analyzes OCR confidence, speaker diarization attribution, and translation completeness
to compile a structured quality_report.json per chapter.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

from src.translation.schemas import (
    PageDetection,
    PageQuality,
    PipelineMetadata,
    QualityFlag,
    QualityReport,
    TranslationResponse,
)

logger = logging.getLogger(__name__)


def calculate_estimated_cost_usd(
    input_tokens: int,
    output_tokens: int,
    vision_calls: int = 0,
    price_per_m_input: float = 0.15,
    price_per_m_output: float = 0.60,
    price_per_image: float = 0.005,
) -> float:
    """Calculate estimated LLM API cost in USD based on Gemini 2.5 Flash pricing.

    Args:
        input_tokens: Total prompt tokens sent.
        output_tokens: Total completion tokens received.
        vision_calls: Number of multimodal image requests dispatched.
        price_per_m_input: Price per 1,000,000 input tokens.
        price_per_m_output: Price per 1,000,000 output tokens.
        price_per_image: Approximate price per vision image.

    Returns:
        Estimated cost in USD rounded to 4 decimal places.
    """
    cost_input = (input_tokens / 1_000_000.0) * price_per_m_input
    cost_output = (output_tokens / 1_000_000.0) * price_per_m_output
    cost_vision = vision_calls * price_per_image
    return round(cost_input + cost_output + cost_vision, 4)


def generate_quality_report(
    chapter_name: str,
    detections: Sequence[PageDetection],
    translations: Sequence[TranslationResponse | None],
    processing_time_seconds: float,
    llm_provider: str = "gemini-2.5-flash",
    ocr_model: str = "manga-ocr",
    magi_version: str = "v2",
    ocr_confidence_threshold: float = 0.50,
    estimated_tokens_per_page: int = 1200,
    vision_calls_count: int = 0,
) -> QualityReport:
    """Audit chapter results and construct a comprehensive QualityReport.

    Flags audited:
        - `low_ocr_confidence`: Text boxes where OCR confidence is below threshold.
        - `no_speaker_attribution`: Dialogue text boxes without an attributed speaker or cluster ID.
        - `missing_translation`: Text boxes that failed to receive an English translation.

    Args:
        chapter_name: Identifier for the processed chapter.
        detections: List of PageDetection models.
        translations: List of TranslationResponse models per page.
        processing_time_seconds: Total pipeline execution runtime.
        llm_provider: Identifier of translation model used.
        ocr_model: Identifier of OCR engine.
        magi_version: Version of Magi used.
        ocr_confidence_threshold: Threshold below which OCR is flagged.
        estimated_tokens_per_page: Token heuristic for cost calculation.
        vision_calls_count: Count of vision attachments.

    Returns:
        Populated QualityReport Pydantic model.
    """
    total_pages = len(detections)
    total_text_boxes = sum(len(d.text_boxes) for d in detections)
    total_sfx = sum(sum(1 for tb in d.text_boxes if tb.is_sfx) for d in detections)

    per_page_quality: list[PageQuality] = []

    for idx, det in enumerate(detections):
        page_num = idx + 1
        flags: list[QualityFlag] = []
        trans_res = translations[idx] if idx < len(translations) else None
        trans_map = {t.id: t.english for t in trans_res.translations} if trans_res else {}

        for tb in det.text_boxes:
            # 1. Flag: Low OCR confidence
            if tb.ocr_confidence < ocr_confidence_threshold and tb.ocr_text:
                flags.append(
                    QualityFlag(
                        text_box_id=tb.id,
                        issue="low_ocr_confidence",
                        details=f"OCR confidence {tb.ocr_confidence:.2f} is below threshold {ocr_confidence_threshold:.2f}",
                        ocr_text=tb.ocr_text[:30],
                    )
                )

            # 2. Flag: No speaker attribution for dialogue
            if tb.is_essential and not tb.speaker_name and tb.speaker_cluster_id is None:
                flags.append(
                    QualityFlag(
                        text_box_id=tb.id,
                        issue="no_speaker_attribution",
                        details="Dialogue bubble could not be associated with a speaker character",
                    )
                )

            # 3. Flag: Missing translation
            if trans_res and tb.id not in trans_map:
                flags.append(
                    QualityFlag(
                        text_box_id=tb.id,
                        issue="missing_translation",
                        details="No English translation was returned for this text box",
                    )
                )

        # Vision check is recommended if low OCR confidence is detected
        vision_recommended = any(f.issue == "low_ocr_confidence" for f in flags)

        per_page_quality.append(
            PageQuality(
                page=page_num,
                num_panels=len(det.panels),
                num_text_boxes=len(det.text_boxes),
                num_characters_detected=len(det.characters),
                flags=flags,
                vision_check_recommended=vision_recommended,
            )
        )

    # Estimate API cost
    total_input_tokens = total_pages * estimated_tokens_per_page
    total_output_tokens = total_pages * 300
    est_cost = calculate_estimated_cost_usd(
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        vision_calls=vision_calls_count,
    )

    metadata = PipelineMetadata(
        magi_version=magi_version,
        ocr_model=ocr_model,
        llm_provider=llm_provider,
        total_api_cost_estimate_usd=est_cost,
        processing_time_seconds=round(processing_time_seconds, 2),
    )

    return QualityReport(
        chapter=chapter_name,
        total_pages=total_pages,
        total_text_boxes=total_text_boxes,
        total_sfx=total_sfx,
        per_page=per_page_quality,
        pipeline_metadata=metadata,
    )


def save_quality_report(report: QualityReport, output_path: str | Path) -> Path:
    """Serialize QualityReport to quality_report.json on disk.

    Args:
        report: Populated QualityReport instance.
        output_path: Path to write quality_report.json.

    Returns:
        Resolved Path of the saved report.
    """
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    json_str = json.dumps(report.model_dump(), ensure_ascii=False, indent=2)
    dest.write_text(json_str, encoding="utf-8")
    logger.info("Saved quality report to: %s", dest)
    return dest
