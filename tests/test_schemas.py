"""Unit tests for Pydantic v2 schemas in src/translation/schemas.py."""

import pytest
from pydantic import ValidationError

from src.translation.schemas import (
    BoundingBox,
    ChapterInfo,
    CharacterBox,
    DialogueEntry,
    PageDetection,
    PageQuality,
    PipelineMetadata,
    QualityFlag,
    QualityReport,
    RollingContext,
    TextBox,
    TranslationRequest,
    TranslationRequestItem,
    TranslationResponse,
)


class TestBoundingBox:
    """Tests for BoundingBox spatial model."""

    def test_valid_box(self):
        box = BoundingBox(x1=0.1, y1=0.2, x2=0.5, y2=0.8)
        assert box.x1 == 0.1
        assert box.y1 == 0.2
        assert box.x2 == 0.5
        assert box.y2 == 0.8
        assert box.width == pytest.approx(0.4)
        assert box.height == pytest.approx(0.6)
        assert box.area == pytest.approx(0.24)

    def test_parse_from_sequence(self):
        box = BoundingBox.model_validate([0.1, 0.2, 0.3, 0.4])
        assert box.x1 == 0.1
        assert box.y1 == 0.2
        assert box.x2 == 0.3
        assert box.y2 == 0.4

    def test_parse_invalid_sequence_length(self):
        with pytest.raises(ValidationError):
            BoundingBox.model_validate([0.1, 0.2, 0.3])

    def test_auto_reordering_inverted_coords(self):
        # x1 > x2 and y1 > y2 should be automatically swapped/ordered
        box = BoundingBox(x1=0.8, y1=0.9, x2=0.2, y2=0.3)
        assert box.x1 == 0.2
        assert box.y1 == 0.3
        assert box.x2 == 0.8
        assert box.y2 == 0.9

    def test_clamping_out_of_bounds(self):
        box = BoundingBox(x1=-0.5, y1=0.1, x2=1.5, y2=0.9)
        assert box.x1 == 0.0
        assert box.x2 == 1.0

    def test_to_pixels(self):
        box = BoundingBox(x1=0.25, y1=0.5, x2=0.75, y2=1.0)
        px1, py1, px2, py2 = box.to_pixels(1000, 2000)
        assert px1 == 250
        assert py1 == 1000
        assert px2 == 750
        assert py2 == 2000


class TestPageDetection:
    """Tests for PageDetection and text/character representations."""

    def test_page_detection_construction(self):
        tb = TextBox(
            id=0,
            bbox=[0.1, 0.1, 0.3, 0.3],
            ocr_text="こんにちは",
            is_essential=True,
            speaker_name="Luffy",
        )
        cb = CharacterBox(
            id=0,
            bbox=[0.4, 0.4, 0.8, 0.8],
            name="Luffy",
            cluster_id=1,
        )
        page = PageDetection(
            page_index=0,
            image_width=1000,
            image_height=1500,
            text_boxes=[tb],
            characters=[cb],
            text_character_associations=[(0, 0)],
            character_names=["Luffy"],
            character_cluster_labels=[1],
        )

        assert page.page_index == 0
        assert len(page.text_boxes) == 1
        assert page.text_boxes[0].ocr_text == "こんにちは"
        assert page.text_boxes[0].speaker_name == "Luffy"
        assert page.text_character_associations == [(0, 0)]


class TestTranslationSchemas:
    """Tests for translation request, response, and structured outputs."""

    def test_translation_response_schema(self):
        response_json = {
            "translations": [
                {"id": 0, "english": "I'm gonna be King of the Pirates!", "translator_note": None},
                {"id": 1, "english": "Boom!", "translator_note": "SFX sound effect"},
            ],
            "scene_summary_update": "Luffy makes his famous declaration.",
            "confidence": 0.95,
        }
        res = TranslationResponse.model_validate(response_json)
        assert len(res.translations) == 2
        assert res.translations[0].id == 0
        assert res.translations[0].english == "I'm gonna be King of the Pirates!"
        assert res.confidence == 0.95
        assert res.scene_summary_update == "Luffy makes his famous declaration."

    def test_translation_request_construction(self):
        ctx = RollingContext(
            chapter_info=ChapterInfo(title="One Piece", user_context="Shonen adventure"),
            recent_dialogue=[
                DialogueEntry(page=0, speaker="Luffy", japanese="行くぞ！", english="Let's go!", is_sfx=False)
            ],
        )
        req = TranslationRequest(
            page_index=1,
            text_boxes=[
                TranslationRequestItem(id=0, speaker="Luffy", japanese="おれは海賊王になる！", is_sfx=False)
            ],
            context=ctx,
        )
        assert req.page_index == 1
        assert req.text_boxes[0].id == 0
        assert req.context.chapter_info.title == "One Piece"
        assert len(req.context.recent_dialogue) == 1


class TestQualityReport:
    """Tests for quality report schema matching prompt requirements."""

    def test_quality_report_serialization(self):
        report = QualityReport(
            chapter="chapter_01",
            total_pages=1,
            total_text_boxes=2,
            total_sfx=1,
            per_page=[
                PageQuality(
                    page=1,
                    num_panels=3,
                    num_text_boxes=2,
                    num_characters_detected=1,
                    flags=[
                        QualityFlag(text_box_id=1, issue="low_ocr_confidence", ocr_text="??")
                    ],
                    vision_check_recommended=False,
                )
            ],
            pipeline_metadata=PipelineMetadata(
                magi_version="v2",
                ocr_model="manga-ocr",
                llm_provider="gemini-2.5-flash",
                total_api_cost_estimate_usd=0.015,
                processing_time_seconds=12.4,
            ),
        )

        data = report.model_dump()
        assert data["chapter"] == "chapter_01"
        assert data["per_page"][0]["flags"][0]["issue"] == "low_ocr_confidence"
        assert data["pipeline_metadata"]["magi_version"] == "v2"
