"""Unit tests for quality audit and reporting module."""

from src.translation.schemas import (
    PageDetection,
    QualityReport,
    TextBox,
    TranslationItem,
    TranslationResponse,
)
from src.utils.quality_report import (
    calculate_estimated_cost_usd,
    generate_quality_report,
    save_quality_report,
)


class TestQualityReportGeneration:
    """Test suite for pipeline quality auditing and cost estimation."""

    def test_cost_calculation(self):
        # 1M input ($0.15), 1M output ($0.60), 2 vision calls ($0.01) -> $0.76
        cost = calculate_estimated_cost_usd(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            vision_calls=2,
            price_per_m_input=0.15,
            price_per_m_output=0.60,
            price_per_image=0.005,
        )
        assert cost == 0.76

    def test_quality_report_flags_and_audit(self):
        # Page with 1 low confidence box and 1 unassigned dialogue
        tb_low_conf = TextBox(
            id=0,
            bbox=[0.1, 0.1, 0.4, 0.4],
            ocr_text="???",
            ocr_confidence=0.3,
            is_essential=True,
            speaker_name="Luffy",
        )
        tb_no_speaker = TextBox(
            id=1,
            bbox=[0.5, 0.1, 0.8, 0.4],
            ocr_text="誰だ？",
            ocr_confidence=0.95,
            is_essential=True,
            speaker_name=None,
            speaker_cluster_id=None,
        )

        det = PageDetection(page_index=0, text_boxes=[tb_low_conf, tb_no_speaker])
        trans = TranslationResponse(
            translations=[TranslationItem(id=0, english="What?")],
            confidence=0.9,
        )

        report = generate_quality_report(
            chapter_name="test_chap",
            detections=[det],
            translations=[trans],
            processing_time_seconds=5.2,
            ocr_confidence_threshold=0.5,
        )

        assert isinstance(report, QualityReport)
        assert report.chapter == "test_chap"
        assert report.total_pages == 1
        assert report.total_text_boxes == 2

        page1_flags = report.per_page[0].flags
        issues = [f.issue for f in page1_flags]
        assert "low_ocr_confidence" in issues
        assert "no_speaker_attribution" in issues
        assert "missing_translation" in issues  # id 1 had no translation in trans
        assert report.per_page[0].vision_check_recommended is True

    def test_save_quality_report(self, tmp_path):
        det = PageDetection(page_index=0)
        report = generate_quality_report(
            chapter_name="save_test",
            detections=[det],
            translations=[None],
            processing_time_seconds=1.0,
        )

        dest_file = tmp_path / "quality_report.json"
        saved = save_quality_report(report, dest_file)
        assert saved.exists()
        assert "save_test" in dest_file.read_text(encoding="utf-8")
