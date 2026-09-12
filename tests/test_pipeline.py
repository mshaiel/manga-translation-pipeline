"""Integration tests for the end-to-end stage-batched MangaTranslationPipeline."""

from PIL import Image

from src.pipeline import MangaTranslationPipeline


class TestPipelineIntegration:
    """Test suite for full pipeline orchestration and checkpointing."""

    def test_end_to_end_pipeline_mock_run(self, tmp_path):
        # Create 2 synthetic test manga pages
        p1 = tmp_path / "page_01.png"
        p2 = tmp_path / "page_02.png"
        Image.new("RGB", (200, 300), color=(255, 255, 255)).save(p1)
        Image.new("RGB", (200, 300), color=(240, 240, 240)).save(p2)

        pipeline = MangaTranslationPipeline(
            config={
                "pipeline": {
                    "output_dir": str(tmp_path / "output"),
                    "checkpoints_dir": str(tmp_path / "checkpoints"),
                    "enable_checkpoints": True,
                },
                "mock_models": True,
            }
        )

        results = pipeline.run(
            chapter_pages=[p1, p2],
            chapter_name="test_chapter",
            user_context="Pirate adventure story",
            resume_from_checkpoints=False,
        )

        # 1. Assert memory outputs
        assert len(results["translated_images"]) == 2
        assert results["quality_report"].total_pages == 2

        # 2. Assert disk exports
        pdf_path = results["pdf_path"]
        cbz_path = results["cbz_path"]
        rep_path = results["quality_report_path"]

        assert pdf_path.exists()
        assert cbz_path.exists()
        assert rep_path.exists()
        assert (results["png_dir"] / "001.png").exists()
        assert (results["png_dir"] / "002.png").exists()

        # 3. Assert checkpoints were created
        cp_dir = tmp_path / "checkpoints"
        assert (cp_dir / "stage1_detection.json").exists()
        assert (cp_dir / "stage2_ocr.json").exists()
        assert (cp_dir / "stage3_translation.json").exists()

    def test_pipeline_resume_from_checkpoints(self, tmp_path):
        # Run pipeline once to generate checkpoints
        p1 = tmp_path / "page_01.png"
        Image.new("RGB", (100, 100), color=(255, 255, 255)).save(p1)

        cfg = {
            "pipeline": {
                "output_dir": str(tmp_path / "output"),
                "checkpoints_dir": str(tmp_path / "checkpoints"),
                "enable_checkpoints": True,
            },
            "mock_models": True,
        }

        pipeline1 = MangaTranslationPipeline(config=cfg)
        pipeline1.run(chapter_pages=[p1], chapter_name="resume_test", resume_from_checkpoints=False)

        # Second pipeline run with resume_from_checkpoints=True
        pipeline2 = MangaTranslationPipeline(config=cfg)
        results = pipeline2.run(chapter_pages=[p1], chapter_name="resume_test", resume_from_checkpoints=True)

        assert results["quality_report"].total_pages == 1
        assert results["pdf_path"].exists()
