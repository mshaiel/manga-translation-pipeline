"""Unit tests for ChapterExporter (PNG, PDF, CBZ)."""

import zipfile

from PIL import Image

from src.export.exporter import ChapterExporter


class TestChapterExporter:
    """Test suite for multi-format chapter packaging."""

    def test_export_png(self, tmp_path):
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        out_png = tmp_path / "page_01.png"

        saved_path = ChapterExporter.export_page_png(img, out_png)
        assert saved_path.exists()

        with Image.open(saved_path) as loaded:
            assert loaded.size == (100, 100)
            assert loaded.format == "PNG"

    def test_export_pdf(self, tmp_path):
        img1 = Image.new("RGB", (100, 150), color=(255, 255, 255))
        img2 = Image.new("RGB", (100, 150), color=(200, 200, 200))
        out_pdf = tmp_path / "chapter.pdf"

        saved_path = ChapterExporter.export_chapter_pdf([img1, img2], out_pdf)
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

    def test_export_cbz(self, tmp_path):
        img1 = Image.new("RGB", (100, 150), color=(255, 255, 255))
        img2 = Image.new("RGB", (100, 150), color=(0, 0, 0))
        out_cbz = tmp_path / "chapter.cbz"

        saved_path = ChapterExporter.export_chapter_cbz([img1, img2], out_cbz)
        assert saved_path.exists()

        # Inspect CBZ zip contents
        with zipfile.ZipFile(saved_path, "r") as zip_ref:
            names = zip_ref.namelist()
            assert "001.png" in names
            assert "002.png" in names
            assert len(names) == 2
