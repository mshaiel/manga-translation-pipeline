"""Chapter export utilities for compiling translated manga into PNG, PDF, and CBZ formats.

Supports:
    - High-resolution individual PNG page exports.
    - Combined chapter PDF generation via Pillow.
    - Standard Comic Book Zip (CBZ) archive creation for reader apps (Tachiyomi, CDisplayEx).
"""

from __future__ import annotations

import io
import logging
import zipfile
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from src.utils.image_utils import load_pil_image

logger = logging.getLogger(__name__)


class ChapterExporter:
    """Manages multi-format export of translated manga chapters."""

    @staticmethod
    def export_page_png(
        image: Image.Image | str | Path,
        output_path: str | Path,
        optimize: bool = True,
    ) -> Path:
        """Export a single manga page image as a high-quality PNG.

        Args:
            image: PIL Image or source file path.
            output_path: Target .png destination path.
            optimize: Whether to apply PNG optimization.

        Returns:
            Resolved Path of the saved PNG.
        """
        pil_img = load_pil_image(image)
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        pil_img.save(dest, format="PNG", optimize=optimize)
        logger.debug("Exported page PNG to: %s", dest)
        return dest

    @staticmethod
    def export_chapter_pdf(
        page_images: Sequence[Image.Image | str | Path],
        output_path: str | Path,
        dpi: int = 150,
    ) -> Path:
        """Compile a sequence of translated pages into a unified multi-page PDF.

        Args:
            page_images: Sequence of page PIL images or file paths.
            output_path: Destination path ending in .pdf.
            dpi: Target resolution metadata.

        Returns:
            Resolved Path of the compiled PDF.
        """
        if not page_images:
            raise ValueError("Cannot generate PDF from empty page list.")

        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        # Load all pages as RGB PIL images
        pil_pages: list[Image.Image] = [load_pil_image(p) for p in page_images]

        first_page = pil_pages[0]
        remaining_pages = pil_pages[1:] if len(pil_pages) > 1 else []

        first_page.save(
            dest,
            format="PDF",
            save_all=True,
            append_images=remaining_pages,
            resolution=dpi,
        )

        logger.info("Compiled %d-page chapter PDF to: %s", len(pil_pages), dest)
        return dest

    @staticmethod
    def export_chapter_cbz(
        page_images: Sequence[Image.Image | str | Path],
        output_path: str | Path,
    ) -> Path:
        """Package translated pages into a standard Comic Book Zip (.cbz) archive.

        Pages are compressed as sequentially indexed PNG files (001.png, 002.png, ...).

        Args:
            page_images: Sequence of page PIL images or file paths.
            output_path: Destination path ending in .cbz.

        Returns:
            Resolved Path of the created CBZ archive.
        """
        if not page_images:
            raise ValueError("Cannot generate CBZ archive from empty page list.")

        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(dest, mode="w", compression=zipfile.ZIP_DEFLATED) as cbz:
            for idx, page in enumerate(page_images, start=1):
                pil_img = load_pil_image(page)
                filename = f"{idx:03d}.png"

                # Save PNG bytes into zip entry
                buf = io.BytesIO()
                pil_img.save(buf, format="PNG")
                cbz.writestr(filename, buf.getvalue())

        logger.info("Packaged %d pages into CBZ archive: %s", len(page_images), dest)
        return dest
