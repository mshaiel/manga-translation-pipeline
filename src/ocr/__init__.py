"""Japanese Manga OCR and SFX classification package."""

from src.ocr.manga_ocr_engine import MangaOcrEngine
from src.ocr.sfx_classifier import is_sfx_candidate

__all__ = ["MangaOcrEngine", "is_sfx_candidate"]
