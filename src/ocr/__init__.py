"""Japanese Manga OCR and SFX classification package."""

from src.ocr.box_merger import merge_adjacent_text_boxes
from src.ocr.manga_ocr_engine import MangaOcrEngine
from src.ocr.sfx_classifier import is_sfx_candidate

__all__ = ["MangaOcrEngine", "is_sfx_candidate", "merge_adjacent_text_boxes"]
