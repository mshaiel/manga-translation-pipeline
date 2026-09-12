"""Rolling JSON Context Memory manager for Japanese-to-English translation.

Japanese is a pro-drop language (subjects are frequently omitted). This module maintains
a cross-page sliding window of recent dialogue, cumulative character sightings, and
periodic scene summaries, solving pronoun ambiguity by grounding the LLM in chapter context.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from src.translation.schemas import (
    ChapterInfo,
    CharacterSeen,
    DialogueEntry,
    PageDetection,
    RollingContext,
    TextBox,
    TranslationItem,
)

logger = logging.getLogger(__name__)


class RollingContextManager:
    """Manages the rolling context memory across chapter pages.

    Maintains:
        1. Chapter metadata and user-injected background context.
        2. Cumulative character registry (`characters_seen`).
        3. Sliding-window dialogue history (`recent_dialogue`, defaults to last 6 pages).
        4. High-level scene summary updated periodically (every N pages).
    """

    def __init__(
        self,
        sliding_window_pages: int = 6,
        scene_summary_interval: int = 5,
        max_context_tokens: int = 2000,
    ) -> None:
        """Initialize Rolling Context Manager.

        Args:
            sliding_window_pages: Number of past pages of dialogue retained in history.
            scene_summary_interval: Page interval at which scene summaries are updated.
            max_context_tokens: Soft token budget for context serialization (~2000 tokens).
        """
        self.sliding_window_pages = sliding_window_pages
        self.scene_summary_interval = scene_summary_interval
        self.max_context_tokens = max_context_tokens

        self.context = RollingContext(
            chapter_info=ChapterInfo(),
            characters_seen={},
            recent_dialogue=[],
            scene_summary="",
        )

    def set_chapter_info(
        self,
        title: str | None = None,
        genre: str | None = None,
        user_context: str | None = None,
    ) -> None:
        """Update chapter metadata and narrative guidance."""
        if title is not None:
            self.context.chapter_info.title = title
        if genre is not None:
            self.context.chapter_info.genre = genre
        if user_context is not None:
            self.context.chapter_info.user_context = user_context

    def load_user_context_from_file(self, file_path: str | Path) -> None:
        """Read free-text story context from a user_context.txt file.

        Args:
            file_path: Path to user_context.txt.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("User context file not found at: %s", path)
            return

        text = path.read_text(encoding="utf-8").strip()
        if text:
            self.context.chapter_info.user_context = text
            logger.info("Loaded user context (%d chars) from %s", len(text), path)

    def register_page_characters(self, page_index: int, detection: PageDetection) -> None:
        """Update the cumulative character registry with characters appearing on this page.

        Characters are never dropped from memory across the chapter.

        Args:
            page_index: 0-indexed page index (converted to 1-indexed for display).
            detection: PageDetection containing detected characters and names.
        """
        page_num = page_index + 1

        for char in detection.characters:
            char_key = char.name or f"Character_Cluster_{char.cluster_id or char.id}"

            if char_key in self.context.characters_seen:
                record = self.context.characters_seen[char_key]
                record.last_seen_page = page_num
                if char.cluster_id is not None:
                    record.cluster_id = char.cluster_id
            else:
                self.context.characters_seen[char_key] = CharacterSeen(
                    first_appeared_page=page_num,
                    last_seen_page=page_num,
                    cluster_id=char.cluster_id,
                    description=f"Appeared in chapter (cluster {char.cluster_id})",
                )

    def append_translated_page(
        self,
        page_index: int,
        text_boxes: list[TextBox],
        translations: list[TranslationItem],
        scene_summary_update: str | None = None,
    ) -> None:
        """Append translated lines to recent dialogue history and prune sliding window.

        Args:
            page_index: 0-indexed page index (stored as 1-indexed page).
            text_boxes: Original TextBoxes on this page.
            translations: Translated items matching text_boxes 1:1 by id.
            scene_summary_update: Optional summary update returned by LLM.
        """
        page_num = page_index + 1
        trans_map = {t.id: t.english for t in translations}

        for tb in text_boxes:
            translated_en = trans_map.get(tb.id, "")
            speaker = tb.speaker_name or (
                f"Cluster_{tb.speaker_cluster_id}" if tb.speaker_cluster_id is not None else None
            )

            entry = DialogueEntry(
                page=page_num,
                speaker=speaker or ("SFX" if tb.is_sfx else "unsure"),
                japanese=tb.ocr_text,
                english=translated_en,
                is_sfx=tb.is_sfx,
            )
            self.context.recent_dialogue.append(entry)

        # Update scene summary if provided
        if scene_summary_update and scene_summary_update.strip():
            self.context.scene_summary = scene_summary_update.strip()
            logger.info("Updated scene summary on page %d: %s", page_num, self.context.scene_summary)

        # Prune older dialogue outside sliding window
        self.prune_sliding_window(current_page_num=page_num)

    def prune_sliding_window(self, current_page_num: int) -> None:
        """Drop dialogue entries older than the sliding window threshold.

        Keeps the last `sliding_window_pages` (default 6 pages).
        """
        min_retained_page = max(1, current_page_num - self.sliding_window_pages + 1)
        before_count = len(self.context.recent_dialogue)

        self.context.recent_dialogue = [
            d for d in self.context.recent_dialogue if d.page >= min_retained_page
        ]

        dropped = before_count - len(self.context.recent_dialogue)
        if dropped > 0:
            logger.debug("Pruned %d dialogue entries outside %d-page window", dropped, self.sliding_window_pages)

    def should_request_scene_summary(self, page_index: int) -> bool:
        """Determine whether the LLM should be asked to update the scene summary.

        Triggers every `scene_summary_interval` pages (e.g., page 5, 10, 15...).
        """
        page_num = page_index + 1
        return (page_num % self.scene_summary_interval) == 0

    def estimate_token_count(self) -> int:
        """Approximate the token size of the serialized context memory.

        Uses a fast heuristic: 1 token ~= 4 characters for English / 1 token ~= 1-2 characters for Japanese.
        """
        serialized = self.to_json()
        return max(1, len(serialized) // 3)

    def get_context(self) -> RollingContext:
        """Return a copy of the current RollingContext model."""
        return self.context.model_copy(deep=True)

    def to_json(self, indent: int | None = None) -> str:
        """Serialize rolling context memory to clean JSON string."""
        data = self.context.model_dump(exclude_none=True)
        return json.dumps(data, ensure_ascii=False, indent=indent)

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Restore context memory from a dictionary (e.g., loaded from checkpoint)."""
        self.context = RollingContext.model_validate(data)
