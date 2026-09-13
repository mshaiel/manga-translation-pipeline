"""Prompt construction utilities for structured Japanese-to-English manga translation.

Assembles translation requests containing reading-order dialogue, speaker attribution hints,
rolling JSON context memory, and multimodal vision verification instructions.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from src.translation.schemas import RollingContext, TextBox, TranslationRequestItem


def build_dialogue_payload(
    text_boxes: Sequence[TextBox | TranslationRequestItem],
) -> list[dict[str, Any]]:
    """Convert ordered TextBoxes or TranslationRequestItems into clean JSON items for the translation prompt.

    Args:
        text_boxes: TextBoxes or TranslationRequestItems sorted in manga reading order.

    Returns:
        List of dictionaries with keys: 'id', 'speaker', 'japanese', 'is_sfx'.
    """
    items: list[dict[str, Any]] = []

    for tb in text_boxes:
        if isinstance(tb, TranslationRequestItem):
            items.append(tb.model_dump(exclude_none=False))
            continue

        speaker_val: str | None = getattr(tb, "speaker", None) or getattr(tb, "speaker_name", None)
        cluster_id = getattr(tb, "speaker_cluster_id", None)
        if not speaker_val and cluster_id is not None:
            speaker_val = f"Cluster_{cluster_id}"

        item = TranslationRequestItem(
            id=tb.id,
            speaker=speaker_val,
            japanese=getattr(tb, "japanese", None) or getattr(tb, "ocr_text", ""),
            is_sfx=bool(getattr(tb, "is_sfx", False)),
        )
        items.append(item.model_dump(exclude_none=False))

    return items


def build_translation_prompt(
    text_boxes: Sequence[TextBox | TranslationRequestItem],
    context: RollingContext,
    vision_mode: bool = False,
    request_scene_summary: bool = False,
) -> str:
    """Construct the complete prompt for the LLM translation call.

    Args:
        text_boxes: Dialogue and SFX text boxes on the page in reading order.
        context: Current rolling JSON context memory.
        vision_mode: Whether a panel or page crop is attached as multimodal input.
        request_scene_summary: Whether to instruct the LLM to update the scene summary.

    Returns:
        Formatted prompt string adhering to the project specification.
    """
    # 1. Format Context
    context_json_str = json.dumps(
        context.model_dump(exclude_none=True),
        ensure_ascii=False,
        indent=2,
    )

    # 2. Format Dialogue Payload in Reading Order
    dialogue_payload = build_dialogue_payload(text_boxes)
    dialogue_json_str = json.dumps(
        dialogue_payload,
        ensure_ascii=False,
        indent=2,
    )

    # 3. Assemble Instructions
    instructions = [
        "- Translate naturally, not literally. Manga dialogue should feel alive and punchy.",
        "- Preserve each character's speech style, tone, and distinct personality.",
        "- For SFX (is_sfx=true), provide a short, dynamic English comic equivalent (e.g., 'BOOM!', 'THUD', 'RUMBLE').",
        "- Use the speaker attribution and rolling context to resolve pronouns — Japanese frequently drops grammatical subjects.",
    ]

    if vision_mode:
        instructions.append(
            "- MULTIMODAL VERIFICATION: A visual image of the manga panel/page is attached. "
            "Use it to verify or correct the OCR text if words appear corrupted, truncated, or ambiguous."
        )

    if request_scene_summary:
        instructions.append(
            "- SCENE SUMMARY: Provide a concise 2-3 sentence update in `scene_summary_update` summarizing "
            "the current narrative events and character interactions."
        )
    else:
        instructions.append(
            "- Keep `scene_summary_update` concise or null if no significant narrative transition occurred."
        )

    instructions.append("- Return strictly valid JSON matching the requested response schema with no preamble.")

    instructions_text = "\n".join(instructions)

    prompt = f"""You are an expert Japanese-to-English manga translator. Translate the following manga page dialogue.

CONTEXT:
{context_json_str}

THIS PAGE'S DIALOGUE (in reading order):
{dialogue_json_str}

INSTRUCTIONS:
{instructions_text}
"""
    return prompt.strip()
