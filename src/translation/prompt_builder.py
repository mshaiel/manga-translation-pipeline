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
        "- Translate all dialogue and narration naturally into fluent, punchy English that reads like a professional manga translation (think Viz Media or official English releases).",
        "- Use contractions, slang, exclamations, and varied sentence structure. Avoid stiff or literal translations. Manga dialogue should feel alive and match the energy of the scene.",
        "- Preserve each character's distinct speech patterns. A tough character should sound tough. A timid character should sound timid.",
        "- For SFX (is_sfx=true), provide a short English comic sound equivalent (BOOM, CRASH, THUD, etc.).",
        "- Use the speaker attribution and rolling context to resolve omitted subjects — Japanese frequently drops pronouns.",
        "- If the Japanese text for an item is empty, garbled, or unrecognizable, return the original text as-is in the 'english' field and add a translator_note saying \"OCR unclear\".",
        "- INDEPENDENT CLAUSES: Each dialogue item corresponds to a separate speech bubble or panel. Do NOT repeat, echo, or prepend phrases, questions, or exclamations from previous items into subsequent items (e.g. do NOT repeat 'Right?!' or 'Wait!' at the start of the next item unless it is explicitly present in that item's Japanese text).",
        "- STRICT 1:1 FIDELITY: Every item's 'english' field must translate ONLY the Japanese text provided in that specific item's 'japanese' field. Never merge, repeat, or bleed sentences across different item IDs.",
        "- Do NOT invent dialogue that isn't present in the source text.",
        "- Return valid JSON matching the response schema.",
    ]

    if vision_mode:
        instructions.append(
            "- MULTIMODAL VERIFICATION: A visual image of the manga page is attached for scene context, speaker identification, and reading order. "
            "Translate strictly the dialogue corresponding to each item's Japanese text. Do NOT borrow, duplicate, or transfer dialogue from one speech bubble into another. "
            "If an item contains silence, dots, or an ellipsis, return '...' in the 'english' field. Never invent spoken dialogue for silent characters."
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
