"""Pydantic v2 schemas for the manga translation pipeline.

Provides type safety, serialization, and validation for all data flowing
between detection, OCR, context memory, LLM translation, typesetting, and quality reporting.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ==============================================================================
# Geometry & Spatial Models
# ==============================================================================

class BoundingBox(BaseModel):
    """Normalized bounding box represented as [x1, y1, x2, y2] in [0.0, 1.0]."""

    x1: float = Field(..., description="Normalized top-left X coordinate in [0.0, 1.0]")
    y1: float = Field(..., description="Normalized top-left Y coordinate in [0.0, 1.0]")
    x2: float = Field(..., description="Normalized bottom-right X coordinate in [0.0, 1.0]")
    y2: float = Field(..., description="Normalized bottom-right Y coordinate in [0.0, 1.0]")

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def parse_from_sequence(cls, data: Any) -> Any:
        """Allow initializing BoundingBox directly from a list or tuple of 4 floats."""
        if isinstance(data, (list, tuple)):
            if len(data) != 4:
                raise ValueError(f"Expected 4 coordinates for BoundingBox, got {len(data)}")
            return {"x1": float(data[0]), "y1": float(data[1]), "x2": float(data[2]), "y2": float(data[3])}
        return data

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBox:
        """Clamp coordinates to [0.0, 1.0] and ensure x1 <= x2, y1 <= y2."""
        clamped_x1 = max(0.0, min(1.0, self.x1))
        clamped_y1 = max(0.0, min(1.0, self.y1))
        clamped_x2 = max(0.0, min(1.0, self.x2))
        clamped_y2 = max(0.0, min(1.0, self.y2))

        self.x1 = min(clamped_x1, clamped_x2)
        self.y1 = min(clamped_y1, clamped_y2)
        self.x2 = max(clamped_x1, clamped_x2)
        self.y2 = max(clamped_y1, clamped_y2)
        return self

    def to_pixels(self, width: int, height: int, clip: bool = True) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute integer pixel bounds (px1, py1, px2, py2)."""
        from src.utils.image_utils import bbox_normalized_to_pixel
        return bbox_normalized_to_pixel([self.x1, self.y1, self.x2, self.y2], width, height, clip=clip)

    def to_list(self) -> list[float]:
        """Return coordinates as a list of floats [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def width(self) -> float:
        """Normalized box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Normalized box height."""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """Normalized box area."""
        return self.width * self.height


# ==============================================================================
# Detection & Page Layout Models
# ==============================================================================

class PanelBox(BaseModel):
    """Detected comic panel."""
    id: int = Field(..., description="Unique index of the panel within the page")
    bbox: BoundingBox = Field(..., description="Normalized bounding box of the panel")
    reading_order_index: int | None = Field(None, description="Sequential reading order index (0-indexed)")


class CharacterBox(BaseModel):
    """Detected character appearance box."""
    id: int = Field(..., description="Unique index of character detection on the page")
    bbox: BoundingBox = Field(..., description="Normalized bounding box of character")
    cluster_id: int | None = Field(None, description="Identity cluster ID cross-page")
    name: str | None = Field(None, description="Character name if matched to character bank")


class TextBox(BaseModel):
    """Detected text bubble or sound effect region."""
    id: int = Field(..., description="Index mapping 1:1 with Magi text output")
    bbox: BoundingBox = Field(..., description="Normalized bounding box of the text region")
    ocr_text: str = Field(default="", description="Recognized Japanese text from manga-ocr")
    is_essential: bool = Field(default=True, description="True for dialogue/narration; False for SFX from Magi")
    is_sfx: bool = Field(default=False, description="True if routed to SFX rendering style")
    has_tail: bool = Field(default=False, description="True if text box has an associated speech bubble tail")
    speaker_name: str | None = Field(default=None, description="Attributed character name or 'narration'")
    speaker_cluster_id: int | None = Field(default=None, description="Magi cluster ID if name unknown")
    speaker_confidence: float = Field(default=1.0, description="Confidence of speaker attribution in [0.0, 1.0]")
    ocr_confidence: float = Field(default=1.0, description="Confidence of OCR recognition in [0.0, 1.0]")
    is_silence: bool = Field(default=False, description="True if text region represents silence, vertical dots, or pause")
    panel_id: int | None = Field(default=None, description="ID of the containing panel")
    reading_order_index: int | None = Field(default=None, description="Sequential reading order index on the page")


class PageDetection(BaseModel):
    """Full structured detection results for a single manga page."""
    page_index: int = Field(..., description="0-indexed page number")
    image_width: int | None = Field(default=None, description="Original image pixel width")
    image_height: int | None = Field(default=None, description="Original image pixel height")
    panels: list[PanelBox] = Field(default_factory=list, description="Detected panels")
    text_boxes: list[TextBox] = Field(default_factory=list, description="Detected text regions")
    characters: list[CharacterBox] = Field(default_factory=list, description="Detected characters")
    text_character_associations: list[tuple[int, int]] = Field(
        default_factory=list,
        description="List of (text_idx, char_idx) speaker linkages from Magi",
    )
    character_names: list[str] = Field(default_factory=list, description="Matched character names")
    character_cluster_labels: list[int] = Field(default_factory=list, description="Cluster labels per character")


# ==============================================================================
# Rolling Context Memory Models
# ==============================================================================

class ChapterInfo(BaseModel):
    """High-level metadata and user-provided story context for a chapter."""
    title: str | None = Field(default=None, description="Title of the chapter or series")
    genre: str | None = Field(default=None, description="Genre tags (e.g., shonen, romance, comedy)")
    user_context: str | None = Field(
        default=None,
        description="Free-text narrative context injected by user (e.g., character secrets, tone guidance)",
    )


class CharacterSeen(BaseModel):
    """Cumulative record of a recognized character across pages."""
    first_appeared_page: int = Field(..., description="First page index where character was detected")
    last_seen_page: int = Field(..., description="Most recent page index where character was detected")
    cluster_id: int | str | None = Field(default=None, description="Magi identity cluster identifier")
    description: str | None = Field(default=None, description="Cumulative context-derived character summary")


class DialogueEntry(BaseModel):
    """Historical record of translated dialogue for the rolling context window."""
    page: int = Field(..., description="Page index where dialogue occurred")
    speaker: str | None = Field(default=None, description="Speaker name, 'narration', or 'unsure'")
    japanese: str = Field(..., description="Original OCR text")
    english: str = Field(..., description="Translated English text")
    is_sfx: bool = Field(default=False, description="Whether this entry was a sound effect")


class RollingContext(BaseModel):
    """Sliding-window context memory injected into each LLM translation prompt."""
    chapter_info: ChapterInfo = Field(default_factory=ChapterInfo, description="Chapter context and user prompts")
    characters_seen: dict[str, CharacterSeen] = Field(
        default_factory=dict,
        description="Cumulative registry of characters seen across the chapter",
    )
    recent_dialogue: list[DialogueEntry] = Field(
        default_factory=list,
        description="Recent dialogue history within the sliding window (last N pages)",
    )
    scene_summary: str = Field(
        default="",
        description="Compressed summary of recent plot events, updated periodically by the LLM",
    )


# ==============================================================================
# Translation Request & Response Models (LLM Interface)
# ==============================================================================

class TranslationRequestItem(BaseModel):
    """Single dialogue/SFX line sent to the LLM for translation."""
    id: int = Field(..., description="Maps 1:1 to Magi's text box index")
    speaker: str | None = Field(default=None, description="Attributed speaker name, 'narration', or null")
    japanese: str = Field(..., description="Original OCR Japanese text")
    is_sfx: bool = Field(default=False, description="Whether this item is a sound effect")

    @property
    def speaker_name(self) -> str | None:
        """Alias for speaker for duck typing with TextBox."""
        return self.speaker

    @property
    def speaker_cluster_id(self) -> int | None:
        """Alias for speaker_cluster_id for duck typing with TextBox."""
        return None

    @property
    def ocr_text(self) -> str:
        """Alias for ocr_text for duck typing with TextBox."""
        return self.japanese


class TranslationItem(BaseModel):
    """Single translated line returned by the LLM."""
    id: int = Field(..., description="Maps 1:1 back to Magi's text box index")
    english: str = Field(..., description="Natural, idiomatic English translation")
    translator_note: str | None = Field(default=None, description="Optional translator note or ambiguity flag")


class TranslationRequest(BaseModel):
    """Complete per-page payload dispatched to the TranslationProvider."""
    page_index: int = Field(..., description="Index of the page being translated")
    text_boxes: list[TranslationRequestItem] = Field(..., description="Dialogue items in reading order")
    context: RollingContext = Field(..., description="Current rolling memory state")
    page_image_path: str | None = Field(default=None, description="Path to page or panel crop for vision mode")
    vision_mode: bool = Field(default=False, description="Whether multimodal image attachment is activated")


class TranslationResponse(BaseModel):
    """Structured response schema enforced from the TranslationProvider."""
    translations: list[TranslationItem] = Field(
        ...,
        description="List of translations mapping 1:1 by id to the requested text boxes",
    )
    scene_summary_update: str | None = Field(
        default=None,
        description="Updated 2-3 sentence summary of current events if requested",
    )
    confidence: float = Field(default=1.0, description="Self-reported translation confidence in [0.0, 1.0]")


# ==============================================================================
# Quality Assurance & Pipeline Reporting Models
# ==============================================================================

class QualityFlag(BaseModel):
    """Anomaly or warning flag raised during pipeline execution."""
    text_box_id: int | None = Field(default=None, description="Associated text box index if applicable")
    issue: str = Field(..., description="Issue type code (e.g., 'low_ocr_confidence', 'no_speaker_attribution')")
    details: str | None = Field(default=None, description="Human-readable explanation or context")
    ocr_text: str | None = Field(default=None, description="Snippet of problematic text")


class PageQuality(BaseModel):
    """Per-page quality audit metrics."""
    page: int = Field(..., description="Page index (1-indexed for reporting)")
    num_panels: int = Field(..., description="Number of detected panels")
    num_text_boxes: int = Field(..., description="Total text boxes detected")
    num_characters_detected: int = Field(..., description="Total character boxes detected")
    flags: list[QualityFlag] = Field(default_factory=list, description="Quality flags raised for this page")
    vision_check_recommended: bool = Field(
        default=False,
        description="True if OCR or diarization uncertainty suggests visual inspection",
    )


class PipelineMetadata(BaseModel):
    """Execution metadata and billing cost estimates."""
    magi_version: str = Field(default="v2", description="Version of Magi used for detection/diarization")
    ocr_model: str = Field(default="manga-ocr", description="OCR engine identifier")
    llm_provider: str = Field(default="gemini-2.5-flash", description="Translation LLM identifier")
    total_api_cost_estimate_usd: float = Field(default=0.0, description="Estimated total API cost in USD")
    processing_time_seconds: float = Field(default=0.0, description="Total pipeline execution runtime in seconds")


class QualityReport(BaseModel):
    """Comprehensive chapter quality report serialized to quality_report.json."""
    chapter: str = Field(..., description="Identifier or folder name of the chapter")
    total_pages: int = Field(..., description="Total number of pages processed")
    total_text_boxes: int = Field(..., description="Total text boxes across all pages")
    total_sfx: int = Field(..., description="Total sound effects classified")
    per_page: list[PageQuality] = Field(..., description="Quality breakdown per page")
    pipeline_metadata: PipelineMetadata = Field(..., description="Execution metadata and costs")


# ==============================================================================
# End-to-End Page Execution Result
# ==============================================================================

class PageResult(BaseModel):
    """Composite state and artifact output for a single processed page."""
    page_index: int = Field(..., description="0-indexed page index")
    original_image_path: str = Field(..., description="Path to original input image")
    translated_image_path: str | None = Field(default=None, description="Path to rendered translated image")
    detection: PageDetection = Field(..., description="Detection and layout parsing results")
    translation: TranslationResponse | None = Field(default=None, description="LLM translation results")
    quality_flags: list[QualityFlag] = Field(default_factory=list, description="Quality flags for this page")
