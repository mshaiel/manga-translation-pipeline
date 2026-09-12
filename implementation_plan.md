# AI-Powered Manga Translation Pipeline — Master Implementation Plan

This implementation plan outlines the architecture, component design, phase breakdown, and verification steps for building the AI-Powered Manga Translation Pipeline. The project is designed as a production-grade, portfolio-defining codebase for international Master's admissions and scholarship committees, optimized for stage-batched execution on Google Colab (T4 GPU) and local development.

---

## Architecture & System Design Overview

```
Input: Chapter Manga Images + [Optional] Character Bank + [Optional] User Context
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Detection & Speaker Diarization (ragavsachdeva/magiv2)           │
│ • Chapter-wide inference (do_ocr=False)                                   │
│ • Predicts: panels, text boxes, characters (normalized 0-1 coordinates)   │
│ • Associates text boxes to character speakers via visual tail cues        │
│ • Separates essential dialogue from SFX candidates (is_essential_text)    │
│ • Unload: explicit VRAM release (del model, empty_cache, gc.collect)      │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Japanese Manga OCR (kha-white/manga-ocr)                         │
│ • Crops text bounding boxes from original image (converted to pixels)     │
│ • Feeds PIL crops into manga-ocr (handles vertical/furigana/stylized)     │
│ • Unload: explicit VRAM release (del mocr, empty_cache, gc.collect)       │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: SFX Classification & Routing                                     │
│ • Evaluates Magi's is_essential_text flag + Katakana/confidence heuristics│
│ • Routes dialogue to inpainting and SFX to Viz-style offset subtitles     │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Reading Order & Rolling Context Assembly                         │
│ • Sorts panels Right-to-Left, Top-to-Bottom (column/row partitioning)     │
│ • Sorts text boxes within panels R-to-L, Top-to-Bottom                    │
│ • Assembles 6-page sliding window dialogue, cumulative characters,        │
│   periodic scene summaries (every 5 pages), and injected user context     │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Structured Translation LLM (Gemini 2.5 Flash / Fallback)         │
│ • Provider-swappable interface (TranslationProvider ABC)                  │
│ • Enforced JSON output via response_schema mapping 1:1 to text IDs        │
│ • Optional vision mode: attaches panel crops on low confidence            │
│ • Exponential backoff retry logic (1s, 2s, 4s, 8s, max 3)                 │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 6: Typesetting Engine (Pillow)                                      │
│ • Dialogue: solid white bounding box inpaint -> dynamic font shrink loop  │
│ • SFX: semi-transparent offset subtitle badge near original art           │
│ • Manga typography: Anime Ace 2.0 / CC-licensed font                      │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 7: Exporters & Quality Assessment                                   │
│ • Generates translated PNGs, compiled PDF (reportlab), and CBZ archive    │
│ • Produces quality_report.json (flags, confidence, cost & runtime stats)  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Phase Breakdown (Phase-by-Phase Execution)

We will proceed strictly phase by phase. Each phase has concrete deliverables and verification gates before moving to the next.

```
Phase 1: Project Scaffolding, Core Foundation & Schemas
   │
   ▼
Phase 2: Vision & Diarization Engine (Magi v2) & Reading Order
   │
   ▼
Phase 3: Japanese OCR (manga-ocr) & SFX Routing
   │
   ▼
Phase 4: Rolling Context Memory & Translation Prompt Engineering
   │
   ▼
Phase 5: Translation Engine (Provider-Swappable LLM Wrapper)
   │
   ▼
Phase 6: Typesetting Engine & Multi-Format Exporters
   │
   ▼
Phase 7: End-to-End Pipeline Orchestration & Quality Reporting
   │
   ▼
Phase 8: Colab Notebook & Interactive Gradio Application
   │
   ▼
Phase 9: Evaluation Benchmark Suite, Example Assets & Final Polish
```

---

### Phase 1: Project Scaffolding, Core Foundation & Schemas
**Objective:** Set up the package structure, dependency management, shared utilities, and Pydantic schemas.

* **Components to Build:**
  1. `pyproject.toml` & `requirements.txt`: Clean dependency specification with version pinning.
  2. `config/default_config.yaml` & `config/.env.example`: Central configuration (paths, model IDs, context limits, font paths, retry limits).
  3. `assets/fonts/`: Download and bundle a redistributable, CC-licensed manga font (e.g. *Anime Ace 2.0* or *CC Wild Words*) + `LICENSE.txt`.
  4. `src/utils/gpu_utils.py`:
     - Memory cleanup context manager (`managed_gpu_memory`).
     - Explicit VRAM flushing (`torch.cuda.empty_cache()`, `gc.collect()`).
     - Hardware inspection utilities (detecting CUDA device, VRAM tracking).
  5. `src/utils/image_utils.py`:
     - Image loading: Grayscale -> RGB conversion (`Image.open(p).convert("L").convert("RGB") -> np.array`).
     - Coordinate transformations: Normalized `[0, 1]` to pixel coordinates and vice versa with bounds clipping.
     - Cropping utilities for text boxes and panels returning PIL Images.
  6. `src/translation/schemas.py`:
     - `BoundingBox`: `[x1, y1, x2, y2]` with `to_pixels(w, h)` and `to_dict()` helpers.
     - `TextBox`: id, bbox, ocr_text, is_essential, speaker_name, speaker_cluster_id, ocr_confidence.
     - `PageDetection`: page_index, panels, text_boxes, characters, associations.
     - `TranslationItem`: id, english, translator_note.
     - `TranslationResponse`: translations list, scene_summary_update, confidence.
     - `RollingContext`: chapter_info, characters_seen, recent_dialogue, scene_summary.
     - `QualityReport`: chapter summary, per-page metrics, flags, pipeline metadata.
* **Verification Gate:**
  - Automated tests in `tests/test_schemas.py` and `tests/test_utils.py` validating normalization roundtrips, image conversion, and schema serialization.

---

### Phase 2: Vision & Diarization Engine (Magi v2) & Reading Order
**Objective:** Implement the chapter-wide detection and character/speaker diarization layer and manga reading-order algorithm.

* **Components to Build:**
  1. `src/detection/magi_detector.py`:
     - Hugging Face `AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True)`.
     - Chapter-wide prediction orchestrator calling `do_chapter_wide_prediction(chapter_pages, character_bank, do_ocr=False)`.
     - Output parser converting raw Magi dictionaries into strongly typed `PageDetection` Pydantic models.
     - Character bank handling (named crops vs empty bank cluster fallback).
     - Strict model unloading and VRAM clearing after stage completion.
  2. `src/typesetting/reading_order.py`:
     - Manga panel ordering heuristic: Right-to-Left, Top-to-Bottom via vertical column clustering and row sorting.
     - Text box ordering within panels: Bounding box containment checking, then R-to-L, Top-to-Bottom sorting.
     - Unassociated text box handling (floating text, edge narration).
* **Verification Gate:**
  - `tests/test_reading_order.py` with synthetic multi-panel and multi-bubble test fixtures ensuring R-to-L ordering is mathematically correct.
  - Schema mapping verification confirming raw Magi v2 outputs cleanly translate into `PageDetection`.

---

### Phase 3: Japanese OCR (manga-ocr) & SFX Routing
**Objective:** Implement the Japanese text recognition layer and non-essential text/SFX heuristic classifier.

* **Components to Build:**
  1. `src/ocr/manga_ocr_engine.py`:
     - Wrapper around `manga_ocr.MangaOcr`.
     - Sequential/batched crop OCR over detected text boxes.
     - Error handling for invalid/degenerate crops (zero-width or out-of-bounds).
     - Complete GPU unloading and cache flushing after chapter text extraction.
  2. `src/ocr/sfx_classifier.py`:
     - Multi-signal SFX candidate identification:
       - Primary signal: Magi's `is_essential_text == False`.
       - Secondary heuristic: 1-2 character isolated tokens or pure Katakana strings with low OCR confidence.
     - Tags `TextBox.is_sfx = True/False`.
* **Verification Gate:**
  - Integration test feeding mock crops through the OCR engine.
  - Test suite asserting correct classification of dialogue bubbles vs SFX candidates.

---

### Phase 4: Rolling Context Memory & Translation Prompt Engineering
**Objective:** Build the linguistic context engine that solves Japanese pro-drop using cross-page dialogue history and visual speaker cues.

* **Components to Build:**
  1. `src/context/memory_manager.py`:
     - Rolling memory lifecycle management (`RollingContext`).
     - 6-page sliding window for `recent_dialogue` (pruning older dialogue).
     - Cumulative character registry (`characters_seen`).
     - Periodic `scene_summary` trigger (every 5 pages or per chapter transition).
     - Token count budgeting (< 2,000 tokens) to guarantee cost and latency predictability.
     - Ingestion of external story context (`user_context.txt` or UI input).
  2. `src/translation/prompt_builder.py`:
     - Constructs structured per-page translation prompts.
     - Formats reading-order dialogue with speaker tags (`Luffy: [おれは海賊王になる！]`).
     - Injects rolling JSON context memory.
     - Multimodal vision prompt adaptation (instructions referencing attached panel crops when vision mode is enabled).
* **Verification Gate:**
  - `tests/test_context_memory.py`: Verifies sliding window eviction at 6 pages, cumulative character retention, and serialization.
  - `tests/test_prompt_builder.py`: Validates prompt formatting, speaker tag presence, and token budget compliance.

---

### Phase 5: Translation Engine (Provider-Swappable LLM Wrapper)
**Objective:** Implement resilient LLM translation with strict JSON schema enforcement and vision fallback.

* **Components to Build:**
  1. `src/translation/base_provider.py`:
     - Abstract Base Class `TranslationProvider` defining `translate(request: TranslationRequest) -> TranslationResponse`.
  2. `src/translation/gemini_provider.py`:
     - Gemini 2.5 Flash implementation via modern SDK (`google-genai` / `google-generativeai`).
     - Enforced structured JSON via `response_mime_type="application/json"` and `response_schema=TranslationResponse`.
     - Multimodal support: attaches panel crop image when requested.
     - Resilient retry handler with exponential backoff (1s, 2s, 4s, 8s, max 3 retries) and malformed response recovery.
  3. `src/translation/openai_provider.py`:
     - Fallback provider (GPT-4o-mini) using Pydantic structured outputs (`beta.chat.completions.parse`).
* **Verification Gate:**
  - `tests/test_translation.py`: Mock API responses testing JSON schema validation, retry backoff on 429/500 errors, and vision payload assembly.

---

### Phase 6: Typesetting Engine & Multi-Format Exporters
**Objective:** Build clean text rendering and multi-format document packaging.

* **Components to Build:**
  1. `src/typesetting/renderer.py`:
     - Dialogue rendering:
       - Solid white bounding box erase/inpaint.
       - Dynamic font-fit algorithm: starts at max font size, loops `textwrap.wrap()` and `draw.textbbox()` until height and width fit.
       - Centered horizontal and vertical alignment.
     - SFX rendering (Viz Media style):
       - Leaves original manga art untouched.
       - Renders small, italicized translated label near the bounding box.
       - Uses semi-transparent background badge for contrast.
  2. `src/export/exporter.py`:
     - PNG exporter for individual translated pages.
     - PDF compiler (`reportlab` / Pillow) for full chapter compilation.
     - CBZ archiver (standard Comic Book Zip format).
* **Verification Gate:**
  - `tests/test_typesetting.py`: Renders test sentences into synthetic bounding boxes of various aspect ratios, asserting no text overflows the bounding box.
  - Exporter tests ensuring generated PDF and CBZ files are valid and openable.

---

### Phase 7: End-to-End Pipeline Orchestration & Quality Reporting
**Objective:** Wire all components into the stage-batched pipeline with checkpoint persistence and automated quality reporting.

* **Components to Build:**
  1. `src/utils/quality_report.py`:
     - Aggregates per-page quality flags: low OCR confidence, unassigned speakers, high-frequency SFX, vision mode activations.
     - Compiles API token usage, cost estimation (~$0.02/chapter), and per-stage execution durations into `quality_report.json`.
  2. `src/pipeline.py`:
     - Main orchestrator executing Stage 1 -> Unload -> Stage 2 -> Unload -> Stage 3/4 -> Stage 5 -> Stage 6 -> Stage 7.
     - Checkpoint manager: saves stage intermediate outputs to disk (`checkpoints/stage_*.json`), allowing a disconnected Colab session to resume without re-running heavy GPU stages.
* **Verification Gate:**
  - End-to-end dry run test using mock detectors and mock LLM calls validating full lifecycle execution and checkpoint restoration.

---

### Phase 8: Colab Notebook & Interactive Gradio Application
**Objective:** Provide the interactive deployment layers (thin Colab runner and full browser-based demo).

* **Components to Build:**
  1. `notebooks/manga_translation_colab.ipynb`:
     - 5-cell clean design: Setup/Install, Configuration/Secrets (`userdata`), Run Pipeline, Review/Export, Evaluation.
     - Direct calls to `src/` modules; zero embedded business logic.
  2. `src/ui/app.py` (Gradio Interface):
     - Multi-image upload for manga pages + optional character bank crops + user context text field.
     - Controls: Vision mode toggle, context window slider, "Translate Chapter" action.
     - Gallery view with side-by-side original vs translated comparison.
     - One-click PDF/CBZ download and quality report viewer.
* **Verification Gate:**
  - Local Gradio launch test verifying UI layout and event bindings.
  - Colab notebook syntax and cell execution validation.

---

### Phase 9: Evaluation Benchmark Suite, Example Assets & Final Polish
**Objective:** Equip the repository with quantitative research benchmarks, open-source sample data, and flagship documentation.

* **Components to Build:**
  1. `evaluation/eval_ocr.py`: Character Error Rate (CER) calculation using `jiwer` against ground truth.
  2. `evaluation/eval_detection.py`: COCO-style mAP evaluation for panels, text boxes, and characters on PopMangaX.
  3. `evaluation/eval_translation.py`: BLEU (`sacrebleu`) / COMET evaluation comparing *Pipeline with Context* vs. *Raw Translation without Context*.
  4. `examples/`: Sample Creative Commons manga chapter (e.g. *Pepper & Carrot* by David Revoy) + character bank + `user_context.txt`.
  5. `README.md`: Flagship presentation with Mermaid architecture diagrams, side-by-side hero visual, benchmark tables, cost breakdown, citations, and licenses.
* **Verification Gate:**
  - Evaluation scripts execute cleanly on sample mock datasets.
  - Linter/type-check pass (`flake8` / `ruff`, `mypy`) ensuring zero warnings, no dead code, and 100% type-annotated APIs.

---

## Proposed Changes by Component

```
manga-translation-pipeline/
├── pyproject.toml                         # [NEW] Project build & package metadata
├── requirements.txt                       # [NEW] Pinned dependencies
├── .gitignore                             # [NEW] Git ignore rules
├── LICENSE                                # [NEW] MIT License
├── README.md                              # [NEW] Comprehensive documentation
│
├── config/
│   ├── default_config.yaml                # [NEW] Default pipeline configuration
│   └── .env.example                       # [NEW] Environment variable template
│
├── assets/
│   └── fonts/
│       ├── manga_font.ttf                 # [NEW] CC-licensed Manga font
│       └── LICENSE.txt                    # [NEW] Font redistribution license
│
├── src/
│   ├── __init__.py                        # [NEW] Package init
│   ├── pipeline.py                        # [NEW] Stage-batched pipeline orchestrator & checkpointing
│   │
│   ├── detection/
│   │   ├── __init__.py                    # [NEW]
│   │   └── magi_detector.py               # [NEW] Magi v2 model loader, inference & VRAM unloading
│   │
│   ├── ocr/
│   │   ├── __init__.py                    # [NEW]
│   │   ├── manga_ocr_engine.py            # [NEW] manga-ocr text extraction & VRAM unloading
│   │   └── sfx_classifier.py              # [NEW] Heuristic and Magi-signal SFX classifier
│   │
│   ├── context/
│   │   ├── __init__.py                    # [NEW]
│   │   └── memory_manager.py              # [NEW] Rolling JSON memory (6-page sliding window)
│   │
│   ├── translation/
│   │   ├── __init__.py                    # [NEW]
│   │   ├── schemas.py                     # [NEW] Pydantic v2 schemas
│   │   ├── base_provider.py               # [NEW] Abstract TranslationProvider interface
│   │   ├── gemini_provider.py             # [NEW] Gemini 2.5 Flash implementation with retries
│   │   ├── openai_provider.py             # [NEW] OpenAI fallback implementation
│   │   └── prompt_builder.py              # [NEW] Translation prompt assembly
│   │
│   ├── typesetting/
│   │   ├── __init__.py                    # [NEW]
│   │   ├── reading_order.py               # [NEW] R-to-L, Top-to-Bottom panel/text sorting
│   │   └── renderer.py                    # [NEW] Pillow inpainting, dynamic font-fit & SFX labels
│   │
│   ├── export/
│   │   ├── __init__.py                    # [NEW]
│   │   └── exporter.py                    # [NEW] PNG, PDF (reportlab), CBZ export
│   │
│   ├── ui/
│   │   ├── __init__.py                    # [NEW]
│   │   └── app.py                         # [NEW] Gradio application
│   │
│   └── utils/
│       ├── __init__.py                    # [NEW]
│       ├── image_utils.py                 # [NEW] Image conversions, bboxes, cropping
│       ├── gpu_utils.py                   # [NEW] Context managers, empty_cache, GC
│       └── quality_report.py              # [NEW] quality_report.json generation
│
├── notebooks/
│   └── manga_translation_colab.ipynb      # [NEW] 5-cell thin Colab execution notebook
│
├── evaluation/
│   ├── eval_ocr.py                        # [NEW] CER evaluation
│   ├── eval_detection.py                  # [NEW] Detection mAP evaluation
│   └── eval_translation.py                # [NEW] BLEU/COMET evaluation
│
├── tests/
│   ├── test_schemas.py                    # [NEW] Pydantic model serialization tests
│   ├── test_utils.py                      # [NEW] Image & GPU utility tests
│   ├── test_reading_order.py              # [NEW] Reading order sorting tests
│   ├── test_context_memory.py             # [NEW] Sliding window & token budget tests
│   ├── test_prompt_builder.py             # [NEW] Prompt formatting tests
│   └── test_typesetting.py                # [NEW] Font-fit & rendering tests
│
└── examples/
    ├── input/
    │   ├── pages/                         # [NEW] Sample CC manga pages
    │   ├── character_bank/                # [NEW] Sample character crops
    │   └── user_context.txt               # [NEW] Sample story background
    └── output/                            # [NEW] Directory for translated outputs
```

---

## Verification Plan

### Automated Testing
* **Unit Tests (`pytest`):**
  - Schemas: Pydantic validation of coordinates, translations, and quality reports.
  - Image Utils: Bounding box normalization, clipping, and crop extraction.
  - Reading Order: Right-to-left panel and text ordering logic against complex multi-panel layouts.
  - Context Memory: 6-page sliding window eviction, cumulative character tracking, and token bounds.
  - Prompt Builder: Prompt structure, speaker tag insertion, and vision instructions.
  - Typesetting: Font-fit convergence within bounding boxes without text truncation.
  - Exporter: Generation and structural verification of PNG, PDF, and CBZ files.
* **Pipeline Integration:**
  - Mock-driven end-to-end pipeline run verifying stage data flow, checkpoint persistence, and resume functionality.

### Manual Verification
* Visual inspection of rendered typesetting on sample manga panels (dialogue contrast, font scaling, SFX subtitle placement).
* Validation of Colab notebook execution flow and secrets handling in Google Colab.
* Verification of Gradio interface components and side-by-side gallery viewer.
