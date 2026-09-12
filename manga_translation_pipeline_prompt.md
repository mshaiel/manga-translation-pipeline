# AI-Powered Manga Translation Pipeline — Full Implementation Prompt

## Who You're Building This For

I'm a recent BSAI (Bachelor of Science in Artificial Intelligence) graduate building this as a flagship portfolio project for international Master's program applications and scholarships. I need it to demonstrate strong system design and resource-efficient architecture, not just working code — help me build something that's genuinely eye-catching and impressive to technical reviewers/interviewers.

**This is a portfolio centerpiece. Structure and code quality must read as production-minded, not a one-off Colab script.** Every module should have docstrings, type hints, and clean separation of concerns.

**This project will be uploaded to GitHub as a public repository.** It will be reviewed by Master's admissions committees and scholarship panels. Code quality, documentation, commit structure, and README presentation must be flawless. No placeholder code, no TODOs left behind, no sloppy imports, no dead code. Every file should look like it belongs in a professional open-source project.

---

## Hardware & Execution Constraints

- **Local machine:** GTX 1050, 4GB VRAM — light dev/testing only, cannot run the full pipeline.
- **Primary execution:** Google Colab (free tier, T4 GPU, ~15GB VRAM). GPU sessions last 4-5 hours per account; I have ~4 accounts.
- **Budget:** Minimal. Use the cheapest viable LLM API. Prefer Gemini 2.5 Flash (verify current pricing — as of mid-2025 it's one of the cheapest multimodal APIs).

**IMPORTANT — Colab-first execution model:**
- **ALL heavy work runs on Colab**, not locally. This includes: downloading models (Magi, manga-ocr), downloading datasets, running the full pipeline, and running the Gradio demo.
- **Nothing large is downloaded or executed locally.** The local machine is only used for writing code and pushing to GitHub.
- The workflow is: write code locally → push to GitHub → clone repo in Colab → run everything there.
- All model weights are downloaded at runtime in Colab (HuggingFace's `from_pretrained` handles this automatically — they cache to Colab's ephemeral disk).
- Datasets are downloaded and stored on Google Drive from within Colab.
- Design all file paths to work in a Colab environment (`/content/`, Google Drive mounts at `/content/drive/MyDrive/`).

---

## System Architecture Overview

```
Input: Folder of manga page images (one chapter)
       + (optional) character_bank/ folder with named character crop images
       + (optional) user_context.txt with free-text story context

Pipeline (stage-batched):
  ┌─────────────────────────────────────────────────────────────┐
  │ STAGE 1: Detection + Diarization (Magiv2)                  │
  │  - Detects: panels, text boxes, character boxes             │
  │  - Clusters characters by identity across pages             │
  │  - Links text boxes → speakers via tail direction/proximity │
  │  - Also provides reading order for text boxes               │
  │  - Output: per-page detection results dict                  │
  │  - Has built-in OCR (do_ocr=True) — BUT we use manga-ocr   │
  │    for Japanese because Magi's OCR is English-trained        │
  └──────────────────────┬──────────────────────────────────────┘
                         │ unload Magi from GPU
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STAGE 2: Japanese OCR (manga-ocr, kha-white)               │
  │  - Crops each text bounding box from the original image     │
  │  - Runs manga-ocr on each crop                              │
  │  - Purpose-built for Japanese manga: vertical/horizontal,   │
  │    furigana, stylized fonts, multi-line bubbles              │
  │  - Output: OCR text per text box                             │
  └──────────────────────┬──────────────────────────────────────┘
                         │ unload manga-ocr from GPU
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STAGE 3: SFX Classification                                 │
  │  - Text boxes Magi marks as NOT "is_essential_text" →       │
  │    classified as SFX candidates                              │
  │  - Also flag low-OCR-confidence regions as SFX candidates   │
  │  - SFX get translated but rendered as offset subtitles      │
  │    (Viz Media style — original art untouched)                │
  └──────────────────────┬──────────────────────────────────────┘
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STAGE 4: Context Assembly                                    │
  │  - Build rolling JSON context memory (sliding window)       │
  │  - Merge user-provided story context                         │
  │  - Assemble per-page translation prompts with:               │
  │    - OCR'd Japanese text in reading order                    │
  │    - Speaker attribution per text box                        │
  │    - Rolling context from previous pages                     │
  │    - Character names from character bank                     │
  └──────────────────────┬──────────────────────────────────────┘
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STAGE 5: Translation (LLM API — Gemini 2.5 Flash)           │
  │  - Default: text-only call (OCR + context + speaker info)   │
  │  - Optional vision mode: attach cropped panel image          │
  │    when OCR confidence is low or user toggles it             │
  │  - Structured JSON output enforced via response_schema       │
  │  - Provider-swappable via thin wrapper                       │
  │  - Retry with exponential backoff                            │
  └──────────────────────┬──────────────────────────────────────┘
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STAGE 6: Typesetting (Pillow)                               │
  │  - Dialogue: inpaint original text region white, then        │
  │    render translated text with dynamic font-fit              │
  │  - SFX: render as small offset subtitle near original       │
  │  - Export: individual PNGs + combined chapter PDF + CBZ      │
  └──────────────────────┬──────────────────────────────────────┘
                         ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STAGE 7: Quality Report                                      │
  │  - Per-page JSON report with confidence scores               │
  │  - Flags: low OCR confidence, unsure speaker attribution,   │
  │    SFX candidates, vision-check recommendations              │
  └─────────────────────────────────────────────────────────────┘

Output: translated_pages/ folder + chapter.pdf + chapter.cbz + quality_report.json
```

---

## Key Architectural Decision: Stage-Batched Execution

Use **stage-batched** processing (NOT per-page sequential):
1. Run Magi across ALL pages → unload model → `torch.cuda.empty_cache()`
2. Run manga-ocr across ALL text crops → unload model → `torch.cuda.empty_cache()`
3. Hit LLM API for all pages (no local GPU needed)
4. Typeset all pages (CPU only)

**Why:** Lower peak VRAM, enables independent benchmarking of each stage, more architecturally interesting for portfolio. Add explicit model loading/unloading with context managers.

---

## Detailed Component Specifications

### 1. Magi v2 — Detection & Speaker Diarization

**Repository:** https://github.com/ragavsachdeva/magi
**Paper (v2):** "Tails Tell Tales: Chapter-wide Manga Transcriptions with Character Names" (ACCV 2024, arXiv:2408.00298)
**HuggingFace model ID:** `ragavsachdeva/magiv2`
**Note:** There is also a v3 (`ragavsachdeva/magiv3`) based on Florence2 architecture (arXiv:2503.23344), but v2 is the most battle-tested for our use case and uses a simpler API. Use v2.

**Installation:** Magi v2 loads via HuggingFace `transformers.AutoModel` with `trust_remote_code=True`. No separate pip install needed beyond `transformers`, `torch`, `numpy`, `Pillow`.

**How to load and run Magiv2:**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()
```

**Input format:**
- `chapter_pages`: list of numpy arrays (H, W, 3), uint8 RGB. Images should be read as grayscale then converted to RGB: `Image.open(path).convert("L").convert("RGB")` then `np.array(image)`
- `character_bank`: dict with keys `"images"` (list of numpy arrays of character face crops) and `"names"` (list of corresponding name strings). If no character bank available, you can pass an empty bank and Magi will still detect/cluster characters but won't name them.

**Main inference call:**
```python
with torch.no_grad():
    per_page_results = model.do_chapter_wide_prediction(chapter_pages, character_bank, use_tqdm=True, do_ocr=False)
```
Set `do_ocr=False` because we use manga-ocr for Japanese text instead of Magi's built-in English OCR.

**Output format** — `per_page_results` is a list of dicts, one per page. Each dict contains:
- `"panels"`: list of panel bounding boxes `[x1, y1, x2, y2]` (normalized 0-1 coordinates)
- `"texts"`: list of text bounding boxes `[x1, y1, x2, y2]` (normalized 0-1 coordinates)
- `"characters"`: list of character bounding boxes `[x1, y1, x2, y2]` (normalized 0-1 coordinates)
- `"text_character_associations"`: list of tuples `(text_idx, char_idx)` mapping each text box to its speaker character
- `"character_names"`: list of character name strings (from character bank matching)
- `"is_essential_text"`: list of booleans — `True` for dialogue/narration, `False` for SFX/non-essential
- `"character_cluster_labels"`: cluster IDs for character identity across pages
- `"ocr"`: list of OCR strings (only populated if `do_ocr=True` — we skip this)

**Critical:** Bounding box coordinates are normalized (0-1). To get pixel coordinates, multiply by image width/height: `pixel_x1 = bbox[0] * image_width`.

**VRAM:** Magiv2 uses ~3-4GB VRAM on a T4. Must be fully unloaded before loading manga-ocr.

**Unloading pattern:**
```python
del model
torch.cuda.empty_cache()
import gc; gc.collect()
```

---

### 2. manga-ocr — Japanese OCR

**Repository:** https://github.com/kha-white/manga-ocr
**HuggingFace model ID:** `kha-white/manga-ocr-base`
**Install:** `pip install manga-ocr`

**How to use:**
```python
from manga_ocr import MangaOcr
mocr = MangaOcr()  # downloads model on first run, auto-uses GPU if available
text = mocr(pil_image_crop)  # pass a PIL Image of the text region crop
```

**Integration with Magi:**
1. For each page, take Magi's `"texts"` bounding boxes (normalized coordinates)
2. Convert to pixel coordinates
3. Crop that region from the original full-resolution page image
4. Pass each crop to `mocr()` to get the Japanese text string
5. Store the results in the same order as Magi's text box list

**Capabilities:** Handles vertical/horizontal Japanese text, furigana, stylized manga fonts, multi-line text in a single bubble — all in one forward pass per crop. No line segmentation needed.

**VRAM:** ~1-2GB. Must unload after processing all crops.

---

### 3. SFX Handling

**Classification:** A text box is an SFX candidate if:
- `is_essential_text[i]` is `False` (from Magi), OR
- OCR confidence is very low (heuristic: very short string like 1-2 chars, or mostly katakana — SFX in manga is typically katakana)

**Rendering approach (Viz Media style):**
- Do NOT inpaint or erase the original SFX art
- Render a small, semi-transparent subtitle label near the original SFX location
- Use a smaller font size than dialogue, italic style
- Position: offset below or beside the original text box (avoid overlapping art)

---

### 4. Rolling JSON Context Memory

**Purpose:** Japanese is a pro-drop language (subjects are frequently omitted). Without cross-page context, the LLM will mistranslate pronouns and speakers. The rolling context solves this.

**Schema:**
```json
{
  "chapter_info": {
    "title": "string or null",
    "genre": "string or null",
    "user_context": "free-text string from user input"
  },
  "characters_seen": {
    "character_name": {
      "first_appeared_page": 1,
      "last_seen_page": 5,
      "cluster_id": "int from Magi",
      "description": "built from context over time"
    }
  },
  "recent_dialogue": [
    {
      "page": 3,
      "speaker": "character_name or 'narration' or 'unsure'",
      "japanese": "original OCR text",
      "english": "translated text",
      "is_sfx": false
    }
  ],
  "scene_summary": "2-3 sentence summary of what's happening, updated every N pages by the LLM"
}
```

**Sliding window strategy:**
- `recent_dialogue` keeps the last **6 pages** of dialogue entries
- Older entries are dropped (not summarized — keep it simple)
- `scene_summary` is updated by asking the LLM to summarize every 5 pages — this compressed representation carries forward the gist without blowing up token count
- `characters_seen` is cumulative (never dropped)
- Target: keep the full context injection under ~2000 tokens per translation call

**Update flow:** After each page is translated, append its dialogue to `recent_dialogue`, update `characters_seen`, and conditionally update `scene_summary`.

---

### 5. User Context Injection

A simple text file (`user_context.txt`) the user places in the input folder, or a text field in the Gradio UI. Contents are injected verbatim into the `chapter_info.user_context` field of the rolling memory.

Examples:
- "Character A is secretly the villain"
- "This is a shoujo manga, so dialogue should be emotionally expressive"
- "The characters are in medieval Japan, use period-appropriate English"

---

### 6. Translation LLM — API Wrapper

**Primary model:** Gemini 2.5 Flash (or whatever is cheapest with structured output support at build time — verify current pricing).
**Fallback:** GPT-4o-mini or Claude Haiku.

**Provider-swappable wrapper design:**
Create an abstract base class `TranslationProvider` with a single method `translate(prompt, image=None) -> TranslationResult`. Implement `GeminiProvider`, `OpenAIProvider`, etc. The pipeline calls the provider through the interface, never directly.

**Translation prompt structure (per page):**
```
You are an expert Japanese-to-English manga translator. Translate the following manga page dialogue.

CONTEXT:
{rolling_json_context}

THIS PAGE'S DIALOGUE (in reading order):
[
  {"id": 0, "speaker": "Luffy", "japanese": "おれは海賊王になる！", "is_sfx": false},
  {"id": 1, "speaker": "narration", "japanese": "その日、少年の運命は変わった。", "is_sfx": false},
  {"id": 2, "speaker": null, "japanese": "ドーン", "is_sfx": true}
]

INSTRUCTIONS:
- Translate naturally, not literally. Manga dialogue should feel alive.
- Preserve each character's speech style and personality.
- For SFX (is_sfx=true), provide a short English equivalent.
- Use the speaker attribution to resolve pronouns — Japanese often drops subjects.
- Return ONLY the JSON response below, no other text.
```

**Structured output schema (enforce via Gemini's `response_schema` or function calling):**
```json
{
  "translations": [
    {
      "id": 0,
      "english": "I'm gonna be King of the Pirates!",
      "translator_note": "optional — flag if uncertain"
    }
  ],
  "scene_summary_update": "Luffy declares his dream as the narrator foreshadows destiny.",
  "confidence": 0.92
}
```

**The `id` field maps 1:1 back to Magi's text box index**, which maps back to the bounding box for typesetting. This is the critical linkage.

**Vision mode (optional toggle):**
When OCR confidence is low or user enables it:
- Attach the cropped panel image to the API call
- Modify the prompt to say: "I've attached the panel image. Use it to verify/correct the OCR if the text seems wrong."
- This reuses the same LLM API's multimodal capability — no separate VLM needed

**Retry logic:**
- Exponential backoff: 1s, 2s, 4s, 8s, max 3 retries
- On schema validation failure: retry with a "your previous response was malformed, return valid JSON" appended
- On provider failure: optionally fall back to secondary provider

---

### 7. Typesetting Engine (Pillow-based)

**For dialogue (is_essential_text=True):**
1. Get the bounding box from Magi (convert normalized to pixel coords)
2. Inpaint the original text: fill the bounding box region with white (simple approach — no need for complex inpainting since manga bubbles are white)
3. Dynamic font-fit algorithm:
   - Start at a maximum font size (e.g., 24px)
   - Use `PIL.ImageFont.truetype()` with a manga-appropriate font (use CC-licensed "Anime Ace" or "Wild Words" or "Komika" font — include the .ttf in the repo)
   - Use `draw.textbbox()` to measure the rendered text size
   - Word-wrap the text using `textwrap.wrap()` to fit the box width
   - Shrink font size in a loop until wrapped text fits within the bounding box height
   - Center-align horizontally and vertically within the box
   - Render with `draw.text()`

**For SFX (is_sfx=True):**
1. Do NOT erase the original art
2. Render a small italic label offset below or beside the original text box
3. Use a semi-transparent background rectangle behind the subtitle text for readability
4. Smaller font, muted color (e.g., gray on semi-transparent white bg)

**Font:** Include a free manga-style font in the repo. "Anime Ace 2.0" (Blambot, free for non-commercial) or "CC Wild Words" are good choices. Document the license in README.

---

### 8. Reading Order

Manga reads right-to-left, top-to-bottom. Magi provides panel bounding boxes — use them to establish reading order:

1. **Panel ordering:** Sort panels right-to-left, top-to-bottom. Use a simple heuristic: sort by column (right-to-left) then by row (top-to-bottom). Specifically, divide the page into vertical columns based on panel x-coordinates, then sort within each column by y-coordinate.
2. **Text box ordering within panels:** For each panel, find which text boxes fall inside it (check bbox overlap). Sort those text boxes right-to-left, top-to-bottom within the panel.
3. **Unassociated text boxes:** Any text boxes not inside a panel (e.g., narration boxes at page edges) should be placed at the beginning or end of the page's dialogue.

This ordered list is what gets sent to the LLM for translation, ensuring the conversation flows correctly.

---

### 9. Quality Report

Generate a `quality_report.json` after processing each chapter:
```json
{
  "chapter": "chapter_name",
  "total_pages": 20,
  "total_text_boxes": 156,
  "total_sfx": 23,
  "per_page": [
    {
      "page": 1,
      "num_panels": 5,
      "num_text_boxes": 8,
      "num_characters_detected": 3,
      "flags": [
        {"text_box_id": 3, "issue": "low_ocr_confidence", "ocr_text": "??"},
        {"text_box_id": 5, "issue": "no_speaker_attribution"}
      ],
      "vision_check_recommended": false
    }
  ],
  "pipeline_metadata": {
    "magi_version": "v2",
    "ocr_model": "manga-ocr",
    "llm_provider": "gemini-2.5-flash",
    "total_api_cost_estimate_usd": 0.03,
    "processing_time_seconds": 245
  }
}
```

---

## Project Structure

```
manga-translation-pipeline/
├── README.md
├── requirements.txt
├── pyproject.toml
├── LICENSE
│
├── config/
│   ├── default_config.yaml
│   └── .env.example
│
├── assets/
│   └── fonts/
│       └── manga_font.ttf
│
├── src/
│   ├── __init__.py
│   │
│   ├── pipeline.py                    # Main orchestrator — runs the full stage-batched pipeline
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   └── magi_detector.py           # Loads Magiv2, runs detection, returns structured results, handles unloading
│   │
│   ├── ocr/
│   │   ├── __init__.py
│   │   └── manga_ocr_engine.py        # Loads manga-ocr, processes text crops, handles unloading
│   │
│   ├── translation/
│   │   ├── __init__.py
│   │   ├── base_provider.py           # Abstract TranslationProvider base class
│   │   ├── gemini_provider.py         # Gemini API implementation
│   │   ├── openai_provider.py         # OpenAI API implementation (fallback)
│   │   ├── prompt_builder.py          # Constructs translation prompts from context + OCR
│   │   └── schemas.py                 # Pydantic models for translation request/response
│   │
│   ├── context/
│   │   ├── __init__.py
│   │   └── memory_manager.py          # Rolling JSON context: update, window, serialize
│   │
│   ├── typesetting/
│   │   ├── __init__.py
│   │   ├── renderer.py                # Pillow-based text rendering, font-fit, SFX subtitles
│   │   └── reading_order.py           # Panel + text box ordering (right-to-left, top-to-bottom)
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   └── exporter.py                # PNG, PDF (via Pillow/reportlab), CBZ (zipfile) export
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py             # Image loading, cropping, bbox conversion helpers
│       ├── gpu_utils.py               # Model loading/unloading, VRAM management, context managers
│       └── quality_report.py          # Generate quality_report.json
│
├── notebooks/
│   └── manga_translation_colab.ipynb  # Main Colab notebook — thin wrapper that calls src/ modules
│
├── tests/
│   ├── test_reading_order.py
│   ├── test_context_memory.py
│   ├── test_typesetting.py
│   └── test_prompt_builder.py
│
├── evaluation/
│   ├── eval_ocr.py                    # OCR accuracy on Manga109
│   ├── eval_detection.py              # Detection metrics on PopManga
│   └── eval_translation.py            # BLEU/COMET on small test set
│
└── examples/
    ├── input/
    │   ├── pages/
    │   ├── character_bank/
    │   └── user_context.txt
    └── output/
```

---

## Colab Notebook Design

The Colab notebook (`manga_translation_colab.ipynb`) should be a **thin orchestration layer**, not contain business logic. Structure:

**Cell 1: Setup & Install**
- Clone the repo, pip install requirements
- Mount Google Drive for persistent storage of results

**Cell 2: Configuration**
- API key input (use Colab's `userdata` secrets, not hardcoded)
- Path to input images
- Optional: character bank path, user context
- Pipeline settings (vision mode toggle, context window size, etc.)

**Cell 3: Run Pipeline**
- Single call to `pipeline.run()` with the config
- Progress bars via tqdm
- Displays results inline

**Cell 4: Review & Export**
- Display translated pages side-by-side with originals
- Show quality report highlights
- Download links for PDF/CBZ

**Cell 5 (optional): Evaluation**
- Run benchmarks on test datasets

---

## Gradio UI (Colab-hosted)

Build a simple Gradio interface that runs inside the Colab notebook. This transforms the project from "run my notebook" to "try my app":

**Interface layout:**
- **Left panel:** Upload manga pages (multi-file upload), optional character bank upload, text area for user context
- **Center controls:** "Translate Chapter" button, toggles (vision mode, context window size slider)
- **Right panel:** Side-by-side original/translated image viewer (gallery component), download buttons for PDF/CBZ
- **Bottom:** Quality report summary, processing log

Use `gr.Blocks()` for the layout. The interface calls the same `pipeline.run()` function.

---

## Datasets & Evaluation

### Datasets for Testing/Benchmarking

| Dataset | What It Is | Where to Get It | Use Case |
|---------|-----------|-----------------|----------|
| **Manga109** | 109 manga volumes with annotations. Standard academic Japanese manga dataset. What manga-ocr was trained on. | http://www.manga109.org/en/download.html (free, requires academic registration form) | OCR accuracy benchmarking, detection evaluation |
| **PopMangaX (Test)** | Magi's own evaluation benchmark. Annotated with panels, text boxes, characters, speech-bubble tails, dialogue vs SFX labels. English manga from MangaPlus. | https://huggingface.co/datasets/ragavsachdeva/popmanga_test | Speaker diarization accuracy evaluation |
| **PopCharacters** | Character crop images for Magi's character bank feature. | https://huggingface.co/datasets/ragavsachdeva/popcharacters | Testing character bank / named speaker attribution |
| **Roboflow manga speech-bubble** | ~8,500 annotated manga pages with speech bubble bounding boxes. | Search Roboflow for "manga speech bubble" | Fallback if custom detector fine-tuning ever needed |
| **MangaDex bulk download** | Tool to download large-scale manga from MangaDex. | https://github.com/EMACC99/mangadex | Gathering test data (respect copyright) |

### For Public Portfolio Demo
Use **public-domain or Creative Commons licensed** manga/webcomics for any images shown on GitHub or in the demo:
- **Pepper & Carrot** (CC-BY, David Revoy) — webcomic available in multiple languages including Japanese
- **MangaPlus free chapters** (Shueisha's official free platform) — the PopMangaX test images come from here
- Or create a small synthetic test set

### Evaluation Metrics

Implement at minimum:
1. **OCR Character Error Rate (CER):** Compare manga-ocr output against ground truth on a Manga109 test subset. Use `jiwer` or `editdistance` library.
2. **Detection mAP:** Run Magi on PopMangaX test set, compute mean Average Precision for panels, text boxes, characters using standard COCO-style evaluation.
3. **Speaker Attribution Accuracy:** On PopMangaX, compare Magi's text-to-character associations against ground truth. Report accuracy as % of correctly attributed text boxes.
4. **Translation Quality (small scale):** Take 5-10 pages where you know both the Japanese and a professional English translation (e.g., official Viz releases). Compare your pipeline's output using BLEU score (`sacrebleu` library) and optionally COMET (`unbabel-comet`). Include a comparison: your pipeline (with context) vs. raw LLM translation (without context) — this demonstrates the value of your architecture.

---

## Pydantic Schemas (define in `src/translation/schemas.py`)

Use Pydantic v2 models for all structured data flowing through the pipeline. This enforces type safety and makes the architecture legible. Key models to define:

- `BoundingBox`: with `x1, y1, x2, y2` (float, normalized 0-1) + helper method `to_pixels(width, height)`
- `TextBox`: bbox, ocr_text, is_essential, speaker_name, speaker_confidence, ocr_confidence
- `PageDetection`: page_index, panels list, text_boxes list, characters list, text_character_associations
- `TranslationRequest`: text_boxes (with Japanese), context_json, page_image_path (optional for vision)
- `TranslationResponse`: translations list (id, english, note), scene_summary_update, confidence
- `PageResult`: original_image_path, translated_image, detection, translation, quality_flags
- `QualityReport`: full chapter quality report
- `RollingContext`: the context memory schema

---

## API Cost Estimation

For a typical 20-page manga chapter:
- ~150 text boxes x ~50 chars avg OCR = ~7,500 input chars Japanese
- Context window: ~2,000 tokens
- Per-page prompt: ~500-800 tokens input, ~200-400 tokens output
- Total per chapter: ~15,000-20,000 input tokens + ~5,000-8,000 output tokens
- At Gemini 2.5 Flash rates (~$0.15/M input, ~$0.60/M output as of mid-2025): **~$0.01-0.02 per chapter**
- Vision mode adds ~$0.01-0.02 per image attachment

**This is negligible.** Include this cost estimate in the README.

---

## README Structure

The project README should include:
1. **Hero image:** Side-by-side Japanese original and English translated page
2. **Architecture diagram** (Mermaid): Input -> Magi -> OCR -> Context -> LLM -> Typeset -> Output
3. **Features list** with badges
4. **Quick start** (3 commands: clone, install, run)
5. **How it works** — brief explanation of each stage
6. **Evaluation results** — table with metrics
7. **Cost analysis** — show it's essentially free to run
8. **Limitations & future work**
9. **Citation** section (cite Magi papers, manga-ocr)
10. **License**

---

## Implementation Order (recommended build sequence)

Build in this order so you have a working pipeline ASAP, then iterate:

1. **`utils/image_utils.py` + `utils/gpu_utils.py`** — Image loading helpers and GPU context managers. Foundation for everything.
2. **`detection/magi_detector.py`** — Get Magi running, verify output format on a test image. This is the most critical component.
3. **`ocr/manga_ocr_engine.py`** — Crop text boxes from Magi's output, run manga-ocr. Verify Japanese text output.
4. **`typesetting/reading_order.py`** — Implement panel/text ordering. Test with Magi's output.
5. **`translation/schemas.py`** — Define all Pydantic models.
6. **`translation/base_provider.py` + `gemini_provider.py`** — Get LLM translation working with hardcoded test input.
7. **`translation/prompt_builder.py`** — Build the prompt assembly logic.
8. **`context/memory_manager.py`** — Rolling context with sliding window.
9. **`typesetting/renderer.py`** — Pillow rendering. Test with translated text on a real page.
10. **`export/exporter.py`** — PDF + CBZ generation.
11. **`pipeline.py`** — Wire everything together.
12. **`utils/quality_report.py`** — Generate quality JSON.
13. **Colab notebook** — Thin wrapper.
14. **Gradio UI** — Interactive demo.
15. **Evaluation scripts** — Benchmarking.
16. **README** — Documentation with architecture diagram and results.

---

## Critical Technical Notes

1. **Magi bbox coordinates are normalized (0-1).** Always multiply by image dimensions before cropping or rendering.

2. **manga-ocr expects a PIL Image**, not a numpy array. Convert crops: `Image.fromarray(crop_array)`.

3. **Magi loads custom model code** — always use `trust_remote_code=True`.

4. **Font licensing matters for portfolio.** Only use fonts with licenses that allow redistribution. Document the license.

5. **API keys:** Never commit API keys. Use Colab's built-in secrets (`google.colab.userdata.get('GEMINI_API_KEY')`) or `.env` files with `python-dotenv`.

6. **Gemini structured output:** Use `response_mime_type="application/json"` and `response_schema` parameter in the `generate_content` call. This enforces JSON output without relying on prompt-only instructions.

7. **The `is_essential_text` field from Magi is your SFX classifier.** This is the primary signal — don't build a separate classifier unless Magi's classification proves insufficient.

8. **Character bank is optional but highly recommended.** Without it, Magi can still detect and cluster characters, but won't be able to name them. The speaker attribution in the rolling context will show cluster IDs instead of names.

9. **Colab session management:** Save intermediate results to Google Drive after each stage. If a session dies mid-pipeline, you can resume from the last completed stage. Implement checkpoint logic in the pipeline orchestrator.

10. **Type hints everywhere.** Use Python 3.10+ type hint syntax. This is a portfolio project — code readability matters as much as functionality.

---

## Dependencies (requirements.txt)

```
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
manga-ocr>=0.1.11
Pillow>=10.0.0
numpy>=1.24.0
pydantic>=2.0.0
google-generativeai>=0.5.0
python-dotenv>=1.0.0
tqdm>=4.65.0
gradio>=4.0.0
pyyaml>=6.0
reportlab>=4.0
sacrebleu>=2.3.0
jiwer>=3.0.0
```

---

## What Success Looks Like

When a technical reviewer looks at this project, they should see:
1. **Clean architecture** — not a single monolithic notebook, but a well-structured Python package
2. **Smart model orchestration** — explicit VRAM management, stage batching, model lifecycle
3. **Novel speaker diarization** — using vision (Magi) to solve a linguistic problem (Japanese pro-drop)
4. **Context-aware translation** — rolling memory that demonstrably improves translation quality
5. **Quantitative evaluation** — real numbers, not just "it works"
6. **Interactive demo** — Gradio UI that lets anyone try it
7. **Cost-efficient design** — runs on free Colab with $0.02/chapter API costs
8. **Production-quality code** — type hints, docstrings, Pydantic schemas, error handling, retry logic
