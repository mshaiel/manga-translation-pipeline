"""Gradio-based interactive application for the Manga Translation Pipeline.

Provides a browser interface for uploading manga pages, configuring story context,
triggering stage-batched translation, previewing side-by-side results, and downloading
compiled PDF and CBZ files.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_ui(pipeline_instance: Any | None = None) -> Any:
    """Construct the Gradio interface Blocks application.

    Args:
        pipeline_instance: Optional existing MangaTranslationPipeline instance.

    Returns:
        gr.Blocks instance ready to `.launch()`.
    """
    try:
        import gradio as gr
    except ImportError as err:
        raise ImportError(
            "Gradio is not installed. Please install it using `pip install gradio>=4.0.0`."
        ) from err

    from src.pipeline import MangaTranslationPipeline

    def handle_translation(
        page_files: list[Any] | None,
        character_crop_files: list[Any] | None,
        user_context_text: str,
        api_key_input: str,
        sliding_window: int,
        mock_mode_toggle: bool,
        progress: Any = gr.Progress(track_tqdm=True),  # noqa: B008
    ) -> tuple[list[Any], str | None, str | None, dict[str, Any], str]:
        """Process user uploads through the stage-batched pipeline."""
        if not page_files:
            return [], None, None, {}, "⚠️ Error: Please upload at least one manga page."

        # Setup temporary execution directory
        temp_dir = Path(tempfile.mkdtemp(prefix="manga_trans_ui_"))
        pages_input_dir = temp_dir / "input_pages"
        pages_input_dir.mkdir(parents=True, exist_ok=True)

        input_paths: list[Path] = []
        for idx, file_obj in enumerate(page_files):
            source_path = Path(file_obj.name if hasattr(file_obj, "name") else file_obj)
            ext = source_path.suffix or ".png"
            dest_path = pages_input_dir / f"page_{idx+1:03d}{ext}"
            dest_path.write_bytes(source_path.read_bytes())
            input_paths.append(dest_path)

        # Character bank processing
        character_bank: dict[str, Any] | None = None
        if character_crop_files:
            bank_images: list[Any] = []
            bank_names: list[str] = []
            for crop_obj in character_crop_files:
                crop_path = Path(crop_obj.name if hasattr(crop_obj, "name") else crop_obj)
                name = crop_path.stem
                bank_images.append(str(crop_path))
                bank_names.append(name)
            character_bank = {"images": bank_images, "names": bank_names}

        # Override or set API key if user typed one in
        if api_key_input.strip():
            os.environ["GEMINI_API_KEY"] = api_key_input.strip()

        # Build pipeline
        pipeline = pipeline_instance or MangaTranslationPipeline(
            config={
                "pipeline": {
                    "output_dir": str(temp_dir / "output"),
                    "checkpoints_dir": str(temp_dir / "checkpoints"),
                    "enable_checkpoints": True,
                },
                "context": {
                    "sliding_window_pages": int(sliding_window),
                },
                "mock_models": mock_mode_toggle,
            }
        )

        progress(0.1, desc="Starting Manga Translation Pipeline...")

        try:
            results = pipeline.run(
                chapter_pages=input_paths,
                chapter_name="web_chapter",
                character_bank=character_bank,
                user_context=user_context_text.strip() or None,
                resume_from_checkpoints=False,
            )

            gallery_items: list[Any] = []
            for orig_path, trans_img in zip(input_paths, results["translated_images"], strict=False):
                gallery_items.append(str(orig_path))
                gallery_items.append(trans_img)

            pdf_path_str = str(results["pdf_path"]) if results["pdf_path"].exists() else None
            cbz_path_str = str(results["cbz_path"]) if results["cbz_path"].exists() else None
            quality_data = results["quality_report"].model_dump()

            status_msg = (
                f"✅ Successfully translated {results['quality_report'].total_pages} pages! "
                f"Estimated API cost: ${results['quality_report'].pipeline_metadata.total_api_cost_estimate_usd:.4f}"
            )

            return gallery_items, pdf_path_str, cbz_path_str, quality_data, status_msg

        except Exception as exc:
            logger.exception("Pipeline execution failed in Gradio UI")
            return [], None, None, {}, f"❌ Error during translation: {exc}"

    with gr.Blocks(title="AI Manga Translation Pipeline") as demo:
        gr.Markdown(
            """
            # 📖 AI-Powered Manga Translation Pipeline
            ### High-Fidelity Japanese Manga Translation with Visual Speaker Diarization
            Combines **Magi v2** (detection & character diarization), **manga-ocr** (Japanese OCR),
            **Rolling Context Memory** (6-page sliding window), and **Gemini 2.5 Flash** structured translation.
            """
        )

        with gr.Row():
            # Left Column: Inputs & Story Context
            with gr.Column(scale=1):
                gr.Markdown("### 1. Upload Manga Pages & Context")
                page_upload = gr.File(
                    label="Manga Page Images (Multi-page)",
                    file_count="multiple",
                    file_types=["image"],
                )
                char_upload = gr.File(
                    label="Optional: Character Bank (Face crops with character name as filename)",
                    file_count="multiple",
                    file_types=["image"],
                )
                user_context_box = gr.Textbox(
                    label="Story & Narrative Context (user_context.txt)",
                    placeholder="e.g., Setting is Edo period Japan. Character A is secretly the antagonist...",
                    lines=3,
                )
                api_key_box = gr.Textbox(
                    label="Gemini API Key (Optional if already configured in environment)",
                    type="password",
                    placeholder="AIzaSy...",
                )

                with gr.Accordion("⚙️ Advanced Pipeline Settings", open=False):
                    window_slider = gr.Slider(
                        minimum=2,
                        maximum=10,
                        value=6,
                        step=1,
                        label="Rolling Context Dialogue Window (Pages)",
                    )
                    mock_toggle = gr.Checkbox(
                        label="Dry Run / Mock Mode (Test UI without downloading weights)",
                        value=False,
                    )

                translate_btn = gr.Button("🚀 Translate Chapter", variant="primary", size="lg")

            # Right Column: Visual Gallery & Downloads
            with gr.Column(scale=2):
                gr.Markdown("### 2. Translated Chapter Preview (Original vs. English)")
                status_box = gr.Markdown("Upload pages and click **Translate Chapter** to begin.")

                results_gallery = gr.Gallery(
                    label="Side-by-Side Chapter Pages",
                    columns=2,
                    height=500,
                    object_fit="contain",
                )

                with gr.Row():
                    pdf_download = gr.File(label="📥 Download Chapter PDF")
                    cbz_download = gr.File(label="📦 Download Comic Archive (.cbz)")

        # Bottom Section: Quality Audit Report
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 📊 Chapter Quality & Cost Audit (quality_report.json)")
                quality_json = gr.JSON(label="Pipeline Quality Audit")

        # Wire click event
        translate_btn.click(
            fn=handle_translation,
            inputs=[
                page_upload,
                char_upload,
                user_context_box,
                api_key_box,
                window_slider,
                mock_toggle,
            ],
            outputs=[
                results_gallery,
                pdf_download,
                cbz_download,
                quality_json,
                status_box,
            ],
        )

    return demo


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False)
