"""Unit tests for Gradio UI construction."""

import pytest


def test_create_ui():
    """Verify that create_ui constructs a valid Gradio Blocks instance."""
    try:
        import gradio as gr
    except ImportError:
        pytest.skip("Gradio is not installed in local environment (runs primarily in Colab).")

    from src.ui.app import create_ui

    app = create_ui()
    assert isinstance(app, gr.Blocks)
    assert "AI Manga Translation Pipeline" in app.title
