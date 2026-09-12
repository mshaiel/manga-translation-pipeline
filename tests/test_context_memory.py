"""Unit tests for RollingContextManager in src/context/memory_manager.py."""

from src.context.memory_manager import RollingContextManager
from src.translation.schemas import CharacterBox, PageDetection, TextBox, TranslationItem


class TestRollingContextManager:
    """Test suite for sliding window context memory and character registry."""

    def test_init_state(self):
        manager = RollingContextManager(sliding_window_pages=6, scene_summary_interval=5)
        ctx = manager.get_context()
        assert ctx.chapter_info.title is None
        assert len(ctx.characters_seen) == 0
        assert len(ctx.recent_dialogue) == 0
        assert ctx.scene_summary == ""

    def test_set_chapter_info_and_user_context(self, tmp_path):
        manager = RollingContextManager()
        manager.set_chapter_info(title="Chainsaw Man", genre="Dark Fantasy")

        ctx_file = tmp_path / "user_context.txt"
        ctx_file.write_text("Denji is trying to live a normal life.", encoding="utf-8")
        manager.load_user_context_from_file(ctx_file)

        ctx = manager.get_context()
        assert ctx.chapter_info.title == "Chainsaw Man"
        assert ctx.chapter_info.genre == "Dark Fantasy"
        assert ctx.chapter_info.user_context == "Denji is trying to live a normal life."

    def test_register_page_characters_cumulative(self):
        manager = RollingContextManager()

        # Page 0 (Page 1 in 1-index): Char 1 (Luffy) and Char 2 (Zoro)
        det_p1 = PageDetection(
            page_index=0,
            characters=[
                CharacterBox(id=0, bbox=[0.1, 0.1, 0.4, 0.4], name="Luffy", cluster_id=1),
                CharacterBox(id=1, bbox=[0.5, 0.1, 0.8, 0.4], name="Zoro", cluster_id=2),
            ],
        )
        manager.register_page_characters(0, det_p1)

        ctx = manager.get_context()
        assert "Luffy" in ctx.characters_seen
        assert ctx.characters_seen["Luffy"].first_appeared_page == 1
        assert ctx.characters_seen["Luffy"].last_seen_page == 1
        assert "Zoro" in ctx.characters_seen

        # Page 2 (Page 3 in 1-index): Luffy seen again, Nami appears for the first time
        det_p3 = PageDetection(
            page_index=2,
            characters=[
                CharacterBox(id=0, bbox=[0.1, 0.1, 0.4, 0.4], name="Luffy", cluster_id=1),
                CharacterBox(id=1, bbox=[0.5, 0.1, 0.8, 0.4], name="Nami", cluster_id=3),
            ],
        )
        manager.register_page_characters(2, det_p3)

        ctx_updated = manager.get_context()
        # Luffy's first appearance remains 1, last seen updated to 3
        assert ctx_updated.characters_seen["Luffy"].first_appeared_page == 1
        assert ctx_updated.characters_seen["Luffy"].last_seen_page == 3
        # Zoro is still retained (cumulative!)
        assert "Zoro" in ctx_updated.characters_seen
        assert "Nami" in ctx_updated.characters_seen
        assert ctx_updated.characters_seen["Nami"].first_appeared_page == 3

    def test_sliding_window_pruning(self):
        # Sliding window of 3 pages for testing
        manager = RollingContextManager(sliding_window_pages=3)

        # Append pages 1, 2, 3, 4, 5 sequentially
        for p in range(5):
            tb = [TextBox(id=0, bbox=[0.1, 0.1, 0.4, 0.4], ocr_text=f"Page {p+1} text", speaker_name="Hero")]
            tr = [TranslationItem(id=0, english=f"Page {p+1} translation")]
            manager.append_translated_page(p, tb, tr)

        ctx = manager.get_context()
        # Should only keep the last 3 pages: pages 3, 4, 5
        retained_pages = {d.page for d in ctx.recent_dialogue}
        assert retained_pages == {3, 4, 5}
        assert 1 not in retained_pages
        assert 2 not in retained_pages

    def test_scene_summary_update_interval(self):
        manager = RollingContextManager(scene_summary_interval=5)
        assert manager.should_request_scene_summary(0) is False  # Page 1
        assert manager.should_request_scene_summary(3) is False  # Page 4
        assert manager.should_request_scene_summary(4) is True   # Page 5!
        assert manager.should_request_scene_summary(9) is True   # Page 10!

    def test_token_estimation_and_serialization(self):
        manager = RollingContextManager()
        manager.set_chapter_info(title="Bleach", user_context="Shinigami battle")
        json_str = manager.to_json()
        assert "Bleach" in json_str
        assert manager.estimate_token_count() > 0
