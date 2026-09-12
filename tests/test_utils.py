"""Unit tests for utility functions in src/utils/."""

import numpy as np
import pytest
from PIL import Image

from src.utils.gpu_utils import (
    clear_gpu_memory,
    get_device,
    get_vram_usage_mb,
    is_cuda_available,
    managed_gpu_memory,
)
from src.utils.image_utils import (
    bbox_normalized_to_pixel,
    bbox_pixel_to_normalized,
    crop_image,
    load_image_for_magi,
    load_pil_image,
)


class TestImageUtils:
    """Tests for image loading, coordinate conversion, and cropping."""

    @pytest.fixture
    def sample_image(self, tmp_path):
        """Creates a dummy synthetic test image (100x200)."""
        img_path = tmp_path / "test_manga_page.png"
        img = Image.new("RGB", (100, 200), color=(255, 255, 255))
        img.save(img_path)
        return img_path

    def test_load_pil_image(self, sample_image):
        pil_img = load_pil_image(sample_image)
        assert isinstance(pil_img, Image.Image)
        assert pil_img.size == (100, 200)
        assert pil_img.mode == "RGB"

    def test_load_image_for_magi(self, sample_image):
        arr = load_image_for_magi(sample_image)
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (200, 100, 3)
        assert arr.dtype == np.uint8

    def test_bbox_normalized_to_pixel(self):
        # 100 width, 200 height
        px1, py1, px2, py2 = bbox_normalized_to_pixel([0.1, 0.2, 0.5, 0.8], width=100, height=200)
        assert (px1, py1, px2, py2) == (10, 40, 50, 160)

    def test_bbox_normalized_to_pixel_clamping(self):
        px1, py1, px2, py2 = bbox_normalized_to_pixel([-0.2, 0.1, 1.5, 0.9], width=100, height=200)
        assert px1 == 0
        assert px2 == 100

    def test_bbox_pixel_to_normalized(self):
        nx1, ny1, nx2, ny2 = bbox_pixel_to_normalized([10, 40, 50, 160], width=100, height=200)
        assert nx1 == pytest.approx(0.1)
        assert ny1 == pytest.approx(0.2)
        assert nx2 == pytest.approx(0.5)
        assert ny2 == pytest.approx(0.8)

    def test_crop_image_normalized(self, sample_image):
        img = load_pil_image(sample_image)
        crop = crop_image(img, [0.1, 0.2, 0.5, 0.8], normalized=True)
        assert crop.size == (40, 120)

    def test_crop_image_zero_dimension_protection(self, sample_image):
        img = load_pil_image(sample_image)
        # Bbox with zero width/height should be safeguarded to at least 1px
        crop = crop_image(img, [0.5, 0.5, 0.5, 0.5], normalized=True)
        assert crop.size[0] >= 1
        assert crop.size[1] >= 1


class TestGpuUtils:
    """Tests for GPU resource manager and fallback behavior."""

    def test_get_device(self):
        device = get_device("cuda")
        if is_cuda_available():
            assert device.startswith("cuda")
        else:
            assert device == "cpu"

    def test_get_vram_usage(self):
        vram = get_vram_usage_mb()
        assert "allocated_mb" in vram
        assert "reserved_mb" in vram
        assert isinstance(vram["allocated_mb"], float)

    def test_clear_gpu_memory(self):
        # Should execute safely on both CPU and CUDA environments without errors
        clear_gpu_memory()

    def test_managed_gpu_memory_context(self):
        with managed_gpu_memory("TestStage"):
            x = list(range(1000))
            assert len(x) == 1000
