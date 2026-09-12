"""GPU and VRAM resource management utilities.

Provides explicit lifecycle helpers, memory measurement, and context managers
for strict model loading and unloading across stage-batched pipeline steps.
"""

from __future__ import annotations

import contextlib
import gc
import logging
from collections.abc import Generator
from typing import Any

logger = logging.getLogger(__name__)

# Attempt to import torch safely; allows running CPU / schema code without torch installed locally
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


def is_cuda_available() -> bool:
    """Check if PyTorch is installed and a CUDA device is accessible."""
    if not TORCH_AVAILABLE or torch is None:
        return False
    return bool(torch.cuda.is_available())


def get_device(preferred: str = "cuda") -> str:
    """Resolve the computing device ('cuda' or 'cpu').

    Args:
        preferred: Requested device ('cuda' or 'cpu').

    Returns:
        'cuda' if CUDA is accessible and preferred, otherwise 'cpu'.
    """
    if preferred.startswith("cuda") and is_cuda_available():
        return preferred
    return "cpu"


def get_vram_usage_mb() -> dict[str, float]:
    """Retrieve current GPU VRAM allocation and reservation metrics in Megabytes.

    Returns:
        Dictionary with keys 'allocated_mb', 'reserved_mb', 'max_allocated_mb'.
        Returns 0.0 values if running on CPU.
    """
    if not is_cuda_available():
        return {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0}

    assert torch is not None
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)

    return {
        "allocated_mb": round(allocated, 2),
        "reserved_mb": round(reserved, 2),
        "max_allocated_mb": round(max_allocated, 2),
    }


def clear_gpu_memory(*models_to_delete: Any) -> None:
    """Unload model references, run Python garbage collection, and empty CUDA cache.

    Implements the recommended stage-batched unloading pattern:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    Args:
        *models_to_delete: Optional references to deep learning models or heavy tensors.
    """
    # Delete explicit references
    for model in models_to_delete:
        del model

    # Trigger garbage collection
    gc.collect()

    # Flush PyTorch CUDA cache if available
    if is_cuda_available() and torch is not None:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        vram = get_vram_usage_mb()
        logger.debug("GPU cache flushed. Current VRAM: %s", vram)


@contextlib.contextmanager
def managed_gpu_memory(stage_name: str = "PipelineStage") -> Generator[None, None, None]:
    """Context manager for stage-batched GPU execution.

    Records VRAM usage before entry and guarantees garbage collection and
    CUDA cache clearing upon exit, even if exceptions occur.

    Args:
        stage_name: Human-readable identifier for log output.

    Yields:
        None
    """
    initial_vram = get_vram_usage_mb()
    logger.info("Entering [%s] - Initial VRAM: %s", stage_name, initial_vram)
    try:
        yield
    finally:
        clear_gpu_memory()
        final_vram = get_vram_usage_mb()
        logger.info(
            "Exited [%s] - Final VRAM: %s (Allocated Delta: %.2f MB)",
            stage_name,
            final_vram,
            final_vram["allocated_mb"] - initial_vram["allocated_mb"],
        )
