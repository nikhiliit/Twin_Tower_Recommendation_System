"""Automatic device detection and environment configuration.

Detects the best available hardware accelerator (NVIDIA CUDA,
Apple MPS, or CPU) and configures the environment accordingly,
including OpenMP fixes for macOS.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    """Container for detected device information.

    Attributes:
        device: ``torch.device`` to use.
        device_type: String identifier (``'cuda'``, ``'mps'``,
            ``'cpu'``).
        device_name: Human-readable name (e.g. GPU model).
        gpu_memory_gb: Total GPU memory in GB (0.0 for CPU).
        n_gpus: Number of available GPUs.
        is_apple_silicon: Whether running on Apple Silicon.
    """

    device: torch.device
    device_type: str
    device_name: str
    gpu_memory_gb: float
    n_gpus: int
    is_apple_silicon: bool


def detect_device(
    preferred: str = "auto",
) -> DeviceInfo:
    """Detect the best available compute device.

    Priority order:
        1. NVIDIA CUDA GPU (if available)
        2. Apple MPS (if on macOS with Apple Silicon)
        3. CPU (fallback)

    Also applies environment fixes automatically:
        - Sets ``KMP_DUPLICATE_LIB_OK=TRUE`` on macOS to prevent
          OpenMP conflicts between FAISS and PyTorch.
        - Sets ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`` for MPS
          to prevent OOM issues.

    Args:
        preferred: Preferred device. Use ``'auto'`` for automatic
            detection, or force with ``'cuda'``, ``'mps'``,
            ``'cpu'``.

    Returns:
        ``DeviceInfo`` with detected device details.

    Example:
        >>> info = detect_device()
        >>> model.to(info.device)
    """
    is_apple = platform.system() == "Darwin"
    is_arm = platform.machine() in ("arm64", "aarch64")
    is_apple_silicon = is_apple and is_arm

    # Apply macOS environment fixes unconditionally.
    if is_apple:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # --- CUDA ---
    if preferred in ("auto", "cuda") and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = (
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3)
        )
        n_gpus = torch.cuda.device_count()

        logger.info(
            "Detected NVIDIA CUDA: %s (%.1f GB) x %d.",
            gpu_name,
            gpu_mem,
            n_gpus,
        )
        return DeviceInfo(
            device=device,
            device_type="cuda",
            device_name=gpu_name,
            gpu_memory_gb=round(gpu_mem, 1),
            n_gpus=n_gpus,
            is_apple_silicon=False,
        )

    # --- Apple MPS ---
    if (
        preferred in ("auto", "mps")
        and is_apple_silicon
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

        device = torch.device("mps")
        chip_name = _get_apple_chip_name()

        logger.info(
            "Detected Apple MPS: %s.", chip_name
        )
        return DeviceInfo(
            device=device,
            device_type="mps",
            device_name=chip_name,
            gpu_memory_gb=0.0,  # Shared memory, not queryable.
            n_gpus=1,
            is_apple_silicon=True,
        )

    # --- CPU fallback ---
    cpu_name = platform.processor() or "Unknown CPU"
    if is_apple_silicon:
        cpu_name = _get_apple_chip_name()

    logger.info("Using CPU: %s.", cpu_name)
    return DeviceInfo(
        device=torch.device("cpu"),
        device_type="cpu",
        device_name=cpu_name,
        gpu_memory_gb=0.0,
        n_gpus=0,
        is_apple_silicon=is_apple_silicon,
    )


def _get_apple_chip_name() -> str:
    """Get Apple Silicon chip name via sysctl.

    Returns:
        Chip name string (e.g. ``'Apple M2 Pro'``) or
        ``'Apple Silicon'`` if detection fails.
    """
    try:
        import subprocess

        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        name = result.stdout.strip()
        return name if name else "Apple Silicon"
    except Exception:
        return "Apple Silicon"


def get_optimal_workers(device_info: DeviceInfo) -> int:
    """Suggest optimal DataLoader workers for the device.

    Args:
        device_info: Detected device information.

    Returns:
        Recommended number of worker processes.
    """
    cpu_count = os.cpu_count() or 4

    if device_info.device_type == "cuda":
        # GPU: use more workers to keep GPU fed.
        return min(8, cpu_count)
    elif device_info.device_type == "mps":
        # MPS: moderate workers, avoid oversubscription.
        return min(4, cpu_count)
    else:
        # CPU: fewer workers to avoid contention.
        return min(4, max(1, cpu_count - 1))


def get_optimal_batch_size(
    device_info: DeviceInfo,
    default: int = 1024,
) -> int:
    """Suggest optimal batch size for the device.

    Args:
        device_info: Detected device information.
        default: Default batch size from config.

    Returns:
        Recommended batch size.
    """
    if device_info.device_type == "cuda":
        if device_info.gpu_memory_gb >= 16:
            return max(default, 4096)
        elif device_info.gpu_memory_gb >= 8:
            return max(default, 2048)
        else:
            return default
    elif device_info.device_type == "mps":
        return min(default, 2048)
    else:
        return min(default, 1024)


def print_system_info(device_info: DeviceInfo) -> None:
    """Print a formatted summary of the detected system.

    Args:
        device_info: Detected device information.
    """
    lines = [
        "",
        "=" * 50,
        "  SYSTEM CONFIGURATION",
        "=" * 50,
        f"  Platform:    {platform.system()} "
        f"{platform.machine()}",
        f"  Python:      {sys.version.split()[0]}",
        f"  PyTorch:     {torch.__version__}",
        f"  Device:      {device_info.device_type.upper()}",
        f"  Device Name: {device_info.device_name}",
    ]

    if device_info.device_type == "cuda":
        lines.append(
            f"  GPU Memory:  {device_info.gpu_memory_gb} GB"
        )
        lines.append(f"  GPU Count:   {device_info.n_gpus}")

    if device_info.is_apple_silicon:
        lines.append("  Apple MPS:   Available")

    lines.append(
        f"  Workers:     {get_optimal_workers(device_info)}"
    )
    lines.append(
        f"  Batch Size:  {get_optimal_batch_size(device_info)}"
    )
    lines.append("=" * 50)
    lines.append("")

    print("\n".join(lines))
    for line in lines:
        if line.strip():
            logger.info(line.strip())
