"""Google Colab environment utilities.

Auto-detection, Drive mounting, path management, and GPU setup for
running ChronoTick 2 benchmarks and fine-tuning on Colab.
"""

from __future__ import annotations

from pathlib import Path


def is_colab() -> bool:
    """Detect if running inside Google Colab."""
    try:
        import google.colab  # type: ignore[import-not-found]  # noqa: F401

        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> Path:
    """Mount Google Drive in Colab.

    Args:
        mount_point: Where to mount Drive.

    Returns:
        Path to the mounted Drive root (My Drive).
    """
    from google.colab import drive  # type: ignore[import-not-found]

    drive.mount(mount_point)
    return Path(mount_point) / "MyDrive"


def get_data_dir(
    colab_path: str = "/content/drive/MyDrive/chronotick2/data",
    local_path: str | None = None,
) -> Path:
    """Get the data directory, adapting to Colab vs. local environment.

    Args:
        colab_path: Path on Drive where sensor data is stored.
        local_path: Override for local development.

    Returns:
        Path to the data directory.
    """
    if local_path is not None:
        return Path(local_path)

    if is_colab():
        return Path(colab_path)

    # Local development: relative to tick2/
    from tick2.data.preprocessing import DEFAULT_DATA_DIR

    return DEFAULT_DATA_DIR


def get_output_dir(
    colab_path: str = "/content/drive/MyDrive/chronotick2/results",
    local_path: str | None = None,
) -> Path:
    """Get the output directory for benchmark results.

    Args:
        colab_path: Path on Drive for saving results.
        local_path: Override for local development.

    Returns:
        Path to the output directory (created if necessary).
    """
    if local_path is not None:
        p = Path(local_path)
    elif is_colab():
        p = Path(colab_path)
    else:
        p = Path(__file__).resolve().parents[3] / "results"

    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_gpu() -> str:
    """Configure GPU for optimal usage and return device string.

    Returns:
        "cuda" if GPU available, "cpu" otherwise.
    """
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

            # Enable TF32 for Ampere+ GPUs (faster matmuls)
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]

            return "cuda"
        else:
            print("No GPU detected, using CPU")
            return "cpu"
    except ImportError:
        print("PyTorch not installed, using CPU")
        return "cpu"


def install_model_deps(model_name: str) -> None:
    """Install model-specific dependencies in Colab.

    Args:
        model_name: Registry model name.
    """
    import subprocess
    import sys

    install_cmds: dict[str, list[str]] = {
        "chronos2-small": ["chronos-forecasting[extras]>=2.2"],
        "chronos2-base": ["chronos-forecasting[extras]>=2.2"],
        "moirai-1.1-small": ["uni2ts>=2.0"],
        "moirai-1.1-base": ["uni2ts>=2.0"],
        "moirai-1.1-large": ["uni2ts>=2.0"],
        "moirai-2.0-small": ["uni2ts>=2.0"],
        "toto": ["toto-ts"],
        "granite-ttm": ["granite-tsfm>=0.3.3"],
    }

    if model_name in install_cmds:
        for pkg in install_cmds[model_name]:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
    elif model_name == "timesfm-2.5":
        print("TimesFM 2.5 requires source install:")
        print("  !git clone https://github.com/google-research/timesfm")
        print('  !cd timesfm && pip install -e ".[torch]"')
        print("Run these commands manually in a Colab cell.")
    else:
        print(f"Unknown model: {model_name}")
