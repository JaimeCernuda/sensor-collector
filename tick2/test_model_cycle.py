"""Smoke test: install -> load -> predict -> uninstall cycle for each model.

Uses synthetic data (no sensor CSVs needed). Runs on CPU with n_samples=1.
Tests the exact install/cleanup/verify cycle from the notebook.
"""

import importlib
import subprocess
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Model configs — mirrors notebook cell 2
# ---------------------------------------------------------------------------
MODEL_CONFIGS = [
    {
        "name": "chronos2-small",
        "cleanup": [],
        "install": ["uv pip install -q chronos-forecasting[extras]>=2.2"],
        "verify": "chronos",
        "group": "A",
    },
    {
        "name": "granite-ttm",
        "cleanup": [],
        "install": ["uv pip install -q granite-tsfm>=0.3.3"],
        "verify": "tsfm_public",
        "group": "A",
    },
    # timesfm needs source install — skip for local smoke test
    # toto-ts pins exact torch version — test separately
    {
        "name": "toto",
        "cleanup": [
            "uv pip uninstall -q chronos-forecasting granite-tsfm 2>/dev/null; true",
        ],
        "install": [
            'uv pip install -q "setuptools<81"',
            "uv pip install -q toto-ts",
        ],
        "verify": "toto",
        "group": "B",
    },
    {
        "name": "moirai-1.1-small",
        "cleanup": [
            "uv pip uninstall -q toto-ts 2>/dev/null; true",
        ],
        "install": ["uv pip install -q uni2ts"],
        "verify": "uni2ts",
        "group": "C",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False


def run_cmd(cmd: str, timeout: int = 600) -> tuple[bool, str]:
    """Run a shell command. Returns (success, stderr_tail)."""
    print(f"    $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        tail = result.stderr.strip().split("\n")[-5:]
        return False, "\n".join(tail)
    return True, ""


def install_model(cfg: dict) -> bool:
    """Run cleanup + install + verify for a model."""
    name, verify = cfg["name"], cfg["verify"]

    # Cleanup
    for cmd in cfg.get("cleanup", []):
        run_cmd(cmd, timeout=120)

    # Check if already importable (and no cleanup was needed)
    importlib.invalidate_caches()
    if not cfg.get("cleanup") and _can_import(verify):
        print(f"    [{verify}] already available")
        return True

    # Install
    for cmd in cfg["install"]:
        ok, err = run_cmd(cmd)
        if not ok:
            print(f"    [FAIL] {err}")
            return False

    importlib.invalidate_caches()
    if _can_import(verify):
        print(f"    [{verify}] ready")
        return True

    print(f"    [FAIL] {verify} not importable after install")
    return False


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
np.random.seed(42)
CONTEXT = np.cumsum(np.random.randn(512) * 0.01).astype(np.float32)  # fake drift
COVARIATES = np.random.randn(512, 3).astype(np.float32)  # 3 fake features
HORIZON = 60

# ---------------------------------------------------------------------------
# Test loop
# ---------------------------------------------------------------------------
results = {}

for cfg in MODEL_CONFIGS:
    name = cfg["name"]
    print(f"\n{'='*60}")
    print(f"  {name}  [Group {cfg['group']}]")
    print(f"{'='*60}")

    # Step 1: Install
    print("  [1/3] Installing...")
    if not install_model(cfg):
        results[name] = "INSTALL_FAILED"
        continue

    # Step 2: Load
    print("  [2/3] Loading model...")
    model = None
    try:
        from tick2.models.registry import get_model
        model = get_model(name)
        t0 = time.time()
        model.load(device="cpu")
        load_time = time.time() - t0
        print(f"    Loaded in {load_time:.1f}s, ~{model.memory_footprint_mb():.0f} MB")
    except Exception as e:
        print(f"    [FAIL] Load error: {e}")
        results[name] = f"LOAD_FAILED: {e}"
        continue

    # Step 3: Predict
    print("  [3/3] Running inference...")
    try:
        # Univariate
        pred = model.predict(CONTEXT, HORIZON)
        assert pred.point_forecast.shape == (HORIZON,), \
            f"Expected ({HORIZON},), got {pred.point_forecast.shape}"
        print(f"    Univariate OK: shape={pred.point_forecast.shape}, "
              f"time={pred.inference_time_ms:.1f}ms")

        if pred.quantile_lo is not None:
            print(f"    Quantiles: lo={pred.quantile_lo.shape}, hi={pred.quantile_hi.shape}")

        # Multivariate (if supported)
        if model.supports_covariates:
            pred_cov = model.predict(CONTEXT, HORIZON, covariates=COVARIATES)
            print(f"    Multivariate OK: shape={pred_cov.point_forecast.shape}, "
                  f"time={pred_cov.inference_time_ms:.1f}ms")
        else:
            print(f"    Covariates: not supported (skipped)")

        results[name] = "OK"

    except Exception as e:
        print(f"    [FAIL] Predict error: {e}")
        import traceback
        traceback.print_exc()
        results[name] = f"PREDICT_FAILED: {e}"

    finally:
        del model

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
for name, status in results.items():
    icon = "OK" if status == "OK" else "FAIL"
    print(f"  [{icon:4s}] {name}: {status}")
