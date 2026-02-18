# ChronoTick 2: TSFM Model Landscape (February 2026)

## Executive Summary

The TSFM field has **not** undergone a paradigm shift since ChronoTick 1
(mid-2025). The same core models remain dominant. The main evolution is
incremental maturation: covariate support becoming standard, fine-tuning APIs
stabilizing, and training corpora growing. The only genuinely new entrant is
Toto (Datadog), trained on observability telemetry.

Key implication for Tick 2: the research contribution is not "better models
exist" but rather "fine-tuning these models on drift data unlocks accuracy
that zero-shot cannot achieve," especially for multivariate covariate
utilization (which MUSEval benchmarks show still fails in zero-shot mode).

---

## Model Registry

### Tier 1: Professional-Grade (active development, stable APIs, fine-tuning support)

#### Chronos-2 (Amazon, October 2025)

| Property | Value |
|----------|-------|
| Architecture | Encoder-only, cuboid/group attention |
| Parameters | 28M (small), 120M (base) |
| Context length | Up to 8192 steps |
| Max prediction | Up to 1024 steps |
| Covariates | Yes: past-only, known-future, categorical |
| Probabilistic | Yes: quantile forecasts (configurable levels) |
| Fine-tuning | Yes: LoRA (default) or full, via AutoGluon only |
| License | Apache 2.0 |

**HuggingFace IDs:**
- `amazon/chronos-2` (120M, flagship)
- `autogluon/chronos-2-small` (28M)

**Installation:**
```bash
# Inference only
pip install "chronos-forecasting[extras]>=2.2"

# Fine-tuning (pulls chronos-forecasting as dependency)
pip install "autogluon-timeseries>=1.5"
```

**Inference API:**
```python
from chronos import BaseChronosPipeline

pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cuda")
pred_df = pipeline.predict_df(
    context_df,                    # long-format: target + past covariates
    future_df=future_df,           # known-future covariate columns only
    prediction_length=60,
    quantile_levels=[0.1, 0.5, 0.9],
    id_column="id",
    timestamp_column="timestamp",
    target="target",
)
```

**Covariate rules:**
- Columns in both `context_df` and `future_df` = known-future covariates
- Columns only in `context_df` (besides target) = past-only covariates
- Supports both float and categorical columns

**Fine-tuning API (AutoGluon):**
```python
from autogluon.timeseries import TimeSeriesPredictor

predictor = TimeSeriesPredictor(
    known_covariates_names=["sensor_temp", "cpu_freq"],
).fit(
    train_data,
    hyperparameters={
        "Chronos2": {
            "fine_tune": True,
            "fine_tune_lr": 1e-4,
            "fine_tune_steps": 1000,
            "fine_tune_batch_size": 32,
            # "fine_tune_mode": "full",  # default is LoRA
        }
    },
)
```

**Relevance to ChronoTick:**
Best overall candidate. Rich covariate support matches our sensor feature set.
LoRA fine-tuning via AutoGluon is the most polished FT experience available.
The 28M small model performs within 1% of the 120M base, attractive for
deployment on commodity hardware.

**What changed since Tick 1:**
Chronos 1 (T5-based) was univariate only with no covariates. Chronos-2 is a
completely new encoder-only architecture with native multivariate and covariate
support. Chronos-Bolt (late 2024) was an intermediate step: faster but still
univariate.

---

#### Moirai (Salesforce, August 2025)

| Property | Value |
|----------|-------|
| Architecture | Moirai 2.0: decoder-only transformer. Moirai 1.1: masked encoder (BERT-style) |
| Parameters | 2.0: small only (size TBD). 1.1: 14M (small), 91M (base), 311M (large) |
| Context length | Configurable via patch size |
| Covariates | Yes: Any-Variate Attention (variable channel count) |
| Probabilistic | Yes: quantile forecasts |
| Fine-tuning | Yes: via uni2ts CLI |
| License | Apache 2.0 |

**HuggingFace IDs:**
- `Salesforce/moirai-2.0-R-small` (only 2.0 size released)
- `Salesforce/moirai-1.1-R-small` / `moirai-1.1-R-base` / `moirai-1.1-R-large`

**Installation:**
```bash
pip install "uni2ts>=2.0"
```

**Fine-tuning (uni2ts CLI):**
```bash
# Prepare dataset
python -m uni2ts.data.builder.simple my_data dataset/my_data.csv \
  --date_offset '2024-01-01' --normalize

# Fine-tune
python -m cli.train \
  -cp conf/finetune \
  model=moirai_1.0_R_small \
  model.patch_size=32 \
  model.context_length=512 \
  model.prediction_length=60 \
  data=my_data
```

**Relevance to ChronoTick:**
Any-Variate Attention handles variable sensor counts across machines (67-181
columns) without schema changes. For fine-tuning with size comparisons, use the
1.1 family (small/base/large); 2.0 only has small.

**What changed since Tick 1:**
Moirai 1.0 was masked-encoder only. Moirai 2.0 switched to decoder-only
architecture, claims 96% smaller than 1.0-Large while matching accuracy.
Fine-tuning support matured via uni2ts 2.0.

---

#### Granite TTM r2.1 (IBM, January 2026)

| Property | Value |
|----------|-------|
| Architecture | TinyTimeMixer (MLP-Mixer, no attention) |
| Parameters | 1-5M (extremely small) |
| Context length | 512, 1024, or 1536 steps |
| Max prediction | 96, 192, 336, or 720 steps |
| Covariates | Yes: channel-mixing during fine-tuning, exogenous variables |
| Probabilistic | Point forecasts only (no native quantiles) |
| Fine-tuning | Yes: channel-independent, channel-mix, and exogenous modes |
| License | Apache 2.0 |

**HuggingFace ID:**
- `ibm-granite/granite-timeseries-ttm-r2` (branch-per-variant scheme)
- Branches: `{context}-{horizon}[-ft][-mae]-r2` (e.g., `512-96-r2`, `1024-96-ft-r2.1`)

**Installation:**
```bash
pip install "granite-tsfm>=0.3.3"
```

**Inference API:**
```python
from tsfm_public.toolkit.get_model import get_model

model = get_model(
    model_path="ibm-granite/granite-timeseries-ttm-r2",
    context_length=512,
    prediction_length=96,
)
```

**Channel-mix fine-tuning:**
Documented in `notebooks/tutorial/ttm_channel_mix_finetuning.ipynb` in the
granite-tsfm repo. Enables decoder cross-channel interaction to capture
multivariate correlations.

**Relevance to ChronoTick:**
Uniquely small (1-5M params) and CPU-capable. Channel-mixing FT is the
interesting experiment: does cross-sensor interaction during fine-tuning improve
drift prediction? Point-only output is a limitation (no uncertainty bounds).

**What changed since Tick 1:**
TTM existed in r1 but was less documented. r2/r2.1 expanded training data to
~1B samples, added channel-mix fine-tuning tutorials, and improved the
granite-tsfm package.

---

### Tier 2: Professional-Grade, Custom FT Required (no official FT support)

Both TimesFM 2.5 and Toto are standard PyTorch `nn.Module` models with
accessible weights and differentiable forward passes. Fine-tuning is
**technically feasible** but requires writing custom training loops because
no official FT code exists. Neither model has `torch.no_grad()` on its
`forward()` method; gradients flow normally.

#### TimesFM 2.5 (Google, October 2025)

| Property | Value |
|----------|-------|
| Architecture | Decoder-only transformer (GPT-style) |
| Parameters | 200M |
| Context length | Up to 16384 steps |
| Max prediction | Up to 1024 steps (with continuous quantile head) |
| Covariates | Yes: via `forecast_with_covariates()`, requires JAX |
| Probabilistic | Yes: native quantile output (10 levels) |
| Fine-tuning | No official support for 2.5; custom loop feasible (~200-300 LoC) |
| License | Apache 2.0 |

**HuggingFace ID:**
- `google/timesfm-2.5-200m-pytorch`
- `google/timesfm-2.5-200m-flax`

**Installation (from source only, NOT on PyPI):**
```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
pip install -e ".[torch]"       # PyTorch inference
pip install -e ".[xreg]"        # adds covariate support (requires JAX)
```

Note: `pip install timesfm==1.3.0` (PyPI) loads 1.0/2.0 checkpoints only.
Cannot load the 2.5 model.

**Inference API:**
```python
import timesfm

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=1024,
    max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
))

point, quantiles = model.forecast(
    horizon=60,
    inputs=[series_array],
)
```

**Covariate API:**
```python
model.compile(timesfm.ForecastConfig(
    max_context=512,
    max_horizon=60,
    return_backcast=True,         # REQUIRED for covariates
))

point_out, xreg_out = model.forecast_with_covariates(
    inputs=[context_series],
    dynamic_numerical_covariates={
        "temperature": [[*ctx_temps, *horizon_temps]],
    },
    xreg_mode="xreg + timesfm",  # linear model on covariates, TSFM on residuals
)
```

The covariate mechanism is a linear regression layer (XReg) that runs on JAX,
even if the main model uses PyTorch. Two modes:
- `"xreg + timesfm"`: fit linear model on covariates first, forecast residuals
- `"timesfm + xreg"`: forecast first, fit linear model on residuals

**Fine-tuning status:**
- v1 codebase has `FinetuningConfig` + `TimesFMFinetuner` for 1.0/2.0
- v1 has PEFT scripts (LoRA, DoRA, linear probing) for 1.0
- None of this is ported to 2.5 (different architecture, incompatible forward signature)
- v1 `forward(input_ts, input_padding, freq)` vs v2.5 `forward(inputs, masks, decode_caches)`
- In-context fine-tuning (ICML 2025 paper): code NOT released (GitHub issue
  #293 open since August 2025, zero response)
- GitHub issues #327 ("How to finetune v2.5?"), #261 ("Fine-tuning folder
  missing") both open with zero maintainer response

**Custom fine-tuning feasibility:**
The model IS a standard `nn.Module`. Weights are accessible, gradients flow
through `forward()`. The `decode()` method wraps in `torch.no_grad()`, but
`forward()` itself does not. A custom training loop is feasible:
- Call `model.forward()` directly (not `decode()`)
- Handle RevIN normalization manually (extractable from existing codebase)
- Build a patch-level dataset (patch size 32, `[B, num_patches, 32]`)
- Define loss on dual output heads (point projection + quantile spread)
- Estimated effort: ~200-300 LoC, a few days of engineering
- Nobody has published v2.5 FT code publicly as of February 2026

**Relevance to ChronoTick:**
Continuity with Tick 1 (which used TimesFM). Primary value is zero-shot
benchmarking and covariate ablation. Custom FT is feasible but would be a
novel engineering contribution (nobody has done it for 2.5). Could also
fine-tune the older 2.0 (500M) checkpoint using v1 code as a comparison point.

**What changed since Tick 1:**
v2.5 added continuous quantile head, 16K context, covariate support (XReg),
and flip invariance. Accuracy improved. But fine-tuning regressed: v1 had
working FT code, v2.5 does not.

---

#### Toto (Datadog, May 2025)

| Property | Value |
|----------|-------|
| Architecture | Decoder-only, time-wise + space-wise attention blocks |
| Parameters | 151M |
| Context length | Up to 4096 steps |
| Covariates | Native multivariate (all channels are equal) |
| Probabilistic | Yes: sample-based forecasts (MixtureOfStudentT, quantiles from samples) |
| Fine-tuning | No official support; custom loop feasible (~1-2 days engineering) |
| License | Apache 2.0 |

**HuggingFace ID:**
- `Datadog/Toto-Open-Base-1.0`

**Installation:**
```bash
pip install toto-ts
```

**Inference API:**
```python
import torch
from toto.model.toto import Toto
from toto.inference.forecaster import TotoForecaster
from toto.data.util.dataset import MaskedTimeseries

toto = Toto.from_pretrained("Datadog/Toto-Open-Base-1.0")
toto.to("cuda")

forecaster = TotoForecaster(toto.model)

inputs = MaskedTimeseries(
    series=torch.randn(n_channels, seq_len).to("cuda"),
    padding_mask=torch.full((n_channels, seq_len), True, dtype=torch.bool).to("cuda"),
    id_mask=torch.zeros(n_channels, seq_len).to("cuda"),
    timestamp_seconds=torch.zeros(n_channels, seq_len).to("cuda"),
    time_interval_seconds=torch.full((n_channels,), 1.0).to("cuda"),
)

forecast = forecaster.forecast(inputs, prediction_length=60, num_samples=256)
# forecast.median, forecast.samples, forecast.quantile(0.1)
```

**Custom fine-tuning feasibility:**
The model IS a standard `nn.Module` (`TotoBackbone` extends `torch.nn.Module`).
All weights are differentiable. Dropout=0.1 is configured, confirming training
was intended. Datadog's own "TotoFT" variant exists in their papers but
weights/code are not released. GitHub issues #37 and #42 asking for FT have
been ignored for 8+ months.

What you need to build:
- **Loss function**: paper specifies 57% NLL + 43% Cauchy robust loss.
  `distribution.log_prob()` is differentiable. NLL component is straightforward.
  Cauchy robust loss: `log(0.5 * ((x - x_hat) / delta)^2 + 1)`.
  Paper's training config: AdamW, lr=0.001, cosine annealing, 5k warmup, 193k steps.
- **Target alignment**: patch size 64, stride 64. Distribution at position t
  predicts t+1. Must handle the affine transform (loc/scale returned separately
  in `TotoOutput`, not applied inside `forward()`).
- **Data prep**: `forward()` expects `[batch, variate, time_steps]` inputs with
  padding_mask and id_mask. Existing `MaskedTimeseries` utilities help.
- Estimated effort: ~1-2 days, medium difficulty. Main gotchas are the
  affine-transformed distribution handling and next-patch target alignment.

**Relevance to ChronoTick:**
Trained on 2.36 trillion tokens, ~70% from Datadog's production observability
telemetry (CPU, memory, network, disk metrics). This is nearly identical to
ChronoTick's sensor suite. Valuable as domain-proximity baseline. Custom FT
would be novel (nobody has published it). If Toto outperforms general-purpose
models zero-shot, it validates that domain-relevant pretraining matters. If
fine-tuned Toto beats fine-tuned Chronos-2, it validates domain pretraining +
FT as the optimal strategy.

**What changed since Tick 1:**
Toto did not exist during Tick 1. It is the only genuinely new entrant.

---

### Tier 3: Research-Grade (excluded from Tick 2)

| Model | Why Excluded |
|-------|-------------|
| **MOMENT** (CMU) | Channel-independent (no cross-sensor correlations), no covariate mechanism, no updates since ICML 2024 |
| **Time-MoE** (ICLR 2025) | Univariate only, no covariate support, limited FT API |
| **Sundial** (Tsinghua) | Univariate only, fine-tuning "coming soon" (still not released) |
| **Lag-Llama** | Univariate only, no active development in 2025 |
| **Timer** (Tsinghua) | Univariate, research toolkit only |
| **TimeGPT** (Nixtla) | Proprietary API-only, no open weights, no local FT |

---

## Capability Matrix

| Capability | Chronos-2 | TimesFM 2.5 | Toto | Moirai | Granite TTM |
|-----------|-----------|-------------|------|--------|-------------|
| Zero-shot inference | Yes | Yes | Yes | Yes | Yes |
| Covariate support | Native | XReg (linear, needs JAX) | Native multivariate | Any-Variate Attention | Channel-mix FT |
| Probabilistic output | Quantiles | Quantiles (10 levels) | Samples | Quantiles | No (point only) |
| Fine-tuning | LoRA / full (AutoGluon) | Custom loop (feasible) | Custom loop (feasible) | Yes (uni2ts CLI) | Yes (channel-mix) |
| CPU inference | Yes (slow) | Yes (slow) | No (needs GPU) | Yes (slow) | Yes (fast, 1-5M) |
| Min VRAM (inference) | ~0.5 GB (small) | ~2 GB | ~2 GB | ~0.5 GB (small) | ~0.1 GB |
| Max context | 8192 | 16384 | 4096 | Configurable | 1536 |
| Max horizon | 1024 | 1024 | Configurable | Configurable | 720 |

---

## Fine-Tuning Feasibility Summary

| Model | FT Method | Framework | Data Format | VRAM for FT | Status |
|-------|-----------|-----------|------------|-------------|--------|
| **Chronos-2** | LoRA (default) or full | AutoGluon-TimeSeries >= 1.5 | TimeSeriesDataFrame (long format) | ~4 GB LoRA, ~8 GB full | Production-ready |
| **Moirai 1.1/2.0** | Full parameter | uni2ts >= 2.0 CLI | CSV via `uni2ts.data.builder` | ~4-8 GB depending on size | Working, CLI-based |
| **Granite TTM** | Channel-mix or channel-independent | granite-tsfm >= 0.3.3 | PyTorch tensors (batch, channels, seq) | ~1 GB (tiny model) | Documented with tutorials |
| **TimesFM 2.5** | Custom PyTorch loop | Call `forward()` directly (not `decode()`) | Patch tensors `[B, num_patches, 32]` | ~6 GB | Custom only; ~200-300 LoC; nobody has done it for 2.5 |
| **Toto** | Custom PyTorch loop | Call `model(inputs, mask, id_mask)` | `[B, variate, time_steps]` tensors | ~4 GB | Custom only; ~1-2 days; loss = 57% NLL + 43% Cauchy |

---

## Key Research Findings

### MUSEval Warning (Synthefy, 2025)
The largest multivariate evaluation benchmark found that adding correlated
variates provided **no benefit** to zero-shot TSFM inference for TimesFM 2.5,
Toto, and others. This means:
- Zero-shot multivariate does not automatically leverage sensor correlations
- Fine-tuning is likely required to teach models cross-variate relationships
- This directly motivates Tick 2's core thesis

### RL / Continuous Learning (State of the Art)
- **Time-R1** (NeurIPS 2025): Two-stage RL (CoT SFT + GRIP reward). Very early research, LLM-based, not applicable to standard TSFMs.
- **In-context fine-tuning** (Google, ICML 2025): Few-shot adaptation without weight updates. Promising concept but code not released.
- **CoRA** (October 2025): Lightweight covariate-aware adapter on frozen TSFM backbones. Could be interesting for adding covariate conditioning to univariate models.
- **Assessment**: Standard supervised fine-tuning (LoRA, full) is the only production-ready path. RL for time series is 1-2 years from practical use.

---

## Recommended Experiment Design for Tick 2

### Models to benchmark (zero-shot): all 5
### Models to fine-tune:

**Tier A -- Official FT support (low risk):**
- Chronos-2 (LoRA/full via AutoGluon)
- Moirai (full via uni2ts CLI)
- Granite TTM (channel-mix via granite-tsfm)

**Tier B -- Custom FT (medium risk, novel contribution):**
- TimesFM 2.5 (custom PyTorch loop, ~200-300 LoC)
- Toto (custom PyTorch loop, ~1-2 days, loss from paper)

Writing custom FT for TimesFM 2.5 or Toto would itself be a contribution
(nobody has published this). However, it adds engineering risk. Recommended
approach: implement Tier A first, then attempt Tier B if time permits.

### Comparison axes:
1. Zero-shot 2026 models vs. Tick 1 results (same data, new models)
2. Zero-shot with vs. without covariates (MUSEval validation)
3. Fine-tuned vs. zero-shot (core thesis)
4. Fine-tuned with vs. without covariates (does FT unlock multivariate benefit?)
5. Cross-machine generalization (train on machine A, test on B)
6. Model size efficiency (Granite 1-5M vs. Chronos 28-120M vs. Moirai 14-311M)
7. Domain pretraining value (Toto zero-shot vs. others; Toto FT vs. others FT if Tier B done)
