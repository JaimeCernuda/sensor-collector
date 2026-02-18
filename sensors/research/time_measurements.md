# Time Measurement Approach

## Overview

We measure clock drift using three independent, cross-validating signals — all
from software, no GPS/PPS hardware required. Graham (NSDI '22, Section 5.2)
proved NTP-only learning achieves drift prediction within 0.5 ppm of PPS ground
truth.

## Clock Sources

### CLOCK_REALTIME

- **What**: Wall-clock time (Unix epoch), adjusted by NTP.
- **Use**: Timestamp for each row. Not a drift signal itself (jumps on step
  corrections).
- **Read**: `clock_gettime(CLOCK_REALTIME)` via ctypes.

### CLOCK_MONOTONIC

- **What**: Time since boot, NTP-disciplined. NTP slews this clock by adjusting
  its frequency — never steps it.
- **Use**: Provides the "disciplined" reference. Comparison against RAW reveals
  cumulative NTP correction.
- **Read**: `clock_gettime(CLOCK_MONOTONIC)` via ctypes.

### CLOCK_MONOTONIC_RAW

- **What**: Raw TSC-derived clock since boot. No NTP adjustments — represents
  the hardware oscillator's native drift.
- **Use**: The "free-running" clock. Divergence from MONOTONIC is the integral
  of all NTP corrections = integral of drift.
- **Read**: `clock_gettime(CLOCK_MONOTONIC_RAW)` via ctypes.

### Derived: mono_minus_raw

```
mono_minus_raw_ns = ts_monotonic_ns - ts_mono_raw_ns
```

This grows over time as NTP corrects the raw oscillator's drift. Its derivative
is the instantaneous drift rate. Over 24 hours at typical 10 ppm drift,
`mono_minus_raw` grows by ~864 ms.

## Kernel NTP State: adjtimex()

The `adjtimex()` syscall reads the kernel's NTP adjustment state:

| Field | Meaning |
|-------|---------|
| `freq` | Frequency correction in scaled PPM (value / 2^16 = PPM) |
| `offset` | Current offset being slewed (ns) |
| `maxerror` | Maximum estimated error (us) |
| `esterror` | Estimated error (us) |
| `status` | NTP status bitmask (STA_PLL, STA_FLL, etc.) |
| `tick` | Microseconds per tick (nominally 10000) |

`adj_freq` is the most important: it's the kernel's current best estimate of
how much to speed up or slow down the clock. This is what NTP/chrony computes
from its offset measurements.

## Chrony Tracking

`chronyc -c tracking` outputs CSV with chrony's own estimates:

| Field | Index | Meaning |
|-------|-------|---------|
| Frequency | 4 | Residual frequency error (PPM) |
| Residual freq | 5 | Remaining after last correction |
| Skew | 6 | Error bound on frequency (PPM) |
| Root delay | 7 | RTT to stratum-1 (seconds) |
| Root dispersion | 8 | Dispersion (seconds) |
| Last offset | 10 | Last measured offset (seconds) |
| Stratum | 2 | Stratum number |

## Cross-Validation

The three signals must be consistent:

1. **Rate of `mono_minus_raw`** ≈ **`adj_freq / 65536`** (both in PPM).
   The kernel applies `adj_freq` to discipline MONOTONIC, so the accumulated
   difference between MONOTONIC and RAW is the integral of `adj_freq`.

2. **`chrony_freq_ppm`** ≈ **`adj_freq / 65536`**. Chrony computes the
   frequency correction and passes it to the kernel via `adjtimex()`.

3. **`chrony_offset_ns`** should be small (< 1 ms usually) and mean-zero over
   long periods, confirming NTP discipline is working.

Discrepancies indicate:
- Large `chrony_skew_ppm` → unstable frequency estimate
- `adj_status` without STA_PLL → NTP not synchronized
- `mono_minus_raw` rate diverging from `adj_freq` → measurement timing issue

## Graham's Approach vs Ours

**Graham (NSDI '22)**:
- Used NTP offset queries as training signal (Section 5.2)
- Temperature sensor as sole predictor
- Achieved 100 ppb drift prediction (vs 200 ppm uncorrected)
- Proved no GPS/PPS needed for learning

**Our approach**:
- Same NTP-based drift signal (we use three cross-validating measures)
- Multiple predictors: temperature, power, load, I/O, C-states
- Goal: determine if multivariate model improves beyond temperature-only
- 4 machines provide hardware diversity for generalization testing

## Implementation Notes

- All three clocks read in immediate succession (< 2 us total) to minimize
  timing skew between them.
- `adjtimex()` accessed via ctypes (Linux-only, no dependencies).
- `chronyc` called as subprocess; output parsed from CSV mode (`-c` flag).
- On systems without chrony, chrony fields are omitted from the CSV.
