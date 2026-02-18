# Experiment Design: Multivariate Clock Drift Data Collection

## Objective

Collect synchronized sensor and time-measurement data across 4 Linux machines
to build a multivariate clock-drift model. Inspired by Graham et al. (NSDI '22),
who showed temperature alone reduces drift from 200 ppm to 100 ppb using a
learned correction. We extend this by collecting all available environmental
covariates (CPU load, power, C-state residency, memory pressure, I/O) alongside
three independent drift signals.

## Machines

| Machine | Role | Root | Stressed | Approx Columns |
|---------|------|------|----------|----------------|
| chameleon | Full collector | Yes | Yes (stress-ng) | ~142 |
| homelab | Full collector | Yes | No (services running) | ~80 |
| ares (master) | Lesser collector | No | No (shared cluster) | ~66 |
| ares-comp-10 | Lesser collector | No | Yes (Python stress) | ~116 |

## Measurements

### Drift Signals (every row, all machines)

1. **MONOTONIC vs MONOTONIC_RAW divergence** — cumulative NTP correction integral.
2. **adjtimex()** — kernel frequency/offset adjustments applied by NTP/chrony.
3. **chronyc tracking** — chrony's own frequency and offset estimates.

These three are cross-validating: `mono_minus_raw` is the integral of
`adj_freq`, and `chrony_freq_ppm` should track `adj_freq / 65536`.

### Sensor Covariates

Per-machine sensor sets are documented in `sensor_inventory.md` (Final
Collection Set section). Categories:

- Temperatures (hwmon, IPMI, thermal zones)
- CPU frequency (per-core)
- CPU utilization (aggregate user/sys/idle/iowait)
- RAPL energy (root-only)
- turbostat (root-only background subprocess)
- IPMI (chameleon-only background subprocess)
- Load average
- Memory (free, available, cached, dirty, buffers)
- C-state residency (2 representative CPUs per socket)
- Disk I/O (reads, writes, bytes)
- Network I/O (rx/tx bytes and packets)
- Context switches and interrupts

### Sampling Rate

1 Hz — all sensors in a single CSV row per tick. Matches Graham's approach
and hardware sensor update rates.

## Stress Schedule

Applied to chameleon (stress-ng) and ares-comp-10 (Python multiprocessing).
Cycling through mixed workload modes to generate temperature variation:

| Offset | Mode | Duration | Purpose |
|--------|------|----------|---------|
| 0:00 | CPU 100% all cores | 45 min | Raise package/core temps |
| 0:45 | Idle | 15 min | Cool-down |
| 1:00 | Memory (alloc/free) | 45 min | DRAM controller heat |
| 1:45 | Idle | 15 min | Cool-down |
| 2:00 | CPU + Memory | 45 min | Peak power draw |
| 2:45 | Idle | 15 min | Cool-down |
| 3:00 | Disk I/O | 45 min | Chipset/PCH heat |
| 3:45 | Idle | 15 min | Cool-down |

4-hour cycle repeats 6 times = 24 hours.

Homelab and ares master run ambient (no artificial stress) — they provide
baseline drift data under natural workloads.

## Timeline

1. **Validation run** (30 minutes, all 4 nodes simultaneously)
   - Verify CSV headers, sensor data, clock monotonicity
   - Confirm `mono_minus_raw` divergence is reasonable
   - Check `adj_freq` / `chrony_freq_ppm` in sane range (< 100 PPM)
   - Row count ~1800, sample jitter < 10 ms, CPU < 2%
   - 5-minute stress burst to test coexistence

2. **24-hour collection** (coordinated start, all 4 nodes)
   - Start stress scripts on chameleon + ares-comp-10
   - Periodic spot-checks via SSH
   - Transfer CSVs back after completion

## Storage

- Local CSV per machine at `~/drift_data/`
- Metadata JSON alongside each CSV (hostname, kernel, column schema, start time)
- Transfer back to localhost after collection for analysis

## Analysis Plan

1. Data quality: completeness, monotonicity, jitter distribution
2. Exploratory: time-series plots of drift vs temperature, power, load
3. Feature engineering: rolling means, derivatives, lag features
4. Modeling: multivariate regression (OLS, Ridge, random forest) predicting
   `adj_freq` or `mono_minus_raw` rate of change from sensor covariates
5. Comparison: temperature-only model vs full multivariate model
6. Cross-machine generalization: train on one, test on another

## References

- Graham, M. et al. "Distributed Clock Synchronization with Application-Level
  Time Queries." NSDI '22. Section 5.2: NTP-only learning works within 0.5 ppm
  of PPS ground truth.
