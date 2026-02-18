# ChronoTick — Project Instructions

## Project Overview

ChronoTick is a software-defined oscillator compensation system that applies
zero-shot time series foundation models to predict and correct clock drift on
commodity hardware. The paper targets an IEEE conference (IEEEtran two-column,
10-page content limit). Key results: sub-millisecond accuracy (0.49–0.91 ms MAE),
2.4–3.3x improvement over commodity NTP, 94.9% uncertainty coverage.

Three contributions:
1. **Zero-shot foundation model drift prediction** — no per-device calibration
2. **Self-improving prediction pipeline** — retrospective correction from NTP ground truth
3. **Multivariate compensation** — commodity sensors (temps, freq, memory, I/O, C-states)

## Language & Tooling

- **Python 3.9+**, stdlib-only for sensor collector (no runtime deps)
- Analysis pipeline requires: pandas, numpy, scikit-learn, shap, matplotlib, seaborn
- `uv` for project management, `uv run` for execution
- `ruff` for linting/formatting, `pyright` for type checking
- `pytest` for testing

## Repository Structure

```
sensor_gathering/                         # project root
├── CLAUDE.md                             # this file
├── pyproject.toml                        # uv project config
├── paper/                                # ChronoTick IEEE paper
│   ├── CLAUDE.md                         # paper-specific instructions
│   ├── main.tex                          # paper entry point
│   ├── chronotick.bib                    # bibliography
│   ├── sections/                         # current paper sections
│   │   ├── 1.a.introduction.tex
│   │   ├── 2.related-work.tex
│   │   ├── 3.design.tex
│   │   ├── 4.evaluations.tex
│   │   └── 5.conclusions.tex
│   ├── figures/                          # paper figures (PDF + PNG)
│   ├── tables/                           # LaTeX tables
│   └── old/                              # stale drafts (do not use)
├── sensors/                              # sensor collector codebase
│   ├── src/sensor_collector/             # main package (stdlib-only)
│   │   ├── __main__.py                   # python3 -m sensor_collector
│   │   ├── cli.py                        # argparse CLI
│   │   ├── config.py                     # configuration dataclass
│   │   ├── clocks.py                     # clock_gettime_ns + adjtimex ctypes
│   │   ├── chrony.py                     # chronyc -c tracking parser
│   │   ├── discovery.py                  # walk sysfs, detect capabilities
│   │   ├── schema.py                     # build column list from inventory
│   │   ├── writer.py                     # buffered CSV + metadata JSON
│   │   ├── collector.py                  # main 1 Hz collection loop
│   │   ├── peer_clock.py                 # UDP peer clock offset probing
│   │   └── sensors/                      # sensor reader modules
│   │       ├── hwmon.py, cpufreq.py, rapl.py, procfs.py
│   │       ├── ipmi.py, turbostat.py, cstate.py
│   │       └── network.py, diskstats.py, thermal.py
│   ├── scripts/                          # deploy, stress, start/stop
│   ├── tests/                            # pytest unit tests
│   ├── research/                         # experiment design docs
│   ├── analysis/                         # SHAP sensor importance pipeline
│   │   ├── load_data.py                  # CSV loading + preprocessing
│   │   ├── models.py                     # GBR fitting + SHAP analysis
│   │   ├── figures.py                    # figure generation
│   │   └── run.py                        # pipeline entry point
│   ├── data/                             # collected CSV snapshots
│   │   ├── 8h_snapshot/                  # 4 machines x 8 hours
│   │   ├── 16h_snapshot/                 # 4 machines x 16 hours
│   │   └── 24h_snapshot/                 # 4 machines x 24 hours
│   ├── paper_figures_sensors/            # generated SHAP analysis figures
│   └── nsdi22-paper-najafi_1/            # reference: Graham NSDI'22 paper
└── background/                           # background materials
```

## Paper Notes

- **10 pages of content** limit; citations overflow beyond page 10 and do not count.
- Target venue: IEEE conference (IEEEtran two-column format).
- Section files use numbered prefixes (`sections/1.a.introduction.tex`, etc.).
  Files in `paper/old/` are stale — do not use.
- Do NOT use em-dashes (`---`) in LaTeX prose. Use commas, semicolons, or restructure.

## Data Collection Testbed

| Label | Machine | Hardware | Sensors | Workload |
|-------|---------|----------|---------|----------|
| Commodity (idle) | homelab | i7-6700, 16 GB | ~67 cols | Ambient |
| Commodity (stressed) | chameleon | 2x Xeon Gold 6126, 192 GB | ~137 cols | Cyclic stress-ng |
| HPC (idle) | ares | 2x Xeon Silver 4114, 95 GB | ~80 cols | Shared cluster |
| HPC (stressed) | ares-comp-10 | 2x Xeon Silver 4114, 47 GB | ~102 cols | Cyclic Python stress |

### Remote Access

| Name | Access | OS |
|------|--------|----|
| homelab | `ssh homelab` | Debian (5.10.0-35-amd64) |
| ares | `ssh ares` | Ubuntu (5.15.0-164-generic) |
| ares-comp-10 | `ssh ares` then `ssh ares-comp-10` | Ubuntu (5.15.0-143-generic) |
| chameleon | `ssh chameleon` (user `cc`, key `~/.ssh/Chameleon`) | Ubuntu (6.8.0-64-generic) |

### Deployment

- Sensor collector deployed via GitHub: `https://github.com/JaimeCernuda/sensor-collector`
- On each machine: `cd ~/sensor_collector && git pull && PYTHONPATH=src python3 -m sensor_collector`
- **chameleon**: `sudo env PYTHONPATH=src python3 -m sensor_collector -o /home/cc/drift_data`
- **ares, ares-comp-10**: `--no-root-sensors` (no sudo)
- **ares-comp-10**: no internet; deploy via `ssh ares "rsync -a --delete ~/sensor_collector/ ares-comp-10:~/sensor_collector/"`
- **homelab**: Python 3.9.2 — no `slots=True` on dataclasses, no `zip(strict=True)`

### Peer Clock (UDP 19777)

Only the ares <-> ares-comp-10 LAN link is active. Cross-internet UDP blocked
by firewalls/NAT everywhere else.

| Machine | `--peers` value |
|---------|----------------|
| ares | `ares-comp-10=172.20.101.10:19777` |
| ares-comp-10 | `ares=172.20.1.1:19777` |

## Analysis Pipeline

Runs the multivariate SHAP sensor importance analysis (supports Contribution C3):

```bash
cd sensors/
uv sync --group analysis
uv run --group analysis python -m analysis.run
```

Generates figures in `sensors/paper_figures_sensors/` and copies to `paper/figures/`.

## Lessons Learned

- **`sudo` drops env vars.** Use `sudo env PYTHONPATH=src` on chameleon, plus `-o /home/cc/drift_data`.
- **ares-comp-10 has no internet.** Deploy via rsync from ares, not `git pull`.
- **NFS shared home on ares cluster.** Both ares nodes write to `~/drift_data/`; CSV filenames include hostname.
- **Double-SSH quoting from Windows/Git Bash is fragile.** Use `echo "cmd" | ssh ares "ssh ares-comp-10 bash -s"`.
- **Cross-internet UDP blocked.** All non-LAN nodes blocked by firewalls/NAT.
