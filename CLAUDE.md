# Sensor Gathering - Project Notes

## Project Overview

Multivariate clock drift data collection across 4 Linux machines, inspired by
Graham (NSDI '22). Collects synchronized sensor + time-measurement data at 1 Hz
for 24 hours to build multivariate drift prediction models.

## Language & Tooling

- **Python 3.11+**, stdlib-only (no runtime dependencies)
- `uv` for project management, `uv run` for execution
- `ruff` for linting/formatting, `pyright` for type checking
- `pytest` for testing

## Project Structure

```
sensor_gathering/
  pyproject.toml                          # uv project, no runtime deps
  research/                               # experiment design docs
  src/sensor_collector/                   # main package
    __main__.py                           # python3 -m sensor_collector
    cli.py                                # argparse
    config.py                             # configuration dataclass
    clocks.py                             # clock_gettime_ns + adjtimex ctypes
    chrony.py                             # chronyc -c tracking parser
    discovery.py                          # walk sysfs, detect capabilities
    schema.py                             # build column list from inventory
    writer.py                             # buffered CSV + metadata JSON
    collector.py                          # main 1Hz loop
    sensors/                              # sensor reader modules
      hwmon.py, cpufreq.py, rapl.py, procfs.py,
      ipmi.py, turbostat.py, cstate.py,
      network.py, diskstats.py, thermal.py
  scripts/                               # deploy, stress, start/stop
  tests/                                 # pytest unit tests
```

## Remote Machines

| Name | Access | Hostname | OS | Status |
|------|--------|----------|----|--------|
| localhost | direct | Desktop | Windows 11 | OK |
| homelab | `ssh homelab` | Server | Debian (5.10.0-35-amd64) | OK |
| ares | `ssh ares` | ares.ares.local | Ubuntu (5.15.0-164-generic) | OK |
| ares-comp-10 | `ssh ares` then `ssh ares-comp-10` | ares-comp-10 | Ubuntu (5.15.0-143-generic) | OK |
| chameleon | `ssh chameleon` | sensor | Ubuntu (6.8.0-64-generic) | OK |

### Connection Notes

- **ares-comp-10** is only reachable from inside the ares master node (jump host required).
- **chameleon** (129.114.108.185, user `cc`) is on a separate Chameleon Cloud cluster. Key: `~/.ssh/Chameleon`.
