# Sensor Gathering - Project Notes

## Project Overview

Multivariate clock drift data collection across 4 Linux machines, inspired by
Graham (NSDI '22). Collects synchronized sensor + time-measurement data at 1 Hz
for 24 hours to build multivariate drift prediction models.

## Language & Tooling

- **Python 3.9+**, stdlib-only (no runtime dependencies)
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
    peer_clock.py                         # UDP peer clock offset probing
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
- **ares & ares-comp-10** share NFS home directory — CSV files from both appear in `~/drift_data/`.
- **ares-comp-10** has no internet access — deploy via `rsync` from ares, not `git pull`.

### Deployment Notes

- Code is deployed via GitHub: `https://github.com/JaimeCernuda/sensor-collector`
- On each machine: `cd ~/sensor_collector && git pull && PYTHONPATH=src python3 -m sensor_collector`
- **chameleon**: must run with `sudo env PYTHONPATH=src` and `-o /home/cc/drift_data`
  (sudo changes home to /root; turbostat needs root for MSR access)
- **ares, ares-comp-10**: use `--no-root-sensors` (no sudo available)
- **ares-comp-10**: no internet; deploy with `ssh ares "rsync -a --delete ~/sensor_collector/ ares-comp-10:~/sensor_collector/"`
- **homelab**: Python 3.9.2 — no `slots=True` on dataclasses, no `zip(strict=True)`
- Data output: `~/drift_data/` on each machine

### Peer Clock Probing

UDP port **19777** is used for pairwise clock offset measurement between nodes.
Each node runs a UDP server and probes its peers every 2 seconds using NTP-style
4-timestamp exchanges.

Per-machine `--peers` arguments:

| Machine | `--peers` value |
|---------|----------------|
| homelab | `ares=216.47.152.168:19777,chameleon=129.114.108.185:19777` |
| chameleon | `ares=216.47.152.168:19777` |
| ares | `chameleon=129.114.108.185:19777,ares-comp-10=172.20.101.10:19777` |
| ares-comp-10 | `ares=172.20.1.1:19777` |

Firewall: ensure UDP 19777 is open on all nodes.

**Status (verified 2026-02-13):** LAN peer links (ares <-> ares-comp-10) work
with ~0.1ms offset, ~0.3ms RTT. Cross-internet links (homelab, chameleon, ares)
blocked by firewalls/NAT — needs:
- **homelab**: UDP 19777 port forward on home router
- **ares**: university firewall exception for UDP 19777 inbound
- **chameleon**: Chameleon Cloud security group rule for UDP 19777 inbound
