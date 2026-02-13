# Sensor Inventory — All Machines

Probed 2026-02-12. This documents every sensor and metric available on each
node for the 24-hour parallel data collection.

---

## Machine Summary

| | Chameleon | Homelab | Ares (master) | Ares-comp-10 |
|--|-----------|---------|---------------|-------------|
| **Hostname** | sensor | Server | ares.ares.local | ares-comp-10 |
| **CPU** | 2x Xeon Gold 6126 | i7-6700 | 2x Xeon Silver 4114 | 2x Xeon Silver 4114 |
| **Cores/Threads** | 24C/48T | 4C/8T | 20C/40T | 20C/40T |
| **RAM** | 192 GB ECC | 16 GB | 95 GB | 47 GB |
| **Kernel** | 6.8.0-64 (Ubuntu 24.04) | 5.10.0-35 (Debian) | 5.15.0-164 (Ubuntu) | 5.15.0-143 (Ubuntu) |
| **Storage** | 224G SATA SSD | 256G NVMe + 12T HDD | 44T RAID + 1.8T M.2 | 32G + 477G SSD + 932G HDD + 239G NVMe |
| **Can install** | Yes (full sudo) | Yes | **No** | **No** |
| **Role** | Full collector | Full collector | Lesser collector | Lesser collector |

---

## Chameleon — Full Collector

### Hardware Sensors (lm-sensors + IPMI)

**lm-sensors (27 temp + 1 power):**
```
coretemp-isa-0000: Package id 0 + 12 cores (Core 0,1,3,4,5,6,8,9,10,11,12,13) = 13 temps
coretemp-isa-0001: Package id 1 + 12 cores (Core 0,1,2,3,4,5,6,8,9,11,12,13) = 13 temps
pch_lewisburg: 1 temp (chipset, 28C)
power_meter-acpi-0: 1 power reading (108W, 2s interval)
```

**IPMI analog sensors (19 readings, requires sudo):**
```
Temp x2:          CPU socket temps (22C, 22C)
Inlet Temp:       19C
Exhaust Temp:     21C
Fan1-Fan6:        6x RPM (8640-8880)
Current 1-2:      2x PSU amps (0.4A, 0.2A)
Voltage 1-2:      2x PSU volts (210V, 210V)
Pwr Consumption:  88W (system total)
CPU/MEM/IO/SYS Usage: 4x percent (0%)
```

### System Metrics

| Source | Count | Access |
|--------|-------|--------|
| CPU frequency (scaling_cur_freq) | 48 per-core readings | user |
| C-states (POLL, C1, C1E, C6) | 4 states x 48 CPUs = 192 counters | user |
| RAPL energy (pkg-0, pkg-1, dram x2) | 4 counters | **root** |
| /proc/interrupts | 142 IRQs x 48 CPUs = 6,816 | user |
| /proc/softirqs | 11 types x 48 CPUs = 528 | user |
| /proc/vmstat | 178 counters | user |
| /proc/meminfo | 55 counters | user |
| Network interfaces | 5 (eno1np0 active, eno2np1, eno3, eno4, lo) | user |
| turbostat fields | 32 (Avg_MHz, Busy%, Bzy_MHz, TSC_MHz, IPC, IRQ, **SMI**, C-states, CoreTmp, PkgTmp, PkgWatt, RAMWatt, UncMHz, etc.) | root |
| perf events | 9,521 available | root |
| Thermal zones | 3 (pch_lewisburg, x86_pkg_temp x2) | user |
| EDAC ECC | 4 MCs x 2 counters = 8 | user |

### Tools Installed
lm-sensors, ipmitool, linux-tools (turbostat, perf)

### Network
5 interfaces: eno1np0 (10 Gbps, active), eno2np1, eno3, eno4, lo

### Disk
1x 224G SATA SSD (MZ7KM240HMHQ0D3), no NVMe

---

## Homelab — Full Collector

### Hardware Sensors (lm-sensors)

**lm-sensors (9 temp + 0 power):**
```
coretemp-isa-0000: Package id 0 + 4 cores = 5 temps (36C pkg, 32-35C cores)
pch_skylake: 1 temp (59.5C — hot!)
acpitz-acpi-0: 2 temps (27.8C, 29.8C)
nvme-pci-0200: 1 temp (43.9C — Samsung PM951)
```

### System Metrics

| Source | Count | Access |
|--------|-------|--------|
| CPU frequency (scaling_cur_freq) | 8 per-core readings | user |
| C-states (POLL, C1, C1E, C3, C6, C7s, C8) | 7 states x 8 CPUs = 56 counters | user |
| RAPL energy (package-0, core, uncore, dram) | 4 counters | **root** |
| /proc/interrupts | 41 IRQs x 8 CPUs = 328 | user |
| /proc/softirqs | 11 types x 8 CPUs = 88 | user |
| /proc/vmstat | 154 counters | user |
| /proc/meminfo | 51 counters | user |
| Network interfaces | 66 (1 physical enp0s31f6, lots of docker bridges/veths) | user |
| perf events | 1,699 available | user |
| Thermal zones | 4 (acpitz x2, pch_skylake, x86_pkg_temp) | user |

### Tools Installed
lm-sensors, linux-perf

### Missing
No IPMI (no /dev/ipmi0), no turbostat

### Network
enp0s31f6 (physical), 18 docker bridges, ~47 veths, lo

### Disk
nvme0n1: 239G Samsung PM951, sda: 10.9T Seagate HDD, sr0: DVD

---

## Ares (master) — Lesser Collector

### Hardware Sensors (sysfs only, no lm-sensors)

**hwmon (23 temps, no voltage/fan/power):**
```
pch_lewisburg: 1 temp (chipset)
coretemp socket 0: 11 temps (package + 10 cores)
coretemp socket 1: 11 temps (package + 10 cores)
(hwmon0 exists but unnamed, no sensors)
```

### System Metrics

| Source | Count | Access |
|--------|-------|--------|
| CPU frequency | **NOT AVAILABLE** (no cpufreq driver) | — |
| C-states (POLL, C1, C1E, C6) | 4 states x 40 CPUs = 160 counters | user |
| RAPL energy (pkg-0, dram, pkg-1, dram) | 4 counters | **root** (no sudo) |
| /proc/interrupts | 289 IRQs x 40 CPUs = 11,560 | user |
| /proc/softirqs | 11 types x 40 CPUs = 440 | user |
| /proc/vmstat | 159 counters | user |
| /proc/meminfo | 51 counters | user |
| Thermal zones | 3 (pch_lewisburg, x86_pkg_temp x2) | user |

### Tools Available
ipmitool (installed but no /dev/ipmi0 — unusable), no lm-sensors, no perf, no turbostat

### Limitations
- **No cpufreq** — cannot read per-core frequency
- **No RAPL** — energy_uj is root-only and we don't have sudo
- **No IPMI** — tool exists but no IPMI device
- **No lm-sensors** — hwmon temps readable via sysfs directly

### Network
9 interfaces: eno1, eno2, ens1np0, ens4f0, ens4f1, enx7ed30aefbc07, docker0, virbr0, lo

### Disk
sda: 43.7T RAID (LSI 930-8i), sdb: 1.8T ThinkSystem M.2

---

## Ares-comp-10 — Lesser Collector

### Hardware Sensors (sysfs only)

**hwmon (25 temps via sysfs):**
```
nvme: 1 temp (NVMe drive)
pch_lewisburg: 1 temp (chipset)
coretemp socket 0: ~11 temps
coretemp socket 1: ~11 temps
(hwmon0 unnamed, 1 extra — possibly acpi)
```

### System Metrics

| Source | Count | Access |
|--------|-------|--------|
| CPU frequency (scaling_cur_freq) | 40 per-core readings | user |
| C-states | ~4 states x 40 CPUs = 160 counters (est.) | user |
| RAPL energy (pkg-0, pkg-1) | 2+ counters | **root** (no sudo) |
| /proc/interrupts | 209 IRQs x 40 CPUs = 8,360 | user |
| /proc/softirqs | 11 types x 40 CPUs = 440 | user |
| /proc/vmstat | 159 counters | user |
| /proc/meminfo | 51 counters | user |
| Thermal zones | 3 (pch_lewisburg, x86_pkg_temp x2) | user |

### Limitations
- **No RAPL** — root-only, no sudo
- **No IPMI tools**
- **No lm-sensors, perf, turbostat**
- Has cpufreq (unlike ares master)
- Has NVMe temp (Toshiba RD400)

### Network
6 interfaces: eno1, eno2, enp47s0np0, enx7ed30aef36df, docker0, lo

### Disk
sda: 30G LITEON SSD, sdb: 477G Samsung 860, sdc: 932G Seagate HDD, nvme0n1: 239G Toshiba RD400

---

## Comparison: What Each Collector Can Sample

| Sensor Category | Chameleon | Homelab | Ares | Ares-comp-10 |
|----------------|-----------|---------|------|-------------|
| **CPU core temps** | 26 (coretemp) | 5 (coretemp) | 22 (sysfs) | ~22 (sysfs) |
| **Chipset temp** | 1 (PCH) | 1 (PCH) | 1 (PCH) | 1 (PCH) |
| **NVMe temp** | — | 1 | — | 1 |
| **ACPI/ambient temps** | — | 2 (acpitz) | — | — |
| **IPMI temps** | 4 (inlet/exhaust/CPU) | — | — | — |
| **Fan RPM** | 6 (IPMI) | — | — | — |
| **PSU current** | 2 (IPMI) | — | — | — |
| **PSU voltage** | 2 (IPMI) | — | — | — |
| **System power** | 2 (ACPI 108W + IPMI 88W) | — | — | — |
| **CPU frequency** | 48 (cpufreq) | 8 (cpufreq) | **none** | 40 (cpufreq) |
| **C-state counters** | 192 | 56 | 160 | ~160 |
| **RAPL energy** | 4 (root) | 4 (root) | **none** (no sudo) | **none** (no sudo) |
| **turbostat** | 32 fields (root) | 32 fields (root) | — | — |
| **perf events** | 9,521 | 1,699 | — | — |
| **Interrupts** | 6,816 | 328 | 11,560 | 8,360 |
| **Softirqs** | 528 | 88 | 440 | 440 |
| **vmstat** | 178 | 154 | 159 | 159 |
| **meminfo** | 55 | 51 | 51 | 51 |

---

## Derived / Computed Metrics (non-sensor)

These come from /proc and are available on ALL machines without any tools.
They are computed as deltas between samples (counters are cumulative).

| Metric | Source | Per-CPU? | Available on |
|--------|--------|----------|-------------|
| **CPU utilization** (user/sys/idle/iowait/irq/softirq %) | /proc/stat | yes | all |
| **Context switches/sec** | /proc/stat (ctxt) | total | all |
| **Interrupts/sec** (total + per-IRQ) | /proc/stat + /proc/interrupts | yes | all |
| **Softirqs/sec** (NET_RX, TIMER, SCHED, RCU, etc.) | /proc/softirqs | yes | all |
| **Processes created/sec** | /proc/stat (processes) | total | all |
| **Procs running / blocked** | /proc/stat | total | all |
| **Load average** (1/5/15 min) | /proc/loadavg | total | all |
| **Memory pressure** (free, cached, buffers, dirty) | /proc/meminfo | total | all |
| **Page faults/sec** (major + minor) | /proc/vmstat (pgfault, pgmajfault) | total | all |
| **Swap activity** (pswpin, pswpout) | /proc/vmstat | total | all |
| **TLB shootdowns/sec** | /proc/vmstat (nr_tlb_*) | total | all |
| **NUMA stats** (local/remote allocs) | /proc/vmstat (numa_*) | total | multi-socket only |
| **Disk I/O** (reads/writes/sec, bytes/sec, queue depth) | /proc/diskstats | per-disk | all |
| **Network I/O** (bytes/packets/errors/drops per interface) | /sys/class/net/*/statistics/ | per-iface | all |
| **C-state residency** (time in each state) | cpuidle sysfs | yes | all |
| **CPU frequency** (current scaling freq) | cpufreq sysfs | yes | chameleon, homelab, ares-comp-10 |
| **RAPL power** (watts, derived from energy_uj delta) | powercap sysfs | per-domain | chameleon, homelab (root) |
| **SMI count** | turbostat (MSR 0x34) | per-socket | chameleon, homelab (root) |
| **IPC** (instructions per cycle) | turbostat or perf | per-core | chameleon, homelab (root) |
| **PSI pressure stall** (some/full % for cpu, io, memory) | /proc/pressure/ | total | chameleon (kernel 4.20+) |

### Why these matter for clock drift modeling

- **CPU utilization** — drives heat generation, directly affects crystal temperature
- **Context switches / interrupts** — high rates indicate system load, correlate with power draw
- **C-state transitions** — entering/exiting deep sleep changes power delivery, can perturb PLL
- **Memory pressure / swap** — memory-intensive workloads stress the memory controller and DRAM power
- **Disk/Network I/O** — PCIe traffic generates heat near chipset, affects PCH temperature
- **Load average** — smoothed workload proxy, useful for slow-changing drift
- **SMI count** — system management interrupts are invisible to OS, steal cycles, and directly cause apparent clock jumps
- **IPC** — workload intensity proxy; high IPC means more power per cycle

---

## Practical Sensor Sets for Collection

### Full Collectors (chameleon, homelab) — run as root

**High-frequency (1-10 Hz):**
- All hwmon temps (sysfs reads)
- CPU frequency per-core (sysfs)
- RAPL energy counters (derive power as delta/dt)
- turbostat snapshot: actual MHz, Busy%, IPC, SMI, C-state%, PkgWatt, RAMWatt
- /proc/stat per-CPU (derive utilization %)

**Medium-frequency (0.1-1 Hz):**
- IPMI sensors (chameleon only): fans, current, voltage, power
- /proc/vmstat, /proc/meminfo (page faults, swap, memory pressure)
- /proc/interrupts, /proc/softirqs (rates)
- /proc/diskstats, network sysfs (I/O rates)
- /proc/loadavg
- C-state residency counters

### Lesser Collectors (ares, ares-comp-10) — user only, no root

**High-frequency (1-10 Hz):**
- hwmon temps via sysfs (23-25 sensors)
- CPU frequency per-core (ares-comp-10 only, 40 readings)
- /proc/stat per-CPU (derive utilization %)

**Medium-frequency (0.1-1 Hz):**
- /proc/vmstat, /proc/meminfo
- /proc/interrupts, /proc/softirqs (rates)
- /proc/diskstats, network sysfs
- /proc/loadavg
- C-state residency counters
- Thermal zones (3 readings)

---

## Final Collection Set

Definitive per-machine sensor list for the 24-hour data collection. All machines
sample at 1 Hz with every field in a single CSV row.

### Time Fields (all machines, every row, ~2 us)

| Field | Source | Purpose |
|-------|--------|---------|
| `ts_realtime_ns` | `clock_gettime(CLOCK_REALTIME)` | Wall clock timestamp |
| `ts_monotonic_ns` | `clock_gettime(CLOCK_MONOTONIC)` | NTP-disciplined clock |
| `ts_mono_raw_ns` | `clock_gettime(CLOCK_MONOTONIC_RAW)` | Raw TSC, no NTP adjustments |
| `mono_minus_raw_ns` | computed | Cumulative NTP correction = drift integral |
| `adj_freq` | `adjtimex()` via ctypes | Kernel frequency correction (PPM × 65536) |
| `adj_offset` | `adjtimex()` | Current offset being slewed |
| `adj_maxerror` | `adjtimex()` | Max estimated error |
| `adj_esterror` | `adjtimex()` | Estimated error |
| `adj_status` | `adjtimex()` | NTP status bitmask |
| `adj_tick` | `adjtimex()` | Microseconds per tick |
| `chrony_freq_ppm` | `chronyc -c tracking` | Chrony frequency estimate |
| `chrony_offset_ns` | `chronyc -c tracking` | Last offset measurement |
| `chrony_skew_ppm` | `chronyc -c tracking` | Error in frequency estimate |
| `chrony_root_delay_ns` | `chronyc -c tracking` | RTT to reference |
| `chrony_root_dispersion_ns` | `chronyc -c tracking` | Dispersion |
| `chrony_stratum` | `chronyc -c tracking` | Stratum |

### Chameleon (~142 columns, full, root, stressed)

| Category | Sensors | Count | Read Time | Relevance |
|----------|---------|-------|-----------|-----------|
| Temperatures (sysfs hwmon) | coretemp 2 sockets (13+13), PCH | 27 | ~1.5 ms | Primary predictor |
| IPMI temps (bg subprocess) | CPU×2, inlet, exhaust | 4 | bg | Primary predictor |
| IPMI power/fans (bg) | fans×6, current×2, voltage×2, power | 11 | bg | Power/airflow |
| CPU frequency (sysfs) | 48 per-core | 48 | ~2.5 ms | Heat generation proxy |
| CPU utilization (/proc/stat) | user%, sys%, idle%, iowait% aggregate | 4 | ~0.5 ms | Thermal load |
| RAPL energy (sysfs) | pkg-0, pkg-1, dram-0, dram-1 | 4 | ~0.2 ms | Power measurement |
| turbostat (bg subprocess) | Avg_MHz, Busy%, IPC, SMI, PkgWatt, RAMWatt, C-state% | 13 | bg | SMI count, power |
| Load average (/proc/loadavg) | 1min, 5min, 15min, running, blocked | 5 | ~0.1 ms | Workload proxy |
| Memory (/proc/meminfo) | free, available, cached, dirty, buffers | 5 | ~0.2 ms | DRAM power |
| C-state residency (sysfs) | 2 representative CPUs × 4 states | 8 | ~0.4 ms | Power state transitions |
| Thermal zones (sysfs) | 3 zones | 3 | ~0.2 ms | Extra temp sources |
| Disk I/O (/proc/diskstats) | sda: reads, writes, read_bytes, write_bytes | 4 | ~0.1 ms | PCH/chipset heat |
| Network I/O (sysfs) | eno1np0: rx/tx bytes, rx/tx packets | 4 | ~0.2 ms | PCH/chipset heat |
| Context switches (/proc/stat) | ctxt, total interrupts | 2 | free | Load indicator |

`ipmi_age_ms` and `turbo_age_ms` columns record staleness of background values.

### Homelab (~80 columns, full, root, NOT stressed)

| Category | Count | Notes |
|----------|-------|-------|
| Temperatures | 9 | coretemp(5), PCH, ACPI(2), NVMe |
| CPU frequency | 8 | |
| CPU utilization | 4 | |
| RAPL energy | 4 | package-0, core, uncore, dram |
| turbostat | 13 | bg subprocess |
| Load average | 5 | |
| Memory | 5 | |
| C-state residency | 14 | 2 CPUs × 7 states (C1, C1E, C3, C6, C7s, C8) |
| Thermal zones | 4 | |
| Disk I/O | 8 | nvme0n1 + sda |
| Network I/O | 4 | enp0s31f6 |
| Context switches | 2 | |
| No IPMI | | |

### Ares Master (~66 columns, lesser, no root, NOT stressed)

| Category | Count | Notes |
|----------|-------|-------|
| Temperatures | 23 | coretemp(11+11), PCH |
| CPU frequency | 0 | No cpufreq driver |
| CPU utilization | 4 | |
| Load average | 5 | |
| Memory | 5 | |
| C-state residency | 8 | 2 CPUs × 4 states |
| Thermal zones | 3 | |
| Disk I/O | 8 | sda + sdb |
| Network I/O | 8 | eno1 + ens1np0 |
| Context switches | 2 | |
| No RAPL, turbostat, IPMI | | |

### Ares-comp-10 (~116 columns, lesser, no root, stressed)

| Category | Count | Notes |
|----------|-------|-------|
| Temperatures | 25 | coretemp(~11+11), PCH, NVMe |
| CPU frequency | 40 | |
| CPU utilization | 4 | |
| Load average | 5 | |
| Memory | 5 | |
| C-state residency | 8 | 2 CPUs × ~4 states |
| Thermal zones | 3 | |
| Disk I/O | 16 | sda + sdb + sdc + nvme0n1 |
| Network I/O | 8 | eno1 + enp47s0np0 |
| Context switches | 2 | |
| No RAPL, turbostat, IPMI | | |
