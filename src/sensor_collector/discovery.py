"""Auto-detect available sensors and capabilities at startup.

Walks sysfs, checks for tools, and builds an inventory of what this
machine can provide. Used by schema.py to construct the CSV column list.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import CollectorConfig


@dataclass
class HwmonEntry:
    """A discovered hwmon temperature sensor."""

    hwmon_path: str  # e.g., /sys/class/hwmon/hwmon3
    name: str  # e.g., coretemp
    index: int  # e.g., 1 from temp1_input
    label: str  # e.g., "Package id 0" or "temp1"
    input_path: str  # full path to temp*_input


@dataclass
class ThermalZoneEntry:
    """A discovered thermal zone."""

    zone_index: int
    zone_type: str  # e.g., "x86_pkg_temp"
    temp_path: str  # /sys/class/thermal/thermal_zone{N}/temp


@dataclass
class CpuCstateEntry:
    """C-states available on a specific CPU."""

    cpu_index: int
    states: list[tuple[int, str]]  # [(state_index, state_name), ...]


@dataclass
class RaplDomainEntry:
    """A discovered RAPL energy domain."""

    path: str  # path to energy_uj
    name: str  # e.g., "package-0", "dram"


@dataclass
class MachineInventory:
    """Everything discovered about this machine's sensor capabilities."""

    # Hardware temperatures
    hwmon_sensors: list[HwmonEntry] = field(default_factory=list)

    # Thermal zones
    thermal_zones: list[ThermalZoneEntry] = field(default_factory=list)

    # CPUs with cpufreq support
    cpufreq_cpus: list[int] = field(default_factory=list)

    # C-state info per representative CPU
    cstate_cpus: list[CpuCstateEntry] = field(default_factory=list)

    # RAPL energy domains (empty if not root)
    rapl_domains: list[RaplDomainEntry] = field(default_factory=list)

    # Network interfaces to monitor
    net_interfaces: list[str] = field(default_factory=list)

    # Disk devices to monitor
    disk_devices: list[str] = field(default_factory=list)

    # Tool availability
    has_chrony: bool = False
    has_ipmitool: bool = False
    has_turbostat: bool = False
    is_root: bool = False


def _discover_hwmon() -> list[HwmonEntry]:
    """Walk /sys/class/hwmon/ and find all temperature sensors."""
    entries: list[HwmonEntry] = []
    hwmon_base = Path("/sys/class/hwmon")
    if not hwmon_base.exists():
        return entries

    for hwmon_dir in sorted(hwmon_base.iterdir()):
        name_path = hwmon_dir / "name"
        if not name_path.exists():
            continue
        try:
            name = name_path.read_text().strip()
        except OSError:
            continue

        # Find all temp*_input files
        for f in sorted(hwmon_dir.iterdir()):
            fname = f.name
            if fname.startswith("temp") and fname.endswith("_input"):
                idx_str = fname[4:-6]  # extract N from tempN_input
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue

                # Try to read label
                label_path = hwmon_dir / f"temp{idx}_label"
                if label_path.exists():
                    try:
                        label = label_path.read_text().strip()
                    except OSError:
                        label = f"temp{idx}"
                else:
                    label = f"temp{idx}"

                entries.append(
                    HwmonEntry(
                        hwmon_path=str(hwmon_dir),
                        name=name,
                        index=idx,
                        label=label,
                        input_path=str(f),
                    )
                )
    return entries


def _discover_thermal_zones() -> list[ThermalZoneEntry]:
    """Find all thermal zones in /sys/class/thermal/."""
    entries: list[ThermalZoneEntry] = []
    tz_base = Path("/sys/class/thermal")
    if not tz_base.exists():
        return entries

    for tz_dir in sorted(tz_base.iterdir()):
        if not tz_dir.name.startswith("thermal_zone"):
            continue
        try:
            zone_idx = int(tz_dir.name[len("thermal_zone") :])
        except ValueError:
            continue

        type_path = tz_dir / "type"
        temp_path = tz_dir / "temp"
        if not temp_path.exists():
            continue

        try:
            if type_path.exists():
                zone_type = type_path.read_text().strip()
            else:
                zone_type = f"zone{zone_idx}"
        except OSError:
            zone_type = f"zone{zone_idx}"

        entries.append(
            ThermalZoneEntry(
                zone_index=zone_idx,
                zone_type=zone_type,
                temp_path=str(temp_path),
            )
        )
    return entries


def _discover_cpufreq() -> list[int]:
    """Find CPUs with cpufreq scaling_cur_freq."""
    cpus: list[int] = []
    cpu_base = Path("/sys/devices/system/cpu")
    if not cpu_base.exists():
        return cpus

    for cpu_dir in sorted(cpu_base.iterdir()):
        if not cpu_dir.name.startswith("cpu"):
            continue
        cpu_str = cpu_dir.name[3:]
        try:
            cpu_idx = int(cpu_str)
        except ValueError:
            continue
        freq_path = cpu_dir / "cpufreq" / "scaling_cur_freq"
        if freq_path.exists():
            cpus.append(cpu_idx)
    return cpus


def _discover_cstates(cpus_per_socket: int) -> list[CpuCstateEntry]:
    """Find C-states on representative CPUs (first N per socket).

    We pick the first `cpus_per_socket` CPUs from each physical package
    to keep column count manageable.
    """
    cpu_base = Path("/sys/devices/system/cpu")
    if not cpu_base.exists():
        return []

    # Group CPUs by physical package
    packages: dict[int, list[int]] = {}
    for cpu_dir in sorted(cpu_base.iterdir()):
        if not cpu_dir.name.startswith("cpu"):
            continue
        cpu_str = cpu_dir.name[3:]
        try:
            cpu_idx = int(cpu_str)
        except ValueError:
            continue

        cpuidle_dir = cpu_dir / "cpuidle"
        if not cpuidle_dir.exists():
            continue

        # Determine package
        topo_path = cpu_dir / "topology" / "physical_package_id"
        try:
            pkg = int(topo_path.read_text().strip()) if topo_path.exists() else 0
        except (OSError, ValueError):
            pkg = 0

        packages.setdefault(pkg, []).append(cpu_idx)

    entries: list[CpuCstateEntry] = []
    for _pkg_id in sorted(packages):
        cpus = sorted(packages[_pkg_id])[:cpus_per_socket]
        for cpu_idx in cpus:
            cpuidle_dir = cpu_base / f"cpu{cpu_idx}" / "cpuidle"
            states: list[tuple[int, str]] = []
            for state_dir in sorted(cpuidle_dir.iterdir()):
                if not state_dir.name.startswith("state"):
                    continue
                try:
                    state_idx = int(state_dir.name[5:])
                except ValueError:
                    continue
                name_path = state_dir / "name"
                try:
                    if name_path.exists():
                        state_name = name_path.read_text().strip()
                    else:
                        state_name = f"state{state_idx}"
                except OSError:
                    state_name = f"state{state_idx}"
                states.append((state_idx, state_name))

            if states:
                entries.append(CpuCstateEntry(cpu_index=cpu_idx, states=states))

    return entries


def _discover_rapl() -> list[RaplDomainEntry]:
    """Find RAPL energy domains. Returns empty if not root."""
    entries: list[RaplDomainEntry] = []
    rapl_base = Path("/sys/class/powercap")
    if not rapl_base.exists():
        return entries

    for domain_dir in sorted(rapl_base.iterdir()):
        if not domain_dir.name.startswith("intel-rapl:"):
            continue

        energy_path = domain_dir / "energy_uj"
        name_path = domain_dir / "name"

        if not energy_path.exists():
            continue

        # Test read access
        try:
            energy_path.read_text()
        except PermissionError:
            continue

        try:
            if name_path.exists():
                name = name_path.read_text().strip()
            else:
                name = domain_dir.name
        except OSError:
            name = domain_dir.name

        entries.append(RaplDomainEntry(path=str(energy_path), name=name))

        # Check subdomains (e.g., intel-rapl:0:0 for dram under package-0)
        for sub_dir in sorted(domain_dir.iterdir()):
            if not sub_dir.is_dir():
                continue
            if not sub_dir.name.startswith("intel-rapl:"):
                continue
            sub_energy = sub_dir / "energy_uj"
            sub_name_path = sub_dir / "name"
            if not sub_energy.exists():
                continue
            try:
                sub_energy.read_text()
            except PermissionError:
                continue
            try:
                if sub_name_path.exists():
                    sub_name = sub_name_path.read_text().strip()
                else:
                    sub_name = sub_dir.name
            except OSError:
                sub_name = sub_dir.name
            entries.append(RaplDomainEntry(path=str(sub_energy), name=sub_name))

    return entries


def _discover_net_interfaces(config_interfaces: list[str]) -> list[str]:
    """Find network interfaces to monitor.

    If config specifies interfaces, use those. Otherwise auto-detect
    physical interfaces (exclude lo, docker, veth, virbr).
    """
    if config_interfaces:
        return config_interfaces

    net_base = Path("/sys/class/net")
    if not net_base.exists():
        return []

    skip_prefixes = ("lo", "docker", "veth", "virbr", "br-")
    interfaces: list[str] = []
    for iface_dir in sorted(net_base.iterdir()):
        name = iface_dir.name
        if any(name.startswith(p) for p in skip_prefixes):
            continue
        # Check it has statistics
        if (iface_dir / "statistics" / "rx_bytes").exists():
            interfaces.append(name)
    return interfaces


def _discover_disk_devices(config_devices: list[str]) -> list[str]:
    """Find disk devices to monitor from /proc/diskstats.

    If config specifies devices, use those. Otherwise auto-detect
    whole-disk devices (sd*, nvme*n*, no partitions).
    """
    if config_devices:
        return config_devices

    diskstats = Path("/proc/diskstats")
    if not diskstats.exists():
        return []

    devices: list[str] = []
    try:
        for line in diskstats.read_text().splitlines():
            parts = line.split()
            if len(parts) < 14:
                continue
            dev = parts[2]
            # Include whole disks: sd[a-z], nvme[0-9]n[0-9]
            # Exclude partitions: sda1, nvme0n1p1
            is_whole_sd = dev.startswith("sd") and len(dev) == 3
            is_whole_nvme = dev.startswith("nvme") and "p" not in dev
            if is_whole_sd or is_whole_nvme:
                devices.append(dev)
    except OSError:
        pass
    return devices


def discover_machine(config: CollectorConfig) -> MachineInventory:
    """Run full discovery and return a MachineInventory."""
    inv = MachineInventory()
    inv.is_root = os.geteuid() == 0

    inv.hwmon_sensors = _discover_hwmon()
    inv.thermal_zones = _discover_thermal_zones()
    inv.cpufreq_cpus = _discover_cpufreq()
    inv.cstate_cpus = _discover_cstates(config.cstate_cpus_per_socket)
    inv.net_interfaces = _discover_net_interfaces(config.net_interfaces)
    inv.disk_devices = _discover_disk_devices(config.disk_devices)

    if config.try_root_sensors:
        inv.rapl_domains = _discover_rapl()

    inv.has_chrony = shutil.which("chronyc") is not None

    if config.try_root_sensors:
        inv.has_ipmitool = shutil.which("ipmitool") is not None
        inv.has_turbostat = shutil.which("turbostat") is not None
    else:
        inv.has_ipmitool = False
        inv.has_turbostat = False

    return inv


def print_inventory(inv: MachineInventory) -> None:
    """Print a human-readable summary of the discovered inventory."""
    print("=== Machine Inventory ===")
    print(f"  Root: {inv.is_root}")
    print(f"  hwmon sensors: {len(inv.hwmon_sensors)}")
    print(f"  Thermal zones: {len(inv.thermal_zones)}")
    print(f"  cpufreq CPUs: {len(inv.cpufreq_cpus)}")
    print(f"  C-state CPUs: {len(inv.cstate_cpus)}")
    print(f"  RAPL domains: {len(inv.rapl_domains)}")
    print(f"  Network interfaces: {inv.net_interfaces}")
    print(f"  Disk devices: {inv.disk_devices}")
    print(f"  Chrony: {inv.has_chrony}")
    print(f"  IPMI: {inv.has_ipmitool}")
    print(f"  Turbostat: {inv.has_turbostat}")
