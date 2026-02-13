"""Build the CSV column list from a MachineInventory.

Constructs the ordered list of column names and instantiates all sensor
readers based on what was discovered at startup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .config import CollectorConfig
    from .discovery import MachineInventory


class SensorReader(Protocol):
    """Protocol for all sensor reader classes."""

    def read(self) -> dict[str, int | float | str]: ...


@dataclass
class SensorSchema:
    """The complete schema: ordered column names and instantiated readers."""

    columns: list[str] = field(default_factory=list)
    readers: list[SensorReader] = field(default_factory=list)
    stoppable: list[object] = field(default_factory=list)  # readers with stop()


def build_schema(inventory: MachineInventory, config: CollectorConfig) -> SensorSchema:
    """Build column list and instantiate readers from inventory."""
    from .chrony import ChronyReader
    from .clocks import ClockReader
    from .sensors.cpufreq import CpufreqReader
    from .sensors.cstate import CpuCstateInfo, CstateReader
    from .sensors.diskstats import DiskstatsReader
    from .sensors.hwmon import HwmonReader, HwmonSensor
    from .sensors.ipmi import IpmiReader
    from .sensors.network import NetworkReader
    from .sensors.procfs import ProcfsReader
    from .sensors.rapl import RaplDomain, RaplReader
    from .sensors.thermal import ThermalReader, ThermalZone
    from .sensors.turbostat import TurbostatReader

    schema = SensorSchema()

    def _add(reader: SensorReader, columns: list[str]) -> None:
        schema.columns.extend(columns)
        schema.readers.append(reader)

    # 1. Clocks (always first)
    clock_reader = ClockReader()
    _add(clock_reader, ClockReader.COLUMNS)

    # 2. Chrony (if available)
    if inventory.has_chrony:
        chrony_reader = ChronyReader()
        _add(chrony_reader, ChronyReader.COLUMNS)

    # 3. hwmon temperatures
    if inventory.hwmon_sensors:
        hwmon_sensors = [
            HwmonSensor(
                path=Path(s.input_path),
                name=s.name,
                label=s.label,
            )
            for s in inventory.hwmon_sensors
        ]
        hwmon_reader = HwmonReader(hwmon_sensors)
        _add(hwmon_reader, hwmon_reader.columns)

    # 4. Thermal zones
    if inventory.thermal_zones:
        zones = [
            ThermalZone(
                index=tz.zone_index,
                zone_type=tz.zone_type,
                temp_path=Path(tz.temp_path),
            )
            for tz in inventory.thermal_zones
        ]
        thermal_reader = ThermalReader(zones)
        _add(thermal_reader, thermal_reader.columns)

    # 5. CPU frequency
    if inventory.cpufreq_cpus:
        cpufreq_reader = CpufreqReader(inventory.cpufreq_cpus)
        _add(cpufreq_reader, cpufreq_reader.columns)

    # 6. procfs (always available on Linux)
    procfs_reader = ProcfsReader()
    _add(procfs_reader, list(ProcfsReader.COLUMNS))

    # 7. C-state residency
    if inventory.cstate_cpus:
        cstate_infos = [
            CpuCstateInfo(cpu_index=cpu.cpu_index, states=cpu.states)
            for cpu in inventory.cstate_cpus
        ]
        cstate_reader = CstateReader(cstate_infos)
        _add(cstate_reader, cstate_reader.columns)

    # 8. RAPL energy
    if inventory.rapl_domains:
        rapl_domains = [
            RaplDomain(name=d.name, energy_path=Path(d.path))
            for d in inventory.rapl_domains
        ]
        rapl_reader = RaplReader(rapl_domains)
        _add(rapl_reader, rapl_reader.columns)

    # 9. Network I/O
    if inventory.net_interfaces:
        net_reader = NetworkReader(inventory.net_interfaces)
        _add(net_reader, net_reader.columns)

    # 10. Disk I/O
    if inventory.disk_devices:
        disk_reader = DiskstatsReader(inventory.disk_devices)
        _add(disk_reader, disk_reader.columns)

    # 11. IPMI (background subprocess)
    if inventory.has_ipmitool:
        ipmi_reader = IpmiReader()
        if ipmi_reader.COLUMNS:
            _add(ipmi_reader, list(ipmi_reader.COLUMNS))
            schema.stoppable.append(ipmi_reader)

    # 12. turbostat (background subprocess)
    if inventory.has_turbostat:
        turbostat_reader = TurbostatReader()
        if turbostat_reader.COLUMNS:
            _add(turbostat_reader, list(turbostat_reader.COLUMNS))
            schema.stoppable.append(turbostat_reader)

    # 13. Peer clock (UDP probing)
    if config.peers:
        from .peer_clock import PeerClockReader, PeerConfig

        peer_configs = [
            PeerConfig(name=name, host=host, port=port)
            for name, host, port in config.peers
        ]
        peer_reader = PeerClockReader(peer_configs, config.listen_port)
        _add(peer_reader, peer_reader.columns)
        schema.stoppable.append(peer_reader)

    return schema
