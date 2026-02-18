"""Configuration for the sensor collector."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CollectorConfig:
    """Runtime configuration for the sensor collector."""

    # Output directory for CSV and metadata files
    output_dir: Path = field(default_factory=lambda: Path.home() / "drift_data")

    # Sampling interval in seconds
    interval: float = 1.0

    # CSV flush interval (flush every N rows)
    flush_every: int = 60

    # Maximum run duration in seconds (0 = unlimited)
    duration: int = 0

    # Specific network interfaces to monitor (empty = auto-detect)
    net_interfaces: list[str] = field(default_factory=list)

    # Specific disk devices to monitor (empty = auto-detect)
    disk_devices: list[str] = field(default_factory=list)

    # Number of representative CPUs per socket for C-state sampling
    cstate_cpus_per_socket: int = 2

    # Whether to attempt root-only sensors (RAPL, turbostat, IPMI)
    try_root_sensors: bool = True

    # Peer clock probing: list of (name, host, port) tuples
    peers: list[tuple[str, str, int]] = field(default_factory=list)

    # UDP listen port for peer clock server
    listen_port: int = 19777

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
