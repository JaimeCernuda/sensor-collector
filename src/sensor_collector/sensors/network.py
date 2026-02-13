"""Network I/O counters from sysfs.

Reads per-interface cumulative byte and packet counters from
/sys/class/net/{iface}/statistics/.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

# Statistics files to read for each interface
_STAT_FILES: list[str] = [
    "rx_bytes",
    "tx_bytes",
    "rx_packets",
    "tx_packets",
]


class NetworkReader:
    """Read network interface I/O counters from sysfs.

    Each interface produces four columns:
      - ``net_{iface}_rx_bytes``
      - ``net_{iface}_tx_bytes``
      - ``net_{iface}_rx_packets``
      - ``net_{iface}_tx_packets``

    Values are cumulative counters as reported by the kernel.
    """

    COLUMNS: ClassVar[list[str]] = []  # populated per-instance via property

    def __init__(self, interfaces: list[str]) -> None:
        self._interfaces = interfaces
        self._columns: list[str] = []
        self._paths: list[tuple[str, Path]] = []  # (column_name, stat_path)

        for iface in self._interfaces:
            for stat in _STAT_FILES:
                col = f"net_{iface}_{stat}"
                stat_path = Path(f"/sys/class/net/{iface}/statistics/{stat}")
                self._columns.append(col)
                self._paths.append((col, stat_path))

    @property
    def columns(self) -> list[str]:
        """Return the column names for this reader instance."""
        return list(self._columns)

    @classmethod
    def discover_interfaces(
        cls,
        sysfs_root: str = "/sys/class/net",
        skip_loopback: bool = True,
        skip_virtual: bool = True,
    ) -> list[str]:
        """Discover network interfaces with statistics available in sysfs.

        Args:
            sysfs_root: Base path to the net class directory.
            skip_loopback: If True, exclude the ``lo`` interface.
            skip_virtual: If True, exclude interfaces whose sysfs entry is
                under /sys/devices/virtual/ (bridges, veth, etc.).

        Returns:
            Sorted list of interface names.
        """
        root = Path(sysfs_root)
        interfaces: list[str] = []

        if not root.is_dir():
            return interfaces

        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            iface = entry.name

            if skip_loopback and iface == "lo":
                continue

            stats_dir = entry / "statistics"
            if not stats_dir.is_dir():
                continue

            if skip_virtual:
                # Resolve the symlink to check if it points into virtual devices
                try:
                    resolved = entry.resolve()
                    if "/devices/virtual/" in str(resolved):
                        continue
                except (OSError, ValueError):
                    pass

            interfaces.append(iface)

        return sorted(interfaces)

    def read(self) -> dict[str, int | float | str]:
        """Read cumulative network I/O counters for all monitored interfaces.

        Returns:
            Dict mapping column names to cumulative counter values (int),
            or empty string if a counter could not be read.
        """
        result: dict[str, int | float | str] = {}
        for col, stat_path in self._paths:
            try:
                raw = stat_path.read_text().strip()
                result[col] = int(raw)
            except (FileNotFoundError, PermissionError, ValueError):
                result[col] = ""
        return result
