"""CPU frequency readings from sysfs cpufreq interface.

Reads per-CPU scaling_cur_freq (KHz) and reports MHz as a float.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar


class CpufreqReader:
    """Read current CPU frequencies from sysfs.

    Each monitored CPU produces a column ``cpu{N}_freq_mhz`` with the current
    frequency in megahertz (float, KHz / 1000).
    """

    COLUMNS: ClassVar[list[str]] = []  # populated per-instance via property

    def __init__(
        self,
        cpu_indices: list[int],
        sysfs_root: str = "/sys/devices/system/cpu",
    ) -> None:
        self._cpu_indices = sorted(cpu_indices)
        root = Path(sysfs_root)
        self._paths = {
            idx: root / f"cpu{idx}" / "cpufreq" / "scaling_cur_freq"
            for idx in self._cpu_indices
        }
        self._columns = [f"cpu{idx}_freq_mhz" for idx in self._cpu_indices]

    @property
    def columns(self) -> list[str]:
        """Return the column names for this reader instance."""
        return list(self._columns)

    @classmethod
    def discover_cpufreq(cls, sysfs_root: str = "/sys/devices/system/cpu") -> list[int]:
        """Discover CPU indices that have a cpufreq/scaling_cur_freq file.

        Args:
            sysfs_root: Base path to the CPU sysfs directory.

        Returns:
            Sorted list of CPU indices with cpufreq support.
        """
        root = Path(sysfs_root)
        indices: list[int] = []

        if not root.is_dir():
            return indices

        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("cpu"):
                continue
            suffix = entry.name[3:]
            if not suffix.isdigit():
                continue
            freq_file = entry / "cpufreq" / "scaling_cur_freq"
            if freq_file.exists():
                indices.append(int(suffix))

        return sorted(indices)

    def read(self) -> dict[str, int | float | str]:
        """Read current frequencies for all monitored CPUs.

        Returns:
            Dict mapping column names to MHz float values, or empty string
            if a CPU frequency could not be read.
        """
        result: dict[str, int | float | str] = {}
        for idx, col in zip(self._cpu_indices, self._columns):
            try:
                raw = self._paths[idx].read_text().strip()
                result[col] = int(raw) / 1000.0
            except (FileNotFoundError, PermissionError, ValueError):
                result[col] = ""
        return result
