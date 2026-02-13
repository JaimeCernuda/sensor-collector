"""CPU C-state residency counters from sysfs cpuidle interface.

Reads per-CPU idle state names and cumulative time-in-state (microseconds)
from /sys/devices/system/cpu/cpu{N}/cpuidle/state{S}/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class CpuCstateInfo:
    """Discovered C-state layout for a single CPU."""

    cpu_index: int
    states: list[tuple[int, str]]  # [(state_index, state_name), ...]


class CstateReader:
    """Read CPU C-state residency counters from sysfs.

    Each (cpu, state) pair produces a column ``cstate_cpu{N}_{state_name}_us``
    with the cumulative microseconds spent in that state.
    """

    COLUMNS: ClassVar[list[str]] = []  # populated per-instance via property

    def __init__(
        self,
        cpu_cstates: list[CpuCstateInfo],
        sysfs_root: str = "/sys/devices/system/cpu",
    ) -> None:
        self._cpu_cstates = cpu_cstates
        self._columns: list[str] = []
        self._paths: list[tuple[str, Path]] = []  # (column_name, time_path)
        root = Path(sysfs_root)

        for info in self._cpu_cstates:
            for state_idx, state_name in info.states:
                col = f"cstate_cpu{info.cpu_index}_{state_name}_us"
                time_path = (
                    root
                    / f"cpu{info.cpu_index}"
                    / "cpuidle"
                    / f"state{state_idx}"
                    / "time"
                )
                self._columns.append(col)
                self._paths.append((col, time_path))

    @property
    def columns(self) -> list[str]:
        """Return the column names for this reader instance."""
        return list(self._columns)

    @classmethod
    def discover_cstates(
        cls, sysfs_root: str = "/sys/devices/system/cpu"
    ) -> list[CpuCstateInfo]:
        """Discover available C-states for each CPU.

        Args:
            sysfs_root: Base path to the CPU sysfs directory.

        Returns:
            List of CpuCstateInfo, one per CPU that has cpuidle states.
        """
        root = Path(sysfs_root)
        results: list[CpuCstateInfo] = []

        if not root.is_dir():
            return results

        for cpu_dir in sorted(root.iterdir()):
            if not cpu_dir.is_dir() or not cpu_dir.name.startswith("cpu"):
                continue
            suffix = cpu_dir.name[3:]
            if not suffix.isdigit():
                continue
            cpu_index = int(suffix)

            cpuidle_dir = cpu_dir / "cpuidle"
            if not cpuidle_dir.is_dir():
                continue

            states: list[tuple[int, str]] = []
            for state_dir in sorted(cpuidle_dir.iterdir()):
                if not state_dir.is_dir() or not state_dir.name.startswith("state"):
                    continue
                state_suffix = state_dir.name[5:]
                if not state_suffix.isdigit():
                    continue
                state_index = int(state_suffix)

                name_file = state_dir / "name"
                try:
                    state_name = name_file.read_text().strip()
                except (FileNotFoundError, PermissionError):
                    state_name = f"state{state_index}"

                # Sanitise for column name
                state_name = state_name.replace(" ", "_").replace("-", "_").lower()

                time_file = state_dir / "time"
                if not time_file.exists():
                    continue

                states.append((state_index, state_name))

            if states:
                results.append(CpuCstateInfo(cpu_index=cpu_index, states=states))

        return results

    def read(self) -> dict[str, int | float | str]:
        """Read cumulative C-state residency counters.

        Returns:
            Dict mapping column names to cumulative microsecond values
            (int), or empty string if a counter could not be read.
        """
        result: dict[str, int | float | str] = {}
        for col, time_path in self._paths:
            try:
                raw = time_path.read_text().strip()
                result[col] = int(raw)
            except (FileNotFoundError, PermissionError, ValueError):
                result[col] = ""
        return result
