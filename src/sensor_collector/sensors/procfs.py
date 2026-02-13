"""System metrics from /proc/stat, /proc/meminfo, and /proc/loadavg.

Provides CPU utilization percentages (delta-based), context switches,
interrupts, memory stats, and load averages -- all from procfs with no
external dependencies.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import ClassVar


class ProcfsReader:
    """Read aggregate system metrics from procfs.

    CPU utilization is computed as delta percentages between successive calls
    to ``read()``.  The first call returns zeros for the CPU percentage fields
    because there is no previous sample to diff against.

    Context switches and interrupts are reported as absolute cumulative
    counters.
    """

    COLUMNS: ClassVar[list[str]] = [
        "cpu_user_pct",
        "cpu_sys_pct",
        "cpu_idle_pct",
        "cpu_iowait_pct",
        "ctxt",
        "intr",
        "mem_free_kb",
        "mem_available_kb",
        "mem_cached_kb",
        "mem_dirty_kb",
        "mem_buffers_kb",
        "loadavg_1m",
        "loadavg_5m",
        "loadavg_15m",
        "procs_running",
        "procs_blocked",
    ]

    def __init__(
        self,
        proc_root: str = "/proc",
    ) -> None:
        self._stat_path = Path(proc_root) / "stat"
        self._meminfo_path = Path(proc_root) / "meminfo"
        self._loadavg_path = Path(proc_root) / "loadavg"

        # Previous CPU jiffies for delta calculation (None = first read)
        self._prev_cpu: (
            tuple[int, int, int, int, int, int, int, int, int, int] | None
        ) = None

    @property
    def columns(self) -> list[str]:
        """Return the column names for this reader."""
        return list(self.COLUMNS)

    # ------------------------------------------------------------------
    # /proc/stat helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cpu_line(
        line: str,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]:
        """Parse the aggregate ``cpu`` line from /proc/stat.

        Fields (all in jiffies):
            user, nice, system, idle, iowait, irq, softirq, steal, guest,
            guest_nice

        Returns:
            Tuple of 10 ints.
        """
        parts = line.split()
        # parts[0] == "cpu"; parts[1:] are the counters
        values = [int(p) for p in parts[1:11]]
        # Pad with zeros if the kernel exposes fewer fields
        while len(values) < 10:
            values.append(0)
        return (
            values[0],
            values[1],
            values[2],
            values[3],
            values[4],
            values[5],
            values[6],
            values[7],
            values[8],
            values[9],
        )

    def _read_stat(
        self,
    ) -> dict[str, int | float | str]:
        """Read /proc/stat and return CPU percentages, ctxt, and intr."""
        result: dict[str, int | float | str] = {
            "cpu_user_pct": 0.0,
            "cpu_sys_pct": 0.0,
            "cpu_idle_pct": 0.0,
            "cpu_iowait_pct": 0.0,
            "ctxt": "",
            "intr": "",
        }

        try:
            text = self._stat_path.read_text()
        except (FileNotFoundError, PermissionError):
            self._prev_cpu = None
            return result

        cpu_line: str | None = None
        ctxt_val: int | None = None
        intr_val: int | None = None

        for line in text.splitlines():
            if line.startswith("cpu "):
                cpu_line = line
            elif line.startswith("ctxt "):
                with contextlib.suppress(IndexError, ValueError):
                    ctxt_val = int(line.split()[1])
            elif line.startswith("intr "):
                # First field after "intr" is the total interrupt count
                with contextlib.suppress(IndexError, ValueError):
                    intr_val = int(line.split()[1])

        # Context switches and interrupts (cumulative counters)
        if ctxt_val is not None:
            result["ctxt"] = ctxt_val
        if intr_val is not None:
            result["intr"] = intr_val

        # CPU utilization deltas
        if cpu_line is not None:
            cur = self._parse_cpu_line(cpu_line)
            if self._prev_cpu is not None:
                prev = self._prev_cpu
                d_user = (cur[0] + cur[1]) - (prev[0] + prev[1])  # user + nice
                d_sys = (cur[2] + cur[5] + cur[6] + cur[7]) - (
                    prev[2] + prev[5] + prev[6] + prev[7]
                )  # system + irq + softirq + steal
                d_idle = cur[3] - prev[3]
                d_iowait = cur[4] - prev[4]
                d_total = sum(cur) - sum(prev)

                if d_total > 0:
                    result["cpu_user_pct"] = round(d_user / d_total * 100.0, 2)
                    result["cpu_sys_pct"] = round(d_sys / d_total * 100.0, 2)
                    result["cpu_idle_pct"] = round(d_idle / d_total * 100.0, 2)
                    result["cpu_iowait_pct"] = round(d_iowait / d_total * 100.0, 2)

            self._prev_cpu = cur

        return result

    # ------------------------------------------------------------------
    # /proc/meminfo helpers
    # ------------------------------------------------------------------

    _MEMINFO_KEYS: ClassVar[dict[str, str]] = {
        "MemFree:": "mem_free_kb",
        "MemAvailable:": "mem_available_kb",
        "Cached:": "mem_cached_kb",
        "Dirty:": "mem_dirty_kb",
        "Buffers:": "mem_buffers_kb",
    }

    def _read_meminfo(self) -> dict[str, int | float | str]:
        """Parse /proc/meminfo for selected memory fields (kB)."""
        result: dict[str, int | float | str] = {
            col: "" for col in self._MEMINFO_KEYS.values()
        }

        try:
            text = self._meminfo_path.read_text()
        except (FileNotFoundError, PermissionError):
            return result

        for line in text.splitlines():
            parts = line.split()
            if not parts:
                continue
            key = parts[0]
            if key in self._MEMINFO_KEYS:
                with contextlib.suppress(IndexError, ValueError):
                    result[self._MEMINFO_KEYS[key]] = int(parts[1])

        return result

    # ------------------------------------------------------------------
    # /proc/loadavg helpers
    # ------------------------------------------------------------------

    def _read_loadavg(self) -> dict[str, int | float | str]:
        """Parse /proc/loadavg for load averages and process counts."""
        result: dict[str, int | float | str] = {
            "loadavg_1m": "",
            "loadavg_5m": "",
            "loadavg_15m": "",
            "procs_running": "",
            "procs_blocked": "",
        }

        try:
            text = self._loadavg_path.read_text().strip()
        except (FileNotFoundError, PermissionError):
            return result

        # Format: "0.08 0.03 0.01 1/234 5678"
        parts = text.split()
        try:
            result["loadavg_1m"] = float(parts[0])
            result["loadavg_5m"] = float(parts[1])
            result["loadavg_15m"] = float(parts[2])
        except (IndexError, ValueError):
            pass

        # Running/total from the "running/total" field
        try:
            running_total = parts[3].split("/")
            result["procs_running"] = int(running_total[0])
        except (IndexError, ValueError):
            pass

        # procs_blocked is not in /proc/loadavg; read from /proc/stat
        try:
            stat_text = self._stat_path.read_text()
            for line in stat_text.splitlines():
                if line.startswith("procs_blocked "):
                    result["procs_blocked"] = int(line.split()[1])
                    break
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            pass

        return result

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def read(self) -> dict[str, int | float | str]:
        """Read all procfs metrics and return a flat dict matching COLUMNS.

        Returns:
            Dict with keys from COLUMNS.  Unavailable values are empty strings.
        """
        data: dict[str, int | float | str] = {}
        data.update(self._read_stat())
        data.update(self._read_meminfo())
        data.update(self._read_loadavg())
        return data
