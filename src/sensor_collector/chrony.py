"""Read chrony NTP tracking state via the chronyc CLI.

Parses the CSV output of ``chronyc -c tracking`` to extract clock
synchronization metrics: frequency offset, skew, stratum, root delay,
root dispersion, and last offset.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import ClassVar

# Seconds-to-nanoseconds multiplier
_S_TO_NS = 1_000_000_000.0


def parse_tracking_line(line: str) -> dict[str, int | float | str]:
    """Parse a single CSV line from ``chronyc -c tracking``.

    Args:
        line: A comma-separated line produced by ``chronyc -c tracking``.

    Returns:
        A dict with keys matching :pyattr:`ChronyReader.COLUMNS` and
        numeric values extracted from the CSV fields.  Returns empty
        strings for all columns if the line cannot be parsed.
    """
    empty: dict[str, int | float | str] = {
        "chrony_freq_ppm": "",
        "chrony_offset_ns": "",
        "chrony_skew_ppm": "",
        "chrony_root_delay_ns": "",
        "chrony_root_dispersion_ns": "",
        "chrony_stratum": "",
    }

    fields = line.strip().split(",")
    if len(fields) < 11:
        return empty

    try:
        stratum = int(fields[2])
        freq_ppm = float(fields[4])
        residual_freq_ppm = float(fields[5])  # noqa: F841 — available if needed
        skew_ppm = float(fields[6])
        root_delay_ns = float(fields[7]) * _S_TO_NS
        root_dispersion_ns = float(fields[8]) * _S_TO_NS
        last_offset_ns = float(fields[10]) * _S_TO_NS
    except (ValueError, IndexError):
        return empty

    return {
        "chrony_freq_ppm": freq_ppm,
        "chrony_offset_ns": last_offset_ns,
        "chrony_skew_ppm": skew_ppm,
        "chrony_root_delay_ns": root_delay_ns,
        "chrony_root_dispersion_ns": root_dispersion_ns,
        "chrony_stratum": stratum,
    }


class ChronyReader:
    """Synchronous reader for chrony NTP tracking data.

    Calls ``chronyc -c tracking`` on each :meth:`read` invocation and
    returns parsed results.  If chronyc is not installed or fails, all
    values are returned as empty strings.
    """

    COLUMNS: ClassVar[list[str]] = [
        "chrony_freq_ppm",
        "chrony_offset_ns",
        "chrony_skew_ppm",
        "chrony_root_delay_ns",
        "chrony_root_dispersion_ns",
        "chrony_stratum",
    ]

    @classmethod
    def is_available(cls) -> bool:
        """Return True if ``chronyc`` is found on PATH."""
        return shutil.which("chronyc") is not None

    def read(self) -> dict[str, int | float | str]:
        """Run chronyc and return parsed tracking metrics.

        Returns:
            A dict with keys from :pyattr:`COLUMNS`.  Values are numeric
            on success, or empty strings if the command fails.
        """
        empty: dict[str, int | float | str] = {col: "" for col in self.COLUMNS}

        try:
            result = subprocess.run(
                ["chronyc", "-c", "tracking"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return empty

        if result.returncode != 0:
            return empty

        line = result.stdout.strip()
        if not line:
            return empty

        return parse_tracking_line(line)
