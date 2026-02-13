"""Background turbostat reader.

Runs ``turbostat`` as a long-lived subprocess, parses its periodic
summary output, and caches the latest row for low-latency ``read()``
calls.
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import threading
import time
from typing import ClassVar

log = logging.getLogger(__name__)

# Mapping from turbostat column header to our output column name.
_HEADER_MAP: dict[str, str] = {
    "Avg_MHz": "turbo_avg_mhz",
    "Busy%": "turbo_busy_pct",
    "Bzy_MHz": "turbo_bzy_mhz",
    "TSC_MHz": "turbo_tsc_mhz",
    "IPC": "turbo_ipc",
    "IRQ": "turbo_irq",
    "SMI": "turbo_smi",
    "CoreTmp": "turbo_core_tmp",
    "PkgTmp": "turbo_pkg_tmp",
    "Pkg%pc2": "turbo_pkg_pc2_pct",
    "Pkg%pc6": "turbo_pkg_pc6_pct",
    "PkgWatt": "turbo_pkg_watt",
    "RAMWatt": "turbo_ram_watt",
}


def parse_turbostat_line(
    header: list[str],
    values: list[str],
) -> dict[str, float | str]:
    """Parse a single turbostat summary data line.

    Args:
        header: List of column names from turbostat's header row.
        values: List of whitespace-split values from a data row.

    Returns:
        A dict mapping our canonical column names (``turbo_*``) to
        numeric values.  Columns that cannot be parsed are set to
        empty strings.
    """
    result: dict[str, float | str] = {v: "" for v in _HEADER_MAP.values()}

    for idx, col_name in enumerate(header):
        mapped = _HEADER_MAP.get(col_name)
        if mapped is None:
            continue
        if idx >= len(values):
            continue
        raw = values[idx].strip()
        if not raw or raw == "-":
            result[mapped] = ""
            continue
        try:
            result[mapped] = float(raw)
        except ValueError:
            result[mapped] = ""

    return result


class TurbostatReader:
    """Background reader for ``turbostat`` summary output.

    Launches turbostat as a long-running subprocess and reads its
    periodic summary lines in a daemon thread.  The latest parsed
    row is cached for fast :meth:`read` access.

    If turbostat is not installed or the user lacks permission, the
    reader gracefully degrades: :pyattr:`COLUMNS` remains populated
    but :meth:`read` returns empty strings.
    """

    COLUMNS: ClassVar[list[str]] = [
        "turbo_avg_mhz",
        "turbo_busy_pct",
        "turbo_bzy_mhz",
        "turbo_tsc_mhz",
        "turbo_ipc",
        "turbo_irq",
        "turbo_smi",
        "turbo_core_tmp",
        "turbo_pkg_tmp",
        "turbo_pkg_pc2_pct",
        "turbo_pkg_pc6_pct",
        "turbo_pkg_watt",
        "turbo_ram_watt",
        "turbo_age_ms",
    ]

    _TURBOSTAT_CMD: ClassVar[list[str]] = [
        "turbostat",
        "--interval",
        "2",
        "--num_iterations",
        "0",
        "--show",
        "Avg_MHz,Busy%,Bzy_MHz,TSC_MHz,IPC,IRQ,SMI,CoreTmp,PkgTmp,"
        "Pkg%pc2,Pkg%pc6,PkgWatt,RAMWatt",
        "--Summary",
    ]

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, float | str] = {}
        self._last_ok: float = 0.0
        self._stop_event = threading.Event()
        self._process: subprocess.Popen[str] | None = None
        self._thread: threading.Thread | None = None

        if not self.is_available():
            return

        try:
            self._start()
        except (FileNotFoundError, PermissionError, OSError) as exc:
            log.debug("Failed to start turbostat: %s", exc)

    @classmethod
    def is_available(cls) -> bool:
        """Return True if ``turbostat`` is found on PATH."""
        return shutil.which("turbostat") is not None

    # ------------------------------------------------------------------
    # Subprocess management
    # ------------------------------------------------------------------

    def _start(self) -> None:
        """Launch turbostat and start the reader thread."""
        self._process = subprocess.Popen(
            self._TURBOSTAT_CMD,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="turbostat-reader",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        """Read turbostat stdout line-by-line in a background thread."""
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        header: list[str] | None = None

        try:
            for raw_line in proc.stdout:
                if self._stop_event.is_set():
                    break

                line = raw_line.strip()
                if not line:
                    # A blank line precedes a fresh header + data pair in
                    # turbostat's repeating output.  Reset the header so
                    # the next non-blank line is treated as a new header.
                    header = None
                    continue

                tokens = line.split()

                if header is None:
                    # First non-blank line after a blank (or at start)
                    # is the header row.
                    header = tokens
                    continue

                # Data row — parse it.
                parsed = parse_turbostat_line(header, tokens)
                now = time.monotonic()
                with self._lock:
                    self._latest = parsed
                    self._last_ok = now

                # After consuming one data row, expect the next block
                # starts with a blank line followed by a new header.
                # Some turbostat versions re-print the header every
                # iteration; others don't.  We reset header to None
                # only on blank lines (handled above).
        except Exception:
            log.debug("turbostat reader loop crashed", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> dict[str, int | float | str]:
        """Return the latest turbostat summary values.

        Returns:
            A dict with keys from :pyattr:`COLUMNS`.  Numeric values
            from the last parsed row, empty strings if no data is
            available yet.  Always includes ``turbo_age_ms``.
        """
        with self._lock:
            latest = dict(self._latest)
            last_ok = self._last_ok

        if last_ok == 0.0:
            result: dict[str, int | float | str] = {col: "" for col in self.COLUMNS}
            return result

        age_ms = (time.monotonic() - last_ok) * 1000.0

        result = {}
        for col in self.COLUMNS:
            if col == "turbo_age_ms":
                result[col] = round(age_ms, 1)
            elif col in latest:
                result[col] = latest[col]
            else:
                result[col] = ""

        return result

    def stop(self) -> None:
        """Stop the turbostat subprocess and reader thread."""
        self._stop_event.set()

        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5.0)
            except (OSError, subprocess.TimeoutExpired):
                with contextlib.suppress(OSError):
                    self._process.kill()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
