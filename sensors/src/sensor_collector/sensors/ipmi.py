"""Background IPMI sensor reader using ``ipmitool``.

Runs ``ipmitool sdr elist`` in a background thread, parses temperature,
fan, current, voltage, and power sensors, and caches the latest readings
for low-latency ``read()`` calls.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import threading
import time
from typing import ClassVar

log = logging.getLogger(__name__)

# Refresh interval for the background ipmitool poll.
_POLL_INTERVAL_S = 5.0

# Timeout for a single ipmitool invocation.
_CMD_TIMEOUT_S = 10.0


def _sanitize_name(name: str) -> str:
    """Convert a raw sensor name to a column-safe identifier.

    Lowercases, strips leading/trailing whitespace, and replaces
    non-alphanumeric characters with underscores.  Collapses runs of
    underscores and strips trailing underscores.
    """
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def _parse_elist_line(
    line: str,
) -> tuple[str, str, float | None]:
    """Parse one line of ``ipmitool sdr elist`` output.

    Expected format::

        Sensor Name      | hex | ok  | entity | 42 degrees C

    Returns:
        A tuple of *(sanitized_name, unit, numeric_value)*.
        *unit* is one of ``"temp"``, ``"fan"``, ``"current"``,
        ``"voltage"``, ``"power"``, or ``""`` for unrecognised lines.
        *numeric_value* is ``None`` when parsing fails.
    """
    parts = [p.strip() for p in line.split("|")]
    if len(parts) < 5:
        return ("", "", None)

    raw_name = parts[0]
    status = parts[2].lower()
    value_field = parts[4].lower()

    # Skip sensors that are not readable.
    if status in ("ns", "na", "nr"):
        return ("", "", None)

    sanitized = _sanitize_name(raw_name)
    if not sanitized:
        return ("", "", None)

    # Try to extract a numeric value and determine the unit.
    # ipmitool value fields look like: "42 degrees C", "3600 RPM",
    # "1.200 Volts", "0.312 Amps", "140 Watts"
    unit = ""
    numeric: float | None = None

    if "degrees c" in value_field:
        unit = "temp"
    elif "rpm" in value_field:
        unit = "fan"
    elif "amps" in value_field:
        unit = "current"
    elif "volts" in value_field:
        unit = "voltage"
    elif "watts" in value_field:
        unit = "power"
    else:
        return (sanitized, "", None)

    # Extract the first numeric token.
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value_field)
    if match:
        try:
            numeric = float(match.group())
        except ValueError:
            numeric = None

    return (sanitized, unit, numeric)


def _run_ipmitool_elist() -> str:
    """Run ``ipmitool sdr elist`` and return stdout.

    Raises on failure so callers can handle gracefully.
    """
    result = subprocess.run(
        ["ipmitool", "sdr", "elist"],
        capture_output=True,
        text=True,
        timeout=_CMD_TIMEOUT_S,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ipmitool exited with code {result.returncode}")
    return result.stdout


def _parse_elist_output(output: str) -> dict[str, float]:
    """Parse full ``ipmitool sdr elist`` output into a flat dict.

    Keys are ``ipmi_<sanitized_name>`` and values are floats.
    """
    readings: dict[str, float] = {}
    for line in output.splitlines():
        sanitized, unit, value = _parse_elist_line(line)
        if not sanitized or not unit or value is None:
            continue
        col = f"ipmi_{sanitized}"
        readings[col] = value
    return readings


class IpmiReader:
    """Background-threaded IPMI sensor reader.

    Starts a daemon thread that periodically polls ``ipmitool sdr elist``
    and caches parsed sensor values.  Call :meth:`read` to get the latest
    cached values.

    If ``ipmitool`` is not installed or not accessible, :pyattr:`COLUMNS`
    will be empty and :meth:`read` will return ``{}``.
    """

    COLUMNS: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: dict[str, float] = {}
        self._last_ok: float = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Attempt a discovery pass to learn which sensors exist.
        if not self.is_available():
            return

        try:
            output = _run_ipmitool_elist()
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, RuntimeError):
            return

        discovery = _parse_elist_output(output)
        if not discovery:
            return

        # Set COLUMNS on the *instance* (shadows the class attribute).
        columns = [*sorted(discovery.keys()), "ipmi_age_ms"]
        # We mutate the class variable so other code that inspects
        # IpmiReader.COLUMNS before instantiation can still see the
        # discovered set.  This is intentional for this dynamic sensor.
        self.__class__.COLUMNS = columns  # type: ignore[misc]

        self._latest = discovery
        self._last_ok = time.monotonic()

        # Start the background poller.
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="ipmi-reader",
            daemon=True,
        )
        self._thread.start()

    @classmethod
    def is_available(cls) -> bool:
        """Return True if ``ipmitool`` is found on PATH."""
        return shutil.which("ipmitool") is not None

    # ------------------------------------------------------------------
    # Background poller
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """Periodically run ipmitool and cache results."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=_POLL_INTERVAL_S)
            if self._stop_event.is_set():
                break
            try:
                output = _run_ipmitool_elist()
                readings = _parse_elist_output(output)
                with self._lock:
                    self._latest = readings
                    self._last_ok = time.monotonic()
            except Exception:
                log.debug("ipmitool poll failed", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> dict[str, int | float | str]:
        """Return the latest cached IPMI sensor values.

        Returns:
            A dict with keys from :pyattr:`COLUMNS`.  Numeric values
            from the last successful poll, or empty strings if no data
            is available.  Always includes ``ipmi_age_ms``.
        """
        if not self.COLUMNS:
            return {}

        with self._lock:
            latest = dict(self._latest)
            last_ok = self._last_ok

        if last_ok == 0.0:
            return {col: "" for col in self.COLUMNS}

        age_ms = (time.monotonic() - last_ok) * 1000.0

        result: dict[str, int | float | str] = {}
        for col in self.COLUMNS:
            if col == "ipmi_age_ms":
                result[col] = round(age_ms, 1)
            elif col in latest:
                result[col] = latest[col]
            else:
                result[col] = ""

        return result

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5.0)
