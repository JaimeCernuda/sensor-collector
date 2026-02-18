"""Disk I/O counters from /proc/diskstats.

Parses /proc/diskstats for specific block device names, extracting
cumulative counters for reads, writes, and sectors transferred.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

# Field indices within a /proc/diskstats line (0-indexed after the device name).
# The line format is:
#   major minor name  rd_ios rd_merges rd_sectors rd_ticks
#                      wr_ios wr_merges wr_sectors wr_ticks
#                      ios_in_progress io_ticks weighted_ticks
#
# After splitting, the device name is at index 2.  Fields after the name:
#   [0] reads completed   (rd_ios)
#   [1] reads merged      (rd_merges)
#   [2] sectors read      (rd_sectors)
#   [3] read ticks        (rd_ticks)
#   [4] writes completed  (wr_ios)
#   [5] writes merged     (wr_merges)
#   [6] sectors written   (wr_sectors)
#   [7] write ticks       (wr_ticks)
#
# We want fields 0, 4, 2, 6 (reads, writes, sectors_read, sectors_written).
_FIELD_READS = 0
_FIELD_WRITES = 4
_FIELD_READ_SECTORS = 2
_FIELD_WRITE_SECTORS = 6


class DiskstatsReader:
    """Read disk I/O counters from /proc/diskstats.

    Each monitored device produces four columns:
      - ``disk_{dev}_reads``          -- cumulative reads completed
      - ``disk_{dev}_writes``         -- cumulative writes completed
      - ``disk_{dev}_read_sectors``   -- cumulative sectors read
      - ``disk_{dev}_write_sectors``  -- cumulative sectors written

    Values are cumulative counters as reported by the kernel.
    """

    COLUMNS: ClassVar[list[str]] = []  # populated per-instance via property

    _SUFFIXES: ClassVar[list[tuple[str, int]]] = [
        ("reads", _FIELD_READS),
        ("writes", _FIELD_WRITES),
        ("read_sectors", _FIELD_READ_SECTORS),
        ("write_sectors", _FIELD_WRITE_SECTORS),
    ]

    def __init__(
        self,
        devices: list[str],
        proc_root: str = "/proc",
    ) -> None:
        self._devices = devices
        self._diskstats_path = Path(proc_root) / "diskstats"
        self._columns: list[str] = []

        for dev in self._devices:
            for suffix, _ in self._SUFFIXES:
                self._columns.append(f"disk_{dev}_{suffix}")

    @property
    def columns(self) -> list[str]:
        """Return the column names for this reader instance."""
        return list(self._columns)

    @classmethod
    def discover_devices(
        cls,
        proc_root: str = "/proc",
        skip_partitions: bool = True,
    ) -> list[str]:
        """Discover block devices listed in /proc/diskstats.

        Args:
            proc_root: Base path to the proc filesystem.
            skip_partitions: If True, skip devices whose name ends with a
                digit following a letter sequence (e.g. ``sda1``), keeping
                only whole-disk devices (e.g. ``sda``, ``nvme0n1``).

        Returns:
            Sorted list of device names.
        """
        diskstats_path = Path(proc_root) / "diskstats"
        devices: list[str] = []

        try:
            text = diskstats_path.read_text()
        except (FileNotFoundError, PermissionError):
            return devices

        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 4:
                continue
            dev_name = parts[2]

            # Skip ram, loop, dm- devices
            if dev_name.startswith(("ram", "loop", "dm-")):
                continue

            if skip_partitions:
                # For standard sd* devices, skip partitions like sda1
                is_sd = dev_name[:2] == "sd" and len(dev_name) > 3
                if is_sd and dev_name[-1].isdigit():
                    stripped = dev_name.rstrip("0123456789")
                    if stripped and stripped[-1].isalpha():
                        continue

                # For nvme: keep nvme*n* but skip nvme*n*p*
                if dev_name.startswith("nvme") and "p" in dev_name.split("n", 1)[-1]:
                    continue

            devices.append(dev_name)

        return sorted(set(devices))

    def read(self) -> dict[str, int | float | str]:
        """Read disk I/O counters for all monitored devices.

        Returns:
            Dict mapping column names to cumulative counter values (int),
            or empty string if a device could not be found or parsed.
        """
        # Initialise all columns to empty string
        result: dict[str, int | float | str] = {col: "" for col in self._columns}

        try:
            text = self._diskstats_path.read_text()
        except (FileNotFoundError, PermissionError):
            return result

        # Build a lookup: device_name -> list of int fields after the name
        dev_fields: dict[str, list[int]] = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) < 7:
                continue
            dev_name = parts[2]
            if dev_name in self._devices:
                try:
                    fields = [int(p) for p in parts[3:]]
                except ValueError:
                    continue
                dev_fields[dev_name] = fields

        # Fill in the result dict
        for dev in self._devices:
            fields = dev_fields.get(dev)
            if fields is None:
                continue
            for suffix, field_idx in self._SUFFIXES:
                col = f"disk_{dev}_{suffix}"
                if field_idx < len(fields):
                    result[col] = fields[field_idx]

        return result
