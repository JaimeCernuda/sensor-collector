"""Buffered CSV writer with metadata JSON sidecar.

Writes one CSV row per sample tick and flushes periodically. Writes a
metadata JSON file alongside the CSV at startup with machine info and
column schema.
"""

from __future__ import annotations

import csv
import io
import json
import os
import platform
import socket
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import CollectorConfig


def _generate_filename() -> str:
    """Generate a CSV filename from hostname and start timestamp."""
    hostname = socket.gethostname()
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    return f"drift_{hostname}_{ts}"


class CsvWriter:
    """Buffered CSV writer with metadata JSON sidecar."""

    def __init__(
        self,
        columns: list[str],
        config: CollectorConfig,
    ) -> None:
        self._columns = columns
        self._config = config
        self._flush_every = config.flush_every

        config.output_dir.mkdir(parents=True, exist_ok=True)

        base = _generate_filename()
        self._csv_path = config.output_dir / f"{base}.csv"
        self._meta_path = config.output_dir / f"{base}.meta.json"

        self._file: io.TextIOWrapper | None = None
        self._writer: csv.DictWriter[str] | None = None
        self._row_count = 0

    @property
    def csv_path(self) -> Path:
        """Path to the CSV output file."""
        return self._csv_path

    @property
    def meta_path(self) -> Path:
        """Path to the metadata JSON file."""
        return self._meta_path

    @property
    def row_count(self) -> int:
        """Number of rows written so far."""
        return self._row_count

    def open(self) -> None:
        """Open the CSV file and write headers. Write metadata JSON."""
        self._file = open(  # noqa: SIM115
            self._csv_path, "w", newline="", buffering=1
        )
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._columns, extrasaction="ignore"
        )
        self._writer.writeheader()

        self._write_metadata()

    def _write_metadata(self) -> None:
        """Write metadata JSON sidecar file."""
        meta = {
            "hostname": socket.gethostname(),
            "fqdn": socket.getfqdn(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "python_version": platform.python_version(),
            "start_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "start_time_monotonic_ns": time.monotonic_ns(),
            "csv_file": self._csv_path.name,
            "columns": self._columns,
            "column_count": len(self._columns),
            "interval_s": self._config.interval,
            "flush_every": self._flush_every,
            "config": asdict(self._config),
            "pid": os.getpid(),
        }
        # Convert Path objects to strings for JSON serialization
        meta["config"]["output_dir"] = str(meta["config"]["output_dir"])
        with open(self._meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def write_row(self, row: dict[str, int | float | str]) -> None:
        """Write a single CSV row. Flushes periodically."""
        if self._writer is None:
            raise RuntimeError("CsvWriter not opened; call open() first")

        self._writer.writerow(row)
        self._row_count += 1

        if self._row_count % self._flush_every == 0:
            self.flush()

    def flush(self) -> None:
        """Flush the CSV file to disk."""
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())

    def close(self) -> None:
        """Flush and close the CSV file. Update metadata with final stats."""
        if self._file is not None:
            self.flush()
            self._file.close()
            self._file = None
            self._writer = None

        # Update metadata with final row count
        if self._meta_path.exists():
            with open(self._meta_path) as f:
                meta = json.load(f)
            meta["end_time_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta["total_rows"] = self._row_count
            with open(self._meta_path, "w") as f:
                json.dump(meta, f, indent=2)

    def __enter__(self) -> CsvWriter:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()
