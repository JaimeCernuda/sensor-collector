"""Tests for the CsvWriter module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sensor_collector.config import CollectorConfig
from sensor_collector.writer import CsvWriter

SAMPLE_COLUMNS = ["col_a", "col_b", "col_c"]


def _make_config(tmp_path: Path, flush_every: int = 5) -> CollectorConfig:
    """Create a CollectorConfig pointing at tmp_path."""
    return CollectorConfig(output_dir=tmp_path, flush_every=flush_every)


def _make_row(a: int = 1, b: int = 2, c: int = 3) -> dict[str, int]:
    """Create a sample data row."""
    return {"col_a": a, "col_b": b, "col_c": c}


class TestCsvWriterOpen:
    """Tests for CsvWriter.open() — file and metadata creation."""

    def test_open_creates_csv_file(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        writer.open()
        try:
            assert writer.csv_path.exists()
        finally:
            writer.close()

    def test_open_creates_metadata_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        writer.open()
        try:
            assert writer.meta_path.exists()
        finally:
            writer.close()

    def test_csv_has_header_row(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        writer.open()
        writer.close()

        with open(writer.csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == SAMPLE_COLUMNS

    def test_csv_path_ends_with_csv(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        writer.open()
        writer.close()
        assert writer.csv_path.suffix == ".csv"

    def test_meta_path_ends_with_meta_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        writer.open()
        writer.close()
        assert writer.meta_path.name.endswith(".meta.json")


class TestCsvWriterWriteRow:
    """Tests for CsvWriter.write_row()."""

    def test_write_row_increments_count(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            assert writer.row_count == 0
            writer.write_row(_make_row())
            assert writer.row_count == 1
            writer.write_row(_make_row(4, 5, 6))
            assert writer.row_count == 2

    def test_write_row_without_open_raises(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        with pytest.raises(RuntimeError, match="not opened"):
            writer.write_row(_make_row())

    def test_single_row_written_correctly(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            writer.write_row(_make_row(10, 20, 30))
            csv_path = writer.csv_path

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["col_a"] == "10"
        assert rows[0]["col_b"] == "20"
        assert rows[0]["col_c"] == "30"


class TestCsvWriterFlush:
    """Tests for periodic flush behavior."""

    def test_flush_happens_after_flush_every_rows(self, tmp_path: Path) -> None:
        """After flush_every rows, file should contain all data on disk."""
        config = _make_config(tmp_path, flush_every=3)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            for i in range(3):
                writer.write_row(_make_row(i, i, i))

            # After exactly flush_every rows, data should be on disk
            csv_path = writer.csv_path
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 3

    def test_row_count_tracks_all_writes(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, flush_every=2)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            for i in range(7):
                writer.write_row(_make_row(i, i, i))
            assert writer.row_count == 7


class TestCsvWriterClose:
    """Tests for CsvWriter.close() — metadata finalization."""

    def test_close_updates_metadata_with_total_rows(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        writer.open()
        writer.write_row(_make_row(1, 2, 3))
        writer.write_row(_make_row(4, 5, 6))
        writer.close()

        with open(writer.meta_path) as f:
            meta = json.load(f)
        assert meta["total_rows"] == 2

    def test_close_updates_metadata_with_end_time(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        writer.open()
        writer.close()

        with open(writer.meta_path) as f:
            meta = json.load(f)
        assert "end_time_utc" in meta

    def test_close_idempotent(self, tmp_path: Path) -> None:
        """Calling close() twice should not raise."""
        config = _make_config(tmp_path)
        writer = CsvWriter(SAMPLE_COLUMNS, config)
        writer.open()
        writer.close()
        writer.close()  # Should not raise


class TestCsvWriterContextManager:
    """Tests for using CsvWriter as a context manager."""

    def test_context_manager_creates_files(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            csv_path = writer.csv_path
            meta_path = writer.meta_path

        assert csv_path.exists()
        assert meta_path.exists()

    def test_context_manager_writes_and_closes(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            writer.write_row(_make_row(1, 2, 3))
            meta_path = writer.meta_path

        # After context exit, metadata should have total_rows
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["total_rows"] == 1

    def test_context_manager_returns_self(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            assert isinstance(writer, CsvWriter)


class TestMultipleRows:
    """Tests for writing and reading back multiple rows."""

    def test_multiple_rows_csv_content(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        rows_to_write = [
            _make_row(10, 20, 30),
            _make_row(40, 50, 60),
            _make_row(70, 80, 90),
        ]

        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            for row in rows_to_write:
                writer.write_row(row)
            csv_path = writer.csv_path

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            read_rows = list(reader)

        assert len(read_rows) == 3
        assert read_rows[0] == {"col_a": "10", "col_b": "20", "col_c": "30"}
        assert read_rows[1] == {"col_a": "40", "col_b": "50", "col_c": "60"}
        assert read_rows[2] == {"col_a": "70", "col_b": "80", "col_c": "90"}

    def test_row_count_matches_written(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            for i in range(25):
                writer.write_row(_make_row(i, i * 2, i * 3))
            assert writer.row_count == 25

    def test_csv_header_preserved_with_many_rows(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            for i in range(10):
                writer.write_row(_make_row(i, i, i))
            csv_path = writer.csv_path

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
        assert header == SAMPLE_COLUMNS


class TestMetadataJson:
    """Tests for the metadata JSON sidecar file contents."""

    def test_metadata_contains_hostname(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert "hostname" in meta
        assert isinstance(meta["hostname"], str)
        assert len(meta["hostname"]) > 0

    def test_metadata_contains_columns(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert meta["columns"] == SAMPLE_COLUMNS

    def test_metadata_contains_column_count(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert meta["column_count"] == len(SAMPLE_COLUMNS)

    def test_metadata_contains_start_time(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert "start_time_utc" in meta
        # Should be ISO-ish format ending in Z
        assert meta["start_time_utc"].endswith("Z")

    def test_metadata_contains_interval(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, flush_every=10)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert "interval_s" in meta
        assert meta["interval_s"] == config.interval

    def test_metadata_contains_flush_every(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path, flush_every=10)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert meta["flush_every"] == 10

    def test_metadata_contains_csv_file(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            csv_name = writer.csv_path.name
        meta = self._load_meta(tmp_path)
        assert meta["csv_file"] == csv_name

    def test_metadata_contains_platform_info(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert "platform" in meta
        assert "kernel" in meta
        assert "python_version" in meta

    def test_metadata_contains_pid(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert "pid" in meta
        assert isinstance(meta["pid"], int)

    def test_metadata_contains_config(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config):
            pass
        meta = self._load_meta(tmp_path)
        assert "config" in meta
        assert isinstance(meta["config"], dict)

    def test_metadata_is_valid_json(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with CsvWriter(SAMPLE_COLUMNS, config) as writer:
            meta_path = writer.meta_path
        # json.load will raise if invalid
        with open(meta_path) as f:
            meta = json.load(f)
        assert isinstance(meta, dict)

    @staticmethod
    def _load_meta(tmp_path: Path) -> dict[str, object]:
        """Find and load the single .meta.json file in tmp_path."""
        meta_files = list(tmp_path.glob("*.meta.json"))
        assert len(meta_files) == 1, f"Expected 1 meta file, found {len(meta_files)}"
        with open(meta_files[0]) as f:
            return json.load(f)  # type: ignore[no-any-return]
