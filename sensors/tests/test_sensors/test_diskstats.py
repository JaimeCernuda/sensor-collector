"""Tests for the diskstats sensor reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensor_collector.sensors.diskstats import DiskstatsReader

SAMPLE_DISKSTATS = """\
   8       0 sda 12345 678 90123 4567 89012 345 67890 1234 0 5678 9012
   8       1 sda1 1000 200 3000 400 5000 600 7000 800 0 900 1000
   8      16 sdb 54321 876 32109 7654 21098 543 9876 4321 0 8765 2109
 259       0 nvme0n1 99999 111 222222 333 444444 555 666666 777 0 888 999
 259       1 nvme0n1p1 11111 22 33333 44 55555 66 77777 88 0 99 11
   7       0 loop0 100 0 200 0 0 0 0 0 0 0 0
"""


@pytest.fixture()
def fake_proc(tmp_path: Path) -> Path:
    """Create a fake /proc tree with diskstats."""
    (tmp_path / "diskstats").write_text(SAMPLE_DISKSTATS)
    return tmp_path


class TestDiscoverDevices:
    """Tests for DiskstatsReader.discover_devices()."""

    def test_discovers_whole_disks(self, fake_proc: Path) -> None:
        devices = DiskstatsReader.discover_devices(str(fake_proc))
        assert "sda" in devices
        assert "sdb" in devices
        assert "nvme0n1" in devices

    def test_skips_partitions(self, fake_proc: Path) -> None:
        devices = DiskstatsReader.discover_devices(str(fake_proc))
        assert "sda1" not in devices
        assert "nvme0n1p1" not in devices

    def test_skips_loop_devices(self, fake_proc: Path) -> None:
        devices = DiskstatsReader.discover_devices(str(fake_proc))
        assert "loop0" not in devices

    def test_includes_partitions_when_requested(self, fake_proc: Path) -> None:
        devices = DiskstatsReader.discover_devices(
            str(fake_proc), skip_partitions=False
        )
        assert "sda1" in devices

    def test_nonexistent_proc(self, tmp_path: Path) -> None:
        devices = DiskstatsReader.discover_devices(str(tmp_path / "nonexistent"))
        assert devices == []


class TestDiskstatsReader:
    """Tests for DiskstatsReader.read()."""

    def test_read_values(self, fake_proc: Path) -> None:
        reader = DiskstatsReader(["sda", "nvme0n1"], str(fake_proc))
        data = reader.read()
        # sda: reads=12345, sectors_read=90123, writes=89012, sectors_written=67890
        assert data["disk_sda_reads"] == 12345
        assert data["disk_sda_read_sectors"] == 90123
        assert data["disk_sda_writes"] == 89012
        assert data["disk_sda_write_sectors"] == 67890

        # nvme0n1: reads=99999, sectors_read=222222,
        # writes=444444, sectors_written=666666
        assert data["disk_nvme0n1_reads"] == 99999
        assert data["disk_nvme0n1_read_sectors"] == 222222
        assert data["disk_nvme0n1_writes"] == 444444
        assert data["disk_nvme0n1_write_sectors"] == 666666

    def test_columns_match_read_keys(self, fake_proc: Path) -> None:
        reader = DiskstatsReader(["sda"], str(fake_proc))
        data = reader.read()
        assert set(reader.columns) == set(data.keys())

    def test_missing_device_returns_empty(self, fake_proc: Path) -> None:
        reader = DiskstatsReader(["nonexistent"], str(fake_proc))
        data = reader.read()
        for col in reader.columns:
            assert data[col] == ""

    def test_missing_diskstats_file(self, tmp_path: Path) -> None:
        reader = DiskstatsReader(["sda"], str(tmp_path))
        data = reader.read()
        for col in reader.columns:
            assert data[col] == ""

    def test_empty_devices(self, fake_proc: Path) -> None:
        reader = DiskstatsReader([], str(fake_proc))
        assert reader.read() == {}
        assert reader.columns == []

    def test_four_columns_per_device(self) -> None:
        reader = DiskstatsReader(["sda", "sdb"])
        assert len(reader.columns) == 8
