"""Tests for the cpufreq sensor reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensor_collector.sensors.cpufreq import CpufreqReader


@pytest.fixture()
def fake_cpufreq(tmp_path: Path) -> Path:
    """Create a fake /sys/devices/system/cpu tree with cpufreq."""
    for i in range(4):
        cpu_dir = tmp_path / f"cpu{i}" / "cpufreq"
        cpu_dir.mkdir(parents=True)
        (cpu_dir / "scaling_cur_freq").write_text(f"{2400000 + i * 100000}\n")

    # cpu4 has no cpufreq
    (tmp_path / "cpu4").mkdir()

    # Non-cpu entries should be ignored
    (tmp_path / "cpufreq").mkdir()
    (tmp_path / "cpuidle").mkdir()

    return tmp_path


class TestDiscoverCpufreq:
    """Tests for CpufreqReader.discover_cpufreq()."""

    def test_discovers_cpus(self, fake_cpufreq: Path) -> None:
        indices = CpufreqReader.discover_cpufreq(str(fake_cpufreq))
        assert indices == [0, 1, 2, 3]

    def test_skips_cpu_without_cpufreq(self, fake_cpufreq: Path) -> None:
        indices = CpufreqReader.discover_cpufreq(str(fake_cpufreq))
        assert 4 not in indices

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        indices = CpufreqReader.discover_cpufreq(str(tmp_path / "nonexistent"))
        assert indices == []


class TestCpufreqReader:
    """Tests for CpufreqReader.read()."""

    def test_read_values(self, fake_cpufreq: Path) -> None:
        indices = CpufreqReader.discover_cpufreq(str(fake_cpufreq))
        reader = CpufreqReader(indices, sysfs_root=str(fake_cpufreq))
        data = reader.read()
        assert data["cpu0_freq_mhz"] == pytest.approx(2400.0)
        assert data["cpu1_freq_mhz"] == pytest.approx(2500.0)
        assert data["cpu3_freq_mhz"] == pytest.approx(2700.0)

    def test_columns_match_read_keys(self, fake_cpufreq: Path) -> None:
        indices = CpufreqReader.discover_cpufreq(str(fake_cpufreq))
        reader = CpufreqReader(indices, sysfs_root=str(fake_cpufreq))
        assert set(reader.columns) == set(reader.read().keys())

    def test_missing_file_returns_empty(self) -> None:
        reader = CpufreqReader([99])
        data = reader.read()
        assert data["cpu99_freq_mhz"] == ""

    def test_empty_indices(self) -> None:
        reader = CpufreqReader([])
        assert reader.read() == {}
        assert reader.columns == []
