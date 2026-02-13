"""Tests for the cstate sensor reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensor_collector.sensors.cstate import CpuCstateInfo, CstateReader


@pytest.fixture()
def fake_cpuidle(tmp_path: Path) -> Path:
    """Create a fake /sys/devices/system/cpu tree with cpuidle states."""
    for cpu_idx in range(2):
        for state_idx, (name, time_us) in enumerate(
            [("POLL", "0"), ("C1", "500000"), ("C1E", "1200000")]
        ):
            state_dir = tmp_path / f"cpu{cpu_idx}" / "cpuidle" / f"state{state_idx}"
            state_dir.mkdir(parents=True)
            (state_dir / "name").write_text(f"{name}\n")
            (state_dir / "time").write_text(f"{time_us}\n")

    # cpu2 has no cpuidle directory
    (tmp_path / "cpu2").mkdir()

    # Non-cpu entry
    (tmp_path / "cpufreq").mkdir()

    return tmp_path


class TestDiscoverCstates:
    """Tests for CstateReader.discover_cstates()."""

    def test_discovers_cpus(self, fake_cpuidle: Path) -> None:
        infos = CstateReader.discover_cstates(str(fake_cpuidle))
        assert len(infos) == 2

    def test_discovers_states(self, fake_cpuidle: Path) -> None:
        infos = CstateReader.discover_cstates(str(fake_cpuidle))
        for info in infos:
            assert len(info.states) == 3
            state_names = [s[1] for s in info.states]
            assert "poll" in state_names
            assert "c1" in state_names
            assert "c1e" in state_names

    def test_skips_cpu_without_cpuidle(self, fake_cpuidle: Path) -> None:
        infos = CstateReader.discover_cstates(str(fake_cpuidle))
        cpu_indices = [i.cpu_index for i in infos]
        assert 2 not in cpu_indices

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        infos = CstateReader.discover_cstates(str(tmp_path / "nonexistent"))
        assert infos == []


class TestCstateReader:
    """Tests for CstateReader.read()."""

    def test_read_values(self, fake_cpuidle: Path) -> None:
        infos = CstateReader.discover_cstates(str(fake_cpuidle))
        reader = CstateReader(infos, sysfs_root=str(fake_cpuidle))
        data = reader.read()
        assert data["cstate_cpu0_poll_us"] == 0
        assert data["cstate_cpu0_c1_us"] == 500000
        assert data["cstate_cpu1_c1e_us"] == 1200000

    def test_columns_match_read_keys(self, fake_cpuidle: Path) -> None:
        infos = CstateReader.discover_cstates(str(fake_cpuidle))
        reader = CstateReader(infos, sysfs_root=str(fake_cpuidle))
        assert set(reader.columns) == set(reader.read().keys())

    def test_missing_file_returns_empty(self) -> None:
        info = CpuCstateInfo(cpu_index=99, states=[(0, "fake")])
        reader = CstateReader([info])
        data = reader.read()
        assert data["cstate_cpu99_fake_us"] == ""

    def test_empty_cstates(self) -> None:
        reader = CstateReader([])
        assert reader.read() == {}
        assert reader.columns == []
