"""Tests for the procfs sensor reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensor_collector.sensors.procfs import ProcfsReader

SAMPLE_STAT = """\
cpu  10000 500 3000 80000 200 100 50 10 0 0
cpu0 5000 250 1500 40000 100 50 25 5 0 0
cpu1 5000 250 1500 40000 100 50 25 5 0 0
intr 12345678 50 0 0 0 0 0 0 0
ctxt 98765432
btime 1700000000
processes 5000
procs_running 3
procs_blocked 1
"""

SAMPLE_STAT_2 = """\
cpu  11000 600 3500 81000 250 120 60 15 0 0
cpu0 5500 300 1750 40500 125 60 30 7 0 0
cpu1 5500 300 1750 40500 125 60 30 8 0 0
intr 12346000 55 0 0 0 0 0 0 0
ctxt 98766000
btime 1700000000
processes 5010
procs_running 2
procs_blocked 0
"""

SAMPLE_MEMINFO = """\
MemTotal:       16384000 kB
MemFree:         8000000 kB
MemAvailable:   12000000 kB
Buffers:          500000 kB
Cached:          2000000 kB
SwapCached:            0 kB
Active:          4000000 kB
Inactive:        3000000 kB
Dirty:              1234 kB
Writeback:             0 kB
"""

SAMPLE_LOADAVG = "0.08 0.03 0.01 2/456 12345\n"


@pytest.fixture()
def fake_proc(tmp_path: Path) -> Path:
    """Create a fake /proc tree."""
    (tmp_path / "stat").write_text(SAMPLE_STAT)
    (tmp_path / "meminfo").write_text(SAMPLE_MEMINFO)
    (tmp_path / "loadavg").write_text(SAMPLE_LOADAVG)
    return tmp_path


class TestProcfsReader:
    """Tests for ProcfsReader."""

    def test_columns_class_var(self) -> None:
        assert isinstance(ProcfsReader.COLUMNS, list)
        assert len(ProcfsReader.COLUMNS) == 16
        assert "cpu_user_pct" in ProcfsReader.COLUMNS
        assert "mem_free_kb" in ProcfsReader.COLUMNS
        assert "loadavg_1m" in ProcfsReader.COLUMNS
        assert "procs_blocked" in ProcfsReader.COLUMNS

    def test_columns_property(self, fake_proc: Path) -> None:
        reader = ProcfsReader(str(fake_proc))
        assert reader.columns == ProcfsReader.COLUMNS

    def test_first_read_zeros_cpu(self, fake_proc: Path) -> None:
        reader = ProcfsReader(str(fake_proc))
        data = reader.read()
        assert data["cpu_user_pct"] == 0.0
        assert data["cpu_sys_pct"] == 0.0
        assert data["cpu_idle_pct"] == 0.0
        assert data["cpu_iowait_pct"] == 0.0

    def test_second_read_has_deltas(self, fake_proc: Path) -> None:
        reader = ProcfsReader(str(fake_proc))
        reader.read()  # first read stores baseline

        # Write updated stat
        (fake_proc / "stat").write_text(SAMPLE_STAT_2)
        data = reader.read()

        # Total delta: (11000+600+3500+81000+250+120+60+15)
        #            - (10000+500+3000+80000+200+100+50+10)
        # = 96545 - 93860 = 2685
        total = 2685.0
        d_user = (11000 + 600) - (10000 + 500)  # 1100
        d_sys = (3500 + 120 + 60 + 15) - (3000 + 100 + 50 + 10)  # 535
        d_idle = 81000 - 80000  # 1000
        d_iowait = 250 - 200  # 50

        assert data["cpu_user_pct"] == pytest.approx(d_user / total * 100.0, rel=1e-2)
        assert data["cpu_sys_pct"] == pytest.approx(d_sys / total * 100.0, rel=1e-2)
        assert data["cpu_idle_pct"] == pytest.approx(d_idle / total * 100.0, rel=1e-2)
        assert data["cpu_iowait_pct"] == pytest.approx(
            d_iowait / total * 100.0, rel=1e-2
        )

    def test_ctxt_and_intr(self, fake_proc: Path) -> None:
        reader = ProcfsReader(str(fake_proc))
        data = reader.read()
        assert data["ctxt"] == 98765432
        assert data["intr"] == 12345678

    def test_meminfo(self, fake_proc: Path) -> None:
        reader = ProcfsReader(str(fake_proc))
        data = reader.read()
        assert data["mem_free_kb"] == 8000000
        assert data["mem_available_kb"] == 12000000
        assert data["mem_cached_kb"] == 2000000
        assert data["mem_dirty_kb"] == 1234
        assert data["mem_buffers_kb"] == 500000

    def test_loadavg(self, fake_proc: Path) -> None:
        reader = ProcfsReader(str(fake_proc))
        data = reader.read()
        assert data["loadavg_1m"] == pytest.approx(0.08)
        assert data["loadavg_5m"] == pytest.approx(0.03)
        assert data["loadavg_15m"] == pytest.approx(0.01)
        assert data["procs_running"] == 2

    def test_procs_blocked(self, fake_proc: Path) -> None:
        reader = ProcfsReader(str(fake_proc))
        data = reader.read()
        assert data["procs_blocked"] == 1

    def test_all_columns_present(self, fake_proc: Path) -> None:
        reader = ProcfsReader(str(fake_proc))
        data = reader.read()
        for col in ProcfsReader.COLUMNS:
            assert col in data

    def test_missing_proc_returns_defaults(self, tmp_path: Path) -> None:
        reader = ProcfsReader(str(tmp_path / "nonexistent"))
        data = reader.read()
        assert data["cpu_user_pct"] == 0.0
        assert data["ctxt"] == ""
        assert data["mem_free_kb"] == ""
        assert data["loadavg_1m"] == ""
