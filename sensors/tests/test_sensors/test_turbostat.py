"""Tests for the turbostat background reader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sensor_collector.sensors.turbostat import (
    TurbostatReader,
    parse_turbostat_line,
)

# -- Realistic turbostat header and data ----------------

SAMPLE_HEADER = [
    "Avg_MHz",
    "Busy%",
    "Bzy_MHz",
    "TSC_MHz",
    "IPC",
    "IRQ",
    "SMI",
    "CoreTmp",
    "PkgTmp",
    "Pkg%pc2",
    "Pkg%pc6",
    "PkgWatt",
    "RAMWatt",
]

SAMPLE_VALUES = [
    "800",
    "23.45",
    "3400",
    "3500",
    "1.23",
    "1500",
    "0",
    "65",
    "67",
    "12.34",
    "0.00",
    "45.67",
    "8.90",
]

_MOD = "sensor_collector.sensors.turbostat"
_WHICH = f"{_MOD}.shutil.which"
_POPEN = f"{_MOD}.subprocess.Popen"


class TestParseTurbostatLine:
    """Unit tests for parse_turbostat_line()."""

    def test_valid_line(self) -> None:
        result = parse_turbostat_line(SAMPLE_HEADER, SAMPLE_VALUES)

        assert result["turbo_avg_mhz"] == pytest.approx(800.0)
        assert result["turbo_busy_pct"] == pytest.approx(23.45)
        assert result["turbo_bzy_mhz"] == pytest.approx(3400.0)
        assert result["turbo_tsc_mhz"] == pytest.approx(3500.0)
        assert result["turbo_ipc"] == pytest.approx(1.23)
        assert result["turbo_irq"] == pytest.approx(1500.0)
        assert result["turbo_smi"] == pytest.approx(0.0)
        assert result["turbo_core_tmp"] == pytest.approx(65.0)
        assert result["turbo_pkg_tmp"] == pytest.approx(67.0)
        assert result["turbo_pkg_pc2_pct"] == pytest.approx(12.34)
        assert result["turbo_pkg_pc6_pct"] == pytest.approx(0.0)
        assert result["turbo_pkg_watt"] == pytest.approx(45.67)
        assert result["turbo_ram_watt"] == pytest.approx(8.90)

    def test_dash_values_become_empty(self) -> None:
        values = ["-"] * len(SAMPLE_HEADER)
        result = parse_turbostat_line(SAMPLE_HEADER, values)
        for v in result.values():
            assert v == ""

    def test_empty_values_become_empty(self) -> None:
        values = [""] * len(SAMPLE_HEADER)
        result = parse_turbostat_line(SAMPLE_HEADER, values)
        for v in result.values():
            assert v == ""

    def test_short_values_list(self) -> None:
        result = parse_turbostat_line(SAMPLE_HEADER, SAMPLE_VALUES[:3])
        assert result["turbo_avg_mhz"] == pytest.approx(800.0)
        assert result["turbo_busy_pct"] == pytest.approx(23.45)
        assert result["turbo_bzy_mhz"] == pytest.approx(3400.0)
        # Remaining should be empty
        assert result["turbo_tsc_mhz"] == ""
        assert result["turbo_ram_watt"] == ""

    def test_unknown_header_columns_ignored(self) -> None:
        header = ["Foo", "Bar", "Avg_MHz"]
        values = ["1", "2", "999"]
        result = parse_turbostat_line(header, values)
        assert result["turbo_avg_mhz"] == pytest.approx(999.0)
        # All others remain empty string defaults.
        assert result["turbo_busy_pct"] == ""

    def test_non_numeric_value(self) -> None:
        header = ["Avg_MHz"]
        values = ["not_a_number"]
        result = parse_turbostat_line(header, values)
        assert result["turbo_avg_mhz"] == ""

    def test_all_mapped_columns_present(self) -> None:
        result = parse_turbostat_line(SAMPLE_HEADER, SAMPLE_VALUES)
        expected_keys = {
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
        }
        assert expected_keys == set(result.keys())


class TestTurbostatReader:
    """Unit tests for TurbostatReader."""

    def test_columns_class_var(self) -> None:
        assert isinstance(TurbostatReader.COLUMNS, list)
        assert "turbo_avg_mhz" in TurbostatReader.COLUMNS
        assert "turbo_age_ms" in TurbostatReader.COLUMNS
        assert len(TurbostatReader.COLUMNS) == 14

    @patch(_WHICH, return_value=None)
    def test_is_available_false(self, mock_which: MagicMock) -> None:
        assert TurbostatReader.is_available() is False

    @patch(_WHICH, return_value="/usr/bin/turbostat")
    def test_is_available_true(self, mock_which: MagicMock) -> None:
        assert TurbostatReader.is_available() is True

    @patch(_WHICH, return_value=None)
    def test_not_available_read_returns_all_empty(self, mock_which: MagicMock) -> None:
        reader = TurbostatReader()
        result = reader.read()
        for col in TurbostatReader.COLUMNS:
            assert result[col] == ""

    @patch(_WHICH, return_value=None)
    def test_stop_safe_when_not_started(self, mock_which: MagicMock) -> None:
        reader = TurbostatReader()
        reader.stop()  # Should not raise

    @patch(_WHICH, return_value="/usr/bin/turbostat")
    @patch(_POPEN, side_effect=FileNotFoundError)
    def test_start_failure_graceful(
        self,
        mock_popen: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        reader = TurbostatReader()
        result = reader.read()
        for col in TurbostatReader.COLUMNS:
            assert result[col] == ""

    @patch(_WHICH, return_value="/usr/bin/turbostat")
    @patch(_POPEN, side_effect=PermissionError)
    def test_start_permission_error_graceful(
        self,
        mock_popen: MagicMock,
        mock_which: MagicMock,
    ) -> None:
        reader = TurbostatReader()
        result = reader.read()
        for col in TurbostatReader.COLUMNS:
            assert result[col] == ""
