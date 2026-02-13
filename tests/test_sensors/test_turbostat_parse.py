"""Tests for turbostat line parsing."""

from __future__ import annotations

import pytest

from sensor_collector.sensors.turbostat import parse_turbostat_line

# Realistic turbostat header and values
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


class TestParseTurbostatLine:
    """Unit tests for parse_turbostat_line()."""

    def test_valid_header_and_values(self) -> None:
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

    def test_missing_columns_get_empty_string(self) -> None:
        """When values list is shorter than header, missing columns are empty."""
        result = parse_turbostat_line(SAMPLE_HEADER, SAMPLE_VALUES[:3])

        # First three are present
        assert result["turbo_avg_mhz"] == pytest.approx(800.0)
        assert result["turbo_busy_pct"] == pytest.approx(23.45)
        assert result["turbo_bzy_mhz"] == pytest.approx(3400.0)

        # Rest should be empty strings
        assert result["turbo_tsc_mhz"] == ""
        assert result["turbo_ipc"] == ""
        assert result["turbo_irq"] == ""
        assert result["turbo_smi"] == ""
        assert result["turbo_core_tmp"] == ""
        assert result["turbo_pkg_tmp"] == ""
        assert result["turbo_pkg_pc2_pct"] == ""
        assert result["turbo_pkg_pc6_pct"] == ""
        assert result["turbo_pkg_watt"] == ""
        assert result["turbo_ram_watt"] == ""

    def test_extra_columns_are_ignored(self) -> None:
        """Extra values beyond the header length are safely ignored."""
        header = ["Avg_MHz", "Busy%"]
        values = ["800", "23.45", "9999", "extra", "more"]
        result = parse_turbostat_line(header, values)

        assert result["turbo_avg_mhz"] == pytest.approx(800.0)
        assert result["turbo_busy_pct"] == pytest.approx(23.45)
        # Other mapped columns remain empty since they weren't in header
        assert result["turbo_bzy_mhz"] == ""

    def test_dash_values_get_empty_string(self) -> None:
        """Hyphen/dash values (turbostat's placeholder) become empty strings."""
        values = ["-"] * len(SAMPLE_HEADER)
        result = parse_turbostat_line(SAMPLE_HEADER, values)

        for val in result.values():
            assert val == ""

    def test_empty_string_values_get_empty_string(self) -> None:
        values = [""] * len(SAMPLE_HEADER)
        result = parse_turbostat_line(SAMPLE_HEADER, values)

        for val in result.values():
            assert val == ""

    def test_non_numeric_value_gets_empty_string(self) -> None:
        header = ["Avg_MHz"]
        values = ["not_a_number"]
        result = parse_turbostat_line(header, values)
        assert result["turbo_avg_mhz"] == ""

    def test_unknown_header_columns_ignored(self) -> None:
        """Columns not in the header map are silently skipped."""
        header = ["Foo", "Bar", "Avg_MHz"]
        values = ["1", "2", "999"]
        result = parse_turbostat_line(header, values)

        assert result["turbo_avg_mhz"] == pytest.approx(999.0)
        # All other mapped columns stay empty
        assert result["turbo_busy_pct"] == ""
        assert result["turbo_ram_watt"] == ""

    def test_all_mapped_columns_present_in_result(self) -> None:
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
        assert set(result.keys()) == expected_keys

    def test_empty_header_returns_all_empty(self) -> None:
        result = parse_turbostat_line([], [])
        for val in result.values():
            assert val == ""
