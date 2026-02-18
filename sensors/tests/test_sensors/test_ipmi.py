"""Tests for the IPMI background sensor reader."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sensor_collector.sensors.ipmi import (
    IpmiReader,
    _parse_elist_line,
    _parse_elist_output,
    _sanitize_name,
)

# -- Realistic ipmitool sdr elist output --------------------

SAMPLE_ELIST_OUTPUT = """\
Inlet Temp       | 01h | ok  |  7.1 | 23 degrees C
Exhaust Temp     | 02h | ok  |  7.1 | 37 degrees C
CPU1 Temp        | 03h | ok  |  3.1 | 55 degrees C
CPU2 Temp        | 04h | ok  |  3.2 | 52 degrees C
Fan1             | 30h | ok  | 29.1 | 5400 RPM
Fan2             | 31h | ok  | 29.2 | 5520 RPM
Fan3             | 32h | ns  | 29.3 | No Reading
Planar 12V       | 10h | ok  |  7.1 | 12.192 Volts
Planar 3.3V      | 11h | ok  |  7.1 | 3.384 Volts
Current 1        | 40h | ok  | 10.1 | 0.600 Amps
Pwr Consumption  | 50h | ok  |  7.1 | 140 Watts
"""

_MOD = "sensor_collector.sensors.ipmi"
_WHICH = f"{_MOD}.shutil.which"
_RUN = f"{_MOD}._run_ipmitool_elist"


class TestSanitizeName:
    """Unit tests for _sanitize_name()."""

    def test_basic(self) -> None:
        assert _sanitize_name("Inlet Temp") == "inlet_temp"

    def test_special_chars(self) -> None:
        assert _sanitize_name("Planar 3.3V") == "planar_3_3v"

    def test_leading_trailing_spaces(self) -> None:
        assert _sanitize_name("  Fan1  ") == "fan1"

    def test_multiple_spaces(self) -> None:
        assert _sanitize_name("CPU 1  Temp") == "cpu_1_temp"


class TestParseElistLine:
    """Unit tests for _parse_elist_line()."""

    def test_temperature(self) -> None:
        line = "Inlet Temp       | 01h | ok  |  7.1 | 23 degrees C"
        name, unit, value = _parse_elist_line(line)
        assert name == "inlet_temp"
        assert unit == "temp"
        assert value == 23.0

    def test_fan(self) -> None:
        line = "Fan1             | 30h | ok  | 29.1 | 5400 RPM"
        name, unit, value = _parse_elist_line(line)
        assert name == "fan1"
        assert unit == "fan"
        assert value == 5400.0

    def test_voltage(self) -> None:
        line = "Planar 12V       | 10h | ok  |  7.1 | 12.192 Volts"
        name, unit, value = _parse_elist_line(line)
        assert name == "planar_12v"
        assert unit == "voltage"
        assert value == pytest.approx(12.192)

    def test_current(self) -> None:
        line = "Current 1        | 40h | ok  | 10.1 | 0.600 Amps"
        name, unit, value = _parse_elist_line(line)
        assert name == "current_1"
        assert unit == "current"
        assert value == pytest.approx(0.600)

    def test_power(self) -> None:
        line = "Pwr Consumption  | 50h | ok  |  7.1 | 140 Watts"
        name, unit, value = _parse_elist_line(line)
        assert name == "pwr_consumption"
        assert unit == "power"
        assert value == 140.0

    def test_no_reading_skipped(self) -> None:
        line = "Fan3             | 32h | ns  | 29.3 | No Reading"
        name, _unit, _value = _parse_elist_line(line)
        assert name == ""

    def test_short_line(self) -> None:
        name, _unit, value = _parse_elist_line("too | few")
        assert name == ""
        assert value is None

    def test_unrecognised_unit(self) -> None:
        line = "SomeOther        | 99h | ok  | 1.1 | 42 foos"
        _name, unit, value = _parse_elist_line(line)
        assert unit == ""
        assert value is None


class TestParseElistOutput:
    """Unit tests for _parse_elist_output()."""

    def test_parses_full_output(self) -> None:
        result = _parse_elist_output(SAMPLE_ELIST_OUTPUT)

        assert result["ipmi_inlet_temp"] == 23.0
        assert result["ipmi_exhaust_temp"] == 37.0
        assert result["ipmi_cpu1_temp"] == 55.0
        assert result["ipmi_cpu2_temp"] == 52.0
        assert result["ipmi_fan1"] == 5400.0
        assert result["ipmi_fan2"] == 5520.0
        assert result["ipmi_planar_12v"] == pytest.approx(12.192)
        assert result["ipmi_planar_3_3v"] == pytest.approx(3.384)
        assert result["ipmi_current_1"] == pytest.approx(0.600)
        assert result["ipmi_pwr_consumption"] == 140.0

        # Fan3 had "ns" status, should be excluded.
        assert "ipmi_fan3" not in result

    def test_empty_output(self) -> None:
        assert _parse_elist_output("") == {}


class TestIpmiReader:
    """Unit tests for the IpmiReader class."""

    def setup_method(self) -> None:
        """Save COLUMNS so each test starts clean."""
        self._original_columns = IpmiReader.COLUMNS[:]

    def teardown_method(self) -> None:
        """Restore COLUMNS after each test."""
        IpmiReader.COLUMNS = self._original_columns

    @patch(_WHICH, return_value=None)
    def test_not_available(self, mock_which: MagicMock) -> None:
        reader = IpmiReader()
        assert reader.COLUMNS == [] or reader.read() == {}

    @patch(_WHICH, return_value=None)
    def test_is_available_false(self, mock_which: MagicMock) -> None:
        assert IpmiReader.is_available() is False

    @patch(_WHICH, return_value="/usr/bin/ipmitool")
    def test_is_available_true(self, mock_which: MagicMock) -> None:
        assert IpmiReader.is_available() is True

    @patch(_RUN, return_value=SAMPLE_ELIST_OUTPUT)
    @patch(_WHICH, return_value="/usr/bin/ipmitool")
    def test_discovery_sets_columns(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        reader = IpmiReader()
        try:
            assert "ipmi_age_ms" in reader.COLUMNS
            assert "ipmi_inlet_temp" in reader.COLUMNS
            assert "ipmi_fan1" in reader.COLUMNS
            # Fan3 was ns, should not be in columns
            assert "ipmi_fan3" not in reader.COLUMNS
        finally:
            reader.stop()

    @patch(_RUN, return_value=SAMPLE_ELIST_OUTPUT)
    @patch(_WHICH, return_value="/usr/bin/ipmitool")
    def test_read_returns_values_with_age(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        reader = IpmiReader()
        try:
            result = reader.read()
            assert isinstance(result["ipmi_age_ms"], (int, float))
            assert result["ipmi_inlet_temp"] == 23.0
        finally:
            reader.stop()

    @patch(_RUN, side_effect=FileNotFoundError)
    @patch(_WHICH, return_value="/usr/bin/ipmitool")
    def test_discovery_failure_gives_empty(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        reader = IpmiReader()
        assert reader.read() == {}

    @patch(_RUN, return_value=SAMPLE_ELIST_OUTPUT)
    @patch(_WHICH, return_value="/usr/bin/ipmitool")
    def test_stop_is_idempotent(
        self,
        mock_which: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        reader = IpmiReader()
        reader.stop()
        reader.stop()  # Should not raise
