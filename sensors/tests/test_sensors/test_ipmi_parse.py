"""Tests for IPMI elist parsing functions."""

from __future__ import annotations

import pytest

from sensor_collector.sensors.ipmi import (
    _parse_elist_line,
    _parse_elist_output,
)


class TestParseElistLine:
    """Unit tests for _parse_elist_line()."""

    def test_temperature_line(self) -> None:
        line = "CPU Temp         | 01h | ok  |  3.1 | 42 degrees C"
        name, unit, value = _parse_elist_line(line)
        assert name == "cpu_temp"
        assert unit == "temp"
        assert value == 42.0

    def test_fan_line(self) -> None:
        line = "Fan1             | 02h | ok  |  7.1 | 8640 RPM"
        name, unit, value = _parse_elist_line(line)
        assert name == "fan1"
        assert unit == "fan"
        assert value == 8640.0

    def test_current_line(self) -> None:
        line = "Current 1        | 03h | ok  |  8.1 | 0.400 Amps"
        name, unit, value = _parse_elist_line(line)
        assert name == "current_1"
        assert unit == "current"
        assert value == pytest.approx(0.400)

    def test_voltage_line(self) -> None:
        line = "Voltage 1        | 04h | ok  |  8.1 | 210.000 Volts"
        name, unit, value = _parse_elist_line(line)
        assert name == "voltage_1"
        assert unit == "voltage"
        assert value == pytest.approx(210.0)

    def test_power_line(self) -> None:
        line = "Pwr Consumption  | 05h | ok  | 10.1 | 88 Watts"
        name, unit, value = _parse_elist_line(line)
        assert name == "pwr_consumption"
        assert unit == "power"
        assert value == 88.0

    def test_ns_status_is_skipped(self) -> None:
        line = "Fan3             | 32h | ns  | 29.3 | No Reading"
        name, unit, value = _parse_elist_line(line)
        assert name == ""
        assert unit == ""
        assert value is None

    def test_na_status_is_skipped(self) -> None:
        line = "Fan4             | 33h | na  | 29.4 | No Reading"
        name, _unit, _value = _parse_elist_line(line)
        assert name == ""

    def test_short_line_returns_empty(self) -> None:
        name, unit, value = _parse_elist_line("too | few")
        assert name == ""
        assert unit == ""
        assert value is None

    def test_empty_line_returns_empty(self) -> None:
        name, _unit, value = _parse_elist_line("")
        assert name == ""
        assert value is None

    def test_unrecognised_unit_returns_empty_unit(self) -> None:
        line = "Mystery          | 99h | ok  | 1.1 | 42 frobnicators"
        _name, unit, value = _parse_elist_line(line)
        assert unit == ""
        assert value is None


class TestParseElistOutput:
    """Unit tests for _parse_elist_output()."""

    MULTI_LINE_OUTPUT = (
        "CPU Temp         | 01h | ok  |  3.1 | 42 degrees C\n"
        "Fan1             | 02h | ok  |  7.1 | 8640 RPM\n"
        "Current 1        | 03h | ok  |  8.1 | 0.400 Amps\n"
        "Voltage 1        | 04h | ok  |  8.1 | 210.000 Volts\n"
        "Pwr Consumption  | 05h | ok  | 10.1 | 88 Watts\n"
        "Fan3             | 32h | ns  | 29.3 | No Reading\n"
    )

    def test_full_output_returns_dict_with_ipmi_prefix(self) -> None:
        result = _parse_elist_output(self.MULTI_LINE_OUTPUT)

        assert result["ipmi_cpu_temp"] == 42.0
        assert result["ipmi_fan1"] == 8640.0
        assert result["ipmi_current_1"] == pytest.approx(0.400)
        assert result["ipmi_voltage_1"] == pytest.approx(210.0)
        assert result["ipmi_pwr_consumption"] == 88.0

    def test_all_keys_have_ipmi_prefix(self) -> None:
        result = _parse_elist_output(self.MULTI_LINE_OUTPUT)
        for key in result:
            assert key.startswith("ipmi_"), f"Key {key!r} missing ipmi_ prefix"

    def test_ns_status_excluded(self) -> None:
        result = _parse_elist_output(self.MULTI_LINE_OUTPUT)
        assert "ipmi_fan3" not in result

    def test_empty_output(self) -> None:
        assert _parse_elist_output("") == {}

    def test_all_ns_lines_returns_empty(self) -> None:
        output = "Fan3 | 32h | ns | 29.3 | No Reading\n"
        assert _parse_elist_output(output) == {}
