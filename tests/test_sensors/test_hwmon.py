"""Tests for the hwmon sensor reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensor_collector.sensors.hwmon import HwmonReader, HwmonSensor


@pytest.fixture()
def fake_hwmon(tmp_path: Path) -> Path:
    """Create a fake /sys/class/hwmon tree."""
    hwmon0 = tmp_path / "hwmon0"
    hwmon0.mkdir()
    (hwmon0 / "name").write_text("coretemp\n")
    (hwmon0 / "temp1_input").write_text("45000\n")
    (hwmon0 / "temp1_label").write_text("Core 0\n")
    (hwmon0 / "temp2_input").write_text("47500\n")
    (hwmon0 / "temp2_label").write_text("Core 1\n")

    hwmon1 = tmp_path / "hwmon1"
    hwmon1.mkdir()
    (hwmon1 / "name").write_text("acpitz\n")
    (hwmon1 / "temp1_input").write_text("30000\n")
    # No label file for temp1 -> should fall back to "temp1"

    return tmp_path


class TestDiscoverHwmon:
    """Tests for HwmonReader.discover_hwmon()."""

    def test_discovers_sensors(self, fake_hwmon: Path) -> None:
        sensors = HwmonReader.discover_hwmon(str(fake_hwmon))
        assert len(sensors) == 3

    def test_sensor_names(self, fake_hwmon: Path) -> None:
        sensors = HwmonReader.discover_hwmon(str(fake_hwmon))
        names = [s.name for s in sensors]
        assert "coretemp" in names
        assert "acpitz" in names

    def test_label_fallback(self, fake_hwmon: Path) -> None:
        sensors = HwmonReader.discover_hwmon(str(fake_hwmon))
        acpi = [s for s in sensors if s.name == "acpitz"]
        assert len(acpi) == 1
        assert acpi[0].label == "temp1"

    def test_label_sanitisation(self, fake_hwmon: Path) -> None:
        sensors = HwmonReader.discover_hwmon(str(fake_hwmon))
        core0 = [s for s in sensors if s.label == "core_0"]
        assert len(core0) == 1

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        sensors = HwmonReader.discover_hwmon(str(tmp_path / "nonexistent"))
        assert sensors == []


class TestHwmonReader:
    """Tests for HwmonReader.read()."""

    def test_read_values(self, fake_hwmon: Path) -> None:
        sensors = HwmonReader.discover_hwmon(str(fake_hwmon))
        reader = HwmonReader(sensors)
        data = reader.read()
        assert data["temp_coretemp_core_0_c"] == pytest.approx(45.0)
        assert data["temp_coretemp_core_1_c"] == pytest.approx(47.5)
        assert data["temp_acpitz_temp1_c"] == pytest.approx(30.0)

    def test_columns_match_read_keys(self, fake_hwmon: Path) -> None:
        sensors = HwmonReader.discover_hwmon(str(fake_hwmon))
        reader = HwmonReader(sensors)
        assert set(reader.columns) == set(reader.read().keys())

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        sensor = HwmonSensor(path=tmp_path / "nonexistent", name="test", label="x")
        reader = HwmonReader([sensor])
        data = reader.read()
        assert data["temp_test_x_c"] == ""

    def test_empty_sensors(self) -> None:
        reader = HwmonReader([])
        assert reader.read() == {}
        assert reader.columns == []
