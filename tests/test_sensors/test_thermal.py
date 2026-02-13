"""Tests for the thermal zone sensor reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensor_collector.sensors.thermal import ThermalReader, ThermalZone


@pytest.fixture()
def fake_thermal(tmp_path: Path) -> Path:
    """Create a fake /sys/class/thermal tree."""
    tz0 = tmp_path / "thermal_zone0"
    tz0.mkdir()
    (tz0 / "temp").write_text("50000\n")
    (tz0 / "type").write_text("x86_pkg_temp\n")

    tz1 = tmp_path / "thermal_zone1"
    tz1.mkdir()
    (tz1 / "temp").write_text("42000\n")
    (tz1 / "type").write_text("acpitz\n")

    # thermal_zone2 has no temp file -> should be skipped
    tz2 = tmp_path / "thermal_zone2"
    tz2.mkdir()
    (tz2 / "type").write_text("iwlwifi_1\n")

    # Non-zone entry should be ignored
    (tmp_path / "cooling_device0").mkdir()

    return tmp_path


class TestDiscoverThermal:
    """Tests for ThermalReader.discover_thermal()."""

    def test_discovers_zones(self, fake_thermal: Path) -> None:
        zones = ThermalReader.discover_thermal(str(fake_thermal))
        assert len(zones) == 2

    def test_zone_types(self, fake_thermal: Path) -> None:
        zones = ThermalReader.discover_thermal(str(fake_thermal))
        types = [z.zone_type for z in zones]
        assert "x86_pkg_temp" in types
        assert "acpitz" in types

    def test_skips_zone_without_temp(self, fake_thermal: Path) -> None:
        zones = ThermalReader.discover_thermal(str(fake_thermal))
        indices = [z.index for z in zones]
        assert 2 not in indices

    def test_sorted_by_index(self, fake_thermal: Path) -> None:
        zones = ThermalReader.discover_thermal(str(fake_thermal))
        assert zones[0].index < zones[1].index

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        zones = ThermalReader.discover_thermal(str(tmp_path / "nonexistent"))
        assert zones == []


class TestThermalReader:
    """Tests for ThermalReader.read()."""

    def test_read_values(self, fake_thermal: Path) -> None:
        zones = ThermalReader.discover_thermal(str(fake_thermal))
        reader = ThermalReader(zones)
        data = reader.read()
        assert data["tz_x86_pkg_temp_0_c"] == pytest.approx(50.0)
        assert data["tz_acpitz_1_c"] == pytest.approx(42.0)

    def test_columns_match_read_keys(self, fake_thermal: Path) -> None:
        zones = ThermalReader.discover_thermal(str(fake_thermal))
        reader = ThermalReader(zones)
        assert set(reader.columns) == set(reader.read().keys())

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        zone = ThermalZone(
            index=99,
            zone_type="fake",
            temp_path=tmp_path / "nonexistent",
        )
        reader = ThermalReader([zone])
        data = reader.read()
        assert data["tz_fake_99_c"] == ""

    def test_empty_zones(self) -> None:
        reader = ThermalReader([])
        assert reader.read() == {}
        assert reader.columns == []
