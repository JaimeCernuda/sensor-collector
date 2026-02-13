"""Tests for the RAPL sensor reader.

Note: The RAPL sysfs directory names contain colons (e.g. ``intel-rapl:0``)
which are invalid in Windows paths.  Discovery tests are therefore skipped
on Windows, while the reader tests construct RaplDomain objects directly
with colon-free paths.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sensor_collector.sensors.rapl import RaplDomain, RaplReader

_IS_WINDOWS = sys.platform == "win32"


# -- Fixtures for the RaplReader (colon-free, works on all platforms) ---------


@pytest.fixture()
def fake_energy_files(tmp_path: Path) -> list[tuple[str, Path]]:
    """Create fake energy_uj files and return (name, path) pairs."""
    entries: list[tuple[str, Path]] = []

    for name, value in [
        ("package_0", "123456789"),
        ("package_0_core", "987654321"),
        ("package_0_uncore", "111222333"),
        ("psys", "444555666"),
    ]:
        energy_path = tmp_path / f"{name}_energy_uj"
        energy_path.write_text(f"{value}\n")
        entries.append((name, energy_path))

    return entries


# -- Fixtures that use real sysfs-like paths (Linux only) --------------------


@pytest.fixture()
def fake_rapl(tmp_path: Path) -> Path:
    """Create a fake /sys/class/powercap tree with RAPL domains.

    Only usable on platforms that allow colons in directory names (Linux).
    """
    rapl0 = tmp_path / "intel-rapl:0"
    rapl0.mkdir()
    (rapl0 / "name").write_text("package-0\n")
    (rapl0 / "energy_uj").write_text("123456789\n")

    rapl0_0 = rapl0 / "intel-rapl:0:0"
    rapl0_0.mkdir()
    (rapl0_0 / "name").write_text("core\n")
    (rapl0_0 / "energy_uj").write_text("987654321\n")

    rapl0_1 = rapl0 / "intel-rapl:0:1"
    rapl0_1.mkdir()
    (rapl0_1 / "name").write_text("uncore\n")
    (rapl0_1 / "energy_uj").write_text("111222333\n")

    rapl1 = tmp_path / "intel-rapl:1"
    rapl1.mkdir()
    (rapl1 / "name").write_text("psys\n")
    (rapl1 / "energy_uj").write_text("444555666\n")

    return tmp_path


# -- Discovery tests (Linux only) -------------------------------------------


@pytest.mark.skipif(_IS_WINDOWS, reason="Colons not allowed in Windows paths")
class TestDiscoverRapl:
    """Tests for RaplReader.discover_rapl()."""

    def test_discovers_domains(self, fake_rapl: Path) -> None:
        domains = RaplReader.discover_rapl(str(fake_rapl))
        assert len(domains) == 4

    def test_domain_names(self, fake_rapl: Path) -> None:
        domains = RaplReader.discover_rapl(str(fake_rapl))
        names = [d.name for d in domains]
        assert "package_0" in names
        assert "psys" in names
        assert "package_0_core" in names
        assert "package_0_uncore" in names

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        domains = RaplReader.discover_rapl(str(tmp_path / "nonexistent"))
        assert domains == []

    def test_empty_powercap(self, tmp_path: Path) -> None:
        domains = RaplReader.discover_rapl(str(tmp_path))
        assert domains == []


# -- Reader tests (all platforms) --------------------------------------------


class TestRaplReader:
    """Tests for RaplReader.read()."""

    def test_read_values(self, fake_energy_files: list[tuple[str, Path]]) -> None:
        domains = [
            RaplDomain(name=name, energy_path=path) for name, path in fake_energy_files
        ]
        reader = RaplReader(domains)
        data = reader.read()
        assert data["rapl_package_0_uj"] == 123456789
        assert data["rapl_package_0_core_uj"] == 987654321
        assert data["rapl_package_0_uncore_uj"] == 111222333
        assert data["rapl_psys_uj"] == 444555666

    def test_columns_match_read_keys(
        self, fake_energy_files: list[tuple[str, Path]]
    ) -> None:
        domains = [
            RaplDomain(name=name, energy_path=path) for name, path in fake_energy_files
        ]
        reader = RaplReader(domains)
        assert set(reader.columns) == set(reader.read().keys())

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        domain = RaplDomain(name="fake", energy_path=tmp_path / "nonexistent")
        reader = RaplReader([domain])
        data = reader.read()
        assert data["rapl_fake_uj"] == ""

    def test_empty_domains(self) -> None:
        reader = RaplReader([])
        assert reader.read() == {}
        assert reader.columns == []
