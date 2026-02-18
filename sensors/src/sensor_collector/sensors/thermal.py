"""Thermal zone temperatures from sysfs.

Reads /sys/class/thermal/thermal_zone{N}/temp (millidegrees) and type,
reporting Celsius floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class ThermalZone:
    """Description of a single thermal zone."""

    index: int  # Zone number
    zone_type: str  # Contents of the "type" file (e.g. "x86_pkg_temp")
    temp_path: Path  # Full path to the "temp" file


class ThermalReader:
    """Read thermal zone temperatures from sysfs.

    Each zone produces a column ``tz_{type}_{N}_c`` with the temperature
    in degrees Celsius (float, millidegrees / 1000).
    """

    COLUMNS: ClassVar[list[str]] = []  # populated per-instance via property

    def __init__(self, zones: list[ThermalZone]) -> None:
        self._zones = zones
        self._columns = [f"tz_{z.zone_type}_{z.index}_c" for z in self._zones]

    @property
    def columns(self) -> list[str]:
        """Return the column names for this reader instance."""
        return list(self._columns)

    @classmethod
    def discover_thermal(
        cls, sysfs_root: str = "/sys/class/thermal"
    ) -> list[ThermalZone]:
        """Discover available thermal zones in sysfs.

        Args:
            sysfs_root: Base path to the thermal class directory.

        Returns:
            A list of discovered ThermalZone descriptors, sorted by index.
        """
        root = Path(sysfs_root)
        zones: list[ThermalZone] = []

        if not root.is_dir():
            return zones

        for entry in sorted(root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("thermal_zone"):
                continue
            suffix = entry.name[len("thermal_zone") :]
            if not suffix.isdigit():
                continue
            index = int(suffix)

            temp_path = entry / "temp"
            if not temp_path.exists():
                continue

            type_path = entry / "type"
            try:
                zone_type = type_path.read_text().strip()
            except (FileNotFoundError, PermissionError):
                zone_type = f"zone{index}"

            # Sanitise for column name
            zone_type = zone_type.replace(" ", "_").replace("-", "_").lower()

            zones.append(
                ThermalZone(index=index, zone_type=zone_type, temp_path=temp_path)
            )

        return sorted(zones, key=lambda z: z.index)

    def read(self) -> dict[str, int | float | str]:
        """Read all thermal zone temperatures.

        Returns:
            Dict mapping column names to Celsius float values, or empty string
            if a zone could not be read.
        """
        result: dict[str, int | float | str] = {}
        for zone, col in zip(self._zones, self._columns):
            try:
                raw = zone.temp_path.read_text().strip()
                result[col] = int(raw) / 1000.0
            except (FileNotFoundError, PermissionError, ValueError):
                result[col] = ""
        return result
