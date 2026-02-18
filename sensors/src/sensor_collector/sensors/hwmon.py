"""Hardware monitor temperatures from sysfs hwmon interface.

Walks /sys/class/hwmon/hwmon*/ to discover temperature sensors, then reads
millidegree values and converts to Celsius floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class HwmonSensor:
    """Description of a single hwmon temperature input."""

    path: Path  # Full path to temp*_input file
    name: str  # Contents of the hwmon device's "name" file
    label: str  # Contents of temp*_label, or fallback index like "temp1"


class HwmonReader:
    """Read hardware monitor temperatures from sysfs.

    Each sensor produces a column named ``temp_{name}_{label}_c`` with the
    temperature in degrees Celsius (float, millidegrees / 1000).
    """

    COLUMNS: ClassVar[list[str]] = []  # populated per-instance via property

    def __init__(self, sensors: list[HwmonSensor]) -> None:
        self._sensors = sensors
        self._columns = [f"temp_{s.name}_{s.label}_c" for s in self._sensors]

    @property
    def columns(self) -> list[str]:
        """Return the column names for this reader instance."""
        return list(self._columns)

    @classmethod
    def discover_hwmon(cls, sysfs_root: str = "/sys/class/hwmon") -> list[HwmonSensor]:
        """Walk sysfs to discover available hwmon temperature sensors.

        Args:
            sysfs_root: Base path to the hwmon class directory.

        Returns:
            A list of discovered HwmonSensor descriptors.
        """
        root = Path(sysfs_root)
        sensors: list[HwmonSensor] = []

        if not root.is_dir():
            return sensors

        for hwmon_dir in sorted(root.iterdir()):
            if not hwmon_dir.is_dir():
                continue

            # Read the device name
            name_file = hwmon_dir / "name"
            try:
                name = name_file.read_text().strip()
            except (FileNotFoundError, PermissionError):
                name = hwmon_dir.name

            # Sanitise name for column usage
            name = name.replace(" ", "_").replace("-", "_").lower()

            # Find all temp*_input files
            for input_file in sorted(hwmon_dir.glob("temp*_input")):
                # Derive the index stem, e.g. "temp1" from "temp1_input"
                stem = input_file.name.removesuffix("_input")

                # Try to read the corresponding label file
                label_file = hwmon_dir / f"{stem}_label"
                try:
                    label = label_file.read_text().strip()
                except (FileNotFoundError, PermissionError):
                    label = stem  # fallback: "temp1", "temp2", ...

                label = label.replace(" ", "_").replace("-", "_").lower()

                sensors.append(HwmonSensor(path=input_file, name=name, label=label))

        return sensors

    def read(self) -> dict[str, int | float | str]:
        """Read all hwmon temperature sensors.

        Returns:
            Dict mapping column names to Celsius float values, or empty string
            if a sensor could not be read.
        """
        result: dict[str, int | float | str] = {}
        for sensor, col in zip(self._sensors, self._columns):
            try:
                raw = sensor.path.read_text().strip()
                result[col] = int(raw) / 1000.0
            except (FileNotFoundError, PermissionError, ValueError):
                result[col] = ""
        return result
