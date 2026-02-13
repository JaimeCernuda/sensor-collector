"""RAPL energy counters from sysfs powercap interface.

Reads Intel Running Average Power Limit (RAPL) energy counters from
/sys/class/powercap/intel-rapl:*/.  Requires root on most systems.
Handles PermissionError gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class RaplDomain:
    """Description of a single RAPL energy domain or subdomain."""

    name: str  # Contents of the "name" file (e.g. "package-0", "core")
    energy_path: Path  # Full path to the energy_uj file


class RaplReader:
    """Read RAPL energy counters from sysfs.

    Each domain produces a column ``rapl_{name}_uj`` with the cumulative
    energy in microjoules.  On systems where RAPL is not accessible (no
    permission or not available), columns return empty strings.
    """

    COLUMNS: ClassVar[list[str]] = []  # populated per-instance via property

    def __init__(self, domains: list[RaplDomain]) -> None:
        self._domains = domains
        self._columns = [f"rapl_{d.name}_uj" for d in self._domains]

    @property
    def columns(self) -> list[str]:
        """Return the column names for this reader instance."""
        return list(self._columns)

    @classmethod
    def discover_rapl(cls, sysfs_root: str = "/sys/class/powercap") -> list[RaplDomain]:
        """Discover available RAPL energy domains in sysfs.

        Scans both top-level domains (``intel-rapl:N``) and subdomains
        (``intel-rapl:N:M``).  Returns an empty list if the powercap
        directory does not exist or is not readable.

        Args:
            sysfs_root: Base path to the powercap class directory.

        Returns:
            List of RaplDomain descriptors.  Empty if RAPL is unavailable.
        """
        root = Path(sysfs_root)
        domains: list[RaplDomain] = []

        if not root.is_dir():
            return domains

        # Collect both top-level and subdomain directories
        rapl_dirs: list[Path] = []
        try:
            for entry in sorted(root.iterdir()):
                if entry.is_dir() and entry.name.startswith("intel-rapl:"):
                    rapl_dirs.append(entry)
                    # Check for subdomains
                    try:
                        for sub_entry in sorted(entry.iterdir()):
                            if sub_entry.is_dir() and sub_entry.name.startswith(
                                "intel-rapl:"
                            ):
                                rapl_dirs.append(sub_entry)
                    except PermissionError:
                        continue
        except PermissionError:
            return domains

        for rapl_dir in rapl_dirs:
            energy_path = rapl_dir / "energy_uj"
            name_path = rapl_dir / "name"

            # Both files must be present
            try:
                if not energy_path.exists():
                    continue
            except PermissionError:
                continue

            try:
                name = name_path.read_text().strip()
            except (FileNotFoundError, PermissionError):
                name = rapl_dir.name

            # Sanitise for column name
            name = name.replace(" ", "_").replace("-", "_").lower()

            # Disambiguate: if this is a subdomain, prefix with the parent domain name
            parent = rapl_dir.parent
            if parent.name.startswith("intel-rapl:") and parent != root:
                try:
                    parent_name = (parent / "name").read_text().strip()
                    parent_name = (
                        parent_name.replace(" ", "_").replace("-", "_").lower()
                    )
                    name = f"{parent_name}_{name}"
                except (FileNotFoundError, PermissionError):
                    pass

            # Verify we can actually read the energy counter
            try:
                energy_path.read_text()
            except PermissionError:
                continue
            except FileNotFoundError:
                continue

            domains.append(RaplDomain(name=name, energy_path=energy_path))

        return domains

    def read(self) -> dict[str, int | float | str]:
        """Read cumulative RAPL energy counters.

        Returns:
            Dict mapping column names to cumulative energy in microjoules
            (int), or empty string if a domain could not be read.
        """
        result: dict[str, int | float | str] = {}
        for domain, col in zip(self._domains, self._columns, strict=True):
            try:
                raw = domain.energy_path.read_text().strip()
                result[col] = int(raw)
            except (FileNotFoundError, PermissionError, ValueError):
                result[col] = ""
        return result
