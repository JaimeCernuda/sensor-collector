"""Tests for the network sensor reader."""

from __future__ import annotations

from pathlib import Path

import pytest

from sensor_collector.sensors.network import NetworkReader


@pytest.fixture()
def fake_net(tmp_path: Path) -> Path:
    """Create a fake /sys/class/net tree."""
    for iface in ["eth0", "wlan0"]:
        stats_dir = tmp_path / iface / "statistics"
        stats_dir.mkdir(parents=True)
        (stats_dir / "rx_bytes").write_text("123456789\n")
        (stats_dir / "tx_bytes").write_text("987654321\n")
        (stats_dir / "rx_packets").write_text("100000\n")
        (stats_dir / "tx_packets").write_text("200000\n")

    # lo interface (should be skipped by default)
    lo_stats = tmp_path / "lo" / "statistics"
    lo_stats.mkdir(parents=True)
    (lo_stats / "rx_bytes").write_text("0\n")
    (lo_stats / "tx_bytes").write_text("0\n")
    (lo_stats / "rx_packets").write_text("0\n")
    (lo_stats / "tx_packets").write_text("0\n")

    return tmp_path


class TestDiscoverInterfaces:
    """Tests for NetworkReader.discover_interfaces()."""

    def test_discovers_interfaces(self, fake_net: Path) -> None:
        ifaces = NetworkReader.discover_interfaces(
            str(fake_net), skip_loopback=True, skip_virtual=False
        )
        assert "eth0" in ifaces
        assert "wlan0" in ifaces

    def test_skips_loopback(self, fake_net: Path) -> None:
        ifaces = NetworkReader.discover_interfaces(
            str(fake_net), skip_loopback=True, skip_virtual=False
        )
        assert "lo" not in ifaces

    def test_includes_loopback_when_requested(self, fake_net: Path) -> None:
        ifaces = NetworkReader.discover_interfaces(
            str(fake_net), skip_loopback=False, skip_virtual=False
        )
        assert "lo" in ifaces

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        ifaces = NetworkReader.discover_interfaces(str(tmp_path / "nonexistent"))
        assert ifaces == []


class TestNetworkReader:
    """Tests for NetworkReader.read()."""

    def test_read_values(self, fake_net: Path) -> None:
        reader = NetworkReader(["eth0", "wlan0"])
        # Override paths to point to our fake tree
        reader._paths = []
        reader._columns = []
        for iface in ["eth0", "wlan0"]:
            for stat in ["rx_bytes", "tx_bytes", "rx_packets", "tx_packets"]:
                col = f"net_{iface}_{stat}"
                reader._columns.append(col)
                reader._paths.append((col, fake_net / iface / "statistics" / stat))

        data = reader.read()
        assert data["net_eth0_rx_bytes"] == 123456789
        assert data["net_eth0_tx_bytes"] == 987654321
        assert data["net_wlan0_rx_packets"] == 100000

    def test_columns_count(self) -> None:
        reader = NetworkReader(["eth0"])
        assert len(reader.columns) == 4

    def test_missing_file_returns_empty(self) -> None:
        reader = NetworkReader(["nonexistent0"])
        data = reader.read()
        for col in reader.columns:
            assert data[col] == ""

    def test_empty_interfaces(self) -> None:
        reader = NetworkReader([])
        assert reader.read() == {}
        assert reader.columns == []
