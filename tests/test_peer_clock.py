"""Tests for the peer clock UDP probing module."""

from __future__ import annotations

import struct
import time

import pytest

from sensor_collector.peer_clock import (
    MSG_PROBE,
    MSG_RESPONSE,
    PROBE_FMT,
    PROBE_SIZE,
    RESPONSE_FMT,
    RESPONSE_SIZE,
    PeerClockReader,
    PeerConfig,
    pack_probe,
    pack_response,
    parse_peer_arg,
    unpack_probe,
    unpack_response,
)


class TestParsePeerArg:
    """Tests for parse_peer_arg()."""

    def test_single_peer(self) -> None:
        result = parse_peer_arg("ares=216.47.152.168:19777")
        assert len(result) == 1
        assert result[0].name == "ares"
        assert result[0].host == "216.47.152.168"
        assert result[0].port == 19777

    def test_multiple_peers(self) -> None:
        result = parse_peer_arg(
            "ares=216.47.152.168:19777,chameleon=129.114.108.185:19777"
        )
        assert len(result) == 2
        assert result[0].name == "ares"
        assert result[1].name == "chameleon"
        assert result[1].host == "129.114.108.185"

    def test_empty_string(self) -> None:
        assert parse_peer_arg("") == []

    def test_whitespace_only(self) -> None:
        assert parse_peer_arg("   ") == []

    def test_missing_equals(self) -> None:
        with pytest.raises(ValueError, match="expected name=host:port"):
            parse_peer_arg("ares216.47.152.168:19777")

    def test_missing_port(self) -> None:
        with pytest.raises(ValueError, match="expected host:port"):
            parse_peer_arg("ares=216.47.152.168")

    def test_invalid_port(self) -> None:
        with pytest.raises(ValueError, match="Invalid port"):
            parse_peer_arg("ares=216.47.152.168:notaport")

    def test_port_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            parse_peer_arg("ares=216.47.152.168:99999")

    def test_empty_name(self) -> None:
        with pytest.raises(ValueError, match="Empty peer name"):
            parse_peer_arg("=216.47.152.168:19777")

    def test_empty_host(self) -> None:
        with pytest.raises(ValueError, match="Empty host"):
            parse_peer_arg("ares=:19777")

    def test_with_spaces(self) -> None:
        result = parse_peer_arg(" ares = 216.47.152.168:19777 ")
        assert len(result) == 1
        assert result[0].name == "ares"
        assert result[0].host == "216.47.152.168"


class TestProtocolPacking:
    """Tests for probe/response pack/unpack functions."""

    def test_probe_round_trip(self) -> None:
        t1 = time.time_ns()
        data = pack_probe(t1)
        assert len(data) == PROBE_SIZE
        assert unpack_probe(data) == t1

    def test_probe_format(self) -> None:
        t1 = 1_000_000_000
        data = pack_probe(t1)
        msg_type, value = struct.unpack(PROBE_FMT, data)
        assert msg_type == MSG_PROBE
        assert value == t1

    def test_response_round_trip(self) -> None:
        t1 = time.time_ns()
        t2 = t1 + 1000
        t3 = t1 + 2000
        data = pack_response(t1, t2, t3)
        assert len(data) == RESPONSE_SIZE
        assert unpack_response(data) == (t1, t2, t3)

    def test_response_format(self) -> None:
        t1, t2, t3 = 100, 200, 300
        data = pack_response(t1, t2, t3)
        msg_type, v1, v2, v3 = struct.unpack(RESPONSE_FMT, data)
        assert msg_type == MSG_RESPONSE
        assert (v1, v2, v3) == (t1, t2, t3)

    def test_unpack_probe_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            unpack_probe(b"\x01\x00")

    def test_unpack_probe_wrong_type(self) -> None:
        data = struct.pack(PROBE_FMT, 0x02, 12345)
        with pytest.raises(ValueError, match="Expected probe type"):
            unpack_probe(data)

    def test_unpack_response_too_short(self) -> None:
        with pytest.raises(ValueError, match="too short"):
            unpack_response(b"\x02" + b"\x00" * 10)

    def test_unpack_response_wrong_type(self) -> None:
        data = struct.pack(RESPONSE_FMT, 0x01, 1, 2, 3)
        with pytest.raises(ValueError, match="Expected response type"):
            unpack_response(data)

    def test_negative_timestamp(self) -> None:
        """Signed timestamps should round-trip correctly."""
        t1 = -1_000_000
        data = pack_probe(t1)
        assert unpack_probe(data) == t1


class TestPeerClockReaderColumns:
    """Tests for column generation from peer configs."""

    def test_single_peer_columns(self) -> None:
        peers = [PeerConfig(name="ares", host="1.2.3.4", port=19777)]
        # Don't start the reader (it would bind a socket), just test
        # the column building logic directly.
        expected = [
            "peer_ares_offset_ns",
            "peer_ares_rtt_ns",
            "peer_age_ms",
        ]
        # Use the internal method via a minimal approach
        reader = PeerClockReader.__new__(PeerClockReader)
        reader._peers = peers
        cols = reader._build_columns()
        assert cols == expected

    def test_multiple_peer_columns(self) -> None:
        peers = [
            PeerConfig(name="ares", host="1.2.3.4", port=19777),
            PeerConfig(name="chameleon", host="5.6.7.8", port=19777),
        ]
        reader = PeerClockReader.__new__(PeerClockReader)
        reader._peers = peers
        cols = reader._build_columns()
        assert cols == [
            "peer_ares_offset_ns",
            "peer_ares_rtt_ns",
            "peer_chameleon_offset_ns",
            "peer_chameleon_rtt_ns",
            "peer_age_ms",
        ]


class TestPeerClockReaderLoopback:
    """Integration tests using localhost loopback."""

    def test_loopback_probe(self) -> None:
        """Start a PeerClockReader probing itself on localhost."""
        peers = [PeerConfig(name="self", host="127.0.0.1", port=19778)]
        reader = PeerClockReader(peers, listen_port=19778)
        try:
            # Wait for at least one probe cycle to complete.
            deadline = time.monotonic() + 10.0
            result: dict[str, int | float | str] = {}
            while time.monotonic() < deadline:
                result = reader.read()
                if result.get("peer_self_offset_ns") != "":
                    break
                time.sleep(0.2)

            assert isinstance(result["peer_self_offset_ns"], int)
            assert isinstance(result["peer_self_rtt_ns"], int)
            assert isinstance(result["peer_age_ms"], (int, float))

            # Loopback offset should be very small (< 10ms = 10_000_000 ns).
            offset = result["peer_self_offset_ns"]
            assert isinstance(offset, int)
            assert abs(offset) < 10_000_000

            # RTT should be small (< 10ms).
            rtt = result["peer_self_rtt_ns"]
            assert isinstance(rtt, int)
            assert rtt >= 0
            assert rtt < 10_000_000
        finally:
            reader.stop()

    def test_read_before_probes(self) -> None:
        """read() returns empty strings when no probes have completed."""
        peers = [PeerConfig(name="missing", host="192.0.2.1", port=19779)]
        reader = PeerClockReader(peers, listen_port=19779)
        try:
            result = reader.read()
            assert result["peer_missing_offset_ns"] == ""
            assert result["peer_missing_rtt_ns"] == ""
            assert result["peer_age_ms"] == ""
        finally:
            reader.stop()

    def test_stop_is_idempotent(self) -> None:
        """Calling stop() twice should not raise."""
        peers = [PeerConfig(name="test", host="127.0.0.1", port=19780)]
        reader = PeerClockReader(peers, listen_port=19780)
        reader.stop()
        reader.stop()  # Should not raise

    def test_columns_property(self) -> None:
        """columns property returns a copy of the column list."""
        peers = [PeerConfig(name="node1", host="127.0.0.1", port=19781)]
        reader = PeerClockReader(peers, listen_port=19781)
        try:
            cols = reader.columns
            assert "peer_node1_offset_ns" in cols
            assert "peer_node1_rtt_ns" in cols
            assert "peer_age_ms" in cols
            # Verify it's a copy
            cols.append("extra")
            assert "extra" not in reader.columns
        finally:
            reader.stop()
