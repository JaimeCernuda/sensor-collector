"""Peer clock UDP probing for pairwise relative clock offsets.

Implements an NTP-style 4-timestamp exchange over UDP to measure
inter-node clock offsets and round-trip times, independent of NTP.

Protocol (binary, network byte order):
    Probe    (9 bytes):  type=0x01 (1B) + t1_ns (8B signed big-endian)
    Response (25 bytes): type=0x02 (1B) + t1_ns (8B) + t2_ns (8B) + t3_ns (8B)

Offset = ((T2 - T1) + (T3 - T4)) // 2
RTT    = (T4 - T1) - (T3 - T2)
"""

from __future__ import annotations

import contextlib
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Protocol constants
MSG_PROBE: int = 0x01
MSG_RESPONSE: int = 0x02

# struct formats (network byte order, big-endian)
PROBE_FMT = "!Bq"  # type (1B) + t1_ns (8B signed)
PROBE_SIZE = struct.calcsize(PROBE_FMT)  # 9 bytes

RESPONSE_FMT = "!Bqqq"  # type (1B) + t1_ns + t2_ns + t3_ns (each 8B signed)
RESPONSE_SIZE = struct.calcsize(RESPONSE_FMT)  # 25 bytes

# Default probe interval per peer
_PROBE_INTERVAL_S = 2.0

# Socket receive timeout (so threads can check stop events)
_RECV_TIMEOUT_S = 1.0

# Maximum expected UDP datagram size
_MAX_DGRAM = 64


@dataclass
class PeerConfig:
    """Configuration for a single peer node."""

    name: str
    host: str
    port: int


def parse_peer_arg(arg: str) -> list[PeerConfig]:
    """Parse a peer specification string into a list of PeerConfig.

    Format: ``name=host:port[,name=host:port,...]``

    Args:
        arg: Comma-separated peer specifications.

    Returns:
        List of parsed PeerConfig objects.

    Raises:
        ValueError: If the format is invalid.
    """
    if not arg or not arg.strip():
        return []

    peers: list[PeerConfig] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue

        if "=" not in part:
            raise ValueError(f"Invalid peer spec {part!r}: expected name=host:port")

        name, rest = part.split("=", 1)
        name = name.strip()
        rest = rest.strip()

        if not name:
            raise ValueError(f"Empty peer name in {part!r}")

        if ":" not in rest:
            raise ValueError(f"Invalid peer address {rest!r}: expected host:port")

        host, port_str = rest.rsplit(":", 1)
        host = host.strip()

        try:
            port = int(port_str)
        except ValueError:
            raise ValueError(f"Invalid port {port_str!r} in {part!r}") from None

        if not (1 <= port <= 65535):
            raise ValueError(f"Port {port} out of range in {part!r}")

        if not host:
            raise ValueError(f"Empty host in {part!r}")

        peers.append(PeerConfig(name=name, host=host, port=port))

    return peers


def pack_probe(t1_ns: int) -> bytes:
    """Pack a probe message."""
    return struct.pack(PROBE_FMT, MSG_PROBE, t1_ns)


def unpack_probe(data: bytes) -> int:
    """Unpack a probe message, returning t1_ns.

    Raises:
        ValueError: If the message is malformed.
    """
    if len(data) < PROBE_SIZE:
        raise ValueError(f"Probe too short: {len(data)} bytes")
    msg_type, t1_ns = struct.unpack(PROBE_FMT, data[:PROBE_SIZE])
    if msg_type != MSG_PROBE:
        raise ValueError(f"Expected probe type 0x01, got 0x{msg_type:02x}")
    return t1_ns


def pack_response(t1_ns: int, t2_ns: int, t3_ns: int) -> bytes:
    """Pack a response message."""
    return struct.pack(RESPONSE_FMT, MSG_RESPONSE, t1_ns, t2_ns, t3_ns)


def unpack_response(data: bytes) -> tuple[int, int, int]:
    """Unpack a response message, returning (t1_ns, t2_ns, t3_ns).

    Raises:
        ValueError: If the message is malformed.
    """
    if len(data) < RESPONSE_SIZE:
        raise ValueError(f"Response too short: {len(data)} bytes")
    msg_type, t1_ns, t2_ns, t3_ns = struct.unpack(RESPONSE_FMT, data[:RESPONSE_SIZE])
    if msg_type != MSG_RESPONSE:
        raise ValueError(f"Expected response type 0x02, got 0x{msg_type:02x}")
    return t1_ns, t2_ns, t3_ns


class PeerClockReader:
    """UDP peer clock offset reader with background server and prober threads.

    Starts a UDP server (daemon thread) that responds to incoming probes,
    and a prober thread (daemon) that periodically sends probes to each
    configured peer and computes offset/RTT from the 4-timestamp exchange.
    """

    def __init__(
        self,
        peers: list[PeerConfig],
        listen_port: int = 19777,
    ) -> None:
        self._peers = list(peers)
        self._listen_port = listen_port

        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Latest offset/RTT per peer name, updated by the prober thread.
        self._latest: dict[str, dict[str, int | float | str]] = {}
        self._last_ok: dict[str, float] = {}  # monotonic timestamps

        self._columns = self._build_columns()

        # Create the shared UDP socket.
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(_RECV_TIMEOUT_S)
        self._sock.bind(("0.0.0.0", self._listen_port))

        log.info("Peer clock: listening on UDP port %d", self._listen_port)

        # Start background threads.
        self._server_thread = threading.Thread(
            target=self._server_loop,
            name="peer-clock-server",
            daemon=True,
        )
        self._server_thread.start()

        self._prober_thread = threading.Thread(
            target=self._prober_loop,
            name="peer-clock-prober",
            daemon=True,
        )
        self._prober_thread.start()

    def _build_columns(self) -> list[str]:
        """Build the ordered list of output column names."""
        cols: list[str] = []
        for peer in self._peers:
            cols.append(f"peer_{peer.name}_offset_ns")
            cols.append(f"peer_{peer.name}_rtt_ns")
        cols.append("peer_age_ms")
        return cols

    @property
    def columns(self) -> list[str]:
        """Return the list of CSV column names for this reader."""
        return list(self._columns)

    # ------------------------------------------------------------------
    # Server thread: respond to incoming probes
    # ------------------------------------------------------------------

    def _server_loop(self) -> None:
        """Listen for incoming probe packets and send responses."""
        while not self._stop_event.is_set():
            try:
                data, addr = self._sock.recvfrom(_MAX_DGRAM)
            except TimeoutError:
                continue
            except OSError:
                if self._stop_event.is_set():
                    break
                log.debug("Peer clock server recv error", exc_info=True)
                continue

            t2_ns = time.time_ns()

            if len(data) < PROBE_SIZE:
                continue

            try:
                msg_type = data[0]
            except IndexError:
                continue

            if msg_type != MSG_PROBE:
                continue

            try:
                t1_ns = unpack_probe(data)
            except ValueError:
                continue

            t3_ns = time.time_ns()
            response = pack_response(t1_ns, t2_ns, t3_ns)

            try:
                self._sock.sendto(response, addr)
            except OSError:
                log.debug("Peer clock: failed to send response to %s", addr)

    # ------------------------------------------------------------------
    # Prober thread: send probes and compute offsets
    # ------------------------------------------------------------------

    def _prober_loop(self) -> None:
        """Periodically probe each peer and compute offset/RTT."""
        # Use a separate socket for sending probes and receiving responses,
        # so we don't mix server traffic with probe responses.
        probe_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe_sock.settimeout(_RECV_TIMEOUT_S)

        try:
            while not self._stop_event.is_set():
                for peer in self._peers:
                    if self._stop_event.is_set():
                        break
                    self._probe_one(probe_sock, peer)

                # Wait for the probe interval, but check stop event.
                self._stop_event.wait(timeout=_PROBE_INTERVAL_S)
        finally:
            probe_sock.close()

    def _probe_one(
        self,
        sock: socket.socket,
        peer: PeerConfig,
    ) -> None:
        """Send a single probe to a peer and process the response."""
        t1_ns = time.time_ns()
        probe = pack_probe(t1_ns)

        try:
            sock.sendto(probe, (peer.host, peer.port))
        except OSError:
            log.debug("Peer clock: failed to send probe to %s", peer.name)
            return

        try:
            data, _addr = sock.recvfrom(_MAX_DGRAM)
        except TimeoutError:
            log.debug("Peer clock: timeout waiting for %s", peer.name)
            return
        except OSError:
            log.debug("Peer clock: recv error from %s", peer.name)
            return

        t4_ns = time.time_ns()

        try:
            resp_t1, t2_ns, t3_ns = unpack_response(data)
        except ValueError:
            log.debug("Peer clock: bad response from %s", peer.name)
            return

        # Verify the response matches our probe.
        if resp_t1 != t1_ns:
            log.debug("Peer clock: t1 mismatch from %s", peer.name)
            return

        offset_ns = ((t2_ns - t1_ns) + (t3_ns - t4_ns)) // 2
        rtt_ns = (t4_ns - t1_ns) - (t3_ns - t2_ns)

        now = time.monotonic()
        with self._lock:
            self._latest[peer.name] = {
                f"peer_{peer.name}_offset_ns": offset_ns,
                f"peer_{peer.name}_rtt_ns": rtt_ns,
            }
            self._last_ok[peer.name] = now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read(self) -> dict[str, int | float | str]:
        """Return the latest cached peer clock values.

        Returns:
            A dict with keys from :attr:`columns`.  Contains offset_ns
            and rtt_ns per peer, plus a single ``peer_age_ms`` reporting
            the maximum age across all peers.  Values are empty strings
            if no data is available yet.
        """
        with self._lock:
            latest = {k: dict(v) for k, v in self._latest.items()}
            last_ok = dict(self._last_ok)

        now = time.monotonic()
        result: dict[str, int | float | str] = {}

        max_age_ms = 0.0
        any_data = False

        for peer in self._peers:
            offset_col = f"peer_{peer.name}_offset_ns"
            rtt_col = f"peer_{peer.name}_rtt_ns"

            peer_data = latest.get(peer.name, {})
            if peer_data:
                any_data = True
                result[offset_col] = peer_data.get(offset_col, "")
                result[rtt_col] = peer_data.get(rtt_col, "")

                peer_age_ms = (now - last_ok.get(peer.name, now)) * 1000.0
                if peer_age_ms > max_age_ms:
                    max_age_ms = peer_age_ms
            else:
                result[offset_col] = ""
                result[rtt_col] = ""

        if any_data:
            result["peer_age_ms"] = round(max_age_ms, 1)
        else:
            result["peer_age_ms"] = ""

        return result

    def stop(self) -> None:
        """Stop the background threads and close the socket."""
        self._stop_event.set()

        # Close the socket to unblock any recvfrom calls.
        with contextlib.suppress(OSError):
            self._sock.close()

        if self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)
        if self._prober_thread.is_alive():
            self._prober_thread.join(timeout=5.0)
