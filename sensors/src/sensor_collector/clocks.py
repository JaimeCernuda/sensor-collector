"""Read Linux clocks and NTP kernel state via ctypes.

Provides three clock readings (REALTIME, MONOTONIC, MONOTONIC_RAW) and
the full adjtimex() state — all without any third-party dependencies.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from typing import ClassVar

# Linux clock IDs
CLOCK_REALTIME = 0
CLOCK_MONOTONIC = 1
CLOCK_MONOTONIC_RAW = 4


class _Timespec(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_nsec", ctypes.c_long),
    ]


class _Timex(ctypes.Structure):
    """Linux struct timex for adjtimex(2).

    See <sys/timex.h> and adjtimex(2) man page.
    """

    _fields_ = [
        ("modes", ctypes.c_uint),
        ("offset", ctypes.c_long),
        ("freq", ctypes.c_long),
        ("maxerror", ctypes.c_long),
        ("esterror", ctypes.c_long),
        ("status", ctypes.c_int),
        ("constant", ctypes.c_long),
        ("precision", ctypes.c_long),
        ("tolerance", ctypes.c_long),
        ("time_tv_sec", ctypes.c_long),
        ("time_tv_usec", ctypes.c_long),
        ("tick", ctypes.c_long),
        ("ppsfreq", ctypes.c_long),
        ("jitter", ctypes.c_long),
        ("shift", ctypes.c_int),
        ("stabil", ctypes.c_long),
        ("jitcnt", ctypes.c_long),
        ("calcnt", ctypes.c_long),
        ("errcnt", ctypes.c_long),
        ("stbcnt", ctypes.c_long),
        ("tai", ctypes.c_int),
        # Padding to match kernel struct size
        ("_pad0", ctypes.c_int),
        ("_pad1", ctypes.c_long),
        ("_pad2", ctypes.c_long),
        ("_pad3", ctypes.c_long),
        ("_pad4", ctypes.c_long),
        ("_pad5", ctypes.c_long),
        ("_pad6", ctypes.c_long),
        ("_pad7", ctypes.c_long),
        ("_pad8", ctypes.c_long),
        ("_pad9", ctypes.c_long),
        ("_pad10", ctypes.c_long),
        ("_pad11", ctypes.c_long),
    ]


@dataclass(frozen=True)
class ClockReading:
    """Snapshot of all three Linux clocks taken in rapid succession."""

    ts_realtime_ns: int
    ts_monotonic_ns: int
    ts_mono_raw_ns: int
    mono_minus_raw_ns: int


@dataclass(frozen=True)
class AdjTimexReading:
    """Subset of adjtimex() fields relevant to drift measurement."""

    freq: int  # Frequency correction (scaled PPM: value / 2^16 = PPM)
    offset: int  # Current offset being slewed (ns on modern kernels)
    maxerror: int  # Maximum estimated error (us)
    esterror: int  # Estimated error (us)
    status: int  # NTP status bitmask
    tick: int  # Microseconds per tick


class ClockReader:
    """Read Linux clocks and NTP kernel state via ctypes."""

    COLUMNS: ClassVar[list[str]] = [
        "ts_realtime_ns",
        "ts_monotonic_ns",
        "ts_mono_raw_ns",
        "mono_minus_raw_ns",
        "adj_freq",
        "adj_offset",
        "adj_maxerror",
        "adj_esterror",
        "adj_status",
        "adj_tick",
    ]

    def __init__(self) -> None:
        libc_name = ctypes.util.find_library("c")
        if libc_name is None:
            raise OSError("Cannot find libc")
        self._libc = ctypes.CDLL(libc_name, use_errno=True)

        self._clock_gettime = self._libc.clock_gettime
        self._clock_gettime.argtypes = [ctypes.c_int, ctypes.POINTER(_Timespec)]
        self._clock_gettime.restype = ctypes.c_int

        self._adjtimex = self._libc.adjtimex
        self._adjtimex.argtypes = [ctypes.POINTER(_Timex)]
        self._adjtimex.restype = ctypes.c_int

    def _clock_gettime_ns(self, clock_id: int) -> int:
        """Read a clock and return nanoseconds."""
        ts = _Timespec()
        ret = self._clock_gettime(clock_id, ctypes.byref(ts))
        if ret != 0:
            errno = ctypes.get_errno()
            raise OSError(errno, f"clock_gettime({clock_id}) failed")
        return ts.tv_sec * 1_000_000_000 + ts.tv_nsec

    def read_clocks(self) -> ClockReading:
        """Read all three clocks in rapid succession (< 2 us total)."""
        rt = self._clock_gettime_ns(CLOCK_REALTIME)
        mono = self._clock_gettime_ns(CLOCK_MONOTONIC)
        raw = self._clock_gettime_ns(CLOCK_MONOTONIC_RAW)
        return ClockReading(
            ts_realtime_ns=rt,
            ts_monotonic_ns=mono,
            ts_mono_raw_ns=raw,
            mono_minus_raw_ns=mono - raw,
        )

    def read_adjtimex(self) -> AdjTimexReading:
        """Read kernel NTP adjustment state via adjtimex(2)."""
        tx = _Timex()
        tx.modes = 0  # Read-only query
        ret = self._adjtimex(ctypes.byref(tx))
        if ret < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, "adjtimex() failed")
        return AdjTimexReading(
            freq=tx.freq,
            offset=tx.offset,
            maxerror=tx.maxerror,
            esterror=tx.esterror,
            status=tx.status,
            tick=tx.tick,
        )

    def read(self) -> dict[str, int | float | str]:
        """Read clocks and adjtimex, return flat dict matching COLUMNS."""
        clk = self.read_clocks()
        adj = self.read_adjtimex()
        return {
            "ts_realtime_ns": clk.ts_realtime_ns,
            "ts_monotonic_ns": clk.ts_monotonic_ns,
            "ts_mono_raw_ns": clk.ts_mono_raw_ns,
            "mono_minus_raw_ns": clk.mono_minus_raw_ns,
            "adj_freq": adj.freq,
            "adj_offset": adj.offset,
            "adj_maxerror": adj.maxerror,
            "adj_esterror": adj.esterror,
            "adj_status": adj.status,
            "adj_tick": adj.tick,
        }
