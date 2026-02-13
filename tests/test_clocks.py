"""Tests for the ClockReader module."""

from __future__ import annotations

import ctypes
import sys

import pytest

from sensor_collector.clocks import (
    ClockReader,
    _Timespec,
    _Timex,
)

linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux only")


class TestColumns:
    """Tests for ClockReader.COLUMNS class variable."""

    def test_columns_is_list(self) -> None:
        assert isinstance(ClockReader.COLUMNS, list)

    def test_columns_has_ten_entries(self) -> None:
        assert len(ClockReader.COLUMNS) == 10

    def test_columns_expected_names(self) -> None:
        expected = [
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
        assert expected == ClockReader.COLUMNS

    def test_columns_all_strings(self) -> None:
        for col in ClockReader.COLUMNS:
            assert isinstance(col, str)

    def test_columns_no_duplicates(self) -> None:
        assert len(ClockReader.COLUMNS) == len(set(ClockReader.COLUMNS))


class TestCtypesStructures:
    """Tests for _Timespec and _Timex ctypes structures."""

    def test_timespec_has_tv_sec_field(self) -> None:
        ts = _Timespec()
        ts.tv_sec = 42
        assert ts.tv_sec == 42

    def test_timespec_has_tv_nsec_field(self) -> None:
        ts = _Timespec()
        ts.tv_nsec = 123456789
        assert ts.tv_nsec == 123456789

    def test_timespec_field_count(self) -> None:
        assert len(_Timespec._fields_) == 2

    def test_timespec_size_is_two_longs(self) -> None:
        expected = 2 * ctypes.sizeof(ctypes.c_long)
        assert ctypes.sizeof(_Timespec) == expected

    def test_timex_has_modes_field(self) -> None:
        tx = _Timex()
        tx.modes = 0
        assert tx.modes == 0

    def test_timex_has_tick_field(self) -> None:
        tx = _Timex()
        tx.tick = 10000
        assert tx.tick == 10000

    def test_timex_has_freq_field(self) -> None:
        tx = _Timex()
        tx.freq = 12345
        assert tx.freq == 12345

    def test_timex_has_offset_field(self) -> None:
        tx = _Timex()
        tx.offset = -100
        assert tx.offset == -100

    def test_timex_has_status_field(self) -> None:
        tx = _Timex()
        tx.status = 1
        assert tx.status == 1

    def test_timex_field_count(self) -> None:
        # 21 real fields + 12 padding fields = 33
        assert len(_Timex._fields_) == 33

    def test_timex_field_names_include_key_fields(self) -> None:
        names = [name for name, _ in _Timex._fields_]
        for key in ("modes", "offset", "freq", "tick", "status", "precision"):
            assert key in names


@linux_only
class TestReadClocks:
    """Tests for ClockReader.read_clocks() on Linux."""

    def test_reader_creation(self) -> None:
        reader = ClockReader()
        assert reader is not None

    def test_read_clocks_returns_positive_nanoseconds(self) -> None:
        reader = ClockReader()
        clk = reader.read_clocks()
        assert clk.ts_realtime_ns > 0
        assert clk.ts_monotonic_ns > 0
        assert clk.ts_mono_raw_ns > 0

    def test_realtime_positive(self) -> None:
        reader = ClockReader()
        clk = reader.read_clocks()
        assert clk.ts_realtime_ns > 0

    def test_monotonic_positive(self) -> None:
        reader = ClockReader()
        clk = reader.read_clocks()
        assert clk.ts_monotonic_ns > 0

    def test_mono_raw_positive(self) -> None:
        reader = ClockReader()
        clk = reader.read_clocks()
        assert clk.ts_mono_raw_ns > 0

    def test_mono_minus_raw_computed_correctly(self) -> None:
        reader = ClockReader()
        clk = reader.read_clocks()
        assert clk.mono_minus_raw_ns == clk.ts_monotonic_ns - clk.ts_mono_raw_ns

    def test_realtime_is_plausible_epoch(self) -> None:
        """Realtime clock should be after 2020-01-01 in nanoseconds."""
        reader = ClockReader()
        clk = reader.read_clocks()
        jan_2020_ns = 1_577_836_800 * 1_000_000_000
        assert clk.ts_realtime_ns > jan_2020_ns

    def test_all_values_are_ints(self) -> None:
        reader = ClockReader()
        clk = reader.read_clocks()
        assert isinstance(clk.ts_realtime_ns, int)
        assert isinstance(clk.ts_monotonic_ns, int)
        assert isinstance(clk.ts_mono_raw_ns, int)
        assert isinstance(clk.mono_minus_raw_ns, int)


@linux_only
class TestReadAdjtimex:
    """Tests for ClockReader.read_adjtimex() on Linux."""

    def test_tick_near_10000(self) -> None:
        """Default tick should be near 10000 us/tick."""
        reader = ClockReader()
        adj = reader.read_adjtimex()
        assert 9000 <= adj.tick <= 11000

    def test_returns_adjtimex_reading(self) -> None:
        from sensor_collector.clocks import AdjTimexReading

        reader = ClockReader()
        adj = reader.read_adjtimex()
        assert isinstance(adj, AdjTimexReading)

    def test_maxerror_non_negative(self) -> None:
        reader = ClockReader()
        adj = reader.read_adjtimex()
        assert adj.maxerror >= 0

    def test_esterror_non_negative(self) -> None:
        reader = ClockReader()
        adj = reader.read_adjtimex()
        assert adj.esterror >= 0

    def test_status_is_int(self) -> None:
        reader = ClockReader()
        adj = reader.read_adjtimex()
        assert isinstance(adj.status, int)

    def test_freq_is_int(self) -> None:
        reader = ClockReader()
        adj = reader.read_adjtimex()
        assert isinstance(adj.freq, int)


@linux_only
class TestReadAll:
    """Tests for ClockReader.read_all() on Linux."""

    def test_returns_dict(self) -> None:
        reader = ClockReader()
        result = reader.read_all()
        assert isinstance(result, dict)

    def test_keys_match_columns(self) -> None:
        reader = ClockReader()
        result = reader.read_all()
        assert set(result.keys()) == set(ClockReader.COLUMNS)

    def test_all_columns_present(self) -> None:
        reader = ClockReader()
        result = reader.read_all()
        for col in ClockReader.COLUMNS:
            assert col in result

    def test_all_values_are_ints(self) -> None:
        reader = ClockReader()
        result = reader.read_all()
        for key, value in result.items():
            assert isinstance(value, int), f"{key} is {type(value)}, expected int"

    def test_clock_values_positive(self) -> None:
        reader = ClockReader()
        result = reader.read_all()
        assert result["ts_realtime_ns"] > 0
        assert result["ts_monotonic_ns"] > 0
        assert result["ts_mono_raw_ns"] > 0

    def test_adj_tick_near_10000(self) -> None:
        reader = ClockReader()
        result = reader.read_all()
        assert 9000 <= result["adj_tick"] <= 11000
