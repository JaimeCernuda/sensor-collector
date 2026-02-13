"""Tests for the chrony NTP tracking reader."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from sensor_collector.chrony import ChronyReader, parse_tracking_line

# -- Synthetic chronyc -c tracking output matching field indices --
# Indices: 0=ref_id, 1=ref_ip, 2=stratum, 3=ref_time,
#          4=system_time(s), 5=last_offset(s), 6=rms_offset(s),
#          7=frequency(ppm), 8=resid_freq(ppm), 9=skew(ppm),
#          10=root_delay(s), 11=root_dispersion(s),
#          12=update_interval(s), 13=leap_status
# fmt: off
SAMPLE_LINE = ",".join([
    "C3B20501",          # 0  ref_id
    "195.178.5.1",       # 1  ref_ip
    "2",                 # 2  stratum
    "1707843210.123",    # 3  ref_time
    "0.000012345",       # 4  system_time (seconds)
    "-0.000000456",      # 5  last_offset (seconds)
    "0.000000789",       # 6  rms_offset (seconds)
    "0.012345",          # 7  frequency (PPM)
    "0.000678",          # 8  residual freq (PPM)
    "0.001234",          # 9  skew (PPM)
    "0.004567",          # 10 root delay (seconds)
    "0.000890",          # 11 root dispersion (seconds)
    "64.0",              # 12 update interval
    "Normal",            # 13 leap status
])
# fmt: on

S_TO_NS = 1_000_000_000.0

_MOD = "sensor_collector.chrony"


class TestParseTrackingLine:
    """Unit tests for parse_tracking_line()."""

    def test_valid_line(self) -> None:
        result = parse_tracking_line(SAMPLE_LINE)

        assert result["chrony_stratum"] == 2
        assert result["chrony_freq_ppm"] == pytest.approx(0.012345)
        assert result["chrony_skew_ppm"] == pytest.approx(0.001234)
        assert result["chrony_root_delay_ns"] == pytest.approx(0.004567 * S_TO_NS)
        assert result["chrony_root_dispersion_ns"] == pytest.approx(0.000890 * S_TO_NS)
        assert result["chrony_offset_ns"] == pytest.approx(-0.000000456 * S_TO_NS)

    def test_empty_line_returns_empty_strings(self) -> None:
        result = parse_tracking_line("")
        for col in ChronyReader.COLUMNS:
            assert result[col] == ""

    def test_short_line_returns_empty_strings(self) -> None:
        result = parse_tracking_line("a,b,c")
        for col in ChronyReader.COLUMNS:
            assert result[col] == ""

    def test_non_numeric_fields_return_empty_strings(self) -> None:
        bad = "ref,name,notanint,time,freq,res,skew,delay,disp,upd,offset"
        result = parse_tracking_line(bad)
        for col in ChronyReader.COLUMNS:
            assert result[col] == ""

    def test_all_columns_present(self) -> None:
        result = parse_tracking_line(SAMPLE_LINE)
        for col in ChronyReader.COLUMNS:
            assert col in result

    def test_stratum_is_int(self) -> None:
        result = parse_tracking_line(SAMPLE_LINE)
        assert isinstance(result["chrony_stratum"], int)


class TestChronyReader:
    """Unit tests for the ChronyReader class."""

    def test_columns_class_var(self) -> None:
        assert isinstance(ChronyReader.COLUMNS, list)
        assert len(ChronyReader.COLUMNS) == 6
        assert "chrony_freq_ppm" in ChronyReader.COLUMNS
        assert "chrony_stratum" in ChronyReader.COLUMNS

    @patch(f"{_MOD}.shutil.which", return_value=None)
    def test_is_available_false_when_not_installed(self, mock_which: MagicMock) -> None:
        assert ChronyReader.is_available() is False

    @patch(f"{_MOD}.shutil.which", return_value="/usr/bin/chronyc")
    def test_is_available_true_when_installed(self, mock_which: MagicMock) -> None:
        assert ChronyReader.is_available() is True

    @patch(f"{_MOD}.subprocess.run")
    def test_read_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=SAMPLE_LINE + "\n",
        )
        reader = ChronyReader()
        result = reader.read()

        assert result["chrony_stratum"] == 2
        assert isinstance(result["chrony_freq_ppm"], float)
        mock_run.assert_called_once_with(
            ["chronyc", "-c", "tracking"],
            capture_output=True,
            text=True,
            timeout=2,
        )

    @patch(
        f"{_MOD}.subprocess.run",
        side_effect=FileNotFoundError,
    )
    def test_read_chronyc_not_found(self, mock_run: MagicMock) -> None:
        reader = ChronyReader()
        result = reader.read()
        for col in ChronyReader.COLUMNS:
            assert result[col] == ""

    @patch(
        f"{_MOD}.subprocess.run",
        side_effect=subprocess.TimeoutExpired("chronyc", 2),
    )
    def test_read_timeout(self, mock_run: MagicMock) -> None:
        reader = ChronyReader()
        result = reader.read()
        for col in ChronyReader.COLUMNS:
            assert result[col] == ""

    @patch(f"{_MOD}.subprocess.run")
    def test_read_nonzero_exit(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
        reader = ChronyReader()
        result = reader.read()
        for col in ChronyReader.COLUMNS:
            assert result[col] == ""

    @patch(f"{_MOD}.subprocess.run")
    def test_read_empty_stdout(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="")
        reader = ChronyReader()
        result = reader.read()
        for col in ChronyReader.COLUMNS:
            assert result[col] == ""
