"""Tests for data preprocessing module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tick2.data.preprocessing import (
    TARGET_COL,
    _deduplicate_columns,
    _is_cumulative,
    _is_excluded,
    categorize_column,
    get_feature_cols,
    load_machine,
)


class TestDeduplicateColumns:
    def test_no_duplicates(self) -> None:
        cols = ["a", "b", "c"]
        assert _deduplicate_columns(cols) == ["a", "b", "c"]

    def test_single_duplicate(self) -> None:
        cols = ["a", "b", "a"]
        result = _deduplicate_columns(cols)
        assert result == ["a", "b", "a_sock1"]

    def test_triple_duplicate(self) -> None:
        cols = ["x", "x", "x"]
        result = _deduplicate_columns(cols)
        assert len(set(result)) == 3  # all unique
        assert result[0] == "x"
        assert result[1] == "x_sock1"

    def test_empty(self) -> None:
        assert _deduplicate_columns([]) == []


class TestIsCumulative:
    def test_cstate_cumulative(self) -> None:
        assert _is_cumulative("cstate_C1_time")

    def test_rapl_cumulative(self) -> None:
        assert _is_cumulative("rapl_pkg_energy")

    def test_network_cumulative(self) -> None:
        assert _is_cumulative("net_eth0_rx_bytes")

    def test_disk_cumulative(self) -> None:
        assert _is_cumulative("disk_sda_reads")

    def test_ctxt_cumulative(self) -> None:
        assert _is_cumulative("ctxt")

    def test_intr_cumulative(self) -> None:
        assert _is_cumulative("intr")

    def test_temperature_not_cumulative(self) -> None:
        assert not _is_cumulative("temp_coretemp_Core_0")

    def test_freq_not_cumulative(self) -> None:
        assert not _is_cumulative("cpu0_freq_mhz")


class TestIsExcluded:
    def test_timestamp_excluded(self) -> None:
        assert _is_excluded("ts_realtime_ns")
        assert _is_excluded("ts_monotonic_ns")

    def test_target_excluded(self) -> None:
        assert _is_excluded("adj_freq")

    def test_chrony_excluded(self) -> None:
        assert _is_excluded("chrony_offset")

    def test_age_excluded(self) -> None:
        assert _is_excluded("ipmi_temp_age_ms")

    def test_mono_minus_raw_excluded(self) -> None:
        assert _is_excluded("mono_minus_raw")

    def test_feature_not_excluded(self) -> None:
        assert not _is_excluded("temp_coretemp_Core_0")
        assert not _is_excluded("cpu0_freq_mhz")


class TestCategorizeColumn:
    def test_cpu_core_temp(self) -> None:
        assert categorize_column("temp_coretemp_Core_0") == "CPU Core Temp"

    def test_cpu_package_temp(self) -> None:
        assert categorize_column("temp_coretemp_Package_id_0") == "CPU Package Temp"

    def test_cpu_frequency(self) -> None:
        assert categorize_column("cpu0_freq_mhz") == "CPU Frequency"

    def test_power(self) -> None:
        assert categorize_column("rapl_pkg_energy") == "Power"

    def test_memory(self) -> None:
        assert categorize_column("mem_available") == "Memory"

    def test_io_disk(self) -> None:
        assert categorize_column("disk_sda_reads") == "I/O"

    def test_io_network(self) -> None:
        assert categorize_column("net_eth0_rx_bytes") == "I/O"

    def test_system(self) -> None:
        assert categorize_column("ctxt") == "System"

    def test_unknown(self) -> None:
        assert categorize_column("totally_unknown_sensor") is None


class TestGetFeatureCols:
    def test_excludes_target(self) -> None:
        df = pd.DataFrame(
            {
                "feat_a": [1, 2],
                "feat_b": [3, 4],
                TARGET_COL: [5, 6],
            }
        )
        cols = get_feature_cols(df)
        assert TARGET_COL not in cols
        assert "feat_a" in cols
        assert "feat_b" in cols

    def test_empty_df(self) -> None:
        df = pd.DataFrame({TARGET_COL: [1, 2]})
        assert get_feature_cols(df) == []


class TestLoadMachine:
    """Integration tests that require actual data files."""

    @pytest.fixture
    def data_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "src" / "tick2" / "data"

    def test_load_machine_smoke(self, data_dir: Path) -> None:
        """Smoke test: load homelab if data exists, else skip."""
        from tick2.data.preprocessing import DEFAULT_DATA_DIR

        csv_path = DEFAULT_DATA_DIR / "24h_snapshot" / "homelab.csv"
        if not csv_path.exists():
            pytest.skip(f"Data file not found: {csv_path}")

        df, cats = load_machine("homelab")
        assert TARGET_COL in df.columns
        assert len(df) > 0
        assert len(cats) > 0
        # All feature columns should have a category
        for col in get_feature_cols(df):
            assert col in cats, f"{col} missing from categories"

    def test_load_machine_no_nans(self) -> None:
        """Loaded data should have no NaN values after preprocessing."""
        from tick2.data.preprocessing import DEFAULT_DATA_DIR

        csv_path = DEFAULT_DATA_DIR / "24h_snapshot" / "homelab.csv"
        if not csv_path.exists():
            pytest.skip("Data file not found")

        df, _ = load_machine("homelab")
        assert not df.isna().any().any(), "NaN values found in preprocessed data"

    def test_load_machine_with_aggregation(self) -> None:
        """Test loading with temporal aggregation."""
        from tick2.data.preprocessing import DEFAULT_DATA_DIR

        csv_path = DEFAULT_DATA_DIR / "24h_snapshot" / "homelab.csv"
        if not csv_path.exists():
            pytest.skip("Data file not found")

        df_raw, _ = load_machine("homelab")
        df_agg, _ = load_machine("homelab", agg_seconds=60)

        # Aggregated should have fewer rows
        assert len(df_agg) < len(df_raw)
        # Roughly 60x fewer rows
        ratio = len(df_raw) / len(df_agg)
        assert 50 < ratio < 70, f"Unexpected aggregation ratio: {ratio}"
