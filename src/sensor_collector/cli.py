"""Command-line interface for the sensor collector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CollectorConfig


def parse_args(argv: list[str] | None = None) -> CollectorConfig:
    """Parse command-line arguments and return a CollectorConfig."""
    parser = argparse.ArgumentParser(
        prog="sensor-collector",
        description="Collect sensor and clock drift data at 1 Hz",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path.home() / "drift_data",
        help="Output directory for CSV and metadata files (default: ~/drift_data)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=60,
        help="Flush CSV every N rows (default: 60)",
    )
    parser.add_argument(
        "-d",
        "--duration",
        type=int,
        default=0,
        help="Run duration in seconds, 0 for unlimited (default: 0)",
    )
    parser.add_argument(
        "--net",
        nargs="*",
        default=[],
        help="Network interfaces to monitor (default: auto-detect)",
    )
    parser.add_argument(
        "--disk",
        nargs="*",
        default=[],
        help="Disk devices to monitor (default: auto-detect)",
    )
    parser.add_argument(
        "--cstate-cpus-per-socket",
        type=int,
        default=2,
        help="Representative CPUs per socket for C-state (default: 2)",
    )
    parser.add_argument(
        "--no-root-sensors",
        action="store_true",
        help="Skip root-only sensors (RAPL, turbostat, IPMI)",
    )

    args = parser.parse_args(argv)

    return CollectorConfig(
        output_dir=args.output_dir,
        interval=args.interval,
        flush_every=args.flush_every,
        duration=args.duration,
        net_interfaces=args.net,
        disk_devices=args.disk,
        cstate_cpus_per_socket=args.cstate_cpus_per_socket,
        try_root_sensors=not args.no_root_sensors,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point for the sensor collector CLI."""
    config = parse_args(argv)

    # Import here to avoid circular imports and so --help works on any platform
    from .collector import run_collector

    try:
        run_collector(config)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(0)
