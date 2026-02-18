"""Main 1 Hz collection loop with signal handling.

Orchestrates discovery, schema construction, and the sampling loop.
Handles SIGTERM/SIGINT for graceful shutdown.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import TYPE_CHECKING

from .discovery import discover_machine, print_inventory
from .schema import build_schema
from .writer import CsvWriter

if TYPE_CHECKING:
    from .config import CollectorConfig


_shutdown_requested = False


def _signal_handler(signum: int, frame: object) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global _shutdown_requested
    _shutdown_requested = True


def run_collector(config: CollectorConfig) -> None:
    """Run the main collection loop."""
    global _shutdown_requested
    _shutdown_requested = False

    # Register signal handlers
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Discovery
    print("Discovering sensors...", file=sys.stderr)
    inventory = discover_machine(config)
    print_inventory(inventory)

    # Build schema
    print("\nBuilding schema...", file=sys.stderr)
    schema = build_schema(inventory, config)
    print(f"  Total columns: {len(schema.columns)}", file=sys.stderr)
    print(f"  Readers: {len(schema.readers)}", file=sys.stderr)

    # Open writer
    writer = CsvWriter(columns=schema.columns, config=config)

    start_mono = time.monotonic()
    try:
        with writer:
            print(f"\nCollecting to {writer.csv_path}", file=sys.stderr)
            print(
                f"  Interval: {config.interval}s, "
                f"flush every {config.flush_every} rows",
                file=sys.stderr,
            )
            if config.duration > 0:
                print(f"  Duration: {config.duration}s", file=sys.stderr)
            print("  Press Ctrl+C to stop.\n", file=sys.stderr)

            next_tick = time.monotonic()

            while not _shutdown_requested:
                # Check duration limit
                if config.duration > 0:
                    elapsed = time.monotonic() - start_mono
                    if elapsed >= config.duration:
                        print(
                            f"\nDuration limit reached ({config.duration}s).",
                            file=sys.stderr,
                        )
                        break

                # Collect all readings into a single row
                row: dict[str, int | float | str] = {}
                for reader in schema.readers:
                    try:
                        data = reader.read()
                        row.update(data)
                    except Exception as e:
                        print(
                            f"Warning: {type(reader).__name__}.read() failed: {e}",
                            file=sys.stderr,
                        )

                writer.write_row(row)

                # Progress indicator every 60 rows
                if writer.row_count % 60 == 0:
                    elapsed = time.monotonic() - start_mono
                    print(
                        f"  [{writer.row_count} rows, {elapsed:.0f}s elapsed]",
                        file=sys.stderr,
                    )

                # Sleep until next tick (compensate for read time)
                next_tick += config.interval
                sleep_time = next_tick - time.monotonic()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # We're behind schedule — skip ahead to avoid drift
                    missed = int(-sleep_time / config.interval)
                    if missed > 0:
                        print(
                            f"Warning: missed {missed} tick(s), resynchronizing",
                            file=sys.stderr,
                        )
                    next_tick = time.monotonic()

    finally:
        # Stop background subprocesses
        for stoppable in schema.stoppable:
            try:
                stoppable.stop()  # type: ignore[attr-defined]
            except Exception as e:
                print(
                    f"Warning: failed to stop {type(stoppable).__name__}: {e}",
                    file=sys.stderr,
                )

        total_elapsed = time.monotonic() - start_mono
        print(
            f"\nDone. {writer.row_count} rows in {total_elapsed:.1f}s "
            f"({writer.csv_path})",
            file=sys.stderr,
        )
