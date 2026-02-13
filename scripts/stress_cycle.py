#!/usr/bin/env python3
"""Python-only stress cycle for ares-comp-10 (no install required).

Cycles through CPU, memory, CPU+mem, and I/O stress modes with idle
cool-down periods. One full cycle = 4 hours, repeats 6 times = 24 hours.

Usage: python3 stress_cycle.py [num_cycles]
  Default: 6 cycles (24 hours)
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

STRESS_DURATION = 2700  # 45 minutes
IDLE_DURATION = 900  # 15 minutes
IO_BLOCK_SIZE = 1024 * 1024  # 1 MB
IO_FILE_SIZE = 512 * 1024 * 1024  # 512 MB

_shutdown = False


def _signal_handler(signum: int, frame: object) -> None:
    global _shutdown
    _shutdown = True


def log(msg: str) -> None:
    """Print a timestamped log message."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{ts} UTC] {msg}", flush=True)


def _cpu_worker(duration: int) -> None:
    """Burn CPU for `duration` seconds."""
    end = time.monotonic() + duration
    x = 0.0
    while time.monotonic() < end:
        for _ in range(100_000):
            x = (x + 1.0) * 0.999999
        if _shutdown:
            return


def _mem_worker(duration: int, size_mb: int) -> None:
    """Allocate and thrash memory for `duration` seconds."""
    end = time.monotonic() + duration
    size_bytes = size_mb * 1024 * 1024
    while time.monotonic() < end:
        # Allocate a large bytearray and write to it
        data = bytearray(size_bytes)
        # Touch every page (4KB stride)
        for i in range(0, len(data), 4096):
            data[i] = 0xFF
        del data
        if _shutdown:
            return


def _io_worker(duration: int, tmpdir: str) -> None:
    """Read/write files for `duration` seconds."""
    end = time.monotonic() + duration
    pid = os.getpid()
    filepath = os.path.join(tmpdir, f"stress_io_{pid}.tmp")
    block = os.urandom(IO_BLOCK_SIZE)

    try:
        while time.monotonic() < end:
            # Write
            with open(filepath, "wb") as f:
                written = 0
                while written < IO_FILE_SIZE and time.monotonic() < end:
                    f.write(block)
                    written += IO_BLOCK_SIZE
                f.flush()
                os.fsync(f.fileno())

            if _shutdown or time.monotonic() >= end:
                break

            # Read back
            with open(filepath, "rb") as f:
                while f.read(IO_BLOCK_SIZE) and time.monotonic() < end:
                    pass

            if _shutdown:
                break
    finally:
        try:
            os.unlink(filepath)
        except FileNotFoundError:
            pass


def run_stress(
    mode: str, duration: int, ncpus: int, mem_per_worker_mb: int
) -> None:
    """Run a stress phase with the given mode."""
    log(f"=== {mode.upper()} STRESS for {duration // 60} min ===")

    procs: list[multiprocessing.Process] = []

    if mode == "cpu":
        for _ in range(ncpus):
            p = multiprocessing.Process(target=_cpu_worker, args=(duration,))
            p.start()
            procs.append(p)

    elif mode == "memory":
        for _ in range(ncpus):
            p = multiprocessing.Process(
                target=_mem_worker, args=(duration, mem_per_worker_mb)
            )
            p.start()
            procs.append(p)

    elif mode == "cpu_memory":
        half = max(ncpus // 2, 1)
        for _ in range(half):
            p = multiprocessing.Process(target=_cpu_worker, args=(duration,))
            p.start()
            procs.append(p)
        for _ in range(half):
            p = multiprocessing.Process(
                target=_mem_worker, args=(duration, mem_per_worker_mb)
            )
            p.start()
            procs.append(p)

    elif mode == "io":
        tmpdir = tempfile.mkdtemp(prefix="stress_io_")
        num_io = min(4, ncpus)
        for _ in range(num_io):
            p = multiprocessing.Process(
                target=_io_worker, args=(duration, tmpdir)
            )
            p.start()
            procs.append(p)

    # Wait for all workers
    for p in procs:
        p.join()

    log(f"=== {mode.upper()} STRESS COMPLETE ===")


def idle_phase(duration: int) -> None:
    """Sleep for the idle/cool-down period."""
    log(f"=== IDLE (cool-down) for {duration // 60} min ===")
    end = time.monotonic() + duration
    while time.monotonic() < end and not _shutdown:
        time.sleep(min(10, end - time.monotonic()))


def main() -> None:
    """Run the stress cycle."""
    global _shutdown

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    num_cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ncpus = multiprocessing.cpu_count()

    # Memory per worker: 75% total / ncpus
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total_mb = int(line.split()[1]) // 1024
                    break
            else:
                mem_total_mb = 4096
    except OSError:
        mem_total_mb = 4096

    mem_per_worker_mb = (mem_total_mb * 75 // 100) // ncpus

    log(f"Starting stress cycle: {num_cycles} cycles, {ncpus} CPUs")
    log(f"Memory per worker: {mem_per_worker_mb} MB")
    log(f"Each cycle: 4 hours (45min stress + 15min idle x 4 modes)")

    modes = ["cpu", "memory", "cpu_memory", "io"]

    for cycle in range(1, num_cycles + 1):
        if _shutdown:
            break
        log(f">>> CYCLE {cycle} / {num_cycles} <<<")

        for mode in modes:
            if _shutdown:
                break
            run_stress(mode, STRESS_DURATION, ncpus, mem_per_worker_mb)
            if _shutdown:
                break
            idle_phase(IDLE_DURATION)

        log(f"<<< CYCLE {cycle} COMPLETE >>>")

    log(f"All {num_cycles} cycles complete.")


if __name__ == "__main__":
    main()
