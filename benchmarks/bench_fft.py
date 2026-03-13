"""Benchmark polars-rfft vs numpy.fft on various signal sizes.

Usage:
    pip install numpy
    python benchmarks/bench_fft.py --sizes 1024 4096 16384 65536 262144 1048576
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
import polars as pl
from polars_rfft import rfft


def bench(fn, warmup: int = 2, repeats: int = 5) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    return median


def run_benchmarks(sizes: list[int]) -> None:
    rows: list[dict] = []

    for n in sizes:
        # Generate a test signal: sum of sinusoids
        t = np.linspace(0, 1, n, endpoint=False)
        signal = np.sin(2 * math.pi * 50 * t) + 0.5 * np.sin(2 * math.pi * 120 * t)
        signal_list = signal.tolist()

        df = pl.DataFrame({"x": signal_list})

        # ── Forward FFT ──
        polars_fft_ms = bench(
            lambda: df.select(rfft(pl.col("x")).fft()),
        ) * 1000

        numpy_fft_ms = bench(
            lambda: np.fft.fft(signal),
        ) * 1000

        rows.append({
            "n": n,
            "operation": "fft",
            "polars_ms": round(polars_fft_ms, 1),
            "numpy_ms": round(numpy_fft_ms, 1),
            "speedup": round(numpy_fft_ms / polars_fft_ms, 1) if polars_fft_ms > 0 else float("inf"),
        })

        # ── Inverse FFT (roundtrip) ──
        fft_result = df.select(rfft(pl.col("x")).fft().alias("f"))

        polars_ifft_ms = bench(
            lambda: fft_result.select(rfft(pl.col("f")).ifft_real()),
        ) * 1000

        fft_np = np.fft.fft(signal)
        numpy_ifft_ms = bench(
            lambda: np.fft.ifft(fft_np),
        ) * 1000

        rows.append({
            "n": n,
            "operation": "ifft",
            "polars_ms": round(polars_ifft_ms, 1),
            "numpy_ms": round(numpy_ifft_ms, 1),
            "speedup": round(numpy_ifft_ms / polars_ifft_ms, 1) if polars_ifft_ms > 0 else float("inf"),
        })

        # ── Magnitude ──
        polars_mag_ms = bench(
            lambda: fft_result.select(rfft(pl.col("f")).magnitude()),
        ) * 1000

        numpy_mag_ms = bench(
            lambda: np.abs(fft_np),
        ) * 1000

        rows.append({
            "n": n,
            "operation": "magnitude",
            "polars_ms": round(polars_mag_ms, 1),
            "numpy_ms": round(numpy_mag_ms, 1),
            "speedup": round(numpy_mag_ms / polars_mag_ms, 1) if polars_mag_ms > 0 else float("inf"),
        })

    # Print results
    print(f"\n{'n':>10} {'operation':>12} {'polars (ms)':>12} {'numpy (ms)':>12} {'speedup':>8}")
    print("-" * 58)
    for r in rows:
        marker = "<" if r["speedup"] < 1.0 else ""
        print(
            f"{r['n']:>10} {r['operation']:>12} {r['polars_ms']:>12.1f} {r['numpy_ms']:>12.1f} {r['speedup']:>7.1f}x{marker}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark polars-rfft vs numpy.fft")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 4096, 16384, 65536, 262144, 1048576],
        help="Signal sizes to benchmark",
    )
    args = parser.parse_args()
    run_benchmarks(args.sizes)
