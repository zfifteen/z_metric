#!/usr/bin/env python3
"""
stream_driver.py

Stream through integers, detect primes, manage a 360-frame
streaming window of Z-embeddings, and print a concise summary
with performance statistics.
"""

import argparse
import time

import numpy as np
from prime_hologram_harness import HarnessDatabase, is_prime, embed_z
from prime_hologram_db import StreamingDatabase
from predictor import rolling_predict


def main():
    # ----- Parse arguments -----
    parser = argparse.ArgumentParser(
        description="Stream next primes, update 360-window, and forecast"
    )
    parser.add_argument(
        "--coords", required=True, help="Path to harness coords .npy"
    )
    parser.add_argument(
        "--primes", required=True, help="Path to harness primes .txt"
    )
    parser.add_argument(
        "--forecast",
        action="store_true",
        help="Enable forecasting of next-prime embeddings"
    )
    parser.add_argument(
        "--count", type=int, default=10000,
        help="How many integers to scan beyond the harness primes"
    )
    args = parser.parse_args()

    # ----- Load and seed streaming window -----
    harness = HarnessDatabase.load(args.coords, args.primes)
    initial_primes = harness.primes.tolist()
    window_size, dims = len(initial_primes), harness.coords.shape[1]

    stream_db = StreamingDatabase(window_size=window_size, dims=dims)
    for p, coord in zip(initial_primes, harness.coords):
        stream_db.add_point(p, coord)

    # ----- Initialize counters & timer -----
    start_time    = time.time()
    ints_scanned  = 0
    primes_found  = 0
    forecasts     = 0

    current = initial_primes[-1] + 1
    end     = current + args.count

    # ----- Streaming loop -----
    for n in range(current, end):
        ints_scanned += 1

        if not is_prime(n):
            continue

        primes_found += 1
        zvec = embed_z(n, dims)
        stream_db.add_point(n, zvec)
        print(f"Test: found prime {n}")

        if args.forecast:
            forecasts += 1
            buffer = stream_db.coords
            z_pred = rolling_predict(buffer)
            cands  = stream_db.query_radius(z_pred, radius=4.0)
            print(f"Forecast #{forecasts}: candidates = {cands}")

    # ----- Summary -----
    total_time = time.time() - start_time
    tests_per_s  = ints_scanned / total_time if total_time > 0 else float("inf")
    primes_per_s = primes_found  / total_time if total_time > 0 else float("inf")

    print("\nStreaming complete.\n")
    print("Summary:")
    print(f"  Total integers scanned: {ints_scanned}")
    print(f"  Primes found:           {primes_found}")
    if args.forecast:
        print(f"  Forecasts made:         {forecasts}")
    print(f"  Total runtime:          {total_time:.2f} sec")
    print(f"  Tests per second:       {tests_per_s:.2f}")
    print(f"  Primes per second:      {primes_per_s:.2f}")
    print(f"  Avg time per test:      {total_time/ints_scanned*1e3:.2f} ms")
    if args.forecast and forecasts:
        print(f"  Avg time per forecast:  {total_time/forecasts*1e3:.2f} ms")


if __name__ == "__main__":
    main()
