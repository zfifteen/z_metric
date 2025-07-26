#!/usr/bin/env python3
"""
stream_driver.py

Stream integers until a specified number of primes is found,
manage a 360-frame streaming window of Z-embeddings, and
print concise predictions plus a final performance summary.
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
        description="Stream until a set number of primes is found, update window, and forecast"
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
        "--prime-count", type=int, default=10000,
        help="How many new primes to find after the harness primes"
    )
    args = parser.parse_args()

    # ----- Load and seed streaming window -----
    harness = HarnessDatabase.load(args.coords, args.primes)
    initial_primes = harness.primes.tolist()
    initial_count = len(initial_primes)
    dims = harness.coords.shape[1]

    stream_db = StreamingDatabase(window_size=initial_count, dims=dims)
    for p, coord in zip(initial_primes, harness.coords):
        stream_db.add_point(p, coord)

    # ----- Initialize counters, trackers & timer -----
    start_time     = time.time()
    ints_scanned   = 0
    primes_found   = 0
    forecasts      = 0

    last_prime     = None
    last_prime_z   = None
    last_prime_idx = None

    current = initial_primes[-1] + 1

    # ----- Streaming loop (by prime count) -----
    while primes_found < args.prime_count:
        ints_scanned += 1

        if is_prime(current):
            primes_found += 1
            zvec = embed_z(current, dims)
            stream_db.add_point(current, zvec)

            # Track last-prime details
            last_prime     = current
            last_prime_z   = zvec
            last_prime_idx = initial_count + primes_found

            print(f"Test: found prime {current}")

            if args.forecast:
                forecasts += 1
                buffer = stream_db.coords
                z_pred = rolling_predict(buffer)
                cands  = stream_db.query_radius(z_pred, radius=4.0)
                print(f"Forecast #{forecasts}: candidates = {cands}")

        current += 1

    # ----- Summary -----
    total_time   = time.time() - start_time
    tests_per_s  = ints_scanned / total_time if total_time > 0 else float("inf")
    primes_per_s = primes_found  / total_time if total_time > 0 else float("inf")

    print("\nStreaming complete.\n")
    print("Summary:")
    print(f"  Total integers scanned: {ints_scanned}")
    print(f"  Primes requested:       {args.prime_count}")
    print(f"  Primes found:           {primes_found}")
    if args.forecast:
        print(f"  Forecasts made:         {forecasts}")
    print(f"  Total runtime:          {total_time:.2f} sec")
    print(f"  Tests per second:       {tests_per_s:.2f}")
    print(f"  Primes per second:      {primes_per_s:.2f}")
    print(f"  Avg time per test:      {total_time/ints_scanned*1e3:.2f} ms")
    if args.forecast and forecasts:
        print(f"  Avg time per forecast:  {total_time/forecasts*1e3:.2f} ms")

    if last_prime is not None:
        print("\nLast Prime Details:")
        print(f"  Last prime found:       {last_prime}")
        print(f"  Prime ordinal:          {last_prime_idx}")
        print(f"  Z-embedding:            {last_prime_z.tolist()}")


if __name__ == "__main__":
    main()
