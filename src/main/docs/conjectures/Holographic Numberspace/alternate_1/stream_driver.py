#!/usr/bin/env python3
"""
stream_driver.py

Stream through integers, detect primes, and manage a 360-frame
streaming window of Z-embeddings. Optionally forecast next primes.
"""

import argparse
import logging

import numpy as np
from prime_hologram_harness import HarnessDatabase, is_prime, embed_z
from prime_hologram_db import StreamingDatabase
from predictor import rolling_predict

# Logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    p = argparse.ArgumentParser(
        description="Stream next primes, update 360-window, and forecast"
    )
    p.add_argument(
        "--coords", required=True, help="Path to harness coords .npy"
    )
    p.add_argument(
        "--primes", required=True, help="Path to harness primes .txt"
    )
    p.add_argument(
        "--forecast",
        action="store_true",
        help="Forecast next prime geodesic each time window is full",
    )
    p.add_argument(
        "--count", type=int, default=10000,
        help="Number of integers to scan after the harness primes"
    )
    args = p.parse_args()

    # 1) Load harness DB
    harness = HarnessDatabase.load(args.coords, args.primes)
    initial_primes = harness.primes.tolist()

    # 2) Initialize streaming window with first 360 prime embeddings
    window_size = len(initial_primes)
    dims = harness.coords.shape[1]
    stream_db = StreamingDatabase(window_size=window_size, dims=dims)

    logger.info(f"Seeding streaming window with first {window_size} primes")
    for n, coord in zip(initial_primes, harness.coords):
        stream_db.add_point(n, coord)

    # 3) Stream next integers
    current = initial_primes[-1] + 1
    end = current + args.count
    logger.info(f"Scanning integers {current} to {end-1} for primes")

    while current < end:
        if is_prime(current):
            zvec = embed_z(current, dims)
            stream_db.add_point(current, zvec)
            logger.info(f"Prime detected & added: {current}")

            if args.forecast:
                # perform AR forecast
                buffer = stream_db.coords  # shape (360, D)
                z_pred = rolling_predict(buffer)
                # snap using streaming DB
                cands = stream_db.query_radius(z_pred, radius=4.0)
                logger.info(f"Forecast candidates for next prime: {cands}")

        current += 1

    logger.info("Streaming complete.")


if __name__ == "__main__":
    main()
