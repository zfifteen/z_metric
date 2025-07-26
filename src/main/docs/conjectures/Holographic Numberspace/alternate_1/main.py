#!/usr/bin/env python3
"""
main.py

Orchestrates batch and streaming prime prediction using the Z‐Metric framework.
Modes:
  - batch : evaluate deterministic projection accuracy up to N
  - stream: continuously predict next prime geodesic in real time
"""

import argparse
import numpy as np
from lattice import load_holograph_db, compute_z_vector, query_neighbors
from predictor import rolling_predict
from math import exp, log


def run_batch(N, db_path, buffer_size, radius):
    # Load precomputed holographic DB
    indices, coords, tree = load_holograph_db(db_path)

    # Build static Z‐vectors for n=2..N
    ns = np.arange(2, N + 1, dtype=int)
    static_z = np.vstack([compute_z_vector(n) for n in ns])

    predictions = []
    errors = []

    # Slide a window of `buffer_size` and predict next geodesic
    for i in range(buffer_size, len(ns)):
        window = static_z[i - buffer_size : i]
        z_pred = rolling_predict(window)

        # Snap to nearest lattice points within radius
        neighbors = query_neighbors(tree, coords, indices, z_pred, radius)
        pred_n = neighbors[0][0] if neighbors else None
        actual_n = ns[i]

        if pred_n is not None:
            err = abs(pred_n - actual_n)
            predictions.append((actual_n, pred_n, err))
            errors.append(err)

    # Summarize results
    errs = np.array(errors, dtype=float)
    print(f"\nBatch mode up to N={N}")
    print(f"Buffer size: {buffer_size} | Snap radius: {radius}")
    print(f"Predictions made: {len(predictions)}")
    print(f"Mean absolute error: {errs.mean():.2f}")
    print(f"Max absolute error : {errs.max():.2f}")

    # Optional: print first few predictions
    print("\nSample predictions (actual → predicted | error):")
    for actual, pred, err in predictions[:10]:
        print(f"  {actual} → {pred} | Δ={err}")


def run_stream(start_n, db_path, buffer_size, radius):
    # Load precomputed holographic DB
    indices, coords, tree = load_holograph_db(db_path)

    # Initialize rolling buffer of Z‐vectors
    window_ns = list(range(start_n, start_n + buffer_size))
    buffer = np.vstack([compute_z_vector(n) for n in window_ns])
    next_n = start_n + buffer_size

    print(f"Streaming mode: starting at n={next_n}")
    try:
        while True:
            # Predict next Z‐vector
            z_pred = rolling_predict(buffer)

            # Snap to nearest candidate
            neighbors = query_neighbors(tree, coords, indices, z_pred, radius)
            pred_n = neighbors[0][0] if neighbors else None

            print(f"Predicted next prime ≈ {pred_n} (at n={next_n})")

            # Slide window: drop oldest, add actual next embedding
            z_actual = compute_z_vector(next_n)
            buffer = np.vstack([buffer[1:], z_actual])

            next_n += 1
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Z-Metric prime prediction: batch or stream mode."
    )
    parser.add_argument(
        "--mode", "-m", choices=["batch", "stream"], default="batch",
        help="Operation mode: 'batch' to evaluate up to N, 'stream' for live."
    )
    parser.add_argument(
        "--max", "-N", type=int, default=50021,
        help="Maximum integer for batch mode (default: 50021)."
    )
    parser.add_argument(
        "--db", "-d", type=str, default="holograph_db.npz",
        help="Path to holographic lattice DB (default: holograph_db.npz)."
    )
    parser.add_argument(
        "--buffer", "-b", type=int, default=360,
        help="Rolling buffer size (default: 360)."
    )
    parser.add_argument(
        "--radius", "-r", type=float, default=4.0,
        help="KD-tree snap radius Δ (default: 4.0)."
    )
    args = parser.parse_args()

    if args.mode == "batch":
        run_batch(args.max, args.db, args.buffer, args.radius)
    else:
        # For streaming, start_n must be >= 2
        start = 2
        run_stream(start, args.db, args.buffer, args.radius)
