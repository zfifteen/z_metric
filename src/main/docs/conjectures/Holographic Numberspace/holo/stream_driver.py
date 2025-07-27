#!/usr/bin/env python3
"""
stream_driver.py

Stream integers until the total specified primes is reached (including harness),
manage a 360-frame streaming window of Z-embeddings, and
print only the final performance summary with Miller-Rabin sanity.
Integrates a neural-inspired layer to adaptively refine Z(n) via learned deviation.
Updated with review fixes: deviation prediction, gamma scaling, full-vector forecasting,
consistent window shapes, enhanced gap estimation, robustness guards, summary-only output,
ordinal prime indexing, adjusted target for total primes (-initial for new search),
factorization-avoidant primality via Miller-Rabin (no trial division),
and Z'-metric pre-filter on embeddings for prime proxy (dynamic axes 0,2,4; adaptive theta).
Handles general dims >=3; fixed AttributeError by maintaining window_primes in sync with adds.
Restored original stats in summary per user request.
"""

import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn

from prime_hologram_harness import HarnessDatabase, embed_z  # Removed is_prime; using Miller-Rabin
from prime_hologram_db import StreamingDatabase

def miller_rabin(n, k=10):
    """Miller-Rabin probabilistic primality test (factorization-avoidant)."""
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n < 2:
        return False
    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

class ZPredictor(nn.Module):
    """
    Neural module for Z refinement: predicts deviation ΔZ for scaling.
    Incorporates learnable epsilons, input-dependent hidden layer, and gamma scaling.
    Inputs: n (prime), delta (gap); Output: ΔZ (deviation for correction).
    Architecture: 2-in MLP with hidden=num_epsilons, relu, gamma-normalized refinement.
    """
    def __init__(self, num_epsilons=20, gamma=1.0):
        super().__init__()
        self.fc1 = nn.Linear(2, num_epsilons)
        self.fc2 = nn.Linear(num_epsilons, 1)
        self.gamma = gamma  # Now used in forward for scaling

    def forward(self, x_n, x_delta):
        input = torch.cat((x_n, x_delta), dim=1)
        hidden = torch.relu(self.fc1(input))
        out = self.fc2(hidden)
        refinement = out / self.gamma  # Explicit gamma utilization
        return refinement

def compute_theta(window_primes, coords, axes=(0,2,4), epsilon=0.01):
    """Compute adaptive theta as min Z' from window primes - epsilon."""
    if len(window_primes) < 2:
        return -np.inf  # No filter if insufficient data
    primes = np.array(window_primes)
    gaps = np.diff(primes, prepend=primes[0] - 2)  # Dummy gap for first
    i, j, k = [ax % coords.shape[1] for ax in axes]  # Modulo for general dims
    zi, zj, zk = coords[:, i], coords[:, j], coords[:, k]
    product = zi * zj * zk
    gm = np.sign(product) * np.abs(product)**(1/3)
    Zp = gm / np.exp(gaps)
    min_Zp = np.min(Zp)
    return min_Zp - epsilon

def main():
    # ----- Parse arguments -----
    parser = argparse.ArgumentParser(
        description="Stream until the total set number of primes is reached (including harness), update window, and forecast with neural refinement"
    )
    parser.add_argument("--coords", required=True, help="Path to harness coords .npy")
    parser.add_argument("--primes", required=True, help="Path to harness primes .txt")
    parser.add_argument("--forecast", action="store_true",
                        help="Enable forecasting of next-prime embeddings")
    parser.add_argument("--prime-count", type=int, default=10000,
                        help="Total primes to reach (including initial harness primes)")
    parser.add_argument("--tune-freq", type=int, default=10,
                        help="Online fine-tuning frequency (every N primes)")
    args = parser.parse_args()

    # ----- Load and seed streaming window -----
    harness = HarnessDatabase.load(args.coords, args.primes)
    initial_primes = harness.primes.tolist()
    initial_count = len(initial_primes)
    dims = harness.coords.shape[1]
    if dims < 3:
        raise ValueError("Require dims >=3 for Z' pre-filter")

    stream_db = StreamingDatabase(window_size=initial_count, dims=dims)
    window_primes = initial_primes.copy()  # Track primes in sync with rolling window
    for p, coord in zip(initial_primes, harness.coords):
        stream_db.add_point(p, coord)

    # ----- Adjust target for new primes -----
    new_target = max(0, args.prime_count - initial_count)

    # ----- Neural predictor setup -----
    model = ZPredictor(num_epsilons=20, gamma=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Prepare initial training data (N samples via dummy gap)
    primes_arr = np.array(initial_primes, dtype=np.float32)
    gaps = np.diff(primes_arr)  # N-1 gaps
    dummy_gap = np.mean(gaps) if len(gaps) > 0 else 1.0  # Empirical mean for dummy
    gaps = np.insert(gaps, 0, dummy_gap)  # Now N gaps
    classical_Z = primes_arr / np.exp(gaps)  # N values

    # Empirical deviation proxy: ΔZ = classical_Z - mean(classical_Z) for non-trivial learning
    mean_Z = np.mean(classical_Z)
    deviations = classical_Z - mean_Z  # Targets: deviations from mean for refinement

    X_n = torch.from_numpy(primes_arr).unsqueeze(1)
    X_delta = torch.from_numpy(gaps).unsqueeze(1)
    Y_dev = torch.from_numpy(deviations).unsqueeze(1)  # Train on deviations

    # Pretrain for stability
    for epoch in range(50):
        optimizer.zero_grad()
        dev_pred = model(X_n, X_delta)
        loss = loss_fn(dev_pred, Y_dev)
        loss.backward()
        optimizer.step()

    # ----- Initialize counters, trackers & timer -----
    start_time = time.time()
    ints_scanned = 0
    primes_found = 0
    forecasts = 0
    mr_calls = 0

    last_prime = initial_primes[-1]
    last_prime_idx = initial_count
    last_gap = primes_arr[-1] - primes_arr[-2] if len(primes_arr) > 1 else dummy_gap
    recent_gaps = list(gaps[-5:])  # For moving-average forecast enhancement

    current = last_prime + 1
    if current % 2 == 0:  # Skip even if starting even (post-odd prime)
        current += 1

    # Initial theta from harness
    theta = compute_theta(window_primes, stream_db.coords)

    # ----- Streaming loop (by adjusted new prime count; factorization-avoidant) -----
    while primes_found < new_target:
        ints_scanned += 1

        gap = current - last_prime

        zvec = embed_z(current, dims)
        i, j, k = 0, 2, 4
        zi, zj, zk = zvec[i % dims], zvec[j % dims], zvec[k % dims]  # Modulo for safety
        product = zi * zj * zk
        gm = np.sign(product) * np.abs(product)**(1/3)
        Zp = gm / np.exp(gap) if gap > 0 else 0

        if Zp >= theta:
            mr_calls += 1
            if miller_rabin(current):
                primes_found += 1

                # Classical gap and Z (with NaN/zero guards)
                delta_n = gap  # Already computed
                if delta_n == 0: delta_n = 1e-6  # Rare edge
                exp_delta = np.exp(delta_n)
                classical_Z = current / exp_delta if exp_delta != 0 else 1.0

                # Neural deviation prediction
                n_tensor = torch.tensor([[float(current)]], dtype=torch.float32)
                delta_tensor = torch.tensor([[float(delta_n)]], dtype=torch.float32)
                with torch.no_grad():
                    dev_Z = model(n_tensor, delta_tensor).item()

                # Apply correction: multiplicative for scale (Z_corrected = classical_Z * (1 + dev_Z))
                Z_corrected = classical_Z * (1 + dev_Z)

                # Embedding with guard against NaN
                base_vec = zvec  # Already computed
                zvec = base_vec * (Z_corrected / classical_Z) if classical_Z != 0 else base_vec

                stream_db.add_point(current, zvec)
                window_primes.append(current)
                if len(window_primes) > initial_count:
                    window_primes = window_primes[1:]  # Shift to sync with rolling window

                # Update trackers
                last_prime = current
                last_gap = delta_n
                recent_gaps.append(delta_n)
                recent_gaps = recent_gaps[-5:]  # Rolling last 5 for avg

                # ----- Online fine-tuning -----
                if primes_found % args.tune_freq == 0:
                    # Append new data (using empirical deviation)
                    new_classical_Z = current / np.exp(delta_n)
                    new_dev = new_classical_Z - mean_Z  # Update proxy; could recompute mean
                    X_n = torch.cat([X_n, n_tensor], dim=0)[-initial_count:]
                    X_delta = torch.cat([X_delta, delta_tensor], dim=0)[-initial_count:]
                    Y_dev = torch.cat([Y_dev, torch.tensor([[new_dev]], dtype=torch.float32)], dim=0)[-initial_count:]

                    optimizer.zero_grad()
                    dev_pred = model(X_n, X_delta)
                    loss = loss_fn(dev_pred, Y_dev)
                    loss.backward()
                    optimizer.step()

                    # Adaptive theta recompute
                    theta = compute_theta(window_primes, stream_db.coords)

                # ----- Forecasting using model-based Z prediction (silent) -----
                if args.forecast:
                    forecasts += 1
                    # Enhanced gap estimate: moving avg of recent gaps
                    next_gap = np.mean(recent_gaps) if recent_gaps else last_gap
                    next_n = current + next_gap
                    next_n_t = torch.tensor([[float(next_n)]], dtype=torch.float32)
                    next_gap_t = torch.tensor([[float(next_gap)]], dtype=torch.float32)
                    with torch.no_grad():
                        dev_next = model(next_n_t, next_gap_t).item()

                    # Classical next Z with guard
                    exp_next_gap = np.exp(next_gap)
                    classical_next_Z = next_n / exp_next_gap if exp_next_gap != 0 else 1.0
                    Z_next = classical_next_Z * (1 + dev_next)

                    # Full vector for query
                    next_base = embed_z(int(next_n), dims)  # Cast to int for embed_z
                    next_vec = next_base * (Z_next / classical_next_Z) if classical_next_Z != 0 else next_base
                    _ = stream_db.query_radius(next_vec, radius=4.0)  # Silent query

        current += 2  # Increment by 2 to skip evens (factorization-avoidant optimization)

    # ----- Summary -----
    total_time = time.time() - start_time
    tests_per_s = ints_scanned / total_time if total_time > 0 else float("inf")
    primes_per_s = primes_found / total_time if total_time > 0 else float("inf")

    print("\nStreaming complete.\n")
    print("Summary:")
    print(f"  Total integers scanned: {ints_scanned}")
    print(f"  Primes requested:       {new_target}")
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
        print(f"  The {last_prime_idx}th prime found is {last_prime}")
        print(f"  Z-embedding:            {stream_db.coords[-1].tolist()}")
        mr_pass = miller_rabin(last_prime, k=10)
        print(f"  Sanity check (Miller-Rabin): {'Passed' if mr_pass else 'Failed'}")

if __name__ == "__main__":
    main()