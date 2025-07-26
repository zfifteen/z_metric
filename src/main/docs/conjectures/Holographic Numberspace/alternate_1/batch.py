#!/usr/bin/env python3
"""
batch.py

Batch script to build the 6D holographic NumberSpace database.
It invokes lattice.build_holograph_db to
factor-free compute Z-embeddings for all n = 2…N
and serialize them into a compressed .npz file.
"""

import argparse
from lattice import build_holograph_db

def main():
    parser = argparse.ArgumentParser(
        description="Build 6D holographic NumberSpace DB."
    )
    parser.add_argument(
        "--max", "-N", type=int, default=100000,
        help="Maximum integer N to embed (default: 100000)"
    )
    parser.add_argument(
        "--out", "-o", type=str, default="holograph_db.npz",
        help="Output .npz path (default: holograph_db.npz)"
    )
    args = parser.parse_args()

    print(f"▶ Building holographic DB up to N = {args.max}")
    build_holograph_db(N=args.max, out_path=args.out)
    print("✔ Build complete.")

if __name__ == "__main__":
    main()
