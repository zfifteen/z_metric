#!/usr/bin/env python3
"""
prime_hologram_harness.py

Immutable batch “harness” database for prime embeddings.
Provides logging and a minimal CLI so you can load, query, and save.
"""

import sys
import argparse
import logging

from typing import Union, Sequence

import numpy as np
from scipy.spatial import cKDTree

# Configure root logger
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# By convention: row i of coords ↦ integer n = i + OFFSET
OFFSET = 2


class HarnessDatabase:
    """
    Immutable batch database. Embeddings never change order or size.
    Uses implicit mapping: index i → n = i + OFFSET.
    """

    def __init__(self, coords: np.ndarray, offset: int = OFFSET):
        if coords.dtype != np.float64:
            raise ValueError("coords must be float64")
        if coords.ndim != 2:
            raise ValueError("coords must be 2-dimensional")

        self.coords = coords
        self.offset = offset
        self.tree = cKDTree(self.coords)
        logger.info(
            f"HarnessDatabase initialized: "
            f"{coords.shape[0]} points, dim={coords.shape[1]}, offset={self.offset}"
        )

    @classmethod
    def load(cls, coords_path: str, offset: int = OFFSET) -> "HarnessDatabase":
        logger.info(f"Loading coords from '{coords_path}'")
        coords = np.load(coords_path)
        return cls(coords, offset)

    def save(self, coords_path: str) -> None:
        logger.info(f"Saving coords to '{coords_path}'")
        np.save(coords_path, self.coords)

    def query_radius(
        self, point: Union[Sequence[float], np.ndarray], radius: float
    ) -> np.ndarray:
        pt = np.asarray(point, dtype=np.float64)
        logger.info(f"Querying radius={radius} around point={pt.tolist()}")
        idxs = self.tree.query_ball_point(pt, radius)
        result = np.array(idxs, dtype=np.int32) + self.offset
        logger.info(f"Found {len(result)} hits: {result.tolist()}")
        return result


def main():
    parser = argparse.ArgumentParser(
        description="Harness DB CLI: load, query, and save prime embeddings"
    )
    sub = parser.add_subparsers(dest="command")

    # load-and-query
    q = sub.add_parser("query", help="Load a .npy coords file and query it")
    q.add_argument(
        "--coords", required=True, help="Path to .npy file holding coords array"
    )
    q.add_argument(
        "--point", required=True, nargs="+", type=float, help="Embedding to query"
    )
    q.add_argument(
        "--radius", required=True, type=float, help="Search radius"
    )

    # save (for completeness)
    s = sub.add_parser("save", help="Save a coords array to .npy")
    s.add_argument(
        "--coords-path", required=True, help="Output .npy file path"
    )
    s.add_argument(
        "--from-txt",
        required=True,
        help="Path to a whitespace-delimited txt file of floats"
    )

    args = parser.parse_args()

    if args.command == "query":
        db = HarnessDatabase.load(args.coords)
        db.query_radius(args.point, args.radius)

    elif args.command == "save":
        logger.info(f"Loading raw coords from '{args.from_txt}'")
        raw = np.loadtxt(args.from_txt, dtype=np.float64)
        if raw.ndim != 2:
            logger.error("Input text file must be a 2D array")
            sys.exit(1)
        db = HarnessDatabase(raw)
        db.save(args.coords_path)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
