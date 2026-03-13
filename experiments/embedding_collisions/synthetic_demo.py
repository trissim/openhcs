#!/usr/bin/env python3
from __future__ import annotations

from typing import Sequence
from pathlib import Path
import sys
import importlib

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    core = importlib.import_module("experiments.embedding_collisions.core")
else:
    from . import core

exact_distortion = core.exact_distortion
feasible_zero_error = core.feasible_zero_error
synthetic_records_from_fiber_sizes = core.synthetic_records_from_fiber_sizes
uniform_worst_fiber_distortion_floor = core.uniform_worst_fiber_distortion_floor
zero_error_threshold_bits = core.zero_error_threshold_bits


def brute_force_best_success_uniform(
    fiber_sizes: Sequence[int], tag_alphabet: int
) -> float:
    total_items = sum(fiber_sizes)
    total_correct = sum(min(size, tag_alphabet) for size in fiber_sizes)
    return total_correct / total_items if total_items else 1.0


def main() -> None:
    fiber_sizes = [5, 5, 3, 1]
    records = synthetic_records_from_fiber_sizes(fiber_sizes)
    weights = [[1.0 for _ in range(size)] for size in fiber_sizes]
    a_pi = max(fiber_sizes)

    print("Synthetic Lean-style verification")
    print(f"  fiber sizes: {fiber_sizes}")
    print(f"  A_pi: {a_pi}")
    print(f"  zero-error threshold bits: {zero_error_threshold_bits(a_pi)}")

    for bits in range(0, 4):
        tag_alphabet = 2**bits
        exact = exact_distortion(weights, tag_alphabet)
        brute_success = brute_force_best_success_uniform(fiber_sizes, tag_alphabet)
        brute_distortion = 1.0 - brute_success
        floor = uniform_worst_fiber_distortion_floor(a_pi, bits)
        print(
            f"  L={bits} bits | T={tag_alphabet:2d} | exact empirical D={exact:.4f} | "
            f"worst-fiber floor={floor:.4f} | zero-error feasible={feasible_zero_error(a_pi, bits)}"
        )
        if abs(exact - brute_distortion) > 1e-12:
            raise AssertionError(
                "Exact empirical distortion disagrees with brute-force calculation"
            )

    if len(records) != sum(fiber_sizes):
        raise AssertionError(
            "Synthetic record generation produced the wrong number of items"
        )
    print("  verification passed")


if __name__ == "__main__":
    main()
