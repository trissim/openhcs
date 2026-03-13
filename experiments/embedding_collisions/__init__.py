"""Embedding-collision experiments for Paper 1.

This package mirrors the finite fiber geometry used in the Lean development:

- worst-fiber collision multiplicity `A_pi`
- zero-error feasibility threshold `2^L >= A_pi`
- uniform worst-fiber distortion floor `max(0, 1 - 2^L / a)`
- exact finite recoverable mass under a uniform per-fiber tag budget

The code is intentionally lightweight and reproducible. Real embedding backends are
optional; the collision algebra works for any fixed representation map.
"""
