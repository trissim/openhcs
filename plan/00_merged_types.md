# Plan: Merged Type Definitions for Certification

## Unified Type Hierarchy

All certification types live in `dq_dock_engine/docking/core.py`.

### Core Shared Types

```python
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum, auto

class CertificationDecision(Enum):
    CERTIFIED_BETTER = auto()   # |gap| > 2*bound, A is definitively better
    CERTIFIED_WORSE = auto()     # |gap| > 2*bound, A is definitively worse
    UNCERTIFIED = auto()         # |gap| ≤ 2*bound, cannot certify

@dataclass(frozen=True)
class GapCertification:
    """
    Gap-based certification for ranking decisions.
    
    From LatticeSum.lean::lj6_tail_bound:
    - If |E_A - E_B| > 2 * M/R³, ranking is CERTIFIED
    - Otherwise, ranking is UNCERTIFIED
    
    Works for both:
    - Option 1 (native in batch): certify top vs native
    - Option 2 (gap-based): certify top vs runner-up
    """
    decision: CertificationDecision
    energy_gap: float  # |E_top - E_other|, always positive
    error_bound: float  # Lean-proven truncation error (kcal/mol)
    
    @property
    def is_certified(self) -> bool:
        return self.decision in (CertificationDecision.CERTIFIED_BETTER,
                                 CertificationDecision.CERTIFIED_WORSE)
    
    @property
    def confidence(self) -> float:
        if self.is_certified:
            return 1.0
        two_bound = 2 * self.error_bound
        return self.energy_gap / two_bound if two_bound > 0 else 0.0
    
    def summary(self) -> str:
        if self.decision == CertificationDecision.CERTIFIED_BETTER:
            return f"CERTIFIED BETTER (gap={self.energy_gap:.3f} > 2×{self.error_bound:.6f})"
        elif self.decision == CertificationDecision.CERTIFIED_WORSE:
            return f"CERTIFIED WORSE (gap={self.energy_gap:.3f} > 2×{self.error_bound:.6f})"
        else:
            return f"UNCERTIFIED (gap={self.energy_gap:.3f} ≤ 2×{self.error_bound:.6f})"
    
    @staticmethod
    def from_energies(energy_a: float, energy_b: float, error_bound: float) -> "GapCertification":
        """
        Create certification from two energies (order-independent).
        
        Lean theorem: |E_A - E_B| > 2 * ε_bound → certified ranking
        """
        gap = abs(energy_a - energy_b)
        two_bound = 2 * error_bound
        
        if gap > two_bound:
            # Certified: determine which is better
            decision = (CertificationDecision.CERTIFIED_BETTER
                        if energy_a < energy_b
                        else CertificationDecision.CERTIFIED_WORSE)
        else:
            decision = CertificationDecision.UNCERTIFIED
        
        return GapCertification(
            decision=decision,
            energy_gap=gap,
            error_bound=error_bound,
        )
```

### Option 1: Native-In-Batch Result (Extends GapCertification)

```python
@dataclass(frozen=True)
class NativeCertification(GapCertification):
    """
    Certification when native pose is included in the batch (Option 1).
    
    Adds native_rank for benchmarking/validation reporting.
    """
    native_rank: int  # 1 = native ranked first
    
    @property
    def is_native_ranked_first(self) -> bool:
        return self.native_rank == 1
    
    def summary(self) -> str:
        base = super().summary()
        return f"Native #{self.native_rank}: {base}"


@dataclass(frozen=True)
class BenchmarkResult:
    """Type-safe result from benchmark docking run."""
    success: bool
    energy: float
    rmsd: float
    time: float
    n_atoms: int
    formal_status: str
    certified: bool | None = None
    confidence: float | None = None
    energy_gap: float | None = None
    native_rank: int | None = None
    
    @classmethod
    def from_certification(
        cls,
        pose_energy: float,
        pose_rmsd: float,
        elapsed: float,
        n_atoms: int,
        formal_status: str,
        cert: NativeCertification | GapCertification | None,
    ) -> "BenchmarkResult":
        """Construct from pose info + optional certification."""
        kwargs = dict(
            success=True,
            energy=pose_energy,
            rmsd=pose_rmsd,
            time=elapsed,
            n_atoms=n_atoms,
            formal_status=formal_status,
        )
        if cert is not None:
            kwargs.update(
                certified=cert.is_certified,
                confidence=cert.confidence,
                energy_gap=cert.energy_gap,
            )
        if isinstance(cert, NativeCertification):
            kwargs["native_rank"] = cert.native_rank
        return cls(**kwargs)
```

### Option 2: Scoring Result (Uses GapCertification)

```python
@dataclass(frozen=True)
class CertifiedBatchResult:
    """
    Result of batch scoring with Lean-proven error bounds.
    
    Key output for gap-based certification.
    """
    scores: jnp.ndarray  # (N_poses,)
    error_bound: float   # Per-pose error bound (same for all poses)
    target_error: float  # Target error used (kcal/mol)
    cutoff_radius: float # Actual cutoff radius used (Å)
    
    def certify_gap(self, idx_a: int, idx_b: int) -> GapCertification:
        """
        Certify whether pose at idx_a is better than pose at idx_b.
        
        From LatticeSum.lean::lj6_tail_bound:
        - gap = |E_a - E_b|
        - If gap > 2 * error_bound → CERTIFIED
        """
        gap = abs(float(self.scores[idx_a]) - float(self.scores[idx_b]))
        return GapCertification.from_energies(
            float(self.scores[idx_a]),
            float(self.scores[idx_b]),
            self.error_bound,
        )
    
    def certify_top_k(self, k: int = 1) -> list[GapCertification]:
        """
        Certify whether the top k poses are correctly ranked.
        
        Returns list of (k-1) certifications for consecutive pairs:
        [(top vs 2nd), (top vs 3rd), ..., (top vs kth)]
        """
        sorted_indices = jnp.argsort(self.scores)
        best_idx = int(sorted_indices[0])
        
        certifications = []
        for i in range(1, k):
            if i >= len(sorted_indices):
                break
            cert = self.certify_gap(best_idx, int(sorted_indices[i]))
            certifications.append(cert)
        
        return certifications
```

### Type Hierarchy Summary

```
CertificationResult (ABC)
├── GapCertification          # Option 2: gap-based, no native needed
│   └── NativeCertification   # Option 1: native in batch, adds native_rank
```

## Bug Fixes Applied

### Option 1: Native Rank Computation (Fixed)

**Old (broken):**
```python
# L158 in option1_native_in_batch.md - WRONG
native_rank = int(jnp.argsort(scores)[::-1].tolist().index(native_idx)) + 1
```

`argsort` returns indices sorted ascending by energy (low=best). Reversing gives worst-first, then `.index()` finds native's position — wrong ordering.

**Fixed:**
```python
native_rank = int((scores < scores[native_idx]).sum()) + 1
```

Count poses with strictly better (lower) energy than native, then +1 = native's rank.

### Option 2: CertificationEngine.certify (Fixed)

**Old (broken):**
```python
# L213-214 in option2_gap_based_certification.md - UNREACHABLE
if energy_a > energy_b:
    raise ValueError(...)
```

Then `certify_top_poses` calls `certify(scores[best], scores[other])` where `best < other` (lower energy = better). Since `scores[best] ≤ scores[other]` by construction, it always enters the `gap > 0` path — `CERTIFIED_BETTER` is unreachable.

**Fixed:**
```python
@staticmethod
def from_energies(energy_a: float, energy_b: float, error_bound: float) -> GapCertification:
    gap = abs(energy_a - energy_b)  # Order-independent
    ...
```

The Lean theorem checks `|E_A - E_B| > 2×bound` regardless of which is better. Using `abs()` makes it order-independent, matching the theorem.

## Files to Modify

| File | Change |
|------|--------|
| `dq_dock_engine/docking/core.py` | Add all types above (unified) |
| `dq_dock_engine/docking/scoring.py` | Add `CertifiedBatchResult` (uses `GapCertification`) |
| `dq_dock_engine/docking/pipeline.py` | Return `GapCertification` for CERTIFIED mode |
| `dq_dock_engine/docking/certification.py` | Keep `CertificationEngine`, use fixed `GapCertification` |
| `dq_dock_engine/benchmark/benchmark_pdb.py` | Use `BenchmarkResult.from_certification()` |

## Decision Tree

```
User asks: "Is this docking certified?"
    │
    ├── Have native pose? ──YES──> Use Option 1 (NativeCertification)
    │                               "Native ranked #{native_rank}"
    │                               Fixes 1ajx benchmark failure
    │
    └── Have native pose? ──NO────> Use Option 2 (GapCertification)
                                    "Top pose certified better than 2nd"
                                    Production docking use case
```
