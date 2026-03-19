# Plan: Option 2 - Gap-Based Certification Without Native Pose

## Problem Statement

Option 1 requires the **native pose** to establish an energy baseline. But in real docking:
- You **don't have** the native pose
- You only have **sampled poses**

For true certification, we need to prove that **pose A is better than pose B** without knowing the answer.

**Mathematical basis (from Lean theorems):**
```
|E_A - E_B| > 2 * ε_bound → certified ranking decision
```

Where `ε_bound` is the Lean-proven truncation error.

## OpenHCS Principles

| Principle | Implementation |
|-----------|----------------|
| **Type Safety** | `GapCertification` + `CertifiedBatchResult` (from `00_merged_types.md`) |
| **ABC/Polymorphism** | `GapCertification` base, separate scoring from decision |
| **Orthogonality** | Scoring → Gap computation → Decision, three separate steps |
| **Mathematical Simplification** | Centralize gap calculation, inline simple checks |
| **Fail-Loud** | Raise if gap ≤ 0 (invalid), no silent fallback |

## Types (See `00_merged_types.md`)

All types are defined in `dq_dock_engine/docking/core.py` and shared between Option 1 and Option 2:

- `CertificationDecision` — enum: `CERTIFIED_BETTER`, `CERTIFIED_WORSE`, `UNCERTIFIED`
- `GapCertification` — base type with `decision`, `energy_gap`, `error_bound`, `confidence`, `from_energies()`
- `CertifiedBatchResult` — batch scoring result with `certify_gap()` and `certify_top_k()`

## Files to Modify

### 1. `dq_dock_engine/docking/core.py`

Add from `00_merged_types.md`:
- `CertificationDecision` enum
- `GapCertification` dataclass with `from_energies` factory
- `CertifiedBatchResult` dataclass

### 2. `dq_dock_engine/docking/scoring.py`

Add `score_certified_batch`:

```python
def score_certified_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    target_error: float = 0.001,
    epsilon: float = _EPSILON_KCAL_MOL,
) -> CertifiedBatchResult:
    """
    Score batch of poses with Lean-proven error bounds.
    
    Returns:
        CertifiedBatchResult with scores and error_bound for gap certification.
    """
    scores, error_bound = score_certified_lj(
        receptor_coords, poses_coords, receptor_radii, ligand_radii,
        target_error=target_error, epsilon=epsilon
    )
    
    R = optimal_cutoff(target_error, s=6.0)
    
    return CertifiedBatchResult(
        scores=scores,
        error_bound=error_bound,
        target_error=target_error,
        cutoff_radius=R
    )
```

### 3. `dq_dock_engine/docking/certification.py` (NEW)

Create dedicated certification engine:

```python
"""
Gap-Based Certification Engine

Orthogonal to scoring - handles certification decisions only.
"""

from dq_dock_engine.docking.core import GapCertification, CertificationDecision

class CertificationEngine:
    """
    Engine for gap-based certification decisions.
    
    Uses Lean-proven bounds from lattice sums.
    """
    
    def __init__(self, error_bound: float):
        """
        Args:
            error_bound: Lean-proven truncation error (kcal/mol)
        """
        self.error_bound = error_bound
    
    def certify(
        self, 
        energy_a: float, 
        energy_b: float,
    ) -> GapCertification:
        """
        Certify whether pose A is better than pose B.
        
        Fails loudly if energies are invalid (NaN, Inf).
        Order-independent: works regardless of which energy is lower.
        """
        return GapCertification.from_energies(energy_a, energy_b, self.error_bound)
    
    def certify_top_poses(
        self,
        scores: list[float],
        top_k: int = 1,
    ) -> list[GapCertification]:
        """
        Certify the top k poses.
        
        Args:
            scores: List of pose energies (lower is better)
            top_k: Number of top poses to certify
            
        Returns:
            List of GapCertifications for consecutive pairs
        """
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        best_idx = sorted_indices[0]
        
        certifications = []
        for i in range(1, min(top_k, len(sorted_indices))):
            cert = self.certify(scores[best_idx], scores[sorted_indices[i]])
            certifications.append(cert)
        
        return certifications
```

### 4. `dq_dock_engine/docking/pipeline.py`

Wire in certification engine:

```python
from dq_dock_engine.docking.certification import CertificationEngine

def run_docking_pipeline(
    # ... existing params ...
) -> tuple[List[ScoredPose], GapCertification | None]:
    """
    Returns:
        (best_poses, top_certification)
        - best_poses: Ranked list of poses
        - top_certification: GapCertification for top pose vs 2nd (CERTIFIED mode only)
    """
    # ... existing scoring code ...
    
    if config is not None and config.mode == DockingMode.CERTIFIED:
        # Create certification engine with proven bound
        _, error_bound = score_certified_lj(
            protein_coords, poses_coords[:1], receptor_radii, ligand_radii,
            target_error=config.target_error
        )
        
        engine = CertificationEngine(error_bound=float(error_bound))
        
        # Certify top pose
        scores_list = [float(s) for s in final_scores]
        certifications = engine.certify_top_poses(scores_list, top_k=2)
        
        top_cert = certifications[0] if certifications else None
        return best_poses, top_cert
    
    return best_poses, None
```

### 5. `dq_dock_engine/benchmark/benchmark_pdb.py`

Report certification results with type safety:

```python
from dq_dock_engine.docking.core import GapCertification

def run_dq_dock(
    # ... existing params ...
) -> GapBenchmarkResult:
    # ...
    
    best_poses, top_cert = run_docking_pipeline(
        # ... existing params ...
    )
    
    # Type-safe result construction
    return BenchmarkResult.from_certification(
        pose_energy=best_poses[0].energy,
        pose_rmsd=rmsd,
        elapsed=elapsed,
        n_atoms=len(pocket_coords) + len(ligand_coords),
        formal_status=formal_status.name,
        cert=top_cert,
    )
```

## Comparison: Option 1 vs Option 2

| Aspect | Option 1 (Native in Batch) | Option 2 (Gap-Based) |
|--------|---------------------------|---------------------|
| **Use case** | Benchmarking, validation | Production docking |
| **Requires native** | Yes | No |
| **Certification type** | "Native is best" | "Pose A is better than B" |
| **Real-world applicable** | No | Yes |
| **Lean theorem used** | Ranking | Gap > 2×bound |
| **Result type** | `NativeCertification` | `GapCertification` |

## Mathematical Foundation

From `LatticeSum.lean::lj6_tail_bound`:

```
For any two poses A, B with truncated energies E_A, E_B:
|E_A - E_B| ≤ |E_true_A - E_true_B| + 2·M/R³

If |E_A - E_B| > 2·M/R³, then:
  E_true_A < E_true_B  (certified ranking)
```

The 2× factor accounts for worst-case: each truncated sum could be off by M/R³ in opposite directions.

## Bug Fix: Order-Independent Certification

**Old (broken):** `CertificationEngine.certify` asserted `energy_a ≤ energy_b`, making `CERTIFIED_BETTER` unreachable when called from `certify_top_poses` (which passes scores in the wrong argument order for that branch).

**Fixed:** `GapCertification.from_energies` uses `abs(energy_a - energy_b)`, making it order-independent — matching the Lean theorem which checks `|E_A - E_B| > 2×bound` regardless of which is better.

```python
gap = abs(energy_a - energy_b)
two_bound = 2 * error_bound

if gap > two_bound:
    decision = (CERTIFIED_BETTER if energy_a < energy_b else CERTIFIED_WORSE)
```

## Summary of Changes

| File | Change |
|------|--------|
| `core.py` | Add `CertificationDecision`, `GapCertification`, `CertifiedBatchResult` |
| `scoring.py` | Add `score_certified_batch` |
| `certification.py` | NEW: `CertificationEngine` class |
| `pipeline.py` | Return `GapCertification` for top pose |
| `benchmark_pdb.py` | Use `BenchmarkResult.from_certification` |

## Benefits

1. **Orthogonal**: Certification is separate from scoring
2. **Type Safe**: Named tuples and dataclasses with typed fields
3. **Fail-Loud**: Invalid inputs raise, no silent defaults
4. **Real-world**: Works without knowing the native pose
5. **Mathematical**: Direct translation of Lean theorem to code
6. **Shared**: Types shared with Option 1, no duplication
