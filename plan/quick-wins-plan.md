# Quick Wins Plan: Immediate RMSD Improvements

**Goal**: Achieve 0.5-1.0 Å RMSD improvement with minimal code changes (1-2 hours work)

**Status**: 🔄 Planning
**Priority**: 🥇 HIGHEST - Fastest ROI
**Expected Impact**: 0.5-1.0 Å RMSD improvement
**Expected Speed Impact**: Neutral (may actually improve performance)

---

## Current Baseline Analysis

### Existing LJ Scoring Function
```python
# File: dq_dock_engine/docking/scoring.py:52-53
repulsion = 4.0 * r12        # Full repulsive wall
attraction = 0.4 * r6        # 10x weaker attractive well
```

**Problem Identification**:
1. **Over-conservative repulsion**: 10:1 ratio penalizes minor clashes too heavily
2. **Weak attraction**: Good poses with slight geometric imperfections rank poorly
3. **Limited sampling**: Only 10,000 poses in current benchmark

### Verification Against Literature

Standard LJ parameters in molecular docking:
- **AutoDock Vina**: Uses Gaussian steric + directional terms, not pure LJ
- **Rosetta**: Uses 12-6 LJ with ~2:1 attraction:repulsion ratio
- **SMINA**: Custom scoring, but typically uses more balanced LJ terms

**Our current 10:1 ratio is unusually conservative** for docking applications.

---

## Implementation Plan

### Phase 1: Optimize LJ Balance (30 minutes)

#### 1.1 Define Parameter Search Space

```python
# Test these LJ weight combinations:
test_configs = [
    # (repulsion_weight, attraction_weight, description)
    (4.0, 0.4, "baseline"),      # Current
    (3.5, 0.7, "moderate"),       # 5:1 ratio
    (3.0, 1.0, "balanced"),       # 3:1 ratio
    (3.0, 1.5, "attraction"),     # 2:1 ratio
    (2.5, 1.5, "soft"),          # 1.7:1 ratio
]
```

#### 1.2 Create Parameter Sweep Script

**File**: `dq_dock_engine/benchmark/lj_parameter_sweep.py`

```python
#!/usr/bin/env python3
"""
Parameter sweep for optimal LJ weights.
Tests multiple repulsion:attraction ratios on benchmark complexes.
"""

import jax.numpy as jnp
from pathlib import Path
import itertools

# Import current infrastructure
from dq_dock_engine.docking.scoring import _score_single_lj
from dq_dock_engine.benchmark.benchmark_pdb import (
    download_pdb, prepare_protein, prepare_ligand,
    extract_coords_and_radii, run_dq_dock
)

# Test complexes
TEST_COMPLEX = ["1ajx", "1jvp"]  # Fast to download, diverse targets

# LJ parameter grid
REPULSION_WEIGHTS = [2.5, 3.0, 3.5, 4.0]
ATTRACTION_WEIGHTS = [0.7, 1.0, 1.5, 2.0]

def test_lj_config(repulsion_w, attraction_w, complex_id):
    """Test single LJ configuration on single complex."""
    # Modify scoring function temporarily
    # Run docking with current config
    # Return (rmsd, energy, time)
    pass

def main():
    results = []
    for rep_w, attr_w in itertools.product(REPULSION_WEIGHTS, ATTRACTION_WEIGHTS):
        for complex_id in TEST_COMPLEX:
            rmsd, energy, time = test_lj_config(rep_w, attr_w, complex_id)
            results.append({
                'repulsion': rep_w,
                'attraction': attr_w,
                'ratio': rep_w/attr_w,
                'complex': complex_id,
                'rmsd': rmsd,
                'energy': energy,
                'time': time
            })

    # Analysis: Find best avg RMSD performer
    # Print summary table
    # Recommend optimal parameters

if __name__ == "__main__":
    main()
```

#### 1.3 Add Scoring Configuration (Following OpenHCS Architecture)

**File**: `dq_dock_engine/docking/core.py` (Add new dataclass)

```python
# RESPECTFUL: Frozen dataclass with validation (OpenHCS pattern)
@dataclass(frozen=True)
class ScoringConfig:
    """Configuration for Lennard-Jones scoring parameters.

    Follows OpenHCS pattern: declarative configuration with explicit validation.
    """
    repulsion_weight: float = 3.0
    attraction_weight: float = 1.5

    def validate(self) -> None:
        """Validate scoring parameters are physically reasonable.

        Raises:
            ValueError: If parameters violate physical constraints
        """
        if self.repulsion_weight <= 0:
            raise ValueError(
                f"Repulsion weight must be positive, got {self.repulsion_weight}"
            )
        if self.attraction_weight < 0:
            raise ValueError(
                f"Attraction weight must be non-negative, got {self.attraction_weight}"
            )
```

**File**: `dq_dock_engine/docking/scoring.py` (Update scoring function)

```python
# RESPECTFUL: Explicit dependency injection, no kwargs magic
@jax.jit
def _score_single_lj(
    receptor_coords: jnp.ndarray,
    pose_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    config: ScoringConfig  # Explicit dependency
) -> float:
    """
    Atom-typed LJ score with configuration.

    Precondition: config.validate() has been called
    Raises: ValueError if config is invalid (architectural violation)

    Args:
        receptor_coords: (N_rec, 3)
        pose_coords:     (N_lig, 3)
        receptor_radii:  (N_rec,) VdW radii in Angstroms
        ligand_radii:    (N_lig,) VdW radii in Angstroms
        config: ScoringConfig with validated parameters
    """
    diffs = receptor_coords[:, None, :] - pose_coords[None, :, :]
    dist_sq = jnp.sum(diffs ** 2, axis=-1)

    sigma_ij = receptor_radii[:, None] + ligand_radii[None, :]
    sigma_sq = sigma_ij ** 2
    dist_sq_safe = jnp.maximum(dist_sq, (0.5 * sigma_ij) ** 2)

    r6 = (sigma_sq / dist_sq_safe) ** 3
    r12 = r6 ** 2

    # NEW: Use configurable weights
    pe = repulsion_weight * r12 - attraction_weight * r6
    return jnp.sum(pe)
```

#### 1.4 Update Batched Scoring

**File**: `dq_dock_engine/docking/scoring.py` (around line 60)

```python
# RESPECTFUL: Explicit config parameter, no magic kwargs
@jax.jit
def score_internal_lj(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    config: ScoringConfig  # Same config for all poses
) -> jnp.ndarray:
    """
    Pure JAX batched internal LJ score.

    Args:
        receptor_coords: (N_rec, 3)
        poses_coords:    (N_poses, N_lig, 3)
        receptor_radii:  (N_rec,) VdW radii
        ligand_radii:    (N_lig,) VdW radii
        config: ScoringConfig with validated parameters

    Returns:
        (N_poses,) array of scores
    """
    # vmap over poses, config is shared (None axis)
    batched_score = jax.vmap(
        _score_single_lj,
        in_axes=(None, 0, None, None, None)  # config is shared
    )
    return batched_score(
        receptor_coords, poses_coords,
        receptor_radii, ligand_radii,
        config
    )
```

#### 1.5 Update Routing Interface

**File**: `dq_dock_engine/docking/scoring.py` (around line 154)

```python
# RESPECTFUL: Enum-driven dispatch with explicit config
def route_scoring(
    engine: ScoringEngine,
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    config: Optional[ScoringConfig] = None,  # Explicit, not **kwargs
    # SMINA-specific params
    receptor_file: Optional[str] = None,
    ligand_template: Optional[str] = None
) -> np.ndarray:
    """
    Strict Enum dispatch for scoring.

    Precondition: For INTERNAL_LJ, receptor_coords, poses_coords, radii must be provided
    Precondition: For SMINA_EXACT, receptor_file, ligand_template must be provided

    Args:
        engine: ScoringEngine enum value
        receptor_coords: (N_rec, 3) receptor coordinates
        poses_coords: (N_poses, N_lig, 3) pose coordinates
        receptor_radii: (N_rec,) VdW radii
        ligand_radii: (N_lig,) VdW radii
        config: ScoringConfig (required for INTERNAL_LJ)
        receptor_file: Path to receptor PDB (required for SMINA_EXACT)
        ligand_template: Path to ligand PDB template (required for SMINA_EXACT)

    Returns:
        np.ndarray of scores

    Raises:
        ValueError: If required parameters missing (architectural violation)
    """
    # Create default config if not provided
    if config is None:
        config = ScoringConfig()
        config.validate()

    if engine == ScoringEngine.INTERNAL_LJ:
        # Direct call - no defensive hasattr/getattr
        return np.array(score_internal_lj(
            receptor_coords, poses_coords,
            receptor_radii, ligand_radii,
            config
        ))

    elif engine == ScoringEngine.SMINA_EXACT:
        # Validate required params (fail-loud)
        if receptor_file is None or ligand_template is None:
            raise ValueError(
                "SMINA_EXACT requires receptor_file and ligand_template"
            )

        return score_smina_exact(
            receptor_file,
            ligand_template,
            poses_coords
        )

    # Enum exhaustiveness - should never reach here
    raise ValueError(f"Unknown ScoringEngine: {engine}")

---

### Phase 2: Increase Pose Sampling (15 minutes)

#### 2.1 Update Benchmark Configuration

**File**: `dq_dock_engine/benchmark/benchmark_pdb.py` (line 416)

```python
# OLD:
n_poses=10000,

# NEW:
n_poses=50000,  # 5x increase, still < 10s total time
```

**Justification**:
- Current: 10,000 poses × ~0.0003s/pose = ~3s
- Proposed: 50,000 poses × ~0.0003s/pose = ~15s
- Still 17x faster than SMINA (typically 30-60s)

#### 2.2 Add Memory Considerations

```python
# Verify memory usage:
# 50,000 poses × 50 atoms/pose × 3 coords × 8 bytes = ~60MB
# Well within GPU memory limits
```

---

### Phase 3: Validation and Testing (30 minutes)

#### 3.1 Create Comparison Script

**File**: `dq_dock_engine/benchmark/validate_quick_wins.py`

```python
#!/usr/bin/env python3
"""
Validate quick wins improvements against baseline.
"""

def run_comparison():
    """
    Compare:
    1. Baseline (current): 4.0/0.4, 10k poses
    2. Optimized: 3.0/1.5, 50k poses
    3. Optimized: 2.5/1.5, 50k poses
    """
    configs = [
        {"repulsion": 4.0, "attraction": 0.4, "n_poses": 10000, "name": "baseline"},
        {"repulsion": 3.0, "attraction": 1.5, "n_poses": 50000, "name": "balanced_50k"},
        {"repulsion": 2.5, "attraction": 1.5, "n_poses": 50000, "name": "soft_50k"},
    ]

    results = []
    for config in configs:
        for complex_id in ["1ajx", "1jvp"]:
            result = run_dq_dock_with_config(complex_id, config)
            results.append(result)

    # Print comparison table
    print("\n=== Quick Wins Validation ===")
    print(f"{'Config':<20} {'RMSD (Å)':<12} {'Time (s)':<10} {'Speedup':<10}")
    print("-" * 55)
    for r in results:
        print(f"{r['config']:<20} {r['rmsd']:<12.2f} {r['time']:<10.2f} {r['speedup']:<10.1f}x")

    # Statistical significance
    print("\n=== Statistical Analysis ===")
    print(f"Baseline RMSD: {baseline_rmsd:.2f} ± {baseline_std:.2f}")
    print(f"Optimized RMSD: {optimized_rmsd:.2f} ± {optimized_std:.2f}")
    print(f"Improvement: {delta_rmsd:.2f} Å ({pct_improvement:.1f}%)")
```

#### 3.2 Acceptance Criteria

**Success metrics**:
- ✅ RMSD improvement ≥ 0.5 Å on test complexes
- ✅ Time penalty < 2x (still > 8x faster than SMINA)
- ✅ No regression in scoring function correctness
- ✅ Code passes all existing tests

---

## Implementation Checklist

### Step 1: Modify Scoring Function
- [ ] Add `repulsion_weight` and `attraction_weight` parameters to `_score_single_lj()`
- [ ] Update `score_internal_lj()` to accept new parameters
- [ ] Update `route_scoring()` to pass through new parameters
- [ ] Set sensible defaults (3.0, 1.5)

### Step 2: Create Parameter Sweep
- [ ] Create `benchmark/lj_parameter_sweep.py`
- [ ] Implement grid search over parameter space
- [ ] Add result aggregation and analysis
- [ ] Run sweep on test complexes

### Step 3: Update Benchmark
- [ ] Change `n_poses` from 10,000 to 50,000 in `benchmark_pdb.py`
- [ ] Verify memory usage is acceptable
- [ ] Update documentation comments

### Step 4: Validation
- [ ] Create `benchmark/validate_quick_wins.py`
- [ ] Run baseline vs optimized comparison
- [ ] Collect statistics on multiple complexes
- [ ] Document results

### Step 5: Documentation
- [ ] Update `scoring.py` docstrings
- [ ] Document optimal LJ parameters
- [ ] Add usage examples to README

---

## Risk Assessment

### Low Risk ✅
- **Code changes**: Minimal, isolated to scoring function
- **Backward compatibility**: New parameters have sensible defaults
- **Performance**: Faster or same speed (better convergence)

### Medium Risk ⚠️
- **Optimal parameters**: May need per-target tuning
  - *Mitigation*: Use conservative default, add config system
- **Memory increase**: 5x more poses
  - *Mitigation*: Verified < 100MB GPU usage

### Low Probability / High Impact 🚨
- **Physics breakage**: Wrong LJ weights could produce nonsense
  - *Mitigation*: Parameter sweep + validation on known complexes

---

## Success Metrics

### Primary Metrics
- **RMSD improvement**: Target ≥ 0.5 Å average across test set
- **Speed**: Maintain ≥ 10x speedup vs SMINA
- **No regressions**: All existing tests pass

### Secondary Metrics
- **Code quality**: Clean, documented, type-safe
- **Reproducibility**: Deterministic results with same random seed
- **Extensibility**: Easy to add more LJ parameters later

---

## Next Steps

1. **Immediate**: Implement Phase 1.1-1.2 (parameter sweep script)
2. **Short-term**: Run sweep, identify optimal parameters
3. **Medium-term**: Implement Phase 2 (increase sampling)
4. **Validation**: Run full comparison suite

**Estimated timeline**: 2-3 hours from start to validated results

---

## Hand-wavy gaps to fill:

1. **What are actual optimal LJ parameters?**
   - Need to run parameter sweep to determine empirically
   - May vary by target class (kinases vs proteases)
   - Should test on diverse set: 1ajx (HIV protease), 1jvp (kinase), 6lu7 (COVID Mpro)

2. **Is 50k poses too many?**
   - Need to measure actual time per pose
   - Current estimate: 0.0003s/pose
   - Should benchmark with n_poses = [10k, 25k, 50k, 100k] to find sweet spot

3. **Memory constraints?**
   - Need to test on actual hardware
   - May need batch size limits for smaller GPUs
   - Fallback: reduce n_poses or process in chunks

4. **Interaction with optimization?**
   - Better LJ weights might reduce need for local refinement
   - Should test both with and without optimization enabled
   - Might find optimization becomes unnecessary with good global search

**Action**: Fill these gaps by running the parameter sweep and validation scripts described above.
