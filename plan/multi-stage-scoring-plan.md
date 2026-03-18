# Multi-Stage Scoring Pipeline Plan

**Goal**: Implement coarse-to-fine scoring pipeline for 1-2 Å RMSD improvement

**Status**: 🔄 Planning
**Priority**: 🥈 HIGH - Major impact with moderate effort
**Expected Impact**: 1-2 Å RMSD improvement
**Expected Speed Impact**: Neutral or positive (faster overall due to early rejection)

---

## Problem Statement

### Current Limitation
All 10,000 poses receive full expensive LJ scoring:
```
10,000 poses × 0.3ms/pose = 3.0 seconds total
```

**Inefficiency**: Most poses are obviously bad (major clashes, outside pocket) but get full computation anyway.

### Solution: Progressive Filtering
```
Stage 1: 10,000 poses × 0.01ms/pose = 0.1s  → Keep 2,000 (20%)
Stage 2:  2,000 poses × 0.1ms/pose  = 0.2s  → Keep   500 (5%)
Stage 3:    500 poses × 0.3ms/pose  = 0.15s → Final ranking
Total: 0.45s (6.7x faster) with better discrimination
```

---

## Architecture Design

### Stage 1: Geometric Pre-filtering (Coarse)

**Purpose**: Rapid elimination of impossible poses
**Time budget**: ≤ 0.01ms per pose (100 poses/ms)

#### 1.1 Clash Detection Grid

**Algorithm**:
```python
# Voxel-based clash detection
# File: dq_dock_engine/docking/scoring_stages.py

@jax.jit
def detect_clashes_grid(
    pose_coords: jnp.ndarray,  # (N_lig, 3)
    receptor_voxel: jnp.ndarray,  # (V, V, V) binary occupancy grid
    voxel_size: float = 0.5,  # Angstroms
    clash_threshold: int = 3  # Allowed ligand atoms in occupied voxels
) -> bool:
    """
    Fast voxel-based clash detection.

    Returns True if pose has acceptable clashes, False otherwise.
    """
    # Convert pose coords to voxel indices
    voxel_indices = (pose_coords / voxel_size).astype(int32)

    # Count unique occupied voxels
    occupied_voxels = set()
    clash_count = 0
    for idx in voxel_indices:
        if receptor_voxel[idx]:
            clash_count += 1
            if clash_count > clash_threshold:
                return False  # Reject pose

    return True  # Accept pose
```

**Precomputation** (once per receptor):
```python
@jax.jit
def build_receptor_voxel_grid(
    receptor_coords: jnp.ndarray,
    center: jnp.ndarray,
    box_size: float = 20.0,
    voxel_size: float = 0.5
) -> jnp.ndarray:
    """
    Build binary occupancy grid for receptor.

    Returns: (V, V, V) boolean array where V = box_size/voxel_size
    """
    V = int(box_size / voxel_size)
    grid = jnp.zeros((V, V, V), dtype=bool)

    # Map receptor coords to voxels
    rel_coords = receptor_coords - center  # Center around origin
    voxel_indices = (rel_coords / voxel_size).astype(int32)

    # Mark occupied voxels
    for idx in voxel_indices:
        if 0 <= idx[0] < V and 0 <= idx[1] < V and 0 <= idx[2] < V:
            grid = grid.at[idx].set(True)

    return grid
```

**Complexity**: O(N_lig) per pose vs O(N_rec × N_lig) for full scoring

#### 1.2 Shape Complementarity Score

**Purpose**: Reward poses that match pocket shape

**Algorithm**:
```python
@jax.jit
def shape_complementarity_score(
    pose_coords: jnp.ndarray,
    receptor_voxel: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    voxel_size: float = 0.5
) -> float:
    """
    Fast shape-based scoring using distance transforms.

    Higher score = better shape fit.
    """
    # Distance transform of receptor (precomputed)
    # dt[i,j,k] = distance to nearest receptor atom

    # For each ligand atom, find distance to receptor surface
    # Score based on how well ligand fills pocket cavities

    # Simplified: Count atoms in "sweet spot" (1-3 Å from receptor)
    surface_voxels = (dt > 1.0) & (dt < 3.0)

    ligand_voxels = voxelize_ligand(pose_coords, ligand_radii, voxel_size)
    overlap = jnp.sum(ligand_voxels & surface_voxels)

    return overlap
```

**Precomputation** (once per receptor):
```python
from scipy.ndimage import distance_transform_edt

def build_distance_transform(receptor_voxel: jnp.ndarray) -> jnp.ndarray:
    """
    Compute distance transform of receptor occupancy.

    Returns: Distance to nearest occupied voxel for each voxel.
    """
    return distance_transform_edt(~receptor_voxel)
```

#### 1.3 Pocket Containment Check

**Purpose**: Reject poses outside binding pocket

```python
@jax.jit
def check_pocket_containment(
    pose_coords: jnp.ndarray,
    pocket_center: jnp.ndarray,
    pocket_radius: float = 10.0
) -> bool:
    """
    Ensure ligand stays within pocket region.

    Returns True if ≥ 80% of ligand atoms within pocket_radius.
    """
    distances = jnp.linalg.norm(pose_coords - pocket_center, axis=1)
    in_pocket = jnp.sum(distances < pocket_radius)
    fraction = in_pocket / len(pose_coords)
    return fraction > 0.8
```

**Stage 1 Pipeline**:
```python
@jax.jit
def stage1_score_pose(
    pose_coords: jnp.ndarray,
    receptor_voxel: jnp.ndarray,
    receptor_dt: jnp.ndarray,
    pocket_center: jnp.ndarray,
    ligand_radii: jnp.ndarray
) -> tuple[bool, float]:
    """
    Stage 1: Fast geometric filtering.

    Returns: (keep_pose, score)
    """
    # Check 1: Pocket containment
    if not check_pocket_containment(pose_coords, pocket_center):
        return (False, -jnp.inf)

    # Check 2: Major clashes
    if not detect_clashes_grid(pose_coords, receptor_voxel):
        return (False, -jnp.inf)

    # Check 3: Shape complementarity (keep score for ranking)
    shape_score = shape_complementarity_score(
        pose_coords, receptor_dt, ligand_radii
    )

    # Keep top 20% based on shape score
    return (True, shape_score)
```

---

### Stage 2: Medium-Fidelity Scoring

**Purpose**: More accurate physics for promising poses
**Time budget**: ≤ 0.1ms per pose

#### 2.1 Simplified LJ Potential

**Optimizations for speed**:
```python
@jax.jit
def stage2_lj_score(
    pose_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    repulsion_weight: float = 3.0,
    attraction_weight: float = 1.5
) -> float:
    """
    Simplified LJ with cutoff for speed.

    Cutoff: 8 Å (ignore distant atom pairs)
    """
    # Spatial hashing for neighbor search (precomputed)
    # Only compute LJ for nearby atom pairs
    nearby_pairs = get_neighbors_within_cutoff(pose_coords, receptor_coords, cutoff=8.0)

    # Compute LJ only for nearby pairs
    total_energy = 0.0
    for (i, j) in nearby_pairs:
        r_sq = distance_squared(pose_coords[i], receptor_coords[j])
        sigma = ligand_radii[i] + receptor_radii[j]
        sigma_sq = sigma ** 2

        r6 = (sigma_sq / r_sq) ** 3
        r12 = r6 ** 2
        total_energy += repulsion_weight * r12 - attraction_weight * r6

    return total_energy
```

**Precomputation** (once per receptor):
```python
def build_spatial_hash(
    receptor_coords: jnp.ndarray,
    cell_size: float = 8.0  # Cutoff distance
) -> dict:
    """
    Build spatial hash grid for fast neighbor queries.

    Returns: Dict mapping cell_id -> list of atom indices
    """
    grid = {}
    for idx, coord in enumerate(receptor_coords):
        cell_id = tuple((coord / cell_size).astype(int))
        if cell_id not in grid:
            grid[cell_id] = []
        grid[cell_id].append(idx)
    return grid

def get_neighbors_within_cutoff(
    query_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    spatial_hash: dict,
    cell_size: float = 8.0
) -> list[tuple[int, int]]:
    """
    Get all atom pairs within cutoff distance.

    Returns: List of (ligand_idx, receptor_idx) tuples
    """
    pairs = []
    for i, q_coord in enumerate(query_coords):
        cell_id = tuple((q_coord / cell_size).astype(int))

        # Check 3x3x3 neighborhood
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    neighbor_cell = (cell_id[0]+dx, cell_id[1]+dy, cell_id[2]+dz)
                    if neighbor_cell in spatial_hash:
                        for j in spatial_hash[neighbor_cell]:
                            dist = jnp.linalg.norm(q_coord - receptor_coords[j])
                            if dist < cell_size:
                                pairs.append((i, j))
    return pairs
```

#### 2.2 Crude Electrostatics

**Simplified Coulomb potential**:
```python
@jax.jit
def stage2_electrostatics(
    pose_coords: jnp.ndarray,
    pose_charges: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    dielectric: float = 4.0  # Distance-dependent dielectric
) -> float:
    """
    Simple Coulomb potential with distance-dependent dielectric.

    Cutoff: 6 Å for electrostatics (shorter than LJ cutoff)
    """
    # Only compute for nearby pairs (from spatial hash)
    nearby_pairs = get_neighbors_within_cutoff(
        pose_coords, receptor_coords, cutoff=6.0
    )

    total_elec = 0.0
    for i, j in nearby_pairs:
        r = jnp.linalg.norm(pose_coords[i] - receptor_coords[j])
        q_prod = pose_charges[i] * receptor_charges[j]

        # Distance-dependent dielectric: ε = 4r
        # Avoids singularity at r=0
        total_elec += q_prod / (dielectric * r)

    return total_elec
```

**Charge assignment** (simplified):
```python
# File: dq_dock_engine/docking/charges.py

# Simple charge rules (more accurate than nothing)
ATOM_CHARGES = {
    # Backbone
    'N': -0.5,   # Amide nitrogen
    'C': 0.5,    # Carbonyl carbon
    'O': -0.5,   # Carbonyl oxygen
    # Side chains
    'NH2': -0.5, # Amines
    'COO': -1.0, # Carboxylates
    'NH3': 1.0,  # Ammonium
    # Neutral
    'C': 0.0,    # Hydrocarbons
    'S': 0.0,    # Sulfur
}

def assign_charges(elements: list[str], atom_names: list[str]) -> jnp.ndarray:
    """
    Assign crude partial charges based on atom type.

    Returns: (N,) array of charges
    """
    charges = []
    for elem, name in zip(elements, atom_names):
        # Very simple rules
        if elem == 'N':
            charges.append(-0.3)
        elif elem == 'O':
            charges.append(-0.4)
        elif elem in ['Na', 'K']:
            charges.append(1.0)
        elif elem in ['Cl', 'Br']:
            charges.append(-1.0)
        else:
            charges.append(0.0)
    return jnp.array(charges)
```

**Stage 2 Pipeline**:
```python
@jax.jit
def stage2_score_pose(
    pose_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    ligand_charges: jnp.ndarray,
    spatial_hash: dict
) -> float:
    """
    Stage 2: Medium-fidelity scoring with simplified physics.

    Returns: Combined energy score
    """
    # LJ term with cutoff
    lj_energy = stage2_lj_score(
        pose_coords, receptor_coords,
        receptor_radii, ligand_radii
    )

    # Electrostatics with cutoff
    elec_energy = stage2_electrostatics(
        pose_coords, ligand_charges,
        receptor_coords, receptor_charges
    )

    # Weight combination
    # Electrostatics typically weaker than LJ in scoring functions
    return lj_energy + 0.1 * elec_energy
```

---

### Stage 3: Full-Accuracy Scoring

**Purpose**: Final ranking with best physics
**Time budget**: ≤ 0.3ms per pose

**Use existing scoring** (`_score_single_lj` from [scoring.py](../dq_dock_engine/docking/scoring.py)) with optimized parameters.

```python
@jax.jit
def stage3_score_pose(
    pose_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    repulsion_weight: float = 3.0,
    attraction_weight: float = 1.5
) -> float:
    """
    Stage 3: Full-accuracy LJ scoring (existing function).

    This is the current scoring function with optimized weights.
    """
    return _score_single_lj(
        pose_coords, receptor_coords,
        receptor_radii, ligand_radii,
        repulsion_weight, attraction_weight
    )
```

---

## Integration with Existing Pipeline

### Modified Pipeline Function

**File**: `dq_dock_engine/docking/pipeline.py`

```python
def run_multi_stage_docking(
    protein_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_ctx: LigandContext,
    box: DockingBox,
    n_poses: int,
    key: jax.Array,
    top_k: int = 10,
    stage1_keep_ratio: float = 0.2,  # Keep 20%
    stage2_keep_ratio: float = 0.05,  # Keep 5% overall
) -> List[ScoredPose]:
    """
    Multi-stage docking pipeline with progressive filtering.

    Stage 1: Geometric filtering (10k poses → 2k poses)
    Stage 2: Medium-fidelity scoring (2k poses → 500 poses)
    Stage 3: Full-accuracy scoring (500 poses → top_k)
    """

    # --- PRECOMPUTATION (once per receptor) ---
    receptor_voxel = build_receptor_voxel_grid(
        protein_coords, box.center, box.size[0]
    )
    receptor_dt = build_distance_transform(receptor_voxel)
    spatial_hash = build_spatial_hash(protein_coords, cell_size=8.0)
    receptor_charges = assign_charges_from_elements(protein_elements)

    # --- STAGE 1: GEOMETRIC FILTERING ---
    pose_vecs = sample_random_poses(key, box, n_poses)
    batched_coords = apply_poses(ligand_ctx, pose_vecs)

    stage1_scores = []
    stage1_keep_indices = []

    for i, pose_coords in enumerate(batched_coords):
        keep, score = stage1_score_pose(
            pose_coords, receptor_voxel, receptor_dt,
            box.center, ligand_ctx.base_radii
        )
        if keep:
            stage1_keep_indices.append(i)
            stage1_scores.append(score)

    # Keep top 20% by shape score
    n_keep_stage1 = int(n_poses * stage1_keep_ratio)
    stage1_best_indices = jnp.argsort(stage1_scores)[:n_keep_stage1]

    stage1_poses = batched_coords[stage1_best_indices]

    # --- STAGE 2: MEDIUM-FIDELITY SCORING ---
    stage2_scores = []
    for pose_coords in stage1_poses:
        score = stage2_score_pose(
            pose_coords,
            protein_coords, receptor_radii, receptor_charges,
            ligand_ctx.base_radii, ligand_charges,
            spatial_hash
        )
        stage2_scores.append(score)

    # Keep top 5% overall
    n_keep_stage2 = int(n_poses * stage2_keep_ratio)
    stage2_best_indices = jnp.argsort(stage2_scores)[:n_keep_stage2]

    stage2_poses = stage1_poses[stage2_best_indices]

    # --- STAGE 3: FULL-ACCURACY SCORING ---
    stage3_scores = score_internal_lj(
        protein_coords, stage2_poses,
        receptor_radii, ligand_ctx.base_radii,
        repulsion_weight=3.0, attraction_weight=1.5
    )

    # Final ranking
    best_final_indices = jnp.argsort(stage3_scores)[:top_k]

    best_poses = []
    for idx in best_final_indices:
        best_poses.append(ScoredPose(
            coords=stage2_poses[idx],
            energy=float(stage3_scores[idx]),
            engine=ScoringEngine.INTERNAL_LJ
        ))

    return best_poses
```

---

## Performance Analysis

### Expected Timing Breakdown

| Stage | Poses | Time/pose | Total | Keep |
|-------|-------|-----------|-------|------|
| 1 | 10,000 | 0.01ms | 0.1s | 2,000 (20%) |
| 2 | 2,000 | 0.1ms | 0.2s | 500 (5%) |
| 3 | 500 | 0.3ms | 0.15s | 10 (final) |
| **Total** | - | - | **0.45s** | - |

**Speedup**: 6.7x faster than current (3.0s → 0.45s)

### Memory Requirements

- **Voxel grid**: (40/0.5)³ ≈ 512,000 voxels × 1 byte = 0.5 MB
- **Distance transform**: Same size × 4 bytes (float) = 2 MB
- **Spatial hash**: ~N_rec entries × 16 bytes = ~16 KB
- **Total precomputation**: < 3 MB (trivial)

---

## Implementation Plan

### Phase 1: Stage 1 Implementation (1 day)

**Tasks**:
1. Implement `build_receptor_voxel_grid()`
2. Implement `build_distance_transform()`
3. Implement `detect_clashes_grid()`
4. Implement `shape_complementarity_score()`
5. Implement `check_pocket_containment()`
6. Create `stage1_score_pose()` pipeline
7. Unit tests for each function

**Validation**:
- Test on 10,000 random poses
- Verify rejection rate > 70%
- Verify kept poses have reasonable geometry

### Phase 2: Stage 2 Implementation (2 days)

**Tasks**:
1. Implement `build_spatial_hash()`
2. Implement `get_neighbors_within_cutoff()`
3. Implement simplified `stage2_lj_score()`
4. Implement charge assignment (`assign_charges()`)
5. Implement `stage2_electrostatics()`
6. Create `stage2_score_pose()` pipeline
7. Unit tests for each function

**Validation**:
- Verify cutoff gives 10-100x speedup vs full scoring
- Test energy correlation with full scoring
- Ensure no major energy errors

### Phase 3: Integration and Testing (1 day)

**Tasks**:
1. Implement `run_multi_stage_docking()`
2. Add configuration parameters (keep ratios, cutoffs)
3. Update benchmark to use multi-stage
4. Run comparison: single-stage vs multi-stage
5. Profile and optimize hotspots

**Validation**:
- RMSD improvement ≥ 0.5 Å
- Speed improvement ≥ 3x
- No regression in scoring quality

---

## Testing Strategy

### Unit Tests

**File**: `tests/docking/test_scoring_stages.py`

```python
import pytest
import jax.numpy as jnp
from dq_dock_engine.docking.scoring_stages import (
    build_receptor_voxel_grid,
    detect_clashes_grid,
    shape_complementarity_score,
    # ...
)

def test_voxel_grid_construction():
    """Test voxel grid is built correctly."""
    coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    grid = build_receptor_voxel_grid(coords, center=jnp.zeros(3), box_size=10.0)

    assert grid.shape == (20, 20, 20)  # 10Å / 0.5Å voxel
    assert grid[10, 10, 10] == True  # Origin occupied

def test_clash_detection():
    """Test clashes are detected correctly."""
    pose_coords = jnp.array([[0.0, 0.0, 0.0]])  # On top of receptor
    receptor_grid = build_receptor_voxel_grid(...)

    has_clash = detect_clashes_grid(pose_coords, receptor_grid)
    assert has_clash == False  # Should reject

def test_stage1_rejection_rate():
    """Test Stage 1 rejects majority of poses."""
    # Generate 1000 random poses
    # Verify > 700 are rejected
    pass

def test_stage2_correlation():
    """Test Stage 2 scores correlate with Stage 3."""
    # For same poses, compute both Stage 2 and Stage 3 scores
    # Verify correlation > 0.8
    pass
```

### Integration Tests

**File**: `tests/docking/test_multi_stage_pipeline.py`

```python
def test_multi_stage_vs_single_stage():
    """Compare multi-stage vs single-stage results."""
    # Run same complex with both pipelines
    # Verify multi-stage is faster but similar RMSD
    pass

def test_multi_stage_determinism():
    """Verify multi-stage gives same results with same seed."""
    pass

def test_multi_stage_memory_usage():
    """Verify memory usage is reasonable."""
    pass
```

### Benchmark Tests

**File**: `benchmarks/test_multi_stage_performance.py`

```python
def test_stage_timing():
    """Measure actual time per stage."""
    # Should match budget: 0.01ms, 0.1ms, 0.3ms
    pass

def test_rmsd_improvement():
    """Test RMSD improvement on benchmark set."""
    # Run on [1ajx, 1jvp, 6lu7, ...]
    # Verify avg RMSD improvement ≥ 0.5 Å
    pass
```

---

## Risk Assessment

### Technical Risks

**Risk**: Stage 1 rejects good poses
- **Probability**: Medium
- **Impact**: High (misses correct binding pose)
- **Mitigation**:
  - Conservative thresholds (keep 20% not 5%)
  - Extensive validation on known complexes
  - Fallback to single-stage if rejection too aggressive

**Risk**: Stage 2 cutoff causes energy errors
- **Probability**: Low
- **Impact**: Medium (ranking errors)
- **Mitigation**:
  - Test multiple cutoff values (6Å, 8Å, 10Å)
  - Verify energy correlation with full scoring
  - Adjust cutoff based on validation

**Risk**: Spatial hash overhead exceeds savings
- **Probability**: Low
- **Impact**: Low (Stage 2 is small fraction of total)
- **Mitigation**:
  - Profile spatial hash construction
  - Optimize hash function if needed
  - Fall back to naive O(N²) for small proteins

### Engineering Risks

**Risk**: Code complexity increases
- **Probability**: High
- **Impact**: Medium (harder to maintain)
- **Mitigation**:
  - Clear function boundaries
  - Extensive documentation
  - Unit tests for each stage

**Risk**: Configuration complexity
- **Probability**: Medium
- **Impact**: Medium (many parameters to tune)
- **Mitigation**:
  - Provide sensible defaults
  - Document parameter effects
  - Create tuning guide

---

## Success Criteria

### Primary Metrics
- ✅ RMSD improvement ≥ 0.5 Å on test set
- ✅ Speed improvement ≥ 3x vs single-stage
- ✅ Stage rejection rates: 70-90% (Stage 1), 75-95% (Stage 2)
- ✅ Energy correlation (Stage 2 vs Stage 3) > 0.8

### Secondary Metrics
- ✅ All tests pass
- ✅ Memory usage < 100 MB additional
- ✅ Code quality maintained (linting, type hints)
- ✅ Documentation complete

### Stretch Goals
- 🎯 RMSD improvement ≥ 1.0 Å
- 🎯 Speed improvement ≥ 5x
- ✅ Stage 2 energy correlation > 0.9

---

## Hand-wavy gaps to fill:

1. **Optimal keep ratios?**
   - Need to experiment with different ratios
   - Test: [0.1, 0.15, 0.2, 0.25] for Stage 1
   - Test: [0.02, 0.05, 0.1] for Stage 2
   - Find Pareto frontier of speed vs accuracy

2. **Optimal cutoff distances?**
   - LJ cutoff: test [6Å, 8Å, 10Å]
   - Electrostatics cutoff: test [4Å, 6Å, 8Å]
   - Balance speed vs accuracy

3. **Charge assignment accuracy?**
   - Current: Very crude rules
   - Could use: AM1-BCC, Gasteiger, etc.
   - Trade-off: accuracy vs computation time

4. **Voxel grid resolution?**
   - Test [0.3Å, 0.5Å, 0.75Å, 1.0Å]
   - Finer grid = more accurate but slower
   - Find sweet spot

5. **Spatial hash cell size?**
   - Should match cutoff distance
   - Test different sizes for performance

**Action**: Create experimental framework to sweep these parameters systematically.

---

## Next Steps

1. **Implement Stage 1** (1 day)
   - Build voxel grid infrastructure
   - Implement clash detection
   - Test rejection rates

2. **Implement Stage 2** (2 days)
   - Build spatial hash
   - Implement simplified scoring
   - Validate energy correlation

3. **Integrate and Test** (1 day)
   - Connect all stages
   - Run benchmarks
   - Optimize performance

4. **Validate on Real Complexes** (1 day)
   - Test on diverse targets
   - Collect RMSD statistics
   - Iterate on parameters

**Total timeline**: 5 days to fully validated multi-stage pipeline

---

## References and Prior Art

1. **DOCK 6**: Uses multi-stage scoring (grid-based filtering → full scoring)
2. **AutoDock Vina**: Multi-threaded scoring with early termination
3. **Glide**: Hierarchical filtering (HTVS → SP → XP)
4. **FRED**: Multi-stage docking with progressive filtering

Key insight: All major docking programs use multi-stage approaches. We're implementing proven technology.
