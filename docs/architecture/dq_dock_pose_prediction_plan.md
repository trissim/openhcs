# DQ-Dock Pose Prediction Implementation Plan

## Current Status

### ✅ What DQ-Dock HAS (core infrastructure):

| Component | Status | Notes |
|-----------|--------|-------|
| **srank computation** | ✅ Complete | Identifies decision-relevant atoms |
| **MD engine** | ✅ Complete | VelocityVerlet, Langevin integrators |
| **Tractability routing** | ✅ Complete | Selects algorithm based on srank (TRACTABLE vs HARD) |
| **Energy/potential** | ✅ Complete | LennardJones potential |
| **CLI** | ✅ Complete | Can run docking simulations |

### ❌ What's MISSING for full pose prediction:

| Component | Status | What It Does |
|-----------|--------|--------------|
| **Ligand placement/search** | ❌ Missing | Generate ligand poses in binding pocket |
| **Docking box specification** | ❌ Missing | Define search space (center, size) |
| **Protein-ligand scoring** | ⚠️ Simple LJ | Need full force field (AMBER/CHARMM-like) |
| **Pose scoring/ranking** | ❌ Missing | Rank multiple poses by affinity |
| **RMSD to native** | ❌ Not integrated | Compare to crystal structure |
| **Exhaustiveness search** | ❌ Missing | How thoroughly to search |

## Current "Docking" = Just MD Simulation

Looking at `cli.py:38-64`:

```python
def _cmd_dock(args):
    # Loads protein + ligand
    # Computes srank
    # Runs MD on the combined system
    # Returns FINAL ENERGY only
```

**Problem:** It's running MD, not docking! There's no:
- Ligand sampling/placement
- Multiple pose generation
- Scoring and ranking

## Implementation Plan

### Phase 1: Core Docking Infrastructure (Priority: HIGH)

#### 1.1 Ligand Placement Module
- [ ] Create `dq_dock_engine/docking/ligand_placement.py`
- [ ] Implement random ligand placement in binding pocket
- [ ] Implement grid-based ligand placement
- [ ] Implement fragment-based ligand assembly
- [ ] Add torsional angle sampling for flexible ligands

#### 1.2 Docking Box Specification
- [ ] Add `DockingBox` dataclass with center, size
- [ ] Auto-detect binding pocket from protein structure
- [ ] Support manual specification of box parameters
- [ ] Validate box doesn't extend outside protein

#### 1.3 Scoring Function Integration
- [ ] Create `dq_dock_engine/docking/scoring.py`
- [ ] Integrate SMINA/QuickVina as external scorer
- [ ] Add internal scoring (improved LJ + electrostatics)
- [ ] Support hybrid scoring (internal + external)

### Phase 2: Pose Generation & Ranking (Priority: HIGH)

#### 2.1 Multi-Pose Generation
- [ ] Implement genetic algorithm for pose generation
- [ ] Implement Monte Carlo sampling
- [ ] Add parallel pose evaluation
- [ ] Track pose history for analysis

#### 2.2 Pose Ranking
- [ ] Add scoring-based ranking
- [ ] Add clustering for diverse poses
- [ ] Implement RMSD-based deduplication
- [ ] Add filtering by energy thresholds

### Phase 3: Evaluation & Benchmarking (Priority: MEDIUM)

#### 3.1 RMSD Computation
- [ ] Create `dq_dock_engine/docking/metrics.py`
- [ ] Implement Kabsch RMSD algorithm
- [ ] Add heavy-atom RMSD
- [ ] Add ligand-only RMSD (ignoring protein movement)

#### 3.2 Benchmark Infrastructure
- [ ] Integrate with existing PDBbind loader
- [ ] Add CASF benchmark support
- [ ] Implement success rate metrics (RMSD < 2Å)
- [ ] Add correlation plots (predicted vs experimental)

#### 3.3 SMINA Comparison
- [ ] Run side-by-side DQ-Dock vs SMINA
- [ ] Compare RMSD to crystal structures
- [ ] Compare timing/speedup
- [ ] Generate comparative figures

### Phase 4: Advanced Features (Priority: LOW)

#### 4.1 Protein Flexibility
- [ ] Add side-chain rotamer sampling
- [ ] Implement induced fit docking
- [ ] Add backbone flexibility options

#### 4.2 Water Modeling
- [ ] Add water placement
- [ ] Support water toggle in scoring
- [ ] Add hydration site analysis

#### 4.3 Co-factors & Ions
- [ ] Support metal ion handling
- [ ] Add co-factor preservation
- [ ] Handle modified residues

## File Structure

```
dq_dock_engine/
├── docking/
│   ├── __init__.py
│   ├── ligand_placement.py    # NEW: Ligand pose generation
│   ├── scoring.py             # NEW: Scoring functions
│   ├── search.py              # NEW: Search algorithms
│   ├── metrics.py             # NEW: RMSD, etc.
│   └── pipeline.py            # NEW: End-to-end docking
├── benchmark/
│   ├── benchmark_pdb.py       # Existing
│   └── sota_comparison.py    # Existing
└── ...
```

## Key Design Decisions

### 1. Hybrid Architecture
Use srank to identify relevant pocket atoms, then use efficient sampling in reduced space. The key insight is that srank tells us which atoms matter for binding - we can sample ligand poses while keeping only relevant protein atoms.

### 2. External Scoring Integration
Rather than reimplementing a full scoring function, integrate SMINA as the primary scorer. This gives:
- Accurate binding affinity predictions
- RMSD to native in output
- Industry-standard results

### 3. Benchmark-First Development
Before implementing features, define the benchmark:
- Download PDBbind refined set (~5000 complexes)
- Define success metrics (RMSD < 2Å)
- Run SMINA as baseline

## Quick Start (MVP)

For the minimum viable product:

1. Add ligand placement in binding pocket
2. Use SMINA to score each pose
3. Return best pose by score
4. Compare RMSD to crystal

```python
# Pseudocode for MVP
def dock_mvp(protein, ligand, pocket_center):
    # 1. Generate ligand poses in pocket
    poses = generate_poses(ligand, pocket_center, n=100)
    
    # 2. Score each pose with SMINA
    scored = [(pose, score_with_smina(protein, pose)) for pose in poses]
    
    # 3. Return best
    return min(scored, key=lambda x: x[1])
```

## References

- AutoDock Vina: https://github.com/ccsb-scripps/AutoDock-Vina
- SMINA: https://sourceforge.net/projects/smina/
- PDBbind: http://www.pdbbind.org.cn/
- CASF: https://casf.biocaddie.org/

## Timeline Estimate

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| Phase 1 | Core Infrastructure | 2-3 weeks |
| Phase 2 | Pose Generation | 2-3 weeks |
| Phase 3 | Evaluation | 1-2 weeks |
| Phase 4 | Advanced Features | Ongoing |

---

*This plan was generated based on analysis of the current DQ-Dock codebase.*
