# Rich-Chemistry Optimization: 180° Flip Regression

## Summary

Two compounds (1hk4, 2ceq) dock with **0.00 Å Kabsch RMSD** to the native but with a **180° rigid-body rotation**. The internal conformation is atomically identical; the optimizer found a flipped local minimum instead of the native basin.

This is a regression introduced by enabling EXTENDED_RICH exact scoring inside the optimization loop, where previously only LJ scoring was used.

---

## Observed Failure Mode

| Compound | Raw RMSD | Kabsch RMSD | Rotation Angle | Native Rank | Energy Gap |
|----------|----------|-------------|----------------|-------------|------------|
| 1hk4     | 12.00 Å  | **0.00 Å**  | **180°**       | 1           | 1.55 kcal/mol |
| 2ceq     | 5.77 Å   | **0.00 Å**  | **180°**       | 1           | —          |

**Key diagnostic:** `kabsch_rmsd(docked, native) == 0.00 Å` for both. The two poses are related by a pure rigid-body transformation. The internal atom-atom geometry is perfect; only the orientation in space is wrong.

**Visual signature (see benchmark plots):**
- XY/XZ projections show the two molecules with identical shape but one flipped
- YZ projection shows a clean translation (centroid offset ~4–9 Å)

Pose files for inspection:
- [1hk4 native](../../benchmark_results_metal_dqonly/pdb_redocking_20260322_161650_poses/1hk4_native_ligand.pdb)
- [1hk4 docked](../../benchmark_results_metal_dqonly/pdb_redocking_20260322_161650_poses/1hk4_dq_dock_pose.pdb)
- [2ceq native](../../benchmark_results_metal_dqonly/pdb_redocking_20260322_161650_poses/2ceq_native_ligand.pdb)
- [2ceq docked](../../benchmark_results_metal_dqonly/pdb_redocking_20260322_161650_poses/2ceq_dq_dock_pose.pdb)

PyMOL sessions:
- [1hk4 pose compare](../../benchmark_results_metal_dqonly/pdb_redocking_20260322_161650_pymol/1hk4_pose_compare.pse)

---

## Root Cause: Exact Scoring in Optimization Loop Changed

### The critical code change

File: [`dq_dock_engine/docking/formal_surrogates.py`](../../dq_dock_engine/docking/formal_surrogates.py)

Function: `score_exact_and_coarse_round` (JIT-compiled, called every optimization round)

**Before (committed at `656a108a`):**
```python
# Lines 473–491 in 656a108a
if False: # Temporarily disabled for speed test
    exact_batch = score_certified_rich_chemistry_batch(
        receptor_coords=receptor_coords[retained_indices],
        ...
        rich_chemistry_plan=plan_subset,
        ...
    )
else:
    exact_batch = score_certified_batch(   # <-- LJ-only was active
        receptor_coords=receptor_coords[retained_indices],
        ...
        electrostatics=_subset_electrostatics(electrostatics, retained_indices),
    )
```

**After (current working tree, introduced to fix JIT tracer leak):**
```python
exact_batch = effective_scoring_context.score_exact_batch(
    receptor_coords=receptor_coords[retained_indices],
    ...
)
```

When `scoring_context.exact_chemistry_mode == ExactChemistryMode.EXTENDED_RICH`, `score_exact_batch` calls `score_certified_rich_chemistry_batch`, which adds:
- Screened Coulomb electrostatics
- Contact/desolvation surrogate
- Directional H-bond scoring
- Metal coordination scoring
- Pi-stacking scoring

**Before the change:** the optimization loop minimized LJ energy only. Rich chemistry was only used for the *final scoring/ranking* step.

**After the change:** the optimization loop minimizes full EXTENDED_RICH energy. The energy landscape is different — it has additional anisotropic terms that create different local minima.

---

## Why This Produces a 180° Flip

The 180° rotation is a precise symmetry artifact. Many drug-like molecules have approximate bilateral symmetry (thyroxine in 1hk4 has a symmetric diphenyl ether backbone; 2ceq's ligand has similar pseudo-symmetry). Under LJ-only scoring, this symmetry means both orientations have nearly equal steric energy, so the optimizer's trajectory determines which basin it falls into — and historically it found the native basin.

Under EXTENDED_RICH scoring:
1. Directional H-bond terms prefer specific N-H···O orientations → the 180°-rotated pose breaks or inverts these vectors
2. Pi-stacking terms (`PiStackingInteractionTerm`) score face-alignment → a 180° flip preserves face alignment (cos² of 180° = 1)
3. Metal coordination is isotropic → unchanged by flip
4. Net effect: the 180°-rotated pose may score **better** under rich chemistry than the native orientation for certain ligands, creating a false global minimum in the optimization landscape

The optimizer (`refine_poses_certified` in [`formal_optimizer.py`](../../dq_dock_engine/docking/formal_optimizer.py)) starts from randomly sampled global poses. For pseudo-symmetric ligands, ~half of the initial poses will be closer to the flipped orientation. Under EXTENDED_RICH scoring those poses descend into the flipped local minimum and cannot escape via small perturbation steps.

---

## What the Lean Theorems Certify (and What They Don't)

The certified path is proven in [`DecisionQuotient/`](../../docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/):

**What IS proven:**
- Given a fixed set of sampled poses, the optimizer correctly identifies the **top-1 among those poses** (Lean: `FLO9`, `FLO8`, `CP2/TK11/TK12`)
- The coarse-exact gap bound δ is valid: `|exact(a) - coarse(a)| ≤ δ` for all poses
- The energy gap proof comparing native vs. sampled poses is sound

**What is NOT proven:**
- That the sampled+optimized pose set contains any pose near the crystal structure
- That the optimization landscape under EXTENDED_RICH scoring has the native pose as the unique local minimum
- That the 180°-rotated pose has higher energy than the native under EXTENDED_RICH scoring

**Native Rank = 1 is still proved** because the pre-optimization native energy is compared against the post-optimization sampled poses — and the native wins. But the sampled poses are all stuck in the flipped basin, so the "winner" among sampled poses is a flipped pose.

---

## Files to Audit

### Core optimization loop

| File | Lines | Issue |
|------|-------|-------|
| [`formal_surrogates.py`](../../dq_dock_engine/docking/formal_surrogates.py) | `score_exact_and_coarse_round` (~440–510) | Exact scoring was LJ; now EXTENDED_RICH. Does the formal certificate still apply? |
| [`formal_optimizer.py`](../../dq_dock_engine/docking/formal_optimizer.py) | `_refine_round`, `refine_poses_certified` | Passes `scoring_context` to JIT core; pre-subsetting introduced here |
| [`scoring_context.py`](../../dq_dock_engine/docking/scoring_context.py) | `score_exact_batch` | Routes to rich chemistry when `exact_chemistry_mode == EXTENDED_RICH` |

### Rich chemistry scoring

| File | Lines | Issue |
|------|-------|-------|
| [`scoring.py`](../../dq_dock_engine/docking/scoring.py) | `score_certified_rich_chemistry_batch` | Is the aggregate score antisymmetric under 180° rotation? Pi-stacking face alignment is symmetric. |
| [`chemistry_runtime.py`](../../dq_dock_engine/docking/chemistry_runtime.py) | `PiStackingInteractionTerm._pair_scores` | `face_alignment = |cos θ|` — symmetric under 180° flip. Creates degenerate score for flipped poses. |

### Lean theorems

| File | Relevant theorem | Question |
|------|-----------------|----------|
| [`DirectionalHBondApproximation.lean`](../../docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DirectionalHBondApproximation.lean) | `exact_vs_coarse_directionalHBond_certified_top1` | Proves coarse approximates exact — but does not prove the native is the unique optimum |
| [`PiStackingApproximation.lean`](../../docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/PiStackingApproximation.lean) | `scaledPiStackingRadial_lipschitz` | Pi-stacking uses `|cos θ|` which is symmetric under 180° flip — the score is identical for pose and flipped pose |

---

## The Pi-Stacking Symmetry Problem

The most likely culprit is this line in [`chemistry_runtime.py:273`](../../dq_dock_engine/docking/chemistry_runtime.py):

```python
face_alignment = jnp.abs(
    jnp.sum(
        receptor_normals[None, :, None, :] * ligand_normals[:, None, :, :],
        axis=-1,
    )
)
```

`|cos θ|` is symmetric under θ → θ + 180°. A ligand flipped 180° has **identical pi-stacking score** to the correctly oriented ligand, but a **different H-bond score** (directional H-bonds are NOT symmetric under 180° flip). If the H-bond term is weak or absent for a given compound, the pi-stacking term creates a degenerate energy landscape where the flipped pose is indistinguishable from the native.

Under LJ-only optimization (old behavior), this degeneracy didn't matter because LJ scores are radially symmetric — the optimizer would converge based on purely geometric fit, not orientation. Under EXTENDED_RICH optimization (new behavior), the combined landscape may have a slightly lower energy at the flipped pose due to accidental complementarity.

---

## Proposed Fixes (for auditing agents)

### Option A: Restore LJ-only optimization, keep rich chemistry for ranking only
Restore `if False:` gate in `score_exact_and_coarse_round` (or equivalent) so the optimization loop uses LJ-only, and rich chemistry is applied only at the final ranking step. This was the original design intent.

**Risk:** The Lean certificate for the optimization step would again be proven over LJ, not over the final rich chemistry scoring function.

### Option B: Fix the H-bond directionality to break the 180° degeneracy
Ensure that the H-bond scoring term (which IS directional) always has non-zero contribution for the failing compounds, so the combined EXTENDED_RICH landscape is not degenerate under 180° rotation.

### Option C: Increase n_poses or add diversified initialization
Sample both orientations explicitly in the initial pose set so the optimizer has a chance to find the native basin regardless of energy landscape.

### Option D: Use EXTENDED_RICH for ranking only, not for the optimization gradient
Keep the optimization using LJ scoring (smooth, radially symmetric) for convergence, but use EXTENDED_RICH for selecting among the final optimized poses.

---

## Reproduction

```bash
# Run 1hk4 with current code
python -m dq_dock_engine.benchmark.benchmark_pdb --pdbs 1hk4 --n-poses 10000

# Compute Kabsch RMSD
python3 -c "
import numpy as np
# ... (see kabsch_rmsd computation above)
"
```

The failure is deterministic — same seed produces same 180°-flipped result.
