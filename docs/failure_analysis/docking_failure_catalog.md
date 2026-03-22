# Docking Failure Catalog

Benchmark: `benchmark_results_metal_dqonly/pdb_redocking_20260322_161650`
Scoring mode: `EXTENDED_RICH` (exact in optimization loop)
n_poses = 10,000 · n_opt_steps = 10

---

## Quick Reference

| PDB   | Raw RMSD | Kabsch RMSD | Rotation | Centroid Δ | Native Rank | Energy Gap | Category |
|-------|----------|-------------|----------|-----------|-------------|------------|----------|
| 1hk4  | 12.00 Å  | 0.000 Å     | **180°** | 11.0 Å    | 1           | +1.55 kcal/mol | Class A — rigid flip |
| 2ceq  | 5.77 Å   | 0.000 Å     | **180°** | 4.46 Å    | 24          | +2.76 kcal/mol | Class A — rigid flip |
| 5er1  | 6.58 Å   | 0.000 Å     | **90°**  | 4.64 Å    | 1           | +0.44 kcal/mol | Class A — rigid flip |
| 1fh7  | 6.49 Å   | 0.000 Å     | **90°**  | 1.77 Å    | 2           | +1.40 kcal/mol | Class A — rigid flip |
| 1q8t  | 1.43 Å   | 0.000 Å     | 12°      | 1.18 Å    | 8           | +4.74 kcal/mol | Class B — tail displacement |

**Kabsch RMSD = 0.000 Å** for all five compounds: every failure is a pure rigid-body transformation
of a geometrically perfect pose. The optimizer found the correct molecular conformation in all cases;
it placed it in the wrong orientation.

---

## Class A — Rigid-Body Orientation Flip (4 compounds)

### Diagnostic signature

`kabsch_rmsd(docked, native) == 0.000 Å` with a non-zero rotation angle.
The two poses share identical internal atom–atom geometry; only the global orientation differs.
This means the energy landscape has (at least) two local minima that are related by a rigid-body
rotation, and the optimizer fell into the wrong one.

### Sub-class A1 — 180° flip (1hk4, 2ceq)

**Rotation:** 180° about an axis roughly perpendicular to the ligand's long axis.

**Root cause — pi-stacking score symmetry under 180° flip:**

In [`dq_dock_engine/docking/chemistry_runtime.py:273`](../../dq_dock_engine/docking/chemistry_runtime.py):

```python
face_alignment = jnp.abs(
    jnp.sum(
        receptor_normals[None, :, None, :] * ligand_normals[:, None, :, :],
        axis=-1,
    )
)
```

`|cos θ|` is symmetric under θ → θ + 180°, so a ligand flipped 180° has **identical pi-stacking
score** to the correctly-oriented ligand. The H-bond term (`DirectionalHBondApproximation.lean`)
uses signed `cos θ` and IS antisymmetric under flip — but only when H-bond strengths are non-zero.

**Why 1hk4 and 2ceq specifically fail:**
- 1hk4 ligand = thyroxine (iodinated diphenyl ether). Primarily hydrophobic; weak H-bond donors.
- 2ceq ligand = small pseudo-symmetric molecule with limited H-bond donors.

For these compounds, H-bond strengths ≈ 0, so the flip penalty from the directional H-bond term
≈ 0. The pi-stacking symmetry dominates → degenerate landscape → optimizer falls into whichever
orientation appears first in the sampled poses (≈50/50 chance).

**Lean impact:**
`PiStackingApproximation.lean` models `face_alignment = |cos θ|` explicitly.
The Lean proof is consistent with the code but does not detect the degeneracy.

---

### Sub-class A2 — 90° rotation (5er1, 1fh7)

**Rotation:** 90° about a vertical axis through the binding pocket.

**Root cause — molecular symmetry:**

Both ligands have approximate 4-fold or quasi-4-fold rotational symmetry:

- **5er1** (renin-inhibitor analogue): Atom names follow a `CA/CA1/CA2/CA3`, `CB/CB1/CB2/CB3`
  pattern — four equivalent branches. A 90° rotation maps the molecule onto itself. Under *any*
  scoring function, both orientations score identically.

- **1fh7** (xylanase inhibitor, xylobiose/xylose sugar): The pyranose ring bound in the glycosidase
  active site sits in a pseudo-symmetric channel. The C2-symmetric sugar scaffold combined with the
  pocket's near-4-fold hydrogen-bond geometry creates a 90°-degenerate landscape.

**Compounding factor — H-bond contacts scored as VdW clashes:**

The native pose for both ligands has polar–polar distances well within H-bond range but below the
all-heavy-atom VdW contact sum, so the LJ term penalizes the native orientation:

| Compound | Clash in native | Distance | VdW sum | Clash |
|----------|-----------------|----------|---------|-------|
| 5er1 | CB – GLY217 O | 2.69 Å | 3.22 Å | +0.53 Å |
| 5er1 | O – ASP32 OD1 | 2.72 Å | 3.04 Å | +0.32 Å |
| 1fh7 | O2 – ASN126 ND2 | 2.65 Å | 3.07 Å | +0.42 Å |
| 1fh7 | O3 – HIS80 NE2  | 2.72 Å | 3.07 Å | +0.35 Å |
| 1fh7 | O2 – TRP273 NE1 | 2.73 Å | 3.07 Å | +0.34 Å |
| 1fh7 | O3 – LYS47 NZ   | 2.78 Å | 3.07 Å | +0.29 Å |

All of these are genuine hydrogen bonds (O-H···N or N-H···O, normal H-bond range 2.5–3.3 Å)
that the all-heavy-atom LJ scoring interprets as clashes. The rotated pose avoids these apparent
clashes (docked 1fh7 has only 2 minor clashes vs. 5 in the native), so the optimizer actively
prefers the wrong orientation.

This means for 90°-rotation failures, there are **two compounding causes**:
1. Molecular symmetry makes both orientations score equally (unfixable without asymmetric
   chemical features)
2. H-bond contacts penalised as VdW clashes tilt the landscape toward the rotated pose (fixable)

---

## Class B — Tail Displacement (1q8t)

**Rotation:** 12° · **Centroid offset:** 1.18 Å · **RMSD:** 1.43 Å

### Ligand and failure characterization

1q8t = PKA + staurosporine analog. The staurosporine scaffold is a rigid tetracyclic core anchored
by N11···VAL123 (backbone N–H, preserved at 2.91 Å in both poses) and N43···THR51 O (H-bond,
preserved at 2.84 Å native / 2.87 Å docked). The core moves only 0.09–0.24 Å.

The failure is localised to the **aminocyclohexane tail**: C42 moves 2.39 Å, N43 moves 2.40 Å.

### Root cause — C-H···O contacts misidentified as VdW clashes

In the crystal structure, C42 sits at:

| Contact | Distance | VdW sum | Apparent clash |
|---------|----------|---------|----------------|
| C42 – ASP184 OD1 | 2.93 Å | 3.22 Å | **+0.29 Å** |
| C42 – ASN171 OD1 | 2.96 Å | 3.22 Å | **+0.26 Å** |

C42 is the methylene adjacent to the primary amine N43. In explicit-H geometry, the C42 methylene
hydrogens form **C-H···O contacts** with ASP184 and ASN171 carboxylate oxygens — a well-documented
interaction in kinase-ligand binding. At C···O = 2.93 Å, the C–H hydrogen is ≈1.85 Å from the
oxygen, well within C-H···O H-bond range (1.7–2.5 Å).

The all-heavy-atom LJ scoring applies strong repulsion (r⁻¹²) at 2.93 Å (sum of vdW radii 3.22 Å),
penalising the correct pose by ~4.7 kcal/mol. The optimizer correctly minimizes energy by shifting
C42 2.39 Å away from ASP184/ASN171, landing in a local minimum where these contacts are broken.

**New false contact created in docked pose:**
N43 drifts 2.4 Å toward ARG18, bringing N43···ARG18 CD from 6.04 Å → 3.85 Å. This may generate
a spurious screened-Coulomb or contact-surrogate benefit (ARG18 is positively charged; N43 is a
primary amine). This adds a ~4.74 kcal/mol energy gap on top of the lost clash penalty.

---

## Root Causes Summary

### Issue 1: EXTENDED_RICH scoring active in optimization loop

This is the **enabling change** that triggered all five failures.

Before commit `656a108a`, the optimization loop used LJ-only scoring (`if False:` gate).
EXTENDED_RICH was applied only at the final ranking step. The current code runs EXTENDED_RICH
inside every optimization round, introducing orientation-dependent anisotropic terms into the
energy landscape.

Under LJ-only optimization:
- No pi-stacking symmetry problem (LJ is radially symmetric)
- C-H···O distances still appear as clashes, but the LJ-only landscape has broader basins that
  tolerate small offsets better

Fix: Restore LJ-only optimization inside the refinement loop; apply EXTENDED_RICH only for
final ranking. See `score_exact_and_coarse_round` in
[`dq_dock_engine/docking/formal_surrogates.py`](../../dq_dock_engine/docking/formal_surrogates.py).

---

### Issue 2: All-heavy-atom LJ radii penalize genuine H-bond contacts

Affects: 5er1, 1fh7 (Class A2 compounding factor), 1q8t (Class B primary driver).

H-bonds have donor···acceptor heavy-atom distances of 2.5–3.3 Å for N/O donors, and C-H···O
contacts have C···O distances of 2.7–3.2 Å. The LJ sum-of-radii for these pairs:

| Pair   | Sum of vdW radii |
|--------|-----------------|
| N – O  | 3.07 Å          |
| O – O  | 3.04 Å          |
| C – O  | 3.22 Å          |
| C – N  | 3.25 Å          |

All genuine H-bond distances fall inside the LJ repulsion zone under default radii.

**Fix options:**
- **Option A**: Reduce effective C radius to 1.40 Å (or N/O to 1.35 Å) when paired with a
  heteroatom — specifically for contacts below 3.5 Å. This allows C-H···O and X-H···Y contacts to
  fall near the LJ minimum rather than the repulsive wall.
- **Option B**: Add explicit polar-H representation for N-H and O-H donors. Most rigorous but
  requires infrastructure changes throughout `chemistry_runtime.py`.
- **Option C**: Use the `CertifiedDirectionalHBondSpec` to encode C42-type C-H donors as weak
  H-bond donors with appropriate strength constants.

The fix is in the `CertifiedScreenedCoulombSpec` and LJ parameterisation in
[`dq_dock_engine/docking/scoring.py`](../../dq_dock_engine/docking/scoring.py) and
[`dq_dock_engine/docking/chemistry_runtime.py`](../../dq_dock_engine/docking/chemistry_runtime.py).

---

### Issue 3: Ligand molecular symmetry (Class A2 only)

5er1 and 1fh7 have approximate rotational symmetry that makes multiple orientations physically
equivalent. This is not fixable by improving the scoring function — the orientations genuinely
score identically because they are related by a molecular symmetry operation.

**Fix options:**
- **Option A (detection)**: Compute the ligand's point group symmetry order before docking. If
  order > 1, sample initial poses to cover all symmetry-related orientations, and report all
  equivalent native poses for RMSD evaluation.
- **Option B (evaluation)**: Compute Kabsch RMSD against all symmetry-equivalent poses of the
  crystal structure, report the minimum. A 90° rotation of a C4-symmetric molecule should score
  RMSD = 0.000 Å.
- **Option C (practical)**: Accept that for symmetric molecules, the reported pose is one of
  several physically correct answers. Flag these during benchmarking rather than counting as
  failures.

---

## Refactoring Priority

| Priority | Issue | Affected compounds | Estimated fix difficulty |
|----------|-------|--------------------|--------------------------|
| **P1** | Restore LJ-only optimization loop | 1hk4, 2ceq, 5er1, 1fh7 | Low — revert one code branch |
| **P2** | Calibrate H-bond heavy-atom radii | 5er1, 1fh7, 1q8t | Medium — requires LJ reparameterisation + re-benchmarking |
| **P3** | Symmetric ligand detection + RMSD correction | 5er1, 1fh7 | Medium — RDKit symmetry analysis |
| **P4** | C-H donor encoding in DirectionalHBond | 1q8t, others | High — requires chemical assignment rules |

---

## Related Files

| File | Relevance |
|------|-----------|
| [`dq_dock_engine/docking/formal_surrogates.py`](../../dq_dock_engine/docking/formal_surrogates.py) | `score_exact_and_coarse_round` — switch EXTENDED_RICH back to ranking-only |
| [`dq_dock_engine/docking/chemistry_runtime.py`](../../dq_dock_engine/docking/chemistry_runtime.py) | `PiStackingInteractionTerm._pair_scores` — `\|cos θ\|` symmetry; LJ radii parameterisation |
| [`dq_dock_engine/docking/scoring.py`](../../dq_dock_engine/docking/scoring.py) | `score_certified_rich_chemistry_batch` — aggregate score composition |
| [`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/PiStackingApproximation.lean`](../papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/PiStackingApproximation.lean) | `scaledPiStackingRadial_lipschitz` — certifies `\|cos θ\|` formulation |
| [`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DirectionalHBondApproximation.lean`](../papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DirectionalHBondApproximation.lean) | `exact_vs_coarse_directionalHBond_certified_top1` — signed face term |
| [`docs/failure_analysis/rich_chemistry_180deg_flip_regression.md`](rich_chemistry_180deg_flip_regression.md) | Detailed analysis of 1hk4 / 2ceq 180° flip |
| [`docs/failure_analysis/native_basin_proof_tractability.md`](native_basin_proof_tractability.md) | Assessment of how far the Lean proofs are from certifying native-basin convergence |
