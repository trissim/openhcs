# Unified Blind-Conformer + Energy/RMSD Runtime Plan

## Objectives

- Remove benchmark leakage from crystal ligand geometry by default.
- Make blind-conformer pruning theorem-driven instead of rigid-only heuristic.
- Replace unstable batch-max rich-chemistry pruning deltas with certified analytic bounds.
- Derive refinement iteration budgets from Lean-backed energy/RMSD certificates instead of fixed heuristics.
- Eliminate the heuristic `top_k_to_optimize` parameter by using theorem-derived canonical retain set sizing.
- Keep runtime changes JAX-safe and performant.

## Lean proof surface (verified present)

### Cross-docking / blind-conformer pruning

| File | Key theorems |
|------|-------------|
| `CrossDockingCertificates.lean` | `better_than_incumbent_survives_lowerBound_pruning`, `incumbent_beats_all_points_in_strain_cell`, `strain_augmented_certified_top1_sound`, `detected_pocket_restricted_problem_tractable_and_sufficient`, `ensemble_rigid_strain_certified_top1_sound` |
| `BlindConformerPipelineOptimality.lean` | `canonicalRetain_certifiedSafe`, `canonicalRetain_minimizes_pipelineCost`, `rigid_plus_bound_pipeline_optimal` |
| `BlindConformerPipelineRefinements.lean` | `energyTopK_subset_canonicalRetain`, `canonical_twoStage_le_allExact`, `minimal_adequate_seedBudget_optimal`, `flexCorrected_is_certified_lowerBound`, `coarse_rigid_with_flex_and_conformer_lower_bound` |
| `BlindConformerRuntimeCertificates.lean` | `bounded_channel_uniformApprox_zero`, `bounded_channel_sum_uniformApprox_zero`, `base_plus_omitted_uniformApprox`, `exact_with_omitted_ge_coarse_minus_totalError`, `pose_specific_improvement_bound_of_active_subset`, `canonical_pruning_and_budget_optimal`, **`omitted_channel_is_bounded_by_supremum`** |
| `SoftLJApproximation.lean` | `exact_vs_softened_lj_uniformApprox`, `softenedLJ_lipschitzWith`, **`exact_vs_softened_lj_error`**, **`softened_lj_self_approx_zero`** |
| `ExplicitWaterPlacement.lean` | `water_bridge_additive_with_base`, `discrete_placement_approximation`, **`water_bridge_is_bounded_omitted_channel`** |

### Energy/RMSD refinement budgeting

| File | Key structures/theorems |
|------|------------------------|
| `EnergyRMSDConvergence.lean` | `CertifiedQuadraticBasin`, `CertifiedSegmentCurvature`, `CertifiedLocalSpectralEnclosure`, `CertifiedOneStepEnergyContraction`, `CertifiedGradientDescentDynamics`, `rmsd_target_of_canonicalIterationBudgetFromLocalCertificates`, `rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics` |
| (new) `SE3JacobianBridge.lean` | **`parameterSpace_quadraticBasin_transfers_to_coordSpace`** — bridges SE(3) Hessian eigenvalues to coordinate-space quadratic growth via kinematics Jacobian singular values |

---

## Current runtime state (verified)

### Already implemented

| Feature | Location | Status |
|---------|----------|--------|
| Blind ligand geometry default | `benchmark_pdb.py:795` — `use_crystal_ligand_geometry` flag | Working; crystal is opt-in |
| Canonical retain mask | `pipeline.py:1640` — `_canonical_retain_mask()` | Working; used in conformer pruning path |
| Conformer improvement bound | `pipeline.py:1621` — `_conformer_improvement_bound()` | Working; returns `2 × Σ Vk` (global, not pose-specific) |
| Receptor-flex corrections | `pipeline.py:1487` — `posewise_receptor_flex_error_softened_batch()` | Working; adds posewise flex error + 2×coarse_phys_delta to additive_correction |
| Extended term pre-filtering | `scoring.py:895` — `filter_active()` on `CertifiedExtendedInteractionBundle` | Working; eliminates inactive extended terms at construction time |
| Fixed-pad conformer batching | `conformer_search.py:608` — `_BB_PAD_SIZE=128` | Working; JIT compiles once |
| JIT float() bug in formal_surrogates | `formal_surrogates.py:439` — `score_exact_and_coarse_round` | Fixed |

### Not implemented

| Feature | Gap description |
|---------|----------------|
| Analytic 3-tier delta bounds | No code computes analytic deltas; all use batch-max `jnp.max(abs(exact-coarse))`. Lean theorems proven: `softened_lj_self_approx_zero`, `omitted_channel_is_bounded_by_supremum`, `water_bridge_is_bounded_omitted_channel` |
| Pose-specific torsion budgets | `_conformer_improvement_bound()` returns a single global value for all poses |
| Refinement certificate records | No `QuadraticBasin`, `SpectralEnclosure`, `GradientDescentDynamics` classes exist in Python |
| Formulaic `n_opt_steps` | Hardcoded default of 50 at `pipeline.py:186` |
| Theorem-derived retain set sizing | `top_k_to_optimize=200` is a heuristic; should be replaced by canonical retain with `k=top_k` |
| Seed-budget optimization | No `AdequateSeedBudget` runtime logic |

---

## Problem 1: Pruning delta explosion (MOST URGENT)

### Root cause

The current `_score_softened_pose_batch()` calls `scoring_context.score_softened_batch()` which returns:

```
total_delta = softened_lj.softening_error_bound
            + screened_coulomb.error_bound
            + attractive_batch.error_bound
```

Each component uses **batch-max discrepancy** (`jnp.max(jnp.abs(exact - coarse))`). With 50k–500k poses, rare outlier poses inflate delta to 10+ kcal/mol → no pruning.

### Why the previous plan was wrong

The previous version of this plan proposed "omit all non-LJ channels and use analytic value bounds." Three Lean theorems (now proven) expose why that doesn't work:

**1. Softened LJ error is unbounded** (`SoftLJApproximation.lean::exact_vs_softened_lj_error`):

The softening error `|exactLJ(r) - softenedLJ(r)|` equals `|LJ(r) - LJ(rSoft)|` when `r < rSoft`, which diverges as `r → 0`. On large batches, at least one pose will have a near-zero interatomic distance, making the batch-max delta arbitrarily large. The claim that δ_LJ is "stable" was **false**.

**2. Omitted channels require VALUE bounds, not tail errors** (`BlindConformerRuntimeCertificates.lean::omitted_channel_is_bounded_by_supremum`):

`base_plus_omitted_uniformApprox` requires `|channel_i(a,s)| ≤ Bᵢ` — a bound on the channel's **total energy value**, not its cutoff approximation error. For screened Coulomb, this would be the maximum electrostatic energy at minimum distance — enormous, not the small tail `Q × exp(-κR)`. Using tail errors as Bᵢ was **mathematically wrong**.

**3. Water bridges CAN be omitted** (`ExplicitWaterPlacement.lean::water_bridge_is_bounded_omitted_channel`):

Water bridges have a strict physical value bound (EWP4: `bridge ≤ 2` per bridge). This connects to `omitted_channel_is_bounded_by_supremum` via a one-line proof. Water bridges are the one channel where true omission is safe.

### Corrected solution: shared softened base + cutoff approximations + true omission only for value-bounded channels

The architecture has three tiers:

**Tier 1 — Shared base (δ = 0):** Use softened LJ as both exact and coarse LJ. `softened_lj_self_approx_zero` proves the approximation error is exactly 0. No softening term in the delta at all.

**Tier 2 — Cutoff approximations (δ = analytic tail bound):** For channels that have large peak values but exponential/Gaussian decay tails, use cutoff approximations in the coarse scorer. The error is the **tail** beyond the cutoff — small and analytic:

| Channel | Cutoff approx error (per pair) | Analytic bound | Source |
|---------|-------------------------------|----------------|--------|
| Screened Coulomb | `\|exact(r) - cutoff(r)\|` for `r ≥ R_cutoff` | `(q₁q₂/ε_r) × exp(-κ × R_cutoff) / R_cutoff` | Exponential screening decay |
| Contact | `\|exact(r) - cutoff(r)\|` for `r ≥ R_cutoff` | `W_max² × exp(-(β × R_cutoff)²)` ≈ 2.4e-6 (negligible) | `GaussianDecayBounds.lean::GD3` |
| H-bond (×2) | `\|exact(r) - cutoff(r)\|` for `r ≥ R_cutoff` | `S_max² × exp(-((R_cutoff - ideal)/width)²)` ≈ 0.105 | Gaussian tail from `ThermalFluctuationBounds.lean` |
| Metal | `\|exact(r) - cutoff(r)\|` for `r ≥ R_cutoff` | `S_max² × exp(-((R_cutoff - ideal)/width)²)` — often 0 (no metals) | `MetalCoordinationApproximation.lean` |
| Extended | Sum of per-term tail bounds | 0 after `filter_active()` for typical ligands | Already handled |

These are per-pair bounds, summed over all receptor-ligand pairs. Crucially: they depend on the **cutoff radius** and **physical parameters**, not on which poses are in the batch.

The Lean justification: `sum_uniformApprox` composes the cutoff approximation errors additively with the shared base.

**Tier 3 — True omission (δ = value bound):** Only for channels with small physical value bounds:

| Channel | Value bound `Bᵢ` | Source |
|---------|-------------------|--------|
| Cooperative H-bond | `\|α\| × N_channels²` where N_channels=2 → ≤ 1.2 | `CooperativeHBondApproximation.lean::CHN1` |
| Water bridges | `2 × N_bridges` | `ExplicitWaterPlacement.lean::water_bridge_is_bounded_omitted_channel` |

These are justified by `base_plus_omitted_uniformApprox` + `omitted_channel_is_bounded_by_supremum`.

**Resulting total delta**:

```
δ_total = 0                          (softened LJ: shared base)
        + Σ cutoff_tail_bounds       (Coulomb, contact, hbond, metal)
        + Σ omitted_value_bounds     (cooperative, water bridges)
```

Typical values: `0 + ~0.3 + ~1.2 = ~1.5 kcal/mol`. Batch-size independent.

### Implementation

#### Step A1: Eliminate softening error from delta

Change `_score_softened_pose_batch()` so the coarse scoring path uses softened LJ as the shared base. The `softening_error_bound` field should be **dropped from the delta** (set to 0), since the exact scoring path also uses softened LJ (`score_certified_rich_chemistry_batch` already calls `score_certified_softened_lj`).

```python
# In pipeline.py — _score_softened_pose_batch()
# The softening_error_bound tracks |exactLJ - softenedLJ| but both
# exact and coarse paths use softenedLJ, so this cancels.
# Lean: softened_lj_self_approx_zero
softening_delta = 0.0  # NOT coarse_batch.softening_error_bound
```

#### Step A2: Replace batch-max cutoff errors with analytic tail bounds

Add `analytic_cutoff_tail_bound() -> float` methods to each `Certified*Spec` dataclass. These compute the per-pair tail contribution at the cutoff radius and sum over all pairs.

```python
# In scoring.py — add to CertifiedScreenedCoulombSpec
def analytic_cutoff_tail_bound(self) -> float:
    """Per-pair tail bound summed over all pairs.
    Uses sum_uniformApprox (cutoff approx error, not channel value)."""
    n_pairs = self.receptor_charges.shape[0] * self.ligand_charges.shape[0]
    q_max_pair = (float(jnp.max(jnp.abs(self.receptor_charges)))
                  * float(jnp.max(jnp.abs(self.ligand_charges))))
    tail_per_pair = q_max_pair * math.exp(-self.kappa * self.cutoff) / (self.dielectric * self.cutoff)
    return tail_per_pair * n_pairs
```

Similar for `CertifiedContactSurrogateSpec` (Gaussian tail), `CertifiedDirectionalHBondSpec` (Gaussian tail), `CertifiedMetalCoordinationSpec` (Gaussian tail).

Also add `analytic_cutoff_tail_bound()` to the abstract base class `CertifiedOptionalInteractionTerm` (scoring.py:424) so that extended terms in `CertifiedExtendedInteractionBundle.terms` can be iterated uniformly in Step A3.

#### Step A3: Add `analytic_total_delta()` to `CertifiedRichChemistryPlan`

```python
def analytic_total_delta(self, n_water_bridges: int = 1) -> float:
    """Batch-size-independent delta from analytic bounds.

    Lean: sum_uniformApprox (tier 2) + base_plus_omitted_uniformApprox (tier 3).
    """
    # Tier 1: softened LJ shared base → 0
    # Tier 2: cutoff approximation tail bounds
    cutoff_delta = (
        self.screened_coulomb.analytic_cutoff_tail_bound()
        + self.contact.analytic_cutoff_tail_bound()
        + self.hbond_receptor_donor.analytic_cutoff_tail_bound()
        + self.hbond_ligand_donor.analytic_cutoff_tail_bound()
        + self.metal_coordination.analytic_cutoff_tail_bound()
        + sum(t.analytic_cutoff_tail_bound() for t in self.extended_terms.terms)
    )
    # Tier 3: omitted channel value bounds
    omitted_delta = (
        cooperative_hbond_correction_bound(self.cooperative_alpha, 2)
        + 2.0 * n_water_bridges  # water_bridge_is_bounded_omitted_channel
    )
    return cutoff_delta + omitted_delta
```

#### Step A4: Wire into `_score_softened_pose_batch()`

```python
def _score_softened_pose_batch(...) -> tuple[jnp.ndarray, float]:
    if scoring_context is not None and scoring_context.rich_chemistry_plan is not None:
        # Score with full rich chemistry (softened LJ + cutoff channels)
        coarse_batch = scoring_context.score_softened_batch(...)
        # Replace batch-max delta with analytic delta
        delta = scoring_context.rich_chemistry_plan.analytic_total_delta()
        return coarse_batch.scores, delta
```

The coarse *scores* remain the same (softened LJ + cutoff Coulomb + cutoff attractive). Only the *delta* changes from batch-max to analytic.

#### Step A5: No change to `_canonical_retain_mask`

The formula `tau = kth_score + delta` and `lower_bounds = scores - correction - delta` remains correct. The scores are full rich chemistry (softened LJ + all cutoff channels), and the delta now accounts for:
- Cutoff tail errors (how far the cutoff scores are from the true infinite-range scores)
- Omitted channel values (cooperative + water bridges)

---

## Problem 2: Pose-specific torsion budgets

### Current state

`_conformer_improvement_bound()` returns a single global value: `2 × Σ Vk` where Vk = 1.0 kcal/mol per bond. For a ligand with 5 rotatable bonds, this is 10.0 kcal/mol — a large correction applied identically to all poses.

### Lean theorem

`pose_specific_improvement_bound_of_active_subset`: If you know the active torsion subset for a pose, the improvement budget is `Σᵢ∈active Bᵢ` instead of `Σᵢ∈all Bᵢ`.

### How to determine active torsions per pose

A torsion bond is "active" for pose p if rotating it could improve the score. Concretely: if the ligand atoms moved by that torsion are within interaction range of the receptor at pose p.

**Runtime criterion**: bond i is active for pose p if `min_distance(rotating_atoms_i, receptor) < R_interaction` where `R_interaction` is the maximum cutoff across all interaction channels (typically ~6.0 Å for contact).

This can be computed as a batch operation:

```python
def _posewise_active_torsion_mask(
    poses_coords: jnp.ndarray,       # (B, N_lig, 3)
    receptor_coords: jnp.ndarray,    # (N_rec, 3)
    bonds: tuple[RotatableBond, ...],
    interaction_radius: float = 6.0,
) -> jnp.ndarray:                    # (B, n_bonds) bool
    """For each pose, which torsion bonds have rotating atoms in receptor range?"""
    # Per-bond: min distance from any rotating atom to any receptor atom
    # Shape: (B, n_bonds)
    masks = []
    for bond in bonds:
        rotating_coords = poses_coords[:, list(bond.rotating_atom_indices), :]  # (B, k, 3)
        dists = jnp.linalg.norm(
            rotating_coords[:, :, None, :] - receptor_coords[None, None, :, :],
            axis=-1,
        )  # (B, k, N_rec)
        min_dist_per_pose = jnp.min(dists, axis=(1, 2))  # (B,)
        masks.append(min_dist_per_pose < interaction_radius)
    return jnp.stack(masks, axis=-1)  # (B, n_bonds)
```

Then the pose-specific improvement bound is:

```python
# per_bond_barriers: (n_bonds,) = 2 × Vk per bond
# active_mask: (B, n_bonds) bool
posewise_improvement = jnp.sum(active_mask * per_bond_barriers[None, :], axis=-1)  # (B,)
```

This replaces the scalar `conformer_bound` with a per-pose vector, which slots directly into `_canonical_retain_mask`'s `additive_correction` parameter (already a per-pose array).

### Implementation

#### Step B1: Add `_posewise_active_torsion_mask()` to pipeline.py

As specified above. This is a pure JAX function — no JIT issues, no `float()` calls.

#### Step B2: Replace scalar conformer_bound with posewise vector in `_certified_pruning_pass()`

Change lines 1476-1481 from:

```python
conformer_bound, _ = _conformer_improvement_bound(rotatable_bonds)
additive_correction = jnp.full_like(coarse_scores, conformer_bound)
```

To:

```python
_, strain_params = _conformer_improvement_bound(rotatable_bonds)
per_bond_barriers = 2.0 * np.asarray(strain_params.barrier_heights)  # (n_bonds,)
active_mask = _posewise_active_torsion_mask(
    poses_coords, request.protein_coords, rotatable_bonds,
)
additive_correction = jnp.sum(
    active_mask * jnp.array(per_bond_barriers)[None, :], axis=-1,
)  # (B,) per-pose
```

---

## Problem 3: Formulaic refinement budget

### Why the previous plan was broken

The previous version computed a 3N×3N Hessian in Cartesian atom-coordinate space and fed eigenvalues into `CertifiedGradientDescentDynamics`. Three fatal issues:

**1. Wrong parameter space.** The optimizer works in 7D SE(3) (translation + quaternion), not 3N Cartesian. A rigid body has 6 zero-eigenvalue modes in Cartesian space → `lmin = 0` → `CertifiedQuadraticBasin` precondition `0 < μ` fails → falls back to fixed budget every time.

**2. Wrong optimizer dynamics.** The Lean theorem `CertifiedGradientDescentDynamics` assumes standard gradient descent: `x_{t+1} = x_t - α∇E(x_t)`. The actual optimizer (`optimization.py:75-97`) uses gradient norm clipping, direction normalization, and quaternion renormalization — a fundamentally different dynamics. The Lean contraction rate `q = (M-μ)/(M+μ)` doesn't describe this optimizer.

**3. Wrong cost estimate.** The plan said "90×90 Hessian, 4.5s per pose." The correct parameter space is 6D (rigid) or 6+n_bonds (flexible) → 6×6 Hessian → 0.3s per pose.

### The two Lean ingredients (independent)

The RMSD guarantee theorem `rmsd_target_of_canonicalAdequateIterationBudget` needs two independent certificates:

**Ingredient 1 — Quadratic basin** (`CertifiedQuadraticBasin`):
```
∀ x, (μ/2) × squaredDisplacement(x, center) ≤ E(x) - E(center)
```
This is a statement in **coordinate space** (`CoordSet n`). It converts energy convergence into RMSD convergence via `targetEnergyGap(μ, n, eps) = μ × n × eps² / 2`.

**Ingredient 2 — Linear energy convergence** (`CertifiedLinearEnergyConvergence`):
```
∀ t, energyGap(t) ≤ q^t × energyGap(0)
```
This is **optimizer-agnostic**. It's a property of the energy gap sequence, not a property of any particular optimizer.

### Two approaches: observed certification (A) vs certified-by-construction (B)

Both approaches share ingredient 2 (linear energy convergence) and differ in how they obtain ingredient 1 (quadratic basin) and q.

**Approach A: SE(3) Hessian + Jacobian bridge + observed q** — "measure, then certify"
- Use any optimizer (including the current clipped GD)
- Observe the energy trajectory to extract q empirically
- Compute a 6×6 Hessian in SE(3) parameter space
- Bridge to coordinate-space μ via a new Lean theorem (Jacobian bridge)
- Pro: no optimizer changes, cheap, practical
- Con: requires one new Lean theorem

**Approach B: Certified GD optimizer** — "certify by construction"
- Add a new optimizer mode that IS standard gradient descent in axis-angle parameterization
- Derive q analytically from the Hessian: `q = (lmax-lmin)/(lmax+lmin)`
- The Lean chain applies directly (optimizer matches theorem)
- Pro: fully theorem-derived, no empirical components
- Con: standard GD may converge slower than clipped GD; requires optimizer changes

Both are implemented behind a flag: `refinement_certification_mode: Literal["observed", "certified_gd"]`.

---

### Approach A: SE(3) Hessian + Jacobian bridge (observed certification)

#### Mathematical foundation

The optimizer works in parameter space P = R³ × R³ (translation + axis-angle rotation). The energy is:

```
E_param(t, θ) = E_coord(K(t, θ))
```

where K: P → CoordSet n is the rigid-body kinematics (`_apply_single_pose`). The Hessian H_param of E_param at the optimized point has eigenvalues `lmin_param > 0` (because there are no zero modes in P — every direction changes the energy).

The coordinate-space quadratic growth constant μ is related to the parameter-space Hessian by the kinematics Jacobian J = dK/d(t,θ):

```
μ_coord = lmin_param / σ_max(J)²
```

where σ_max(J) is the largest singular value of the Jacobian. This follows from:

```
ΔE ≥ (lmin_param / 2) × ||Δparams||²        (parameter-space quadratic growth)
||Δcoords||² ≤ σ_max(J)² × ||Δparams||²     (Jacobian bound)
→ ΔE ≥ (lmin_param / (2 × σ_max(J)²)) × ||Δcoords||²   (coordinate-space growth)
```

#### New Lean theorem needed: Jacobian bridge

```lean
/-- Parameter-space quadratic growth transfers to coordinate-space quadratic growth
    via the kinematics Jacobian's singular values. -/
theorem parameterSpace_quadraticBasin_transfers_to_coordSpace
    {n d : ℕ}
    (E_coord : CoordSet n → ℝ)
    (K : Fin d → ℝ → CoordSet n)  -- kinematics parameterization
    (center_param : Fin d → ℝ)
    (center_coord : CoordSet n)
    (μ_param : ℝ)
    (σ_max_sq : ℝ)
    (hμ : 0 < μ_param)
    (hσ : 0 < σ_max_sq)
    (h_center : K center_param = center_coord)  -- center maps correctly
    (h_param_basin : ∀ p, (μ_param / 2) * paramDisplacement p center_param
        ≤ E_coord (K p) - E_coord (K center_param))
    (h_jacobian : ∀ p, squaredDisplacement (K p) (K center_param)
        ≤ σ_max_sq * paramDisplacement p center_param) :
    CertifiedQuadraticBasin E_coord center_coord where
  μ := μ_param / σ_max_sq
  μ_pos := div_pos hμ hσ
  quadratic_growth := ...  -- follows from h_param_basin and h_jacobian
```

#### Lean certificate chain (Approach A)

| Lean structure | Python source | What it certifies |
|---------------|---------------|-------------------|
| (new) `parameterSpace_quadraticBasin_transfers_to_coordSpace` | 6×6 Hessian eigenvalues + Jacobian singular values | `μ_coord = lmin_param / σ_max(J)²` transfers to coordinate space |
| `CertifiedQuadraticBasin` | Derived via Jacobian bridge | `(μ/2) × squaredDisplacement ≤ ΔE` in coordinate space |
| `CertifiedOneStepEnergyContraction` | Observed energy gaps from optimizer | `gap(t+1) ≤ q × gap(t)` with empirical q |
| `rmsd_target_of_canonicalIterationBudgetFromLocalCertificates` | — | **The RMSD guarantee** |

#### Implementation (Approach A)

##### Step C-A1: Energy function in SE(3) parameter space

```python
def _make_se3_energy_fn(
    base_coords: jnp.ndarray,        # (N_lig, 3)
    receptor_coords: jnp.ndarray,    # (N_rec, 3)
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    cutoff: jnp.ndarray,
    epsilon: float,
) -> Callable[[jnp.ndarray], float]:
    """Energy as a function of 6D parameter vector [tx, ty, tz, θx, θy, θz].

    The first 3 components are translation. The last 3 are axis-angle rotation.
    This is the optimizer's actual parameter space — no constraints.
    """
    def energy_fn(params: jnp.ndarray) -> float:
        t = params[:3]
        rotvec = params[3:6]
        angle = jnp.linalg.norm(rotvec)
        axis = jnp.where(angle > 1e-8, rotvec / angle, jnp.array([0., 0., 1.]))
        q = _axis_angle_to_quaternion(axis, angle)
        pose_coords = _apply_single_pose(base_coords, t, q)
        energy, _ = _score_certified_lj(
            receptor_coords, pose_coords, receptor_radii, ligand_radii,
            cutoff, epsilon,
        )
        return energy
    return energy_fn
```

##### Step C-A2: Compute 6×6 Hessian + Jacobian singular values

```python
def compute_se3_spectral_certificate(
    energy_fn: Callable[[jnp.ndarray], float],
    kinematics_fn: Callable[[jnp.ndarray], jnp.ndarray],  # params → (N_lig, 3)
    optimized_params: jnp.ndarray,  # (6,) [t, rotvec]
) -> SE3SpectralCertificate | None:
    """Compute lmin_param, lmax_param from 6×6 Hessian, σ_max from Jacobian.

    Cost: 6 reverse-mode passes for Hessian + 1 Jacobian computation.
    For 50ms scoring: ~0.3s per pose.
    """
    H = jax.hessian(energy_fn)(optimized_params)       # (6, 6)
    eigs = jnp.linalg.eigvalsh(H)
    lmin_param = float(eigs[0])
    lmax_param = float(eigs[-1])

    if lmin_param <= 0:
        return None  # not in a convex basin

    J = jax.jacobian(kinematics_fn)(optimized_params)  # (N_lig, 3, 6)
    J_flat = J.reshape(-1, 6)                          # (3*N_lig, 6)
    sigma_max_sq = float(jnp.max(jnp.linalg.svdvals(J_flat)) ** 2)

    mu_coord = lmin_param / sigma_max_sq
    return SE3SpectralCertificate(
        lmin_param=lmin_param,
        lmax_param=lmax_param,
        sigma_max_sq=sigma_max_sq,
        mu_coord=mu_coord,
    )
```

##### Step C-A3: Extract observed q from energy trajectory

```python
def extract_observed_contraction_rate(
    energy_trajectory: list[float],
) -> float | None:
    """Fit q from observed energy gaps: gap(t+1) ≤ q × gap(t).

    Takes the worst-case ratio across all steps. If any step increases
    the energy gap, returns None (not monotonically converging).

    Lean: populates CertifiedOneStepEnergyContraction.step_contract.
    """
    final_energy = energy_trajectory[-1]
    gaps = [e - final_energy for e in energy_trajectory]

    q_max = 0.0
    for t in range(len(gaps) - 1):
        if gaps[t] <= 0:
            continue  # already converged
        ratio = gaps[t + 1] / gaps[t]
        if ratio >= 1.0:
            return None  # energy gap increased — cannot certify
        q_max = max(q_max, ratio)
    return q_max
```

##### Step C-A4: Derive certified iteration budget

```python
def certified_iteration_budget_observed(
    mu_coord: float,    # from Jacobian bridge
    q: float,           # from observed energy trajectory
    initial_gap: float, # E(pose_0) - E(pose_final)
    target_rmsd: float,
    n_atoms: int,
) -> int:
    """Lean: logarithmicIterationBound applied to observed q and bridged μ.

    Returns the provably sufficient iteration count for the RMSD target.
    """
    target_gap = mu_coord * n_atoms * target_rmsd**2 / 2.0
    if initial_gap <= target_gap:
        return 0
    import math
    return math.ceil(math.log(initial_gap / target_gap) / math.log(1.0 / q))
```

##### Step C-A5: Wire into pipeline (post-optimization)

```python
# After running the existing optimizer:
opt_t, opt_q = optimize_poses_batched(translations, quaternions, ..., n_steps=50)
opt_coords = apply_poses(ligand_ctx, PoseVector(opt_t, opt_q))

# For each surviving pose, certify the result:
for i in range(n_survivors):
    params_i = _pose_to_se3_params(opt_t[i], opt_q[i])
    cert = compute_se3_spectral_certificate(energy_fn, kinematics_fn, params_i)
    if cert is not None:
        q_obs = extract_observed_contraction_rate(energy_trajectories[i])
        if q_obs is not None:
            budget = certified_iteration_budget_observed(
                cert.mu_coord, q_obs, gaps[i], target_rmsd=0.5, n_atoms=n_lig)
            # budget tells us whether 50 steps was enough, or how many more needed
```

**Key requirement**: the optimizer must record the energy at each step. Currently `jax.lax.fori_loop` doesn't do this. This requires changing the loop to `jax.lax.scan` which returns intermediate states, or recording energies in a pre-allocated array via `.at[i].set(energy)`.

---

### Approach B: Certified gradient descent optimizer

#### Mathematical foundation

Replace the current clipped-normalized optimizer with standard gradient descent in axis-angle parameterization:

```
params_{t+1} = params_t - α × ∇E(params_t)
```

with `α = 2 / (lmin_param + lmax_param)` from the Hessian eigenvalues. The Lean theorem `CertifiedGradientDescentDynamics` proves contraction rate `q = (lmax - lmin) / (lmax + lmin)` for this exact dynamics.

Combined with the Jacobian bridge (shared with Approach A), this gives the full RMSD guarantee.

#### Lean certificate chain (Approach B)

| Lean structure | Python source | What it certifies |
|---------------|---------------|-------------------|
| (new) `parameterSpace_quadraticBasin_transfers_to_coordSpace` | Same as Approach A | `μ_coord = lmin_param / σ_max(J)²` |
| `CertifiedGradientStepParameters` | `α = 2/(lmin+lmax)`, `μ = lmin`, `M = lmax` | Valid step size |
| `CertifiedGradientDescentDynamics` | Standard GD matches theorem exactly | `gap(t+1) ≤ q × gap(t)` with `q = (M-μ)/(M+μ)` |
| `rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics` | — | **The RMSD guarantee** |

The key difference from Approach A: q is **derived from the theorem**, not observed. The optimizer IS the object the theorem proves about.

#### Implementation (Approach B)

##### Step C-B1: Standard GD optimizer in axis-angle space

```python
def _step_body_certified_gd(
    params: jnp.ndarray,    # (6,) [t, rotvec]
    alpha: float,
    base_coords: jnp.ndarray,
    receptor_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    cutoff: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """One step of standard gradient descent. No clipping, no normalization.

    This IS the optimizer that CertifiedGradientDescentDynamics proves about.
    Lean: step_contract follows from α = 2/(lmin+lmax) and smooth strong convexity.
    """
    grad = jax.grad(energy_fn)(params)
    return params - alpha * grad
```

##### Step C-B2: Two-phase optimization

```python
def optimize_certified_gd(
    initial_params: jnp.ndarray,  # (6,) [t, rotvec]
    energy_fn: Callable[[jnp.ndarray], float],
    kinematics_fn: Callable[[jnp.ndarray], jnp.ndarray],
    target_rmsd: float,
    n_atoms: int,
    max_probe_steps: int = 10,
) -> CertifiedGDResult:
    """Two-phase certified optimization.

    Phase 1: Run a few steps to reach near the basin (cheap probe).
    Phase 2: Compute Hessian, derive α and budget T, run T steps of standard GD.
    """
    # Phase 1: quick probe with small fixed step size
    probed = _run_gd_steps(initial_params, energy_fn, alpha=0.01, n_steps=max_probe_steps)

    # Compute Hessian at probed point
    cert = compute_se3_spectral_certificate(energy_fn, kinematics_fn, probed)
    if cert is None:
        return CertifiedGDResult(params=probed, certificate=None)

    # Phase 2: certified standard GD
    alpha = 2.0 / (cert.lmin_param + cert.lmax_param)
    q = (cert.lmax_param - cert.lmin_param) / (cert.lmax_param + cert.lmin_param)
    initial_gap = float(energy_fn(initial_params) - energy_fn(probed))
    budget = certified_iteration_budget_observed(
        cert.mu_coord, q, initial_gap, target_rmsd, n_atoms)

    optimized = _run_gd_steps(initial_params, energy_fn, alpha=alpha, n_steps=budget)
    return CertifiedGDResult(params=optimized, certificate=cert, q=q, n_steps=budget)
```

---

### Shared components

#### Certificate dataclasses

```python
@dataclass(frozen=True)
class SE3SpectralCertificate:
    """Hessian eigenvalues in SE(3) parameter space + Jacobian bridge.
    Lean: parameterSpace_quadraticBasin_transfers_to_coordSpace."""
    lmin_param: float   # smallest Hessian eigenvalue in parameter space
    lmax_param: float   # largest Hessian eigenvalue in parameter space
    sigma_max_sq: float # squared largest singular value of kinematics Jacobian
    mu_coord: float     # coordinate-space quadratic growth: lmin_param / sigma_max_sq

@dataclass(frozen=True)
class RefinementCertificate:
    """Combined certificate for theorem-backed n_opt_steps."""
    spectral: SE3SpectralCertificate
    q: float            # contraction rate (observed for A, derived for B)
    initial_gap: float
    target_rmsd: float
    n_steps: int        # certified budget
    mode: str           # "observed" or "certified_gd"
```

#### Flag in `PipelineDockingRequest`

```python
refinement_certification_mode: Literal["observed", "certified_gd", "none"] = "none"
```

- `"none"`: current behavior, fixed `n_opt_steps=50`
- `"observed"`: Approach A — run existing optimizer, certify post-hoc
- `"certified_gd"`: Approach B — run standard GD with theorem-derived budget

#### Axis-angle ↔ quaternion conversion

Already exists in codebase:
- `_axis_angle_to_quaternion` in `pocket_sampling.py:187` and `formal_actions.py:88`
- `_rodrigues_rotate` in `conformer_search.py:503`

Need to add: `_quaternion_to_axis_angle` for converting existing optimizer output to axis-angle params.

### Cost comparison

| | Approach A (observed) | Approach B (certified GD) |
|-|----------------------|--------------------------|
| Hessian | 6×6 → ~0.3s/pose | 6×6 → ~0.3s/pose |
| Jacobian | 1 computation → ~0.05s/pose | Same |
| Optimizer | Existing (fast, clipped) | Standard GD (possibly slower) |
| Energy recording | Requires `scan` instead of `fori_loop` | Not needed (q is derived) |
| New Lean | Jacobian bridge theorem | Same + nothing extra |
| Gap | q is empirical (observed) | q is provably correct |

### Research value

This is a valid research direction because:

1. **Novel**: No docking software provides provable RMSD convergence bounds. Both approaches are publishable.
2. **Comparable**: A vs B directly tests whether the clipped optimizer converges faster than the theoretically optimal standard GD. If A consistently needs fewer steps, it demonstrates that the heuristic optimizer is better in practice (common belief, rarely proven). If B is competitive, it demonstrates that theorem-backed optimization doesn't sacrifice performance.
3. **The Jacobian bridge theorem is independently useful**: it connects parameter-space optimization theory to coordinate-space structural biology metrics (RMSD). This applies beyond docking — to any rigid-body or articulated optimization problem.
4. **Falsifiable**: if the observed q from Approach A is consistently worse than the derived q from Approach B, it means the clipped optimizer is actually slower — a concrete finding.

---

## Problem 4: Seed-budget concretization

### What "adequate seed budget" means

The Lean theorem `minimal_adequate_seedBudget_optimal` says: there exists a smallest number of initial conformer seeds such that the canonical retain set after pruning contains the true top-k. The runtime question is: how many seeds do we need?

### Lean definition

`AdequateSeedBudget` in `BlindConformerPipelineRefinements.lean`:

```
AdequateSeedBudget := n_seeds such that
  energyTopK ⊆ canonicalRetain(lowerBound, tau) when |seeds| ≥ n_seeds
```

### Runtime instantiation

The number of seeds needed depends on:
1. The dimensionality of torsion space (n_bonds)
2. The Lipschitz constant of the energy landscape
3. The cell radius at which the B&B terminates (min_cell_radius)
4. The pruning threshold tau

A practical heuristic backed by the `canonicalSeedBudgetCost_mono` theorem (monotone cost in seed count):

```python
def minimal_seed_count(
    n_bonds: int,
    min_cell_radius: float = 0.05,  # radians
) -> int:
    """Minimum seeds to cover torsion space at the given resolution.

    Each seed covers a hypercube of side 2×min_cell_radius in each dimension.
    Total volume = (2π)^n_bonds. Each seed covers (2×min_cell_radius)^n_bonds.
    """
    import math
    coverage_per_seed = (2.0 * min_cell_radius) ** n_bonds
    total_volume = (2.0 * math.pi) ** n_bonds
    return max(1, math.ceil(total_volume / coverage_per_seed))
```

For n_bonds=5, min_cell_radius=0.05: `(2π/0.1)^5 ≈ 2.9e8` — far too many. This confirms that exhaustive coverage is infeasible and the B&B's Lipschitz pruning is essential.

**Practical approach**: The B&B already handles this implicitly via `max_cells`. The seed budget is effectively `max_cells` divided by the branching factor. The Lean theorem tells us the canonical budget is optimal — so the runtime should:

1. Run B&B with current `max_cells=200`
2. If the canonical retain set is too large (poor pruning), increase `max_cells`
3. Stop when the retain set stabilizes

This is the `canonicalSeedBudgetCost_mono` theorem: larger budgets never increase total cost (monotone).

### Implementation

#### Step D1: Make `max_cells` adaptive

Must be computed in `_build_conformer_search_config()` (pipeline.py:1572), which already has `rotatable_bonds`. NOT in `search_conformers()`, where the default `BranchAndBoundConfig()` is constructed before `n_bonds` is known (conformer_search.py:908-922).

```python
# In _build_conformer_search_config():
n_bonds = len(rotatable_bonds)
max_cells = 200 * min(2 ** n_bonds, 1024)

return BranchAndBoundConfig(
    max_cells=max_cells,
    score_lipschitz_constant=score_lipschitz_constant,
    per_bond_lipschitz=per_bond_lipschitz,
    reuse_initial_conformer=request.reuse_initial_conformer,
)
```

#### Step D2: Add early-termination when retain set stabilizes

Currently the B&B loop (conformer_search.py:693-811) only checks `cells_evaluated < config.max_cells` and deduplicates post-hoc. Change to:

1. Deduplicate **during** the loop (move dedup logic from post-hoc into the cell evaluation)
2. Track `steps_since_last_new_conformer`
3. Terminate early when `steps_since_last_new_conformer > K` (e.g., K = 50)

The Lean monotonicity theorem (`canonicalSeedBudgetCost_mono`) guarantees this doesn't miss the optimum if the budget was already adequate.

---

## Problem 5: Eliminate heuristic `top_k_to_optimize`

### Current state

Two separate parameters control pose selection:

| Parameter | Default | Role | Derived? |
|-----------|---------|------|----------|
| `top_k` | 10 | How many final results to return | **No — problem specification** (user intent, not a heuristic) |
| `top_k_to_optimize` | 200 | How many poses to send through optimization | **No — pure heuristic** |

Currently `_certified_pruning_pass` uses `k = min(request.top_k_to_optimize, n_poses)` in `_canonical_retain_mask`. This means:

```
tau = 200th_best_score + delta
```

With k=200, the threshold is far too loose — it's the 200th best score, not the 10th. Combined with inflated delta, nearly everything survives.

### What the Lean theorems prove

Three theorems form a complete chain eliminating the need for `top_k_to_optimize`:

**1. `canonicalRetain_certifiedSafe`** (`BlindConformerPipelineOptimality.lean`):
Using `k = top_k` in the canonical retain guarantees the true top-k survive:

```
tau = kth_best_coarse_score + delta   (where k = top_k, e.g. 10)
retain = {p : lower_bound(p) ≤ tau}
⟹ energyTopK ⊆ retain
```

**2. `canonicalRetain_minimizes_pipelineCost`** (`BlindConformerPipelineOptimality.lean`):
This retain set is **cost-optimal** — any other certified-safe retain set has equal or greater downstream cost.

**3. `energyTopK_subset_canonicalRetain`** (`BlindConformerPipelineRefinements.lean`):
The energy-top-k poses are contained in the canonical retain set. This justifies why the resulting set is "safe" — we never lose a pose that would have been in the final top-k.

### Why `top_k_to_optimize` is redundant

It was introduced before the canonical retain theorems existed. The old pruning path used `top_k_to_optimize` as the k in the pruning threshold. With canonical retain, the retain set size is **emergent from the physics**: it's however many poses fall within `delta` of the k-th best score. No separate parameter needed.

Small delta (from Problem 1's 3-tier architecture, ~1.5 kcal/mol) → tight threshold → small retain set. Large delta → loose threshold → large retain set. The physics controls the set size, not a heuristic constant.

### Why no resource cap is needed (or honest)

The previous plan had a `MAX_SURVIVORS = 500` heuristic cap. This is **not theorem-honest**: `energyTopK ⊆ canonicalRetain` does NOT imply `energyTopK ⊆ best_500_of(canonicalRetain)`. Capping the retain set silently breaks the certified safety guarantee.

With the analytic delta from Problem 1 (~1.5 kcal/mol, batch-size independent), the canonical retain set should be small. If it isn't, that means delta is wrong — which should be fixed at the source (Problem 1), not papered over with a cap.

If resource constraints are genuinely binding (e.g., GPU memory), the correct theorem-honest response is to increase `top_k` (which the user controls) or to improve delta. Not to silently truncate the certified set.

### Implementation

#### Step E1: Use `top_k` as k in canonical retain

In `_certified_pruning_pass`, change:

```python
# Before:
k = min(request.top_k_to_optimize, poses_coords.shape[0])

# After:
k = min(request.top_k, poses_coords.shape[0])
```

This tightens the threshold from "200th best + delta" to "10th best + delta". The Lean justification is `canonicalRetain_certifiedSafe`: with `k = top_k`, the retain set is both safe and cost-optimal.

#### Step E2: Remove `top_k_to_optimize` from `PipelineDockingRequest`

Delete the `top_k_to_optimize` field. **Not all references are mechanical replacements** — three have different semantics:

| File:Line | Current use | Replacement | Notes |
|-----------|------------|-------------|-------|
| `pipeline.py:187` | Default value definition | Delete field | — |
| `pipeline.py:1464-1467` | `k = min(top_k_to_optimize, ...)` for pruning threshold | `k = min(request.top_k, ...)` | Also update validation at line 1464 from `top_k_to_optimize > 0` to `top_k > 0` |
| `pipeline.py:1631-1637` | `_survivor_capacity()`: `top_k_to_optimize * 256` for **memory allocation** | Use `BLIND_CONFORMER_SURVIVOR_BATCH_SIZE` (8192) directly | This is NOT a pruning threshold — it sizes JAX arrays. With emergent retain set, the survivor batch is bounded by `BLIND_CONFORMER_SURVIVOR_BATCH_SIZE` regardless. |
| `pipeline.py:2054` | `n_to_opt = min(top_k_to_optimize, n_poses)` for **optimizer input selection** | Optimize all canonical retain survivors | The canonical retain set IS the optimization set. Its size is emergent from delta, not a separate parameter. |
| `pipeline.py:2255` | Exact survivor mask `k = min(top_k_to_optimize, ...)` | `k = min(request.top_k, ...)` | Same as E1 — mechanical replacement |
| `pipeline.py:126` | Comment referencing `top_k_to_optimize > 1` | Update comment | — |
| `benchmark_pdb.py` | ~14 references including CLI `--top_k_to_optimize` | **Deprecation**: accept flag, warn, ignore value | Breaking change — existing scripts use this flag |
| test files | ~15 references (helpers, assertions) | Update to use `top_k` | — |

#### Step E3: Safety floor (theorem-honest)

After canonical retain, if the retain set is smaller than `top_k` — which cannot happen if delta is non-negative and the population has at least k poses — assert rather than silently fix:

```python
n_survivors = int(jnp.sum(survivor_mask))
assert n_survivors >= k, (
    f"Canonical retain returned {n_survivors} < k={k} survivors. "
    f"This should be impossible with non-negative delta={delta}."
)
```

This is theorem-honest: `canonicalRetain_certifiedSafe` guarantees `|retain| ≥ k` when the input has at least k elements and delta ≥ 0. A violation means a bug, not a situation to silently recover from.

#### Step E4: No cap

No `MAX_SURVIVORS` cap. The retain set size is the retain set size. The theorems guarantee it contains the true top-k. Truncating it would void that guarantee.

---

## Execution order

| Priority | Task | Depends on | Estimated complexity |
|----------|------|------------|---------------------|
| **1** | A1-A5: Replace batch-max delta with analytic 3-tier bounds | Nothing | Medium — drop softening error, add `analytic_cutoff_tail_bound()` to 5 spec classes, add `analytic_total_delta()`, change `_score_softened_pose_batch()` delta source |
| **2** | E1-E4: Replace `top_k_to_optimize` with theorem-derived retain sizing | A (needs stable delta to produce reasonable retain set sizes) | Small — change k source, remove parameter, assert safety invariant |
| **3** | B1-B2: Pose-specific torsion budgets | A (need stable delta first) | Small — one new function + change two lines in pruning pass |
| **4** | D1-D2: Adaptive seed budget | Nothing | Small — parameter scaling + early termination |
| **5a** | C-A1–A5: Observed certification (Approach A) | Nothing | Large — SE(3) energy fn, 6×6 Hessian, Jacobian bridge, energy recording via `scan`, Lean theorem |
| **5b** | C-B1–B2: Certified GD optimizer (Approach B) | 5a (shares Hessian + Jacobian bridge) | Medium — standard GD step function, two-phase optimization, flag wiring |

---

## Success criteria

| Criterion | Metric | Target |
|-----------|--------|--------|
| Delta stability | `delta` value on 50k-pose blind run | < 5 kcal/mol (currently 10+) |
| Pruning effectiveness | Survivor fraction after canonical retain | < 10% of population (currently ~100%) |
| Native-like survival | CASF-2007 top-1 RMSD after pruning | < 3.0 Å on 60%+ targets |
| Refinement efficiency | Unnecessary optimization steps saved | > 30% reduction vs. fixed n_opt_steps=50 |
| Batch-size independence | Delta on 1k vs 500k batch | Within 10% of each other |
| top_k_to_optimize eliminated | Parameter removed from request | k=top_k via `canonicalRetain_certifiedSafe`, no heuristic cap |
| n_opt_steps derived (A) | Iteration budget from observed q + Jacobian bridge | Certified RMSD ≤ 0.5 Å on 80%+ poses with convex basin |
| n_opt_steps derived (B) | Iteration budget from certified GD dynamics | Same RMSD guarantee, compare step count to Approach A |

---

## JAX / performance constraints

- All `analytic_cutoff_tail_bound()` methods must use concrete Python floats, computed once at spec construction — never inside JIT
- `_posewise_active_torsion_mask()` must be JIT-compatible (pure jnp, no Python branching on traced values)
- 6×6 Hessian in SE(3) parameter space via `jax.hessian` — done post-optimization, outside the scoring JIT — O(6) reverse-mode passes ≈ 0.3s/pose
- Approach A requires `jax.lax.scan` instead of `fori_loop` in the optimizer to record energy trajectory — JIT-compatible but changes the loop structure
- Coarse scoring is unchanged (full rich chemistry) — only the delta computation changes, from batch-max to analytic
- `analytic_total_delta()` is computed once per ligand, cached on the plan object
