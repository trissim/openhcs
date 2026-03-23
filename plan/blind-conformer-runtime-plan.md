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
| `EnergyRMSDConvergence.lean` | `CertifiedQuadraticBasin`, `CertifiedLocalSpectralEnclosure`, `CertifiedGradientDescentDynamics`, `rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics`, `leastAdequateIterationBudget_spec` |

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

### Strategy: measure, then certify

The previous plan tried to derive `lmin`/`lmax` analytically and gave up, proposing heuristic alternatives. The correct approach: **compute the exact Hessian with JAX, extract eigenvalues, then feed them into the Lean certificate chain.** No heuristics needed.

The Lean infrastructure already exists — it's a post-hoc verification layer, not a derivation engine:

1. `jax.hessian` computes the exact Hessian at the optimized pose
2. `jnp.linalg.eigvalsh` extracts eigenvalues → `lmin`, `lmax`
3. These populate `CertifiedLocalSpectralEnclosure` (Lean structure at `EnergyRMSDConvergence.lean:149`)
4. `CertifiedGradientDescentDynamics` wraps the step contraction with `q = (lmax-lmin)/(lmax+lmin)`
5. `rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics` certifies the RMSD guarantee

**This is not an axiom.** We are not assuming `μ`; we are computing it for a specific pose and using Lean to verify that *if the computation is correct*, the RMSD guarantee holds.

### Lean certificate chain

| Lean structure | Python source | What it certifies |
|---------------|---------------|-------------------|
| `CertifiedLocalSpectralEnclosure` | `jax.hessian` eigenvalues at optimized pose | `lmin × ‖x-c‖² ≤ d²E/dt² ≤ lmax × ‖x-c‖²` along all rays |
| `CertifiedGradientStepParameters` | `α = 2/(lmin+lmax)`, `q = (lmax-lmin)/(lmax+lmin)` | Valid step size and contraction factor |
| `CertifiedGradientDescentDynamics` | Observed energy gaps from optimizer | `gap(t+1) ≤ q × gap(t)` |
| `canonicalIterationBudgetFromGradientDescentDynamics` | — | Minimum `t` such that `rmsd(pose_t, center) ≤ eps` |
| `rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics` | — | **The RMSD guarantee** |

### Cost of exact Hessian

For N_lig ≈ 30 atoms → 90-dimensional parameter space → 90×90 Hessian.

`jax.hessian` uses forward-over-reverse mode: O(3N) reverse-mode passes, each O(scoring cost). For 50ms scoring: 90 × 50ms = **4.5s per pose**.

This is done **once per surviving pose after optimization**, not per scoring call. With ~10–50 survivors after Problem 1 pruning, total cost is 45–225s. Acceptable for a theorem-honest result.

### Implementation

#### Step C1: Compute exact Hessian post-optimization

```python
def compute_spectral_certificate(
    energy_fn: Callable[[jnp.ndarray], float],
    optimized_coords: jnp.ndarray,  # (N_lig, 3)
) -> tuple[float, float]:
    """Compute exact lmin, lmax from Hessian eigenvalues.

    Lean: populates CertifiedLocalSpectralEnclosure.
    """
    flat = optimized_coords.ravel()
    H = jax.hessian(lambda x: energy_fn(x.reshape(-1, 3)))(flat)
    eigenvalues = jnp.linalg.eigvalsh(H)
    lmin = float(eigenvalues[0])
    lmax = float(eigenvalues[-1])
    # Precondition check: CertifiedLocalSpectralEnclosure requires 0 < lmin
    if lmin <= 0:
        # Not in a convex basin — cannot certify. Fall back to fixed budget.
        return None
    return lmin, lmax
```

#### Step C2: Derive iteration budget from certificate

```python
def certified_iteration_budget(
    lmin: float,
    lmax: float,
    initial_gap: float,   # E(x0) - E(x_final)
    target_rmsd: float,
    n_atoms: int,
) -> int:
    """Lean: canonicalIterationBudgetFromGradientDescentDynamics.

    Returns the provably minimal iteration count for the RMSD target.
    """
    q = (lmax - lmin) / (lmax + lmin)
    target_gap = lmin * n_atoms * target_rmsd**2 / 2.0  # targetEnergyGap

    if initial_gap <= target_gap:
        return 0

    # Lean: logarithmicIterationBound ≥ canonicalAdequateIterationBudget
    import math
    return math.ceil(math.log(initial_gap / target_gap) / math.log(1.0 / q))
```

#### Step C3: Wire into optimizer loop

```python
# In pipeline.py — after initial optimization:
spectral = compute_spectral_certificate(score_fn, optimized_coords)
if spectral is not None:
    lmin, lmax = spectral
    initial_gap = float(initial_energy - optimized_energy)
    n_steps = certified_iteration_budget(
        lmin, lmax, initial_gap, target_rmsd=0.5, n_atoms=n_lig)
else:
    n_steps = 50  # fallback only when not in a convex basin
```

#### Step C4: Certificate dataclasses

```python
@dataclass(frozen=True)
class SpectralCertificate:
    """Runtime mirror of CertifiedLocalSpectralEnclosure."""
    lmin: float
    lmax: float

@dataclass(frozen=True)
class RefinementCertificate:
    """Combined certificate for theorem-backed n_opt_steps.
    Lean: canonicalIterationBudgetFromGradientDescentDynamics."""
    spectral: SpectralCertificate
    q: float            # contraction rate
    initial_gap: float
    target_rmsd: float
    n_steps: int        # certified budget
```

**No `@conditionally_certified` needed.** The Hessian eigenvalues are exact (not approximate), and the Lean chain from `CertifiedLocalSpectralEnclosure` through `rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics` is fully proven.

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

```python
def adaptive_max_cells(
    n_bonds: int,
    base_max_cells: int = 200,
) -> int:
    """Scale B&B budget with torsion dimensionality."""
    # Each bisection doubles cells; need ~2^n_bonds to explore one level
    return base_max_cells * min(2 ** n_bonds, 1024)
```

#### Step D2: Add early-termination when retain set stabilizes

In the B&B loop, track the number of leaf conformers found. If no new distinct conformers are found after K consecutive batches, terminate early (the Lean monotonicity theorem guarantees this doesn't miss the optimum if the budget is adequate).

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

Delete the `top_k_to_optimize` field and all references:

| File | References to remove |
|------|---------------------|
| `pipeline.py:187` | Default value definition |
| `pipeline.py:1464-1467` | `k = min(request.top_k_to_optimize, ...)` |
| `pipeline.py:1631` | `_survivor_capacity()` reference |
| `pipeline.py:2054` | `n_to_opt = min(request.top_k_to_optimize, ...)` |
| `pipeline.py:2255` | Exact survivor mask reference |
| `benchmark_pdb.py` | ~12 references |
| test files | ~15 references |

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
| **5** | C1-C4: Formulaic refinement budget | Nothing | Large — Hessian estimation, new certificate types, optimizer integration |

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
| n_opt_steps derived | Iteration budget from certificate | Within 2× of converged value on 80%+ targets |

---

## JAX / performance constraints

- All `analytic_cutoff_tail_bound()` methods must use concrete Python floats, computed once at spec construction — never inside JIT
- `_posewise_active_torsion_mask()` must be JIT-compatible (pure jnp, no Python branching on traced values)
- Exact Hessian via `jax.hessian` is done post-optimization, outside the scoring JIT — O(3N) reverse-mode passes
- Coarse scoring is unchanged (full rich chemistry) — only the delta computation changes, from batch-max to analytic
- `analytic_total_delta()` is computed once per ligand, cached on the plan object
