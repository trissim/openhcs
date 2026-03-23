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
| `BlindConformerRuntimeCertificates.lean` | `bounded_channel_uniformApprox_zero`, `bounded_channel_sum_uniformApprox_zero`, `base_plus_omitted_uniformApprox`, `exact_with_omitted_ge_coarse_minus_totalError`, `pose_specific_improvement_bound_of_active_subset`, `canonical_pruning_and_budget_optimal` |

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
| Analytic omitted-channel bounds | No code computes analytic per-channel uniform bounds; all use batch-max `jnp.max(abs(exact-coarse))` |
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

### Lean structures required at runtime

```python
@dataclass(frozen=True)
class QuadraticBasinCertificate:
    """Maps to CertifiedQuadraticBasin in EnergyRMSDConvergence.lean.

    Preconditions:
      - mu > 0
      - For all x: (mu/2) × ||x - center||² ≤ E(x) - E(center)
    """
    mu: float          # Strong convexity constant (smallest curvature)
    center_energy: float

@dataclass(frozen=True)
class SpectralEnclosureCertificate:
    """Maps to CertifiedLocalSpectralEnclosure.

    Preconditions:
      - 0 < lmin ≤ lmax
      - For all x on the segment from center to x:
        lmin × ||x-center||² ≤ d²E/dt² ≤ lmax × ||x-center||²
    """
    lmin: float        # Lower eigenvalue bound (strong convexity)
    lmax: float        # Upper eigenvalue bound (smoothness)

@dataclass(frozen=True)
class GradientDescentCertificate:
    """Maps to CertifiedGradientDescentDynamics.

    Preconditions:
      - initial_gap ≥ 0
      - For all t: gap(t+1) ≤ q × gap(t)
    """
    alpha: float       # Step size = 2 / (lmin + lmax)
    q: float           # Contraction rate = (lmax - lmin) / (lmax + lmin)
    initial_gap: float # E(x0) - E(x*)

@dataclass(frozen=True)
class RefinementBudgetCertificate:
    """Combined certificate for theorem-backed n_opt_steps."""
    basin: QuadraticBasinCertificate
    spectral: SpectralEnclosureCertificate
    dynamics: GradientDescentCertificate
    target_rmsd: float
    n_steps: int
```

### How to numerically estimate lmin and lmax

The spectral enclosure requires bounds on the Hessian eigenvalues along all geodesics from center. At runtime, we can estimate these from **finite-difference Hessian-vector products** at the optimized pose:

```python
def estimate_spectral_bounds(
    energy_fn: Callable,
    center_coords: jnp.ndarray,  # (N, 3) optimized pose
    n_probes: int = 20,
    perturbation_scale: float = 0.01,  # Å
) -> tuple[float, float]:
    """Estimate lmin, lmax via random Rayleigh quotient probing."""
    flat = center_coords.ravel()
    n = len(flat)
    hess_fn = jax.hessian(lambda x: energy_fn(x.reshape(-1, 3)))
    H = hess_fn(flat)  # (3N, 3N) — expensive but done once per pose
    eigenvalues = jnp.linalg.eigvalsh(H)
    lmin = float(jnp.max(jnp.array([eigenvalues[0], 1e-6])))  # floor at 1e-6
    lmax = float(eigenvalues[-1])
    return lmin, lmax
```

**Cost**: One Hessian computation per pose post-optimization. For N_lig ≈ 30 atoms → 90×90 Hessian → ~8100 entries. With JAX autodiff this is O(N²) forward passes ≈ 8100 energy evaluations. For a 50ms energy call, this is ~400s.

**This is too expensive for routine use.** Two alternatives:

#### Alternative C1: Hessian-free diagonal approximation (practical)

Use the diagonal of the Hessian (O(N) cost) as a proxy:

```python
def estimate_spectral_bounds_diagonal(
    energy_fn: Callable,
    center_coords: jnp.ndarray,
    h: float = 0.001,  # Å
) -> tuple[float, float]:
    """Estimate eigenvalue bounds from diagonal Hessian (finite differences)."""
    flat = center_coords.ravel()
    diag = []
    for i in range(len(flat)):
        e_plus = energy_fn((flat.at[i].set(flat[i] + h)).reshape(-1, 3))
        e_minus = energy_fn((flat.at[i].set(flat[i] - h)).reshape(-1, 3))
        e_center = energy_fn(flat.reshape(-1, 3))
        diag.append((e_plus + e_minus - 2 * e_center) / h**2)
    diag = jnp.array(diag)
    # Gershgorin-style: diagonal entries bound eigenvalues when off-diagonal is small
    lmin_est = float(jnp.min(diag))
    lmax_est = float(jnp.max(diag))
    return max(lmin_est, 1e-6), max(lmax_est, lmin_est)
```

Cost: 2N + 1 energy evaluations per pose ≈ 181 calls for N_lig=30 → ~9s at 50ms/call. Acceptable.

**Caveat**: Diagonal approximation is not a rigorous spectral enclosure — it's a heuristic upper/lower bound. To make it theorem-honest, we'd need to prove that the off-diagonal Hessian elements are bounded (which they are for pairwise potentials, but the proof isn't in Lean yet).

#### Alternative C2: Conservative analytic bounds from Lipschitz constant (rigorous but loose)

Use the score Lipschitz constant as a bound on lmax, and use the observed energy gap improvement rate as lmin:

```python
def conservative_spectral_bounds(
    lipschitz_constant: float,  # M from LipschitzStepBounds.lean
    energy_decrease_per_step: float,  # observed from optimizer
    step_displacement: float,  # ||x_{t+1} - x_t||
) -> tuple[float, float]:
    lmax = lipschitz_constant  # Lipschitz of gradient ≤ Hessian norm
    # From observed contraction: gap_decrease ≥ (1/(2*lmax)) × ||grad||²
    # Approximate lmin from observed convergence rate
    lmin = max(energy_decrease_per_step / (step_displacement**2 + 1e-10), 1e-3)
    return lmin, lmax
```

This is theorem-honest (lmax from Lipschitz is a valid upper bound) but conservative (lmin from observation is a lower bound only if convergence is monotone).

### Budget formula

Given `lmin`, `lmax`:

```python
def compute_refinement_budget(
    lmin: float,
    lmax: float,
    initial_gap: float,   # E(x0) - E(x*), estimated as E(x0) - E(x_final)
    target_rmsd: float,   # desired RMSD to optimum
    n_atoms: int,
) -> int:
    """Theorem-backed iteration budget from EnergyRMSDConvergence.lean."""
    alpha = 2.0 / (lmin + lmax)
    q = (lmax - lmin) / (lmax + lmin)  # condition number dependent
    target_gap = lmin * n_atoms * target_rmsd**2 / 2.0

    if initial_gap <= target_gap:
        return 0  # Already converged

    if q >= 1.0 or q <= 0.0:
        return 1000  # Fallback for degenerate cases

    # n = ceil(log(gap0 / target_gap) / log(1/q))
    import math
    return math.ceil(math.log(initial_gap / target_gap) / math.log(1.0 / q))
```

### Implementation plan

#### Step C1: Add certificate dataclasses to `dq_dock_engine/docking/scoring.py`

The four dataclasses above.

#### Step C2: Add diagonal Hessian estimation to pipeline

Call after optimization converges, before deciding whether to refine further.

#### Step C3: Wire into optimizer loop

In `pipeline.py`, replace the hardcoded `n_opt_steps=50` with:

```python
# After initial optimization run (e.g., 10 steps):
lmin, lmax = estimate_spectral_bounds_diagonal(score_fn, optimized_coords)
initial_gap = float(initial_energy - optimized_energy)
n_steps = compute_refinement_budget(lmin, lmax, initial_gap, target_rmsd=0.5, n_atoms=n_lig)
# Run remaining steps
remaining = max(0, n_steps - steps_already_done)
```

#### Step C4: Defer full spectral enclosure proof to later

The diagonal Hessian approach is practically sound but not fully Lean-certified. Mark with `@conditionally_certified` and note the assumption that diagonal approximation bounds the spectrum. A future Lean proof of off-diagonal Hessian bounds for pairwise LJ potentials would close this gap.

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

### What the Lean theorems say

`canonicalRetain_certifiedSafe` + `energyTopK_subset_canonicalRetain`: using `k = top_k` (the actual desired result count) in the canonical retain guarantees that the true top-k survive:

```
tau = kth_best_coarse_score + delta   (where k = top_k, e.g. 10)
retain = {p : lower_bound(p) ≤ tau}
```

The theorem `canonicalRetain_minimizes_pipelineCost` proves this is the **cost-optimal** retain set — any other certified-safe retain set has equal or greater cost.

The retain set size is **emergent**: it's however many poses fall below the theorem-derived threshold. No separate `top_k_to_optimize` parameter needed.

### Why `top_k_to_optimize` exists as a separate parameter

It was introduced before the canonical retain theorems existed. The old pruning path (non-conformer branch, lines 1510-1527) uses `coarse_top1_ambiguity_mask(coarse_scores, delta)` for k=1, which computes a 2δ-wide band around the best score. For k>1, it computes exact scores and uses `certified_pruning_certificate`. Both paths used `top_k_to_optimize` as the k.

With canonical retain now available, the separate parameter is redundant.

### Implementation

#### Step E1: Use `top_k` as k in canonical retain

In `_certified_pruning_pass`, change:

```python
# Before:
k = min(request.top_k_to_optimize, poses_coords.shape[0])

# After:
k = min(request.top_k, poses_coords.shape[0])
```

This tightens the threshold from "200th best + delta" to "10th best + delta".

#### Step E2: Remove `top_k_to_optimize` from `PipelineDockingRequest`

All callers that set `top_k_to_optimize` should use `top_k` instead. The canonical retain set size replaces the old fixed budget.

#### Step E3: Add safety floor for retain set size

After canonical retain, if the retain set is smaller than `top_k` (which shouldn't happen if delta is correct, but defensive), fall back:

```python
n_survivors = int(jnp.sum(survivor_mask))
if n_survivors < k:
    # Delta too small or numerical issue — fall back to top_k_to_optimize-style behavior
    survivor_mask = top_k_with_ties_mask(coarse_scores, k)
```

#### Step E4: Cap retain set for resource management

The canonical retain might be large if delta is still somewhat loose (post Item A). Add a cap:

```python
MAX_SURVIVORS = 500  # hard resource cap
if n_survivors > MAX_SURVIVORS:
    # Take the best MAX_SURVIVORS by coarse score, still theorem-honest
    # (subset of canonical retain is still safe by canonicalRetain_subset_of_certifiedSafe)
    sorted_indices = jnp.argsort(coarse_scores)
    survivor_mask = jnp.zeros_like(survivor_mask)
    survivor_mask = survivor_mask.at[sorted_indices[:MAX_SURVIVORS]].set(True)
```

Wait — this is NOT theorem-honest. Taking a subset of the canonical retain set is safe (won't miss the top-k) only if the subset still contains the top-k. The theorem says `energyTopK ⊆ canonicalRetain`, not that any subset of canonicalRetain contains energyTopK.

Correct approach: if the retain set exceeds the resource cap, keep the best MAX_SURVIVORS by coarse score. This is safe IF `MAX_SURVIVORS ≥ k + (number of poses within delta of kth)`. In practice, for small delta (post Item A) and reasonable k=10, this will hold for MAX_SURVIVORS=500.

---

## Execution order

| Priority | Task | Depends on | Estimated complexity |
|----------|------|------------|---------------------|
| **1** | A1-A4: Replace batch-max delta with analytic omitted-channel bounds | Nothing | Medium — add `uniform_bound()` to 5 spec classes, add `score_softened_lj_only_batch()`, change `_score_softened_pose_batch()` |
| **2** | E1-E4: Replace `top_k_to_optimize` with theorem-derived retain sizing | A (needs stable delta to produce reasonable retain set sizes) | Small — change k source, remove parameter, add safety floor/cap |
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
| top_k_to_optimize eliminated | Parameter removed from request | No heuristic k in pruning path |
| n_opt_steps derived | Iteration budget from certificate | Within 2× of converged value on 80%+ targets |

---

## JAX / performance constraints

- All `uniform_bound()` methods must use concrete Python floats, computed once at spec construction — never inside JIT
- `_posewise_active_torsion_mask()` must be JIT-compatible (pure jnp, no Python branching on traced values)
- Diagonal Hessian estimation is done post-optimization, outside the scoring JIT — O(2N+1) energy calls
- Coarse scoring (LJ-only) is cheaper than full rich chemistry scoring — pruning pass gets faster, not slower
- `analytic_omitted_channel_bound()` is computed once per ligand, cached on the plan object
