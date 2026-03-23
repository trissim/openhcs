# Lean Energy/RMSD Runtime Plan

This note records the new Lean proof surface for theorem-backed docking refinement and
the exact runtime artifacts that must mirror it so `n_opt_steps` becomes formulaic
rather than heuristic.

## Goal

Given a target heavy-atom RMSD tolerance `eps_rmsd`, compute a certified refinement
budget directly from local curvature and optimizer contraction data:

- certify a local quadratic basin around the target pose,
- certify a linear energy contraction factor for the optimizer,
- derive a closed-form sufficient step count,
- optionally compare it to the canonical minimal certified budget.

The runtime should expose certificates matching the Lean structures below, then use
the closed-form logarithmic bound as the default `n_opt_steps`.

## New Lean proof objects

Primary file:

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/EnergyRMSDConvergence.lean`

Handle aliases:

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/HandleAliases.lean`

### Geometry / energy bridge

- `squaredDisplacement`
- `rmsd`
- `targetEnergyGap`
- `rmsd_le_of_quadratic_growth`
- `rmsd_le_of_energyGap_le_target`
- `energyGap_le_of_rmsd_le`

Meaning:

- a certified quadratic basin turns an energy-gap certificate into an RMSD certificate,
- a target RMSD induces a target energy gap `mu * N * eps^2 / 2`.

### Basin and curvature certificates

- `CertifiedQuadraticBasin`
- `CertifiedQuadraticWindow`
- `CertifiedSegmentCurvature`
- `CertifiedRayleighHessianBounds`
- `CertifiedLocalSpectralEnclosure`

Bridges proved:

- `CertifiedSegmentCurvature.toCertifiedQuadraticWindow`
- `CertifiedRayleighHessianBounds.toCertifiedSegmentCurvature`
- `CertifiedRayleighHessianBounds.toCertifiedQuadraticWindow`
- `CertifiedRayleighHessianBounds.toCertifiedQuadraticBasin`
- `CertifiedLocalSpectralEnclosure.toCertifiedRayleighHessianBounds`
- `CertifiedLocalSpectralEnclosure.toCertifiedSegmentCurvature`
- `CertifiedLocalSpectralEnclosure.toCertifiedQuadraticWindow`
- `CertifiedLocalSpectralEnclosure.toCertifiedQuadraticBasin`

Meaning:

- runtime can start from the most concrete object, a local spectral enclosure,
- then Lean lifts it all the way to the quadratic basin/window used by RMSD proofs.

### Optimizer contraction certificates

- `CertifiedGradientStepParameters`
- `CertifiedOneStepEnergyContraction`
- `CertifiedLinearEnergyConvergence`
- `CertifiedGradientDescentDynamics`

Bridges proved:

- `CertifiedOneStepEnergyContraction.toCertifiedLinearEnergyConvergence`
- `CertifiedGradientDescentDynamics.toCertifiedOneStepEnergyContraction`
- `CertifiedGradientDescentDynamics.toCertifiedLinearEnergyConvergence`

Meaning:

- runtime does not need to inject a full geometric-rate proof directly,
- it can certify per-step contraction and let Lean derive the global rate.

### Canonical and formulaic step budgets

- `AdequateIterationBudget`
- `leastAdequateIterationBudget`
- `canonicalAdequateIterationBudget`
- `logarithmicIterationBound`

Budget theorems:

- `adequateIterationBudget_mono`
- `adequateIterationBudget_mono_eps`
- `minimal_adequate_iterationBudget_optimal`
- `exists_adequateIterationBudget`
- `leastAdequateIterationBudget_spec`
- `leastAdequateIterationBudget_minimal`
- `leastAdequateIterationBudget_optimal`
- `canonicalAdequateIterationBudget_spec`
- `canonicalAdequateIterationBudget_optimal`
- `adequateIterationBudget_of_logarithmicIterationBound`
- `canonicalAdequateIterationBudget_le_logarithmicIterationBound`

RMSD consequences:

- `rmsd_target_of_adequateIterationBudget`
- `rmsd_target_of_leastAdequateIterationBudget`
- `rmsd_target_of_canonicalAdequateIterationBudget`
- `rmsd_target_of_logarithmicIterationBound`

Zero-step edge case:

- `adequateIterationBudget_zero_of_initialGap_le_target`
- `rmsd_target_of_zeroIterationBudget`
- `leastAdequateIterationBudget_eq_zero_of_zero_adequate`
- `canonicalAdequateIterationBudget_eq_zero_of_initialGap_le_target`

Meaning:

- Lean now proves a closed-form sufficient step count,
- and also proves the canonical minimal certified budget is no larger than that bound.

### GD-specific wrappers

- `canonicalIterationBudgetFromGradientDescentDynamics`
- `rmsd_target_of_canonicalIterationBudgetFromGradientDescentDynamics`
- `logarithmicIterationBoundFromGradientDescentDynamics`
- `adequateIterationBudget_of_logarithmicIterationBoundFromGradientDescentDynamics`
- `rmsd_target_of_logarithmicIterationBoundFromGradientDescentDynamics`

Meaning:

- once runtime can emit a `CertifiedGradientDescentDynamics`-shaped object,
- `n_opt_steps` can be computed directly from the logarithmic formula.

## Runtime structures to mirror

The Python/JAX runtime should mirror the following proof objects as serializable or
plain dataclass-like certificate records.

### 1. Local spectral enclosure

Mirror of `CertifiedLocalSpectralEnclosure`.

Required runtime fields:

- `lmin`: certified lower eigenvalue bound in the local basin
- `lmax`: certified upper eigenvalue bound in the local basin
- `rayleigh_eval(x, t)`: scalar second-direction value along the segment from `center` to `x`
- evidence that `rayleigh_eval(x, t)` matches the segment second derivative
- evidence that `lmin * ||x-center||^2 <= rayleigh_eval(x,t) <= lmax * ||x-center||^2`
- smoothness/stationarity side conditions along `t in [0,1]`

Expected implementation source:

- Hessian-vector products from JAX/autodiff,
- Lanczos / power iteration / Gershgorin-style certified enclosures,
- interval or conservative floating-point envelopes on the sampled basin.

### 2. Gradient-descent dynamics certificate

Mirror of `CertifiedGradientDescentDynamics`.

Required runtime fields:

- certified step parameters `mu`, `M`, `alpha`
- derived `q = 1 - alpha * mu`
- proof-side assumptions mirrored by runtime checks:
  - `0 < mu <= M`
  - `0 < alpha <= 1 / M`
- a certified one-step inequality
  - `gap_{t+1} <= q * gap_t`

Expected implementation source:

- local descent lemma using the certified smoothness window,
- measured or bounded post-step energy drop,
- conservative contraction estimate if exact symbolic proof is unavailable.

### 3. Energy gap inputs

Runtime must compute:

- `gap0 = E(pose_0) - E(center)`
- `target_gap = mu * N * eps_rmsd^2 / 2`

where `N` is the atom count used by RMSD.

## Formula to implement for `n_opt_steps`

The theorem-backed sufficient budget is:

```text
n_opt_steps = ceil(log(gap0 / target_gap) / log(1 / q))
```

with:

- `target_gap = mu * N * eps_rmsd^2 / 2`
- `0 < q < 1`
- `gap0 > 0`
- `target_gap > 0`

If `gap0 <= target_gap`, Lean proves the certified budget is exactly `0`.

## Runtime algorithm outline

1. Choose the local target pose `center` used for refinement certification.
2. Build a local spectral enclosure around `center`.
3. Lift it conceptually to the quadratic basin/window (`mu`, `M`).
4. Build GD dynamics certificate data and derive `q`.
5. Compute:
   - `gap0`
   - `target_gap`
6. If `gap0 <= target_gap`, set `n_opt_steps = 0`.
7. Else set:
   - `n_opt_steps = ceil(log(gap0 / target_gap) / log(1 / q))`
8. Optionally compare with the canonical minimal certified budget if runtime search for
   the minimal adequate budget is retained for diagnostics.

## Recommended runtime exports

To keep implementation aligned with Lean, add Python records roughly matching:

- `LocalSpectralEnclosureCertificate`
- `RayleighHessianBoundsCertificate`
- `SegmentCurvatureCertificate`
- `QuadraticWindowCertificate`
- `GradientStepParametersCertificate`
- `GradientDescentDynamicsCertificate`
- `RefinementBudgetCertificate`

Each should carry enough numeric data to reconstruct the proof intent and produce
auditable logs.

## Suggested report/debug fields

For each refinement attempt, log:

- `mu`, `M`, `alpha`, `q`
- `atom_count`
- `eps_rmsd`
- `gap0`
- `target_gap`
- `n_opt_steps_formula`
- `n_opt_steps_used`
- `zero_step_certified`
- optional `canonical_budget_search_result`

## Practical interpretation

After these Lean additions, the remaining non-mechanical part is no longer theorem
shape. It is certificate construction:

- numerically certifying local curvature/spectral bounds,
- numerically certifying contraction for the chosen optimizer.

Once those runtime certificates are emitted conservatively, selecting `n_opt_steps`
should be a direct translation of the proved formula rather than a benchmark-tuned
guess.
