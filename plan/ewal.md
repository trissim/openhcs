## Long-Term Goal

Integrate certified electrostatics into the active formal docking runtime through the full chain:

Lean theorem -> `HandleAliases.lean` -> generated Python aliases -> JAX/Python scorer ->
formal optimizer / surrogate path.

The target outcome is not just a new scorer. The target outcome is:

1. the active certified docking path can use electrostatics during local refinement;
2. the theorem handles attached to that path match actual Lean results;
3. electrostatics-sensitive failures such as `1ajp` improve materially;
4. the fast certified inner loop stays tractable.


## Current Progress

Already landed in the repo:

- Lean: canonical trusted `erfc` symbol / bound now live in
  `EwaldSummation.lean`
- Lean: signed-charge envelope theorem for `exactRealEwaldScore` is added in
  `CoulombApproximation.lean`
- Lean: explicit coarse alpha-dependent `R^-3` domination theorem is added in
  `EwaldSummation.lean`
- Lean: generic utility-sum uniform-approx theorem is added in
  `CoarseApproximation.lean`
- Handles: `CB5`, `CB6`, and `APX4` are exported and generated into Python
- Python: a batched LJ + real-space Ewald scorer exists in
  `dq_dock_engine/docking/scoring.py`
- Python: the certified pipeline / formal round path can now accept the new
  scoring family through `CertifiedScoringFamily.LJ_REALSPACE_EWALD`
- Benchmark prep: receptor extraction now filters non-`A` alternate locations,
  matching competitor preparation and fixing RDKit charge alignment on `1ajp`

Current canary result:

- on a focused `1ajp` run using
  `CertifiedScoringFamily.LJ_REALSPACE_EWALD` + `ChargeMethod.GASTEIGER`
  + altloc-filtered receptor prep,
  DQ-Dock reached approximately `2.8 A` RMSD with `native_rank = 1`
  at a modest `128` poses / `10` formal refinement rounds.

Additional scorer-discrimination check:

- after applying the same altloc filtering rule to ligands as well as
  receptors, the altloc-filtered + Gasteiger composite scorer ranks the native
  pose better than the cached DQ pose on all five `exact_five` complexes
  (`1hk4`, `1gni`, `1nhu`, `2d3z`, `2d3u`) when rescored directly
- this means the immediate remaining risk is no longer "does the scorer ever
  discriminate in the right direction?" but rather "does the full docking run
  consistently find those better minima under realistic benchmark budgets?"


## Scope Decision

Phase 1 is deliberately narrower than "full Ewald":

- Add certified real-space electrostatics for receptor-ligand cross interactions.
- Keep reciprocal-space out of the active certified inner loop.
- Treat reciprocal-space as deferred work: either final rescoring or a later
  decision-irrelevance proof.

This is the only mechanically landable path against the current codebase.


## Current Implementation Anchors

These are the actual entry points the plan must fit.

### Runtime / scoring

- `dq_dock_engine/docking/scoring.py`
  - existing certified LJ single-pose kernel: `_score_certified_lj`
  - existing certified LJ batch kernel: `_score_certified_lj_batch`
  - existing certified public entrypoint: `score_certified_lj`
  - existing certified result wrapper: `CertifiedBatchResult`

- `dq_dock_engine/docking/core.py`
  - `ScoringEngine` currently includes only `CERTIFIED_LJ`
  - there is no separate certified score-family selector yet

- `dq_dock_engine/docking/pipeline.py`
  - certified mode currently hardwires `ScoringEngine.CERTIFIED_LJ`
  - `route_scoring(...)` call sites currently pass only coords/radii/poses
  - certified refinement does not use `route_scoring`; it uses the formal path

- `dq_dock_engine/docking/formal_surrogates.py`
  - active exact/coarse certified rounds call `score_certified_batch(...)`
  - this is the real integration point for the active formal runtime

- `dq_dock_engine/docking/charges.py`
  - charge assignment exists today
  - charge assignment is not part of the formal proof chain
  - for certified electrostatics, charges should be treated as explicit inputs

- `dq_dock_engine/docking/pdb_io.py`
  - `LigandContext` already supports `elements` and optional `charges`

### Existing electrostatics code

- `dq_dock_engine/physics/ewald.py`
  - current code is periodic same-system Ewald, not receptor-ligand docking
  - useful as math/kernel reference
  - not directly pluggable into the docking scorer

### Lean proof anchors

- `.../Tractability/EwaldSummation.lean`
  - contains `ewaldRealSpaceCore`
  - contains exponential decay theorem for the envelope

- `.../Tractability/CoulombApproximation.lean`
  - contains finite-domain Coulomb cutoff theorems `CB1` / `CB2`
  - contains packaging theorems for Coulomb / real-space Ewald bounded-potential
    assumptions `CB3` / `CB4`
  - currently duplicates a placeholder `erfc`

- `.../Tractability/CoarseApproximation.lean`
  - already has generic shared-reference approximation composition
  - does not yet have a utility-sum composition theorem


## What Must Be Solved First

These are the necessary problems, in dependency order.

### Problem A: Lean symbol hygiene for `erfc`

Current issue:

- `CoulombApproximation.lean` defines its own placeholder `erfc`
- `EwaldSummation.lean` contains the real-space envelope theory

Required fix:

- define the trusted `erfc` symbol exactly once in `EwaldSummation.lean`
- add the trusted bound axiom there
- import and reuse it from `CoulombApproximation.lean`

Required edit:

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/EwaldSummation.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CoulombApproximation.lean`

Planned theorem/axiom shape:

```lean
noncomputable def erfc (x : R) : R := 0

axiom erfc_abs_le_exp_neg_sq {x : R} (hx : 0 < x) :
  |erfc x| <= Real.exp (-(x ^ 2))
```

Notes:

- use `|erfc x|`, not `erfc x`
- do not add a second `erfc` in `CoulombApproximation.lean`


### Problem B: Real-space Ewald absolute bound with signed charges

Current issue:

- previous draft incorrectly assumed `0 <= q_i * q_j`
- actual electrostatics needs mixed-sign charge pairs

Required fix:

- prove the real-space score is bounded using `|q_i * q_j|`
- tie the exact score to the already-defined envelope `ewaldRealSpaceCore`

Planned theorem shape:

```lean
theorem abs_exactRealEwaldScore_le_charge_envelope
    (q_i q_j alpha r : R) (hr : 0 < r) (ha : 0 < alpha) :
    |exactRealEwaldScore q_i q_j alpha r| <=
      |q_i * q_j| * ewaldRealSpaceCore r alpha
```

This is the minimum bridge needed to connect the runtime scorer to the current
Lean Ewald envelope story.


### Problem C: Easy `R^-3` domination theorem with an alpha-dependent constant

Current issue:

- the old draft tried to prove a sharp constant using calculus on
  `x^3 * exp(-x^2)`
- that is unnecessary and makes the plan brittle
- the old draft also implicitly assumed `alpha * R >= 1`, which is false for
  practical `alpha = 0.2`

Mechanically safer plan:

- use a cruder but easy bound
- from `exp(t) >= t^2 / 2` for `t > 0`, get `exp(-t) <= 2 / t^2`
- substitute `t = (alpha * R)^2`

Then:

```text
exp(-(alpha*R)^2) / R <= 2 / (alpha^4 * R^5) <= 2 / (alpha^4 * R^3)   for R >= 1
```

This gives an explicit alpha-dependent coefficient with no derivative-heavy
proof and no fake `alpha * R >= 1` assumption.

Planned theorem shape:

```lean
theorem ewaldRealSpaceCore_le_alpha_tail
    (alpha R : R) (ha : 0 < alpha) (hR : 1 <= R) :
    ewaldRealSpaceCore R alpha <= (2 / alpha^4) / R^3
```

This is not meant to be tight. It is meant to be easy to prove and sufficient
to plug into the existing `R^-3` cutoff story.


### Problem D: Composite approximation theorem for LJ + electrostatics

Current issue:

- the repo has LJ theorems and Coulomb/Ewald theorems separately
- it does not yet have an explicit theorem saying the sum of two uniformly
  approximated utilities is uniformly approximated by the sum of the radii

Required fix:

- add a small generic theorem in a new file or in `CoarseApproximation.lean`

Planned theorem shape:

```lean
theorem sum_uniformApprox
    (exact1 coarse1 exact2 coarse2 : DecisionProblem A S)
    (d1 d2 : R)
    (h1 : UniformUtilityApprox exact1 coarse1 d1)
    (h2 : UniformUtilityApprox exact2 coarse2 d2) :
    UniformUtilityApprox
      (sumDecisionProblems exact1 exact2)
      (sumDecisionProblems coarse1 coarse2)
      (d1 + d2)
```

Use this to combine:

- exact LJ vs cutoff LJ
- exact real-space electrostatics vs cutoff real-space electrostatics

This avoids inventing a giant one-off composite proof file first.


## Phase 1: Mechanically Landable Deliverable

Phase 1 deliverable:

- a new batched certified scorer for LJ + real-space electrostatics
- available from Python
- backed by Lean theorem handles
- usable in initial / final scoring paths
- not yet wired into the active formal optimizer rounds

This is the right first landing point because it isolates proof and runtime
issues before touching the formal round machinery.


## Exact Phase 1 Code Changes

### Lean files

1. `.../Tractability/EwaldSummation.lean`
   - add canonical `erfc`
   - add trusted `erfc_abs_le_exp_neg_sq`

2. `.../Tractability/CoulombApproximation.lean`
   - remove local duplicate `erfc`
   - define / reuse `exactRealEwaldScore`
   - add signed-charge absolute bound theorem
   - add easy alpha-dependent `R^-3` domination theorem

3. `.../Tractability/CoarseApproximation.lean`
   - add utility-sum composition theorem

4. `.../HandleAliases.lean`
   - export the new real-space bridge and composite theorems


### Python runtime files

1. `dq_dock_engine/docking/scoring.py`
   - add single-pose kernel mirroring `_score_certified_lj`
   - add batched kernel mirroring `_score_certified_lj_batch`
   - no Python loops over poses
   - safe fallback distances must use cutoffs, not tiny constants

2. `dq_dock_engine/docking/core.py`
   - add a score-family selector only if needed
   - do not assume adding a new `ScoringEngine` alone is enough for certified mode

3. `dq_dock_engine/docking/pipeline.py`
   - plumb `receptor_elements`, `ligand_ctx.elements`, and charges into
     non-formal scoring calls
   - fail loudly if composite certified electrostatics is requested without
     charges or a charge assignment path

4. `dq_dock_engine/generated/formal_handle_aliases.py`
   - regenerate after Lean handle additions


## Phase 1 Runtime Shape Requirements

The new scorer must match the existing scorer interface conventions.

### Single-pose kernel

Match `_score_certified_lj(...)`:

```python
@jax.jit
def _score_certified_lj_realspace_electrostatics(
    receptor_coords,
    pose_coords,
    receptor_radii,
    ligand_radii,
    receptor_charges,
    ligand_charges,
    lj_cutoff,
    electrostatic_cutoff,
    alpha,
    epsilon,
):
    ...
```

Input rank for `pose_coords` is `(N_lig, 3)`, not batched.

### Batched kernel

Match `_score_certified_lj_batch(...)`:

```python
@jax.jit
def _score_certified_lj_realspace_electrostatics_batch(
    receptor_coords,
    poses_coords,
    receptor_radii,
    ligand_radii,
    receptor_charges,
    ligand_charges,
    lj_cutoff,
    electrostatic_cutoff,
    alpha,
    epsilon,
):
    ...
```

Input rank for `poses_coords` is `(N_poses, N_lig, 3)`.

### Numerical safety requirements

- use batch-vectorized math only
- no Python loop over poses
- for masked Ewald / Coulomb distances, fallback to the cutoff, not `alpha`
- keep error-bound computation scalar and uniform over all poses


## Phase 1 Charge Plumbing Rules

Electrostatics cannot be silently enabled without charges.

Rules:

- if certified composite scoring is requested and `LigandContext.charges` is
  present, use it
- if receptor charges are supplied, use them
- otherwise require explicit charge assignment through the existing
  `ChargeMethod` path
- do not silently invent charges inside the scorer

This keeps the proof boundary honest: the score is certified relative to the
supplied charges, not relative to a hidden preprocessing heuristic.


## Charge-Quality Risk Already Observed

Current empirical result from the existing `1ajp` benchmark assets:

- simple element-rule charges do not materially improve the native-vs-decoy
  ordering for `1ajp`
- parameter sweeps over `alpha`, `electrostatic_cutoff`, and `dielectric` with
  simple charges still leave the decoy pose strongly preferred
- Gasteiger ligand charges align cleanly on `1ajp`
- Gasteiger receptor charges currently do not align with `parse_structure`
  output on the receptor PDB (`6071` charges vs `6093` receptor atoms)

Implication:

- chemistry-grade charge assignment and atom-order alignment are likely a real
  subproblem, not an optional polish step
- the certified electrostatics path can land with explicit charges as inputs,
  but `1ajp` is unlikely to improve with the current simple-charge fallback


## Phase 2: Active Formal Runtime Integration

Only start this after Phase 1 lands and benchmarks cleanly.

Required work:

1. Generalize `score_certified_batch(...)` or add a sibling function returning
   the same `CertifiedBatchResult` shape for composite scoring.

2. Thread charge arrays through:
   - `dq_dock_engine/docking/formal_surrogates.py`
   - `dq_dock_engine/docking/formal_optimizer.py`
   - any helper that currently assumes certified scoring only needs coords/radii

3. Add a certified score-family selector for the formal path.

4. Update runtime contracts in `dq_dock_engine/docking/formal_handles.py` only
   after the scorer is genuinely used by the formal rounds.

Important constraint:

- the active certified optimizer does not depend on `route_scoring(...)`
- therefore "add enum case + routing" is not enough


## Phase 3: Reciprocal-Space Strategy (Deferred)

Do not block Phase 1 or Phase 2 on this.

Two acceptable long-term options:

### Option A: Final rescoring only

- keep active certified rounds on LJ + real-space electrostatics
- apply reciprocal-space correction only to a tiny finalist set
- formal guarantee becomes conditional on a post-selection correction rule

### Option B: Local decision-irrelevance theorem

- prove reciprocal-space variation is negligible over the local action family
- then omit reciprocal-space from the inner loop entirely while preserving the
  certified decision claim

This should be treated as a distinct theorem project, not Phase 1 plumbing.


## Validation / Success Criteria

Phase 1 must satisfy all of these:

1. unit tests for the new batched scorer shapes and numerical stability
2. handle alias regeneration test still passes
3. existing certified LJ path remains unchanged when electrostatic charges are zero
4. `1ajp` becomes the first canary benchmark
5. on `1ajp`, native-vs-decoy score ordering improves in the expected direction
6. benchmark harness runs without new NaN / tracing / batch-regression issues

Phase 2 must additionally satisfy:

1. formal surrogate path uses the composite certified scorer
2. exact-round certificates still compose with the new error bound
3. staged / singleton branches still produce valid delta accounting


## Work Order

This is the recommended execution sequence.

1. Lean symbol cleanup for `erfc`
2. signed-charge real-space bound
3. easy alpha-dependent `R^-3` domination theorem
4. utility-sum uniform-approx theorem
5. new Python batched composite scorer
6. charge plumbing into non-formal scoring path
7. benchmark on `1ajp`
8. only then integrate into `formal_surrogates.py` and formal rounds
9. defer reciprocal-space until the real-space path proves its value


## Explicit Non-Goals For Now

- do not reuse `dq_dock_engine/physics/ewald.py` as if it were already a docking
  cross-interaction scorer
- do not claim reciprocal-space is solved in Phase 1
- do not add theorem handles to active runtime contracts until the runtime uses
  them
- do not make the certified path depend on `route_scoring(...)` alone
