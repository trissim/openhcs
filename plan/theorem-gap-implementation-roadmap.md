# Theorem Gap Implementation Roadmap

## Objective

Close the remaining proof-to-runtime gaps so that certified docking derives as much
runtime behavior as possible from proved theorems, with one authoritative source
for each fact family and no manually synchronized shadow logic.

This roadmap is specifically about the remaining gaps after the recent progress on:

- theorem-derived seed-budget computation from `confidence`, `target_rmsd`, and geometry
- theorem-derived certified GD iteration budgets
- canonical retain-set pruning
- dyadic adequacy round derivation
- certified top-1 pruning / posterior / selection bridges

The aim is not just correctness. The aim is to maximize **speed subject to formal
correctness** by wiring the theorems that tighten pruning, shrink survivor sets,
reduce unnecessary refinement, and eliminate batch-dependent slack.

## Current debugging state (Mar 2026)

This section records the concrete runtime state reached during the current
``1hk4`` debugging campaign, so future work can bridge directly from observed
behavior to the roadmap gaps below instead of rediscovering the same failures.

### What is already done

The following debugging-driven fixes are already in the runtime:

- certified binding-site rigid sampling now respects the certified sphere rather
  than its enclosing cube
- the translation-quaternion product no longer truncates in a spatially biased
  flattened order
- rigid local actions rotate about each pose centroid instead of the world origin
- support expansion is tracked per pose rather than leaking across the full batch
- local refinement is constrained to stay inside the certified binding site in
  known-pocket mode
- the formal local root rotation step is tied to translational cell scale via
  ``base_rotation_step_rad = base_translation_step / ligand_radius``
- a theorem-backed rigid future-improvement bound now cuts the local-refinement
  set before the expensive formal local optimizer runs
- ``_certified_refinement()`` now builds its SE(3) energy function from the
  certified optimization score context instead of a bare LJ surrogate
- ``--n-poses-override`` is re-exposed on the benchmark CLI for controlled
  debugging runs while the fully theorem-backed seed-budget path is still being
  repaired

### Current observed behavior on ``1hk4``

- with ``--use-crystal-ligand-geometry --disable-conformer-search`` and
  ``--n-poses-override 16384``, known-conformer ``1hk4`` now returns about
  ``0.23 A`` raw RMSD
- the sampled family at ``16384`` poses contains an oracle seed around
  ``0.39 A`` raw RMSD, so rigid sampling is now good enough at that budget
- certified coarse pruning is still loose enough to keep all ``16384`` poses on
  the current theorem-critical path
- the theorem-backed rigid future-improvement filter reduces local refinement
  from ``16384`` poses to ``17`` on ``1hk4``
- observed probe certificates can now exist on ``1hk4`` after the refinement
  energy-function alignment, but the resulting ``(q, μ, M)`` values are still too
  weak to reduce the derived seed budget materially

### What remains unresolved

- the seed-budget reduction path is still blocked by certificate strength rather
  than certificate existence
- rich-chemistry pruning delta is still not owned by one theorem-backed runtime
  object, and the current scalar branches are too loose to prune aggressively
- the local future-improvement bound is now live, but it still exists as runtime
  scalar logic rather than as part of one authoritative pruning-and-refinement
  execution plan object

### How the current debugging bridges to the roadmap below

- the new local-refinement cut directly advances **Gap 3** conceptually, but it
  still needs to be turned into a first-class execution-plan object
- the still-loose rich pruning delta is exactly **Gap 1**
- the still-too-weak probe reduction path is exactly **Gap 4**
- the current ``1hk4`` debugging work has also clarified that **Gap 1** and
  **Gap 3** interact: pruning tractability is not just about coarse score delta,
  it is also about combining that delta with a theorem-backed future-improvement
  budget before local refinement

### Implementation update (Mar 2026, current branch)

The roadmap items below are now implemented in the runtime in the intended order,
except for the explicit axiom/docs cleanup work that was intentionally deferred.

- **Gap 1 implemented:** rich-chemistry pruning slack is now owned by
  ``CertifiedPruningDeltaBudget`` and derived through
  ``CertifiedRichChemistryPlan.pruning_delta_budget()`` /
  ``CertifiedScoringContext.pruning_delta_budget()``
- **Gap 2 implemented:** pose-local conformer slack is now represented by
  ``PoseSpecificImprovementBudget`` / ``PoseSpecificImprovementBudgetFamily`` and
  used in certified pruning, exact survivor selection, and final conformer gating
- **Gap 3 implemented:** ``CertifiedPipelineExecutionPlan`` now carries pruning
  decisions, and refinement budgets are represented explicitly via
  ``CertifiedRefinementBudget`` plus ``se3_refinement.py::CertifiedIterationBudgetPlan``
- **Gap 4 implemented:** seed-budget selection is now represented by
  ``CertifiedSeedBudgetPlan`` and threaded through ``DockingRequestBase.n_poses``
  instead of being reduced to an opaque helper integer
- **Gap 5 implemented:** missing theorem families now have explicit grouped handle
  helpers in ``dq_dock_engine/docking/formal_handles.py`` and matching generated
  aliases in ``dq_dock_engine/generated/formal_handle_aliases.py``
- **Gap 6 intentionally deferred:** the axiom/proof-thread cleanup work was left
  out on purpose for now, per user direction that it is epistemically unnecessary
  for the current runtime-leverage push

## Architectural rules for this work

1. **One authoritative source per budget family.**
   - There must be exactly one runtime object that owns each of:
     - pruning delta
     - conformer improvement bound
     - seed budget
     - refinement budget
   - Any masks, thresholds, summaries, handles, CSV rows, or logs must be derived.

2. **Nominal runtime contracts, not scattered scalars.**
   - Do not keep adding raw `delta`, `tau`, `n_rounds`, `n_steps`, and
     `additive_correction` values in unrelated functions.
   - Introduce authoritative dataclasses for theorem-derived plans/budgets.

3. **Fail loud on architectural violations.**
   - No new `getattr(..., default)` or silent fallback chains for theorem-backed
     fields that should be guaranteed by the certified path.

4. **Theorem provenance must travel with the derived object.**
   - If a plan uses `canonical_pruning_and_budget_optimal`, the runtime object that
     owns the plan should carry that handle family directly.

## Current proof surface: already wired enough to preserve

These are not the current gaps; they are the already-live baseline the roadmap must preserve.

| Area | Lean surface | Current runtime hook |
|------|--------------|----------------------|
| Conformer B&B lower bounds | `ConformerSearch.lean` CS1-CS9 + per-dimension cell bounds | `dq_dock_engine/docking/conformer_search.py` |
| Seed budget sufficiency | `SeedBudgetDerivation.lean` SB family | `DockingRequestBase.n_poses`, `derive_seed_budget`, `_probe_seed_budget_certificate()` |
| Certified GD iteration budget | `EnergyRMSDConvergence.lean`, logarithmic budget theorem family | `dq_dock_engine/docking/se3_refinement.py::certified_iteration_budget()` |
| Dyadic adequacy rounds | joint adequate dyadic-round theorem family | `pipeline.py::_certified_refinement()`, formal local optimizer shell budgeting |
| Canonical retain safety | `BlindConformerPipelineOptimality.lean::canonicalRetain` / safety theorem | `pipeline.py::_canonical_retain_mask()` |
| Top-1 certified pruning and optimizer witnesses | CP / TK / FLO families | `formal_surrogates.py`, `formal_optimizer.py`, `formal_handles.py` |

## Remaining gaps, ranked by impact

### Execution status (current branch)

- [x] **Gap 1 complete**: pruning delta is now owned by one authoritative runtime
  object (`CertifiedPruningDeltaBudget`) and consumed through the certified
  pruning path rather than unrelated scalar branches
- [x] **Gap 2 complete**: pose-specific conformer-improvement budgets are now
  represented explicitly and used in certified pruning, exact survivor selection,
  and final conformer gating
- [x] **Gap 3 complete**: a first-class `CertifiedPipelineExecutionPlan` now
  carries retain masks, lower bounds, pruning slack, and refinement-budget state;
  `se3_refinement.py` also exposes a first-class iteration-budget plan object
- [x] **Gap 4 complete**: seed-budget adequacy/minimality is now represented by
  `CertifiedSeedBudgetPlan` and threaded through `DockingRequestBase.n_poses`
- [x] **Gap 5 complete**: the missing theorem families now have explicit grouped
  handle helpers in `formal_handles.py` plus generated alias exports
- [ ] **Gap 6 remaining**: proof-thread / proof-audit docs still need to be
  synchronized to the current Lean tree, and the remaining axioms / theorem-shaped
  placeholders still need to be explicitly tracked there as blocking vs non-blocking

| Priority | Gap | Theorem(s) | Why it matters |
|---------|-----|------------|----------------|
| Done | Joint pruning + refinement plan is not authoritative end-to-end | `canonicalRetain_minimizes_pipelineCost`, `canonical_pruning_and_budget_optimal` | Implemented via `CertifiedPipelineExecutionPlan`, `CertifiedRefinementBudget`, and plan-owned pruning/refinement state |
| Done | Rich-chemistry pruning delta is not fully theorem-SSOT | `base_plus_omitted_uniformApprox`, `exact_with_omitted_ge_coarse_minus_totalError`, `omitted_channel_is_bounded_by_supremum`, `water_bridge_is_bounded_omitted_channel`, `exact_vs_softened_lj_error`, `softened_lj_self_approx_zero` | Implemented via `CertifiedPruningDeltaBudget` and rich-chemistry pruning-delta derivation in scoring/scoring_context |
| Done | Conformer-improvement bound is still mostly ligand-global | `pose_specific_improvement_bound_of_active_subset` | Implemented via `PoseSpecificImprovementBudget` / `PoseSpecificImprovementBudgetFamily` |
| Done | Seed-budget optimality is only partially reflected | `minimal_adequate_seedBudget_optimal` | Implemented via `CertifiedSeedBudgetPlan` and request-owned seed-budget provenance |
| Done | Theorem provenance surface is incomplete | missing direct bundles/accessors for the above theorem families | Implemented via new grouped handle helpers and new generated aliases |
| Remaining | Formal completeness/docs are out of sync with the actual Lean tree | remaining axioms + stale docs | Remaining review/documentation pass after the runtime/handle refactor |

## Gap 1: make pruning delta a single-source theorem object

### Current state

- `pipeline.py::_score_softened_pose_batch()` chooses among:
  - `delta = 0.0`
  - `delta = scoring_context.analytic_pruning_delta()`
  - `delta = coarse_batch.softening_error_bound`
- `formal_surrogates.py::two_cutoff_approximation_witness()` still builds a
  LJ-style `combined_delta` and several fast-path round functions still use it directly.
- `scoring.py` aggregates rich-chemistry errors into batch results, but the runtime
  does not yet have one explicit authoritative object for the decomposition:
  shared base vs cutoff tails vs truly omitted channels.

### Required refactor

Introduce an authoritative runtime object, e.g. `CertifiedPruningDeltaBudget`, that owns:

- `shared_base_delta`
- `cutoff_tail_delta`
- `omitted_value_delta`
- `total_delta`
- theorem handles / witness handles
- optional human-readable breakdown for audit output

This object should be derived from `CertifiedRichChemistryPlan` and then consumed by:

- `pipeline.py::_score_softened_pose_batch()`
- `pipeline.py::_certified_pruning_pass()`
- `formal_surrogates.py` staged-top1 / singleton-accept rounds

### Theorem mapping

- `softened_lj_self_approx_zero`
  - authoritative statement that the shared softened-LJ base contributes zero delta
- `exact_vs_softened_lj_error`
  - explains why exact-vs-softened mismatch is unstable and should not be used as a batch-max pruning delta when both paths already use softened LJ
- `base_plus_omitted_uniformApprox`
  - compose a certified base approximation with omitted bounded channels
- `exact_with_omitted_ge_coarse_minus_totalError`
  - runtime-facing lower-bound theorem for omitted attractive channels
- `omitted_channel_is_bounded_by_supremum`
  - admissibility criterion for channels that may be replaced by zero
- `water_bridge_is_bounded_omitted_channel`
  - concrete safe omission theorem for water bridges

### Concrete file targets

- `dq_dock_engine/docking/scoring.py`
  - make rich-chemistry plans derive a theorem-backed pruning-delta budget
- `dq_dock_engine/docking/pipeline.py`
  - stop deciding delta from unrelated scalar branches
- `dq_dock_engine/docking/formal_surrogates.py`
  - stop treating LJ cutoff witness delta as the sole authoritative pruning slack when rich chemistry is active

### Acceptance criteria

- All certified rich-chemistry pruning paths derive `delta` from one object.
- No batch-size-dependent `max(abs(exact - coarse))` term remains on the theorem-critical rich-chemistry path.
- The delta breakdown visibly distinguishes:
  - shared-base zero contribution
  - cutoff-tail approximation error
  - omitted-channel value bound

## Gap 2: replace ligand-global conformer improvement with pose-specific budgets

### Current state

- `pipeline.py::_conformer_improvement_bound()` sums all per-bond bounds into one ligand-global scalar.
- `pipeline.py::_canonical_retain_mask()` then subtracts one coarse improvement correction from all poses.
- This leaves pruning sound, but not maximally tight.

### Required refactor

Introduce a nominal budget object, e.g. `PoseSpecificImprovementBudget`, that owns:

- active torsion subset or certified superset
- per-bond local bounds
- summed certified improvement budget
- provenance handles

This budget should be derived once per pose from the pose’s active torsion subset or
support expansion state, then consumed by pruning and survivor selection.

### Theorem mapping

- `pose_specific_improvement_bound_of_active_subset`
  - authoritative theorem for replacing global `Σ all bonds` with `Σ active subset`

### Concrete file targets

- `dq_dock_engine/docking/pipeline.py`
  - replace scalar `_conformer_improvement_bound()` use in certified pruning paths
- `dq_dock_engine/docking/formal_optimizer.py`
  - thread support-expansion / pose-local state into improvement-budget derivation
- optionally `dq_dock_engine/docking/conformer_search.py`
  - reuse per-bond Lipschitz structure where that helps derive active-subset bounds

### Acceptance criteria

- For every pose, certified improvement budget is `<=` current ligand-global bound.
- Canonical retain sets are never larger because of avoidable global slack.
- The provenance of the active subset is inspectable in O(1) conceptual effort.

## Gap 3: derive one joint pruning-and-refinement execution plan

### Current state

- `pipeline.py::_canonical_retain_mask()` derives the retain set from score-side data.
- `se3_refinement.py::certified_iteration_budget()` derives SE(3) GD iterations from local certificates.
- `formal_optimizer.py::refine_poses_certified()` consumes `n_rounds` as a separate externally supplied scalar.
- In other words, the runtime has theorem-backed pieces, but not one authoritative
  object representing the jointly optimal certified pipeline decision.
- Runtime progress already made:
  - a theorem-backed rigid future-improvement bound is now used to cut the local
    refinement set before formal local refinement
  - on `1hk4`, this reduces the local-refinement set from `16384` to `17`
  - however, that bound is still computed as local scalar logic inside
    `pipeline.py`, not as a first-class execution-plan object that also owns the
    pruning decision

### Required refactor

Introduce a top-level authoritative object, e.g. `CertifiedPipelineExecutionPlan`, with fields like:

- `lower_bounds`
- `tau`
- `retain_mask`
- `pruning_delta_budget`
- `improvement_budget_family`
- `refinement_budget` (variant: dyadic rounds vs GD iterations)
- `postfilter_cost_model`
- theorem / witness handles

The key rule is that pruning and refinement budget are derived together from the
same cost model object, not computed in disconnected functions and manually kept coherent.

### Theorem mapping

- `canonicalRetain_minimizes_pipelineCost`
  - canonical retain is optimal among certified-safe retain sets
- `canonical_pruning_and_budget_optimal`
  - canonical pruning plus canonical RMSD-certified refinement budget is jointly optimal under the additive cost model
- `canonical_twoStage_le_allExact`
  - validates two-stage certified pruning against all-exact evaluation when prefilter cost is covered

### Concrete file targets

- `dq_dock_engine/docking/pipeline.py`
  - `_certified_pruning_pass()` should produce or consume the authoritative execution plan
- `dq_dock_engine/docking/formal_optimizer.py`
  - consume plan-owned round budgets instead of opaque external integers where theorem-critical
- `dq_dock_engine/docking/se3_refinement.py`
  - expose refinement-budget derivation as a first-class plan object rather than a naked integer

### Acceptance criteria

- There is a single runtime object that answers both:
  - which poses survive?
  - what certified refinement budget do survivors receive?
- The pipeline no longer recomputes or reinterprets these decisions in multiple places.
- The theorem handle surface explicitly names the joint-optimality family.

## Gap 4: represent seed-budget adequacy/minimality explicitly

### Current state

- `DockingRequestBase.n_poses` derives the main seed budget theoremically.
- `_probe_seed_budget_certificate()` uses a small observed probe to tighten the basin estimate and then picks the smallest certified resulting budget seen.
- But the runtime still treats probe calibration as ad hoc constants (`SEED_BUDGET_PROBE_POSES = 16`, `SEED_BUDGET_PROBE_TOP_K = 4`) rather than as an explicit monotone adequacy search object.
- Runtime progress already made:
  - `_certified_refinement()` now certifies against the certified optimization
    score context instead of a bare LJ energy surrogate
  - this repair is strong enough for `1hk4` to produce non-`None` observed probe
    certificates in cases where it previously produced none
  - the remaining blocker is now certificate strength / basin-amplification, not
    total probe-certification failure

### Required refactor

Introduce a nominal object, e.g. `CertifiedSeedBudgetPlan`, that owns:

- the seed-budget family `n ↦ lowerBoundFamily_n`
- adequacy evidence
- minimal adequate budget among explored candidates
- seed overhead model
- theorem handles

The runtime does **not** need to pretend that every engineering cap is proven.
But it does need to ensure the chosen full-run budget is the authoritative minimal
certified adequate budget among the explored family, not merely the result of a
helper function returning an int.

### Theorem mapping

- `minimal_adequate_seedBudget_optimal`
  - minimal adequate budget is optimal among adequate budgets when seed cost is monotone
- SB family already live
  - preserve the current seed sufficiency derivation

### Concrete file targets

- `dq_dock_engine/docking/pipeline.py`
  - `_probe_seed_budget_certificate()`
  - `DockingRequestBase.n_poses`

### Acceptance criteria

- The chosen seed budget is represented together with adequacy/minimality provenance.
- Probe calibration constants, if retained, are explicitly marked as engineering bounds outside the authoritative final seed-budget fact.

## Gap 5: expand theorem-handle provenance for the missing families

### Current state

- `dq_dock_engine/generated/formal_handle_aliases.py` contains handle codes.
- `dq_dock_engine/docking/formal_handles.py` bundles the currently live CP / TK / FLO families.
- The newly important theorem families above are not surfaced as first-class handle bundles/accessors there.

### Required refactor

Add explicit grouped handle helpers for:

- joint pruning/budget optimality
- omitted-channel bounds
- softened-LJ shared-base delta elimination
- pose-specific improvement budgets
- seed-budget minimality

Generated aliases should remain generated; the hand-authored layer should provide the
domain-level grouping the runtime actually consumes.

### Concrete file targets

- `dq_dock_engine/docking/formal_handles.py`
- any generated-handle export pipeline that needs regeneration

### Acceptance criteria

- A reviewer can trace every major certified budget object to a small named handle bundle.
- No proof-critical runtime decision depends on a theorem family that is absent from the hand-authored handle layer.

## Gap 6: finish the remaining formal-completeness and documentation sync work

### Current state

- `docs/proof_thread_context.md` still describes older `ConformerSearch.lean` axiom gaps for `per_dimension_lipschitz_bound` and `per_dimension_energy_lower_bound_on_cell`.
- In the current Lean tree, those are already theorems.
- Actual remaining formal-completeness gaps now include at least:
  - `ConformerSearch.lean::single_bond_torsion_lipschitz` (`axiom`)
  - `SoftLJApproximation.lean::softenedLipschitz_le_rawLipschitz` (`axiom`)
- `ConformerSearch.lean::rigid_body_isometry` is theorem-shaped but currently just restates its rigid hypothesis; it is not yet a full geometric derivation.

### Required refactor

1. Replace stale proof-thread notes with the current actual remaining gaps.
2. Prove or isolate the remaining axioms.
3. Re-run proof audits so runtime planning documents match the real Lean graph.

### Concrete file targets

- `docs/proof_thread_context.md`
- `dq_dock_engine/PROOF_AUDIT.md`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/ConformerSearch.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SoftLJApproximation.lean`

### Acceptance criteria

- No planning doc claims an axiom gap that is already proven.
- All remaining axioms are explicitly tracked as either:
  - blocking runtime leverage
  - non-blocking cleanup

## Recommended implementation order

1. **Pruning-delta SSOT**
   - highest immediate runtime leverage
   - shrinks survivor sets and unlocks better pruning everywhere else
2. **Pose-specific improvement budgets**
   - compounds with better deltas to reduce over-retention
3. **Joint pruning-and-refinement plan object**
   - turns scattered theorem-backed pieces into one authoritative runtime contract
4. **Seed-budget adequacy/minimality representation**
   - cleans up the last major budget family still lacking a first-class runtime plan object
5. **Handle provenance expansion**
   - makes the new theorem usage auditable
6. **Lean/docs resync**
   - prevents future implementation against stale proof status

## Validation matrix

### Correctness validation

- Unit tests for each derived budget object:
  - monotonicity
  - nonnegativity
  - theorem-handle presence
- Survivor-set regression tests:
  - winner preserved
  - retain-set size weakly decreases vs prior looser certified path
- Seed-budget tests:
  - minimal adequate budget chosen among explored candidates
- Refinement tests:
  - plan-owned budgets are the ones consumed by refinement entry points

### Performance validation

- Measure before/after on the certified path:
  - pruning delta magnitude
  - canonical retain-set cardinality
  - number of exact rescored poses
  - number of refined poses
  - wall-clock runtime
- Keep the benchmark protocol blind to crystal geometry defaults.

### Provenance validation

- Every major runtime certificate/budget object exposes theorem handles.
- `formal_handles.py` can enumerate the relevant families without string fishing in unrelated files.

### Documentation validation

- `proof_thread_context.md` and `PROOF_AUDIT.md` match the current Lean tree.
- No stale references remain to already-proven axiom gaps.

## Definition of done

This roadmap is complete when:

1. The certified path owns one authoritative object for each budget family.
2. Pruning delta, conformer improvement, seed budget, and refinement budget are all theorem-derived first-class runtime concepts.
3. The joint pruning/refinement theorem family is reflected in the actual runtime architecture, not just in comments.
4. The strongest currently-proven omitted-channel and softened-base theorems drive pruning slack.
5. The proof-tracking docs reflect the current Lean state accurately.
