# Certified Docking Pruning Checklist

**Status**: Execution checklist derived from `CERTIFIED_DOCKING_PRUNING_PLAN.md`
**Date**: 2026-03-18
**Goal**: Implement the finite sampled docking proof stack in dependency order, with explicit acceptance gates.

---

## Stage 0: Preconditions

- [ ] Confirm `lake build` succeeds in `docs/papers/paper4_decision_quotient/proofs`
- [ ] Confirm the current theorem anchors still match the plan:
  - [ ] `EpsilonUtilityGap.lean`
  - [ ] `Sufficiency.lean`
  - [ ] `MolecularSrank.lean`
  - [ ] `DiscretizedState.lean`
- [ ] Freeze naming conventions for new files before implementation begins
- [ ] Decide whether sampled action restriction will use:
  - [ ] a subtype over an ambient action family, or
  - [ ] a standalone finite sampled action type

Acceptance gate:

- [ ] The theorem dependency order is fixed and no foundational rename is pending

---

## Stage 1: Finite Sampled Action Layer

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DiscretizedAction.lean`

- [ ] Add `GridMDAction`
- [ ] Add the supporting equivalence to a finite function space
- [ ] Prove / derive `Fintype (GridMDAction NL N)`
- [ ] Add `liftGridAction`
- [ ] Add one sanity lemma relating lifted action atom count to `NL`

Proof sketch checklist:

- [ ] `toFun` / `invFun` are definitionally inverse
- [ ] `Fintype.ofEquiv` closes the finiteness proof
- [ ] `liftGridAction` reuses `liftGridAtom`-style coordinate conversion rather than duplicating logic ad hoc

Acceptance gate:

- [ ] A toy theorem compiles with `[Fintype (GridMDAction NL N)]`

---

## Stage 2: `GridMDState` Coordinate/Product Instances

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/GridMDInstances.lean`

- [ ] Define the flattened coordinate count `3 * NP + 3 * NL`
- [ ] Define `gridMDStateProj`
- [ ] Add `CoordinateSpace (GridMDState NP NL N) ...`
- [ ] Define `gridMDStateReplace`
- [ ] Add `ProductSpace (GridMDState NP NL N) ...`
- [ ] Prove `replace_proj_eq`
- [ ] Prove `replace_proj_ne`

Proof sketch checklist:

- [ ] Case split protein vs ligand coordinate
- [ ] Case split axis `x/y/z`
- [ ] Reduce `replace_proj_eq` by construction
- [ ] Reduce `replace_proj_ne` by coordinate inequality decomposition

Acceptance gate:

- [ ] `DecisionProblem.sufficient_erase_irrelevant'` typechecks on a toy `GridMDState` decision problem

---

## Stage 3: Sampled Docking Problem Wrapper

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDocking.lean`

- [ ] Define sampled-action support object
- [ ] Define restricted sampled action type or subtype
- [ ] Define sampled docking exact decision problem
- [ ] Define sampled docking coarse decision problem
- [ ] Prove sampled action domain finiteness
- [ ] Prove restricted optimum equals ambient optimum intersected with support

Design checklist:

- [ ] Avoid fake utilities for out-of-support actions if a subtype formulation is cleaner
- [ ] Keep exact/coarse score families structurally aligned so later perturbation theorems are wrappers, not rewrites

Acceptance gate:

- [ ] There is a theorem stating the sampled problem's `Opt` is exactly the optimum over the sampled support

---

## Stage 4: Winner-Preservation Wrapper

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingGap.lean`

- [ ] Define the exact-vs-coarse lifted state encoding if needed
- [ ] State `sampled_epsilon_margin_invariance`
- [ ] Prove it by reduction to `epsilon_margin_invariance`
- [ ] Add a corollary specialized to sampled docking exact/coarse score pairs

Proof sketch checklist:

- [ ] If direct reduction is awkward, lift state space to `ScoreFamily × S`
- [ ] Reuse `StrictUtilityGap exactDP a_star s`
- [ ] Keep the proof wrapper-level only; do not duplicate the gap argument from scratch

Acceptance gate:

- [ ] One theorem proves exact/coarse sampled winner preservation with no new axioms

---

## Stage 5: Finite Top-k Objects

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/FiniteTopK.lean`

- [ ] Decide the tie policy
- [ ] Define `topKList`
- [ ] Define `topKSet`
- [ ] Define `kthUtility`
- [ ] Define `survivorSet`
- [ ] Prove cardinality and membership lemmas
- [ ] Prove a threshold exclusion lemma
- [ ] Prove a threshold survivor containment lemma

Design checklist:

- [ ] Do not leave tie semantics implicit
- [ ] Prefer survivor-set theorems over brittle exact-list theorems when possible

Acceptance gate:

- [ ] The repository has a formal top-k object usable by later pruning theorems

---

## Stage 6: Ranking Preservation

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/RankingPreservation.lean`

- [ ] Define `PairwiseGap`
- [ ] Define boundary-gap helper objects
- [ ] Prove `pairwise_order_preserved_of_uniform_error`
- [ ] Prove boundary-order corollaries relevant to top-k

Proof sketch checklist:

- [ ] Use the two one-sided inequalities from `|u_exact - u_coarse| <= delta`
- [ ] Derive `u_coarse a - u_coarse b >= (u_exact a - u_exact b) - 2 * delta`
- [ ] Close the proof with the strict pairwise gap assumption

Acceptance gate:

- [ ] Pairwise order preservation is available as a reusable theorem

---

## Stage 7: Safe Top-k Pruning

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/TopKPreservation.lean`

- [ ] State `coarse_survivor_contains_exact_topK`
- [ ] State `topK_preserved_of_boundary_gap`
- [ ] State exclusion lemmas for actions below the exact kth boundary by more than `2 * delta`
- [ ] Package the theorems in a survivor-containment form suitable for runtime use

Proof sketch checklist:

- [ ] Reduce every top-k proof to a boundary comparison plus Stage 6 pairwise preservation
- [ ] Prove set containment before trying to prove exact top-k equality

Acceptance gate:

- [ ] There is a theorem of the form `exactTopK ⊆ coarseSurvivors`

---

## Stage 8: Uniform Approximation For One Real Scorer Pair

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CoarseApproximation.lean`

- [ ] Define `UniformUtilityApprox`
- [ ] Choose the first exact/coarse scorer pair
- [ ] Prove a finite-domain uniform approximation theorem
- [ ] Feed that theorem into Stage 4 and Stage 7 corollaries

Priority order:

1. [ ] Standard LJ vs cutoff LJ
2. [ ] Standard LJ vs softened LJ on bounded sampled domains
3. [ ] Coulomb tail approximation only after the LJ case works

Acceptance gate:

- [ ] One actual scorer pair has a proved `UniformUtilityApprox`

---

## Stage 9: Cutoff Locality Transfer

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingCutoff.lean`

- [ ] Define outside-cutoff coordinates in the sampled representation
- [ ] Prove sampled irrelevance from `md_relevant_only_if_within_cutoff`
- [ ] Combine with `sufficient_erase_irrelevant'`
- [ ] Prove a retained-coordinate sufficiency theorem for sampled docking

Dependency checklist:

- [ ] `GridMDState` `ProductSpace` exists
- [ ] sampled docking wrapper exists
- [ ] the molecular cutoff theorem is imported cleanly

Acceptance gate:

- [ ] There is a theorem saying inside-cutoff retained coordinates are sufficient under the cutoff boundedness premise

---

## Stage 10: Certificate Packaging

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CertifiedPruning.lean`

- [ ] Define `PruningCertificate`
- [ ] Define `CertifiedSurvivorSet`
- [ ] Package Stage 7 survivor containment as a certificate
- [ ] Add at least one example certificate theorem for sampled docking

Acceptance gate:

- [ ] A pruning theorem now returns a certificate object rather than only a proposition

---

## Medium Research Track

### 11. Uniform error bounds over sampled grids

- [ ] Replace abstract perturbation assumptions with real bounds for selected score pairs
- [ ] Track every constant explicitly
- [ ] Keep the first target simple and finite

### 12. Ranking stability with ties and near-ties

- [ ] Define an ambiguity band object
- [ ] Prove containment of exact top-k inside a certified ambiguity band
- [ ] Avoid claiming exact top-k equality when strict gaps fail

---

## Big Research Track

### 13. Chemistry-side physical bounds

- [ ] Instantiate `SatisfiesBoundedPotential` for standard LJ with cutoff
- [ ] Instantiate Coulomb tail bounds or Ewald-based bounds
- [ ] Delay composite chemistry until simple physical score families are discharged

### 14. Remove `lattice_sum_converges` axiom

- [ ] Replace the axiom in `Tractability/LatticeSum.lean` with a proof
- [ ] Recover explicit constants needed by `CutoffEpsilon.lean`

### 15. Continuous-to-discrete convergence

- [ ] Add `GridConvergence.lean`
- [ ] Prove a resolution-to-utility-error theorem
- [ ] Feed the resulting `eps(res)` into `UniformUtilityApprox`

---

## Runtime Integration Checklist

- [ ] Add proof-status labels in the implementation/docs:
  - [ ] `Certified`
  - [ ] `ConditionallyCertified`
  - [ ] `Heuristic`
- [ ] Benchmark certified pruning separately from heuristic pruning
- [ ] Do not route heuristic stages through proof-backed language without an explicit status downgrade

---

## Stop Conditions

- [ ] Stop introducing new pruning heuristics until Stage 7 exists
- [ ] Stop claiming formal grounding for score families not actually tied to Lean theorems
- [ ] Stop using empirical rank-correlation as a substitute for containment theorems

---

## Final Milestones

- [ ] Milestone A: finite sampled docking is a formal `DecisionProblem`
- [ ] Milestone B: exact-vs-coarse winner preservation is proved
- [ ] Milestone C: exact top-k survivor containment is proved
- [ ] Milestone D: cutoff/pocket pruning is certified as sufficiency-preserving
- [ ] Milestone E: at least one real docking scorer pair discharges a physical approximation theorem
