# Certified Docking Pruning Checklist

**Status**: Execution checklist derived from `CERTIFIED_DOCKING_PRUNING_PLAN.md`
**Date**: 2026-03-18
**Goal**: Implement the finite sampled docking proof stack in dependency order, with explicit acceptance gates.

---

## Stage 0: Preconditions

- [ ] Confirm `lake build` succeeds in `docs/papers/paper4_decision_quotient/proofs`
- [ ] Confirm the current theorem anchors still match the plan:
  - [x] `EpsilonUtilityGap.lean`
  - [x] `Sufficiency.lean`
  - [x] `MolecularSrank.lean`
  - [x] `DiscretizedState.lean`
- [x] Freeze naming conventions for new files before implementation begins
- [ ] Decide whether sampled action restriction will use:
  - [ ] a subtype over an ambient action family, or
  - [x] a standalone finite sampled action type

Acceptance gate:

- [x] The theorem dependency order is fixed and no foundational rename is pending

---

## Stage 1: Finite Sampled Action Layer

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DiscretizedAction.lean`

- [x] Add `GridMDAction`
- [x] Add the supporting equivalence to a finite function space
- [x] Prove / derive `Fintype (GridMDAction NL N)`
- [x] Add `liftGridAction`
- [x] Add one sanity lemma relating lifted action atom count to `NL`

Proof sketch checklist:

- [x] `toFun` / `invFun` are definitionally inverse
- [x] `Fintype.ofEquiv` closes the finiteness proof
- [x] `liftGridAction` reuses `liftGridAtom`-style coordinate conversion rather than duplicating logic ad hoc

Acceptance gate:

- [x] A toy theorem compiles with `[Fintype (GridMDAction NL N)]`

---

## Stage 2: `GridMDState` Coordinate/Product Instances

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/GridMDInstances.lean`

- [x] Define the flattened coordinate count `3 * NP + 3 * NL`
- [x] Define `gridMDStateProj`
- [x] Add `CoordinateSpace (GridMDState NP NL N) ...`
- [x] Define `gridMDStateReplace`
- [x] Add `ProductSpace (GridMDState NP NL N) ...`
- [x] Prove `replace_proj_eq`
- [x] Prove `replace_proj_ne`

Proof sketch checklist:

- [x] Case split protein vs ligand coordinate
- [x] Case split axis `x/y/z`
- [x] Reduce `replace_proj_eq` by construction
- [x] Reduce `replace_proj_ne` by coordinate inequality decomposition

Acceptance gate:

- [ ] `DecisionProblem.sufficient_erase_irrelevant'` typechecks on a toy `GridMDState` decision problem

---

## Stage 3: Sampled Docking Problem Wrapper

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDocking.lean`

- [x] Define sampled-action support object
- [x] Define restricted sampled action type or subtype
- [x] Define sampled docking exact decision problem
- [x] Define sampled docking coarse decision problem
- [x] Prove sampled action domain finiteness
- [x] Prove restricted optimum equals ambient optimum intersected with support

Design checklist:

- [x] Avoid fake utilities for out-of-support actions if a subtype formulation is cleaner
- [x] Keep exact/coarse score families structurally aligned so later perturbation theorems are wrappers, not rewrites

Acceptance gate:

- [x] There is a theorem stating the sampled problem's `Opt` is exactly the optimum over the sampled support

---

## Stage 4: Winner-Preservation Wrapper

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingGap.lean`

- [x] Define the exact-vs-coarse lifted state encoding if needed
- [x] State `sampled_epsilon_margin_invariance`
- [x] Prove it by reduction to `epsilon_margin_invariance`
- [x] Add a corollary specialized to sampled docking exact/coarse score pairs

Proof sketch checklist:

- [x] If direct reduction is awkward, lift state space to `ScoreFamily × S`
- [x] Reuse `StrictUtilityGap exactDP a_star s`
- [x] Keep the proof wrapper-level only; do not duplicate the gap argument from scratch

Acceptance gate:

- [x] One theorem proves exact/coarse sampled winner preservation with no new axioms

---

## Stage 5: Finite Top-k Objects

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/FiniteTopK.lean`

- [x] Decide the tie policy
- [x] Define `topKList`
- [x] Define `topKSet`
- [x] Define `kthUtility`
- [x] Define `survivorSet`
- [x] Prove cardinality and membership lemmas
- [x] Prove a threshold exclusion lemma
- [x] Prove a threshold survivor containment lemma

Design checklist:

- [x] Do not leave tie semantics implicit
- [x] Prefer survivor-set theorems over brittle exact-list theorems when possible

Acceptance gate:

- [x] The repository has a formal top-k object usable by later pruning theorems

---

## Stage 6: Ranking Preservation

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/RankingPreservation.lean`

- [x] Define `PairwiseGap`
- [x] Define boundary-gap helper objects
- [x] Prove `pairwise_order_preserved_of_uniform_error`
- [x] Prove boundary-order corollaries relevant to top-k

Proof sketch checklist:

- [x] Use the two one-sided inequalities from `|u_exact - u_coarse| <= delta`
- [x] Derive `u_coarse a - u_coarse b >= (u_exact a - u_exact b) - 2 * delta`
- [x] Close the proof with the strict pairwise gap assumption

Acceptance gate:

- [x] Pairwise order preservation is available as a reusable theorem

---

## Stage 7: Safe Top-k Pruning

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/TopKPreservation.lean`

- [x] State `coarse_survivor_contains_exact_topK`
- [x] State `topK_preserved_of_boundary_gap`
- [x] State exclusion lemmas for actions below the exact kth boundary by more than `2 * delta`
- [x] Package the theorems in a survivor-containment form suitable for runtime use

Proof sketch checklist:

- [x] Reduce every top-k proof to a boundary comparison plus Stage 6 pairwise preservation
- [x] Prove set containment before trying to prove exact top-k equality

Acceptance gate:

- [x] There is a theorem of the form `exactTopK ⊆ coarseSurvivors`

---

## Stage 8: Uniform Approximation For One Real Scorer Pair

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CoarseApproximation.lean`

- [x] Define `UniformUtilityApprox`
- [x] Choose the first exact/coarse scorer pair
- [x] Prove a finite-domain uniform approximation theorem
- [x] Feed that theorem into Stage 4 and Stage 7 corollaries

Priority order:

1. [x] Standard LJ vs cutoff LJ
2. [ ] Standard LJ vs softened LJ on bounded sampled domains
3. [ ] Coulomb tail approximation only after the LJ case works

Acceptance gate:

- [x] One actual scorer pair has a proved `UniformUtilityApprox`

---

## Stage 9: Cutoff Locality Transfer

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingCutoff.lean`

- [x] Define outside-cutoff coordinates in the sampled representation
- [x] Prove sampled irrelevance from `md_relevant_only_if_within_cutoff`
- [x] Combine with `sufficient_erase_irrelevant'`
- [x] Prove a retained-coordinate sufficiency theorem for sampled docking

Dependency checklist:

- [x] `GridMDState` `ProductSpace` exists
- [x] sampled docking wrapper exists
- [x] the molecular cutoff theorem is imported cleanly

Acceptance gate:

- [ ] There is a theorem saying inside-cutoff retained coordinates are sufficient under the cutoff boundedness premise

---

## Stage 10: Certificate Packaging

### File: `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CertifiedPruning.lean`

- [x] Define `PruningCertificate`
- [x] Define `CertifiedSurvivorSet`
- [x] Package Stage 7 survivor containment as a certificate
- [x] Add at least one example certificate theorem for sampled docking

Acceptance gate:

- [x] A pruning theorem now returns a certificate object rather than only a proposition

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

- [x] Replace the axiom in `Tractability/LatticeSum.lean` with a proof
- [ ] Recover explicit constants needed by `CutoffEpsilon.lean`

### 15. Continuous-to-discrete convergence

- [x] Add `GridConvergence.lean`
- [x] Prove a resolution-to-utility-error theorem
- [x] Feed the resulting `eps(res)` into `UniformUtilityApprox`

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

- [x] Milestone A: finite sampled docking is a formal `DecisionProblem`
- [x] Milestone B: exact-vs-coarse winner preservation is proved
- [x] Milestone C: exact top-k survivor containment is proved
- [x] Milestone D: cutoff/pocket pruning is certified as sufficiency-preserving
- [ ] Milestone E: at least one real docking scorer pair discharges a physical approximation theorem
