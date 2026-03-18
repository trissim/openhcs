# Certified Docking Pruning Plan

**Status**: In progress
**Date**: 2026-03-18
**Scope**: Turn sampled docking and coarse-to-fine pruning into theorem-carrying transformations rather than empirical ranking heuristics.

---

## Executive Summary

The current Lean development already contains the right backbone for certified pruning, but the pieces are not yet assembled into a finite sampled docking story.

The decisive existing theorems are:

- `StrictUtilityGap` and `epsilon_margin_invariance` in `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/EpsilonUtilityGap.lean`
- `isSufficient`, `isRelevant`, and `sufficient_erase_irrelevant'` in `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Sufficiency.lean`
- `OutsideCutoffApproximationBounded`, `md_relevant_only_if_within_cutoff`, and `md_srank_bound` in `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/MolecularSrank.lean`
- `GridMDState` and `liftGridState` in `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DiscretizedState.lean`

What is missing is not another heuristic score. What is missing is a finite sampled docking layer with the right instances and wrapper theorems, so that we can prove statements of the following form:

1. The sampled docking problem is a finite `DecisionProblem`.
2. A coarse score uniformly approximates the exact sampled score within `delta`.
3. If `delta` is smaller than a certified boundary gap, winner preservation and top-k survivor containment follow.
4. If a physical cutoff theorem supplies `OutsideCutoffApproximationBounded`, then outside-cutoff coordinates are formally irrelevant and may be erased.

This plan is organized by proof leverage. The first phase is mostly theorem engineering and instance construction. The later phases become real research.

### Progress Update

Completed theorem-engineering chunks now live in Lean:

- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DiscretizedAction.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/GridMDInstances.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDocking.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingGap.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/FiniteTopK.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/RankingPreservation.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/TopKPreservation.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CertifiedPruning.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CoarseApproximation.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingCutoff.lean`
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/LJApproximation.lean`

Additional progress beyond the initial finite core:

- the sampled restricted problem now inherits a cutoff-locality/srank bridge in `SampledDockingCutoff.lean`
- a first concrete scorer pair, exact LJ versus cutoff LJ, now has a finite-domain `UniformUtilityApprox` theorem in `LJApproximation.lean`
- exact LJ versus softened LJ now has the same finite-domain approximation treatment in `SoftLJApproximation.lean`
- a first Coulomb finite-domain exact/coarse approximation theorem now exists in `CoulombApproximation.lean`
- the old analytic axiom in `Tractability/LatticeSum.lean` has been removed and replaced by explicit 6-power and 12-power geometric tail bounds with concrete constants
- an abstract continuous-to-discrete bridge now exists in `GridConvergence.lean`, isolating the exact Lipschitz/resolution theorem shape needed to derive `eps(res)`-style approximation guarantees
- the runtime benchmark now carries explicit formal-status labels and separates reported results by status, so heuristic execution paths are no longer silently presented as proof-backed
- the full Lean proof project now builds again after repairing the sampled/grid/lattice theorem slice

The next frontier is now narrower and more mathematical:

- strengthen the current generic/conditional sampled cutoff sufficiency bridge into a fully instantiated theorem for the intended sampled state representation
- replace pointwise finite-radius LJ/cutoff witnesses with sharper physically meaningful error bounds
- prove continuous-to-discrete convergence and, if desired, recover a stronger asymptotic lattice-tail estimate without reintroducing axioms

---

## Ground Truth: Existing Formal Assets

This plan assumes the following Lean results are already the canonical anchors.

### A. Winner-preservation under bounded score error

In `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/EpsilonUtilityGap.lean`:

- `StrictOpt`
- `StrictUtilityGap`
- `strict_margin_invariant`
- `epsilon_margin_invariance`

The core theorem says: if every action's utility changes by at most `delta`, and `delta < StrictUtilityGap / 2`, then the optimal set does not change. This is already the exact theorem shape needed for certified coarse score replacement.

### B. Coordinate sufficiency and relevance

In `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Sufficiency.lean`:

- `CoordinateSpace`
- `ProductSpace`
- `DecisionProblem.isSufficient`
- `DecisionProblem.isRelevant`
- `DecisionProblem.sufficient_contains_relevant`
- `DecisionProblem.sufficient_erase_irrelevant'`
- `DecisionProblem.minimalSufficient_iff_relevant`

This is the formal engine for saying that some coordinates can be erased without changing `Opt`.

### C. Docking-specific cutoff locality

In `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/MolecularSrank.lean`:

- `MDBindingProblem`
- `OutsideCutoffApproximationBounded`
- `md_relevant_only_if_within_cutoff`
- `md_srank_bound`
- `docking_small_pocket_bound`

This is the theorem chain that makes cutoff locality matter for docking rather than only for generic finite decision problems.

### D. Finite state discretization

In `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DiscretizedState.lean`:

- `BoundedGrid`
- `BoundedGridAtom`
- `GridMDState`
- `liftGridAtom`
- `liftGridState`

This gives the finite state half of sampled docking. The missing half is the finite action side and the actual sampled docking wrapper.

### E. Witness-search templates

In `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/StochasticSequential/OracleUpperBounds.lean` and `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Physics/AnchorChecks.lean`:

- `inP_of_poly_witness_list`
- `stochasticAnchorExplicitVerifier`
- `stochastic_anchor_inP_explicit_via_witness_schema`
- `stochasticAnchorSufficient_iff_exists_anchor_singleton`

These are not pruning theorems, but they provide the right style for explicit finite witness-based certification.

---

## Main Objective

Construct a theorem stack that certifies pruning in sampled docking.

The target end state is a theorem-carrying runtime story:

1. A finite sampled docking problem is built from grid states and sampled actions.
2. A coarse score comes with a proof obligation `UniformUtilityApprox`.
3. A pruning stage produces a `PruningCertificate` stating exact survivor containment, not just empirical correlation.
4. Cutoff and pocket truncation are justified by a sufficiency theorem, not by benchmarking folklore.

---

## Non-Goals

This plan does **not** attempt to certify all of current docking immediately.

In particular, this plan does not begin by certifying:

- Vinardo
- AD4-style H-bonds
- SASA/desolvation
- charge assignment procedures
- pocket-guided sampling heuristics
- arbitrary multi-stage score funnels

Those may come later, but only after the finite sampled core and certified pruning layer exist.

---

## Phase 1: Surprisingly Small/Easy Theorem Engineering

This phase should be treated as mandatory groundwork. It is the shortest path from the existing theorem base to a real certified pruning story.

### 1. Add a finite sampled action type

**Problem**

`GridMDState` exists, but there is no analogous finite sampled action type. As a result, the repository can talk about sampled states but not about a finite sampled docking problem in the exact sense required by `StrictUtilityGap` and `epsilon_margin_invariance`.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/DiscretizedAction.lean`

**Definitions to add**

```lean
@[reducible] def BoundedGrid (N : Nat) := Fin (2 * N + 1)

structure GridMDAction (NL N : Nat) where
  ligandAtoms : (Fin NL) -> BoundedGridAtom N NL

noncomputable def liftGridAction {NL N : Nat}
    (res : R) (ga : GridMDAction NL N) : MDAction :=
  { ligand := ... }
```

`GridMDAction` should mirror the existing finite-state construction instead of inventing a second encoding style. The ligand part of `MDAction` in `MolecularSrank.lean` is already just a list of atoms, so the lift is conceptually direct.

**Proof sketch**

The `Fintype` proof should follow the exact shape used for `GridMDState` in `DiscretizedState.lean`. The proof is not hard because the type is definitionally a finite function space over finite coordinates.

1. Build an equivalence from `GridMDAction NL N` to `((Fin NL) -> BoundedGridAtom N NL)`.
2. Reuse the fact that function spaces over finite domains and codomains are finite.
3. Apply `Fintype.ofEquiv`.

The Lean proof should look structurally like:

```lean
instance (NL N : Nat) : Fintype (GridMDAction NL N) := by
  let E : GridMDAction NL N ≃ ((Fin NL) -> BoundedGridAtom N NL) :=
    { toFun := fun a => a.ligandAtoms
      invFun := fun f => { ligandAtoms := f }
      left_inv := by intro a; cases a; rfl
      right_inv := by intro f; rfl }
  exact Fintype.ofEquiv _ E.symm
```

This proof is simple because all structure is already finite by construction.

**Why this unlocks leverage**

`EpsilonUtilityGap.lean` requires `[Fintype A]`. Without this action type, the winner-preservation theorem cannot be applied to sampled docking.

**Acceptance criterion**

`lake build` succeeds and a toy sampled docking instance can instantiate `[Fintype (GridMDAction NL N)]` without any axioms.

---

### 2. Add `CoordinateSpace` and `ProductSpace` for `GridMDState`

**Problem**

`GridMDState` is finite, but it is not yet wired into the `CoordinateSpace`/`ProductSpace` abstraction from `Sufficiency.lean`. Until this exists, the strongest sufficiency theorem `sufficient_erase_irrelevant'` cannot be applied to sampled docking states.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/GridMDInstances.lean`

**Definitions to add**

We need a coordinate indexing scheme for `GridMDState NP NL N` with coordinate count `3 * NP + 3 * NL`.

The projection function should map coordinate `i` to the corresponding grid coordinate of the appropriate protein or ligand atom.

```lean
def gridMDStateProj (s : GridMDState NP NL N)
    (i : Fin (3 * NP + 3 * NL)) : BoundedGrid N :=
  ...

instance : CoordinateSpace (GridMDState NP NL N) (3 * NP + 3 * NL) where
  Coord := fun _ => BoundedGrid N
  proj := gridMDStateProj
```

Then define a replacement operator that changes only one coordinate.

```lean
def gridMDStateReplace
    (s : GridMDState NP NL N)
    (i : Fin (3 * NP + 3 * NL))
    (s' : GridMDState NP NL N) : GridMDState NP NL N :=
  ...

instance : ProductSpace (GridMDState NP NL N) (3 * NP + 3 * NL) where
  replace := gridMDStateReplace
  replace_proj_eq := ...
  replace_proj_ne := ...
```

**Proof sketch**

The key is to avoid proving anything geometric. This is purely combinatorial state surgery.

For `replace_proj_eq`:

1. Split on whether the coordinate index points into protein or ligand coordinates.
2. Split again on the axis `x`, `y`, or `z`.
3. By construction, the updated state copies the target coordinate from `s'`.
4. The projection of the replaced state at the replaced coordinate reduces by `rfl` after the case split.

For `replace_proj_ne`:

1. Use the same coordinate decomposition.
2. Show that if `j != i`, then either the atom index differs or the axis differs.
3. In both cases, the replacement function leaves that component unchanged.
4. Therefore the projection agrees with the original state.

This is exactly the sort of proof `ProductSpace` was designed to support.

**Why this unlocks leverage**

Once this exists, the theorem

```lean
DecisionProblem.sufficient_erase_irrelevant'
```

from `Sufficiency.lean` becomes available for finite grid docking states. That is the bridge from local relevance proofs to actual coordinate erasure.

**Acceptance criterion**

Prove a tiny lemma showing that for any irrelevant coordinate `i`, erasing it from a sufficient set preserves sufficiency on `GridMDState` by direct application of `sufficient_erase_irrelevant'`.

---

### 3. Add a sampled docking problem wrapper

**Problem**

The benchmark pipeline informally does: sample poses, evaluate them, rank them. The Lean side currently has no single object representing that finite sampled decision problem.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDocking.lean`

**Definitions to add**

```lean
structure SampledActionFamily (A : Type*) [DecidableEq A] where
  actions : Finset A
  nonempty : actions.Nonempty

def restrictedDecisionProblem
    (dp : DecisionProblem A S)
    (F : SampledActionFamily A) : DecisionProblem A S :=
  { utility := fun a s => if a ∈ F.actions then dp.utility a s else 0 }
```

The exact details of the off-support utility do not matter if we instead define `Opt` over the sampled support only, but the cleanest OpenHCS choice is to define a restricted decision problem whose action domain is the sampled finite type itself rather than reuse the ambient action type. That means the better formulation is:

```lean
structure SampledAction (A : Type*) where
  val : A
  mem_support : val ∈ F.actions
```

and then define a `DecisionProblem (SampledAction A) S`.

**Proof sketch**

There are two important correctness obligations.

First, prove the action domain is finite.

Second, prove that the restricted `Opt` set coincides with the ambient optimum restricted to the sampled support. The proof is conceptually direct:

1. Expand `DecisionProblem.Opt`.
2. For the restricted action type, every action is by construction a sampled action.
3. Utility comparison inside the restricted problem is definitionally the same as utility comparison in the ambient problem on those sampled actions.
4. Therefore `Opt_restricted s` is exactly the sampled slice of `Opt_ambient s`.

This theorem is crucial because it turns an implementation detail into a formal object.

**Acceptance criterion**

Prove a theorem of the form:

```lean
theorem restricted_opt_eq_sampled_slice ...
```

stating that the restricted optimum equals the ambient optimum intersected with the sampled support.

---

### 4. Add a wrapper theorem instantiating `epsilon_margin_invariance`

**Problem**

The repository already has the right winner-preservation theorem, but not one stated directly for exact sampled docking versus coarse sampled docking.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingGap.lean`

**Theorem to add**

```lean
theorem sampled_epsilon_margin_invariance
    (exactDP coarseDP : DecisionProblem A S)
    [Fintype A]
    (s : S) (a_star : A)
    (delta : R)
    (hdelta : 0 <= delta)
    (hStrict : StrictOpt exactDP a_star s)
    (hPerturb : ∀ a, |exactDP.utility a s - coarseDP.utility a s| <= delta)
    (hBound : delta < StrictUtilityGap exactDP a_star s / 2) :
    exactDP.Opt s = coarseDP.Opt s :=
  ...
```

**Proof sketch**

This theorem should be almost trivial after definitions are aligned.

1. Package `exactDP` and `coarseDP` as the two states of the same underlying action domain.
2. Use `s' = s` at the state level and absorb the score change into the different utility functions, or define an auxiliary decision problem on the product `(score_family, state)`.
3. Reduce the theorem to `epsilon_margin_invariance` in `EpsilonUtilityGap.lean`.

If the theorem is hard to state directly because `epsilon_margin_invariance` varies the state rather than the utility function, then define a lifted state space:

```lean
inductive ScoreFamily | exact | coarse

def liftedState := ScoreFamily × S
```

and a lifted decision problem:

```lean
def liftedDP : DecisionProblem A (ScoreFamily × S)
```

with utility switching on `ScoreFamily`. Then apply the existing theorem between `(exact, s)` and `(coarse, s)`.

That proof is clean, explicit, and avoids modifying existing foundational theorems.

**Acceptance criterion**

The theorem compiles using only existing `EpsilonUtilityGap` machinery and no new axioms.

---

### 5. Add finite top-k definitions and threshold objects

**Problem**

The current theorem base talks about winners and optimal sets. Pruning needs top-k containment and threshold-gap arguments.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/FiniteTopK.lean`

**Definitions to add**

```lean
def topKList (u : A -> R) (k : Nat) : List A := ...
def topKSet (u : A -> R) (k : Nat) : Finset A := ...
def kthUtility (u : A -> R) (k : Nat) : R := ...
def survivorSet (u : A -> R) (tau : R) : Finset A := ...
```

The plan must be explicit about ties. There are only two honest choices.

1. Add a deterministic tiebreak order on `A` and define exact top-k.
2. Define a threshold-based survivor set and prove containment statements instead of exact equality.

For pruning, the second choice is usually the right first target because it avoids fake precision.

**Proof sketch**

The first useful theorem is not exact top-k equality. It is the threshold fact:

```lean
theorem not_mem_topK_of_strictly_below_kth ...
```

The proof strategy is finite combinatorics.

1. Let `tau = kthUtility u k`.
2. If `u a < tau`, then there are already at least `k` actions with utility strictly above `u a`.
3. Therefore `a` cannot belong to any deterministic top-k consistent with the declared tie policy.

The second useful theorem is:

```lean
theorem topK_subset_of_survivorSet ...
```

showing that every exact top-k action survives any threshold at or below the kth utility.

**Acceptance criterion**

There is a formal `topK` object, a formal `kthUtility`, and at least two basic threshold lemmas ready for later ranking-preservation proofs.

---

## Phase 2: Medium Work That Converts Winner Preservation Into Pruning Certificates

### 6. Turn winner-preservation into pairwise ranking-preservation

**Problem**

Winner preservation is not enough for stage pruning. If stage 1 keeps only the top few hundred actions, we need to know that actions below the boundary stay below the boundary.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/RankingPreservation.lean`

**Core definitions**

```lean
def PairwiseGap (u : A -> R) (a b : A) : R := u a - u b
def BoundaryGap (u : A -> R) (k : Nat) : R := ...
```

**Core theorem**

```lean
theorem pairwise_order_preserved_of_uniform_error
    (hApprox : ∀ a, |u_exact a - u_coarse a| <= delta)
    (hGap : u_exact a - u_exact b > 2 * delta) :
    u_coarse a > u_coarse b :=
  ...
```

**Proof sketch**

This is a one-line inequality argument once stated correctly.

From the approximation premise:

- `u_coarse a >= u_exact a - delta`
- `u_coarse b <= u_exact b + delta`

Subtract to get:

- `u_coarse a - u_coarse b >= (u_exact a - u_exact b) - 2 * delta`

If the exact pairwise gap is larger than `2 * delta`, the right-hand side is positive, so the coarse ordering is preserved.

This theorem is the ranking analogue of `epsilon_margin_invariance`.

**Acceptance criterion**

The theorem is proved for finite actions with no physical assumptions. It becomes the base lemma for all top-k pruning certificates.

---

### 7. Add threshold-gap lemmas for safe top-k pruning

**Problem**

A pruning stage does not need exact top-k equality. It needs the stronger safety statement that every exact top-k action survives the coarse filter.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/TopKPreservation.lean`

**Theorems to add**

```lean
theorem coarse_survivor_contains_exact_topK ...
theorem topK_preserved_of_boundary_gap ...
theorem action_excluded_if_below_boundary_by_more_than_2delta ...
```

**Proof sketch**

Fix exact score `u_exact`, coarse score `u_coarse`, and boundary action `b_k` representing the kth exact action under the declared top-k semantics.

To prove survivor containment:

1. Let `a` be any exact top-k action.
2. By definition, `u_exact a >= u_exact b_k`.
3. Under uniform approximation, `u_coarse a >= u_exact a - delta` and `u_coarse b_k <= u_exact b_k + delta`.
4. If the boundary gap assumptions hold, every exact top-k action remains above the coarse threshold used to define survivors.
5. Therefore `a` belongs to the coarse survivor set.

To prove exclusion:

1. Let `c` be an action whose exact utility lies more than `2 * delta` below the exact kth boundary.
2. Pairwise order preservation implies `c` stays below the coarse boundary.
3. Therefore `c` may be safely pruned.

This is the first theorem family that directly justifies stage filtering.

**Acceptance criterion**

There exists a theorem proving set containment `exactTopK ⊆ coarseSurvivors` under a uniform approximation hypothesis and an explicit boundary-gap hypothesis.

---

### 8. Prove a uniform coarse-score error bound over the sampled grid

**Problem**

So far the approximation theorem is only conditional. We still need at least one real scorer pair to satisfy the premise.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CoarseApproximation.lean`

**First target**

Do **not** start with composite docking scores. Start with a simple pair such as:

- exact standard LJ
- cutoff LJ, or
- exact standard LJ versus softened standard LJ on a bounded finite grid

**Definitions to add**

```lean
def UniformUtilityApprox
    (u_exact u_coarse : A -> S -> R) (delta : R) : Prop :=
  ∀ a s, |u_exact a s - u_coarse a s| <= delta
```

**Proof sketch**

The exact proof depends on the chosen pair. For standard LJ versus cutoff LJ, the argument should be:

1. Decompose total utility into inside-cutoff and outside-cutoff terms.
2. Inside-cutoff terms cancel exactly.
3. Outside-cutoff terms are bounded by a tail estimate.
4. Sum the tail bound over all omitted pairs.
5. That finite sum yields a uniform `delta` over the bounded sampled domain.

This is the finite, sampled version of the physical story currently abstracted by `SatisfiesBoundedPotential`.

**Acceptance criterion**

One nontrivial scorer pair gets a proved `UniformUtilityApprox` theorem over a bounded sampled domain.

---

## Phase 3: Medium-to-Hard Work That Reconnects Sampled Docking To Cutoff Locality

### 9. Transfer `OutsideCutoffApproximationBounded` into the sampled problem

**Problem**

`MolecularSrank.lean` already tells us that outside-cutoff coordinates are irrelevant if `OutsideCutoffApproximationBounded` holds. But the theorem is written for the molecular binding problem, not yet for the explicit finite sampled docking wrapper.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/SampledDockingCutoff.lean`

**Theorem target**

```lean
theorem sampled_outside_cutoff_coordinates_irrelevant
    (hBounded : OutsideCutoffApproximationBounded prob)
    ... :
    ∀ i, outsideCutoffCoordinate i ->
      sampledDP.toDecisionProblem.isIrrelevant i
```

**Proof sketch**

This should be a direct transport theorem.

1. Show the sampled docking wrapper preserves the underlying coordinate projection relevant to `MolecularSrank`.
2. Invoke `md_relevant_only_if_within_cutoff` by contraposition.
3. Conclude that any coordinate outside the cutoff is irrelevant in the sampled problem.
4. If `GridMDState` has the proper `ProductSpace` instance, apply `sufficient_erase_irrelevant'` to derive a retained-coordinate sufficiency theorem.

This is the exact theorem that turns pocket truncation from a modeling convenience into a proof-backed optimization.

**Acceptance criterion**

There is a theorem stating that the retained inside-cutoff coordinate set is sufficient for the sampled docking problem under the cutoff boundedness premise.

---

### 10. Introduce an explicit `PruningCertificate` abstraction

**Problem**

Even after the theorems exist, the system still needs a formal object describing what has been certified.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CertifiedPruning.lean`

**Definition sketch**

```lean
structure PruningCertificate (A S : Type*) where
  survivors : Finset A
  exactTopK : Finset A
  claim : exactTopK ⊆ survivors
  assumptions : Prop
```

The exact design may separate theorem-side certificates from runtime-side metadata, but the theorem should produce a concrete survivor-containment statement, not just a prose note.

**Proof sketch**

This file is mostly composition.

1. Import `FiniteTopK`, `RankingPreservation`, and `TopKPreservation`.
2. Define `survivors` from the coarse threshold rule.
3. Instantiate the containment theorem from Phase 2.
4. Package the result into `PruningCertificate`.

This gives the codebase a clean type-level distinction between certified and heuristic pruning.

**Acceptance criterion**

At least one theorem produces a `PruningCertificate` for sampled docking with a simple scorer pair.

---

## Phase 4: Big Research Projects

These are real research obligations. They should not block the earlier finite sampled core, but they are necessary if the long-term goal is full physical grounding.

### 11. Derive realistic chemistry-side bounds for actual docking scorers

**Problem**

`SatisfiesBoundedPotential` in `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/CutoffEpsilon.lean` and `OutsideCutoffApproximationBounded` in `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/MolecularSrank.lean` are still assumption schemas. They are not yet proved for the actual score families used in docking.

**Immediate priority order**

1. Standard LJ with cutoff
2. Coulomb with truncated or Ewald-bounded tail
3. Only later, richer chemistry terms

**Why this is hard**

The proof is no longer just finite combinatorics. It requires analytic bounds on omitted or approximated interaction terms, and explicit constants that survive summation over all relevant pairs.

**Proof sketch for LJ cutoff**

1. Separate pair interactions into inside-cutoff and outside-cutoff sets.
2. Inside-cutoff terms are exact.
3. Bound every omitted pair by the tail estimate already formalized or assumed in `LatticeSum.lean`.
4. Sum over all omitted pairs to obtain a global tail bound `delta`.
5. Feed that `delta` into `large_cutoff_implies_bounded` from `CutoffEpsilon.lean`.

**Proof sketch for Coulomb tail**

The Coulomb case is harder because the tail is longer-ranged and may require bounded charge and bounded geometry assumptions. The likely route is:

1. Introduce a bounded-charge hypothesis.
2. Split short-range and long-range terms.
3. Bound long-range terms either via direct geometric summation or via an Ewald decomposition theorem.
4. Show the resulting bound is below half the strict utility gap.

**Acceptance criterion**

Produce one fully instantiated theorem for a nontrivial docking-relevant scorer family. Standard LJ with cutoff is the correct first target.

---

### 12. Certify ranking stability in the presence of ties or near-ties

**Problem**

The current theory assumes strict gaps. Real docking often has dense bands of near-equivalent scores.

**Research direction**

Instead of insisting on exact top-k equality, define an ambiguity band.

```lean
def NearTieBand (u : A -> R) (k : Nat) (eps : R) : Finset A := ...
```

Then prove statements of the form:

```lean
theorem exact_topK_inside_certified_band ...
```

meaning the exact top-k lies inside a larger, certified ambiguity envelope whenever strict gap assumptions fail.

**Proof sketch**

1. Define the exact kth boundary value.
2. Include every action whose utility lies within `eps` of that boundary.
3. Use the same pairwise approximation inequalities as in Phase 2.
4. Conclude that any top-k instability is confined to the ambiguity band.

This is the right formal replacement for pretending noisy or near-tied rankings are exact.

**Acceptance criterion**

There is a theorem giving certified survivor containment with explicit ambiguity bands when strict gaps are unavailable.

---

### 13. Remove the analytic axiom in `LatticeSum.lean`

**Problem**

`lattice_sum_converges` in `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/LatticeSum.lean` is still axiomatic.

**Why this matters**

As long as the tail-convergence core remains axiomatic, the cutoff certification chain is conditional at a mathematically serious point.

**Proof sketch**

The route is likely classical real analysis:

1. Compare the lattice sum to a convergent integral.
2. Establish monotonicity and positivity of the relevant tail terms.
3. Prove convergence for exponents greater than the spatial dimension.
4. Extract an explicit constant `M` rather than only existence.

This is a real theorem-proving project and should be treated as such.

**Acceptance criterion**

The axiom is replaced by a proved theorem strong enough to recover the existing tail bounds used by `CutoffEpsilon.lean`.

---

### 14. Prove continuous-to-discrete convergence

**Problem**

`DiscretizedState.lean` gives the finite grid objects, but not the theorem connecting resolution to utility error.

**Proposed file**

`docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/GridConvergence.lean`

**Theorem target**

```lean
theorem grid_resolution_implies_uniform_utility_error
    (hSmooth : ...)
    (hDomain : ...)
    (hRes : ... ) :
    UniformUtilityApprox U_continuous U_grid (eps res)
```

**Proof sketch**

The likely route is Lipschitz or bounded-gradient control.

1. Assume a bounded domain and a uniform bound on the gradient norm of the utility over that domain.
2. Show that each lifted grid point lies within `O(res)` of its continuous counterpart.
3. Apply the mean value inequality or Lipschitz continuity to bound utility perturbation by `L * O(res)`.
4. Sum across relevant coordinates if necessary.

This theorem is what lets the finite sampled problem stand in for continuous docking without handwaving.

**Acceptance criterion**

The theorem supplies an explicit `eps(res)` usable by Phase 2 approximation theorems.

---

## Integration Into The OpenHCS Runtime Story

The theorem work only matters if the runtime path reflects the proof status.

### Runtime status taxonomy

Every pruning stage in the runtime system should be labeled as one of:

- `Certified`
- `ConditionallyCertified`
- `Heuristic`

`Certified` means all theorem premises are fully discharged in Lean.

`ConditionallyCertified` means the stage is backed by a theorem with explicit unresolved physical hypotheses such as `SatisfiesBoundedPotential`.

`Heuristic` means there is no theorem yet and the stage remains experimental.

This taxonomy must become visible in the benchmark and configuration layer. It is the only honest way to prevent theorem-backed and heuristic pruning from being silently conflated.

### Runtime integration order

1. Add a simple theorem-backed pruning path for sampled LJ docking.
2. Benchmark that path separately from heuristic composite scoring.
3. Only then decide whether more sophisticated scorers can be wrapped by certified approximation theorems.

---

## Proof Dependency DAG

The concrete dependency order is:

1. `DiscretizedAction.lean`
2. `GridMDInstances.lean`
3. `SampledDocking.lean`
4. `SampledDockingGap.lean`
5. `FiniteTopK.lean`
6. `RankingPreservation.lean`
7. `TopKPreservation.lean`
8. `CoarseApproximation.lean`
9. `SampledDockingCutoff.lean`
10. `CertifiedPruning.lean`
11. `GridConvergence.lean`
12. chemistry-specific bound files
13. lattice-sum de-axiomatization

Nothing in steps 1 through 7 should depend on unresolved physical analysis. That is why they should be done first.

---

## Acceptance Gates

### Gate 1: Finite sampled docking exists formally

This gate is passed when:

- `GridMDAction` exists and has `Fintype`
- `GridMDState` has `CoordinateSpace` and `ProductSpace`
- a sampled docking wrapper exists as a finite `DecisionProblem`

### Gate 2: Certified winner preservation exists

This gate is passed when:

- `sampled_epsilon_margin_invariance` is proved
- at least one sampled exact/coarse score pair instantiates it

### Gate 3: Certified pruning exists

This gate is passed when:

- `FiniteTopK` and threshold-gap lemmas exist
- there is a theorem proving `exactTopK ⊆ coarseSurvivors`
- the runtime can attach a `PruningCertificate`

### Gate 4: Physical locality is connected

This gate is passed when:

- the sampled problem inherits cutoff irrelevance from `MolecularSrank`
- retained inside-cutoff coordinates are proved sufficient

### Gate 5: Physical analysis is no longer abstract

This gate is passed when:

- at least one real scorer family discharges `SatisfiesBoundedPotential`
- continuous-to-discrete error is explicit
- the lattice-tail axiom is either removed or isolated behind a clearly marked conditional layer

---

## What This Plan Deliberately Avoids

This plan rejects the following anti-patterns.

1. Adding new pruning heuristics before top-k certification exists.
2. Treating empirical rank correlation as a proof surrogate.
3. Pretending a custom docking score is Lean-grounded when only standard LJ or cutoff locality is actually formalized.
4. Mixing exact and approximate paths without an explicit approximation theorem.
5. Using composite chemistry as the first mechanization target.

---

## Final Recommendation

The shortest path to trustworthy pruning is not to keep refining heuristics. It is to formalize sampled docking itself as a finite decision problem and then stack exact theorem obligations on top of it.

The first implementation milestone should therefore be:

1. finite sampled action type,
2. `GridMDState` instances,
3. sampled docking wrapper,
4. winner-preservation wrapper theorem,
5. finite top-k and pruning-containment theorems.

Once those exist, the codebase can distinguish between pruning that is certified, pruning that is conditionally certified, and pruning that is still heuristic. That is the correct OpenHCS boundary.
