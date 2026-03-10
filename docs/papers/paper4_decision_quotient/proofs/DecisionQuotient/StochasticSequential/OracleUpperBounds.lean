/-
  Paper 4b: Stochastic and Sequential Regimes

  OracleUpperBounds.lean - Scoped witness-over-verifier upper-bound schemas

  This file does NOT formalize full oracle Turing-machine classes such as
  `NP^PP`. Instead, it packages the exact structural pattern used by the paper's
  upper-bound arguments:

  - an existential witness layer,
  - over a verifier relation that already has an exact boolean decider in the
    artifact,
  - matching the paper-level view that stochastic anchor/minimum add
    existential search on top of PP-style stochastic verification.

  The resulting interface is intentionally lightweight and honest: it captures
  the witness-over-verifier shape needed by the paper without claiming a fully
  mechanized oracle-machine complexity library.
-/

import DecisionQuotient.StochasticSequential.Computation
import DecisionQuotient.Physics.AnchorChecks
import DecisionQuotient.ExplicitStateMembership
import Mathlib.Tactic

namespace DecisionQuotient
namespace StochasticSequential

open Classical

/-- A verifier relation whose instances already have exact boolean deciders in
the current artifact. This is the formal interface used by the paper's scoped
oracle-style upper-bound arguments. -/
structure PPStyleVerifier (α ω : Type*) where
  rel : α → ω → Prop
  exact_bool : ∀ x : α, ∀ w : ω, HasExactBoolDecider (rel x w)

/-- Lightweight surrogate for the witness-over-PP pattern underlying the
paper-level `NP^PP` upper bounds.

This is intentionally weaker than a full oracle-TM class formalization: it
records that a language is decided by existence of a witness accepted by a
verifier relation that already has exact boolean deciders in the artifact. -/
def FitsNPOverPPStyle (α ω : Type*) (p : α → Prop) (R : α → ω → Prop) : Prop :=
  (∀ x : α, ∀ w : ω, HasExactBoolDecider (R x w)) ∧
  (∀ x : α, p x ↔ ∃ w : ω, R x w)

/-- Generic constructor: any witness characterization over a verifier relation
with exact boolean deciders fits the scoped `NP-over-PP` schema. -/
theorem fitsNPOverPPStyle_of_exists_witness
    {α ω : Type*} {p : α → Prop} (R : α → ω → Prop)
    (hdec : ∀ x : α, ∀ w : ω, HasExactBoolDecider (R x w))
    (hspec : ∀ x : α, p x ↔ ∃ w : ω, R x w) :
    FitsNPOverPPStyle α ω p R :=
  ⟨hdec, hspec⟩

/-- Any decidable verifier relation is automatically a PP-style verifier in the
artifact's exact-decider sense. -/
theorem hasExactBoolDecider_of_decidable (Q : Prop) [Decidable Q] :
    HasExactBoolDecider Q := by
  refine ⟨decide Q, ?_⟩
  by_cases h : Q <;> simp [h]

/-- Input package for the stochastic anchor query family. -/
structure StochasticAnchorQueryInput (A S : Type*) (n : ℕ)
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] where
  problem : StochasticDecisionProblem A S
  infoSet : Finset (Fin n)

/-- Witness relation for stochastic anchor sufficiency: one anchor state and one
action certify singleton conditional optimality on the anchor fiber. -/
def stochasticAnchorWitnessRel
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticAnchorQueryInput A S n) (wa : S × A) : Prop :=
  fiberOpt q.problem q.infoSet wa.1 = ({wa.2} : Set A)

/-- The stochastic anchor witness relation has exact boolean deciders inside the
current finite artifact. -/
theorem stochasticAnchorWitnessRel_exact
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticAnchorQueryInput A S n) (wa : S × A) :
    HasExactBoolDecider (stochasticAnchorWitnessRel q wa) := by
  classical
  exact hasExactBoolDecider_of_decidable _

/-- The stochastic anchor query fits the scoped `InNPOverPPStyle` schema.

The key collapse theorem is mechanized in `Physics.AnchorChecks`: anchor
sufficiency is equivalent to existence of one anchor fiber with singleton
conditional optimum. -/
theorem stochastic_anchor_query_fits_np_over_ppstyle
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    FitsNPOverPPStyle (StochasticAnchorQueryInput A S n) (S × A)
      (fun q => StochasticAnchorSufficiencyCheck q.problem q.infoSet)
      (fun q wa => stochasticAnchorWitnessRel q wa) := by
  refine fitsNPOverPPStyle_of_exists_witness
    (fun q wa => stochasticAnchorWitnessRel q wa)
    (fun q wa => stochasticAnchorWitnessRel_exact q wa) ?_
  intro q
  constructor
  · intro h
    rw [Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton
      (P := q.problem) (I := q.infoSet)] at h
    rcases h with ⟨s₀, a, ha⟩
    exact ⟨(s₀, a), ha⟩
  · intro h
    rcases h with ⟨⟨s₀, a⟩, ha⟩
    rw [Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton
      (P := q.problem) (I := q.infoSet)]
    exact ⟨s₀, a, ha⟩

/-- Input package for the stochastic minimum query family. -/
structure StochasticMinimumQueryInput (A S : Type*) (n : ℕ)
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] where
  problem : StochasticDecisionProblem A S
  bound : ℕ

/-- Witness relation for stochastic minimum sufficiency: a coordinate subset of
size at most the declared bound that is stochastically sufficient. -/
def stochasticMinimumWitnessRel
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticMinimumQueryInput A S n) (I : Finset (Fin n)) : Prop :=
  I.card ≤ q.bound ∧ StochasticSufficient q.problem I

/-- The stochastic minimum witness relation has exact boolean deciders inside the
current finite artifact. -/
theorem stochasticMinimumWitnessRel_exact
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticMinimumQueryInput A S n) (I : Finset (Fin n)) :
    HasExactBoolDecider (stochasticMinimumWitnessRel q I) := by
  classical
  exact hasExactBoolDecider_of_decidable _

/-- The stochastic minimum query fits the scoped `InNPOverPPStyle` schema. -/
theorem stochastic_minimum_query_fits_np_over_ppstyle
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    FitsNPOverPPStyle (StochasticMinimumQueryInput A S n) (Finset (Fin n))
      (fun q => StochasticMinimumSufficiencyCheck q.problem q.bound)
      (fun q I => stochasticMinimumWitnessRel q I) := by
  refine fitsNPOverPPStyle_of_exists_witness
    (fun q I => stochasticMinimumWitnessRel q I)
    (fun q I => stochasticMinimumWitnessRel_exact q I) ?_
  intro q
  rfl

/-- Summary theorem: both stochastic existential variants fit the scoped
`NP-over-PP` witness schema used by the paper-level upper-bound arguments. -/
theorem stochastic_existential_queries_fit_np_over_ppstyle
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    FitsNPOverPPStyle (StochasticAnchorQueryInput A S n) (S × A)
      (fun q => StochasticAnchorSufficiencyCheck q.problem q.infoSet)
      (fun q wa => stochasticAnchorWitnessRel q wa) ∧
    FitsNPOverPPStyle (StochasticMinimumQueryInput A S n) (Finset (Fin n))
      (fun q => StochasticMinimumSufficiencyCheck q.problem q.bound)
      (fun q I => stochasticMinimumWitnessRel q I) := by
  exact ⟨stochastic_anchor_query_fits_np_over_ppstyle,
    stochastic_minimum_query_fits_np_over_ppstyle⟩

/-! ## Stronger bounded-witness search layer (explicit-state side) -/

/-- A finite witness-list verifier with exact boolean deciders. -/
structure ExactWitnessListVerifier (α ω : Type*) where
  witnesses : α → List ω
  rel : α → ω → Prop
  exact_bool : ∀ x : α, ∀ w : ω, HasExactBoolDecider (rel x w)

/-- Counted search over a finite witness list. -/
noncomputable def countedWitnessListSearch {α ω : Type*}
    (V : ExactWitnessListVerifier α ω) (x : α) : Counted Bool :=
  countedAnyList (V.witnesses x) (fun w => decide (V.rel x w))

theorem countedWitnessListSearch_spec {α ω : Type*}
    (V : ExactWitnessListVerifier α ω) (x : α) :
    (countedWitnessListSearch V x).result = true ↔
      ∃ w ∈ V.witnesses x, V.rel x w := by
  classical
  unfold countedWitnessListSearch
  simpa using countedAnyList_result_true_iff (V.witnesses x) (fun w => decide (V.rel x w))

theorem countedWitnessListSearch_steps_le {α ω : Type*}
    (V : ExactWitnessListVerifier α ω) (x : α) :
    (countedWitnessListSearch V x).steps ≤ (V.witnesses x).length := by
  unfold countedWitnessListSearch
  exact countedAnyList_steps_le_length _ _

/-- Generic abstract-`P` theorem from a polynomially bounded witness list. -/
theorem inP_of_poly_witness_list
    {α ω : Type*} [SizeOf α] (p : α → Prop)
    (V : ExactWitnessListVerifier α ω)
    (hbound : ∃ c k : ℕ, ∀ x : α, (V.witnesses x).length ≤ c * (sizeOf x) ^ k + c)
    (hspec : ∀ x : α, p x ↔ ∃ w ∈ V.witnesses x, V.rel x w) :
    InP p := by
  classical
  rcases hbound with ⟨c, k, hlen⟩
  use (fun x => countedWitnessListSearch V x), c, k
  constructor
  · intro x
    calc
      (countedWitnessListSearch V x).steps ≤ (V.witnesses x).length := countedWitnessListSearch_steps_le V x
      _ ≤ c * (sizeOf x) ^ k + c := hlen x
  · intro x
    rw [countedWitnessListSearch_spec]
    exact (hspec x).symm

/-- Explicit-state witness list for stochastic anchor search: all state/action
pairs. -/
noncomputable def stochasticAnchorExplicitVerifier
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    ExactWitnessListVerifier (StochasticExplicitInput A S n) (S × A) where
  witnesses q := (Finset.univ.product (Finset.univ : Finset A)).toList
  rel q wa := stochasticAnchorWitnessRel ⟨q.problem, q.infoSet⟩ wa
  exact_bool q wa := stochasticAnchorWitnessRel_exact ⟨q.problem, q.infoSet⟩ wa

theorem stochasticAnchorExplicit_witnesses_poly
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    ∃ c k : ℕ, ∀ q : StochasticExplicitInput A S n,
      (stochasticAnchorExplicitVerifier.witnesses q).length ≤ c * (sizeOf q) ^ k + c := by
  refine ⟨1, 2, ?_⟩
  intro q
  calc
    (stochasticAnchorExplicitVerifier.witnesses q).length
        = Fintype.card S * Fintype.card A := by
            simp [stochasticAnchorExplicitVerifier]
    _ ≤ q.stateBudget * q.actionBudget := Nat.mul_le_mul q.state_bound q.action_bound
    _ ≤ (sizeOf q) * (sizeOf q) := Nat.mul_le_mul (stochasticExplicit_size_ge_state q) (stochasticExplicit_size_ge_action q)
    _ = (sizeOf q) ^ 2 := by simp [pow_two]
    _ ≤ 1 * (sizeOf q) ^ 2 + 1 := by omega

theorem stochasticAnchorExplicit_witness_spec
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticExplicitInput A S n) :
    StochasticAnchorSufficiencyCheck q.problem q.infoSet ↔
      ∃ wa ∈ stochasticAnchorExplicitVerifier.witnesses q,
        stochasticAnchorExplicitVerifier.rel q wa := by
  constructor
  · intro h
    rw [Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton
      (P := q.problem) (I := q.infoSet)] at h
    rcases h with ⟨s₀, a, ha⟩
    refine ⟨(s₀, a), ?_, ha⟩
    simp [stochasticAnchorExplicitVerifier]
  · intro h
    rcases h with ⟨⟨s₀, a⟩, _, ha⟩
    rw [Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton
      (P := q.problem) (I := q.infoSet)]
    exact ⟨s₀, a, ha⟩

/-- The explicit-state stochastic anchor query is recovered from the generic
bounded-witness schema. -/
theorem stochastic_anchor_inP_explicit_via_witness_schema
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : StochasticExplicitInput A S n => StochasticAnchorSufficiencyCheck q.problem q.infoSet) := by
  refine inP_of_poly_witness_list
    (fun q : StochasticExplicitInput A S n => StochasticAnchorSufficiencyCheck q.problem q.infoSet)
    stochasticAnchorExplicitVerifier
    stochasticAnchorExplicit_witnesses_poly ?_
  intro q
  exact stochasticAnchorExplicit_witness_spec q

/-- Explicit-state witness list for stochastic minimum search: all coordinate
subsets. -/
noncomputable def stochasticMinimumExplicitVerifier
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    ExactWitnessListVerifier (StochasticMinimumExplicitInput A S n) (Finset (Fin n)) where
  witnesses _ := (Finset.powerset (Finset.univ : Finset (Fin n))).toList
  rel q I := stochasticMinimumWitnessRel ⟨q.problem, q.bound⟩ I
  exact_bool q I := stochasticMinimumWitnessRel_exact ⟨q.problem, q.bound⟩ I

theorem stochasticMinimumExplicit_witnesses_poly
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    ∃ c k : ℕ, ∀ q : StochasticMinimumExplicitInput A S n,
      (stochasticMinimumExplicitVerifier.witnesses q).length ≤ c * (sizeOf q) ^ k + c := by
  refine ⟨1, 1, ?_⟩
  intro q
  calc
    (stochasticMinimumExplicitVerifier.witnesses q).length = 2 ^ n := by
      simp [stochasticMinimumExplicitVerifier]
    _ ≤ q.subsetBudget := q.subset_bound
    _ ≤ sizeOf q := stochasticMinimumExplicit_size_ge_subset q
    _ ≤ 1 * (sizeOf q) ^ 1 + 1 := by simp

theorem stochasticMinimumExplicit_witness_spec
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n]
    (q : StochasticMinimumExplicitInput A S n) :
    StochasticMinimumSufficiencyCheck q.problem q.bound ↔
      ∃ I ∈ stochasticMinimumExplicitVerifier.witnesses q,
        stochasticMinimumExplicitVerifier.rel q I := by
  constructor
  · intro h
    rcases h with ⟨I, hcard, hsuff⟩
    refine ⟨I, ?_, hcard, hsuff⟩
    simp [stochasticMinimumExplicitVerifier]
  · intro h
    rcases h with ⟨I, _, hcard, hsuff⟩
    exact ⟨I, hcard, hsuff⟩

/-- The explicit-state stochastic minimum query is recovered from the generic
bounded-witness schema. -/
theorem stochastic_minimum_inP_explicit_via_witness_schema
    {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] :
    InP (fun q : StochasticMinimumExplicitInput A S n =>
      StochasticMinimumSufficiencyCheck q.problem q.bound) := by
  refine inP_of_poly_witness_list
    (fun q : StochasticMinimumExplicitInput A S n => StochasticMinimumSufficiencyCheck q.problem q.bound)
    stochasticMinimumExplicitVerifier
    stochasticMinimumExplicit_witnesses_poly ?_
  intro q
  exact stochasticMinimumExplicit_witness_spec q

end StochasticSequential
end DecisionQuotient
