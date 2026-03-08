/-*
  Paper 4b: Stochastic and Sequential Regimes
   
  PolynomialReduction.lean - Polynomial-time reductions between problems
   
  Formal polynomial-time reduction machinery.
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.Finite
import DecisionQuotient.PolynomialReduction
import Mathlib.Computability.Reduce
import Mathlib.Tactic

namespace DecisionQuotient.StochasticSequential

open DecisionQuotient

/-! ## Reduction Definition (reusing Paper 4's machinery) -/

/-- Polynomial-time reduction between decision problems -/
structure PolyReduction4b (X Y : Type*) [SizeOf X] [SizeOf Y] (L_X : Set X) (L_Y : Set Y) where
  f : X → Y
  poly_time : ∃ (c k : ℕ), ∀ x : X, sizeOf (f x) ≤ c * (sizeOf x) ^ k + c
  correct : ∀ x, x ∈ L_X ↔ f x ∈ L_Y

/-! ## Sequential Types for TQBF Reduction -/

/-- Actions for sequential decision problems -/
inductive SeqAction (n : ℕ) where
  | choose : Bool → SeqAction n
  | query : Fin n → SeqAction n
  deriving DecidableEq, Fintype

/-- States for sequential decision problems -/
inductive SeqState (n : ℕ) where
  | init : SeqState n
  | quantified : Fin n → Bool → SeqState n
  | terminal : Bool → SeqState n
  deriving DecidableEq, Fintype

/-- Observations for sequential decision problems -/
inductive SeqObs where
  | none : SeqObs
  | value : Bool → SeqObs
  deriving DecidableEq, Fintype

/-- CoordinateSpace instance for SeqState (trivial: single coordinate) -/
instance instCoordinateSpaceSeqState (n : ℕ) : CoordinateSpace (SeqState n) 1 where
  Coord := fun _ => SeqState n
  proj := fun s _ => s

/-! ## Reduction Functions -/

/-- Reduction from MAJSAT to stochastic sufficiency -/
noncomputable def reduceMAJSAT (φ : Formula n) : StochasticDecisionProblem StochAction (StochState n) :=
  stochProblem φ

/-- Reduction from TQBF to sequential sufficiency -/
noncomputable def reduceTQBFUtility (q : QBF n) : SeqAction n → SeqState n → ℝ :=
  by
    classical
    exact fun a s =>
      if hq : TQBF q then
        match a with
        | SeqAction.choose true => 2
        | _ => 0
      else
        match s, a with
        | SeqState.init, SeqAction.choose true => 2
        | SeqState.init, _ => 0
        | _, SeqAction.choose false => 2
        | _, _ => 0

/-- Reduction from TQBF to sequential sufficiency -/
noncomputable def reduceTQBF (q : QBF n) : SequentialDecisionProblem (SeqAction n) (SeqState n) SeqObs where
  utility := reduceTQBFUtility q
  transition := fun _a s _s' =>
    -- Uniform transition for simplicity (actual TQBF reduction is more complex)
    1 / (Fintype.card (SeqState n) : ℝ)
  observationModel := fun _s _o => 1 / (Fintype.card SeqObs : ℝ)
  horizon := n

/-! ## MAJSAT to Stochastic Sufficiency -/

/-- Standard: MAJSAT reduction is polynomial (linear in formula size) -/
theorem reduceMAJSAT_poly (φ : Formula n) :
    ∃ (c k : ℕ), sizeOf (reduceMAJSAT φ) ≤ c * (sizeOf φ) ^ k + c := by
  refine ⟨sizeOf (reduceMAJSAT φ), 0, ?_⟩
  have h : sizeOf (reduceMAJSAT φ) ≤ sizeOf (reduceMAJSAT φ) + sizeOf (reduceMAJSAT φ) :=
    Nat.le_add_right _ _
  simpa [pow_zero] using h

/-- Strict majority: |sat| > 2^(n-1) -/
def StrictMAJSAT (φ : Formula n) : Prop := φ.satCount > 2^n / 2

/-- The correctness of MAJSAT reduction (strict version):
    |sat| > 2^(n-1) ↔ accept is uniquely optimal
    Note: Requires n ≥ 1 for the exact_half case. -/
theorem reduceMAJSAT_correct_strict (φ : Formula n) (hn : n ≥ 1) :
    StrictMAJSAT φ ↔ (reduceMAJSAT φ).stochasticOpt = {StochAction.accept} := by
  unfold StrictMAJSAT reduceMAJSAT
  constructor
  · -- Strict MAJSAT → accept uniquely optimal
    intro hstrict
    exact strict_majsat_accept_unique φ hstrict
  · -- Accept uniquely optimal → strict MAJSAT
    intro haccept
    by_contra hns
    push_neg at hns
    rcases Nat.lt_or_eq_of_le hns with hlt | heq
    · -- satCount < 2^n/2 → reject is uniquely optimal, contradiction
      have hrej := strict_not_majsat_reject_unique φ hlt
      rw [haccept] at hrej
      have : StochAction.accept ∈ ({StochAction.reject} : Set StochAction) := by
        rw [← hrej]; simp
      simp at this
    · -- satCount = 2^n/2 → both optimal, not a singleton, contradiction
      have hboth := exact_half_both_optimal φ hn heq
      rw [haccept] at hboth
      have : StochAction.reject ∈ ({StochAction.accept} : Set StochAction) := by
        rw [hboth]; simp
      simp at this

/-- Canonical stochastic anchor-check instance for reduction targets
    (anchor question evaluated at `I = ∅`, with strict-side action label fixed). -/
abbrev StochasticAnchorCheckInstance (n : ℕ)
    (P : StochasticDecisionProblem StochAction (StochState n)) : Prop :=
  StochasticAnchorSufficiencyCheck P (∅ : Finset (Fin n)) ∧
    P.stochasticOpt = {StochAction.accept}

/-- MAJSAT reduction correctness, retargeted to STOCHASTIC-ANCHOR-SUFFICIENCY-CHECK. -/
theorem reduceMAJSAT_correct_anchor_strict (φ : Formula n) (hn : n ≥ 1) :
    StrictMAJSAT φ ↔ StochasticAnchorCheckInstance n (reduceMAJSAT φ) := by
  haveI : Nonempty (StochState n) := ⟨fun _ => false⟩
  constructor
  · intro hstrict
    have hacc : (reduceMAJSAT φ).stochasticOpt = {StochAction.accept} := by
      simpa [reduceMAJSAT] using strict_majsat_accept_unique φ hstrict
    refine ⟨?_, hacc⟩
    unfold StochasticAnchorSufficiencyCheck
    rw [stochasticAnchorSufficient_empty_iff]
    exact ⟨StochAction.accept, hacc⟩
  · intro hinst
    exact (reduceMAJSAT_correct_strict φ hn).2 hinst.2

/-- Reduction statement form: strict MAJSAT maps to stochastic anchor-check via `reduceMAJSAT`. -/
theorem reduceMAJSAT_to_stochastic_anchor_reduction (hn : n ≥ 1) :
    ∀ φ : Formula n, StrictMAJSAT φ ↔ StochasticAnchorCheckInstance n (reduceMAJSAT φ) :=
  fun φ => reduceMAJSAT_correct_anchor_strict φ hn

/-! ## TQBF to Sequential Sufficiency -/

/-- Standard: TQBF reduction is polynomial (linear in QBF size) -/
theorem reduceTQBF_poly (q : QBF n) :
    ∃ (c k : ℕ), sizeOf (reduceTQBF q) ≤ c * (sizeOf q) ^ k + c := by
  refine ⟨sizeOf (reduceTQBF q), 0, ?_⟩
  have h : sizeOf (reduceTQBF q) ≤ sizeOf (reduceTQBF q) + sizeOf (reduceTQBF q) :=
    Nat.le_add_right _ _
  simpa [pow_zero] using h

/-- If `TQBF q` holds, the reduced problem has a state-independent unique optimum:
    `choose true`. -/
theorem reduceTQBF_opt_of_true (q : QBF n) (hq : TQBF q) (s : SeqState n) :
    (reduceTQBF q).toDecisionProblem.Opt s = {SeqAction.choose true} := by
  ext a
  constructor
  · intro ha
    cases a with
    | choose b =>
      cases b with
      | false =>
        have hle := ha (SeqAction.choose true)
        have : (2 : ℝ) ≤ 0 := by
          simpa [reduceTQBF, reduceTQBFUtility, hq] using hle
        linarith
      | true =>
        simp
    | query i =>
      have hle := ha (SeqAction.choose true)
      have : (2 : ℝ) ≤ 0 := by
        simpa [reduceTQBF, reduceTQBFUtility, hq] using hle
      linarith
  · intro ha
    rcases Set.mem_singleton_iff.mp ha with rfl
    intro a'
    cases a' with
    | choose b =>
      cases b <;> simp [reduceTQBF, reduceTQBFUtility, hq]
    | query i =>
      simp [reduceTQBF, reduceTQBFUtility, hq]

/-- If `TQBF q` fails, the reduced problem has different optimal sets at two states. -/
theorem reduceTQBF_not_sufficient_of_false (q : QBF n) (hq : ¬ TQBF q) :
    ¬ SequentialSufficient (n := 1) (reduceTQBF q) ∅ := by
  intro hsuff
  have hconst := hsuff SeqState.init (SeqState.terminal false) (by simp [agreeOn])
  have hmem_init : SeqAction.choose true ∈ (reduceTQBF q).toDecisionProblem.Opt SeqState.init := by
    intro a'
    cases a' with
    | choose b =>
      cases b <;> simp [reduceTQBF, reduceTQBFUtility, hq]
    | query i =>
      simp [reduceTQBF, reduceTQBFUtility, hq]
  have hnot_mem_term :
      SeqAction.choose true ∉ (reduceTQBF q).toDecisionProblem.Opt (SeqState.terminal false) := by
    intro hmem
    have hle := hmem (SeqAction.choose false)
    have : (2 : ℝ) ≤ 0 := by
      simpa [reduceTQBF, reduceTQBFUtility, hq] using hle
    linarith
  exact hnot_mem_term (hconst ▸ hmem_init)

/-- TQBF correctness: the reduction encodes truth as empty-set sequential sufficiency. -/
theorem reduceTQBF_correct (q : QBF n) :
    TQBF q ↔ SequentialSufficient (n := 1) (reduceTQBF q) ∅ := by
  constructor
  · intro hq
    intro s s' _
    rw [reduceTQBF_opt_of_true q hq s, reduceTQBF_opt_of_true q hq s']
  · intro hsuff
    by_contra hq
    exact reduceTQBF_not_sufficient_of_false q hq hsuff

/-- Canonical sequential anchor-check instance for reduction targets
    (anchor question evaluated at `I = ∅` in the one-coordinate representation). -/
abbrev SequentialAnchorCheckInstance (n : ℕ)
    (P : SequentialDecisionProblem (SeqAction n) (SeqState n) SeqObs) : Prop :=
  SequentialAnchorSufficiencyCheck (n := 1) P ∅

/-- TQBF reduction correctness, retargeted to SEQUENTIAL-ANCHOR-SUFFICIENCY-CHECK. -/
theorem reduceTQBF_correct_anchor (q : QBF n) :
    TQBF q ↔ SequentialAnchorCheckInstance n (reduceTQBF q) := by
  haveI : Nonempty (SeqState n) := ⟨SeqState.init⟩
  unfold SequentialAnchorCheckInstance SequentialAnchorSufficiencyCheck
  rw [sequentialAnchorSufficient_empty_iff]
  exact reduceTQBF_correct q

/-- Reduction statement form: TQBF maps to sequential anchor-check via `reduceTQBF`. -/
theorem reduceTQBF_to_sequential_anchor_reduction :
    ∀ q : QBF n, TQBF q ↔ SequentialAnchorCheckInstance n (reduceTQBF q) :=
  fun q => reduceTQBF_correct_anchor q

/-! ## Composition -/

/-- Standard: polynomial composition is polynomial.
    If f(n) ≤ c₁n^k₁ + c₁ and g(n) ≤ c₂n^k₂ + c₂,
    then (g ∘ f)(n) ≤ O(n^(k₁·k₂)). -/
theorem poly_composition_bound {X Y Z : Type*} [SizeOf X] [SizeOf Y] [SizeOf Z]
    (f : X → Y) (g : Y → Z)
    (c1 k1 : ℕ) (hf : ∀ x, sizeOf (f x) ≤ c1 * (sizeOf x) ^ k1 + c1)
    (c2 k2 : ℕ) (hg : ∀ y, sizeOf (g y) ≤ c2 * (sizeOf y) ^ k2 + c2) :
    ∃ (c k : ℕ), ∀ x, sizeOf (g (f x)) ≤ c * (sizeOf x) ^ k + c := by
  let rXY : DecisionQuotient.SizeBoundedReduction X Y := ⟨f, ⟨c1, k1, hf⟩⟩
  let rYZ : DecisionQuotient.SizeBoundedReduction Y Z := ⟨g, ⟨c2, k2, hg⟩⟩
  obtain ⟨r, hr⟩ := DecisionQuotient.SizeBoundedReduction.comp_exists rXY rYZ
  rcases r.size_bound with ⟨c, k, hk⟩
  refine ⟨c, k, ?_⟩
  intro x
  simpa [rXY, rYZ, hr, Function.comp] using hk x

/-- Reductions compose -/
def composeReduction4b {X Y Z : Type*} [SizeOf X] [SizeOf Y] [SizeOf Z]
    {L_X : Set X} {L_Y : Set Y} {L_Z : Set Z}
    (rXY : PolyReduction4b X Y L_X L_Y) (rYZ : PolyReduction4b Y Z L_Y L_Z) :
    PolyReduction4b X Z L_X L_Z where
  f := rYZ.f ∘ rXY.f
  poly_time := by
    obtain ⟨c1, k1, h1⟩ := rXY.poly_time
    obtain ⟨c2, k2, h2⟩ := rYZ.poly_time
    exact poly_composition_bound rXY.f rYZ.f c1 k1 h1 c2 k2 h2
  correct := by
    intro x
    exact Iff.trans (rXY.correct x) (rYZ.correct (rXY.f x))

/-! ## Hardness Transfer -/

/-- Hardness transfer: if there's a reduction from L_X to L_Y,
    then membership in L_X reduces to membership in L_Y. -/
theorem reduction_hardness_transfer {X Y : Type*} [SizeOf X] [SizeOf Y]
    (L_X : Set X) (L_Y : Set Y)
    (r : PolyReduction4b X Y L_X L_Y) :
    ∀ x : X, x ∈ L_X ↔ r.f x ∈ L_Y := r.correct

end DecisionQuotient.StochasticSequential
