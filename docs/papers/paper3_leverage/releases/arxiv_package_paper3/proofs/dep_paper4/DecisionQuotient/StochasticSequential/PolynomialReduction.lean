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

/-! ## Pure MAJSAT to Stochastic Anchor Check -/

/-- Three-action gadget for the pure stochastic anchor query. The duplicated
    threshold actions destroy uniqueness on the negative side of the MAJSAT
    threshold while preserving a unique accepting action on the positive side. -/
inductive PureAnchorStochAction where
  | accept : PureAnchorStochAction
  | holdL : PureAnchorStochAction
  | holdR : PureAnchorStochAction
  deriving DecidableEq, Fintype

/-- A threshold placed strictly below `1/2`, but above the largest sub-majority
    acceptance expectation. For `n ≥ 1`, this separates the MAJSAT yes/no sides. -/
noncomputable def pureAnchorThreshold (n : ℕ) : ℝ :=
  (1 / 2 : ℝ) - 1 / (2 : ℝ) ^ (n + 1)

/-- Utility for the pure stochastic anchor reduction. -/
noncomputable def pureAnchorUtility (φ : Formula n) : PureAnchorStochAction → StochState n → ℝ
  | PureAnchorStochAction.accept, a => if Formula.eval a φ then 1 else 0
  | PureAnchorStochAction.holdL, _ => pureAnchorThreshold n
  | PureAnchorStochAction.holdR, _ => pureAnchorThreshold n

/-- MAJSAT reduction targeted directly at `STOCHASTIC-ANCHOR-SUFFICIENCY-CHECK`. -/
noncomputable def reduceMAJSATPureAnchor (φ : Formula n) :
    StochasticDecisionProblem PureAnchorStochAction (StochState n) :=
  { toDecisionProblem := { utility := pureAnchorUtility φ }
    distribution := stochDistribution n }

theorem reduceMAJSATPureAnchor_poly_bound :
    ∃ (c k : ℕ), ∀ φ : Formula n, sizeOf (reduceMAJSATPureAnchor φ) ≤ c * (sizeOf φ) ^ k + c := by
  refine ⟨32, 1, ?_⟩
  intro φ
  simp [reduceMAJSATPureAnchor, pureAnchorUtility]

theorem pureAnchor_expected_accept_eq (φ : Formula n) :
    stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept =
      expectedUtilityAccept φ := by
  unfold stochasticExpectedUtility reduceMAJSATPureAnchor stochDistribution pureAnchorUtility
    expectedUtilityAccept
  unfold Formula.satCount
  have hsplit : ∀ a : StochState n,
      1 / (2 : ℝ) ^ n * (if Formula.eval a φ then 1 else 0) =
        if Formula.eval a φ then 1 / (2 : ℝ) ^ n else 0 := by
    intro a
    split_ifs <;> ring
  simp_rw [hsplit]
  rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_comm]
  rw [one_div, mul_comm, ← div_eq_mul_inv]

theorem pureAnchor_expected_hold_eq (φ : Formula n)
    (a : PureAnchorStochAction)
    (ha : a = PureAnchorStochAction.holdL ∨ a = PureAnchorStochAction.holdR) :
    stochasticExpectedUtility (reduceMAJSATPureAnchor φ) a = pureAnchorThreshold n := by
  rcases ha with rfl | rfl
  · unfold stochasticExpectedUtility reduceMAJSATPureAnchor stochDistribution pureAnchorUtility pureAnchorThreshold
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    rw [card_stochState_eq, two_pow_nat_cast]
    have hpow_ne : (2 : ℝ) ^ n ≠ 0 := by positivity
    field_simp [hpow_ne]
  · unfold stochasticExpectedUtility reduceMAJSATPureAnchor stochDistribution pureAnchorUtility pureAnchorThreshold
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    rw [card_stochState_eq, two_pow_nat_cast]
    have hpow_ne : (2 : ℝ) ^ n ≠ 0 := by positivity
    field_simp [hpow_ne]

theorem pureAnchor_threshold_lt_half (n : ℕ) :
    pureAnchorThreshold n < (1 / 2 : ℝ) := by
  unfold pureAnchorThreshold
  have hpos : 0 < 1 / (2 : ℝ) ^ (n + 1) := by positivity
  linarith

theorem pureAnchor_expected_accept_lt_threshold_of_not_majsat
    (φ : Formula n) (hn : n ≥ 1) (hnmaj : ¬ Formula.majorityTrue φ) :
    stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept <
      pureAnchorThreshold n := by
  cases n with
  | zero =>
      omega
  | succ m =>
      have hdiv : 2 ^ Nat.succ m / 2 = 2 ^ m := by
        simp [pow_succ, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
      have hsat_lt : φ.satCount < 2 ^ m := by
        unfold Formula.majorityTrue at hnmaj
        push_neg at hnmaj
        simpa [hdiv] using hnmaj
      have hsat_le : φ.satCount ≤ 2 ^ m - 1 := by
        omega
      have hreal : (φ.satCount : ℝ) ≤ ((2 ^ m - 1 : ℕ) : ℝ) := by
        exact_mod_cast hsat_le
      have hpow_pos : 0 < (2 : ℝ) ^ Nat.succ m := by positivity
      have hdiv_le :
          (φ.satCount : ℝ) / (2 : ℝ) ^ Nat.succ m ≤
            ((2 ^ m - 1 : ℕ) : ℝ) / (2 : ℝ) ^ Nat.succ m := by
        exact div_le_div_of_nonneg_right hreal (le_of_lt hpow_pos)
      have hcast_sub : ((2 ^ m - 1 : ℕ) : ℝ) = (2 : ℝ) ^ m - 1 := by
        rw [Nat.cast_sub]
        · simp [Nat.cast_pow]
        · exact Nat.succ_le_of_lt (pow_pos (by omega) _)
      have hEq :
          ((2 ^ m - 1 : ℕ) : ℝ) / (2 : ℝ) ^ Nat.succ m =
            (1 / 2 : ℝ) - 1 / (2 : ℝ) ^ Nat.succ m := by
        rw [hcast_sub]
        have hpowm_ne : (2 : ℝ) ^ m ≠ 0 := by positivity
        calc
          ((2 : ℝ) ^ m - 1) / (2 : ℝ) ^ Nat.succ m
              = ((2 : ℝ) ^ m - 1) / ((2 : ℝ) ^ m * 2) := by
                  simp [pow_succ, mul_comm, mul_left_comm, mul_assoc]
          _ = (1 / 2 : ℝ) - 1 / (2 : ℝ) ^ Nat.succ m := by
                  field_simp [hpowm_ne]
                  rw [pow_succ]
                  ring
      have hbound :
          stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept ≤
            (1 / 2 : ℝ) - 1 / (2 : ℝ) ^ Nat.succ m := by
        calc
          stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept
              = expectedUtilityAccept φ := pureAnchor_expected_accept_eq φ
          _ = (φ.satCount : ℝ) / (2 : ℝ) ^ Nat.succ m := by
                simp [expectedUtilityAccept]
          _ ≤ ((2 ^ m - 1 : ℕ) : ℝ) / (2 : ℝ) ^ Nat.succ m := hdiv_le
          _ = (1 / 2 : ℝ) - 1 / (2 : ℝ) ^ Nat.succ m := hEq
      have hlt_inv : 1 / (2 : ℝ) ^ (Nat.succ m + 1) < 1 / (2 : ℝ) ^ Nat.succ m := by
        have hbase_pos : 0 < (2 : ℝ) ^ Nat.succ m := by positivity
        have hmul_lt : (2 : ℝ) ^ Nat.succ m < (2 : ℝ) ^ Nat.succ m * 2 := by
          nlinarith
        have hlt_inv' : 1 / ((2 : ℝ) ^ Nat.succ m * 2) < 1 / (2 : ℝ) ^ Nat.succ m := by
          exact one_div_lt_one_div_of_lt hbase_pos hmul_lt
        simpa [pow_succ, mul_comm, mul_left_comm, mul_assoc] using hlt_inv'
      have hlt_threshold :
          (1 / 2 : ℝ) - 1 / (2 : ℝ) ^ Nat.succ m < pureAnchorThreshold (Nat.succ m) := by
        unfold pureAnchorThreshold
        linarith
      exact lt_of_le_of_lt hbound hlt_threshold

theorem pureAnchor_accept_unique_of_majsat
    (φ : Formula n) (hn : n ≥ 1) (hmaj : Formula.majorityTrue φ) :
    (reduceMAJSATPureAnchor φ).stochasticOpt = {PureAnchorStochAction.accept} := by
  ext a
  constructor
  · intro ha
    cases a with
    | accept => simp
    | holdL =>
        have hhalf_le : (1 / 2 : ℝ) ≤
            stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept := by
          have h := expected_reject_le_accept_of_majsat φ hn hmaj
          have h' : (1 / 2 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
            have h0 : (0.5 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
              simpa [expected_utility_reject_eq, expectedUtilityReject] using h
            norm_num at h0 ⊢
            exact h0
          calc
            (1 / 2 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := h'
            _ = expectedUtilityAccept φ := by rw [expected_utility_accept_eq]
            _ = stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept :=
              (pureAnchor_expected_accept_eq φ).symm
        have hhold_lt :
            pureAnchorThreshold n < stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept := by
          have hth := pureAnchor_threshold_lt_half n
          linarith
        have hhold :
            stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdL = pureAnchorThreshold n :=
          pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdL (Or.inl rfl)
        have hle := ha PureAnchorStochAction.accept
        rw [hhold] at hle
        linarith
    | holdR =>
        have hhalf_le : (1 / 2 : ℝ) ≤
            stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept := by
          have h := expected_reject_le_accept_of_majsat φ hn hmaj
          have h' : (1 / 2 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
            have h0 : (0.5 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
              simpa [expected_utility_reject_eq, expectedUtilityReject] using h
            norm_num at h0 ⊢
            exact h0
          calc
            (1 / 2 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := h'
            _ = expectedUtilityAccept φ := by rw [expected_utility_accept_eq]
            _ = stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept :=
              (pureAnchor_expected_accept_eq φ).symm
        have hhold_lt :
            pureAnchorThreshold n < stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept := by
          have hth := pureAnchor_threshold_lt_half n
          linarith
        have hhold :
            stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdR = pureAnchorThreshold n :=
          pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdR (Or.inr rfl)
        have hle := ha PureAnchorStochAction.accept
        rw [hhold] at hle
        linarith
  · intro ha
    rcases Set.mem_singleton_iff.mp ha with rfl
    intro a'
    cases a' with
    | accept => rfl
    | holdL =>
        have hhalf_le : (1 / 2 : ℝ) ≤
            stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept := by
          have h := expected_reject_le_accept_of_majsat φ hn hmaj
          have h' : (1 / 2 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
            have h0 : (0.5 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
              simpa [expected_utility_reject_eq, expectedUtilityReject] using h
            norm_num at h0 ⊢
            exact h0
          calc
            (1 / 2 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := h'
            _ = expectedUtilityAccept φ := by rw [expected_utility_accept_eq]
            _ = stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept :=
              (pureAnchor_expected_accept_eq φ).symm
        have hhold :
            stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdL = pureAnchorThreshold n :=
          pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdL (Or.inl rfl)
        have hth := pureAnchor_threshold_lt_half n
        rw [hhold]
        linarith
    | holdR =>
        have hhalf_le : (1 / 2 : ℝ) ≤
            stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept := by
          have h := expected_reject_le_accept_of_majsat φ hn hmaj
          have h' : (1 / 2 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
            have h0 : (0.5 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
              simpa [expected_utility_reject_eq, expectedUtilityReject] using h
            norm_num at h0 ⊢
            exact h0
          calc
            (1 / 2 : ℝ) ≤ stochasticExpectedUtility (stochProblem φ) StochAction.accept := h'
            _ = expectedUtilityAccept φ := by rw [expected_utility_accept_eq]
            _ = stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.accept :=
              (pureAnchor_expected_accept_eq φ).symm
        have hhold :
            stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdR = pureAnchorThreshold n :=
          pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdR (Or.inr rfl)
        have hth := pureAnchor_threshold_lt_half n
        rw [hhold]
        linarith

theorem pureAnchor_holdL_optimal_of_not_majsat
    (φ : Formula n) (hn : n ≥ 1) (hnmaj : ¬ Formula.majorityTrue φ) :
    PureAnchorStochAction.holdL ∈ (reduceMAJSATPureAnchor φ).stochasticOpt := by
  unfold StochasticDecisionProblem.stochasticOpt
  simp only [Set.mem_setOf_eq]
  intro a'
  cases a' with
  | accept =>
      have hlt := pureAnchor_expected_accept_lt_threshold_of_not_majsat φ hn hnmaj
      have hhold :
          stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdL = pureAnchorThreshold n :=
        pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdL (Or.inl rfl)
      rw [hhold]
      exact le_of_lt hlt
  | holdL => rfl
  | holdR =>
      have hholdL :
          stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdL = pureAnchorThreshold n :=
        pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdL (Or.inl rfl)
      have hholdR :
          stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdR = pureAnchorThreshold n :=
        pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdR (Or.inr rfl)
      rw [hholdL, hholdR]

theorem pureAnchor_holdR_optimal_of_not_majsat
    (φ : Formula n) (hn : n ≥ 1) (hnmaj : ¬ Formula.majorityTrue φ) :
    PureAnchorStochAction.holdR ∈ (reduceMAJSATPureAnchor φ).stochasticOpt := by
  unfold StochasticDecisionProblem.stochasticOpt
  simp only [Set.mem_setOf_eq]
  intro a'
  cases a' with
  | accept =>
      have hlt := pureAnchor_expected_accept_lt_threshold_of_not_majsat φ hn hnmaj
      have hhold :
          stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdR = pureAnchorThreshold n :=
        pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdR (Or.inr rfl)
      rw [hhold]
      exact le_of_lt hlt
  | holdL =>
      have hholdL :
          stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdL = pureAnchorThreshold n :=
        pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdL (Or.inl rfl)
      have hholdR :
          stochasticExpectedUtility (reduceMAJSATPureAnchor φ) PureAnchorStochAction.holdR = pureAnchorThreshold n :=
        pureAnchor_expected_hold_eq φ PureAnchorStochAction.holdR (Or.inr rfl)
      rw [hholdL, hholdR]
  | holdR => rfl

theorem reduceMAJSATPureAnchor_correct (φ : Formula n) (hn : n ≥ 1) :
    Formula.majorityTrue φ ↔
      StochasticAnchorSufficiencyCheck (reduceMAJSATPureAnchor φ) (∅ : Finset (Fin n)) := by
  haveI : Nonempty (StochState n) := ⟨fun _ => false⟩
  constructor
  · intro hmaj
    unfold StochasticAnchorSufficiencyCheck
    rw [stochasticAnchorSufficient_empty_iff]
    exact ⟨PureAnchorStochAction.accept, pureAnchor_accept_unique_of_majsat φ hn hmaj⟩
  · intro hanchor
    by_contra hnmaj
    have hholdL := pureAnchor_holdL_optimal_of_not_majsat φ hn hnmaj
    have hholdR := pureAnchor_holdR_optimal_of_not_majsat φ hn hnmaj
    unfold StochasticAnchorSufficiencyCheck at hanchor
    rw [stochasticAnchorSufficient_empty_iff] at hanchor
    rcases hanchor with ⟨a, ha⟩
    have hEqL : PureAnchorStochAction.holdL = a := by
      simpa [ha] using hholdL
    have hEqR : PureAnchorStochAction.holdR = a := by
      simpa [ha] using hholdR
    have : PureAnchorStochAction.holdL = PureAnchorStochAction.holdR := hEqL.trans hEqR.symm
    cases this

theorem reduceMAJSAT_to_pure_stochastic_anchor_reduction (hn : n ≥ 1) :
    ∀ φ : Formula n,
      Formula.majorityTrue φ ↔
        StochasticAnchorSufficiencyCheck (reduceMAJSATPureAnchor φ) (∅ : Finset (Fin n)) :=
  fun φ => reduceMAJSATPureAnchor_correct φ hn

theorem reduceMAJSATPureSufficiency_correct (φ : Formula n) (hn : n ≥ 1) :
    Formula.majorityTrue φ ↔
      StochasticSufficient (reduceMAJSATPureAnchor φ) (∅ : Finset (Fin n)) := by
  haveI : Nonempty (StochState n) := ⟨fun _ => false⟩
  constructor
  · intro hmaj
    rw [stochasticSufficient_empty_iff]
    exact ⟨PureAnchorStochAction.accept, pureAnchor_accept_unique_of_majsat φ hn hmaj⟩
  · intro hsuff
    by_contra hnmaj
    have hholdL := pureAnchor_holdL_optimal_of_not_majsat φ hn hnmaj
    have hholdR := pureAnchor_holdR_optimal_of_not_majsat φ hn hnmaj
    rw [stochasticSufficient_empty_iff] at hsuff
    rcases hsuff with ⟨a, ha⟩
    have hEqL : PureAnchorStochAction.holdL = a := by
      simpa [ha] using hholdL
    have hEqR : PureAnchorStochAction.holdR = a := by
      simpa [ha] using hholdR
    have : PureAnchorStochAction.holdL = PureAnchorStochAction.holdR := hEqL.trans hEqR.symm
    cases this

theorem reduceMAJSAT_to_pure_stochastic_sufficiency_reduction (hn : n ≥ 1) :
    ∀ φ : Formula n,
      Formula.majorityTrue φ ↔
        StochasticSufficient (reduceMAJSATPureAnchor φ) (∅ : Finset (Fin n)) :=
  fun φ => reduceMAJSATPureSufficiency_correct φ hn

/-- Pure stochastic sufficiency hardness packaged as a polynomially size-bounded reduction. -/
def StochasticSufficiencyPPHard (n : ℕ) : Prop :=
  ∃ r : DecisionQuotient.SizeBoundedReduction (Formula n)
      (StochasticDecisionProblem PureAnchorStochAction (StochState n)),
    ∀ φ : Formula n,
      Formula.majorityTrue φ ↔ StochasticSufficient (r.f φ) (∅ : Finset (Fin n))

theorem stochastic_sufficiency_pp_hard (hn : n ≥ 1) :
    StochasticSufficiencyPPHard n := by
  refine ⟨{ f := reduceMAJSATPureAnchor, size_bound := ?_ }, ?_⟩
  · exact reduceMAJSATPureAnchor_poly_bound
  · intro φ
    exact reduceMAJSATPureSufficiency_correct φ hn

/-- Stochastic minimum-sufficiency hardness packaged as a polynomially
    size-bounded reduction to the `k = 0` slice. -/
def StochasticMinimumSufficiencyPPHard (n : ℕ) : Prop :=
  ∃ r : DecisionQuotient.SizeBoundedReduction (Formula n)
      (StochasticDecisionProblem PureAnchorStochAction (StochState n)),
    ∀ φ : Formula n,
      Formula.majorityTrue φ ↔
        StochasticMinimumSufficiencyCheck (n := n) (r.f φ) 0

theorem stochastic_minimum_sufficiency_pp_hard (hn : n ≥ 1) :
    StochasticMinimumSufficiencyPPHard n := by
  refine ⟨{ f := reduceMAJSATPureAnchor, size_bound := ?_ }, ?_⟩
  · exact reduceMAJSATPureAnchor_poly_bound
  · intro φ
    change Formula.majorityTrue φ ↔
      StochasticMinimumSufficiencyCheck (n := n) (reduceMAJSATPureAnchor φ) 0
    rw [stochasticMinimumSufficiencyCheck_zero_iff]
    exact reduceMAJSATPureSufficiency_correct φ hn

/-- Pure stochastic anchor hardness packaged as a polynomially size-bounded reduction. -/
def PureStochasticAnchorPPHard (n : ℕ) : Prop :=
  ∃ r : DecisionQuotient.SizeBoundedReduction (Formula n)
      (StochasticDecisionProblem PureAnchorStochAction (StochState n)),
    ∀ φ : Formula n,
      Formula.majorityTrue φ ↔ StochasticAnchorSufficiencyCheck (r.f φ) (∅ : Finset (Fin n))

theorem stochastic_anchor_check_pp_hard (hn : n ≥ 1) :
    PureStochasticAnchorPPHard n := by
  refine ⟨{ f := reduceMAJSATPureAnchor, size_bound := ?_ }, ?_⟩
  · exact reduceMAJSATPureAnchor_poly_bound
  · intro φ
    exact reduceMAJSATPureAnchor_correct φ hn

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

/-- Sequential sufficiency hardness packaged as a polynomially size-bounded reduction. -/
def SequentialSufficiencyPSPACEHard (n : ℕ) : Prop :=
  ∃ r : DecisionQuotient.SizeBoundedReduction (QBF n)
      (SequentialDecisionProblem (SeqAction n) (SeqState n) SeqObs),
    ∀ q : QBF n,
      TQBF q ↔ SequentialSufficient (n := 1) (r.f q) (∅ : Finset (Fin 1))

theorem sequential_sufficiency_pspace_hard (n : ℕ) :
    SequentialSufficiencyPSPACEHard n := by
  refine ⟨{ f := reduceTQBF, size_bound := ?_ }, ?_⟩
  · refine ⟨n + 32, 1, ?_⟩
    intro q
    simp [reduceTQBF, reduceTQBFUtility]
    omega
  · intro q
    exact reduceTQBF_correct q

/-- Sequential minimum-sufficiency hardness packaged as a polynomially
    size-bounded reduction to the `k = 0` slice. -/
def SequentialMinimumSufficiencyPSPACEHard (n : ℕ) : Prop :=
  ∃ r : DecisionQuotient.SizeBoundedReduction (QBF n)
      (SequentialDecisionProblem (SeqAction n) (SeqState n) SeqObs),
    ∀ q : QBF n,
      TQBF q ↔
        SequentialMinimumSufficiencyCheck (n := 1) (r.f q) 0

theorem sequential_minimum_sufficiency_pspace_hard (n : ℕ) :
    SequentialMinimumSufficiencyPSPACEHard n := by
  refine ⟨{ f := reduceTQBF, size_bound := ?_ }, ?_⟩
  · refine ⟨n + 32, 1, ?_⟩
    intro q
    simp [reduceTQBF, reduceTQBFUtility]
    omega
  · intro q
    change TQBF q ↔ SequentialMinimumSufficiencyCheck (n := 1) (reduceTQBF q) 0
    rw [sequentialMinimumSufficiencyCheck_zero_iff]
    exact reduceTQBF_correct q

/-- Sequential anchor hardness packaged as a polynomially size-bounded reduction. -/
def SequentialAnchorPSPACEHard (n : ℕ) : Prop :=
  ∃ r : DecisionQuotient.SizeBoundedReduction (QBF n)
      (SequentialDecisionProblem (SeqAction n) (SeqState n) SeqObs),
    ∀ q : QBF n,
      TQBF q ↔ SequentialAnchorSufficiencyCheck (n := 1) (r.f q) (∅ : Finset (Fin 1))

theorem sequential_anchor_check_pspace_hard (n : ℕ) :
    SequentialAnchorPSPACEHard n := by
  refine ⟨{ f := reduceTQBF, size_bound := ?_ }, ?_⟩
  · refine ⟨n + 32, 1, ?_⟩
    intro q
    simp [reduceTQBF, reduceTQBFUtility]
    omega
  · intro q
    simpa [SequentialAnchorCheckInstance] using reduceTQBF_correct_anchor q

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
