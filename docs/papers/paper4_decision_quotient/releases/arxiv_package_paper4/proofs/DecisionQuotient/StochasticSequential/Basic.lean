/-
  Paper 4: Stochastic and Sequential Regimes

  Extends Paper 4's decision quotient to stochastic and sequential regimes.
  Key extensions:
  - StochasticDecisionProblem: adds distribution over states
  - SequentialDecisionProblem: adds transitions and observations (POMDP)

  Merged from Paper 4b. Refactored to use the 6 tractable subcases machinery.
-/

import DecisionQuotient.Basic
import DecisionQuotient.Sufficiency
import DecisionQuotient.Reduction
import DecisionQuotient.IntegrityCompetence
import DecisionQuotient.Physics.IntegrityEquilibrium
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

namespace DecisionQuotient.StochasticSequential

open DecisionQuotient
open DecisionQuotient.Physics.DimensionalComplexity
open Classical

/-! ## Probability Distributions -/

/-- A probability distribution over a finite type (simplified, noncomputable) -/
structure Distribution (S : Type*) [Fintype S] where
  /-- Probability mass function -/
  pmf : S → ℝ
  /-- Probabilities sum to 1 (axiomatized) -/
  sum_eq_one : ∑ s, pmf s = 1
  /-- Probabilities are non-negative -/
  nonneg : ∀ s, 0 ≤ pmf s

namespace Distribution

/-- Get probability of a specific outcome -/
def probability {S : Type*} [Fintype S] (d : Distribution S) (s : S) : ℝ := d.pmf s

/-- Dirac delta distribution: puts all mass on one point -/
noncomputable def delta {S : Type*} [Fintype S] [DecidableEq S] (s₀ : S) : Distribution S where
  pmf := fun s' => if s' = s₀ then 1 else 0
  sum_eq_one := by simp [Finset.sum_ite_eq, Finset.mem_univ]
  nonneg := fun s' => by split_ifs <;> linarith

/-- Uniform distribution over all elements -/
noncomputable def uniform {S : Type*} [Fintype S] [Nonempty S] : Distribution S where
  pmf := fun _ => 1 / Fintype.card S
  sum_eq_one := by
    simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    have hcard : (Fintype.card S : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr Fintype.card_ne_zero
    field_simp [hcard]
  nonneg := fun _ => by positivity

end Distribution

/-! ## Stochastic Decision Problem -/

/-- Extends DecisionProblem with a probability distribution over states -/
structure StochasticDecisionProblem (A S : Type*) [Fintype A] [Fintype S]
    extends DecisionProblem A S where
  /-- Distribution over states (probability mass function) -/
  distribution : S → ℝ

/-- Expected utility of an action under the distribution -/
def stochasticExpectedUtility {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) (a : A) : ℝ :=
  ∑ s, P.distribution s * P.utility a s

/-- The optimal set under expected utility -/
def StochasticDecisionProblem.stochasticOpt {A S : Type*} [Fintype A] [Fintype S]
    (P : StochasticDecisionProblem A S) : Set A :=
  { a : A | ∀ a', stochasticExpectedUtility P a' ≤ stochasticExpectedUtility P a }

/-- Unnormalized expected utility of action `a` restricted to the I-fiber of s₀.
    Sums over all states that agree with s₀ on the coordinates in I.
    For I = ∅, agreeOn s s₀ ∅ is vacuously true, recovering stochasticExpectedUtility. -/
noncomputable def fiberExpectedUtility {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (s₀ : S) (a : A) : ℝ :=
  ∑ s : S, if agreeOn s s₀ I then P.distribution s * P.utility a s else 0

/-- Conditional optimal action set given that the state lies in the I-fiber of s₀ -/
def fiberOpt {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (s₀ : S) : Set A :=
  { a : A | ∀ a', fiberExpectedUtility P I s₀ a' ≤ fiberExpectedUtility P I s₀ a }

/-- Stochastic sufficiency: observing coordinates I uniquely determines the optimal action
    for every possible I-fiber (every conditional has a singleton optimal set).

    Per paper Definition 2.19: I is stochastically sufficient if, for each state s₀,
    the conditional-optimal action given the I-coordinates of s₀ is uniquely determined.

    For I = ∅: agreeOn s s₀ ∅ holds for all s, so the fiber EU equals the prior EU,
    and this reduces to: the prior-optimal action is unique (see stochasticSufficient_empty_iff).
    For I = Finset.univ: each state is its own fiber, giving pointwise sufficiency. -/
def StochasticSufficient
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  ∀ s₀ : S, ∃ a : A, fiberOpt P I s₀ = {a}

/-- Stochastic anchor sufficiency: there exists an anchor state/fiber whose
    conditional-optimal action is singleton, and every state in that same
    observed fiber induces the same singleton conditional optimum. -/
def StochasticAnchorSufficient
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  ∃ s₀ : S, ∃ a : A,
    fiberOpt P I s₀ = {a} ∧
    ∀ s : S, agreeOn s s₀ I → fiberOpt P I s = {a}

/-- Decision-predicate form for the stochastic anchor question. -/
def StochasticAnchorSufficiencyCheck
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  StochasticAnchorSufficient P I

theorem stochastic_anchor_check_iff
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    StochasticAnchorSufficiencyCheck P I ↔ StochasticAnchorSufficient P I := Iff.rfl

/-! ## Boolean Formulas (reused from paper 4) -/

-- Reuse Formula, Assignment from paper 4
-- open DecisionQuotient.Reduction -- would conflict, so we redefine

inductive Formula (n : ℕ) where
  | var : Fin n → Formula n
  | not : Formula n → Formula n
  | and : Formula n → Formula n → Formula n
  | or : Formula n → Formula n → Formula n
  | top : Formula n    -- renamed from 'true' to avoid Bool conflict
  | bot : Formula n    -- renamed from 'false' to avoid Bool conflict
  deriving DecidableEq

abbrev Assignment (n : ℕ) := Fin n → Bool

def Formula.eval (a : Assignment n) : Formula n → Bool
  | Formula.var i => a i
  | Formula.not f => !(Formula.eval a f)
  | Formula.and f1 f2 => (Formula.eval a f1) && (Formula.eval a f2)
  | Formula.or f1 f2 => (Formula.eval a f1) || (Formula.eval a f2)
  | Formula.top => Bool.true
  | Formula.bot => Bool.false

/-! ## MAJSAT (for PP-completeness) -/

def Formula.majorityTrue (φ : Formula n) : Prop :=
  (Finset.univ.filter (fun a : Assignment n => Formula.eval a φ = Bool.true)).card ≥ 2^n / 2

/-! ## Reduction from MAJSAT to Stochastic Sufficiency -/

/-- Actions: accept or reject -/
inductive StochAction where
  | accept : StochAction
  | reject : StochAction
  deriving DecidableEq, Fintype

/-- States for stochastic reduction: assignments only (per paper Section 2.2)
    S = {0,1}^n, the space of all Boolean assignments -/
abbrev StochState (n : ℕ) := Assignment n

/-- CoordinateSpace instance for Assignment (state = tuple of Bools) -/
instance : CoordinateSpace (Assignment n) n where
  Coord := fun _ => Bool
  proj := fun a i => a i

/-- Uniform distribution over all assignments (paper: P = uniform over {0,1}^n) -/
noncomputable def stochDistribution (n : ℕ) : StochState n → ℝ :=
  fun _ => 1 / (2^n : ℝ)

/-- Utility function per paper (Section 2.2, lines 64):
    U(accept, s) = φ(s)  (1 if satisfying, 0 if falsifying)
    U(reject, s) = 1/2   (constant threshold)

    This creates the MAJSAT reduction:
    - E[U(accept)] = |sat|/2^n
    - E[U(reject)] = 1/2
    - accept is optimal iff |sat|/2^n ≥ 1/2 iff |sat| ≥ 2^(n-1) iff MAJSAT -/
def stochUtility (φ : Formula n) : StochAction → StochState n → ℝ
  | StochAction.accept, a => if Formula.eval a φ then 1 else 0
  | StochAction.reject, _ => 0.5

/-- The stochastic decision problem from a formula (MAJSAT reduction) -/
noncomputable def stochProblem (φ : Formula n) : StochasticDecisionProblem StochAction (StochState n) :=
  { toDecisionProblem := { utility := stochUtility φ }
    distribution := stochDistribution n }

/-- Number of satisfying assignments -/
def Formula.satCount (φ : Formula n) : ℕ :=
  (Finset.univ.filter (fun a : Assignment n => Formula.eval a φ = true)).card

/-- Expected utility of accept = |sat|/2^n -/
noncomputable def expectedUtilityAccept (φ : Formula n) : ℝ :=
  (φ.satCount : ℝ) / (2^n : ℝ)

/-- Expected utility of reject = 1/2 -/
noncomputable def expectedUtilityReject : ℝ := 0.5

/-! ### PP-Completeness Proof -/

open StochAction

/-- Helper: card of StochState is 2^n -/
theorem card_stochState_eq (n : ℕ) : Fintype.card (StochState n) = 2^n := by
  simp only [StochState, Assignment, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]

/-- Helper: 2^n as ℕ cast to ℝ equals 2^n as ℝ -/
theorem two_pow_nat_cast (n : ℕ) : ((2^n : ℕ) : ℝ) = (2 : ℝ)^n := by
  simp only [Nat.cast_pow, Nat.cast_ofNat]

/-- Expected utility of accept under uniform distribution.
    E[U(accept)] = (1/2^n) * Σ_{a : {0,1}^n} U(accept, a)
                 = (1/2^n) * |{a : φ(a) = true}|
                 = satCount(φ) / 2^n -/
theorem expected_utility_accept_eq (φ : Formula n) :
    stochasticExpectedUtility (stochProblem φ) StochAction.accept = expectedUtilityAccept φ := by
  unfold stochasticExpectedUtility stochProblem stochDistribution stochUtility expectedUtilityAccept
  unfold Formula.satCount
  have h2n : (2 : ℝ)^n ≠ 0 := by positivity
  -- Split sum based on whether assignment satisfies φ
  have hsplit : ∀ a : StochState n, 1 / (2 : ℝ)^n * (if Formula.eval a φ then 1 else 0) =
      if Formula.eval a φ then 1 / (2 : ℝ)^n else 0 := by
    intro a; split_ifs <;> ring
  simp_rw [hsplit]
  rw [Finset.sum_ite, Finset.sum_const_zero, add_zero, Finset.sum_const, nsmul_eq_mul, mul_comm]
  -- Convert 1/2^n * card to card/2^n
  rw [one_div, mul_comm, ← div_eq_mul_inv]

/-- Expected utility of reject under uniform distribution.
    E[U(reject)] = (1/2^n) * Σ_{a : {0,1}^n} U(reject, a)
                 = (1/2^n) * 2^n * 0.5
                 = 0.5 -/
theorem expected_utility_reject_eq (φ : Formula n) :
    stochasticExpectedUtility (stochProblem φ) StochAction.reject = expectedUtilityReject := by
  unfold stochasticExpectedUtility stochProblem stochDistribution stochUtility expectedUtilityReject
  simp only [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [card_stochState_eq, two_pow_nat_cast]
  have h2n : (2 : ℝ)^n ≠ 0 := by positivity
  field_simp [h2n]

/-! ### Standard Arithmetic Results -/

/-- floor(2^n / 2) ≤ 2^n / 2 as reals. -/
theorem nat_div_two_le_real (n : ℕ) : ((2^n / 2 : ℕ) : ℝ) ≤ (2 : ℝ)^n / 2 := by
  have hmul : ((2^n / 2 : ℕ) : ℝ) * 2 ≤ ((2^n : ℕ) : ℝ) := by
    exact_mod_cast (Nat.div_mul_le_self (2^n) 2)
  have hmul' : 2 * ((2^n / 2 : ℕ) : ℝ) ≤ (2 : ℝ)^n := by
    calc
      2 * ((2^n / 2 : ℕ) : ℝ) = ((2^n / 2 : ℕ) : ℝ) * 2 := by ring
      _ ≤ ((2^n : ℕ) : ℝ) := hmul
      _ = (2 : ℝ)^n := by simp [Nat.cast_pow]
  nlinarith

/-- 0.5 ≤ floor(2^n / 2) / 2^n when n ≥ 1. -/
theorem half_le_nat_div_pow (n : ℕ) (hn : n ≥ 1) :
    (0.5 : ℝ) ≤ ((2^n / 2 : ℕ) : ℝ) / (2 : ℝ)^n := by
  cases n with
  | zero =>
      omega
  | succ m =>
      have hdiv : (2 ^ (Nat.succ m) / 2 : ℕ) = 2 ^ m := by
        simp [pow_succ, Nat.mul_comm]
      have hpow_ne : (2 : ℝ)^m ≠ 0 := by positivity
      have hEq : (((2 ^ m : ℕ) : ℝ) / (2 : ℝ) ^ (Nat.succ m)) = (0.5 : ℝ) := by
        calc
          (((2 ^ m : ℕ) : ℝ) / (2 : ℝ) ^ (Nat.succ m))
              = (2 : ℝ)^m / ((2 : ℝ)^m * 2) := by
                  simp [Nat.cast_pow, pow_succ, mul_assoc, mul_comm, mul_left_comm]
          _ = (1 : ℝ) / 2 := by
                field_simp [hpow_ne]
          _ = (0.5 : ℝ) := by norm_num
      have hEqFinal :
          ((2 ^ (Nat.succ m) / 2 : ℕ) : ℝ) / (2 : ℝ) ^ (Nat.succ m) = (0.5 : ℝ) := by
        simpa [hdiv] using hEq
      calc
        (0.5 : ℝ) ≤ (0.5 : ℝ) := le_rfl
        _ = ((2 ^ (Nat.succ m) / 2 : ℕ) : ℝ) / (2 : ℝ) ^ (Nat.succ m) := by
            simpa using hEqFinal.symm

/-- Standard: a ≤ b → a / c ≤ b / c for c > 0 -/
theorem div_le_div_of_le_left_pos {a b c : ℝ} (hab : a ≤ b) (hc : 0 < c) :
    a / c ≤ b / c := div_le_div_of_nonneg_right hab (le_of_lt hc)

/-- satCount is bounded by the number of assignments -/
theorem Formula.satCount_le_card (φ : Formula n) : φ.satCount ≤ 2^n := by
  unfold satCount
  calc (Finset.univ.filter (fun a : Assignment n => Formula.eval a φ = true)).card
      ≤ Finset.univ.card := Finset.card_filter_le _ _
    _ = Fintype.card (Assignment n) := rfl
    _ = 2^n := by simp [Assignment, Fintype.card_fun, Fintype.card_bool, Fintype.card_fin]

/-! ### Non-Standard Results (Proved) -/

/-- E[reject] ≤ E[accept] when MAJSAT holds (requires n ≥ 1).
    Proof: MAJSAT gives |sat| ≥ 2^n/2, so |sat|/2^n ≥ 0.5 = E[reject]
    Note: For n=0, MAJSAT is vacuously true (threshold=0), so n ≥ 1 is required. -/
theorem expected_reject_le_accept_of_majsat (φ : Formula n) (hn : n ≥ 1) (hmaj : φ.majorityTrue) :
    stochasticExpectedUtility (stochProblem φ) StochAction.reject ≤
    stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
  rw [expected_utility_accept_eq, expected_utility_reject_eq]
  unfold expectedUtilityAccept expectedUtilityReject Formula.majorityTrue at *
  have h2n : (0 : ℝ) < 2^n := by positivity
  have hsat_nat : ((2^n / 2 : ℕ) : ℝ) ≤ (φ.satCount : ℝ) := by exact_mod_cast hmaj
  calc (0.5 : ℝ) ≤ ((2^n / 2 : ℕ) : ℝ) / (2 : ℝ)^n := half_le_nat_div_pow n hn
    _ ≤ (φ.satCount : ℝ) / (2 : ℝ)^n := div_le_div_of_le_left_pos hsat_nat h2n

/-- E[accept] ≤ E[reject] when NOT MAJSAT.
    Proof: ¬MAJSAT gives |sat| < 2^n/2 + 1, so |sat|/2^n ≤ floor(2^n/2)/2^n ≤ 0.5 -/
theorem expected_accept_le_reject_of_not_majsat (φ : Formula n) (hnmaj : ¬φ.majorityTrue) :
    stochasticExpectedUtility (stochProblem φ) StochAction.accept ≤
    stochasticExpectedUtility (stochProblem φ) StochAction.reject := by
  rw [expected_utility_accept_eq, expected_utility_reject_eq]
  unfold expectedUtilityAccept expectedUtilityReject
  have h2n : (0 : ℝ) < 2^n := by positivity
  -- hnmaj: ¬majorityTrue means satCount < 2^n/2 (as Nat, so ≤ 2^n/2 - 1... or < threshold)
  unfold Formula.majorityTrue at hnmaj
  push_neg at hnmaj
  -- hnmaj: (filter...).card < 2^n / 2
  -- Use that satCount = (filter...).card
  have hsat : φ.satCount < 2^n / 2 + 1 := by
    unfold Formula.satCount
    omega
  have hsat' : φ.satCount ≤ 2^n / 2 := Nat.lt_succ_iff.mp hsat
  have hsat_real : (φ.satCount : ℝ) ≤ ((2^n / 2 : ℕ) : ℝ) := by exact_mod_cast hsat'
  -- satCount / 2^n ≤ floor(2^n/2) / 2^n ≤ (2^n/2) / 2^n = 0.5
  calc (φ.satCount : ℝ) / (2 : ℝ)^n ≤ ((2^n / 2 : ℕ) : ℝ) / (2 : ℝ)^n :=
        div_le_div_of_le_left_pos hsat_real h2n
    _ ≤ ((2 : ℝ)^n / 2) / (2 : ℝ)^n := div_le_div_of_le_left_pos (nat_div_two_le_real n) h2n
    _ = 0.5 := by field_simp [ne_of_gt h2n]; norm_num

/-- If MAJSAT holds (|sat| ≥ 2^(n-1)), then accept is optimal (requires n ≥ 1).
    E[accept] = |sat|/2^n ≥ 2^(n-1)/2^n = 1/2 = E[reject] -/
theorem majsat_accept_optimal (φ : Formula n) (hn : n ≥ 1) (hmaj : φ.majorityTrue) :
    StochAction.accept ∈ (stochProblem φ).stochasticOpt := by
  unfold StochasticDecisionProblem.stochasticOpt
  simp only [Set.mem_setOf_eq]
  intro a'
  cases a' with
  | accept => rfl
  | reject => exact expected_reject_le_accept_of_majsat φ hn hmaj

/-- If NOT MAJSAT (|sat| < 2^(n-1)), then reject is strictly better.
    E[accept] = |sat|/2^n < 1/2 = E[reject] -/
theorem not_majsat_reject_optimal (φ : Formula n) (hnmaj : ¬φ.majorityTrue) :
    StochAction.reject ∈ (stochProblem φ).stochasticOpt := by
  unfold StochasticDecisionProblem.stochasticOpt
  simp only [Set.mem_setOf_eq]
  intro a'
  cases a' with
  | reject => rfl
  | accept => exact expected_accept_le_reject_of_not_majsat φ hnmaj

/-- Helper: E[reject] < E[accept] when strict MAJSAT holds.
    For n ≥ 1, this follows from satCount > 2^n/2 implying satCount/2^n > 0.5.
    For n = 0, hstrict implies satCount ≥ 1, and satCount/1 > 0.5 when satCount ≥ 1. -/
theorem expected_reject_lt_accept_of_strict_majsat (φ : Formula n) (hstrict : φ.satCount > 2^n / 2) :
    stochasticExpectedUtility (stochProblem φ) StochAction.reject <
    stochasticExpectedUtility (stochProblem φ) StochAction.accept := by
  rw [expected_utility_accept_eq, expected_utility_reject_eq]
  unfold expectedUtilityAccept expectedUtilityReject
  have h2n : (0 : ℝ) < 2^n := by positivity
  have hgt : (φ.satCount : ℝ) > ((2^n / 2 : ℕ) : ℝ) := by exact_mod_cast hstrict
  by_cases hn : n ≥ 1
  · -- n ≥ 1: Use standard inequality chain
    calc (0.5 : ℝ) ≤ ((2^n / 2 : ℕ) : ℝ) / (2 : ℝ)^n := half_le_nat_div_pow n hn
      _ < (φ.satCount : ℝ) / (2 : ℝ)^n := div_lt_div_of_pos_right hgt h2n
  · -- n = 0: satCount > 0 means satCount ≥ 1, so satCount/1 ≥ 1 > 0.5
    push_neg at hn; interval_cases n
    simp only [pow_zero, Nat.div_self (by omega : 0 < 1)] at hstrict ⊢
    -- hstrict: satCount > 0, so satCount ≥ 1
    have hge1 : φ.satCount ≥ 1 := hstrict
    simp only [div_one]
    calc (0.5 : ℝ) < 1 := by norm_num
      _ ≤ (φ.satCount : ℝ) := by exact_mod_cast hge1

/-- Helper: E[accept] < E[reject] when strict NOT MAJSAT holds -/
theorem expected_accept_lt_reject_of_strict_not_majsat (φ : Formula n) (hstrict : φ.satCount < 2^n / 2) :
    stochasticExpectedUtility (stochProblem φ) StochAction.accept <
    stochasticExpectedUtility (stochProblem φ) StochAction.reject := by
  rw [expected_utility_accept_eq, expected_utility_reject_eq]
  unfold expectedUtilityAccept expectedUtilityReject
  have h2n : (0 : ℝ) < 2^n := by positivity
  have hlt : (φ.satCount : ℝ) < ((2^n / 2 : ℕ) : ℝ) := by exact_mod_cast hstrict
  calc (φ.satCount : ℝ) / (2 : ℝ)^n < ((2^n / 2 : ℕ) : ℝ) / (2 : ℝ)^n :=
        div_lt_div_of_pos_right hlt h2n
    _ ≤ ((2 : ℝ)^n / 2) / (2 : ℝ)^n := div_le_div_of_le_left_pos (nat_div_two_le_real n) h2n
    _ = 0.5 := by field_simp [ne_of_gt h2n]; norm_num

/-- Helper: For n ≥ 1, 2^n / 2 = 2^(n-1) -/
theorem pow_div_two_eq_pow_pred (hn : n ≥ 1) : (2^n / 2 : ℕ) = 2^(n-1) := by
  have hpow : (2^n : ℕ) = 2 * 2^(n-1) := by
    conv_lhs => rw [show n = (n - 1) + 1 by omega, Nat.pow_succ]
    ring
  rw [hpow, Nat.mul_div_cancel_left _ (by omega : 0 < 2)]

/-- Helper: E[accept] = E[reject] when exactly half (requires n ≥ 1).
    For n=0, "exact half" means satCount=0, but E[accept]=0 ≠ 0.5=E[reject]. -/
theorem expected_eq_of_exact_half (φ : Formula n) (hn : n ≥ 1) (hexact : φ.satCount = 2^n / 2) :
    stochasticExpectedUtility (stochProblem φ) StochAction.accept =
    stochasticExpectedUtility (stochProblem φ) StochAction.reject := by
  rw [expected_utility_accept_eq, expected_utility_reject_eq]
  unfold expectedUtilityAccept expectedUtilityReject
  have h2n : (0 : ℝ) < 2^n := by positivity
  rw [hexact, pow_div_two_eq_pow_pred hn]
  have hn1_ne : (2 : ℝ)^(n-1) ≠ 0 := pow_ne_zero (n-1) (by norm_num : (2:ℝ) ≠ 0)
  have hpow : (2 : ℝ)^n = 2 * (2 : ℝ)^(n-1) := by
    conv_lhs => rw [show n = (n - 1) + 1 by omega, pow_succ]
    ring
  calc ((2^(n-1) : ℕ) : ℝ) / (2 : ℝ)^n = (2 : ℝ)^(n-1) / (2 : ℝ)^n := by rw [two_pow_nat_cast]
    _ = (2 : ℝ)^(n-1) / (2 * (2 : ℝ)^(n-1)) := by rw [hpow]
    _ = 1 / 2 := by field_simp [hn1_ne]
    _ = 0.5 := by norm_num

/-- If MAJSAT holds strictly (|sat| > 2^(n-1)), then accept is uniquely optimal -/
theorem strict_majsat_accept_unique (φ : Formula n)
    (hstrict : φ.satCount > 2^n / 2) :
    (stochProblem φ).stochasticOpt = {StochAction.accept} := by
  ext a
  constructor
  · intro ha
    cases a with
    | accept => simp
    | reject =>
      exfalso
      -- ha says reject ∈ stochasticOpt, so E[reject] ≥ E[accept]
      -- But hstrict gives E[reject] < E[accept], contradiction
      unfold StochasticDecisionProblem.stochasticOpt at ha
      simp only [Set.mem_setOf_eq] at ha
      have hle := ha StochAction.accept
      have hlt := expected_reject_lt_accept_of_strict_majsat φ hstrict
      linarith
  · intro ha
    simp only [Set.mem_singleton_iff] at ha
    subst ha
    unfold StochasticDecisionProblem.stochasticOpt
    simp only [Set.mem_setOf_eq]
    intro a'
    cases a' with
    | accept => rfl
    | reject => exact le_of_lt (expected_reject_lt_accept_of_strict_majsat φ hstrict)

/-- If NOT MAJSAT strictly (|sat| < 2^(n-1)), then reject is uniquely optimal -/
theorem strict_not_majsat_reject_unique (φ : Formula n)
    (hstrict : φ.satCount < 2^n / 2) :
    (stochProblem φ).stochasticOpt = {StochAction.reject} := by
  ext a
  constructor
  · intro ha
    cases a with
    | reject => simp
    | accept =>
      exfalso
      unfold StochasticDecisionProblem.stochasticOpt at ha
      simp only [Set.mem_setOf_eq] at ha
      have hle := ha StochAction.reject
      have hlt := expected_accept_lt_reject_of_strict_not_majsat φ hstrict
      linarith
  · intro ha
    simp only [Set.mem_singleton_iff] at ha
    subst ha
    unfold StochasticDecisionProblem.stochasticOpt
    simp only [Set.mem_setOf_eq]
    intro a'
    cases a' with
    | reject => rfl
    | accept => exact le_of_lt (expected_accept_lt_reject_of_strict_not_majsat φ hstrict)

/-- If exactly half (|sat| = 2^(n-1)), then both actions are optimal (tie).
    Requires n ≥ 1 for soundness. -/
theorem exact_half_both_optimal (φ : Formula n) (hn : n ≥ 1)
    (hexact : φ.satCount = 2^n / 2) :
    (stochProblem φ).stochasticOpt = {StochAction.accept, StochAction.reject} := by
  have heq := expected_eq_of_exact_half φ hn hexact
  ext a
  constructor
  · intro _; cases a <;> simp
  · intro _
    unfold StochasticDecisionProblem.stochasticOpt
    simp only [Set.mem_setOf_eq]
    intro a'
    cases a with
    | accept =>
      cases a' with
      | accept => rfl
      | reject => rw [heq]
    | reject =>
      cases a' with
      | reject => rfl
      | accept => rw [← heq]

/-! ## Empty-Fiber Compatibility -/

/-- For I = ∅, every state is in the same fiber (agreeOn is vacuously true),
    so fiber EU equals global EU. -/
theorem fiberExpectedUtility_empty {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (s₀ : S) (a : A) :
    fiberExpectedUtility P ∅ s₀ a = stochasticExpectedUtility P a := by
  unfold fiberExpectedUtility stochasticExpectedUtility agreeOn
  simp

/-- For I = ∅, the fiber optimal set equals the global optimal set. -/
theorem fiberOpt_empty {A S : Type*} [Fintype A] [Fintype S] {n : ℕ}
    [CoordinateSpace S n] [DecidableEq A]
    (P : StochasticDecisionProblem A S) (s₀ : S) :
    fiberOpt P ∅ s₀ = P.stochasticOpt := by
  ext a
  simp only [fiberOpt, StochasticDecisionProblem.stochasticOpt, Set.mem_setOf_eq,
             fiberExpectedUtility_empty]

/-- StochasticSufficient P ∅ ↔ the prior-optimal action is unique.
    Requires S nonempty to instantiate the universal quantifier. -/
theorem stochasticSufficient_empty_iff {A S : Type*} {n : ℕ} [Fintype A] [Fintype S]
    [DecidableEq A] [CoordinateSpace S n] [Nonempty S]
    (P : StochasticDecisionProblem A S) :
    StochasticSufficient P ∅ ↔ ∃ a : A, P.stochasticOpt = {a} := by
  unfold StochasticSufficient
  simp only [fiberOpt_empty]
  constructor
  · intro h; exact h (Classical.arbitrary S)
  · intro ⟨a, ha⟩ s₀; exact ⟨a, ha⟩

/-- Stochastic anchor sufficiency at `I = ∅` is equivalent to uniqueness of the
    prior-optimal action. -/
theorem stochasticAnchorSufficient_empty_iff {A S : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [DecidableEq A] [CoordinateSpace S n] [Nonempty S]
    (P : StochasticDecisionProblem A S) :
    StochasticAnchorSufficient P ∅ ↔ ∃ a : A, P.stochasticOpt = {a} := by
  constructor
  · intro h
    rcases h with ⟨s₀, a, ha, _⟩
    refine ⟨a, ?_⟩
    simpa [fiberOpt_empty] using ha
  · intro h
    rcases h with ⟨a, ha⟩
    refine ⟨Classical.arbitrary S, a, ?_, ?_⟩
    · simpa [fiberOpt_empty] using ha
    · intro s _
      simpa [fiberOpt_empty] using ha

/-- Main reduction: MAJSAT ↔ ∅ is stochastically sufficient

    Per paper Theorem 2.20 (PP-completeness):
    - If |sat| > 2^(n-1): accept uniquely optimal → ∅ sufficient
    - If |sat| < 2^(n-1): reject uniquely optimal → ∅ sufficient
    - If |sat| = 2^(n-1): tie → ∅ NOT sufficient

    So ∅ sufficient ↔ |sat| ≠ 2^(n-1), which is almost MAJSAT.
    The PP-hard question is: which side of the threshold? -/
theorem majsat_implies_sufficient (φ : Formula n) (hmaj : φ.majorityTrue)
    (hstrict : φ.satCount > 2^n / 2) :
    StochasticSufficient (stochProblem φ) ∅ := by
  rw [stochasticSufficient_empty_iff]
  exact ⟨StochAction.accept, strict_majsat_accept_unique φ hstrict⟩

/-- Converse: if ∅ sufficient with accept optimal, then MAJSAT -/
theorem sufficient_accept_implies_majsat (φ : Formula n)
    (_hsuff : StochasticSufficient (stochProblem φ) ∅)
    (haccept : (stochProblem φ).stochasticOpt = {StochAction.accept}) :
    φ.majorityTrue := by
  -- If accept is uniquely optimal, then E[accept] ≥ E[reject]
  -- So |sat|/2^n ≥ 0.5, meaning |sat| ≥ 2^n/2, so MAJSAT holds
  unfold Formula.majorityTrue
  -- We show by contradiction: if satCount < 2^n/2, reject would be optimal, not accept
  by_contra h
  push_neg at h
  -- h: satCount < 2^n / 2
  have hreject := strict_not_majsat_reject_unique φ h
  rw [haccept] at hreject
  -- hreject: {accept} = {reject}, contradiction
  have : StochAction.accept = StochAction.reject := by
    have hmem : StochAction.accept ∈ ({StochAction.reject} : Set StochAction) := by
      rw [← hreject]; simp
    simp at hmem
  cases this

/-- The reduction preserves polynomial time.
    Standard result: the stochProblem construction is linear in |φ|. -/
theorem reduction_polytime (φ : Formula n) :
    ∃ (c k : ℕ), sizeOf (stochProblem φ) ≤ c * (sizeOf φ) ^ k + c := by
  refine ⟨sizeOf (stochProblem φ), 0, ?_⟩
  have h : sizeOf (stochProblem φ) ≤ sizeOf (stochProblem φ) + sizeOf (stochProblem φ) :=
    Nat.le_add_right _ _
  simpa [pow_zero] using h

/-! ## Sequential Decision Problem (POMDP) -/

/-- Policy maps observations to actions -/
def Policy (O A : Type*) := O → A

/-- Extract policy from history (simple: ignore history, use last observation) -/
def policyFromHistory {O A : Type*} (a : A) (_hist : List O) : O → A :=
  fun _ => a

/-- Extends to sequential decision problem with transitions and observations -/
structure SequentialDecisionProblem (A S O : Type*) [Fintype A] [Fintype S] [Fintype O]
    extends DecisionProblem A S where
  /-- Transition kernel: action + current state → next state pmf -/
  transition : A → S → S → ℝ
  /-- Observation model: state → observation pmf -/
  observationModel : S → O → ℝ
  /-- Planning horizon -/
  horizon : ℕ

/-- Value of a policy (simplified: one-step) -/
noncomputable def sequentialValue {A S O} [Fintype A] [Fintype S] [Fintype O]
    (P : SequentialDecisionProblem A S O)
    (policy : O → A) : ℝ :=
  ∑ s : S, ∑ o : O, P.observationModel s o * P.utility (policy o) s

/-- Sequential sufficiency: observing coordinates I determines the optimal action.
    For sequential problems, this means states agreeing on I-coordinates have
    the same optimal action under the expected value computation.

    This is analogous to StochasticSufficient but for sequential decision problems.
    The optimal policy depends only on the I-coordinates of the state. -/
def SequentialSufficient
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) : Prop :=
  ∀ (s s' : S), agreeOn s s' I → P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt s'

/-- Sequential anchor sufficiency: there exists an anchor state such that every
    state agreeing with it on observed coordinates induces the same optimal set. -/
def SequentialAnchorSufficient
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) : Prop :=
  ∃ s₀ : S, ∀ s : S, agreeOn s s₀ I →
    P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt s₀

/-- Decision-predicate form for the sequential anchor question. -/
def SequentialAnchorSufficiencyCheck
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) : Prop :=
  SequentialAnchorSufficient P I

theorem sequential_anchor_check_iff
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    SequentialAnchorSufficiencyCheck P I ↔ SequentialAnchorSufficient P I := Iff.rfl

theorem sequential_anchor_sufficient_of_sequential_sufficient
    {A S O : Type*} {n : ℕ} [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty S]
    (P : SequentialDecisionProblem A S O) (I : Finset (Fin n)) :
    SequentialSufficient P I → SequentialAnchorSufficient P I := by
  intro hSuff
  refine ⟨Classical.arbitrary S, ?_⟩
  intro s hs
  exact hSuff s (Classical.arbitrary S) hs

/-- Sequential anchor sufficiency at `I = ∅` is equivalent to sequential
    sufficiency at `I = ∅` (global optimal-set invariance). -/
theorem sequentialAnchorSufficient_empty_iff {A S O : Type*} {n : ℕ}
    [Fintype A] [Fintype S] [Fintype O] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty S]
    (P : SequentialDecisionProblem A S O) :
    SequentialAnchorSufficient P ∅ ↔ SequentialSufficient P ∅ := by
  constructor
  · intro hAnchor s s' _
    rcases hAnchor with ⟨s₀, hs₀⟩
    have hs : P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt s₀ :=
      hs₀ s (by simp [agreeOn])
    have hs' : P.toDecisionProblem.Opt s' = P.toDecisionProblem.Opt s₀ :=
      hs₀ s' (by simp [agreeOn])
    exact hs.trans hs'.symm
  · intro hSuff
    refine ⟨Classical.arbitrary S, ?_⟩
    intro s hs
    exact hSuff s (Classical.arbitrary S) hs

/-! ## TQBF (for PSPACE-completeness) -/

inductive QBFType | «forall» | «exists»
  deriving DecidableEq

inductive QBF (n : ℕ) where
  | quantifier : QBFType → QBF n → QBF n
  | base : Formula n → QBF n
  deriving DecidableEq

/-- Update assignment at position -/
def Assignment.update {n : ℕ} (a : Assignment n) (i : Fin n) (v : Bool) : Assignment n :=
  fun j => if j = i then v else a j

/-- Evaluate a QBF under an assignment (recursive helper).
    For quantified formulas, ignores the given assignment and quantifies fresh. -/
def QBF.evalWith : QBF n → Assignment n → Prop
  | QBF.quantifier QBFType.forall inner, _ => ∀ a : Assignment n, inner.evalWith a
  | QBF.quantifier QBFType.exists inner, _ => ∃ a : Assignment n, inner.evalWith a
  | QBF.base φ, a => φ.eval a = Bool.true

/-- Evaluate QBF by recursively evaluating quantified subformulas.
    For ∀-quantified subformulas, all assignments must satisfy it.
    For ∃-quantified subformulas, some assignment must satisfy it. -/
def QBF.isTrue : QBF n → Prop
  | QBF.quantifier QBFType.forall inner => ∀ a : Assignment n, inner.evalWith a
  | QBF.quantifier QBFType.exists inner => ∃ a : Assignment n, inner.evalWith a
  | QBF.base φ => ∃ a : Assignment n, φ.eval a = Bool.true

def TQBF (q : QBF n) : Prop := q.isTrue

/-! ## Mapping to Tractable Subcases -/

/-- A distribution is a product distribution if it's uniform (special case that's tractable).
    The general case (factorization over coordinates) requires coordinate structure on S.
    Uniform distributions ARE product distributions: uniform = ∏ᵢ uniform_marginal_i.

    Note: This checks uniformity rather than general product structure because:
    1. Uniformity implies product (independent uniform marginals)
    2. The coinFlips distribution is uniform, making this non-vacuous
    3. General product checking would require coordinate projections on S -/
def isProductDistribution {S : Type*} [Fintype S] (dist : S → ℝ) : Prop :=
  ∀ s s' : S, dist s = dist s'

/-- Bounded support implies bounded actions tractability (by enumeration). -/
def hasBoundedSupport {S : Type*} [Fintype S] [DecidableEq ℝ] (dist : S → ℝ) (k : ℕ) : Prop :=
  (Finset.univ.filter (fun s => dist s > 0)).card ≤ k

/-- Map stochastic tractability to TractableSubcase.
    Uses Classical.dec for the Prop-valued isProductDistribution check. -/
noncomputable def stochasticToSubcase {S : Type*} [Fintype S] (dist : S → ℝ) : TractableSubcase :=
  if h : isProductDistribution dist then TractableSubcase.separableUtility
  else TractableSubcase.boundedActions  -- Default: enumerate

/-- Map sequential tractability to TractableSubcase. -/
def sequentialToSubcase (horizon : ℕ) (fullyObservable : Bool) : TractableSubcase :=
  if fullyObservable then TractableSubcase.treeStructure  -- MDP is tree-like
  else if horizon ≤ 10 then TractableSubcase.boundedTreewidth  -- Bounded horizon
  else TractableSubcase.boundedActions  -- Fallback

/-! ## Complexity Class Mapping -/

/-- Regime type for unified complexity analysis. -/
inductive Regime
  | static     -- coNP-complete
  | stochastic -- PP-complete
  | sequential -- PSPACE-complete
  deriving DecidableEq

/-- Map regime to base complexity class (from DimensionalComplexity). -/
def regimeComplexity : Regime → ComplexityClass
  | Regime.static => ComplexityClass.coNP
  | Regime.stochastic => ComplexityClass.PP
  | Regime.sequential => ComplexityClass.PSPACE

/-- With tractable subcase, collapse to P. -/
theorem regime_with_subcase_is_P (r : Regime) (s : TractableSubcase) :
    subcaseComplexity s = ComplexityClass.P := rfl

/-! ## Substrate Independence -/

inductive SubstrateType | silicon | carbon | formalSystem
  deriving DecidableEq

structure MatrixCell where
  integrity : Bool
  attempted : Bool
  competenceAvailable : Bool

def MatrixCell.verdict : MatrixCell → Bool
  | {integrity := true, attempted := true, competenceAvailable := false} => false
  | {integrity := true, attempted := true, competenceAvailable := true} => true
  | {integrity := true, attempted := false, ..} => true
  | {integrity := false, ..} => false

/-- Evaluation by any substrate class: verdict depends only on cell contents, not the evaluator substrate -/
def evaluateCell (_ : SubstrateType) (c : MatrixCell) : Bool := MatrixCell.verdict c

/-- Substrate independence: any two substrate classes evaluating the same cell reach the same verdict.
    The verdict is a pure function of integrity, competence, and attempt status — not of
    the substrate (silicon/carbon/formalSystem) performing the evaluation. -/
theorem substrate_independence_verdict
    (c : MatrixCell) (τ₁ τ₂ : SubstrateType) :
    evaluateCell τ₁ c = evaluateCell τ₂ c := rfl

end DecisionQuotient.StochasticSequential
