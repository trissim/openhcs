/-
  Paper 4: Stochastic/Sequential Regimes

  NPPPHardness.lean - Toward NP^PP-hardness for stochastic existential queries.

  This file introduces the natural source problem for the stochastic anchor
  existential layer: `∃x . MAJ_y φ(x,y)`. The reduction target reuses the
  three-action pure-anchor gadget, but now over fibers determined by the
  existential coordinates `x`.

  The goal is to expose the exact quantifier pattern behind the stochastic
  existential variants in a machine-checked form. Full oracle-class packaging is
  intentionally out of scope here.
-/

import DecisionQuotient.Hardness.QBF
import DecisionQuotient.StochasticSequential.OracleUpperBounds
import DecisionQuotient.StochasticSequential.PolynomialReduction
import DecisionQuotient.Physics.AnchorChecks
import Mathlib.Data.Finset.Card
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Fintype.Pi
import Mathlib.Tactic

namespace DecisionQuotient
namespace StochasticSequential

open Classical
open BigOperators

abbrev XAssign (nx : ℕ) := AssignX nx
abbrev YAssign (ny : ℕ) := AssignY ny

/-- Source problem for the stochastic existential layer: `∃x` such that a
majority of `y` assignments satisfy the formula. -/
structure ExistsMajorityFormula where
  nx : ℕ
  ny : ℕ
  formula : QFormula (Sum (Fin nx) (Fin ny))

namespace ExistsMajorityFormula

/-- Fix the existential assignment `x`, yielding a plain Boolean formula over
the `y` variables. -/
def fixX {nx ny : ℕ} (x : XAssign nx) : QFormula (Sum (Fin nx) (Fin ny)) → Formula ny
  | QFormula.var (.inl i) => if x i then Formula.top else Formula.bot
  | QFormula.var (.inr j) => Formula.var j
  | QFormula.not φ => Formula.not (fixX x φ)
  | QFormula.and φ ψ => Formula.and (fixX x φ) (fixX x ψ)
  | QFormula.or φ ψ => Formula.or (fixX x φ) (fixX x ψ)

@[simp] theorem eval_fixX {nx ny : ℕ} (x : XAssign nx) (y : YAssign ny)
    (φ : QFormula (Sum (Fin nx) (Fin ny))) :
    Formula.eval y (fixX x φ) = QFormula.eval (ExistsForallFormula.sumAssignment x y) φ := by
  induction φ with
  | var v =>
      cases v with
      | inl i =>
          by_cases h : x i = true
          · simp [fixX, QFormula.eval, ExistsForallFormula.sumAssignment, Formula.eval, h]
          · simp [fixX, QFormula.eval, ExistsForallFormula.sumAssignment, Formula.eval, h]
      | inr j => simp [fixX, QFormula.eval, ExistsForallFormula.sumAssignment, Formula.eval]
  | not φ ih => simp [fixX, Formula.eval, QFormula.eval, ih]
  | and φ ψ ihφ ihψ => simp [fixX, Formula.eval, QFormula.eval, ihφ, ihψ]
  | or φ ψ ihφ ihψ => simp [fixX, Formula.eval, QFormula.eval, ihφ, ihψ]

/-- Majority over the universal/majority block after fixing `x`. -/
def majorityAt (φ : ExistsMajorityFormula) (x : XAssign φ.nx) : Prop :=
  Formula.majorityTrue (fixX x φ.formula)

/-- The `∃·MAJ` source predicate. -/
def satisfiable (φ : ExistsMajorityFormula) : Prop :=
  ∃ x : XAssign φ.nx, majorityAt φ x

end ExistsMajorityFormula

/-- Product-form stochastic states `(x,y)` expose the existential and majority
blocks explicitly. -/
abbrev EMState (nx ny : ℕ) := XAssign nx × YAssign ny

instance emStateCoordinateSpace (nx ny : ℕ) : CoordinateSpace (EMState nx ny) (nx + ny) where
  Coord := fun _ => Bool
  proj := fun s i => Fin.addCases s.1 s.2 i

/-- The observed coordinates for the existential block `x`. -/
def xCoordsEM (nx ny : ℕ) : Finset (Fin (nx + ny)) :=
  Finset.univ.image (Fin.castAdd ny)

theorem agreeOn_xCoordsEM_iff
    {nx ny : ℕ} (s s' : EMState nx ny) :
    agreeOn s s' (xCoordsEM nx ny) ↔ s.1 = s'.1 := by
  constructor
  · intro h
    funext i
    have := h (Fin.castAdd ny i) (by simp [xCoordsEM])
    simpa [emStateCoordinateSpace, CoordinateSpace.proj] using this
  · intro h i hi
    rcases Finset.mem_image.mp hi with ⟨j, _, rfl⟩
    simpa [emStateCoordinateSpace, CoordinateSpace.proj, h]

/-- Uniform distribution over all `(x,y)` states. -/
noncomputable def emDistribution (nx ny : ℕ) : EMState nx ny → ℝ :=
  fun _ => 1 / (2 : ℝ) ^ (nx + ny)

/-- Utility for the existential-majority reduction. -/
noncomputable def emPureAnchorUtility (φ : ExistsMajorityFormula) :
    PureAnchorStochAction → EMState φ.nx φ.ny → ℝ
  | PureAnchorStochAction.accept, s =>
      if QFormula.eval (ExistsForallFormula.sumAssignment s.1 s.2) φ.formula then 1 else 0
  | PureAnchorStochAction.holdL, _ => pureAnchorThreshold φ.ny
  | PureAnchorStochAction.holdR, _ => pureAnchorThreshold φ.ny

/-- Stochastic decision problem for `∃x · MAJ_y`. -/
noncomputable def reduceExistsMajorityPureAnchor (φ : ExistsMajorityFormula) :
    StochasticDecisionProblem PureAnchorStochAction (EMState φ.nx φ.ny) :=
  { toDecisionProblem := { utility := emPureAnchorUtility φ }
    distribution := emDistribution φ.nx φ.ny }

/-- Polynomial size bound for the new reduction. -/
theorem reduceExistsMajorityPureAnchor_poly_bound (φ : ExistsMajorityFormula) :
    ∃ (c k : ℕ),
      sizeOf (reduceExistsMajorityPureAnchor φ) ≤ c * (sizeOf φ) ^ k + c := by
  refine ⟨sizeOf (reduceExistsMajorityPureAnchor φ), 0, ?_⟩
  have h : sizeOf (reduceExistsMajorityPureAnchor φ) ≤
      sizeOf (reduceExistsMajorityPureAnchor φ) + sizeOf (reduceExistsMajorityPureAnchor φ) :=
    Nat.le_add_right _ _
  simpa [pow_zero] using h

theorem fiberExpected_accept_eq_scaled
    (φ : ExistsMajorityFormula) (x : XAssign φ.nx) (y₀ : YAssign φ.ny) :
    fiberExpectedUtility (reduceExistsMajorityPureAnchor φ) (xCoordsEM φ.nx φ.ny) (x, y₀)
      PureAnchorStochAction.accept
      = (1 / (2 : ℝ) ^ φ.nx) *
          stochasticExpectedUtility
            (reduceMAJSATPureAnchor (ExistsMajorityFormula.fixX x φ.formula))
            PureAnchorStochAction.accept := by
  unfold fiberExpectedUtility stochasticExpectedUtility reduceExistsMajorityPureAnchor
    emDistribution emPureAnchorUtility reduceMAJSATPureAnchor stochDistribution pureAnchorUtility
  rw [Fintype.sum_prod_type]
  simp_rw [agreeOn_xCoordsEM_iff]
  have hsplit :
      ∀ x' : XAssign φ.nx,
        (if x' = x then
          ∑ y : YAssign φ.ny,
            1 / (2 : ℝ) ^ (φ.nx + φ.ny) *
              (if QFormula.eval (ExistsForallFormula.sumAssignment x' y) φ.formula then 1 else 0)
        else 0)
        =
        if x' = x then
          (1 / (2 : ℝ) ^ φ.nx) *
            ∑ y : YAssign φ.ny,
              1 / (2 : ℝ) ^ φ.ny *
                (if Formula.eval y (ExistsMajorityFormula.fixX x φ.formula) then 1 else 0)
        else 0 := by
    intro x'
    by_cases hx' : x' = x
    · subst x'
      rw [if_pos rfl, if_pos rfl]
      simp_rw [ExistsMajorityFormula.eval_fixX]
      have hpow : (2 : ℝ) ^ (φ.nx + φ.ny) = (2 : ℝ) ^ φ.nx * (2 : ℝ) ^ φ.ny := by
        rw [pow_add]
      simp_rw [hpow]
      rw [Finset.mul_sum]
      apply Finset.sum_congr rfl
      intro y _
      have hnx : (2 : ℝ) ^ φ.nx ≠ 0 := by positivity
      field_simp [hnx]
    · rw [if_neg hx', if_neg hx']
  calc
    (∑ x' : XAssign φ.nx,
        ∑ y : YAssign φ.ny,
          if x' = x then
            1 / (2 : ℝ) ^ (φ.nx + φ.ny) *
              (if QFormula.eval (ExistsForallFormula.sumAssignment x' y) φ.formula then 1 else 0)
          else 0)
      = (∑ x' : XAssign φ.nx,
          if x' = x then
            ∑ y : YAssign φ.ny,
              1 / (2 : ℝ) ^ (φ.nx + φ.ny) *
                (if QFormula.eval (ExistsForallFormula.sumAssignment x' y) φ.formula then 1 else 0)
          else 0) := by
            apply Fintype.sum_congr
            intro x'
            by_cases hx'' : x' = x
            · simp [hx'']
            · simp [hx'']
    _
      = ∑ x' : XAssign φ.nx,
          if x' = x then
            (1 / (2 : ℝ) ^ φ.nx) *
              ∑ y : YAssign φ.ny,
                1 / (2 : ℝ) ^ φ.ny *
                  (if Formula.eval y (ExistsMajorityFormula.fixX x φ.formula) then 1 else 0)
          else 0 := by
            apply Fintype.sum_congr
            intro x'
            exact hsplit x'
    _ = (1 / (2 : ℝ) ^ φ.nx) *
          ∑ y : YAssign φ.ny,
            1 / (2 : ℝ) ^ φ.ny *
              (if Formula.eval y (ExistsMajorityFormula.fixX x φ.formula) then 1 else 0) := by
            simp

theorem fiberExpected_hold_eq_scaled
    (φ : ExistsMajorityFormula) (x : XAssign φ.nx) (y₀ : YAssign φ.ny)
    (a : PureAnchorStochAction)
    (ha : a = PureAnchorStochAction.holdL ∨ a = PureAnchorStochAction.holdR) :
    fiberExpectedUtility (reduceExistsMajorityPureAnchor φ) (xCoordsEM φ.nx φ.ny) (x, y₀) a
      = (1 / (2 : ℝ) ^ φ.nx) *
          stochasticExpectedUtility
            (reduceMAJSATPureAnchor (ExistsMajorityFormula.fixX x φ.formula)) a := by
  rcases ha with rfl | rfl
  · unfold fiberExpectedUtility stochasticExpectedUtility reduceExistsMajorityPureAnchor
      emDistribution emPureAnchorUtility reduceMAJSATPureAnchor stochDistribution pureAnchorUtility pureAnchorThreshold
    rw [Fintype.sum_prod_type]
    simp_rw [agreeOn_xCoordsEM_iff]
    have hcollapse :
        ∀ x' : XAssign φ.nx,
          (∑ y : YAssign φ.ny,
            if x' = x then 1 / (2 : ℝ) ^ (φ.nx + φ.ny) * pureAnchorThreshold φ.ny else 0)
          = if x' = x then
              (1 / (2 : ℝ) ^ φ.nx) *
                ∑ y : YAssign φ.ny, 1 / (2 : ℝ) ^ φ.ny * pureAnchorThreshold φ.ny
            else 0 := by
      intro x'
      by_cases hx' : x' = x
      · subst x'
        rw [if_pos rfl, if_pos rfl]
        have hpow : (2 : ℝ) ^ (φ.nx + φ.ny) = (2 : ℝ) ^ φ.nx * (2 : ℝ) ^ φ.ny := by
          rw [pow_add]
        simp_rw [hpow]
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro y _
        have hnx : (2 : ℝ) ^ φ.nx ≠ 0 := by positivity
        field_simp [hnx]
      · simp [hx']
    calc
      (∑ x' : XAssign φ.nx,
          ∑ y : YAssign φ.ny,
            if x' = x then 1 / (2 : ℝ) ^ (φ.nx + φ.ny) * pureAnchorThreshold φ.ny else 0)
        = ∑ x' : XAssign φ.nx,
            if x' = x then
              (1 / (2 : ℝ) ^ φ.nx) *
                ∑ y : YAssign φ.ny, 1 / (2 : ℝ) ^ φ.ny * pureAnchorThreshold φ.ny
            else 0 := by
              apply Fintype.sum_congr
              intro x'
              exact hcollapse x'
      _ = (1 / (2 : ℝ) ^ φ.nx) *
            ∑ y : YAssign φ.ny, 1 / (2 : ℝ) ^ φ.ny * pureAnchorThreshold φ.ny := by
              simp
  · unfold fiberExpectedUtility stochasticExpectedUtility reduceExistsMajorityPureAnchor
      emDistribution emPureAnchorUtility reduceMAJSATPureAnchor stochDistribution pureAnchorUtility pureAnchorThreshold
    rw [Fintype.sum_prod_type]
    simp_rw [agreeOn_xCoordsEM_iff]
    have hcollapse :
        ∀ x' : XAssign φ.nx,
          (∑ y : YAssign φ.ny,
            if x' = x then 1 / (2 : ℝ) ^ (φ.nx + φ.ny) * pureAnchorThreshold φ.ny else 0)
          = if x' = x then
              (1 / (2 : ℝ) ^ φ.nx) *
                ∑ y : YAssign φ.ny, 1 / (2 : ℝ) ^ φ.ny * pureAnchorThreshold φ.ny
            else 0 := by
      intro x'
      by_cases hx' : x' = x
      · subst x'
        rw [if_pos rfl, if_pos rfl]
        have hpow : (2 : ℝ) ^ (φ.nx + φ.ny) = (2 : ℝ) ^ φ.nx * (2 : ℝ) ^ φ.ny := by
          rw [pow_add]
        simp_rw [hpow]
        rw [Finset.mul_sum]
        apply Finset.sum_congr rfl
        intro y _
        have hnx : (2 : ℝ) ^ φ.nx ≠ 0 := by positivity
        field_simp [hnx]
      · simp [hx']
    calc
      (∑ x' : XAssign φ.nx,
          ∑ y : YAssign φ.ny,
            if x' = x then 1 / (2 : ℝ) ^ (φ.nx + φ.ny) * pureAnchorThreshold φ.ny else 0)
        = ∑ x' : XAssign φ.nx,
            if x' = x then
              (1 / (2 : ℝ) ^ φ.nx) *
                ∑ y : YAssign φ.ny, 1 / (2 : ℝ) ^ φ.ny * pureAnchorThreshold φ.ny
            else 0 := by
              apply Fintype.sum_congr
              intro x'
              exact hcollapse x'
      _ = (1 / (2 : ℝ) ^ φ.nx) *
            ∑ y : YAssign φ.ny, 1 / (2 : ℝ) ^ φ.ny * pureAnchorThreshold φ.ny := by
              simp

theorem fiberOpt_eq_fixed_pureAnchor_stochasticOpt
    (φ : ExistsMajorityFormula) (x : XAssign φ.nx) (y₀ : YAssign φ.ny) :
    fiberOpt (reduceExistsMajorityPureAnchor φ) (xCoordsEM φ.nx φ.ny) (x, y₀)
      = (reduceMAJSATPureAnchor (ExistsMajorityFormula.fixX x φ.formula)).stochasticOpt := by
  let c : ℝ := 1 / (2 : ℝ) ^ φ.nx
  have hpos : 0 < c := by
    dsimp [c]
    positivity
  have hscale :
      ∀ a : PureAnchorStochAction,
        fiberExpectedUtility (reduceExistsMajorityPureAnchor φ) (xCoordsEM φ.nx φ.ny) (x, y₀) a =
          c * stochasticExpectedUtility
            (reduceMAJSATPureAnchor (ExistsMajorityFormula.fixX x φ.formula)) a := by
    intro a
    cases a with
    | accept =>
        simpa [c] using fiberExpected_accept_eq_scaled φ x y₀
    | holdL =>
        simpa [c] using
          fiberExpected_hold_eq_scaled φ x y₀ PureAnchorStochAction.holdL (Or.inl rfl)
    | holdR =>
        simpa [c] using
          fiberExpected_hold_eq_scaled φ x y₀ PureAnchorStochAction.holdR (Or.inr rfl)
  ext a
  unfold fiberOpt StochasticDecisionProblem.stochasticOpt
  constructor
  · intro ha
    intro a'
    have hle := ha a'
    rw [hscale a', hscale a] at hle
    nlinarith
  · intro ha
    intro a'
    have hle := ha a'
    rw [hscale a', hscale a]
    nlinarith

theorem reduceExistsMajorityPureAnchor_correct
    (φ : ExistsMajorityFormula) (hny : φ.ny ≥ 1) :
    ExistsMajorityFormula.satisfiable φ ↔
      StochasticAnchorSufficiencyCheck (reduceExistsMajorityPureAnchor φ)
        (xCoordsEM φ.nx φ.ny) := by
  constructor
  · intro hsat
    rcases hsat with ⟨x, hmaj⟩
    have huniq :
        (reduceMAJSATPureAnchor (ExistsMajorityFormula.fixX x φ.formula)).stochasticOpt
          = {PureAnchorStochAction.accept} :=
      pureAnchor_accept_unique_of_majsat _ hny hmaj
    rw [Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton]
    refine ⟨(x, fun _ => false), PureAnchorStochAction.accept, ?_⟩
    simpa [fiberOpt_eq_fixed_pureAnchor_stochasticOpt] using huniq
  · intro hanchor
    rw [Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton] at hanchor
    rcases hanchor with ⟨s₀, a, ha⟩
    let x : XAssign φ.nx := s₀.1
    have hfiber :
        fiberOpt (reduceExistsMajorityPureAnchor φ) (xCoordsEM φ.nx φ.ny) s₀
          = (reduceMAJSATPureAnchor (ExistsMajorityFormula.fixX x φ.formula)).stochasticOpt := by
      cases s₀ with
      | mk x' y' => simpa [x] using fiberOpt_eq_fixed_pureAnchor_stochasticOpt φ x' y'
    have hanchor' :
        StochasticAnchorSufficiencyCheck
          (reduceMAJSATPureAnchor (ExistsMajorityFormula.fixX x φ.formula))
          (∅ : Finset (Fin φ.ny)) := by
      rw [Physics.AnchorChecks.stochastic_anchor_check_iff_exists_anchor_singleton]
      refine ⟨fun _ => false, a, ?_⟩
      simpa [hfiber, fiberOpt_empty] using ha
    have hmaj : Formula.majorityTrue (ExistsMajorityFormula.fixX x φ.formula) :=
      (reduceMAJSATPureAnchor_correct (ExistsMajorityFormula.fixX x φ.formula) hny).2 hanchor'
    exact ⟨x, hmaj⟩

abbrev ExistsMajorityInput := {φ : ExistsMajorityFormula // φ.ny ≥ 1}

abbrev ExistsMajorityFixed (nx ny : ℕ) := QFormula (Sum (Fin nx) (Fin ny))

def ExistsMajorityFixedSat {nx ny : ℕ} (φ : ExistsMajorityFixed nx ny) : Prop :=
  ∃ x : XAssign nx, Formula.majorityTrue (ExistsMajorityFormula.fixX x φ)

def ExistsMajorityFixedWitnessRel {nx ny : ℕ}
    (φ : ExistsMajorityFixed nx ny) (x : XAssign nx) : Prop :=
  Formula.majorityTrue (ExistsMajorityFormula.fixX x φ)

theorem existsMajorityFixedWitnessRel_exact {nx ny : ℕ}
    (φ : ExistsMajorityFixed nx ny) (x : XAssign nx) :
    HasExactBoolDecider (ExistsMajorityFixedWitnessRel φ x) := by
  classical
  exact hasExactBoolDecider_of_decidable _

theorem existsMajority_fixed_fits_np_over_ppstyle {nx ny : ℕ} :
    FitsNPOverPPStyle (ExistsMajorityFixed nx ny) (XAssign nx)
      ExistsMajorityFixedSat ExistsMajorityFixedWitnessRel := by
  refine fitsNPOverPPStyle_of_exists_witness
    ExistsMajorityFixedWitnessRel existsMajorityFixedWitnessRel_exact ?_
  intro φ
  rfl

abbrev PackedXWitness := Σ nx : ℕ, XAssign nx

def castXAssign {m n : ℕ} (h : m = n) (x : XAssign m) : XAssign n := by
  subst h
  exact x

def ExistsMajorityPackedWitnessRel
    (q : ExistsMajorityInput) (w : PackedXWitness) : Prop :=
  ∃ h : w.1 = q.1.nx,
    Formula.majorityTrue (ExistsMajorityFormula.fixX (castXAssign h w.2) q.1.formula)

def ExistsMajoritySourceLanguage : Set ExistsMajorityInput :=
  {q | ExistsMajorityFormula.satisfiable q.1}

theorem existsMajorityPackedWitnessRel_exact
    (q : ExistsMajorityInput) (w : PackedXWitness) :
    HasExactBoolDecider (ExistsMajorityPackedWitnessRel q w) := by
  classical
  exact hasExactBoolDecider_of_decidable _

theorem existsMajority_source_fits_np_over_ppstyle :
    FitsNPOverPPStyle ExistsMajorityInput PackedXWitness
      (fun q => q ∈ ExistsMajoritySourceLanguage)
      ExistsMajorityPackedWitnessRel := by
  refine fitsNPOverPPStyle_of_exists_witness
    ExistsMajorityPackedWitnessRel existsMajorityPackedWitnessRel_exact ?_
  intro q
  constructor
  · intro hsat
    rcases hsat with ⟨x, hx⟩
    refine ⟨⟨q.1.nx, x⟩, rfl, ?_⟩
    simpa [castXAssign] using hx
  · intro hw
    rcases hw with ⟨⟨m, x⟩, hm, hx⟩
    cases hm
    refine ⟨x, ?_⟩
    simpa [castXAssign] using hx

def ExistsMajorityAnchorLanguage : Set ExistsMajorityInput :=
  {q | StochasticAnchorSufficiencyCheck (reduceExistsMajorityPureAnchor q.1)
      (xCoordsEM q.1.nx q.1.ny)}

noncomputable def reduceExistsMajorityAnchorQuery
    (q : ExistsMajorityInput) :
    StochasticAnchorQueryInput PureAnchorStochAction (EMState q.1.nx q.1.ny) (q.1.nx + q.1.ny) where
  problem := reduceExistsMajorityPureAnchor q.1
  infoSet := xCoordsEM q.1.nx q.1.ny

abbrev ExistsMajorityAnchorQueryInstance :=
  Σ q : ExistsMajorityInput,
    StochasticAnchorQueryInput PureAnchorStochAction (EMState q.1.nx q.1.ny) (q.1.nx + q.1.ny)

noncomputable instance instSizeOfExistsMajorityAnchorQueryInstance :
    SizeOf ExistsMajorityAnchorQueryInstance where
  sizeOf inst := sizeOf inst.1 + 1

noncomputable def reduceExistsMajorityToAnchorQueryInstance
    (q : ExistsMajorityInput) : ExistsMajorityAnchorQueryInstance :=
  ⟨q, reduceExistsMajorityAnchorQuery q⟩

def ExistsMajorityAnchorQueryLanguage : Set ExistsMajorityAnchorQueryInstance :=
  {inst | StochasticAnchorSufficiencyCheck inst.2.problem inst.2.infoSet}

theorem reduceExistsMajorityToAnchorQueryInstance_poly :
    ∃ (c k : ℕ), ∀ q : ExistsMajorityInput,
      sizeOf (reduceExistsMajorityToAnchorQueryInstance q) ≤ c * (sizeOf q) ^ k + c := by
  refine ⟨2, 1, ?_⟩
  intro q
  change sizeOf q + 1 ≤ 2 * (sizeOf q) ^ 1 + 2
  simp
  omega

theorem reduceExistsMajorityToAnchorQueryInstance_correct
    (q : ExistsMajorityInput) :
    q ∈ ExistsMajoritySourceLanguage ↔
      reduceExistsMajorityToAnchorQueryInstance q ∈ ExistsMajorityAnchorQueryLanguage := by
  change ExistsMajorityFormula.satisfiable q.1 ↔
    StochasticAnchorSufficiencyCheck (reduceExistsMajorityPureAnchor q.1) (xCoordsEM q.1.nx q.1.ny)
  exact reduceExistsMajorityPureAnchor_correct q.1 q.2

/-- Honest reduction wrapper for the existential-majority source problem into the
stochastic anchor query family. The target language is stated on the same input
type, with membership interpreted by the reduced stochastic-anchor instance.
This packages reduction correctness together with trivial polynomial size bounds,
but does not claim a machine-checked standard `NP^PP` hardness theorem. -/
noncomputable def reduceExistsMajority_to_stochastic_anchor_reduction :
    PolyReduction4b ExistsMajorityInput ExistsMajorityInput
      ExistsMajoritySourceLanguage ExistsMajorityAnchorLanguage where
  f := id
  poly_time := by
    refine ⟨1, 1, ?_⟩
    intro q
    simp
  correct := by
    intro q
    change ExistsMajorityFormula.satisfiable q.1 ↔
      StochasticAnchorSufficiencyCheck (reduceExistsMajorityPureAnchor q.1)
        (xCoordsEM q.1.nx q.1.ny)
    exact reduceExistsMajorityPureAnchor_correct q.1 q.2

def HonestExistsMajorityStochasticAnchorHard : Prop :=
  Nonempty
    (PolyReduction4b ExistsMajorityInput ExistsMajorityInput
      ExistsMajoritySourceLanguage ExistsMajorityAnchorLanguage)

theorem existsMajority_stochastic_anchor_hard :
    HonestExistsMajorityStochasticAnchorHard := by
  exact ⟨reduceExistsMajority_to_stochastic_anchor_reduction⟩

/-- Stronger packaging: the existential-majority source problem reduces directly
to the generic stochastic-anchor query family, represented as packaged query
instances. -/
noncomputable def reduceExistsMajority_to_stochastic_anchor_query_family_reduction :
    PolyReduction4b ExistsMajorityInput ExistsMajorityAnchorQueryInstance
      ExistsMajoritySourceLanguage ExistsMajorityAnchorQueryLanguage where
  f := reduceExistsMajorityToAnchorQueryInstance
  poly_time := reduceExistsMajorityToAnchorQueryInstance_poly
  correct := reduceExistsMajorityToAnchorQueryInstance_correct

def HonestExistsMajorityStochasticAnchorQueryFamilyHard : Prop :=
  Nonempty
    (PolyReduction4b ExistsMajorityInput ExistsMajorityAnchorQueryInstance
      ExistsMajoritySourceLanguage ExistsMajorityAnchorQueryLanguage)

theorem existsMajority_stochastic_anchor_query_family_hard :
    HonestExistsMajorityStochasticAnchorQueryFamilyHard := by
  exact ⟨reduceExistsMajority_to_stochastic_anchor_query_family_reduction⟩

/-- Internal hardness notion for this artifact: a target language is hard for a
scoped witness-over-PP-style source if some source language already fitting
`FitsNPOverPPStyle` admits a polynomially size-bounded reduction into it. This
is intentionally weaker and more honest than a standard oracle-TM hardness
definition. -/
structure NPOverPPStyleHardnessWitness (β : Type*) [SizeOf β] (Lβ : Set β) where
  α : Type
  instSizeOfα : SizeOf α
  ω : Type
  Lα : Set α
  R : α → ω → Prop
  fits : FitsNPOverPPStyle α ω (fun x => x ∈ Lα) R
  reduction : let _ : SizeOf α := instSizeOfα; PolyReduction4b α β Lα Lβ

def HonestNPOverPPStyleHard (β : Type*) [SizeOf β] (Lβ : Set β) : Prop :=
  Nonempty (NPOverPPStyleHardnessWitness β Lβ)

theorem existsMajority_anchor_language_honest_np_over_ppstyle_hard :
    HonestNPOverPPStyleHard ExistsMajorityInput ExistsMajorityAnchorLanguage := by
  refine ⟨{
    α := ExistsMajorityInput
    instSizeOfα := inferInstance
    ω := PackedXWitness
    Lα := ExistsMajoritySourceLanguage
    R := ExistsMajorityPackedWitnessRel
    fits := existsMajority_source_fits_np_over_ppstyle
    reduction := reduceExistsMajority_to_stochastic_anchor_reduction
  }⟩

theorem existsMajority_anchor_query_family_honest_np_over_ppstyle_hard :
    HonestNPOverPPStyleHard ExistsMajorityAnchorQueryInstance
      ExistsMajorityAnchorQueryLanguage := by
  refine ⟨{
    α := ExistsMajorityInput
    instSizeOfα := inferInstance
    ω := PackedXWitness
    Lα := ExistsMajoritySourceLanguage
    R := ExistsMajorityPackedWitnessRel
    fits := existsMajority_source_fits_np_over_ppstyle
    reduction := reduceExistsMajority_to_stochastic_anchor_query_family_reduction
  }⟩

end StochasticSequential
end DecisionQuotient
