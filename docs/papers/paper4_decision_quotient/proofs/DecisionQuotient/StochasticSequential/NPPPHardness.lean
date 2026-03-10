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

end StochasticSequential
end DecisionQuotient
