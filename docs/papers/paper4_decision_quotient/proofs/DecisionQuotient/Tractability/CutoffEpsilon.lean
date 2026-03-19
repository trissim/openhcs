/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CutoffEpsilon.lean
  
  Formal linking of the Lattice Sum continuous bound to the decision-theoretic
  epsilon margin cutoff `OutsideCutoffApproximationBounded`.
-/
import DecisionQuotient.Tractability.MolecularSrank
import DecisionQuotient.Tractability.LatticeSum

namespace DecisionQuotient
namespace Tractability

open Tractability.MolecularSrank
open Tractability.LatticeSum
open Classical

/-- 
  A property of a binding problem where the potential is dominated by a bounded lattice tail. 
    1. The maximum possible perturbation from a single atom moving outside the cutoff 
       is bounded by the physical lattice tail sum (e.g. O(1/R^3)).
    2. The minimum required optimal gap is strictly positive.
-/
def SatisfiesBoundedPotential
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (tail_coefficient minimum_gap : ℝ) : Prop :=
  (∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
    (axis : Fin 3) (s s' : MDState) (R : ℝ),
    0 < R →
    (∀ j : Fin (numMDCoordinates prob),
      j ≠ proteinCoordFin prob atomIdx hAtomInProtein axis →
      mdProj prob s j = mdProj prob s' j) →
    ¬ atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
    (∀ a : MDAction, |prob.utility a s - prob.utility a s'| ≤ tail_coefficient * latticeTailSum 6 R))
  ∧
  (0 < minimum_gap)
  ∧
  (∀ (s : MDState) (a_star : MDAction),
    StrictOpt prob.toDecisionProblem a_star s →
    minimum_gap ≤ StrictUtilityGap prob.toDecisionProblem a_star s)

/-- Explicit finite minimum strict gap over all states, using a chosen strict
    optimizer witness at each state. -/
noncomputable def finiteMinimumGap
    (prob : MDBindingProblem)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s }) : ℝ :=
  let gaps : Finset ℝ :=
    (Finset.univ : Finset MDState).image (fun s => StrictUtilityGap prob.toDecisionProblem (w s).1 s)
  gaps.min' <| by
    let s : MDState := Classical.choice ‹Nonempty MDState›
    refine ⟨StrictUtilityGap prob.toDecisionProblem (w s).1 s, ?_⟩
    exact Finset.mem_image.mpr ⟨s, by simp, rfl⟩

theorem finiteMinimumGap_le_strictUtilityGap
    (prob : MDBindingProblem)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s })
    (s : MDState) :
    finiteMinimumGap prob w ≤ StrictUtilityGap prob.toDecisionProblem (w s).1 s := by
  unfold finiteMinimumGap
  apply Finset.min'_le
  exact Finset.mem_image.mpr ⟨s, by simp, rfl⟩

/-- Utility-specific packaging theorem: once the tail perturbation inequality is
    proved for a docking utility, the finite minimum gap yields an explicit
    `SatisfiesBoundedPotential` witness. -/
theorem satisfiesBoundedPotential_of_tailBound_and_finiteGap
    (prob : MDBindingProblem)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (tail_coefficient : ℝ)
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < finiteMinimumGap prob w)
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (numMDCoordinates prob),
        j ≠ proteinCoordFin prob atomIdx hAtomInProtein axis →
        mdProj prob s j = mdProj prob s' j) →
      ¬ atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MDAction, |prob.utility a s - prob.utility a s'| ≤ tail_coefficient * latticeTailSum 6 R)) :
    SatisfiesBoundedPotential prob tail_coefficient (finiteMinimumGap prob w) := by
  constructor
  · intro atomIdx hAtomInProtein axis s s' R hR hSame hOutside
    exact hTail atomIdx hAtomInProtein axis s s' R hR hSame hOutside
  constructor
  · exact hGapPos
  · intro s aStar hStrict
    have hwEq : aStar = (w s).1 := by
      have hOpt1 : prob.toDecisionProblem.Opt s = {aStar} :=
        opt_eq_singleton_of_strict prob.toDecisionProblem aStar s hStrict
      have hOpt2 : prob.toDecisionProblem.Opt s = {(w s).1} :=
        opt_eq_singleton_of_strict prob.toDecisionProblem (w s).1 s (w s).2
      have : ({aStar} : Set MDAction) = {(w s).1} := by simpa [hOpt1] using hOpt2
      simpa using Set.singleton_eq_singleton_iff.mp this
    rw [hwEq]
    exact finiteMinimumGap_le_strictUtilityGap prob w s

/--
  THEOREM: If the structural cutoff R is chosen large enough such that the tail error 
  is strictly less than half the minimum decision gap, the problem satisfies the 
  `OutsideCutoffApproximationBounded` tractability property.
-/
theorem large_cutoff_implies_bounded 
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (tail_coefficient minimum_gap : ℝ)
    (h_bounds : SatisfiesBoundedPotential prob tail_coefficient minimum_gap)
    (h_pos_cutoff : 0 < prob.cutoff)
    (h_cutoff_size : tail_coefficient * latticeTailSum 6 prob.cutoff < minimum_gap / 2)
    (h_tail_pos : 0 ≤ tail_coefficient * latticeTailSum 6 prob.cutoff) :
    OutsideCutoffApproximationBounded prob := by
  intro atomIdx hAtomInProtein axis s s' a_star
  
  -- The delta is exactly the bound from the lattice tail at R = prob.cutoff
  let δ := tail_coefficient * latticeTailSum 6 prob.cutoff
  use δ
  
  constructor
  · exact h_tail_pos
  
  intro h_same h_outside h_opt
  
  constructor
  · -- Bound on |U(s) - U(s')|
    intro a
    exact h_bounds.1 atomIdx hAtomInProtein axis s s' prob.cutoff h_pos_cutoff h_same h_outside a
  
  · -- Bound on δ < Gap / 2
    have h1 : δ < minimum_gap / 2 := h_cutoff_size
    have h2 : minimum_gap / 2 ≤ (StrictUtilityGap prob.toDecisionProblem a_star s) / 2 := by
      apply div_le_div_of_nonneg_right
      · exact h_bounds.2.2 s a_star h_opt
      · linarith
    linarith

end Tractability
end DecisionQuotient
