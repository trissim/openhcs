/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/BlindDockingTractability.lean - Complexity of Blind vs Guided Docking

  KEY INSIGHT: Blind docking = Pocket Detection + Guided Docking
-/

import DecisionQuotient.Basic
import DecisionQuotient.Computation.PocketDetection
import Mathlib.Data.Real.Basic

namespace DecisionQuotient
namespace Tractability
namespace BlindDocking

open Computation

/-- Composite tractability statement for blind docking.

    Long-term, the two input propositions should be replaced by concrete
    complexity statements for pocket detection and guided docking. -/
def BlindDockingTractable
    (pocketDetectionTractable guidedDockingTractable : Prop) : Prop :=
  pocketDetectionTractable ∧ guidedDockingTractable

/--
  A guided docking problem is structurally bounded if its decision-theoretic
  structural rank admits a finite explicit upper bound.
-/
def GuidedDockingSrankBounded (prob : MolecularSrank.MDBindingProblem) : Prop :=
  ∃ K : ℕ,
    @DecisionProblem.srank
      MolecularSrank.MDAction
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates prob)
      (MolecularSrank.mdCoordinateSpaceStruct prob)
      prob.toDecisionProblem ≤ K

/--
  A detected pocket yields an explicit guided-docking srank bound once it is
  bridged into the `MolecularSrank` model.
-/
theorem guided_docking_srank_bounded_of_detected_pocket
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    [Fintype MolecularSrank.MDAction]
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = PocketDetection.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (hStrictAll : ∀ s, ∃ a, Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob) :
    GuidedDockingSrankBounded prob := by
  refine ⟨3 * MolecularSrank.numRelevantAtoms prob + 3 * ligand.numAtoms, ?_⟩
  exact PocketDetection.detected_pocket_md_srank_bound
    input pocket ligand cutoff utility prob hProb hStrictAll hBounded

/--
  Blind docking is tractable once pocket detection is tractable and the guided
  docking stage induced by a detected pocket has an explicit srank bound.
-/
theorem blind_docking_tractable_of_detected_pocket
    {pocketDetectionTractable : Prop}
    (hDetect : pocketDetectionTractable)
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    [Fintype MolecularSrank.MDAction]
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = PocketDetection.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (hStrictAll : ∀ s, ∃ a, Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob) :
    BlindDockingTractable pocketDetectionTractable (GuidedDockingSrankBounded prob) := by
  refine ⟨hDetect, ?_⟩
  exact guided_docking_srank_bounded_of_detected_pocket
    input pocket ligand cutoff utility prob hProb hStrictAll hBounded

/--
  End-to-end bridge: a detected pocket can certify both a concrete binding
  witness and a tractable guided-docking instance.
-/
theorem detected_pocket_binding_and_tractability
    {pocketDetectionTractable : Prop}
    (hDetect : pocketDetectionTractable)
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (r_ligand : ℝ)
    (hLigand : r_ligand ≤ input.probeRadius)
    (hClear : ImplicitSurface.HasLocalClearance pocket.concaveRegion r_ligand hLigand)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    [Fintype MolecularSrank.MDAction]
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = PocketDetection.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (hStrictAll : ∀ s, ∃ a, Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob) :
    BlindDockingTractable pocketDetectionTractable (GuidedDockingSrankBounded prob) ∧
    ∃ (p : _), p ∈ pocket.concaveRegion.points ∧
      ∃ (q : _),
        Geometry3D.distance q p.position ≤ input.probeRadius - r_ligand ∧
        ImplicitSurface.vdwDistance input.mol q ≥ r_ligand := by
  constructor
  · exact blind_docking_tractable_of_detected_pocket
      hDetect input pocket ligand cutoff utility prob hProb hStrictAll hBounded
  · exact PocketDetection.concave_region_is_binding_competent
      input pocket.concaveRegion r_ligand hLigand hClear

/--
  Blind docking is tractable whenever both pocket detection and guided docking
  are tractable. This removes the previous global axiom and makes the remaining
  proof obligations explicit.
-/
theorem blind_docking_tractable
    {pocketDetectionTractable guidedDockingTractable : Prop}
    (hDetect : pocketDetectionTractable)
    (hGuide : guidedDockingTractable) :
    BlindDockingTractable pocketDetectionTractable guidedDockingTractable :=
  ⟨hDetect, hGuide⟩

end BlindDocking
end Tractability
end DecisionQuotient
