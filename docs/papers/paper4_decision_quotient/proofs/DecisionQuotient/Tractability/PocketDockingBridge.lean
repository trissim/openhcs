/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/PocketDockingBridge.lean

  This file bridges geometric pocket detection into the molecular structural-rank
  tractability layer. It keeps geometry concerns in `PocketDetection.lean` and
  packages the MD-facing conversions and srank wrappers in one place.
-/

import DecisionQuotient.Computation.PocketDetection
import DecisionQuotient.Tractability.MolecularSrank
import DecisionQuotient.Tractability.SampledDockingCutoff

namespace DecisionQuotient
namespace Tractability
namespace PocketDockingBridge

open Computation

/--
  Convert a detected pocket into the `BindingSite` structure used by
  `MolecularSrank`.
-/
def detected_pocket_to_binding_site
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input) :
    MolecularSrank.BindingSite :=
  {
    center := pocket.boundingSphere.center
    radius := pocket.boundingSphere.radius
  }

/--
  Convert an `ImplicitSurface.Atom` into the atom representation used by
  `MolecularSrank`. This preserves geometry and fills in non-geometric fields
  with neutral defaults until richer chemistry is connected.
-/
def to_molecular_srank_atom
    (index : ℕ)
    (atom : ImplicitSurface.Atom) :
    MolecularSrank.Atom :=
  {
    index := index
    position := atom.position
    charge := 0
    mass := 1
  }

/-- Convert a protein molecule into the `MolecularSrank` representation. -/
def to_molecular_srank_molecule
    (mol : ImplicitSurface.Molecule) :
    MolecularSrank.Molecule :=
  {
    atoms := mol.mapIdx to_molecular_srank_atom
    numAtoms := mol.length
    size_eq := by simp
  }

/--
  Build an `MDBindingProblem` from a detected pocket.

  This is the concrete bridge from pocket detection into the structural-rank
  tractability layer. The ligand model and scoring utility remain parameters,
  since pocket detection itself only certifies the binding site geometry.
-/
def detected_pocket_to_md_binding_problem
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ) :
    MolecularSrank.MDBindingProblem :=
  {
    protein := to_molecular_srank_molecule input.mol
    ligand := ligand
    bindingSite := detected_pocket_to_binding_site input pocket
    cutoff := cutoff
    utility := utility
  }

/-- The MDBindingProblem bridge preserves the detected binding-site geometry. -/
theorem detected_pocket_to_md_binding_problem_binding_site
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ) :
    (detected_pocket_to_md_binding_problem input pocket ligand cutoff utility).bindingSite =
      detected_pocket_to_binding_site input pocket :=
  rfl

/--
  A detected pocket inherits the molecular-docking structural-rank bound once
  it is packaged as an `MDBindingProblem`.
-/
theorem detected_pocket_md_srank_bound
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    [Fintype MolecularSrank.MDAction]
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (hStrictAll : ∀ s, ∃ a,
      Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob) :
    @DecisionProblem.srank
      MolecularSrank.MDAction
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates prob)
      (MolecularSrank.mdCoordinateSpaceStruct prob)
      prob.toDecisionProblem ≤
      3 * MolecularSrank.numRelevantAtoms prob
      + 3 * ligand.numAtoms := by
  subst prob
  simpa using MolecularSrank.md_srank_bound
    (detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    hStrictAll hBounded

/--
  Small-pocket corollary transferred across the detected-pocket bridge.
-/
theorem detected_pocket_small_pocket_low_srank
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    [Fintype MolecularSrank.MDAction]
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (K L : Nat)
    (hStrictAll : ∀ s, ∃ a,
      Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob)
    (hPocket : MolecularSrank.numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank
      MolecularSrank.MDAction
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates prob)
      (MolecularSrank.mdCoordinateSpaceStruct prob)
      prob.toDecisionProblem ≤ 3 * K + 3 * L := by
  subst prob
  simpa using MolecularSrank.small_pocket_low_srank
    (detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    K L hStrictAll hBounded hPocket hLigand

/--
  Equivalent docking-bound formulation transferred across the detected-pocket bridge.
-/
theorem detected_pocket_docking_small_pocket_bound
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    [Fintype MolecularSrank.MDAction]
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (K L : Nat)
    (hStrictAll : ∀ s, ∃ a,
      Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob)
    (hPocket : MolecularSrank.numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank
      MolecularSrank.MDAction
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates prob)
      (MolecularSrank.mdCoordinateSpaceStruct prob)
      prob.toDecisionProblem ≤ 3 * K + 3 * L := by
  subst prob
  simpa using MolecularSrank.docking_small_pocket_bound
    (detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    K L hStrictAll hBounded hPocket hLigand

/--
  Bundled assumptions for transferring a detected pocket into the guided-docking
  structural-rank theorems.
-/
structure GuidedDockingBridgeData
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input) where
  ligand : MolecularSrank.Molecule
  cutoff : ℝ
  utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ

/--
  Bundled proof obligations required to apply the guided-docking srank bounds to
  a detected pocket after bridging into `MolecularSrank`.
-/
structure GuidedDockingBridgeAssumptions
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (data : GuidedDockingBridgeData input pocket)
    [Fintype MolecularSrank.MDAction] where
  prob : MolecularSrank.MDBindingProblem
  prob_eq : prob = detected_pocket_to_md_binding_problem input pocket data.ligand data.cutoff data.utility
  strictAll : ∀ s, ∃ a, Tractability.StrictOpt prob.toDecisionProblem a s
  bounded : MolecularSrank.OutsideCutoffApproximationBounded prob

/--
  Bundled small-pocket assumptions extending the basic guided-docking bridge.
-/
structure SmallPocketBridgeAssumptions
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (data : GuidedDockingBridgeData input pocket)
    [Fintype MolecularSrank.MDAction]
    extends GuidedDockingBridgeAssumptions input pocket data where
  K : Nat
  L : Nat
  pocketBound : MolecularSrank.numRelevantAtoms prob ≤ K
  ligandBound : prob.ligand.numAtoms ≤ L

/--
  Bundled sampled-docking assumptions extending the small-pocket bridge.
-/
structure SampledSmallPocketBridgeAssumptions
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (data : GuidedDockingBridgeData input pocket)
    [Fintype MolecularSrank.MDAction]
    extends SmallPocketBridgeAssumptions input pocket data where
  samples : SampledDocking.SampledActionFamily MolecularSrank.MDAction
  aStar : MolecularSrank.MDAction
  strictAllSampled : ∀ s,
    Tractability.StrictOpt prob.toDecisionProblem aStar s ∨
    ∃ a, Tractability.StrictOpt prob.toDecisionProblem a s
  capture : ∀ s,
    ∃ a : SampledDocking.SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s

/-- Bundle-driven version of the base guided-docking srank bound. -/
theorem guided_docking_srank_bound_of_bundle
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    [Fintype MolecularSrank.MDAction]
    (data : GuidedDockingBridgeData input pocket)
    (h : GuidedDockingBridgeAssumptions input pocket data) :
    @DecisionProblem.srank
      MolecularSrank.MDAction
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates h.prob)
      (MolecularSrank.mdCoordinateSpaceStruct h.prob)
      h.prob.toDecisionProblem ≤
      3 * MolecularSrank.numRelevantAtoms h.prob + 3 * data.ligand.numAtoms := by
  exact detected_pocket_md_srank_bound
    input pocket data.ligand data.cutoff data.utility h.prob h.prob_eq h.strictAll h.bounded

/-- Bundle-driven version of the small-pocket guided-docking srank bound. -/
theorem guided_docking_small_pocket_bound_of_bundle
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    [Fintype MolecularSrank.MDAction]
    (data : GuidedDockingBridgeData input pocket)
    (h : SmallPocketBridgeAssumptions input pocket data) :
    @DecisionProblem.srank
      MolecularSrank.MDAction
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates h.toGuidedDockingBridgeAssumptions.prob)
      (MolecularSrank.mdCoordinateSpaceStruct h.toGuidedDockingBridgeAssumptions.prob)
      h.toGuidedDockingBridgeAssumptions.prob.toDecisionProblem ≤ 3 * h.K + 3 * h.L := by
  exact detected_pocket_docking_small_pocket_bound
    input pocket data.ligand data.cutoff data.utility
    h.toGuidedDockingBridgeAssumptions.prob h.toGuidedDockingBridgeAssumptions.prob_eq
    h.K h.L h.toGuidedDockingBridgeAssumptions.strictAll
    h.toGuidedDockingBridgeAssumptions.bounded h.pocketBound h.ligandBound

/-- Bundle-driven version of the sampled small-pocket guided-docking srank bound. -/
theorem sampled_guided_docking_small_pocket_bound_of_bundle
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    [Fintype MolecularSrank.MDAction]
    (data : GuidedDockingBridgeData input pocket)
    (h : SampledSmallPocketBridgeAssumptions input pocket data) :
    @DecisionProblem.srank
      (SampledDocking.SupportedAction h.samples)
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates h.toSmallPocketBridgeAssumptions.toGuidedDockingBridgeAssumptions.prob)
      (MolecularSrank.mdCoordinateSpaceStruct h.toSmallPocketBridgeAssumptions.toGuidedDockingBridgeAssumptions.prob)
      (SampledDocking.restrictedDecisionProblem
        h.toSmallPocketBridgeAssumptions.toGuidedDockingBridgeAssumptions.prob.toDecisionProblem h.samples)
      ≤ 3 * h.toSmallPocketBridgeAssumptions.K + 3 * h.toSmallPocketBridgeAssumptions.L := by
  exact Tractability.SampledDockingCutoff.sampled_small_pocket_low_srank
    h.toSmallPocketBridgeAssumptions.toGuidedDockingBridgeAssumptions.prob
    h.samples h.toSmallPocketBridgeAssumptions.K h.toSmallPocketBridgeAssumptions.L h.aStar
    h.strictAllSampled h.toSmallPocketBridgeAssumptions.toGuidedDockingBridgeAssumptions.bounded
    h.capture h.toSmallPocketBridgeAssumptions.pocketBound h.toSmallPocketBridgeAssumptions.ligandBound

end PocketDockingBridge
end Tractability
end DecisionQuotient
