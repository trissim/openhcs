/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/BlindDockingTractability.lean - Complexity of Blind vs Guided Docking

  KEY INSIGHT: Blind docking = Pocket Detection + Guided Docking
-/

import DecisionQuotient.Basic
import DecisionQuotient.Computation.PocketDetection
import DecisionQuotient.Tractability.PocketDockingBridge
import DecisionQuotient.Tractability.SampledDockingCutoff
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.FormalLocalOptimizer
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
    (hProb : prob = PocketDockingBridge.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (hStrictAll : ∀ s, ∃ a, Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob) :
    GuidedDockingSrankBounded prob := by
  refine ⟨3 * MolecularSrank.numRelevantAtoms prob + 3 * ligand.numAtoms, ?_⟩
  exact PocketDockingBridge.detected_pocket_md_srank_bound
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
    (hProb : prob = PocketDockingBridge.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
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
    (hProb : prob = PocketDockingBridge.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
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
  A detected pocket also yields the sampled-docking small-pocket srank bound
  used by the existing accelerated docking infrastructure.
-/
theorem sampled_guided_docking_srank_bound_of_detected_pocket
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    [Fintype MolecularSrank.MDAction]
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = PocketDockingBridge.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (samples : SampledDocking.SampledActionFamily MolecularSrank.MDAction)
    (aStar : MolecularSrank.MDAction)
    (K L : Nat)
    (hStrictAll : ∀ s,
      Tractability.StrictOpt prob.toDecisionProblem aStar s ∨
      ∃ a, Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s,
      ∃ a : SampledDocking.SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hPocket : MolecularSrank.numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank
      (SampledDocking.SupportedAction samples)
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates prob)
      (MolecularSrank.mdCoordinateSpaceStruct prob)
      (SampledDocking.restrictedDecisionProblem prob.toDecisionProblem samples)
      ≤ 3 * K + 3 * L := by
  subst prob
  simpa using Tractability.SampledDockingCutoff.sampled_small_pocket_low_srank
    (PocketDockingBridge.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    samples K L aStar hStrictAll hBounded hCapture hPocket hLigand

/--
  Non-sampled guided docking also inherits the explicit small-pocket srank bound
  through the detected-pocket bridge.
-/
theorem guided_docking_small_pocket_bound_of_detected_pocket
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    [Fintype MolecularSrank.MDAction]
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = PocketDockingBridge.detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (K L : Nat)
    (hStrictAll : ∀ s, ∃ a, Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob)
    (hPocket : MolecularSrank.numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank
      MolecularSrank.MDAction
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates prob)
      (MolecularSrank.mdCoordinateSpaceStruct prob)
      prob.toDecisionProblem ≤ 3 * K + 3 * L := by
  exact PocketDockingBridge.detected_pocket_docking_small_pocket_bound
    input pocket ligand cutoff utility prob hProb K L hStrictAll hBounded hPocket hLigand

/-- Bundle-driven convenience form of the guided-docking srank boundedness theorem. -/
theorem guided_docking_srank_bounded_of_bundle
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    [Fintype MolecularSrank.MDAction]
    (data : PocketDockingBridge.GuidedDockingBridgeData input pocket)
    (h : PocketDockingBridge.GuidedDockingBridgeAssumptions input pocket data) :
    GuidedDockingSrankBounded h.prob := by
  refine ⟨3 * MolecularSrank.numRelevantAtoms h.prob + 3 * data.ligand.numAtoms, ?_⟩
  exact PocketDockingBridge.guided_docking_srank_bound_of_bundle input pocket data h

/-- Bundle-driven convenience form of the guided-docking small-pocket bound. -/
theorem guided_docking_small_pocket_bound_of_bundle
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    [Fintype MolecularSrank.MDAction]
    (data : PocketDockingBridge.GuidedDockingBridgeData input pocket)
    (h : PocketDockingBridge.SmallPocketBridgeAssumptions input pocket data) :
    @DecisionProblem.srank
      MolecularSrank.MDAction
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates h.toGuidedDockingBridgeAssumptions.prob)
      (MolecularSrank.mdCoordinateSpaceStruct h.toGuidedDockingBridgeAssumptions.prob)
      h.toGuidedDockingBridgeAssumptions.prob.toDecisionProblem ≤ 3 * h.K + 3 * h.L := by
  exact PocketDockingBridge.guided_docking_small_pocket_bound_of_bundle input pocket data h

/-- Bundle-driven convenience form of the sampled small-pocket bound. -/
theorem sampled_guided_docking_small_pocket_bound_of_bundle
    (input : PocketDetection.PocketDetectionInput)
    (pocket : PocketDetection.DetectedPocket input)
    [Fintype MolecularSrank.MDAction]
    (data : PocketDockingBridge.GuidedDockingBridgeData input pocket)
    (h : PocketDockingBridge.SampledSmallPocketBridgeAssumptions input pocket data) :
    @DecisionProblem.srank
      (SampledDocking.SupportedAction h.samples)
      MolecularSrank.MDState
      (MolecularSrank.numMDCoordinates h.toSmallPocketBridgeAssumptions.toGuidedDockingBridgeAssumptions.prob)
      (MolecularSrank.mdCoordinateSpaceStruct h.toSmallPocketBridgeAssumptions.toGuidedDockingBridgeAssumptions.prob)
      (SampledDocking.restrictedDecisionProblem
        h.toSmallPocketBridgeAssumptions.toGuidedDockingBridgeAssumptions.prob.toDecisionProblem h.samples)
      ≤ 3 * h.toSmallPocketBridgeAssumptions.K + 3 * h.toSmallPocketBridgeAssumptions.L := by
  exact PocketDockingBridge.sampled_guided_docking_small_pocket_bound_of_bundle input pocket data h

/--
  Detected pockets can feed the certified-pruning layer directly: if a coarse
  score uniformly approximates the exact score on the finite action family, then
  the top-1 survivor set comes with a theorem-backed pruning certificate.
-/
noncomputable def certified_top1_survivor_set_of_detected_pocket
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    CertifiedPruning.CertifiedSurvivorSet A :=
  CertifiedPruning.certifiedSurvivorSet_of_top1_coarse_ambiguityBand
    uExact uCoarse delta hApprox hDelta

/--
  Soundness wrapper for the detected-pocket certified top-1 survivor set.
-/
theorem certified_top1_survivor_set_of_detected_pocket_sound
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    (CertifiedPruning.certificate_of_top1_coarse_ambiguityBand uExact uCoarse delta hApprox hDelta).exactTopK ⊆
      (certified_top1_survivor_set_of_detected_pocket uExact uCoarse delta hApprox hDelta).survivors := by
  simpa [certified_top1_survivor_set_of_detected_pocket]
    using CertifiedPruning.certificate_top1_coarse_ambiguityBand_sound
      uExact uCoarse delta hApprox hDelta

/--
  The canonical finite uniform-error radius for a sampled docking problem is
  always nonnegative.
-/
theorem sampled_docking_finite_uniform_error_nonneg
    {NP NL N : Nat}
    (prob : SampledDocking.SampledDockingProblem NP NL N) :
    0 ≤ CoarseApproximation.finiteUniformErrorRadius
      prob.exactDecisionProblem prob.coarseDecisionProblem := by
  rcases (inferInstance : Nonempty (SampledDocking.SupportedAction prob.samples)) with ⟨a⟩
  rcases (inferInstance : Nonempty (DiscretizedState.GridMDState NP NL N)) with ⟨s⟩
  have hApprox := CoarseApproximation.SampledDocking.SampledDockingProblem.finiteUniformErrorRadius_witnesses prob
  have hAbs : 0 ≤ |prob.exactDecisionProblem.utility a s - prob.coarseDecisionProblem.utility a s| :=
    abs_nonneg _
  unfold CoarseApproximation.UniformUtilityApprox at hApprox
  have hApprox_as : |prob.exactDecisionProblem.utility a s - prob.coarseDecisionProblem.utility a s| ≤
      CoarseApproximation.finiteUniformErrorRadius prob.exactDecisionProblem prob.coarseDecisionProblem := hApprox a s
  linarith

/--
  Every sampled docking problem yields a theorem-backed certified top-1 survivor
  set using its canonical finite uniform-error approximation radius.
-/
noncomputable def sampled_docking_certified_top1_of_finite_uniform_error
    {NP NL N : Nat}
    (prob : SampledDocking.SampledDockingProblem NP NL N)
    (s : DiscretizedState.GridMDState NP NL N) :
    CertifiedPruning.CertifiedSurvivorSet (SampledDocking.SupportedAction prob.samples) :=
  let delta := CoarseApproximation.finiteUniformErrorRadius
    prob.exactDecisionProblem prob.coarseDecisionProblem
  CertifiedPruning.certifiedSurvivorSet_of_top1_coarse_ambiguityBand
    (fun a => prob.exactDecisionProblem.utility a s)
    (fun a => prob.coarseDecisionProblem.utility a s)
    delta
    (fun a => (CoarseApproximation.SampledDocking.SampledDockingProblem.finiteUniformErrorRadius_witnesses prob) a s)
    (sampled_docking_finite_uniform_error_nonneg prob)

/-- Soundness of the canonical sampled-docking certified top-1 survivor set. -/
theorem sampled_docking_certified_top1_of_finite_uniform_error_sound
    {NP NL N : Nat}
    (prob : SampledDocking.SampledDockingProblem NP NL N)
    (s : DiscretizedState.GridMDState NP NL N) :
    (CertifiedPruning.certificate_of_top1_coarse_ambiguityBand
      (fun a => prob.exactDecisionProblem.utility a s)
      (fun a => prob.coarseDecisionProblem.utility a s)
      (CoarseApproximation.finiteUniformErrorRadius prob.exactDecisionProblem prob.coarseDecisionProblem)
      (fun a => (CoarseApproximation.SampledDocking.SampledDockingProblem.finiteUniformErrorRadius_witnesses prob) a s)
      (sampled_docking_finite_uniform_error_nonneg prob)).exactTopK
      ⊆ (sampled_docking_certified_top1_of_finite_uniform_error prob s).survivors := by
  simpa [sampled_docking_certified_top1_of_finite_uniform_error]
    using CertifiedPruning.certificate_top1_coarse_ambiguityBand_sound
      (fun a => prob.exactDecisionProblem.utility a s)
      (fun a => prob.coarseDecisionProblem.utility a s)
      (CoarseApproximation.finiteUniformErrorRadius prob.exactDecisionProblem prob.coarseDecisionProblem)
      (fun a => (CoarseApproximation.SampledDocking.SampledDockingProblem.finiteUniformErrorRadius_witnesses prob) a s)
      (sampled_docking_finite_uniform_error_nonneg prob)

/--
  The canonical finite uniform-error radius also packages directly into a
  runtime-facing optimizer witness for theorem-backed local optimization.
-/
noncomputable def sampled_docking_coherent_optimizer_witness_of_finite_uniform_error
    {NP NL N : Nat}
    (prob : SampledDocking.SampledDockingProblem NP NL N)
    (s : DiscretizedState.GridMDState NP NL N)
    [LinearOrder (SampledDocking.SupportedAction prob.samples)] :
    FormalLocalOptimizer.CoherentOptimizerWitness (SampledDocking.SupportedAction prob.samples) :=
  CoarseApproximation.coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => prob.exactDecisionProblem.utility a s)
    (fun a => prob.coarseDecisionProblem.utility a s)
    (CoarseApproximation.finiteUniformErrorRadius prob.exactDecisionProblem prob.coarseDecisionProblem)
    (fun a => (CoarseApproximation.SampledDocking.SampledDockingProblem.finiteUniformErrorRadius_witnesses prob) a s)
    (sampled_docking_finite_uniform_error_nonneg prob)

noncomputable def sampled_docking_optimizer_witness_of_finite_uniform_error
    {NP NL N : Nat}
    (prob : SampledDocking.SampledDockingProblem NP NL N)
    (s : DiscretizedState.GridMDState NP NL N)
    [LinearOrder (SampledDocking.SupportedAction prob.samples)] :
    FormalLocalOptimizer.OptimizerWitness (SampledDocking.SupportedAction prob.samples) :=
  (sampled_docking_coherent_optimizer_witness_of_finite_uniform_error prob s).toOptimizerWitness

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
