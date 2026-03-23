/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CrossDockingCertificates.lean

  Runtime-facing certificates for site-known, blind-conformer cross docking.

  The goal of this file is not to introduce new physics. Instead, it packages
  existing locality, pruning, strain, receptor-flexibility, and ambiguity-band
  theorems into statements that mirror the Python implementation decisions we
  want to justify later:

  1. lower-bound pruning from any incumbent is sound for all better actions;
  2. better incumbents monotonically increase the set of prunable cells;
  3. Lipschitz cell bounds justify branch-and-bound exclusion directly;
  4. exact strain penalties preserve certified top-1 pruning and optimizer
     witnesses;
  5. detected-pocket sampled docking inherits inside-cutoff sufficiency;
  6. the same detected-pocket bridge gives tractability + sufficient retained
     coordinates in one theorem;
  7. receptor-flexibility error and ligand strain compose into a certified
     rigid-reference survivor set.
-/

import DecisionQuotient.Sufficiency
import DecisionQuotient.Tractability.BlindDockingTractability
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.ConformerSearch
import DecisionQuotient.Tractability.LigandStrainApproximation
import DecisionQuotient.Tractability.PocketDockingBridge
import DecisionQuotient.Tractability.ReceptorFlexibility
import DecisionQuotient.Tractability.SampledDockingCutoff
import Mathlib.Data.Real.Basic
import Mathlib.Topology.MetricSpace.Lipschitz

namespace DecisionQuotient
namespace Tractability
namespace CrossDockingCertificates

open CoarseApproximation
open ConformerSearch
open LigandStrainApproximation
open ReceptorFlexibility
open PocketDockingBridge
open BlindDocking
open SampledDockingCutoff

universe u v w

-- ---------------------------------------------------------------------------
-- Section 1: Lower-bound pruning is seed-safe
-- ---------------------------------------------------------------------------

/-- Survivors under lower-bound pruning with incumbent energy `incumbentEnergy`.

    An action survives precisely when its certified lower bound does not already
    exceed the current incumbent. This is the abstract object behind the Python
    branch-and-bound queues. -/
def lowerBoundSurvivors {A : Type u}
    (lowerBound : A → ℝ) (incumbentEnergy : ℝ) : Set A :=
  { a | ¬ incumbentEnergy < lowerBound a }

/-- Any action whose exact energy is already no worse than the incumbent cannot
    be pruned by a sound lower bound.

    This is the core seed-independence certificate: whether the incumbent came
    from the crystal conformer, a random seed, or a previously discovered blind
    conformer, lower-bound pruning never discards an action that is actually at
    least as good as that incumbent. -/
theorem better_than_incumbent_survives_lowerBound_pruning
    {A : Type u}
    (energy lowerBound : A → ℝ)
    (incumbentEnergy : ℝ)
    (a : A)
    (hLower : ∀ x, lowerBound x ≤ energy x)
    (hBetter : energy a ≤ incumbentEnergy) :
    a ∈ lowerBoundSurvivors lowerBound incumbentEnergy := by
  show ¬ incumbentEnergy < lowerBound a
  intro hPruned
  have hLb := hLower a
  linarith

/-- Better incumbents can only increase pruning power.

    If `inc1 ≤ inc2`, then every action surviving the stricter incumbent `inc1`
    also survives the weaker incumbent `inc2`. Equivalently, lowering the
    incumbent can only shrink the survivor set, never enlarge it. -/
theorem lowerBoundSurvivors_monotone
    {A : Type u}
    (lowerBound : A → ℝ)
    {inc1 inc2 : ℝ}
    (hInc : inc1 ≤ inc2) :
    lowerBoundSurvivors lowerBound inc1 ⊆ lowerBoundSurvivors lowerBound inc2 := by
  intro a ha
  show ¬ inc2 < lowerBound a
  intro hPruned
  exact ha (lt_of_le_of_lt hInc hPruned)

-- ---------------------------------------------------------------------------
-- Section 2: Direct branch-and-bound cell exclusion
-- ---------------------------------------------------------------------------

/-- A current incumbent beats every point in a Lipschitz cell whenever it beats
    the cell's center lower bound.

    This packages `lipschitz_energy_lower_bound_on_ball` into the exact runtime
    form used by conformer branch-and-bound: score the center once, subtract the
    certified slack, and skip the whole cell if the incumbent is already below
    that lower bound. -/
theorem incumbent_beats_all_points_in_cell
    {P : Type*} [PseudoMetricSpace P]
    (energy : P → ℝ)
    (L : NNReal)
    (hLip : LipschitzWith L energy)
    (incumbent center point : P)
    (r : ℝ)
    (hBest : energy incumbent < energy center - L * r)
    (hBall : dist point center ≤ r) :
    energy incumbent < energy point := by
  have hLower :=
    lipschitz_energy_lower_bound_on_ball energy L hLip point center r hBall
  linarith

/-- Strain-aware version of the Lipschitz cell exclusion theorem.

    This is the direct certificate for blind conformer search with an exact
    torsion-strain penalty: if the incumbent beats the center lower bound of the
    combined `(score - strain)` objective, the entire cell is safely excluded. -/
theorem incumbent_beats_all_points_in_strain_cell
    {P : Type*} [PseudoMetricSpace P]
    (score strain : P → ℝ)
    (L : NNReal)
    (hLip : LipschitzWith L (fun p => score p - strain p))
    (incumbent center point : P)
    (r : ℝ)
    (hBest : score incumbent - strain incumbent <
      (score center - strain center) - L * r)
    (hBall : dist point center ≤ r) :
    score incumbent - strain incumbent < score point - strain point := by
  have hLower :=
    strain_energy_lower_bound_on_ball score strain L hLip point center r hBall
  linarith

-- ---------------------------------------------------------------------------
-- Section 3: Strain preserves certified pruning objects
-- ---------------------------------------------------------------------------

/-- Certified top-1 survivor set for an exact/coarse score pair after adding an
    exact strain penalty. -/
noncomputable def strain_augmented_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (exact coarse : DecisionProblem A S)
    (δ : ℝ)
    (strain : A → ℝ)
    (s : S)
    (hApprox : UniformUtilityApprox exact coarse δ)
    (hδ : 0 ≤ δ) :
    CertifiedPruning.CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => (strainAugmentedDecisionProblem exact strain).utility a s)
    (fun a => (strainAugmentedDecisionProblem coarse strain).utility a s)
    δ
    (fun a => strain_preserves_uniformApprox exact coarse δ strain hApprox a s)
    hδ

/-- Soundness of the strain-augmented certified top-1 survivor set. -/
theorem strain_augmented_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (exact coarse : DecisionProblem A S)
    (δ : ℝ)
    (strain : A → ℝ)
    (s : S)
    (hApprox : UniformUtilityApprox exact coarse δ)
    (hδ : 0 ≤ δ) :
    (CertifiedPruning.certificate_of_top1_coarse_ambiguityBand
      (fun a => (strainAugmentedDecisionProblem exact strain).utility a s)
      (fun a => (strainAugmentedDecisionProblem coarse strain).utility a s)
      δ
      (fun a => strain_preserves_uniformApprox exact coarse δ strain hApprox a s)
      hδ).exactTopK ⊆
      (strain_augmented_certified_top1 exact coarse δ strain s hApprox hδ).survivors := by
  simpa [strain_augmented_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => (strainAugmentedDecisionProblem exact strain).utility a s)
      (fun a => (strainAugmentedDecisionProblem coarse strain).utility a s)
      δ
      (fun a => strain_preserves_uniformApprox exact coarse δ strain hApprox a s)
      hδ

/-- Coherent optimizer witness for strain-augmented exact/coarse top-1 search. -/
noncomputable def strain_augmented_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (exact coarse : DecisionProblem A S)
    (δ : ℝ)
    (strain : A → ℝ)
    (s : S)
    (hApprox : UniformUtilityApprox exact coarse δ)
    (hδ : 0 ≤ δ) :
    FormalLocalOptimizer.CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => (strainAugmentedDecisionProblem exact strain).utility a s)
    (fun a => (strainAugmentedDecisionProblem coarse strain).utility a s)
    δ
    (fun a => strain_preserves_uniformApprox exact coarse δ strain hApprox a s)
    hδ

/-- In the strain-augmented coherent witness, every exact top-1 action remains
    inside the runtime support. -/
theorem strain_augmented_exactTop1_subset_support
    {A : Type u} {S : Type v}
    [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (exact coarse : DecisionProblem A S)
    (δ : ℝ)
    (strain : A → ℝ)
    (s : S)
    (hApprox : UniformUtilityApprox exact coarse δ)
    (hδ : 0 ≤ δ) :
    (strain_augmented_coherent_optimizer_witness exact coarse δ strain s hApprox hδ).survivorSet.certificate.exactTopK
      ⊆ (strain_augmented_coherent_optimizer_witness exact coarse δ strain s hApprox hδ).belief.selection.support :=
  (strain_augmented_coherent_optimizer_witness exact coarse δ strain s hApprox hδ).exactTopK_subset_support

-- ---------------------------------------------------------------------------
-- Section 4: Detected-pocket coordinate sufficiency for sampled docking
-- ---------------------------------------------------------------------------

/-- A detected pocket inherits the inside-cutoff sufficiency theorem for sampled
    docking once it is bridged into the `MDBindingProblem` model.

    This is the theorem the Python implementation needs for receptor-atom
    subsetting after pocket detection: the retained inside-cutoff coordinates are
    sufficient for the sampled restricted docking problem. -/
theorem detected_pocket_sampled_insideCutoff_sufficient
    (input : Computation.PocketDetection.PocketDetectionInput)
    (pocket : Computation.PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (samples : SampledDocking.SampledActionFamily MolecularSrank.MDAction)
    [Fintype MolecularSrank.MDAction]
    [ProductSpace MolecularSrank.MDState (MolecularSrank.numMDCoordinates prob)]
    (aStar : MolecularSrank.MDAction)
    (hStrictAll : ∀ s,
      StrictOpt prob.toDecisionProblem aStar s ∨
      ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s,
      ∃ a : SampledDocking.SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hCompat : ∀ s s' i,
      CoordinateSpace.proj s i = CoordinateSpace.proj s' i ↔
        MolecularSrank.mdProj prob s i = MolecularSrank.mdProj prob s' i)
    (hinj : ∀ s s' : MolecularSrank.MDState,
      (∀ i : Fin (MolecularSrank.numMDCoordinates prob),
        CoordinateSpace.proj s i = CoordinateSpace.proj s' i) → s = s') :
    (SampledDocking.restrictedDecisionProblem prob.toDecisionProblem samples).isSufficient
      (MolecularSrank.potentialRelevantCoords prob) := by
  subst hProb
  exact sampled_insideCutoff_sufficient
    (detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    samples aStar hStrictAll hBounded hCapture hCompat hinj

/-- Detected-pocket tractability plus sampled inside-cutoff sufficiency in one
    packaged theorem.

    This is the end-to-end statement behind site-known blind conformer docking:
    detected pockets keep the docking instance tractable, and sampled retained
    coordinates remain sufficient for the restricted problem. -/
theorem detected_pocket_restricted_problem_tractable_and_sufficient
    {pocketDetectionTractable : Prop}
    (hDetect : pocketDetectionTractable)
    (input : Computation.PocketDetection.PocketDetectionInput)
    (pocket : Computation.PocketDetection.DetectedPocket input)
    (ligand : MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    (prob : MolecularSrank.MDBindingProblem)
    (hProb : prob = detected_pocket_to_md_binding_problem input pocket ligand cutoff utility)
    (samples : SampledDocking.SampledActionFamily MolecularSrank.MDAction)
    [Fintype MolecularSrank.MDAction]
    [ProductSpace MolecularSrank.MDState (MolecularSrank.numMDCoordinates prob)]
    (aStar : MolecularSrank.MDAction)
    (hStrictAll : ∀ s,
      StrictOpt prob.toDecisionProblem aStar s ∨
      ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : MolecularSrank.OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s,
      ∃ a : SampledDocking.SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hCompat : ∀ s s' i,
      CoordinateSpace.proj s i = CoordinateSpace.proj s' i ↔
        MolecularSrank.mdProj prob s i = MolecularSrank.mdProj prob s' i)
    (hinj : ∀ s s' : MolecularSrank.MDState,
      (∀ i : Fin (MolecularSrank.numMDCoordinates prob),
        CoordinateSpace.proj s i = CoordinateSpace.proj s' i) → s = s') :
    BlindDockingTractable pocketDetectionTractable (GuidedDockingSrankBounded prob) ∧
    (SampledDocking.restrictedDecisionProblem prob.toDecisionProblem samples).isSufficient
      (MolecularSrank.potentialRelevantCoords prob) := by
  constructor
  · exact blind_docking_tractable_of_detected_pocket
      hDetect input pocket ligand cutoff utility prob hProb
      (fun s => by
        rcases hStrictAll s with h | h
        · exact ⟨aStar, h⟩
        · exact h)
      hBounded
  · exact detected_pocket_sampled_insideCutoff_sufficient
      input pocket ligand cutoff utility prob hProb samples aStar hStrictAll hBounded hCapture hCompat hinj

-- ---------------------------------------------------------------------------
-- Section 5: Receptor flexibility + ligand strain certificates
-- ---------------------------------------------------------------------------

/-- Certified top-1 survivor set when a rigid reference receptor is combined
    with an exact ligand strain penalty.

    This composes receptor-flexibility error control with exact strain addition,
    giving a single survivor object for cross-docking regimes that keep the
    pocket fixed but allow blind ligand torsions and bounded receptor mismatch. -/
noncomputable def ensemble_rigid_strain_certified_top1
    {A : Type u} {S : Type v} {R : Type w}
    [Fintype A] [Fintype S] [Fintype R]
    [DecidableEq A] [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ)
    (strain : A → ℝ)
    (r₀ r : R)
    (s : S) :
    CertifiedPruning.CertifiedSurvivorSet A :=
  strain_augmented_certified_top1
    (flexibleDecisionProblem score r)
    (rigidDecisionProblem score r₀)
    (conformationalErrorRadius score r₀)
    strain
    s
    (rigid_approximates_conformation score r₀ r)
    (conformationalErrorRadius_nonneg score r₀)

/-- Soundness of the receptor-flexibility + ligand-strain certified top-1 set. -/
theorem ensemble_rigid_strain_certified_top1_sound
    {A : Type u} {S : Type v} {R : Type w}
    [Fintype A] [Fintype S] [Fintype R]
    [DecidableEq A] [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ)
    (strain : A → ℝ)
    (r₀ r : R)
    (s : S) :
    (CertifiedPruning.certificate_of_top1_coarse_ambiguityBand
      (fun a =>
        (strainAugmentedDecisionProblem (flexibleDecisionProblem score r) strain).utility a s)
      (fun a =>
        (strainAugmentedDecisionProblem (rigidDecisionProblem score r₀) strain).utility a s)
      (conformationalErrorRadius score r₀)
      (fun a =>
        strain_preserves_uniformApprox
          (flexibleDecisionProblem score r)
          (rigidDecisionProblem score r₀)
          (conformationalErrorRadius score r₀)
          strain
          (rigid_approximates_conformation score r₀ r)
          a s)
      (conformationalErrorRadius_nonneg score r₀)).exactTopK ⊆
      (ensemble_rigid_strain_certified_top1 score strain r₀ r s).survivors := by
  simpa [ensemble_rigid_strain_certified_top1]
    using strain_augmented_certified_top1_sound
      (flexibleDecisionProblem score r)
      (rigidDecisionProblem score r₀)
      (conformationalErrorRadius score r₀)
      strain
      s
      (rigid_approximates_conformation score r₀ r)
      (conformationalErrorRadius_nonneg score r₀)

end CrossDockingCertificates
end Tractability
end DecisionQuotient
