/-
  Paper 4: Decision-Relevant Uncertainty

  Summary.lean - mechanically backed summary aliases

  This module exposes compact theorem names that alias concrete results proved
  in the underlying modules. It does not introduce placeholder statements.
-/

import DecisionQuotient.Hardness
import DecisionQuotient.Tractability.BoundedActions
import DecisionQuotient.Tractability.BoundedStateSpace
import DecisionQuotient.Tractability.SingleAction
import DecisionQuotient.Tractability.SeparableUtility
import DecisionQuotient.Tractability.MultiplicativeSeparable
import DecisionQuotient.Tractability.Dominance
import DecisionQuotient.Tractability.TreeStructure
import DecisionQuotient.Tractability.Tightness
import DecisionQuotient.Tractability.FPT
import DecisionQuotient.Tractability.BlindDockingTractability
import DecisionQuotient.Tractability.PocketDockingBridge
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.LJApproximation
import DecisionQuotient.Tractability.SoftLJApproximation
import DecisionQuotient.Tractability.CoulombApproximation
import DecisionQuotient.Tractability.ContactApproximation
import DecisionQuotient.Tractability.ScreenedCoulombApproximation
import DecisionQuotient.Tractability.NonbondedApproximation
import DecisionQuotient.Tractability.DirectionalHBondApproximation
import DecisionQuotient.Tractability.PiStackingApproximation
import DecisionQuotient.Tractability.PiCationApproximation
import DecisionQuotient.Tractability.HalogenBondApproximation
import DecisionQuotient.Tractability.WaterMediatedHBondApproximation
import DecisionQuotient.Tractability.SignInvariance
import DecisionQuotient.Tractability.MetalCoordinationApproximation
import DecisionQuotient.Tractability.RichChemistryApproximation
import DecisionQuotient.Tractability.ExtendedRichChemistryApproximation
import DecisionQuotient.Tractability.TopKLoweringBridge
import DecisionQuotient.Tractability.SupportExpansion
import DecisionQuotient.Tractability.GridConvergence
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.LigandStrainApproximation
import DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation
import DecisionQuotient.Tractability.CooperativeHBondApproximation
import DecisionQuotient.Tractability.ExplicitWaterPlacement
import DecisionQuotient.Tractability.ReceptorFlexibility
import DecisionQuotient.Dichotomy
import DecisionQuotient.ComplexityMain

namespace DecisionQuotient.Summary

/-- coNP-hardness reduction core for SUFFICIENCY-CHECK. -/
theorem conp_completeness {n : ℕ} (φ : Formula n) :
    (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology :=
  (DecisionQuotient.tautology_iff_sufficient φ).symm

/-- Bounded-actions tractability alias. -/
theorem bounded_actions_tractable
    {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
    {n : ℕ} [CoordinateSpace S n]
    [∀ s s' : S, ∀ I : Finset (Fin n), Decidable (agreeOn s s' I)]
    (k : ℕ) (cdp : ComputableDecisionProblem A S)
    (hcard : Fintype.card A ≤ k) :
    ∃ (decide : Finset (Fin n) → Bool),
      ∀ I, decide I = true ↔ cdp.toAbstract.isSufficient I :=
  DecisionQuotient.sufficiency_poly_bounded_actions (k := k) (cdp := cdp) hcard

/-- Separable-utility tractability alias. -/
theorem separable_utility_tractable
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hsep : SeparableUtility (dp := dp)) :
    ∃ algo : Finset (Fin n) → Bool,
      ∀ I, algo I = true ↔ dp.isSufficient I :=
  DecisionQuotient.sufficiency_poly_separable (dp := dp) hsep

/-- Tree-structured tractability alias. -/
theorem tree_structure_tractable
    {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
    {n : ℕ} [CoordinateSpace S n]
    [∀ s s' : S, ∀ I : Finset (Fin n), Decidable (agreeOn s s' I)]
    (cdp : ComputableDecisionProblem A S)
    (deps : Fin n → Finset (Fin n)) (htree : TreeStructured deps) :
    ∃ algo : Finset (Fin n) → Bool,
      ∀ I, algo I = true ↔ cdp.toAbstract.isSufficient I :=
  DecisionQuotient.sufficiency_poly_tree_structured (cdp := cdp) deps htree

/-- Blind-docking tractability alias via detected-pocket structural rank. -/
theorem blind_docking_detected_pocket_tractable
    {pocketDetectionTractable : Prop}
    (hDetect : pocketDetectionTractable)
    (input : DecisionQuotient.Computation.PocketDetection.PocketDetectionInput)
    (pocket : DecisionQuotient.Computation.PocketDetection.DetectedPocket input)
    (ligand : DecisionQuotient.Tractability.MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : DecisionQuotient.Tractability.MolecularSrank.MDAction →
      DecisionQuotient.Tractability.MolecularSrank.MDState → ℝ)
    [Fintype DecisionQuotient.Tractability.MolecularSrank.MDAction]
    (prob : DecisionQuotient.Tractability.MolecularSrank.MDBindingProblem)
    (hProb : prob =
      DecisionQuotient.Tractability.PocketDockingBridge.detected_pocket_to_md_binding_problem
        input pocket ligand cutoff utility)
    (hStrictAll : ∀ s, ∃ a, DecisionQuotient.Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : DecisionQuotient.Tractability.MolecularSrank.OutsideCutoffApproximationBounded prob) :
    DecisionQuotient.Tractability.BlindDocking.BlindDockingTractable
      pocketDetectionTractable
      (DecisionQuotient.Tractability.BlindDocking.GuidedDockingSrankBounded prob) :=
  DecisionQuotient.Tractability.BlindDocking.blind_docking_tractable_of_detected_pocket
    hDetect input pocket ligand cutoff utility prob hProb hStrictAll hBounded

/-- Sampled-docking srank bound alias via detected pockets. -/
theorem blind_docking_detected_pocket_sampled_srank_bound
    (input : DecisionQuotient.Computation.PocketDetection.PocketDetectionInput)
    (pocket : DecisionQuotient.Computation.PocketDetection.DetectedPocket input)
    (ligand : DecisionQuotient.Tractability.MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : DecisionQuotient.Tractability.MolecularSrank.MDAction →
      DecisionQuotient.Tractability.MolecularSrank.MDState → ℝ)
    [Fintype DecisionQuotient.Tractability.MolecularSrank.MDAction]
    (prob : DecisionQuotient.Tractability.MolecularSrank.MDBindingProblem)
    (hProb : prob =
      DecisionQuotient.Tractability.PocketDockingBridge.detected_pocket_to_md_binding_problem
        input pocket ligand cutoff utility)
    (samples : DecisionQuotient.Tractability.SampledDocking.SampledActionFamily
      DecisionQuotient.Tractability.MolecularSrank.MDAction)
    (aStar : DecisionQuotient.Tractability.MolecularSrank.MDAction)
    (K L : Nat)
    (hStrictAll : ∀ s,
      DecisionQuotient.Tractability.StrictOpt prob.toDecisionProblem aStar s ∨
      ∃ a, DecisionQuotient.Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : DecisionQuotient.Tractability.MolecularSrank.OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s,
      ∃ a : DecisionQuotient.Tractability.SampledDocking.SupportedAction samples,
        a.1 ∈ prob.toDecisionProblem.Opt s)
    (hPocket : DecisionQuotient.Tractability.MolecularSrank.numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank
      (DecisionQuotient.Tractability.SampledDocking.SupportedAction samples)
      DecisionQuotient.Tractability.MolecularSrank.MDState
      (DecisionQuotient.Tractability.MolecularSrank.numMDCoordinates prob)
      (DecisionQuotient.Tractability.MolecularSrank.mdCoordinateSpaceStruct prob)
      (DecisionQuotient.Tractability.SampledDocking.restrictedDecisionProblem prob.toDecisionProblem samples)
      ≤ 3 * K + 3 * L :=
  DecisionQuotient.Tractability.BlindDocking.sampled_guided_docking_srank_bound_of_detected_pocket
    input pocket ligand cutoff utility prob hProb samples aStar K L hStrictAll hBounded hCapture hPocket hLigand

/-- Direct guided-docking small-pocket srank bound alias via detected pockets. -/
theorem blind_docking_detected_pocket_guided_srank_bound
    (input : DecisionQuotient.Computation.PocketDetection.PocketDetectionInput)
    (pocket : DecisionQuotient.Computation.PocketDetection.DetectedPocket input)
    (ligand : DecisionQuotient.Tractability.MolecularSrank.Molecule)
    (cutoff : ℝ)
    (utility : DecisionQuotient.Tractability.MolecularSrank.MDAction →
      DecisionQuotient.Tractability.MolecularSrank.MDState → ℝ)
    [Fintype DecisionQuotient.Tractability.MolecularSrank.MDAction]
    (prob : DecisionQuotient.Tractability.MolecularSrank.MDBindingProblem)
    (hProb : prob =
      DecisionQuotient.Tractability.PocketDockingBridge.detected_pocket_to_md_binding_problem
        input pocket ligand cutoff utility)
    (K L : Nat)
    (hStrictAll : ∀ s, ∃ a, DecisionQuotient.Tractability.StrictOpt prob.toDecisionProblem a s)
    (hBounded : DecisionQuotient.Tractability.MolecularSrank.OutsideCutoffApproximationBounded prob)
    (hPocket : DecisionQuotient.Tractability.MolecularSrank.numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank
      DecisionQuotient.Tractability.MolecularSrank.MDAction
      DecisionQuotient.Tractability.MolecularSrank.MDState
      (DecisionQuotient.Tractability.MolecularSrank.numMDCoordinates prob)
      (DecisionQuotient.Tractability.MolecularSrank.mdCoordinateSpaceStruct prob)
      prob.toDecisionProblem ≤ 3 * K + 3 * L :=
  DecisionQuotient.Tractability.BlindDocking.guided_docking_small_pocket_bound_of_detected_pocket
    input pocket ligand cutoff utility prob hProb K L hStrictAll hBounded hPocket hLigand

/-- Certified top-1 pruning alias for exact/coarse docking scores. -/
noncomputable def blind_docking_detected_pocket_certified_top1
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.BlindDocking.certified_top1_survivor_set_of_detected_pocket
    uExact uCoarse delta hApprox hDelta

/-- Sampled docking yields a canonical certified top-1 survivor set via finite uniform error. -/
noncomputable def sampled_docking_certified_top1
    {NP NL N : Nat}
    (prob : DecisionQuotient.Tractability.SampledDocking.SampledDockingProblem NP NL N)
    (s : DecisionQuotient.Tractability.DiscretizedState.GridMDState NP NL N) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet
      (DecisionQuotient.Tractability.SampledDocking.SupportedAction prob.samples) :=
  DecisionQuotient.Tractability.BlindDocking.sampled_docking_certified_top1_of_finite_uniform_error prob s

/-- Sampled docking coherent optimizer witness alias via finite uniform error. -/
noncomputable def sampled_docking_coherent_optimizer_witness
    {NP NL N : Nat}
    (prob : DecisionQuotient.Tractability.SampledDocking.SampledDockingProblem NP NL N)
    (s : DecisionQuotient.Tractability.DiscretizedState.GridMDState NP NL N)
    [LinearOrder (DecisionQuotient.Tractability.SampledDocking.SupportedAction prob.samples)] :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness
      (DecisionQuotient.Tractability.SampledDocking.SupportedAction prob.samples) :=
  DecisionQuotient.Tractability.BlindDocking.sampled_docking_coherent_optimizer_witness_of_finite_uniform_error prob s

/-- Exact-vs-cutoff Lennard-Jones coherent optimizer witness alias. -/
noncomputable def lj_cutoff_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rc : ℝ) (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.LJApproximation.exact_vs_cutoff_lj_coherent_optimizer_witness distance ε σ rc s

/-- Exact-vs-softened Lennard-Jones coherent optimizer witness alias. -/
noncomputable def lj_softened_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.SoftLJApproximation.exact_vs_softened_lj_coherent_optimizer_witness distance ε σ rSoft s

/-- Exact-vs-cutoff Coulomb coherent optimizer witness alias. -/
noncomputable def coulomb_cutoff_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.CoulombApproximation.exact_vs_cutoff_coulomb_coherent_optimizer_witness q_i q_j rc distance s

/-- Exact-vs-cutoff screened Coulomb coherent optimizer witness alias. -/
noncomputable def screened_coulomb_cutoff_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.ScreenedCoulombApproximation.exact_vs_cutoff_screened_coulomb_coherent_optimizer_witness
    q_i q_j κ rc distance s

/-- Exact-vs-cutoff bounded contact/desolvation coherent optimizer witness alias. -/
noncomputable def contact_cutoff_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (w β rc : ℝ) (distance : A → S → ℝ) (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.ContactApproximation.exact_vs_cutoff_contact_coherent_optimizer_witness
    w β rc distance s

/-- Exact-vs-cutoff additive LJ+Coulomb coherent optimizer witness alias. -/
noncomputable def lj_coulomb_cutoff_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.NonbondedApproximation.exact_vs_cutoff_lj_coulomb_coherent_optimizer_witness
    distance ε σ rcLJ q_i q_j rcC s

/-- Exact-vs-cutoff additive LJ+screened-Coulomb coherent optimizer witness alias. -/
noncomputable def lj_screened_coulomb_cutoff_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.NonbondedApproximation.exact_vs_cutoff_lj_screened_coulomb_coherent_optimizer_witness
    distance ε σ rcLJ q_i q_j κ rcSC s

/-- Generic finite exact-vs-coarse directional H-bond coherent optimizer witness alias. -/
noncomputable def directional_hbond_finite_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.DirectionalHBondApproximation.exact_vs_coarse_directionalHBond_coherent_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

/-- Exact-vs-coarse rich chemistry coherent optimizer witness alias. -/
noncomputable def rich_chemistry_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.RichChemistryApproximation.exact_vs_coarse_richChemistry_coherent_optimizer_witness
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s

/-- Generic negation-preserving coherent optimizer witness alias. -/
noncomputable def negated_uniform_coherent_optimizer_witness
    {A : Type*}
    [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.SignInvariance.coherent_optimizer_witness_of_negated_uniformApprox_top1
    uExact uCoarse delta hApprox hDelta

/-- Exact-vs-coarse attractive polar surrogate coherent optimizer witness alias. -/
noncomputable def attractive_polar_surrogate_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.RichChemistryApproximation.exact_vs_coarse_attractivePolarSurrogate_coherent_optimizer_witness
    distance w β rc radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s

/-- Exact-vs-coarse attractive rich chemistry coherent optimizer witness alias. -/
noncomputable def attractive_rich_chemistry_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.RichChemistryApproximation.exact_vs_coarse_attractiveRichChemistry_coherent_optimizer_witness
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s

/-- Exact-vs-coarse attractive directional H-bond coherent optimizer witness alias. -/
noncomputable def attractive_directional_hbond_finite_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.DirectionalHBondApproximation.exact_vs_coarse_attractiveDirectionalHBond_coherent_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

/-- Far-field Ewald correction coherent optimizer witness alias over a base scorer. -/
noncomputable def ewald_far_field_coherent_optimizer_witness
    {A S : Type*}
    [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uBase : A → S → ℝ)
    (distance : A → S → ℝ)
    (q_i q_j alpha R : ℝ)
    (s : S)
    (ha : 0 < alpha)
    (hR : 1 ≤ R)
    (hFar : ∀ a, R ≤ distance a s) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.CoulombApproximation.additive_exactRealEwald_coherent_optimizer_witness
    uBase distance q_i q_j alpha R s ha hR hFar

/-- Grid-convergence coherent optimizer witness alias. -/
noncomputable def grid_convergence_coherent_optimizer_witness
    {A Scont Sgrid : Type*}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ∀ a sGrid', |uCont a (lift sGrid') - uGrid a sGrid'| ≤ eps res)
    (hEps : 0 ≤ eps res) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.CoarseApproximation.coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => uCont a (lift sGrid))
    (fun a => uGrid a sGrid)
    (eps res)
    (fun a => hApprox a sGrid)
    hEps

/-- Exact-vs-cutoff Lennard-Jones certified top-1 survivor set alias. -/
noncomputable def lj_cutoff_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.LJApproximation.exact_vs_cutoff_lj_certified_top1 distance ε σ rc s

/-- Exact-vs-softened Lennard-Jones certified top-1 survivor set alias. -/
noncomputable def lj_softened_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rSoft : ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.SoftLJApproximation.exact_vs_softened_lj_certified_top1 distance ε σ rSoft s

/-- Exact-vs-cutoff Coulomb certified top-1 survivor set alias. -/
noncomputable def coulomb_cutoff_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.CoulombApproximation.exact_vs_cutoff_coulomb_certified_top1 q_i q_j rc distance s

/-- Exact-vs-cutoff screened Coulomb certified top-1 survivor set alias. -/
noncomputable def screened_coulomb_cutoff_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (q_i q_j κ rc : ℝ) (distance : A → S → ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.ScreenedCoulombApproximation.exact_vs_cutoff_screened_coulomb_certified_top1
    q_i q_j κ rc distance s

/-- Exact-vs-cutoff bounded contact/desolvation certified top-1 survivor set alias. -/
noncomputable def contact_cutoff_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w β rc : ℝ) (distance : A → S → ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.ContactApproximation.exact_vs_cutoff_contact_certified_top1
    w β rc distance s

/-- Exact-vs-cutoff additive LJ+Coulomb certified top-1 survivor set alias. -/
noncomputable def lj_coulomb_cutoff_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j rcC : ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.NonbondedApproximation.exact_vs_cutoff_lj_coulomb_certified_top1
    distance ε σ rcLJ q_i q_j rcC s

/-- Exact-vs-cutoff additive LJ+screened-Coulomb certified top-1 survivor set alias. -/
noncomputable def lj_screened_coulomb_cutoff_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rcLJ q_i q_j κ rcSC : ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.NonbondedApproximation.exact_vs_cutoff_lj_screened_coulomb_certified_top1
    distance ε σ rcLJ q_i q_j κ rcSC s

/-- Generic finite exact-vs-coarse directional H-bond certified top-1 survivor set alias. -/
noncomputable def directional_hbond_finite_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.DirectionalHBondApproximation.exact_vs_coarse_directionalHBond_certified_top1
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

/-- Exact-vs-coarse rich chemistry certified top-1 survivor set alias. -/
noncomputable def rich_chemistry_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.RichChemistryApproximation.exact_vs_coarse_richChemistry_certified_top1
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s

/-- Exact-vs-cutoff bounded metal coordination certified top-1 survivor set alias. -/
noncomputable def metal_coordination_cutoff_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.MetalCoordinationApproximation.exact_vs_cutoff_metalCoordination_certified_top1
    w ideal width rc distance s

/-- Generic negation-preserving certified top-1 survivor set alias. -/
noncomputable def negated_uniform_certified_top1
    {A : Type*}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.SignInvariance.certified_top1_survivor_set_of_negated_uniformApprox
    uExact uCoarse delta hApprox hDelta

/-- Exact-vs-coarse attractive polar surrogate certified top-1 survivor set alias. -/
noncomputable def attractive_polar_surrogate_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rc : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.RichChemistryApproximation.exact_vs_coarse_attractivePolarSurrogate_certified_top1
    distance w β rc radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s

/-- Exact-vs-coarse attractive rich chemistry certified top-1 survivor set alias. -/
noncomputable def attractive_rich_chemistry_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC w β rcCT : ℝ)
    (radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor : A → S → ℝ)
    (radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.RichChemistryApproximation.exact_vs_coarse_attractiveRichChemistry_certified_top1
    distance ε σ rcLJ q_i q_j κ rcSC w β rcCT
    radialExactRecDonor donorExactRecDonor acceptorExactRecDonor radialExactLigDonor donorExactLigDonor acceptorExactLigDonor radialCoarseRecDonor donorCoarseRecDonor acceptorCoarseRecDonor radialCoarseLigDonor donorCoarseLigDonor acceptorCoarseLigDonor s

/-- Exact-vs-coarse attractive extended chemistry certified top-1 survivor set alias. -/
noncomputable def attractive_extended_chemistry_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.ExtendedRichChemistryApproximation.exact_vs_coarse_attractiveExtendedChemistry_certified_top1
    distance w β rcCT
    polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
    polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
    metalW metalIdeal metalWidth metalRc metalDistance
    ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
    pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
    xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
    wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse s

/-- Exact-vs-coarse extended rich chemistry certified top-1 survivor set alias. -/
noncomputable def extended_rich_chemistry_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ)
    (ε σ rcLJ q_i q_j κ rcSC : ℝ)
    (w β rcCT : ℝ)
    (polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor : A → S → ℝ)
    (polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor : A → S → ℝ)
    (metalW metalIdeal metalWidth metalRc : ℝ)
    (metalDistance : A → S → ℝ)
    (ppRadialExact ppFaceExact ppOffsetExact : A → S → ℝ)
    (ppRadialCoarse ppFaceCoarse ppOffsetCoarse : A → S → ℝ)
    (pcRadialExact pcPlaneExact pcCationExact : A → S → ℝ)
    (pcRadialCoarse pcPlaneCoarse pcCationCoarse : A → S → ℝ)
    (xbRadialExact xbDonorExact xbAcceptorExact : A → S → ℝ)
    (xbRadialCoarse xbDonorCoarse xbAcceptorCoarse : A → S → ℝ)
    (wbRadialExact wbWaterExact wbLigandExact : A → S → ℝ)
    (wbRadialCoarse wbWaterCoarse wbLigandCoarse : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.ExtendedRichChemistryApproximation.exact_vs_coarse_extendedRichChemistry_certified_top1
    distance ε σ rcLJ q_i q_j κ rcSC
    w β rcCT
    polarRadialExactRecDonor polarDonorExactRecDonor polarAcceptorExactRecDonor polarRadialExactLigDonor polarDonorExactLigDonor polarAcceptorExactLigDonor
    polarRadialCoarseRecDonor polarDonorCoarseRecDonor polarAcceptorCoarseRecDonor polarRadialCoarseLigDonor polarDonorCoarseLigDonor polarAcceptorCoarseLigDonor
    metalW metalIdeal metalWidth metalRc metalDistance
    ppRadialExact ppFaceExact ppOffsetExact ppRadialCoarse ppFaceCoarse ppOffsetCoarse
    pcRadialExact pcPlaneExact pcCationExact pcRadialCoarse pcPlaneCoarse pcCationCoarse
    xbRadialExact xbDonorExact xbAcceptorExact xbRadialCoarse xbDonorCoarse xbAcceptorCoarse
    wbRadialExact wbWaterExact wbLigandExact wbRadialCoarse wbWaterCoarse wbLigandCoarse s

/-- Exact-vs-cutoff attractive metal coordination certified top-1 survivor set alias. -/
noncomputable def attractive_metal_coordination_cutoff_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance : A → S → ℝ) (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.MetalCoordinationApproximation.exact_vs_cutoff_attractiveMetalCoordination_certified_top1
    w ideal width rc distance s

/-- Exact-vs-coarse attractive directional H-bond certified top-1 survivor set alias. -/
noncomputable def attractive_directional_hbond_finite_certified_top1
    {A S : Type*}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.DirectionalHBondApproximation.exact_vs_coarse_attractiveDirectionalHBond_certified_top1
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

/-- Far-field Ewald correction certified top-1 survivor set alias over a base scorer. -/
noncomputable def ewald_far_field_certified_top1
    {A S : Type*}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (uBase : A → S → ℝ)
    (distance : A → S → ℝ)
    (q_i q_j alpha R : ℝ)
    (s : S)
    (ha : 0 < alpha)
    (hR : 1 ≤ R)
    (hFar : ∀ a, R ≤ distance a s) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.CoulombApproximation.additive_exactRealEwald_certified_top1
    uBase distance q_i q_j alpha R s ha hR hFar

/-- Generic grid-convergence certified top-1 survivor set alias. -/
noncomputable def grid_convergence_certified_top1
    {A Scont Sgrid : Type*}
    [Fintype A] [Fintype Sgrid] [DecidableEq A] [Nonempty A]
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (sGrid : Sgrid)
    (hApprox : ∀ a sGrid', |uCont a (lift sGrid') - uGrid a sGrid'| ≤ eps res)
    (hEps : 0 ≤ eps res) :
    DecisionQuotient.Tractability.CertifiedPruning.CertifiedSurvivorSet A :=
  DecisionQuotient.Tractability.CertifiedPruning.certifiedSurvivorSet_of_top1_coarse_ambiguityBand
    (fun a => uCont a (lift sGrid))
    (fun a => uGrid a sGrid)
    (eps res)
    (fun a => hApprox a sGrid)
    hEps

/-- Coherent optimizer witness alias for generic uniform approximations. -/
noncomputable def uniformApprox_coherent_optimizer_witness
    {A : Type*}
    [Fintype A] [DecidableEq A] [Nonempty A] [LinearOrder A]
    (uExact uCoarse : A → ℝ)
    (delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hDelta : 0 ≤ delta) :
    DecisionQuotient.Tractability.FormalLocalOptimizer.CoherentOptimizerWitness A :=
  DecisionQuotient.Tractability.CoarseApproximation.coherent_optimizer_witness_of_uniformApprox_top1
    uExact uCoarse delta hApprox hDelta

-- ---------------------------------------------------------------------------
-- Ligand strain energy theorem family
-- ---------------------------------------------------------------------------

/-- Cosine torsion strain is bounded in [0, 2Vk]. -/
theorem cosine_torsion_strain_bounded (Vk n φ φ₀ : ℝ) (hVk : 0 ≤ Vk) :
    DecisionQuotient.Tractability.LigandStrainApproximation.BoundedStrain
      (fun φ' => DecisionQuotient.Tractability.LigandStrainApproximation.cosineTorsionStrain Vk n φ' φ₀) (2 * Vk) :=
  DecisionQuotient.Tractability.LigandStrainApproximation.cosineTorsionStrain_bounded Vk n φ₀ hVk

/-- Exact strain penalty preserves uniform approximation error bounds. -/
theorem strain_preserves_uniform_approximation {A : Type*} {S : Type*}
    (exact coarse : DecisionQuotient.DecisionProblem A S) (δ : ℝ)
    (strain : A → ℝ)
    (h : DecisionQuotient.Tractability.CoarseApproximation.UniformUtilityApprox exact coarse δ) :
    DecisionQuotient.Tractability.CoarseApproximation.UniformUtilityApprox
      (DecisionQuotient.Tractability.LigandStrainApproximation.strainAugmentedDecisionProblem exact strain)
      (DecisionQuotient.Tractability.LigandStrainApproximation.strainAugmentedDecisionProblem coarse strain)
      δ :=
  DecisionQuotient.Tractability.LigandStrainApproximation.strain_preserves_uniformApprox exact coarse δ strain h

-- ---------------------------------------------------------------------------
-- Directional metal coordination theorem family
-- ---------------------------------------------------------------------------

/-- Two-factor Lipschitz bound for radial × angular metal coordination. -/
theorem directional_metal_two_factor_lipschitz
    {f1 f2 g1 g2 Lf Lg err : ℝ}
    (hf1 : 0 ≤ f1) (hf1b : f1 ≤ 1)
    (hf2 : 0 ≤ f2) (hf2b : f2 ≤ 1)
    (hg1 : 0 ≤ g1) (hg1b : g1 ≤ 1)
    (hg2 : 0 ≤ g2) (hg2b : g2 ≤ 1)
    (hLf : 0 ≤ Lf) (hLg : 0 ≤ Lg)
    (herr : 0 ≤ err)
    (hRadial : |f1 - f2| ≤ Lf * err)
    (hGeometry : |g1 - g2| ≤ Lg * err) :
    |DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation.directionalMetalScore f1 g1 -
     DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation.directionalMetalScore f2 g2| ≤
      (Lf + Lg) * err :=
  DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation.directionalMetalScore_sub_le_component_sum
    hf1 hf1b hf2 hf2b hg1 hg1b hg2 hg2b hLf hLg herr hRadial hGeometry

/-- Angular factor ∈ [0,1] does not worsen radial cutoff error. -/
theorem angular_factor_tightens_tail
    (w radial angular B : ℝ)
    (h_nonneg : 0 ≤ angular) (h_le_one : angular ≤ 1)
    (h_bound : |w * radial| ≤ B) :
    |w * radial * angular| ≤ B :=
  DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation.angular_factor_tightens_tail
    w radial angular h_nonneg h_le_one h_bound

/-- Directional metal coordination cutoff uniform approximation. -/
theorem directional_metal_cutoff_uniformApprox {A : Type*} {S : Type*}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (w ideal width rc : ℝ) (distance geometry : A → S → ℝ) :
    DecisionQuotient.Tractability.CoarseApproximation.UniformUtilityApprox
      (DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation.exactDirectionalMetalDecisionProblem w ideal width distance geometry)
      (DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation.cutoffDirectionalMetalDecisionProblem w ideal width rc distance geometry)
      (DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation.directionalMetalCutoffErrorRadius w ideal width rc distance geometry) :=
  DecisionQuotient.Tractability.DirectionalMetalCoordinationApproximation.directional_metal_cutoff_uniformApprox w ideal width rc distance geometry

-- ---------------------------------------------------------------------------
-- Cooperative H-bond network theorem family
-- ---------------------------------------------------------------------------

/-- Independent pairwise model is a uniform approximation of the cooperative
    model with error ≤ |α| · N². -/
theorem independent_approximates_cooperative {A : Type*} {S : Type*}
    (N : ℕ) (scores : Fin N → A → S → ℝ) (α : ℝ)
    (h_unit : ∀ a s, DecisionQuotient.Tractability.CooperativeHBondApproximation.AllUnitInterval (fun i => scores i a s)) :
    DecisionQuotient.Tractability.CoarseApproximation.UniformUtilityApprox
      (DecisionQuotient.Tractability.CooperativeHBondApproximation.cooperativeHBondDecisionProblem N scores α)
      (DecisionQuotient.Tractability.CooperativeHBondApproximation.independentHBondDecisionProblem N scores)
      (|α| * (N : ℝ) ^ 2) :=
  DecisionQuotient.Tractability.CooperativeHBondApproximation.independent_approximates_cooperative N scores α h_unit

-- ---------------------------------------------------------------------------
-- Explicit water placement theorem family
-- ---------------------------------------------------------------------------

/-- Discrete water grid approximates continuous placement with Lipschitz error. -/
theorem discrete_water_placement_approximation
    {Wcont Wgrid : Type*}
    [Fintype Wgrid] [Nonempty Wgrid]
    (bridgeCont : Wcont → ℝ) (bridgeGrid : Wgrid → ℝ)
    (nearest : Wcont → Wgrid)
    (h : ℝ)
    (h_approx : ∀ wc, |bridgeCont wc - bridgeGrid (nearest wc)| ≤ h)
    (w_opt : Wcont) :
    bridgeCont w_opt - h ≤ DecisionQuotient.Tractability.ExplicitWaterPlacement.bestWaterBridge bridgeGrid :=
  DecisionQuotient.Tractability.ExplicitWaterPlacement.discrete_placement_approximation
    bridgeCont bridgeGrid nearest h h_approx w_opt

/-- Water bridge composes additively with base chemistry (zero added error). -/
theorem water_bridge_additive_composition {A : Type*} {S : Type*}
    (base_exact base_coarse : DecisionQuotient.DecisionProblem A S)
    (δ_base : ℝ)
    (h_base : DecisionQuotient.Tractability.CoarseApproximation.UniformUtilityApprox base_exact base_coarse δ_base)
    {W : Type*} [Fintype W] [Nonempty W]
    (bridge : W → A → S → ℝ) :
    DecisionQuotient.Tractability.CoarseApproximation.UniformUtilityApprox
      (DecisionQuotient.Tractability.CoarseApproximation.sumDecisionProblems base_exact
        (DecisionQuotient.Tractability.ExplicitWaterPlacement.waterPlacementDecisionProblem bridge))
      (DecisionQuotient.Tractability.CoarseApproximation.sumDecisionProblems base_coarse
        (DecisionQuotient.Tractability.ExplicitWaterPlacement.waterPlacementDecisionProblem bridge))
      (δ_base + 0) :=
  DecisionQuotient.Tractability.ExplicitWaterPlacement.water_bridge_additive_with_base
    base_exact base_coarse δ_base h_base bridge

-- ---------------------------------------------------------------------------
-- Receptor flexibility / ensemble docking theorem family
-- ---------------------------------------------------------------------------

/-- Rigid model at r₀ uniformly approximates the flexible model at any
    single conformation r. -/
theorem rigid_approximates_conformation {A : Type*} {S : Type*}
    {R : Type*} [Fintype A] [Fintype S] [Fintype R]
    [Nonempty A] [Nonempty S] [Nonempty R]
    (score : A → S → R → ℝ) (r₀ r : R) :
    DecisionQuotient.Tractability.CoarseApproximation.UniformUtilityApprox
      (DecisionQuotient.Tractability.ReceptorFlexibility.flexibleDecisionProblem score r)
      (DecisionQuotient.Tractability.ReceptorFlexibility.rigidDecisionProblem score r₀)
      (DecisionQuotient.Tractability.ReceptorFlexibility.conformationalErrorRadius score r₀) :=
  DecisionQuotient.Tractability.ReceptorFlexibility.rigid_approximates_conformation score r₀ r

/-- Boltzmann-weighted ensemble average approximates any reference conformation
    with error ≤ max conformational deviation. -/
theorem boltzmann_ensemble_approximation {R : Type*} [Fintype R] [Nonempty R]
    (weights : R → ℝ) (score : R → ℝ)
    (h_nonneg : ∀ r, 0 ≤ weights r)
    (h_sum_one : Finset.univ.sum weights = 1)
    (r₀ : R) :
    |DecisionQuotient.Tractability.ReceptorFlexibility.boltzmannEnsembleScore weights score - score r₀| ≤
      Finset.univ.sup' Finset.univ_nonempty (fun r => |score r - score r₀|) :=
  DecisionQuotient.Tractability.ReceptorFlexibility.boltzmann_between_extremes
    weights score h_nonneg h_sum_one r₀

/-- Multiplicative-separable tractability alias. -/
theorem multiplicative_separable_empty_sufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (I : Finset (Fin n))
    (hcase :
      (∀ s, hms.stateFactor s > 0) ∨
        (∀ s, hms.stateFactor s < 0) ∨
          (∀ s, hms.stateFactor s = 0)) :
    dp.isSufficient I :=
  DecisionQuotient.MultiplicativeSeparable.empty_sufficient
    (dp := dp) (hms := hms) (I := I) hcase

/-- Strict-dominance tractability alias. -/
theorem strict_global_dominance_empty_sufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hdom : StrictGlobalDominance (dp := dp))
    (I : Finset (Fin n)) :
    dp.isSufficient I :=
  DecisionQuotient.StrictGlobalDominance.empty_sufficient (dp := dp) (hdom := hdom) I

/-- Constant-optimal-set tractability alias. -/
theorem constant_optimal_set_empty_sufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hconst : ConstantOptimalSet (dp := dp))
    (I : Finset (Fin n)) :
    dp.isSufficient I :=
  DecisionQuotient.ConstantOptimalSet.empty_sufficient (dp := dp) (hconst := hconst) I

/-- Single-action tractability alias. -/
theorem single_action_all_sufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S] {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hcard : dp.actions.card = 1)
    (I : Finset (Fin n)) :
    dp.isSufficient I :=
  DecisionQuotient.single_action_all_sufficient (dp := dp) hcard I

/-- Bounded-state-space brute-force tractability alias. -/
theorem bounded_state_space_bruteforce
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [Fintype S]
    (k : ℕ) (hk : Fintype.card S ≤ k)
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (I : Finset (Fin n)) :
    dp.isSufficient I ↔
      ∀ (s₁ : S) (hs₁ : s₁ ∈ dp.states) (s₂ : S) (hs₂ : s₂ ∈ dp.states),
        (∀ i ∈ I, CoordinateSpace.proj s₁ i = CoordinateSpace.proj s₂ i) →
          dp.optimalActions s₁ = dp.optimalActions s₂ :=
  DecisionQuotient.bounded_state_sufficiency_trivial k hk dp I

/-- Tightness alias. -/
theorem tractability_tightness :
    (∀ (n : ℕ) (φ : Formula n),
      (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology) ∧
    (∀ (n : ℕ) (φ : Formula n), (∃ a, φ.eval a = false) →
      ¬∃ (av : ReductionAction → ℝ) (sv : ReductionState n → ℝ),
        ∀ a s, reductionUtility φ a s = av a + sv s) ∧
    (∀ {A S : Type*} [Unique A] (dp : DecisionProblem A S)
      {n : ℕ} [CoordinateSpace S n] (I : Finset (Fin n)), dp.isSufficient I) :=
  DecisionQuotient.tractability_conditions_tight

/-- Parameterized-results alias. -/
theorem parameterized_results :
    (∀ {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
        {n : ℕ} [CoordinateSpace S n]
        [∀ s s' : S, ∀ I : Finset (Fin n), Decidable (agreeOn s s' I)]
        (cdp : ComputableDecisionProblem A S),
        ∃ f : ℕ → ℕ, ∃ algo : Finset (Fin n) → Bool,
          (∀ I, algo I = true ↔ cdp.toAbstract.isSufficient I) ∧
          (∀ k, 1 ≤ f k)) ∧
    (∀ {n : ℕ} (φ : Formula n),
      (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology) :=
  DecisionQuotient.parameterized_complexity_summary

/-- Dichotomy alias in terms of minimal-sufficient-set size relative to state
cardinality. -/
theorem complexity_dichotomy
    {S : Type*} {n : ℕ}
    [Fintype S] [Nonempty S]
    (I : Finset (Fin n)) :
    (I.card ≤ Nat.log 2 (Fintype.card S)) ∨
    (I.card > Nat.log 2 (Fintype.card S)) := by
  by_cases h : I.card ≤ Nat.log 2 (Fintype.card S)
  · exact Or.inl h
  · exact Or.inr (lt_of_not_ge h)

end DecisionQuotient.Summary
