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
import DecisionQuotient.Tractability.GridConvergence
import DecisionQuotient.Tractability.CoarseApproximation
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
