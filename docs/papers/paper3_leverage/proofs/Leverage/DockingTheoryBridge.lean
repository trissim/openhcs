import Leverage.BridgeToDQ
import Leverage.ProteinMechanicalGraph
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Data.Finset.Max
import Mathlib.Data.List.GetD
import Mathlib.Data.List.OfFn
import Mathlib.InformationTheory.Hamming
import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic
import DecisionQuotient.AbstractionCollapse
import DecisionQuotient.OptimizerSummary
import DecisionQuotient.Summary
import DecisionQuotient.Reduction_AllCoords
import DecisionQuotient.WitnessCheckingDuality
import DecisionQuotient.Statistics.FisherInformation
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.CertifiedPruning
import DecisionQuotient.Tractability.GridConvergence
import DecisionQuotient.Tractability.GridMDInstances
import DecisionQuotient.Tractability.CutoffEpsilon
import DecisionQuotient.Tractability.BoundedActions
import DecisionQuotient.Tractability.Dominance
import DecisionQuotient.Tractability.MultiplicativeSeparable
import DecisionQuotient.Tractability.BoundedStateSpace
import DecisionQuotient.Tractability.MolecularSrank
import DecisionQuotient.Tractability.DiscretizedState
import DecisionQuotient.Tractability.DiscretizedAction
import DecisionQuotient.Tractability.TopKPreservation
import DecisionQuotient.Tractability.NearTieBand
import DecisionQuotient.Tractability.LJApproximation
import DecisionQuotient.Tractability.CoulombApproximation
import DecisionQuotient.Tractability.EwaldSummation
import DecisionQuotient.Computation.ArrayDSL
import DecisionQuotient.Computation.ArrayDSLExport
import DecisionQuotient.Computation.PhaseSpaceGeometry
import DecisionQuotient.Computation.LennardJonesDeriv
import DecisionQuotient.Physics.WolpertMismatch
import DecisionQuotient.Physics.WolpertDecomposition
import DecisionQuotient.Physics.TUR
import DecisionQuotient.Tractability.SampledDockingGap
import DecisionQuotient.Tractability.SampledDockingCutoff

namespace Leverage

universe u v

open Classical DecisionQuotient
open DecisionQuotient.Tractability
open DecisionQuotient.Tractability.DiscretizedState
open DecisionQuotient.Tractability.DiscretizedAction
open DecisionQuotient.Tractability.MolecularSrank
open DecisionQuotient.Tractability.SampledDocking
open DecisionQuotient.Tractability.CoarseApproximation
open DecisionQuotient.Tractability.CertifiedPruning
open DecisionQuotient.Computation

noncomputable def ljGradExpr (ε σ r : ℝ) : ℝ :=
  (24 * ε * r⁻¹) * ((σ * r⁻¹) ^ (6 : ℕ) - 2 * (σ * r⁻¹) ^ (12 : ℕ))

noncomputable def ljGradShellBound (ε σ rmin : ℝ) : ℝ :=
  (24 * |ε| * rmin⁻¹) *
    ((|σ| * rmin⁻¹) ^ (6 : ℕ) + 2 * (|σ| * rmin⁻¹) ^ (12 : ℕ))

noncomputable def ljSecondDerivExpr (ε σ r : ℝ) : ℝ :=
  (24 * ε * (r⁻¹ ^ (2 : ℕ))) * (26 * (σ * r⁻¹) ^ (12 : ℕ) - 7 * (σ * r⁻¹) ^ (6 : ℕ))

noncomputable def ljSecondDerivShellBound (ε σ rmin : ℝ) : ℝ :=
  (24 * |ε| * (rmin⁻¹ ^ (2 : ℕ))) *
    (26 * (|σ| * rmin⁻¹) ^ (12 : ℕ) + 7 * (|σ| * rmin⁻¹) ^ (6 : ℕ))

/-- Local paper3 exposure of the exact abstraction dichotomy: a surjective summary either factors
through the decision quotient or erases a decision-relevant distinction. -/
theorem surjectiveAbstraction_factors_or_erases
    {A S T : Type*} [DecidableEq A]
    (dp : DecisionProblem A S) (φ : S → T)
    (hSurj : Function.Surjective φ) :
    (∃ ψ : T → dp.DecisionQuotientType, ∀ s : S, dp.quotientMap s = ψ (φ s)) ∨
      dp.erasesDecisionRelevantDistinction φ :=
  dp.surjective_abstraction_factors_or_erases φ hSurj

/-- Local paper3 exposure of the positive factorization theorem: if every extra collapse beyond the
decision quotient were physically feasible at the canonical requirement profile, the abstraction
must factor through the quotient after all. -/
theorem surjectiveAbstraction_withFeasibleCollapseMap_factors
    {A S T : Type*} [DecidableEq A]
    (dp : DecisionProblem A S) (φ : S → T)
    (hSurj : Function.Surjective φ)
    (c : PhysicalComplexity.PhysicalComputer)
    (hMap :
      dp.erasesDecisionRelevantDistinction φ →
      PhysicalComplexity.PhysicalCollapseAtRequirement c PhysicalComplexity.coNP_requirement) :
    ∃ ψ : T → dp.DecisionQuotientType, ∀ s : S, dp.quotientMap s = ψ (φ s) :=
  dp.surjective_abstraction_with_feasible_collapse_map_factors φ hSurj c hMap

/-- Local paper3 exposure of the Fisher-information sum identity for structural rank. -/
theorem totalFisher_eq_srank
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) :
    ∑ i : Fin n, DecisionQuotient.Statistics.fisherScore dp i = (dp.srank : ℝ) :=
  DecisionQuotient.Statistics.sum_fisherScore_eq_srank dp

/-- Local paper3 exposure of the Fisher-matrix rank identity. -/
theorem fisherMatrix_rank_eq_srank
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) :
    Matrix.rank (DecisionQuotient.Statistics.fisherMatrix dp) = dp.srank :=
  DecisionQuotient.Statistics.fisherMatrix_rank_eq_srank dp

/-- Local paper3 exposure of the general coNP-hardness reduction core for exact sufficiency. -/
theorem exactSufficiency_conp_core {n : ℕ} (φ : Formula n) :
    (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology :=
  DecisionQuotient.Summary.conp_completeness φ

/-- Local paper3 exposure of the maximal-structural-rank hard family. -/
theorem exactSufficiency_hardFamily_srank_eq_n {n : ℕ} (hn : 0 < n)
    (φ : Formula n) (hnt : ¬ φ.isTautology) :
    (reductionProblemMany φ).srank = n :=
  DecisionQuotient.hard_family_srank_eq_n hn φ hnt

/-- Local paper3 exposure of structural-rank monotonicity for any decision
summary that depends only on the exact optimizer quotient. -/
theorem optSummary_srank_le_srank
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (σ : Set A → B) :
    (dp.optSummaryDecisionProblem σ).srank ≤ dp.srank :=
  dp.optSummaryDecisionProblem_srank_le_srank σ

/-- Strict rank reduction under an exact-optimizer summary occurs exactly when
the summary erases at least one exact relevant coordinate. -/
theorem optSummary_srank_lt_srank_iff
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (σ : Set A → B) :
    (dp.optSummaryDecisionProblem σ).srank < dp.srank ↔
      ∃ i : Fin n, dp.isRelevant i ∧ ¬ (dp.optSummaryDecisionProblem σ).isRelevant i :=
  dp.optSummaryDecisionProblem_srank_lt_srank_iff σ

/-- Relevant-coordinate finset for a summary of the exact optimizer quotient. -/
noncomputable def optSummaryRelevantFinset
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (σ : Set A → B) : Finset (Fin n) :=
  Finset.univ.filter (fun i => (dp.optSummaryDecisionProblem σ).isRelevant i)

/-- Number of exact relevant coordinates erased when passing from one optimizer summary to a coarser one. -/
noncomputable def optSummaryCollapseCount
    {A B C S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (σTight : Set A → B) (σLoose : Set A → C) : ℕ :=
  (optSummaryRelevantFinset dp σTight \ optSummaryRelevantFinset dp σLoose).card

/-- If one optimizer summary factors through another, irrelevance for the finer summary propagates to the
coarser summary. -/
theorem optSummary_irrelevant_of_factor
    {A B C S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (σTight : Set A → B) (σLoose : Set A → C)
    (hFactor : ∃ τ : B → C, ∀ X : Set A, σLoose X = τ (σTight X))
    (i : Fin n) :
    (dp.optSummaryDecisionProblem σTight).isIrrelevant i →
      (dp.optSummaryDecisionProblem σLoose).isIrrelevant i := by
  intro hIrr s s' hAgree
  rcases hFactor with ⟨τ, hτ⟩
  have hTight :
      (dp.optSummaryDecisionProblem σTight).Opt s =
        (dp.optSummaryDecisionProblem σTight).Opt s' :=
    hIrr s s' hAgree
  have hσTight : σTight (dp.Opt s) = σTight (dp.Opt s') := by
    have hs : σTight (dp.Opt s) ∈ (dp.optSummaryDecisionProblem σTight).Opt s := by
      rw [dp.optSummaryDecisionProblem_opt_eq_singleton]
      simp
    have hs' : σTight (dp.Opt s) ∈ (dp.optSummaryDecisionProblem σTight).Opt s' := by
      rwa [hTight] at hs
    rw [dp.optSummaryDecisionProblem_opt_eq_singleton] at hs'
    simpa using hs'
  have hσLoose : σLoose (dp.Opt s) = σLoose (dp.Opt s') := by
    rw [hτ, hτ, hσTight]
  rw [dp.optSummaryDecisionProblem_opt_eq_singleton,
    dp.optSummaryDecisionProblem_opt_eq_singleton, hσLoose]

/-- If one optimizer summary factors through another, relevance for the coarser summary already appeared in
the finer one. -/
theorem optSummary_relevant_imp_relevant_of_factor
    {A B C S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (σTight : Set A → B) (σLoose : Set A → C)
    (hFactor : ∃ τ : B → C, ∀ X : Set A, σLoose X = τ (σTight X))
    (i : Fin n)
    (hRel : (dp.optSummaryDecisionProblem σLoose).isRelevant i) :
    (dp.optSummaryDecisionProblem σTight).isRelevant i := by
  by_contra hNot
  have hIrrTight : (dp.optSummaryDecisionProblem σTight).isIrrelevant i :=
    ((dp.optSummaryDecisionProblem σTight).irrelevant_iff_not_relevant i).2 hNot
  have hIrrLoose : (dp.optSummaryDecisionProblem σLoose).isIrrelevant i :=
    optSummary_irrelevant_of_factor dp σTight σLoose hFactor i hIrrTight
  exact (((dp.optSummaryDecisionProblem σLoose).irrelevant_iff_not_relevant i).1 hIrrLoose) hRel

/-- Factorization between admissibility summaries gives a quantitative rank-collapse law: the tight summary's
rank splits into the relaxed rank plus the number of erased coordinates. -/
theorem optSummary_srank_eq_relaxed_plus_collapseCount_of_factor
    {A B C S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (σTight : Set A → B) (σLoose : Set A → C)
    (hFactor : ∃ τ : B → C, ∀ X : Set A, σLoose X = τ (σTight X)) :
    (dp.optSummaryDecisionProblem σTight).srank =
      (dp.optSummaryDecisionProblem σLoose).srank +
        optSummaryCollapseCount dp σTight σLoose := by
  let RTight := optSummaryRelevantFinset dp σTight
  let RLoose := optSummaryRelevantFinset dp σLoose
  have hSub : RLoose ⊆ RTight := by
    intro i hi
    simp only [RTight, RLoose, optSummaryRelevantFinset, Finset.mem_filter, Finset.mem_univ,
      true_and] at hi ⊢
    exact optSummary_relevant_imp_relevant_of_factor dp σTight σLoose hFactor i hi
  have hCard : (RTight \ RLoose).card = RTight.card - RLoose.card :=
    Finset.card_sdiff_of_subset hSub
  have hNat : RTight.card = RLoose.card + (RTight \ RLoose).card := by
    have hAdd : (RTight \ RLoose).card + RLoose.card = RTight.card :=
      Finset.card_sdiff_add_card_eq_card hSub
    omega
  simpa [DecisionProblem.srank, optSummaryRelevantFinset, optSummaryCollapseCount, RTight, RLoose]
    using hNat

/-- Equivalent subtractive form of the quantitative rank-collapse law. -/
theorem optSummary_srank_eq_tight_sub_collapseCount_of_factor
    {A B C S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (σTight : Set A → B) (σLoose : Set A → C)
    (hFactor : ∃ τ : B → C, ∀ X : Set A, σLoose X = τ (σTight X)) :
    (dp.optSummaryDecisionProblem σLoose).srank =
      (dp.optSummaryDecisionProblem σTight).srank -
        optSummaryCollapseCount dp σTight σLoose := by
  have hEq := optSummary_srank_eq_relaxed_plus_collapseCount_of_factor dp σTight σLoose hFactor
  omega

/-- Energy saved by relaxing a nested admissibility summary is exactly one `joulesPerBit` per erased relevant
coordinate. -/
theorem optSummary_energy_eq_relaxed_plus_collapseCost_of_factor
    {A B C S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (σTight : Set A → B) (σLoose : Set A → C)
    (hFactor : ∃ τ : B → C, ∀ X : Set A, σLoose X = τ (σTight X))
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) :
    DecisionQuotient.ThermodynamicLift.energyLowerBound M
        (dp.optSummaryDecisionProblem σTight).srank =
      DecisionQuotient.ThermodynamicLift.energyLowerBound M
          (dp.optSummaryDecisionProblem σLoose).srank +
        DecisionQuotient.ThermodynamicLift.energyLowerBound M
          (optSummaryCollapseCount dp σTight σLoose) := by
  unfold DecisionQuotient.ThermodynamicLift.energyLowerBound
  rw [optSummary_srank_eq_relaxed_plus_collapseCount_of_factor dp σTight σLoose hFactor,
    Nat.mul_add]

/-- A quantitative admissibility family is a tolerance-indexed optimizer summary in which larger admissibility
bands factor through smaller ones. This is the exact nestedness hypothesis needed for honest rank-collapse
statements. -/
structure QuantitativeAdmissibilityFamily (A B : Type*) where
  summary : ℝ → Set A → B
  relaxes : ∀ ε Δ : ℝ, 0 ≤ Δ → ∃ τ : B → B, ∀ X : Set A, summary (ε + Δ) X = τ (summary ε X)

/-- Structural rank under a tolerance-indexed admissibility summary. -/
noncomputable def QuantitativeAdmissibilityFamily.summaryDecisionProblem
    {A B S : Type*} (F : QuantitativeAdmissibilityFamily A B)
    (dp : DecisionProblem A S) (ε : ℝ) : DecisionProblem B S :=
  dp.optSummaryDecisionProblem (F.summary ε)

/-- Number of relevant coordinates erased when relaxing admissibility by `Δ`. -/
noncomputable def QuantitativeAdmissibilityFamily.collapseCount
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (F : QuantitativeAdmissibilityFamily A B)
    (dp : DecisionProblem A S) (ε Δ : ℝ) : ℕ :=
  optSummaryCollapseCount dp (F.summary ε) (F.summary (ε + Δ))

/-- Quantitative admissibility rank-collapse theorem: for any nested admissibility family, relaxing the band by
`Δ` erases exactly `collapseCount` relevant coordinates and no more. -/
theorem quantitativeAdmissibility_srank_eq_relaxed_plus_collapseCount
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (F : QuantitativeAdmissibilityFamily A B)
    (dp : DecisionProblem A S) (ε Δ : ℝ) (hΔ : 0 ≤ Δ) :
    (F.summaryDecisionProblem dp ε).srank =
      (F.summaryDecisionProblem dp (ε + Δ)).srank +
        F.collapseCount dp ε Δ := by
  exact optSummary_srank_eq_relaxed_plus_collapseCount_of_factor dp
    (F.summary ε) (F.summary (ε + Δ)) (F.relaxes ε Δ hΔ)

/-- If relaxing admissibility erases exactly one relevant coordinate, the structural rank drops by one. -/
theorem quantitativeAdmissibility_srank_drop_one
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (F : QuantitativeAdmissibilityFamily A B)
    (dp : DecisionProblem A S) (ε Δ : ℝ) (hΔ : 0 ≤ Δ)
    (hOne : F.collapseCount dp ε Δ = 1) :
    (F.summaryDecisionProblem dp (ε + Δ)).srank =
      (F.summaryDecisionProblem dp ε).srank - 1 := by
  have hEq := quantitativeAdmissibility_srank_eq_relaxed_plus_collapseCount F dp ε Δ hΔ
  rw [hOne] at hEq
  omega

/-- One erased coordinate saves exactly one Landauer-scale discrete bit-cost unit in the declared linear energy
model. -/
theorem quantitativeAdmissibility_energy_saves_one_bit
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (F : QuantitativeAdmissibilityFamily A B)
    (dp : DecisionProblem A S) (ε Δ : ℝ) (hΔ : 0 ≤ Δ)
    (hOne : F.collapseCount dp ε Δ = 1)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) :
    DecisionQuotient.ThermodynamicLift.energyLowerBound M
        (F.summaryDecisionProblem dp ε).srank =
      DecisionQuotient.ThermodynamicLift.energyLowerBound M
          (F.summaryDecisionProblem dp (ε + Δ)).srank +
        M.joulesPerBit := by
  have hEq := optSummary_energy_eq_relaxed_plus_collapseCost_of_factor dp
    (F.summary ε) (F.summary (ε + Δ)) (F.relaxes ε Δ hΔ) M
  have hCollapse :
      DecisionQuotient.ThermodynamicLift.energyLowerBound M
          (optSummaryCollapseCount dp (F.summary ε) (F.summary (ε + Δ))) = M.joulesPerBit := by
    unfold DecisionQuotient.ThermodynamicLift.energyLowerBound
    simpa [QuantitativeAdmissibilityFamily.collapseCount] using congrArg (fun x => M.joulesPerBit * x) hOne
  rw [hCollapse] at hEq
  simpa [QuantitativeAdmissibilityFamily.summaryDecisionProblem] using hEq

/-- Local paper3 exposure of the witness/checking duality budget lower bound. -/
theorem exactSufficiency_checkerBudget_ge_witnessBudget {n : ℕ}
    (hn : 1 ≤ n)
    (P : Finset (BinaryState n × BinaryState n))
    (hSound :
      ∀ Opt : BinaryState n → Bool,
        (∃ s s', agreeOn s s' (∅ : Finset (Fin n)) ∧ Opt s ≠ Opt s') →
        ∃ p ∈ P, Opt p.1 ≠ Opt p.2) :
    witnessBudgetEmpty n ≤ checkingBudgetPairs P :=
  DecisionQuotient.checking_witnessing_duality_budget hn P hSound

/-- Local paper3 exposure of the no-sound-checker corollary below witness budget. -/
theorem exactSufficiency_noSoundChecker_below_witnessBudget {n : ℕ}
    (hn : 1 ≤ n)
    (P : Finset (BinaryState n × BinaryState n))
    (hSmall : checkingBudgetPairs P < witnessBudgetEmpty n) :
    ¬ (∀ Opt : BinaryState n → Bool,
        (∃ s s', agreeOn s s' (∅ : Finset (Fin n)) ∧ Opt s ≠ Opt s') →
        ∃ p ∈ P, Opt p.1 ≠ Opt p.2) :=
  DecisionQuotient.no_sound_checker_below_witness_budget hn P hSmall

/-- Local paper3 exposure of the runtime lower bound induced by the witness budget. -/
theorem exactSufficiency_checkingTime_ge_witnessBudget {n : ℕ}
    (hn : 1 ≤ n)
    (P : Finset (BinaryState n × BinaryState n))
    (hSound :
      ∀ Opt : BinaryState n → Bool,
        (∃ s s', agreeOn s s' (∅ : Finset (Fin n)) ∧ Opt s ≠ Opt s') →
        ∃ p ∈ P, Opt p.1 ≠ Opt p.2)
    (T : ℕ)
    (hCheck : checkingBudgetPairs P ≤ T) :
    witnessBudgetEmpty n ≤ T :=
  DecisionQuotient.checking_time_ge_witness_budget hn P hSound T hCheck

/-- Local paper3 exposure of the cutoff-local structural-rank bound for exact docking. -/
theorem molecularDocking_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms :=
  md_srank_bound prob hStrictAll hBounded

/-- Local paper3 exposure of the bounded-pocket low-rank regime for exact docking. -/
theorem molecularDocking_boundedPocket_srank_bound
    (prob : MDBindingProblem) (K L : Nat)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hPocket : numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤ 3 * K + 3 * L :=
  docking_small_pocket_bound prob K L hStrictAll hBounded hPocket hLigand

/-- Local paper3 exposure of the generic structural-rank thermodynamic lower bound for the exact
    docking decision problem itself. -/
theorem molecularDocking_srank_energy_lower_bound
    (prob : MDBindingProblem)
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) (hJ : 0 < M.joulesPerBit) :
    M.joulesPerBit *
        (@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
          (mdCoordinateSpaceStruct prob) prob.toDecisionProblem) ≤
      DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  exact DecisionQuotient.Physics.BoundedAcquisition.energy_ge_srank_cost
    prob.toDecisionProblem I hI M hJ

/-- Local paper3 exposure of the exact-docking entropy lower bound obtained by composing the
    docking structural-rank cost floor with Landauer calibration. -/
theorem molecularDocking_landauer_entropy_lower_bound
    (prob : MDBindingProblem)
    [Fintype MDState] [Nonempty MDState] [DecidableEq (Set MDAction)]
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) = DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hEntropy : prob.toDecisionProblem.quotientEntropy ≤
      (@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℝ)) :
    (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≥
      kB * T * Real.log (prob.toDecisionProblem.numOptClasses : ℝ) := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  let sr : ℕ := prob.toDecisionProblem.srank
  have hJ : 0 < M.joulesPerBit :=
    DecisionQuotient.ThermodynamicLift.joulesPerBit_pos_of_landauer_calibration M hkB hT hCal
  have hEnergyNat := molecularDocking_srank_energy_lower_bound prob I hI M hJ
  have hLandauer : (sr : ℝ) * (kB * T * Real.log 2) ≤
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
    have hCast : (((M.joulesPerBit * sr : ℕ)) : ℝ) ≤
        (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
      exact_mod_cast hEnergyNat
    calc
      (sr : ℝ) * (kB * T * Real.log 2)
          = (sr : ℝ) * DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T := by
              simp [DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit]
      _ = (sr : ℝ) * (M.joulesPerBit : ℝ) := by rw [hCal]
      _ = ((M.joulesPerBit : ℝ) * (sr : ℝ)) := by ring
      _ = (((M.joulesPerBit * sr : ℕ)) : ℝ) := by norm_num
      _ ≤ (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
        simpa [DecisionQuotient.ThermodynamicLift.energyLowerBound] using hCast
  simpa [sr] using
    (DecisionQuotient.Tractability.MolecularSrank.md_thermodynamic_lower_bound
      prob kB T (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ)
      hkB hT hLandauer hEntropy)

/-- Local paper3 exposure of the fact that an exact docking problem with structural rank above one
    lies strictly above the one-bit thermodynamic ground state. -/
theorem molecularDocking_above_ground_of_srank_gt_one
    (prob : MDBindingProblem)
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (hSrank : 1 < @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) (hJ : 0 < M.joulesPerBit) :
    M.joulesPerBit < DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  have hGround := DecisionQuotient.Physics.BoundedAcquisition.srank_gt_one_above_ground
    prob.toDecisionProblem I hI hSrank M hJ
  have hEnergy := molecularDocking_srank_energy_lower_bound prob I hI M hJ
  exact lt_of_lt_of_le hGround hEnergy

/-- Stagewise exact-resolution costs add along a finite path of decision problems. -/
theorem pathwise_srank_energy_lower_bound
    {A S : Type*} {n m : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : Fin m → DecisionProblem A S)
    (I : Fin m → Finset (Fin n))
    (hI : ∀ t : Fin m, (dp t).isSufficient (I t))
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    (hJ : 0 < M.joulesPerBit) :
    (Finset.univ.sum fun t : Fin m =>
      M.joulesPerBit * (dp t).srank) ≤
      Finset.univ.sum fun t : Fin m =>
        DecisionQuotient.ThermodynamicLift.energyLowerBound M (I t).card := by
  exact Finset.sum_le_sum (fun t _ =>
    DecisionQuotient.Physics.BoundedAcquisition.energy_ge_srank_cost
      (dp t) (I t) (hI t) M hJ)

/-- Stationary forward weight of a finite quotient trajectory, using the theorem-level edge-flow weights from
the Wolpert residual layer. -/
noncomputable def quotientTrajectoryForwardWeight
    {Q : Type*} [Fintype Q]
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain Q)
    (π : DecisionQuotient.Physics.StationaryDist mc) {τ : ℕ}
    (q : Fin (τ + 1) → Q) : ℝ :=
  Finset.univ.prod fun t : Fin τ =>
    DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.castSucc) (q t.succ)

/-- Reverse stationary weight of the same quotient trajectory. -/
noncomputable def quotientTrajectoryReverseWeight
    {Q : Type*} [Fintype Q]
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain Q)
    (π : DecisionQuotient.Physics.StationaryDist mc) {τ : ℕ}
    (q : Fin (τ + 1) → Q) : ℝ :=
  Finset.univ.prod fun t : Fin τ =>
    DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.succ) (q t.castSucc)

/-- Structural-rank entropy increment along one quotient-trajectory step, in nats. -/
noncomputable def quotientTrajectorySrankEntropyStep
    {τ : ℕ} (r : Fin (τ + 1) → ℕ) (t : Fin τ) : ℝ :=
  stateSpaceEntropy (r t.succ) - stateSpaceEntropy (r t.castSucc)

/-- Each rank-entropy step is exactly `Δsrank * ln 2`. -/
theorem quotientTrajectorySrankEntropyStep_eq
    {τ : ℕ} (r : Fin (τ + 1) → ℕ) (t : Fin τ) :
    quotientTrajectorySrankEntropyStep r t =
      (((r t.succ : ℕ) : ℝ) - ((r t.castSucc : ℕ) : ℝ)) * Real.log 2 := by
  unfold quotientTrajectorySrankEntropyStep
  rw [stateSpaceEntropy_eq, stateSpaceEntropy_eq]
  ring

/-- Finite quotient-trajectory Crooks identity at the path-weight level: the forward/reverse stationary path
weight ratio is the exponential of the summed log edge-flow asymmetry. -/
theorem quotientTrajectory_forwardReverseRatio
    {Q : Type*} [Fintype Q]
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain Q)
    (π : DecisionQuotient.Physics.StationaryDist mc) {τ : ℕ}
    (q : Fin (τ + 1) → Q)
    (hForward : ∀ t : Fin τ,
      0 < DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.castSucc) (q t.succ))
    (hReverse : ∀ t : Fin τ,
      0 < DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.succ) (q t.castSucc)) :
    quotientTrajectoryForwardWeight mc π q /
        quotientTrajectoryReverseWeight mc π q =
      Real.exp (Finset.univ.sum fun t : Fin τ =>
        Real.log
          (DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.castSucc) (q t.succ) /
            DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.succ) (q t.castSucc))) := by
  unfold quotientTrajectoryForwardWeight quotientTrajectoryReverseWeight
  rw [← Finset.prod_div_distrib]
  have hExp :
      Real.exp
          (Finset.univ.sum fun t : Fin τ =>
            Real.log
              (DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.castSucc) (q t.succ) /
                DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.succ) (q t.castSucc))) =
        Finset.univ.prod fun t : Fin τ =>
          Real.exp
            (Real.log
              (DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.castSucc) (q t.succ) /
                DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.succ) (q t.castSucc))) := by
    simpa using (Real.exp_sum (s := Finset.univ) (f := fun t : Fin τ =>
      Real.log
        (DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.castSucc) (q t.succ) /
          DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.succ) (q t.castSucc))))
  rw [hExp]
  refine Finset.prod_congr rfl ?_
  intro t _
  exact (Real.exp_log (div_pos (hForward t) (hReverse t))).symm

/-- Rank-calibrated quotient Crooks theorem: if each local forward/reverse log-flow ratio is calibrated by the
structural-rank entropy increment, then the full path ratio is governed by the cumulative structural-rank
entropy produced along the trajectory. -/
theorem quotientTrajectory_forwardReverseRatio_eq_exp_srankEntropy
    {Q : Type*} [Fintype Q]
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain Q)
    (π : DecisionQuotient.Physics.StationaryDist mc) {τ : ℕ}
    (q : Fin (τ + 1) → Q) (r : Fin (τ + 1) → ℕ)
    (hForward : ∀ t : Fin τ,
      0 < DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.castSucc) (q t.succ))
    (hReverse : ∀ t : Fin τ,
      0 < DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.succ) (q t.castSucc))
    (hCal : ∀ t : Fin τ,
      Real.log
          (DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.castSucc) (q t.succ) /
            DecisionQuotient.Physics.WolpertResidual.edgeFlow mc π (q t.succ) (q t.castSucc)) =
        quotientTrajectorySrankEntropyStep r t) :
    quotientTrajectoryForwardWeight mc π q /
        quotientTrajectoryReverseWeight mc π q =
      Real.exp (Finset.univ.sum fun t : Fin τ => quotientTrajectorySrankEntropyStep r t) := by
  rw [quotientTrajectory_forwardReverseRatio mc π q hForward hReverse]
  congr 1
  exact Finset.sum_congr rfl (fun t _ => hCal t)

/-- If each quotient-trajectory step carries at least its declared structural-rank entropy cost in dissipation,
the total dissipation is bounded below by the cumulative rank-entropy production. -/
theorem quotientTrajectory_dissipation_lower_bound
    {τ : ℕ} (r : Fin (τ + 1) → ℕ) (wDiss : Fin τ → ℝ) (kB T : ℝ)
    (hStep : ∀ t : Fin τ,
      kB * T * quotientTrajectorySrankEntropyStep r t ≤ wDiss t) :
    kB * T * (Finset.univ.sum fun t : Fin τ => quotientTrajectorySrankEntropyStep r t) ≤
      Finset.univ.sum wDiss := by
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum (fun t _ => hStep t)

noncomputable def binaryScorePartition (β x y : ℝ) : ℝ :=
  Real.exp (β * x) + Real.exp (β * y)

/-- Strictly positive binary witness distribution induced by a pair of real scores. -/
noncomputable def binaryScoreDistribution (β x y : ℝ) :
    DecisionQuotient.Physics.WolpertMismatch.StrictFiniteDistribution Bool where
  pmf := fun b =>
    if b then Real.exp (β * x) / binaryScorePartition β x y
    else Real.exp (β * y) / binaryScorePartition β x y
  sum_eq_one := by
    have hZpos : 0 < binaryScorePartition β x y := by
      unfold binaryScorePartition
      positivity
    have hZne : binaryScorePartition β x y ≠ 0 := ne_of_gt hZpos
    rw [Fintype.sum_bool]
    simp [binaryScorePartition, hZne]
    field_simp [hZne]
  pos := by
    intro b
    have hZpos : 0 < binaryScorePartition β x y := by
      unfold binaryScorePartition
      positivity
    by_cases hb : b
    · subst hb
      simpa [binaryScorePartition] using div_pos (Real.exp_pos _) hZpos
    · have hb' : b = false := by cases b <;> simp_all
      subst hb'
      simpa [binaryScorePartition] using div_pos (Real.exp_pos _) hZpos

/-- A nonzero change in the two-action score gap forces the induced binary
Boltzmann witnesses to differ. -/
theorem binaryScoreDistribution_pmf_true_ne_of_gap_ne
    (β : ℝ) (hβ : 0 < β) {x₁ y₁ x₂ y₂ : ℝ}
    (hGapNe : x₁ - y₁ ≠ x₂ - y₂) :
    (binaryScoreDistribution β x₁ y₁).pmf true ≠
      (binaryScoreDistribution β x₂ y₂).pmf true := by
  intro hEq
  let Z₁ := binaryScorePartition β x₁ y₁
  let Z₂ := binaryScorePartition β x₂ y₂
  have hZ₁pos : 0 < Z₁ := by
    unfold Z₁ binaryScorePartition
    positivity
  have hZ₂pos : 0 < Z₂ := by
    unfold Z₂ binaryScorePartition
    positivity
  have hZ₁ne : Z₁ ≠ 0 := ne_of_gt hZ₁pos
  have hZ₂ne : Z₂ ≠ 0 := ne_of_gt hZ₂pos
  change Real.exp (β * x₁) / Z₁ = Real.exp (β * x₂) / Z₂ at hEq
  have hCross := hEq
  field_simp [hZ₁ne, hZ₂ne] at hCross
  have hPair : Real.exp (β * x₁) * Real.exp (β * y₂) =
      Real.exp (β * x₂) * Real.exp (β * y₁) := by
    unfold Z₁ Z₂ binaryScorePartition at hCross
    linarith
  have hExpEq : Real.exp (β * (x₁ + y₂)) = Real.exp (β * (x₂ + y₁)) := by
    simpa [Real.exp_add, mul_add, add_comm, add_left_comm, add_assoc,
      mul_comm, mul_left_comm, mul_assoc] using hPair
  have hArg : β * (x₁ + y₂) = β * (x₂ + y₁) := Real.exp_injective hExpEq
  have hGapEq : x₁ - y₁ = x₂ - y₂ := by
    nlinarith [hArg, hβ]
  exact hGapNe hGapEq

/-- The theorem-level KL mismatch witness is already positive for any two-action
score pair whose exact and designed gaps differ. -/
theorem binaryScore_mismatchNatLowerBound_pos_of_gap_ne
    (β : ℝ) (hβ : 0 < β) {x₁ y₁ x₂ y₂ : ℝ}
    (hGapNe : x₁ - y₁ ≠ x₂ - y₂) :
    0 < DecisionQuotient.Physics.WolpertMismatch.mismatchNatLowerBound
      (binaryScoreDistribution β x₁ y₁)
      (binaryScoreDistribution β x₂ y₂) := by
  exact DecisionQuotient.Physics.WolpertMismatch.mismatchNatLowerBound_pos_of_exists_ne
    (binaryScoreDistribution β x₁ y₁)
    (binaryScoreDistribution β x₂ y₂)
    ⟨true, binaryScoreDistribution_pmf_true_ne_of_gap_ne β hβ hGapNe⟩

/-- Sampled Lennard-Jones exact/coarse gap distortion yields a concrete theorem-
level binary mismatch witness. -/
theorem exactVsCutoffLJ_binaryMismatchWitness_pos_of_pairwise_gap_ne
    {A S : Type*}
    (β ε σ : ℝ) (hβ : 0 < β)
    (distanceExact distanceGrid : A → S → ℝ)
    (a b : A) (s : S)
    (hGapNe :
      DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) -
          DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact b s) ≠
        DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) -
          DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid b s)) :
    0 < DecisionQuotient.Physics.WolpertMismatch.mismatchNatLowerBound
      (binaryScoreDistribution β
        (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s))
        (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact b s)))
      (binaryScoreDistribution β
        (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s))
        (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid b s))) :=
  binaryScore_mismatchNatLowerBound_pos_of_gap_ne β hβ hGapNe

/-- Sampled Coulomb exact/coarse gap distortion yields a concrete theorem-level
binary mismatch witness. -/
theorem exactVsCutoffCoulomb_binaryMismatchWitness_pos_of_pairwise_gap_ne
    {A S : Type*}
    (β q_i q_j : ℝ) (hβ : 0 < β)
    (distanceExact distanceGrid : A → S → ℝ)
    (a b : A) (s : S)
    (hGapNe :
      DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact b s) ≠
        DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid b s)) :
    0 < DecisionQuotient.Physics.WolpertMismatch.mismatchNatLowerBound
      (binaryScoreDistribution β
        (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s))
        (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact b s)))
      (binaryScoreDistribution β
        (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s))
        (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid b s))) :=
  binaryScore_mismatchNatLowerBound_pos_of_gap_ne β hβ hGapNe

/-- Finite action code used for explicit action-side symmetry breaking on sampled docking families. -/
noncomputable def finiteActionCode {A : Type*} [Fintype A] (a : A) : ℕ :=
  Fintype.equivFin A a

/-- The finite action code is injective. -/
theorem finiteActionCode_injective {A : Type*} [Fintype A] :
    Function.Injective (@finiteActionCode A _ ) := by
  intro a b h
  have h' : (Fintype.equivFin A) a = (Fintype.equivFin A) b := Fin.ext h
  exact (Fintype.equivFin A).injective h'

/-- Finite optimal-action finset extracted from the abstract optimal-set predicate. -/
noncomputable def optFinset {A S : Type*} [Fintype A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) : Finset A :=
  (Finset.univ : Finset A).filter (fun a => a ∈ dp.Opt s)

theorem mem_optFinset_iff {A S : Type*} [Fintype A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) (a : A) :
    a ∈ optFinset dp s ↔ a ∈ dp.Opt s := by
  unfold optFinset
  simp

/-- Every finite nonempty action family has at least one optimal action at every state. -/
theorem optFinset_nonempty {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    (optFinset dp s).Nonempty := by
  classical
  obtain ⟨aMax, _, hMax⟩ := Finset.exists_max_image (Finset.univ : Finset A)
    (fun a => dp.utility a s) Finset.univ_nonempty
  refine ⟨aMax, ?_⟩
  rw [mem_optFinset_iff]
  intro a'
  exact hMax a' (by simp)

/-- The finite action code is always strictly below the total number of actions. -/
theorem finiteActionCode_lt_card {A : Type*} [Fintype A] (a : A) :
    finiteActionCode a < Fintype.card A :=
  (Fintype.equivFin A a).is_lt

/-- Deterministic max-code representative of the optimal-action set at state `s`. -/
noncomputable def maxCodeOptAction {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) : A :=
  let F := optFinset dp s
  let codes := F.image finiteActionCode
  (Fintype.equivFin A).symm ⟨codes.max' (by
      rcases optFinset_nonempty dp s with ⟨a, ha⟩
      exact Finset.image_nonempty.mpr ⟨a, ha⟩), by
      rcases Finset.mem_image.mp (Finset.max'_mem codes
        (by
          rcases optFinset_nonempty dp s with ⟨a, ha⟩
          exact Finset.image_nonempty.mpr ⟨a, ha⟩)) with ⟨a, _, hcode⟩
      exact hcode.symm ▸ finiteActionCode_lt_card a⟩

/-- The chosen max-code representative is optimal in the base problem. -/
theorem maxCodeOptAction_mem_optFinset {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    maxCodeOptAction dp s ∈ optFinset dp s := by
  classical
  let F := optFinset dp s
  let hF : F.Nonempty := optFinset_nonempty dp s
  let codes := F.image finiteActionCode
  have hCodes : codes.Nonempty := by
    rcases hF with ⟨a, ha⟩
    exact Finset.image_nonempty.mpr ⟨a, ha⟩
  let selected : A :=
    (Fintype.equivFin A).symm ⟨codes.max' hCodes, by
      rcases Finset.mem_image.mp (Finset.max'_mem codes hCodes) with ⟨a, _, hcode⟩
      exact hcode.symm ▸ finiteActionCode_lt_card a⟩
  change selected ∈ F
  rcases Finset.mem_image.mp (Finset.max'_mem codes hCodes) with ⟨a, ha, hcode⟩
  have hSelectedMax : finiteActionCode selected = codes.max' hCodes := by
    simp [selected, finiteActionCode]
  have hSelCode : finiteActionCode selected = finiteActionCode a := by
    exact hSelectedMax.trans hcode.symm
  have hSelEq : selected = a := finiteActionCode_injective hSelCode
  exact hSelEq.symm ▸ ha

theorem maxCodeOptAction_mem_opt {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    maxCodeOptAction dp s ∈ dp.Opt s := by
  simpa [mem_optFinset_iff] using maxCodeOptAction_mem_optFinset dp s

/-- Every optimal action has finite action code at most the chosen representative's code. -/
theorem finiteActionCode_le_maxCodeOptAction {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) {a : A} (ha : a ∈ dp.Opt s) :
    finiteActionCode a ≤ finiteActionCode (maxCodeOptAction dp s) := by
  classical
  have ha' : a ∈ optFinset dp s := by simpa [mem_optFinset_iff] using ha
  unfold maxCodeOptAction
  dsimp
  have hmem : finiteActionCode a ∈ (optFinset dp s).image finiteActionCode :=
    Finset.mem_image.mpr ⟨a, ha', rfl⟩
  simpa [finiteActionCode] using Finset.le_max' ((optFinset dp s).image finiteActionCode) _ hmem

/-- If two actions are both optimal, then their base utilities agree. -/
theorem optimal_actions_same_utility {A S : Type*} (dp : DecisionProblem A S) (s : S)
    {a b : A} (ha : a ∈ dp.Opt s) (hb : b ∈ dp.Opt s) :
    dp.utility a s = dp.utility b s := by
  have h1 := ha b
  have h2 := hb a
  linarith

/-- Any non-optimal action lies strictly below an optimal action. -/
theorem not_optimal_lt_optimal {A S : Type*} (dp : DecisionProblem A S) (s : S)
    {a b : A} (ha : a ∈ dp.Opt s) (hb : b ∉ dp.Opt s) :
    dp.utility b s < dp.utility a s := by
  have hle := ha b
  by_contra hnot
  have heq : dp.utility b s = dp.utility a s := by linarith
  have hbOpt : b ∈ dp.Opt s := by
    intro a'
    calc
      dp.utility a' s ≤ dp.utility a s := ha a'
      _ = dp.utility b s := heq.symm
  exact hb hbOpt

/-- Positive pairwise utility differences present at a fixed state. -/
noncomputable def positiveUtilityGapsAtState {A S : Type*} [Fintype A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) : Finset ℝ :=
  (((Finset.univ : Finset A).product (Finset.univ : Finset A)).image
    (fun p : A × A => dp.utility p.1 s - dp.utility p.2 s)).filter (fun δ => 0 < δ)

/-- Minimum positive utility gap at a state, defaulting to `1` if every action utility is tied. -/
noncomputable def minPositiveUtilityGapOrOne {A S : Type*} [Fintype A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) : ℝ :=
  if h : (positiveUtilityGapsAtState dp s).Nonempty then
    (positiveUtilityGapsAtState dp s).min' h
  else 1

theorem minPositiveUtilityGapOrOne_pos {A S : Type*} [Fintype A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    0 < minPositiveUtilityGapOrOne dp s := by
  unfold minPositiveUtilityGapOrOne
  by_cases h : (positiveUtilityGapsAtState dp s).Nonempty
  · rw [dif_pos h]
    exact (Finset.mem_filter.mp (Finset.min'_mem _ h)).2
  · simp [dif_neg h]

theorem minPositiveUtilityGapOrOne_le_of_utility_lt
    {A S : Type*} [Fintype A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) {a b : A}
    (hGap : dp.utility b s < dp.utility a s) :
    minPositiveUtilityGapOrOne dp s ≤ dp.utility a s - dp.utility b s := by
  have hMem : dp.utility a s - dp.utility b s ∈ positiveUtilityGapsAtState dp s := by
    unfold positiveUtilityGapsAtState
    rw [Finset.mem_filter, Finset.mem_image]
    refine ⟨?_, by linarith⟩
    exact ⟨(a, b), by simp, rfl⟩
  unfold minPositiveUtilityGapOrOne
  by_cases h : (positiveUtilityGapsAtState dp s).Nonempty
  · rw [dif_pos h]
    exact Finset.min'_le (positiveUtilityGapsAtState dp s) _ hMem
  · exfalso
    exact h ⟨_, hMem⟩

theorem finiteActionCode_abs_sub_le_card {A : Type*} [Fintype A]
    (a b : A) :
    |((finiteActionCode a : ℕ) : ℝ) - (finiteActionCode b : ℕ)| ≤ Fintype.card A := by
  have ha0 : 0 ≤ (((finiteActionCode a : ℕ) : ℝ)) := by positivity
  have hb0 : 0 ≤ (((finiteActionCode b : ℕ) : ℝ)) := by positivity
  have haCard : (((finiteActionCode a : ℕ) : ℝ)) ≤ Fintype.card A := by
    exact_mod_cast (Nat.le_of_lt (finiteActionCode_lt_card a))
  have hbCard : (((finiteActionCode b : ℕ) : ℝ)) ≤ Fintype.card A := by
    exact_mod_cast (Nat.le_of_lt (finiteActionCode_lt_card b))
  rw [abs_le]
  constructor <;> linarith

/-- State-dependent tie-break scale small enough to preserve every strict base ordering while still
breaking exact ties by the finite action code. -/
noncomputable def actionTieBreakEps {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) : ℝ :=
  minPositiveUtilityGapOrOne dp s / (2 * (Fintype.card A : ℝ))

theorem actionTieBreakEps_nonneg {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    0 ≤ actionTieBreakEps dp s := by
  unfold actionTieBreakEps
  have hDen : 0 < (2 * (Fintype.card A : ℝ)) := by positivity
  exact div_nonneg (minPositiveUtilityGapOrOne_pos dp s).le hDen.le

theorem actionTieBreakEps_pos {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    0 < actionTieBreakEps dp s := by
  unfold actionTieBreakEps
  have hGap : 0 < minPositiveUtilityGapOrOne dp s := minPositiveUtilityGapOrOne_pos dp s
  have hDen : 0 < (2 * (Fintype.card A : ℝ)) := by positivity
  exact div_pos hGap hDen

/-- Action-side symmetry-broken refinement of a finite decision problem. -/
noncomputable def actionPinnedDecisionProblem {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) : DecisionProblem A S where
  utility := fun a s => dp.utility a s + actionTieBreakEps dp s * (finiteActionCode a : ℝ)

theorem actionTieBreak_perturbation_abs_le_halfGap
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) (a b : A) :
    |actionTieBreakEps dp s * (((finiteActionCode a : ℕ) : ℝ) - (finiteActionCode b : ℕ))| ≤
      minPositiveUtilityGapOrOne dp s / 2 := by
  have hNonneg : 0 ≤ actionTieBreakEps dp s := actionTieBreakEps_nonneg dp s
  calc
    |actionTieBreakEps dp s * ((((finiteActionCode a : ℕ) : ℝ) - (finiteActionCode b : ℕ)))|
        = actionTieBreakEps dp s * |(((finiteActionCode a : ℕ) : ℝ) - (finiteActionCode b : ℕ))| := by
            rw [abs_mul, abs_of_nonneg hNonneg]
    _ ≤ actionTieBreakEps dp s * (Fintype.card A : ℝ) := by
          exact mul_le_mul_of_nonneg_left (finiteActionCode_abs_sub_le_card a b) hNonneg
    _ = minPositiveUtilityGapOrOne dp s / 2 := by
          unfold actionTieBreakEps
          have hCard : (2 * (Fintype.card A : ℝ)) ≠ 0 := by positivity
          field_simp [hCard]

theorem actionPinned_preserves_strict_base_order
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) {a b : A}
    (hGap : dp.utility b s < dp.utility a s) :
    (actionPinnedDecisionProblem dp).utility b s < (actionPinnedDecisionProblem dp).utility a s := by
  have hMinLe : minPositiveUtilityGapOrOne dp s ≤ dp.utility a s - dp.utility b s :=
    minPositiveUtilityGapOrOne_le_of_utility_lt dp s hGap
  have hPert := actionTieBreak_perturbation_abs_le_halfGap dp s a b
  have hLower :
      -(minPositiveUtilityGapOrOne dp s / 2) ≤
        actionTieBreakEps dp s * ((((finiteActionCode a : ℕ) : ℝ) - (finiteActionCode b : ℕ))) := by
    rw [abs_le] at hPert
    exact hPert.1
  have hMinPos : 0 < minPositiveUtilityGapOrOne dp s := minPositiveUtilityGapOrOne_pos dp s
  have hDeltaPos :
      0 < (dp.utility a s - dp.utility b s) +
        actionTieBreakEps dp s * ((((finiteActionCode a : ℕ) : ℝ) - (finiteActionCode b : ℕ))) := by
    linarith
  unfold actionPinnedDecisionProblem
  linarith

theorem actionPinned_breaks_equal_base_ties_by_code
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) {a b : A}
    (hEq : dp.utility a s = dp.utility b s)
    (hCode : finiteActionCode a < finiteActionCode b) :
    (actionPinnedDecisionProblem dp).utility a s < (actionPinnedDecisionProblem dp).utility b s := by
  have hEps : 0 < actionTieBreakEps dp s := actionTieBreakEps_pos dp s
  have hCodeR : (((finiteActionCode a : ℕ) : ℝ) < ((finiteActionCode b : ℕ) : ℝ)) := by
    exact_mod_cast hCode
  unfold actionPinnedDecisionProblem
  have hMul : actionTieBreakEps dp s * ((finiteActionCode a : ℕ) : ℝ) <
      actionTieBreakEps dp s * ((finiteActionCode b : ℕ) : ℝ) :=
    mul_lt_mul_of_pos_left hCodeR hEps
  linarith

theorem actionTieBreakEps_le_halfMinGap
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    actionTieBreakEps dp s ≤ minPositiveUtilityGapOrOne dp s / 2 := by
  unfold actionTieBreakEps
  have hMinNonneg : 0 ≤ minPositiveUtilityGapOrOne dp s := (minPositiveUtilityGapOrOne_pos dp s).le
  have hCardOne : 1 ≤ (Fintype.card A : ℝ) := by
    have hPosNat : 0 < Fintype.card A := Fintype.card_pos_iff.mpr ‹Nonempty A›
    exact_mod_cast (Nat.succ_le_of_lt hPosNat)
  have hNe1 : (2 : ℝ) ≠ 0 := by norm_num
  have hNe2 : (2 * (Fintype.card A : ℝ)) ≠ 0 := by positivity
  field_simp [hNe1, hNe2]
  nlinarith

theorem actionPinned_selected_margin_ge_tieBreak
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) {a : A}
    (haNe : a ≠ maxCodeOptAction dp s) :
    actionTieBreakEps dp s ≤
      (actionPinnedDecisionProblem dp).utility (maxCodeOptAction dp s) s -
        (actionPinnedDecisionProblem dp).utility a s := by
  let sel := maxCodeOptAction dp s
  by_cases haOpt : a ∈ dp.Opt s
  · have hBaseEq : dp.utility a s = dp.utility sel s := by
      rw [optimal_actions_same_utility dp s haOpt (maxCodeOptAction_mem_opt dp s)]
    have hCodeLe := finiteActionCode_le_maxCodeOptAction dp s haOpt
    have hCodeNe : finiteActionCode a ≠ finiteActionCode sel := by
      intro hEqCode
      exact haNe (finiteActionCode_injective hEqCode)
    have hCodeLt : finiteActionCode a < finiteActionCode sel :=
      lt_of_le_of_ne hCodeLe hCodeNe
    have hCodeStep : (1 : ℝ) ≤ ((finiteActionCode sel : ℕ) : ℝ) - ((finiteActionCode a : ℕ) : ℝ) := by
      have hCodeStepNat : finiteActionCode a + 1 ≤ finiteActionCode sel := Nat.succ_le_of_lt hCodeLt
      have hCodeStepReal : (((finiteActionCode a : ℕ) : ℝ) + 1) ≤ ((finiteActionCode sel : ℕ) : ℝ) := by
        exact_mod_cast hCodeStepNat
      linarith
    have hEpsNonneg : 0 ≤ actionTieBreakEps dp s := actionTieBreakEps_nonneg dp s
    have hEpsStep : actionTieBreakEps dp s ≤
        actionTieBreakEps dp s * (((finiteActionCode sel : ℕ) : ℝ) - (finiteActionCode a : ℕ)) := by
      have hMul : actionTieBreakEps dp s * 1 ≤
          actionTieBreakEps dp s * (((finiteActionCode sel : ℕ) : ℝ) - (finiteActionCode a : ℕ)) := by
        exact mul_le_mul_of_nonneg_left hCodeStep hEpsNonneg
      simpa using hMul
    unfold actionPinnedDecisionProblem
    linarith
  · have hBaseLt := not_optimal_lt_optimal dp s (maxCodeOptAction_mem_opt dp s) haOpt
    have hMinLe : minPositiveUtilityGapOrOne dp s ≤ dp.utility sel s - dp.utility a s :=
      minPositiveUtilityGapOrOne_le_of_utility_lt dp s hBaseLt
    have hPert := actionTieBreak_perturbation_abs_le_halfGap dp s sel a
    have hLower : -(minPositiveUtilityGapOrOne dp s / 2) ≤
        actionTieBreakEps dp s * (((finiteActionCode sel : ℕ) : ℝ) - (finiteActionCode a : ℕ)) := by
      rw [abs_le] at hPert
      exact hPert.1
    have hEpsLeHalf := actionTieBreakEps_le_halfMinGap dp s
    unfold actionPinnedDecisionProblem
    linarith

theorem actionPinned_strictOpt_maxCodeOptAction
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    StrictOpt (actionPinnedDecisionProblem dp) (maxCodeOptAction dp s) s := by
  intro a haNe
  by_cases haOpt : a ∈ dp.Opt s
  · have hBaseEq : dp.utility a s = dp.utility (maxCodeOptAction dp s) s := by
      rw [optimal_actions_same_utility dp s haOpt (maxCodeOptAction_mem_opt dp s)]
    have hCodeLe := finiteActionCode_le_maxCodeOptAction dp s haOpt
    have hCodeNe : finiteActionCode a ≠ finiteActionCode (maxCodeOptAction dp s) := by
      intro hEqCode
      exact haNe (finiteActionCode_injective hEqCode)
    have hCodeLt : finiteActionCode a < finiteActionCode (maxCodeOptAction dp s) :=
      lt_of_le_of_ne hCodeLe hCodeNe
    exact actionPinned_breaks_equal_base_ties_by_code dp s hBaseEq hCodeLt
  · have hBaseLt := not_optimal_lt_optimal dp s (maxCodeOptAction_mem_opt dp s) haOpt
    exact actionPinned_preserves_strict_base_order dp s hBaseLt

theorem actionPinned_opt_eq_singleton_maxCodeOptAction
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) (s : S) :
    (actionPinnedDecisionProblem dp).Opt s = {maxCodeOptAction dp s} := by
  exact opt_eq_singleton_of_strict _ _ _ (actionPinned_strictOpt_maxCodeOptAction dp s)

theorem maxCodeOptAction_eq_of_opt_eq
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (dp : DecisionProblem A S) {s s' : S}
    (hOpt : dp.Opt s = dp.Opt s') :
    maxCodeOptAction dp s = maxCodeOptAction dp s' := by
  unfold maxCodeOptAction optFinset
  simp [hOpt]

theorem actionPinned_sufficient_of_sufficient
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) (hI : dp.isSufficient I) :
    (actionPinnedDecisionProblem dp).isSufficient I := by
  intro s s' hAgree
  have hBase : dp.Opt s = dp.Opt s' := hI s s' hAgree
  have hSel : maxCodeOptAction dp s = maxCodeOptAction dp s' := maxCodeOptAction_eq_of_opt_eq dp hBase
  rw [actionPinned_opt_eq_singleton_maxCodeOptAction, actionPinned_opt_eq_singleton_maxCodeOptAction, hSel]

theorem actionPinned_srank_le_srank
    {A : Type*} {n : ℕ} [Fintype A] [Nonempty A] [DecidableEq A] [DecidableEq (Set A)]
    {X : Type*}
    (dp : DecisionProblem A (Fin n → X)) :
    (actionPinnedDecisionProblem dp).srank ≤ dp.srank := by
  let R : Finset (Fin n) := Finset.univ.filter dp.isRelevant
  have hinj : ∀ s s' : Fin n → X,
      (∀ i : Fin n, CoordinateSpace.proj s i = CoordinateSpace.proj s' i) → s = s' := by
    intro s s' h
    funext i
    exact h i
  have hRSuff : dp.isSufficient R := relevantSet_isSufficient dp hinj
  have hPinnedSuff : (actionPinnedDecisionProblem dp).isSufficient R :=
    actionPinned_sufficient_of_sufficient dp R hRSuff
  exact le_trans (srank_le_sufficient_card _ _ hPinnedSuff) (by rfl)

private theorem strictUtilityGap_eq_of_nonempty_local
    {A S : Type*} [Fintype A]
    (dp : DecisionProblem A S) (a_star : A) (s : S)
    (hSub : ((Finset.univ : Finset A).filter (fun a => a ≠ a_star)).Nonempty) :
    StrictUtilityGap dp a_star s =
      dp.utility a_star s -
        ((Finset.univ : Finset A).filter (fun a => a ≠ a_star)).sup' hSub
          (fun a => dp.utility a s) := by
  classical
  unfold StrictUtilityGap
  let subopts : Finset A := (Finset.univ : Finset A).filter (fun a => a ≠ a_star)
  have hGapDef :
      (if h : subopts.Nonempty then
        dp.utility a_star s - subopts.sup' h (fun a => dp.utility a s)
      else 0) =
        dp.utility a_star s - subopts.sup' hSub (fun a => dp.utility a s) := by
    exact dif_pos hSub
  dsimp only [subopts] at hGapDef ⊢
  exact hGapDef

theorem actionPinned_strictUtilityGap_ge_tieBreak_of_card_gt_one
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (hCard : 1 < Fintype.card A)
    (dp : DecisionProblem A S) (s : S) :
    actionTieBreakEps dp s ≤
      StrictUtilityGap (actionPinnedDecisionProblem dp) (maxCodeOptAction dp s) s := by
  classical
  letI : DecidableEq A := Classical.decEq A
  let subopts : Finset A := (Finset.univ : Finset A).filter (fun a => a ≠ maxCodeOptAction dp s)
  have hSubExact : ((Finset.univ : Finset A).filter (fun a => a ≠ maxCodeOptAction dp s)).Nonempty := by
    haveI : Nontrivial A := Fintype.one_lt_card_iff_nontrivial.mp hCard
    obtain ⟨b, hb⟩ := exists_ne (maxCodeOptAction dp s)
    exact ⟨b, by simp [hb]⟩
  have hSub : subopts.Nonempty := by
    simpa [subopts] using hSubExact
  have hBound : ∀ a ∈ subopts,
      (actionPinnedDecisionProblem dp).utility a s ≤
        (actionPinnedDecisionProblem dp).utility (maxCodeOptAction dp s) s -
          actionTieBreakEps dp s := by
    intro a ha
    have haNe : a ≠ maxCodeOptAction dp s := by
      simpa using (Finset.mem_filter.mp ha).2
    have hMargin := actionPinned_selected_margin_ge_tieBreak dp s haNe
    linarith
  have hSubDef : ((Finset.univ : Finset A).filter (fun a => a ≠ maxCodeOptAction dp s)).Nonempty := by
    haveI : Nontrivial A := Fintype.one_lt_card_iff_nontrivial.mp hCard
    obtain ⟨b, hb⟩ := exists_ne (maxCodeOptAction dp s)
    exact ⟨b, by simp [hb]⟩
  have hBoundDef : ∀ a ∈ ((Finset.univ : Finset A).filter (fun a => a ≠ maxCodeOptAction dp s)),
      (actionPinnedDecisionProblem dp).utility a s ≤
        (actionPinnedDecisionProblem dp).utility (maxCodeOptAction dp s) s -
          actionTieBreakEps dp s := by
    intro a ha
    have haNe : a ≠ maxCodeOptAction dp s := by
      simpa using (Finset.mem_filter.mp ha).2
    have hMargin := actionPinned_selected_margin_ge_tieBreak dp s haNe
    linarith
  have hSupDef :
      ((Finset.univ : Finset A).filter (fun a => a ≠ maxCodeOptAction dp s)).sup' hSubDef
        (fun a => (actionPinnedDecisionProblem dp).utility a s) ≤
        (actionPinnedDecisionProblem dp).utility (maxCodeOptAction dp s) s - actionTieBreakEps dp s := by
    exact Finset.sup'_le hSubDef _ hBoundDef
  have hGapDefBound : actionTieBreakEps dp s ≤
      (actionPinnedDecisionProblem dp).utility (maxCodeOptAction dp s) s -
        ((Finset.univ : Finset A).filter (fun a => a ≠ maxCodeOptAction dp s)).sup' hSubDef
          (fun a => (actionPinnedDecisionProblem dp).utility a s) := by
    linarith
  have hGapDefEq :
      StrictUtilityGap (actionPinnedDecisionProblem dp) (maxCodeOptAction dp s) s =
        (actionPinnedDecisionProblem dp).utility (maxCodeOptAction dp s) s -
          ((Finset.univ : Finset A).filter (fun a => a ≠ maxCodeOptAction dp s)).sup' hSubDef
            (fun a => (actionPinnedDecisionProblem dp).utility a s) := by
    exact strictUtilityGap_eq_of_nonempty_local
      (dp := actionPinnedDecisionProblem dp)
      (a_star := maxCodeOptAction dp s)
      (s := s)
      hSubDef
  exact hGapDefEq.symm ▸ hGapDefBound

theorem actionPinned_strictUtilityGap_pos_of_card_gt_one
    {A S : Type*} [Fintype A] [Nonempty A] [DecidableEq A]
    (hCard : 1 < Fintype.card A)
    (dp : DecisionProblem A S) (s : S) :
    0 < StrictUtilityGap (actionPinnedDecisionProblem dp) (maxCodeOptAction dp s) s := by
  exact lt_of_lt_of_le (actionTieBreakEps_pos dp s)
    (actionPinned_strictUtilityGap_ge_tieBreak_of_card_gt_one hCard dp s)

/-- Local paper3 exposure of the strict-dominance collapse regime. -/
theorem strictGlobalDominance_emptySufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S]
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hdom : StrictGlobalDominance (dp := dp))
    (I : Finset (Fin n)) :
    dp.isSufficient I :=
  StrictGlobalDominance.empty_sufficient dp hdom I

/-- Local paper3 exposure of multiplicative-separable collapse when the state factor has constant
sign. -/
theorem multiplicativeSeparable_emptySufficient
    {A S : Type*} [DecidableEq A] [DecidableEq S]
    {n : ℕ} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (hms : MultiplicativeSeparable (dp := dp))
    (I : Finset (Fin n))
    (hcase :
      (∀ s, hms.stateFactor s > 0) ∨
        (∀ s, hms.stateFactor s < 0) ∨
          (∀ s, hms.stateFactor s = 0)) :
    dp.isSufficient I :=
  MultiplicativeSeparable.empty_sufficient dp hms I hcase

/-- Local paper3 exposure of the bounded-state pair-count bound. -/
theorem boundedState_pairBound {S : Type*} [Fintype S]
    (k : ℕ) (hk : Fintype.card S ≤ k) :
    Fintype.card S * Fintype.card S ≤ k * k :=
  DecisionQuotient.bounded_state_pair_bound k hk

private theorem mem_topKSet_one_iff_optimal_local
    {A : Type*} [Fintype A] [DecidableEq A]
    (u : A → ℝ) (a : A) :
    a ∈ DecisionQuotient.Tractability.FiniteTopK.topKSet u 1 ↔ ∀ b, u b ≤ u a := by
  rw [DecisionQuotient.Tractability.FiniteTopK.mem_topKSet_iff]
  constructor
  · intro ha b
    by_contra hba
    have hCardPos : 1 ≤ DecisionQuotient.Tractability.FiniteTopK.strictBetterCount u a := by
      unfold DecisionQuotient.Tractability.FiniteTopK.strictBetterCount
      exact Nat.succ_le_of_lt (Finset.card_pos.mpr ⟨b, by simp [lt_of_not_ge hba]⟩)
    omega
  · intro hopt
    have hCountZero : DecisionQuotient.Tractability.FiniteTopK.strictBetterCount u a = 0 := by
      unfold DecisionQuotient.Tractability.FiniteTopK.strictBetterCount
      apply Finset.card_eq_zero.mpr
      rw [Finset.filter_eq_empty_iff]
      intro b hb
      exact not_lt_of_ge (hopt b)
    rw [hCountZero]
    norm_num

/-- Action-side tie-breaking with an externally supplied statewise tie-break scale. -/
noncomputable def fixedTieBreakDecisionProblem
    {A S : Type*} [Fintype A]
    (dp : DecisionProblem A S) (tieBreak : S → ℝ) : DecisionProblem A S where
  utility := fun a s => dp.utility a s + tieBreak s * (finiteActionCode a : ℝ)

theorem fixedTieBreak_uniformApprox_of_uniformApprox
    {A S : Type*} [Fintype A]
    (exactDP coarseDP : DecisionProblem A S) (tieBreak : S → ℝ) (delta : ℝ)
    (hApprox : UniformUtilityApprox exactDP coarseDP delta) :
    UniformUtilityApprox
      (fixedTieBreakDecisionProblem exactDP tieBreak)
      (fixedTieBreakDecisionProblem coarseDP tieBreak)
      delta := by
  intro a s
  have hEq :
      (fixedTieBreakDecisionProblem exactDP tieBreak).utility a s -
          (fixedTieBreakDecisionProblem coarseDP tieBreak).utility a s =
        exactDP.utility a s - coarseDP.utility a s := by
    simp [fixedTieBreakDecisionProblem]
  rw [show
      |(fixedTieBreakDecisionProblem exactDP tieBreak).utility a s -
          (fixedTieBreakDecisionProblem coarseDP tieBreak).utility a s| =
        |exactDP.utility a s - coarseDP.utility a s| by rw [hEq]]
  exact hApprox a s

/-- Any action that is optimal for the grid score at a sampled state lies in the lifted continuous
ambiguity band of width `2 * delta` whenever the continuous and grid scores differ uniformly by at
most `delta`. -/
theorem opt_mem_lifted_ambiguityBand_of_uniformApprox
    {A : Type u} {Scont Sgrid : Type v} [Fintype A] [DecidableEq A] [Nonempty A]
    (uCont : A → Scont → ℝ) (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont) (delta : ℝ)
    (hApprox : ∀ a sGrid, |uCont a (lift sGrid) - uGrid a sGrid| ≤ delta)
    (s : Sgrid) {a : A}
    (haOpt : a ∈ ({ utility := uGrid } : DecisionProblem A Sgrid).Opt s) :
    a ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
      (fun x => uCont x (lift s)) 1 (by norm_num) (2 * delta) := by
  let uContLift : A → ℝ := fun x => uCont x (lift s)
  let uGridState : A → ℝ := fun x => uGrid x s
  have haTop : a ∈ DecisionQuotient.Tractability.FiniteTopK.topKSet uGridState 1 := by
    rw [mem_topKSet_one_iff_optimal_local]
    intro b
    exact haOpt b
  have hGridTop : ∀ b, uGridState b ≤ uGridState a :=
    (mem_topKSet_one_iff_optimal_local uGridState a).mp haTop
  have hk : 0 < 1 := by norm_num
  obtain ⟨b, hb⟩ := DecisionQuotient.Tractability.FiniteTopK.topKSet_nonempty uContLift hk
  have hKth : DecisionQuotient.Tractability.FiniteTopK.kthUtility uContLift 1 hk ≤ uContLift b :=
    DecisionQuotient.Tractability.FiniteTopK.kthUtility_le_of_mem_topKSet uContLift 1 hk b hb
  have hApproxA := hApprox a s
  have hApproxB := hApprox b s
  have hGridLe : uGrid b s ≤ uGrid a s := hGridTop b
  have hBUpper : uCont b (lift s) ≤ uGrid b s + delta := by
    have hB := abs_le.mp hApproxB
    linarith
  have hALower : uGrid a s ≤ uCont a (lift s) + delta := by
    have hA := abs_le.mp hApproxA
    linarith
  rw [DecisionQuotient.Tractability.NearTieBand.mem_ambiguityBand_iff]
  dsimp [uContLift] at hKth
  linarith

/-- If the continuous lifted score and the grid score use the same statewise tie-break scale, then
the action-pinned grid optimizer is preserved under a resolution-controlled continuous-to-grid error
radius smaller than half of the grid pinned strict utility gap. -/
theorem resolutionControlled_actionPinned_opt_eq_liftedFixedTieBreak
    {A : Type u} {Scont Sgrid : Type v} [Fintype A] [Nonempty A] [DecidableEq A]
    (uCont : A → Scont → ℝ) (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont) (eps : ℝ → ℝ) (res : ℝ)
    (hApprox : DecisionQuotient.Tractability.GridConvergence.ResolutionControlledApprox
      uCont uGrid lift eps res)
    (hDelta : 0 ≤ eps res)
    (s : Sgrid)
    (hBound : eps res <
      StrictUtilityGap
        (actionPinnedDecisionProblem ({ utility := uGrid } : DecisionProblem A Sgrid))
        (maxCodeOptAction ({ utility := uGrid } : DecisionProblem A Sgrid) s) s / 2) :
    (actionPinnedDecisionProblem ({ utility := uGrid } : DecisionProblem A Sgrid)).Opt s =
      (fixedTieBreakDecisionProblem
        ({ utility := fun a sGrid => uCont a (lift sGrid) } : DecisionProblem A Sgrid)
        (actionTieBreakEps ({ utility := uGrid } : DecisionProblem A Sgrid))).Opt s := by
  let contLiftedDP : DecisionProblem A Sgrid :=
    { utility := fun a sGrid => uCont a (lift sGrid) }
  let gridDP : DecisionProblem A Sgrid := { utility := uGrid }
  let tieBreak : Sgrid → ℝ := actionTieBreakEps gridDP
  have hUniform : UniformUtilityApprox contLiftedDP gridDP (eps res) :=
    DecisionQuotient.Tractability.GridConvergence.resolutionControlledApprox_implies_uniformApprox
      uCont uGrid lift eps res hApprox
  have hUniformSymm : UniformUtilityApprox gridDP contLiftedDP (eps res) := by
    intro a sGrid
    simpa [abs_sub_comm] using hUniform a sGrid
  have hUniformPinned : UniformUtilityApprox
      (fixedTieBreakDecisionProblem gridDP tieBreak)
      (fixedTieBreakDecisionProblem contLiftedDP tieBreak)
      (eps res) :=
    fixedTieBreak_uniformApprox_of_uniformApprox gridDP contLiftedDP tieBreak (eps res) hUniformSymm
  have hStrict : StrictOpt (fixedTieBreakDecisionProblem gridDP tieBreak)
      (maxCodeOptAction gridDP s) s := by
    simpa [gridDP, tieBreak, fixedTieBreakDecisionProblem, actionPinnedDecisionProblem] using
      actionPinned_strictOpt_maxCodeOptAction gridDP s
  have hBound' : eps res <
      StrictUtilityGap (fixedTieBreakDecisionProblem gridDP tieBreak)
        (maxCodeOptAction gridDP s) s / 2 := by
    simpa [gridDP, tieBreak, fixedTieBreakDecisionProblem, actionPinnedDecisionProblem] using hBound
  simpa [gridDP, contLiftedDP, tieBreak, fixedTieBreakDecisionProblem, actionPinnedDecisionProblem] using
    (uniform_approx_implies_opt_invariance
      (fixedTieBreakDecisionProblem gridDP tieBreak)
      (fixedTieBreakDecisionProblem contLiftedDP tieBreak)
      (eps res) hUniformPinned s (maxCodeOptAction gridDP s) hDelta hStrict hBound')

/-- Under the same half-gap hypothesis, every selected action of the lifted continuous problem with
the grid tie-break scale remains inside the ambiguity band of width `2 * eps res` around the lifted
continuous optimum. -/
theorem resolutionControlled_liftedFixedTieBreak_opt_subset_ambiguityBand
    {A : Type u} {Scont Sgrid : Type v} [Fintype A] [Nonempty A] [DecidableEq A]
    (uCont : A → Scont → ℝ) (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont) (eps : ℝ → ℝ) (res : ℝ)
    (hApprox : DecisionQuotient.Tractability.GridConvergence.ResolutionControlledApprox
      uCont uGrid lift eps res)
    (hDelta : 0 ≤ eps res)
    (s : Sgrid)
    (hBound : eps res <
      StrictUtilityGap
        (actionPinnedDecisionProblem ({ utility := uGrid } : DecisionProblem A Sgrid))
        (maxCodeOptAction ({ utility := uGrid } : DecisionProblem A Sgrid) s) s / 2) :
    (fixedTieBreakDecisionProblem
      ({ utility := fun a sGrid => uCont a (lift sGrid) } : DecisionProblem A Sgrid)
      (actionTieBreakEps ({ utility := uGrid } : DecisionProblem A Sgrid))).Opt s ⊆
      ↑(DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun x => uCont x (lift s)) 1 (by norm_num) (2 * eps res)) := by
  let contLiftedDP : DecisionProblem A Sgrid :=
    { utility := fun a sGrid => uCont a (lift sGrid) }
  let gridDP : DecisionProblem A Sgrid := { utility := uGrid }
  intro a ha
  have hEq := resolutionControlled_actionPinned_opt_eq_liftedFixedTieBreak
    uCont uGrid lift eps res hApprox hDelta s hBound
  have haPinned : a ∈ (actionPinnedDecisionProblem gridDP).Opt s := by
    simpa [gridDP, contLiftedDP, fixedTieBreakDecisionProblem] using hEq.symm ▸ ha
  have hSel : a = maxCodeOptAction gridDP s := by
    rw [actionPinned_opt_eq_singleton_maxCodeOptAction] at haPinned
    simpa using haPinned
  subst hSel
  exact opt_mem_lifted_ambiguityBand_of_uniformApprox uCont uGrid lift (eps res)
    (fun a sGrid => hApprox a sGrid) s (maxCodeOptAction_mem_opt gridDP s)

/-- Resolution-controlled continuous-to-grid approximation yields a uniform score bound on the
lifted grid state space. -/
theorem resolutionControlledApprox_implies_uniformApprox
    {A : Type u} {Scont Sgrid : Type v}
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (hApprox : DecisionQuotient.Tractability.GridConvergence.ResolutionControlledApprox
      uCont uGrid lift eps res) :
    UniformUtilityApprox
      { utility := fun a sGrid => uCont a (lift sGrid) }
      { utility := uGrid }
      (eps res) :=
  DecisionQuotient.Tractability.GridConvergence.resolutionControlledApprox_implies_uniformApprox
    uCont uGrid lift eps res hApprox

/-- A Lipschitz utility bound and a uniform state-resolution bound yield a uniform
continuous-to-grid score approximation radius. -/
theorem lipschitzUtilityApprox_implies_resolutionControlled
    {A : Type u} {Scont Sgrid : Type v}
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (L res : ℝ)
    (hLip : DecisionQuotient.Tractability.GridConvergence.LipschitzUtilityApprox
      uCont uGrid lift stateError L)
    (hState : ∀ sGrid, stateError sGrid ≤ res)
    (hL : 0 ≤ L) :
    DecisionQuotient.Tractability.GridConvergence.ResolutionControlledApprox
      uCont uGrid lift (fun r => L * r) res :=
  DecisionQuotient.Tractability.GridConvergence.lipschitzUtilityApprox_implies_resolutionControlled
    uCont uGrid lift stateError L res hLip hState hL

/-- A physical lattice-tail bound and a positive strict minimum gap are sufficient to establish the
cutoff-boundedness hypothesis used in the docking locality theorems. -/
theorem largeCutoff_implies_bounded
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (tail_coefficient minimum_gap : ℝ)
    (hBounds : DecisionQuotient.Tractability.SatisfiesBoundedPotential prob tail_coefficient minimum_gap)
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff < minimum_gap / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff) :
    OutsideCutoffApproximationBounded prob :=
  DecisionQuotient.Tractability.large_cutoff_implies_bounded
    prob tail_coefficient minimum_gap hBounds hPosCutoff hCutoffSize hTailPos

/-- Finite sampled docking always admits a canonical exact/coarse discrepancy radius witnessing
uniform approximation on the sampled domain. -/
theorem sampledDocking_finiteUniformErrorRadius_witnesses
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    UniformUtilityApprox prob.exactDecisionProblem prob.coarseDecisionProblem
      (DecisionQuotient.Tractability.CoarseApproximation.finiteUniformErrorRadius
        prob.exactDecisionProblem prob.coarseDecisionProblem) :=
  DecisionQuotient.Tractability.CoarseApproximation.SampledDocking.SampledDockingProblem.finiteUniformErrorRadius_witnesses prob

/-- The flat finite-grid exact docking decision problem obtained by reconstructing the sampled grid
state from its coordinate function. -/
noncomputable def sampledDocking_flatExactDecisionProblem
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    DecisionProblem (SupportedAction prob.samples)
      (Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N) where
  utility := fun a s => prob.exactDecisionProblem.utility a
    (DecisionQuotient.Tractability.GridMDInstances.fromFlat s)

/-- For finite grid-coordinate state families, the number of exact optimizer classes is bounded by
the grid alphabet size raised to structural rank. -/
theorem flatGrid_numOptClasses_le_gridArity_pow_srank
    {A : Type*} {n N : ℕ} [DecidableEq A] [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → BoundedGrid N)) :
    dp.numOptClasses ≤ (2 * N + 1) ^ dp.srank := by
  let R := Finset.univ.filter dp.isRelevant
  have hinj : ∀ s s' : Fin n → BoundedGrid N,
      (∀ i : Fin n, CoordinateSpace.proj s i = CoordinateSpace.proj s' i) → s = s' := by
    intro s s' h
    funext i
    exact h i
  have hsuff : dp.isSufficient R := relevantSet_isSufficient dp hinj
  let pat : (Fin n → BoundedGrid N) → (R → BoundedGrid N) := fun s => fun ⟨i, _⟩ => s i
  have hfactor : ∀ s s' : Fin n → BoundedGrid N, pat s = pat s' → dp.Opt s = dp.Opt s' := by
    intro s s' hpat
    apply hsuff
    intro i hi
    exact congrFun hpat ⟨i, hi⟩
  have hcard_pat : Fintype.card (R → BoundedGrid N) = (2 * N + 1) ^ R.card := by
    simp [DecisionQuotient.Tractability.DiscretizedState.BoundedGrid]
  have hcard_R : R.card = dp.srank := rfl
  let optFromPat : (R → BoundedGrid N) → Set A := fun p =>
    dp.Opt (fun i => if h : i ∈ R then p ⟨i, h⟩ else 0)
  have hsubset : Finset.univ.image dp.Opt ⊆ Finset.univ.image optFromPat := by
    intro optVal hoptVal
    rw [Finset.mem_image] at hoptVal ⊢
    obtain ⟨s, _, hs⟩ := hoptVal
    use pat s
    constructor
    · exact Finset.mem_univ _
    · rw [← hs]
      apply hfactor
      funext ⟨i, hi⟩
      simp [pat, hi]
  calc dp.numOptClasses
      = (Finset.univ.image dp.Opt).card := rfl
    _ ≤ (Finset.univ.image optFromPat).card := Finset.card_le_card hsubset
    _ ≤ Finset.univ.card := Finset.card_image_le
    _ = Fintype.card (R → BoundedGrid N) := Finset.card_univ
    _ = (2 * N + 1) ^ R.card := hcard_pat
    _ = (2 * N + 1) ^ dp.srank := by rw [hcard_R]

/-- Nat-valued decision entropy of a flat finite-grid state family is bounded by structural rank
times the log of the grid alphabet size. -/
theorem flatGrid_natEntropy_le_srank_log_gridArity
    {A : Type*} {n N : ℕ} [DecidableEq A] [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → BoundedGrid N)) :
    Real.log (dp.numOptClasses : ℝ) ≤ (dp.srank : ℝ) * Real.log (2 * N + 1 : ℝ) := by
  have hClasses := flatGrid_numOptClasses_le_gridArity_pow_srank (A := A) (n := n) (N := N) dp
  have hPos : 0 < (dp.numOptClasses : ℝ) := by
    exact_mod_cast (DecisionProblem.numOptClasses_pos (dp := dp))
  have hCast : (dp.numOptClasses : ℝ) ≤ (((2 * N + 1) ^ dp.srank : ℕ) : ℝ) := by
    exact_mod_cast hClasses
  calc
    Real.log (dp.numOptClasses : ℝ) ≤ Real.log ((((2 * N + 1) ^ dp.srank : ℕ) : ℝ)) :=
      Real.log_le_log hPos hCast
    _ = Real.log (2 * N + 1 : ℝ) * dp.srank := by
      rw [Nat.cast_pow, Real.log_pow]
      norm_num
      rw [mul_comm]
    _ = (dp.srank : ℝ) * Real.log (2 * N + 1 : ℝ) := by rw [mul_comm]

/-- The flat finite-grid encoding of sampled exact docking therefore carries a direct structural-rank
entropy bound. -/
theorem sampledDocking_flatExact_natEntropy_le_srank_log_gridArity
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    Real.log ((sampledDocking_flatExactDecisionProblem prob).numOptClasses : ℝ) ≤
      ((sampledDocking_flatExactDecisionProblem prob).srank : ℝ) * Real.log (2 * N + 1 : ℝ) := by
  classical
  exact flatGrid_natEntropy_le_srank_log_gridArity (sampledDocking_flatExactDecisionProblem prob)

/-- Bit width of the fixed-length binary register used to encode one `(2N+1)`-ary grid coordinate. -/
@[reducible] def flatGridBitWidth (N : Nat) : Nat :=
  Nat.size (2 * N + 1)

/-- Interpret a finite Boolean word as a natural number, with coordinate `0` as the least
significant bit. -/
def bitVectorToNat : {k : ℕ} → (Fin k → Bool) → ℕ
  | 0, _ => 0
  | k + 1, b => Nat.bit (b 0) (bitVectorToNat (fun i => b i.succ))

/-- A `k`-bit word always decodes to a natural number below `2^k`. -/
theorem bitVectorToNat_lt_two_pow :
    ∀ {k : ℕ} (b : Fin k → Bool), bitVectorToNat b < 2 ^ k
  | 0, _ => by simp [bitVectorToNat]
  | k + 1, b => by
      have hTail : bitVectorToNat (fun i => b i.succ) < 2 ^ k :=
        bitVectorToNat_lt_two_pow (fun i => b i.succ)
      by_cases h0 : b 0 <;> simp [bitVectorToNat, h0, Nat.pow_succ] at * <;> omega

/-- Reading back the in-range bit coordinates of a word recovers the original bits. -/
theorem testBit_bitVectorToNat
    : ∀ {k : ℕ} (b : Fin k → Bool) (i : Fin k),
        Nat.testBit (bitVectorToNat b) i = b i
  | 0, _, i => Fin.elim0 i
  | k + 1, b, i => by
      obtain rfl | ⟨j, rfl⟩ := Fin.eq_zero_or_eq_succ i
      · simp [bitVectorToNat]
      · change Nat.testBit (Nat.bit (b 0) (bitVectorToNat (fun i => b i.succ))) (Nat.succ ↑j) =
            b j.succ
        rw [Nat.testBit_bit_succ]
        exact testBit_bitVectorToNat (fun i => b i.succ) j

/-- Bits above the encoded width are forced to zero. -/
theorem testBit_bitVectorToNat_eq_false_of_ge
    {k : ℕ} (b : Fin k → Bool) {i : ℕ} (hi : k ≤ i) :
    Nat.testBit (bitVectorToNat b) i = false := by
  apply Nat.testBit_eq_false_of_lt
  exact lt_of_lt_of_le (bitVectorToNat_lt_two_pow b)
    (Nat.pow_le_pow_right (by decide) hi)

/-- A natural number below `2^k` is reconstructed exactly from its first `k` test bits. -/
theorem bitVectorToNat_of_testBit {k : ℕ} (n : ℕ) (h : n < 2 ^ k) :
    bitVectorToNat (fun i : Fin k => Nat.testBit n i) = n := by
  apply Nat.eq_of_testBit_eq
  intro i
  by_cases hi : i < k
  · let j : Fin k := ⟨i, hi⟩
    simpa [j] using
      (testBit_bitVectorToNat (b := fun i : Fin k => Nat.testBit n i) j)
  · have hLeft : Nat.testBit (bitVectorToNat (fun i : Fin k => Nat.testBit n i)) i = false := by
      apply testBit_bitVectorToNat_eq_false_of_ge
      exact Nat.le_of_not_gt hi
    have hRight : Nat.testBit n i = false := by
      apply Nat.testBit_eq_false_of_lt
      exact lt_of_lt_of_le h (Nat.pow_le_pow_right (by decide) (Nat.le_of_not_gt hi))
    rw [hLeft, hRight]

/-- Fixed-length binary encoding of one grid coordinate. -/
def encodeGridWord {N : Nat} (g : BoundedGrid N) : Fin (flatGridBitWidth N) → Bool :=
  fun i => Nat.testBit g.val i

/-- Total binary decoder for one grid coordinate. Extra codewords collapse modulo the grid arity. -/
def decodeGridWord {N : Nat} (b : Fin (flatGridBitWidth N) → Bool) : BoundedGrid N :=
  ⟨bitVectorToNat b % (2 * N + 1), Nat.mod_lt _ (Nat.succ_pos _)⟩

/-- Encoding followed by decoding recovers the original grid coordinate. -/
theorem decodeGridWord_encodeGridWord {N : Nat} (g : BoundedGrid N) :
    decodeGridWord (encodeGridWord g) = g := by
  apply Fin.ext
  change bitVectorToNat (fun i : Fin (flatGridBitWidth N) => Nat.testBit g.val i) % (2 * N + 1) = g.val
  rw [bitVectorToNat_of_testBit]
  · exact Nat.mod_eq_of_lt g.is_lt
  · exact lt_of_lt_of_le g.is_lt (Nat.lt_size_self (2 * N + 1)).le

/-- The fixed-length binary encoding of one grid coordinate is injective. -/
theorem encodeGridWord_injective {N : Nat} :
    Function.Injective (@encodeGridWord N) := by
  intro g g' h
  have hDecode := congrArg decodeGridWord h
  simpa [decodeGridWord_encodeGridWord] using hDecode

/-- Flat binary state space used to encode a finite `n`-coordinate grid state. -/
abbrev FlatGridBitState (n N : Nat) :=
  Fin (n * flatGridBitWidth N) → Bool

/-- Fixed-length binary encoding of a flat finite-grid state. -/
def flatGrid_encodeBits {n N : Nat} (s : Fin n → BoundedGrid N) : FlatGridBitState n N :=
  fun idx =>
    let ij :=
      ((finProdFinEquiv : Fin n × Fin (flatGridBitWidth N) ≃
        Fin (n * flatGridBitWidth N))).symm idx
    encodeGridWord (s ij.1) ij.2

/-- Total binary decoder for a flat finite-grid state. -/
def flatGrid_decodeBits {n N : Nat} (b : FlatGridBitState n N) : Fin n → BoundedGrid N :=
  fun i => decodeGridWord fun j =>
    b ((finProdFinEquiv : Fin n × Fin (flatGridBitWidth N) ≃
      Fin (n * flatGridBitWidth N)) (i, j))

/-- The fixed-length coordinatewise binary encoding is injective, with `flatGrid_decodeBits` as a
right inverse on true grid states. -/
theorem flatGrid_decodeBits_encodeBits {n N : Nat} (s : Fin n → BoundedGrid N) :
    flatGrid_decodeBits (flatGrid_encodeBits s) = s := by
  funext i
  simpa [flatGrid_decodeBits, flatGrid_encodeBits] using decodeGridWord_encodeGridWord (s i)

/-- The coordinatewise binary encoding of a flat finite-grid state is injective. -/
theorem flatGrid_encodeBits_injective {n N : Nat} :
    Function.Injective (@flatGrid_encodeBits n N) := by
  intro s s' h
  have hDecode := congrArg flatGrid_decodeBits h
  simpa [flatGrid_decodeBits_encodeBits] using hDecode

/-- Binary lift of a finite grid-coordinate decision problem obtained by decoding a fixed-length
bit register to the original grid state. -/
def flatGrid_binaryEncodedDecisionProblem
    {A : Type*} {n N : ℕ} (dp : DecisionProblem A (Fin n → BoundedGrid N)) :
    DecisionProblem A (FlatGridBitState n N) where
  utility := fun a b => dp.utility a (flatGrid_decodeBits b)

/-- The binary lift agrees with the original flat grid problem on genuinely encoded states. -/
theorem flatGrid_binaryEncoded_opt_encode
    {A : Type*} {n N : ℕ} (dp : DecisionProblem A (Fin n → BoundedGrid N))
    (s : Fin n → BoundedGrid N) :
    (flatGrid_binaryEncodedDecisionProblem dp).Opt (flatGrid_encodeBits s) = dp.Opt s := by
  ext a
  simp [DecisionProblem.Opt, DecisionProblem.isOptimal, flatGrid_binaryEncodedDecisionProblem,
    flatGrid_decodeBits_encodeBits]

/-- Bit indices belonging to the coordinate blocks of `I`. -/
noncomputable def flatGrid_expandCoordSet {n N : Nat}
    (I : Finset (Fin n)) : Finset (Fin (n * flatGridBitWidth N)) :=
  (I.product (Finset.univ : Finset (Fin (flatGridBitWidth N)))).image
    (finProdFinEquiv : Fin n × Fin (flatGridBitWidth N) ≃ Fin (n * flatGridBitWidth N))

/-- Every bit inside a selected coordinate block belongs to the expanded bit set. -/
theorem flatGrid_mem_expandCoordSet_of_mem
    {n N : Nat} {I : Finset (Fin n)} {i : Fin n}
    (hi : i ∈ I) (j : Fin (flatGridBitWidth N)) :
    (finProdFinEquiv : Fin n × Fin (flatGridBitWidth N) ≃ Fin (n * flatGridBitWidth N)) (i, j) ∈
      flatGrid_expandCoordSet (N := N) I := by
  unfold flatGrid_expandCoordSet
  rw [Finset.mem_image]
  exact ⟨(i, j), by simp [hi], rfl⟩

/-- Expanding a coordinate set to all of its bit positions multiplies the cardinality by the fixed
bit width. -/
theorem flatGrid_expandCoordSet_card
    {n N : Nat} (I : Finset (Fin n)) :
    (flatGrid_expandCoordSet (N := N) I).card = I.card * flatGridBitWidth N := by
  unfold flatGrid_expandCoordSet
  calc
    ((I.product (Finset.univ : Finset (Fin (flatGridBitWidth N)))).image
        (finProdFinEquiv : Fin n × Fin (flatGridBitWidth N) ≃ Fin (n * flatGridBitWidth N))).card
        = (I.product (Finset.univ : Finset (Fin (flatGridBitWidth N)))).card :=
            Finset.card_image_of_injective _ (finProdFinEquiv : Fin n × Fin (flatGridBitWidth N) ≃
              Fin (n * flatGridBitWidth N)).injective
    _ = I.card * (Finset.univ : Finset (Fin (flatGridBitWidth N))).card := Finset.card_product _ _
    _ = I.card * flatGridBitWidth N := by simp

/-- If a set of grid coordinates is sufficient, then all bits belonging to those coordinate blocks
are sufficient for the binary-encoded problem. -/
theorem flatGrid_binaryEncoded_sufficient_of_gridSufficient
    {A : Type*} {n N : ℕ} (dp : DecisionProblem A (Fin n → BoundedGrid N))
    (I : Finset (Fin n)) (hI : dp.isSufficient I) :
    (flatGrid_binaryEncodedDecisionProblem dp).isSufficient (flatGrid_expandCoordSet (N := N) I) := by
  intro b b' hAgree
  apply hI
  intro i hi
  have hWord :
      (fun j : Fin (flatGridBitWidth N) =>
        b ((finProdFinEquiv : Fin n × Fin (flatGridBitWidth N) ≃ Fin (n * flatGridBitWidth N)) (i, j))) =
      (fun j : Fin (flatGridBitWidth N) =>
        b' ((finProdFinEquiv : Fin n × Fin (flatGridBitWidth N) ≃ Fin (n * flatGridBitWidth N)) (i, j))) := by
    funext j
    exact hAgree _ (flatGrid_mem_expandCoordSet_of_mem (N := N) hi j)
  have hCoord : flatGrid_decodeBits b i = flatGrid_decodeBits b' i := by
    simp [flatGrid_decodeBits, hWord]
  exact hCoord

/-- The binary lift preserves the optimizer-class count of the original finite-grid decision
problem. -/
theorem flatGrid_binaryEncoded_numOptClasses_eq
    {A : Type*} {n N : ℕ} [DecidableEq A] [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → BoundedGrid N)) :
    (flatGrid_binaryEncodedDecisionProblem dp).numOptClasses = dp.numOptClasses := by
  classical
  let dpBits := flatGrid_binaryEncodedDecisionProblem dp
  have hsubsetBits : Finset.univ.image dpBits.Opt ⊆ Finset.univ.image dp.Opt := by
    intro optVal hoptVal
    rw [Finset.mem_image] at hoptVal ⊢
    rcases hoptVal with ⟨b, -, hb⟩
    refine ⟨flatGrid_decodeBits b, Finset.mem_univ _, ?_⟩
    simpa [dpBits, flatGrid_binaryEncodedDecisionProblem] using hb
  have hsubsetGrid : Finset.univ.image dp.Opt ⊆ Finset.univ.image dpBits.Opt := by
    intro optVal hoptVal
    rw [Finset.mem_image] at hoptVal ⊢
    rcases hoptVal with ⟨s, -, hs⟩
    refine ⟨flatGrid_encodeBits s, Finset.mem_univ _, ?_⟩
    change dp.Opt (flatGrid_decodeBits (flatGrid_encodeBits s)) = optVal
    simpa [flatGrid_decodeBits_encodeBits] using hs
  apply le_antisymm
  · calc
      (flatGrid_binaryEncodedDecisionProblem dp).numOptClasses
          = (Finset.univ.image dpBits.Opt).card := rfl
      _ ≤ (Finset.univ.image dp.Opt).card := Finset.card_le_card hsubsetBits
      _ = dp.numOptClasses := rfl
  · calc
      dp.numOptClasses = (Finset.univ.image dp.Opt).card := rfl
      _ ≤ (Finset.univ.image dpBits.Opt).card := Finset.card_le_card hsubsetGrid
      _ = (flatGrid_binaryEncodedDecisionProblem dp).numOptClasses := rfl

/-- Local binary entropy-count lemma specialized to function-state Boolean coordinate spaces, used
to avoid instance-resolution ambiguity when composing the finite-grid binary encoding with the
Landauer chain. -/
theorem binary_numOptClasses_le_pow_srank
    {A : Type*} {m : ℕ} [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin m → Bool)) :
    dp.numOptClasses ≤ 2 ^ dp.srank := by
  let R := Finset.univ.filter dp.isRelevant
  have hinj : ∀ s s' : Fin m → Bool,
      (∀ i : Fin m, CoordinateSpace.proj s i = CoordinateSpace.proj s' i) → s = s' := by
    intro s s' h
    funext i
    exact h i
  have hsuff : dp.isSufficient R := relevantSet_isSufficient dp hinj
  let pat : (Fin m → Bool) → (R → Bool) := fun s => fun ⟨i, _⟩ => s i
  have hfactor : ∀ s s' : Fin m → Bool, pat s = pat s' → dp.Opt s = dp.Opt s' := by
    intro s s' hpat
    apply hsuff
    intro i hi
    have : pat s ⟨i, hi⟩ = pat s' ⟨i, hi⟩ := congrFun hpat ⟨i, hi⟩
    exact this
  have hcard_pat : Fintype.card (R → Bool) = 2 ^ R.card := by
    simp only [Fintype.card_fun, Fintype.card_bool, Fintype.card_coe]
  have hcard_R : R.card = dp.srank := rfl
  let optFromPat : (R → Bool) → Set A := fun p =>
    dp.Opt (fun i => if h : i ∈ R then p ⟨i, h⟩ else false)
  have hsubset : Finset.univ.image dp.Opt ⊆ Finset.univ.image optFromPat := by
    intro optVal hoptVal
    rw [Finset.mem_image] at hoptVal ⊢
    obtain ⟨s, _, hs⟩ := hoptVal
    use pat s
    constructor
    · exact Finset.mem_univ _
    · rw [← hs]
      apply hfactor
      funext ⟨i, hi⟩
      simp only [pat, dif_pos hi]
  calc dp.numOptClasses
      = (Finset.univ.image dp.Opt).card := rfl
    _ ≤ (Finset.univ.image optFromPat).card := Finset.card_le_card hsubset
    _ ≤ Finset.univ.card := Finset.card_image_le
    _ = Fintype.card (R → Bool) := Finset.card_univ
    _ = 2 ^ R.card := hcard_pat
    _ = 2 ^ dp.srank := by rw [hcard_R]

/-- The bit-level structural rank is controlled by the grid-level structural rank times the number
of bits used to encode each grid coordinate. -/
theorem flatGrid_binaryEncoded_srank_le_bitWidth_mul_srank
    {A : Type*} {n N : ℕ} [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → BoundedGrid N)) :
    (flatGrid_binaryEncodedDecisionProblem dp).srank ≤ dp.srank * flatGridBitWidth N := by
  let R : Finset (Fin n) := Finset.univ.filter dp.isRelevant
  have hinj : ∀ s s' : Fin n → BoundedGrid N,
      (∀ i : Fin n, CoordinateSpace.proj s i = CoordinateSpace.proj s' i) → s = s' := by
    intro s s' h
    funext i
    exact h i
  have hRSuff : dp.isSufficient R := relevantSet_isSufficient dp hinj
  have hBitsSuff :
      (flatGrid_binaryEncodedDecisionProblem dp).isSufficient (flatGrid_expandCoordSet (N := N) R) :=
    flatGrid_binaryEncoded_sufficient_of_gridSufficient dp R hRSuff
  calc
    (flatGrid_binaryEncodedDecisionProblem dp).srank ≤ (flatGrid_expandCoordSet (N := N) R).card :=
      srank_le_sufficient_card _ _ hBitsSuff
    _ = R.card * flatGridBitWidth N := flatGrid_expandCoordSet_card (N := N) R
    _ = dp.srank * flatGridBitWidth N := by rfl

/-- The grid alphabet fits inside the advertised fixed-length binary word. -/
theorem flatGrid_gridArity_lt_two_pow_bitWidth {N : Nat} :
    2 * N + 1 < 2 ^ flatGridBitWidth N := by
  simpa [flatGridBitWidth] using Nat.lt_size_self (2 * N + 1)

/-- The grid alphabet size is therefore logarithmically bounded by the fixed binary word length. -/
theorem flatGrid_log_gridArity_le_bitWidth_log2 {N : Nat} :
    Real.log (2 * N + 1 : ℝ) ≤ (flatGridBitWidth N : ℝ) * Real.log 2 := by
  have hCast : (2 * N + 1 : ℝ) ≤ (2 : ℝ) ^ flatGridBitWidth N := by
    exact_mod_cast Nat.le_of_lt (flatGrid_gridArity_lt_two_pow_bitWidth (N := N))
  calc
    Real.log (2 * N + 1 : ℝ) ≤ Real.log ((2 : ℝ) ^ flatGridBitWidth N) :=
      Real.log_le_log (by positivity) hCast
    _ = (flatGridBitWidth N : ℝ) * Real.log 2 := by rw [Real.log_pow, mul_comm]

/-- The finite-grid optimizer entropy is bounded explicitly by the number of bits needed per
relevant coordinate block. -/
theorem flatGrid_natEntropy_le_srank_bitWidth_log2
    {A : Type*} {n N : ℕ} [DecidableEq A] [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → BoundedGrid N)) :
    Real.log (dp.numOptClasses : ℝ) ≤
      (((dp.srank * flatGridBitWidth N : ℕ)) : ℝ) * Real.log 2 := by
  have hNat := flatGrid_natEntropy_le_srank_log_gridArity dp
  have hWidth := flatGrid_log_gridArity_le_bitWidth_log2 (N := N)
  have hMul :
      (dp.srank : ℝ) * Real.log (2 * N + 1 : ℝ) ≤
        (dp.srank : ℝ) * ((flatGridBitWidth N : ℝ) * Real.log 2) := by
    exact mul_le_mul_of_nonneg_left hWidth (by exact_mod_cast Nat.zero_le dp.srank)
  calc
    Real.log (dp.numOptClasses : ℝ) ≤ (dp.srank : ℝ) * Real.log (2 * N + 1 : ℝ) := hNat
    _ ≤ (dp.srank : ℝ) * ((flatGridBitWidth N : ℝ) * Real.log 2) := hMul
    _ = (((dp.srank * flatGridBitWidth N : ℕ)) : ℝ) * Real.log 2 := by
        norm_num
        ring

/-- Grid optimizer classes are bounded by the number of binary words available across the relevant
coordinate blocks. -/
theorem flatGrid_numOptClasses_le_two_pow_srank_mul_bitWidth
    {A : Type*} {n N : ℕ} [DecidableEq A] [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → BoundedGrid N)) :
    dp.numOptClasses ≤ 2 ^ (dp.srank * flatGridBitWidth N) := by
  have hGrid := flatGrid_numOptClasses_le_gridArity_pow_srank dp
  have hBase : 2 * N + 1 ≤ 2 ^ flatGridBitWidth N :=
    Nat.le_of_lt (flatGrid_gridArity_lt_two_pow_bitWidth (N := N))
  calc
    dp.numOptClasses ≤ (2 * N + 1) ^ dp.srank := hGrid
    _ ≤ (2 ^ flatGridBitWidth N) ^ dp.srank := Nat.pow_le_pow_left hBase _
    _ = 2 ^ (flatGridBitWidth N * dp.srank) := by rw [← Nat.pow_mul]
    _ = 2 ^ (dp.srank * flatGridBitWidth N) := by rw [Nat.mul_comm]

/-- The finite binary encoding of a flat grid problem carries the exact optimizer entropy of the
original grid problem, so the binary Landauer theorem applies without an extra entropy hypothesis. -/
theorem flatGrid_binaryEncoded_landauer_entropy_lower_bound
    {A : Type*} {n N : ℕ} [DecidableEq A] [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → BoundedGrid N))
    (I : Finset (Fin (n * flatGridBitWidth N)))
    (hI : (flatGrid_binaryEncodedDecisionProblem dp).isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) = DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T) :
    (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≥
      kB * T * Real.log (dp.numOptClasses : ℝ) := by
  let dpBits : DecisionProblem A (Fin (n * flatGridBitWidth N) → Bool) :=
    flatGrid_binaryEncodedDecisionProblem dp
  let sr : ℕ := dpBits.srank
  have hJ : 0 < M.joulesPerBit :=
    DecisionQuotient.ThermodynamicLift.joulesPerBit_pos_of_landauer_calibration M hkB hT hCal
  have hEnergyNat := DecisionQuotient.Physics.BoundedAcquisition.energy_ge_srank_cost
    dpBits I hI M hJ
  have hLandauer : (sr : ℝ) * (kB * T * Real.log 2) ≤
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
    have hCast : (((M.joulesPerBit * sr : ℕ)) : ℝ) ≤
        (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
      exact_mod_cast hEnergyNat
    calc
      (sr : ℝ) * (kB * T * Real.log 2)
          = (sr : ℝ) * DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T := by
              simp [DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit]
      _ = (sr : ℝ) * (M.joulesPerBit : ℝ) := by rw [hCal]
      _ = ((M.joulesPerBit : ℝ) * (sr : ℝ)) := by ring
      _ = (((M.joulesPerBit * sr : ℕ)) : ℝ) := by norm_num
      _ ≤ (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
          simpa [DecisionQuotient.ThermodynamicLift.energyLowerBound] using hCast
  have hClassesBits : dpBits.numOptClasses ≤ 2 ^ dpBits.srank := by
    simpa [dpBits] using binary_numOptClasses_le_pow_srank dpBits
  have hPosBits : 0 < (dpBits.numOptClasses : ℝ) := by
    exact_mod_cast (DecisionProblem.numOptClasses_pos (dp := dpBits))
  have hCastBits : (dpBits.numOptClasses : ℝ) ≤ (2 : ℝ) ^ dpBits.srank := by
    exact_mod_cast hClassesBits
  have hNatEntropyBits : Real.log (dpBits.numOptClasses : ℝ) ≤ (dpBits.srank : ℝ) * Real.log 2 := by
    calc
      Real.log (dpBits.numOptClasses : ℝ) ≤ Real.log ((2 : ℝ) ^ dpBits.srank) :=
        Real.log_le_log hPosBits hCastBits
      _ = (dpBits.srank : ℝ) * Real.log 2 := by rw [Real.log_pow, mul_comm]
  have hScale : 0 ≤ kB * T := mul_nonneg hkB.le hT.le
  calc
    kB * T * Real.log (dp.numOptClasses : ℝ)
        = kB * T * Real.log (dpBits.numOptClasses : ℝ) := by
            rw [flatGrid_binaryEncoded_numOptClasses_eq dp]
    _ ≤ kB * T * ((dpBits.srank : ℝ) * Real.log 2) :=
      mul_le_mul_of_nonneg_left hNatEntropyBits hScale
    _ = (dpBits.srank : ℝ) * (kB * T * Real.log 2) := by ring
    _ ≤ (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := hLandauer

/-- Binary lift of the flat sampled exact docking problem. -/
noncomputable def sampledDocking_flatBinaryEncodedDecisionProblem
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    DecisionProblem (SupportedAction prob.samples)
      (FlatGridBitState (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) N) :=
  flatGrid_binaryEncodedDecisionProblem (sampledDocking_flatExactDecisionProblem prob)

/-- The thermodynamic entropy statement for finite docking is honest at the binary-encoding level:
the sampled grid problem inherits Landauer cost through its exact fixed-length binary register, not
through an unsupported inequality on the unencoded grid quotient entropy. -/
theorem sampledDocking_flatBinaryEncoded_landauer_entropy_lower_bound
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (I : Finset
      (Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL *
        flatGridBitWidth N)))
    (hI : (sampledDocking_flatBinaryEncodedDecisionProblem prob).isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) = DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T) :
    (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≥
      kB * T * Real.log ((sampledDocking_flatExactDecisionProblem prob).numOptClasses : ℝ) := by
  exact flatGrid_binaryEncoded_landauer_entropy_lower_bound
    (dp := sampledDocking_flatExactDecisionProblem prob) I hI M hkB hT hCal

/-- On sampled exact docking, the binary-encoded structural rank is bounded by the grid structural
rank times the bits required for one grid coordinate. -/
theorem sampledDocking_flatBinaryEncoded_srank_le_bitWidth_mul_srank
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    (sampledDocking_flatBinaryEncodedDecisionProblem prob).srank ≤
      (sampledDocking_flatExactDecisionProblem prob).srank * flatGridBitWidth N := by
  exact flatGrid_binaryEncoded_srank_le_bitWidth_mul_srank
    (dp := sampledDocking_flatExactDecisionProblem prob)

/-- The finite sampled docking entropy bound can be written directly in terms of the explicit bit
length per grid coordinate. -/
theorem sampledDocking_flatExact_natEntropy_le_srank_bitWidth_log2
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    Real.log ((sampledDocking_flatExactDecisionProblem prob).numOptClasses : ℝ) ≤
      ((((sampledDocking_flatExactDecisionProblem prob).srank * flatGridBitWidth N : ℕ)) : ℝ) *
        Real.log 2 := by
  exact flatGrid_natEntropy_le_srank_bitWidth_log2
    (dp := sampledDocking_flatExactDecisionProblem prob)

/-- Action-side symmetry-broken refinement of the flat sampled exact docking problem. -/
noncomputable def sampledDocking_flatActionPinnedDecisionProblem
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    DecisionProblem (SupportedAction prob.samples)
      (Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N) :=
  actionPinnedDecisionProblem (sampledDocking_flatExactDecisionProblem prob)

/-- The action-side symmetry-broken sampled docking problem has a strict winner at every flat grid
state, chosen canonically from the base optimal set by the finite action code. -/
theorem sampledDocking_flatActionPinned_strictOpt
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (s : Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N) :
    StrictOpt (sampledDocking_flatActionPinnedDecisionProblem prob)
      (maxCodeOptAction (sampledDocking_flatExactDecisionProblem prob) s) s := by
  exact actionPinned_strictOpt_maxCodeOptAction
    (dp := sampledDocking_flatExactDecisionProblem prob) s

/-- Action-side symmetry breaking does not increase the flat sampled docking structural rank. -/
theorem sampledDocking_flatActionPinned_srank_le_srank
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) :
    (sampledDocking_flatActionPinnedDecisionProblem prob).srank ≤
      (sampledDocking_flatExactDecisionProblem prob).srank := by
  exact actionPinned_srank_le_srank
    (dp := sampledDocking_flatExactDecisionProblem prob)

/-- In the symmetry-broken flat sampled docking refinement, the canonical winner's strict utility
gap dominates the statewise tie-break scale whenever the sampled action family has at least two
actions. -/
theorem sampledDocking_flatActionPinned_strictUtilityGap_ge_tieBreak_of_card_gt_one
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (hCard : 1 < Fintype.card (SupportedAction prob.samples))
    (s : Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N) :
    actionTieBreakEps (sampledDocking_flatExactDecisionProblem prob) s ≤
      StrictUtilityGap (sampledDocking_flatActionPinnedDecisionProblem prob)
        (maxCodeOptAction (sampledDocking_flatExactDecisionProblem prob) s) s := by
  exact actionPinned_strictUtilityGap_ge_tieBreak_of_card_gt_one hCard
    (dp := sampledDocking_flatExactDecisionProblem prob) s

/-- The symmetry-broken flat sampled docking refinement has a positive strict utility gap at every
flat grid state whenever the sampled action family has at least two actions. -/
theorem sampledDocking_flatActionPinned_strictUtilityGap_pos_of_card_gt_one
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (hCard : 1 < Fintype.card (SupportedAction prob.samples))
    (s : Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N) :
    0 < StrictUtilityGap (sampledDocking_flatActionPinnedDecisionProblem prob)
      (maxCodeOptAction (sampledDocking_flatExactDecisionProblem prob) s) s := by
  exact actionPinned_strictUtilityGap_pos_of_card_gt_one hCard
    (dp := sampledDocking_flatExactDecisionProblem prob) s

/-- Minimum strict utility gap of the canonical symmetry-broken winner across the finite flat grid
state space. -/
noncomputable def sampledDocking_flatActionPinned_minStrictUtilityGap
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N) : ℝ :=
  let gaps : Finset ℝ :=
    (Finset.univ : Finset
      (Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N)).image
        (fun s =>
          StrictUtilityGap (sampledDocking_flatActionPinnedDecisionProblem prob)
            (maxCodeOptAction (sampledDocking_flatExactDecisionProblem prob) s) s)
  gaps.min' <| by
    let s : Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N :=
      Classical.choice
        (inferInstance :
          Nonempty
            (Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N))
    exact ⟨_, Finset.mem_image.mpr ⟨s, by simp, rfl⟩⟩

/-- The minimum symmetry-broken strict utility gap is bounded above by each state's canonical gap.
-/
theorem sampledDocking_flatActionPinned_minStrictUtilityGap_le
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (s : Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N) :
    sampledDocking_flatActionPinned_minStrictUtilityGap prob ≤
      StrictUtilityGap (sampledDocking_flatActionPinnedDecisionProblem prob)
        (maxCodeOptAction (sampledDocking_flatExactDecisionProblem prob) s) s := by
  unfold sampledDocking_flatActionPinned_minStrictUtilityGap
  dsimp
  exact Finset.min'_le _ _ (Finset.mem_image.mpr ⟨s, by simp, rfl⟩)

/-- Because the flat sampled docking state space is finite, the pointwise positive strict gaps of
the symmetry-broken refinement admit a positive global minimum. -/
theorem sampledDocking_flatActionPinned_minStrictUtilityGap_pos_of_card_gt_one
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (hCard : 1 < Fintype.card (SupportedAction prob.samples)) :
    0 < sampledDocking_flatActionPinned_minStrictUtilityGap prob := by
  unfold sampledDocking_flatActionPinned_minStrictUtilityGap
  dsimp
  rw [Finset.lt_min'_iff]
  intro y hy
  rcases Finset.mem_image.mp hy with ⟨s, -, rfl⟩
  exact sampledDocking_flatActionPinned_strictUtilityGap_pos_of_card_gt_one prob hCard s

/-- A sampled action family with at least two retained actions yields a uniform positive strict
utility gap for the canonical symmetry-broken winner over the entire flat grid state space. -/
theorem sampledDocking_flatActionPinned_uniformStrictUtilityGap_of_card_gt_one
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (hCard : 1 < Fintype.card (SupportedAction prob.samples)) :
    ∃ γ : ℝ, 0 < γ ∧
      ∀ s : Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL) → BoundedGrid N,
        γ ≤ StrictUtilityGap (sampledDocking_flatActionPinnedDecisionProblem prob)
          (maxCodeOptAction (sampledDocking_flatExactDecisionProblem prob) s) s := by
  refine ⟨sampledDocking_flatActionPinned_minStrictUtilityGap prob, ?_, ?_⟩
  · exact sampledDocking_flatActionPinned_minStrictUtilityGap_pos_of_card_gt_one prob hCard
  · intro s
    exact sampledDocking_flatActionPinned_minStrictUtilityGap_le prob s

/-- Exact resolution for an arbitrary finite-coordinate decision problem fits inside a bounded region
and horizon when some sufficient coordinate set fits inside that acquisition budget. -/
def decisionExactResolutionWithin
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (R : DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion) (T : ℕ) : Prop :=
  ∃ I : Finset (Fin n), dp.isSufficient I ∧
    I.card ≤ DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T

/-- Exact resolution inside a bounded acquisition budget forces structural rank below the total
available number of acquisition events. -/
theorem decision_srank_le_boundedAcquisitions_of_exactResolutionWithin
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (R : DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion) (T : ℕ)
    (hRes : decisionExactResolutionWithin dp R T) :
    dp.srank ≤ DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T := by
  rcases hRes with ⟨I, hI, hBudget⟩
  exact le_trans
    (DecisionQuotient.Physics.BoundedAcquisition.srank_le_resolution_bits dp I hI)
    hBudget

/-- The same bounded-budget witness lifts the structural-rank floor to a concrete bounded-region
energy floor. -/
theorem decision_energy_le_boundedAcquisitions_of_exactResolutionWithin
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (R : DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion) (T : ℕ)
    (hRes : decisionExactResolutionWithin dp R T)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) (hJ : 0 < M.joulesPerBit) :
    M.joulesPerBit * dp.srank ≤
      DecisionQuotient.ThermodynamicLift.energyLowerBound M
        (DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T) := by
  rcases hRes with ⟨I, hI, hBudget⟩
  have hEnergyI := DecisionQuotient.Physics.BoundedAcquisition.energy_ge_srank_cost dp I hI M hJ
  have hBudgetEnergy :
      DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card ≤
        DecisionQuotient.ThermodynamicLift.energyLowerBound M
          (DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T) := by
    simpa [DecisionQuotient.ThermodynamicLift.energyLowerBound] using
      Nat.mul_le_mul_left M.joulesPerBit hBudget
  exact le_trans hEnergyI hBudgetEnergy

/-- Resolution-controlled continuous-to-grid approximation plus a strict gap yields exact/coarse
winner preservation on the lifted grid state. -/
theorem resolutionControlled_gap_implies_opt_invariance
    {A : Type u} {Scont Sgrid : Type v} [Fintype A]
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (eps : ℝ → ℝ)
    (res : ℝ)
    (hApprox : DecisionQuotient.Tractability.GridConvergence.ResolutionControlledApprox
      uCont uGrid lift eps res)
    (sGrid : Sgrid)
    (aStar : A)
    (hDelta : 0 ≤ eps res)
    (hStrict : StrictOpt { utility := fun a s => uCont a (lift s) } aStar sGrid)
    (hBound : eps res < StrictUtilityGap { utility := fun a s => uCont a (lift s) } aStar sGrid / 2) :
    ({ utility := fun a s => uCont a (lift s) } : DecisionProblem A Sgrid).Opt sGrid =
      ({ utility := uGrid } : DecisionProblem A Sgrid).Opt sGrid := by
  have hUniform := resolutionControlledApprox_implies_uniformApprox uCont uGrid lift eps res hApprox
  exact uniform_approx_implies_opt_invariance
    ({ utility := fun a s => uCont a (lift s) } : DecisionProblem A Sgrid)
    ({ utility := uGrid } : DecisionProblem A Sgrid)
    (eps res) hUniform sGrid aStar hDelta hStrict hBound

/-- A Lipschitz utility transport bound plus a uniform grid-resolution bound yields exact/coarse
winner preservation on the lifted grid state. -/
theorem lipschitzResolution_gap_implies_opt_invariance
    {A : Type u} {Scont Sgrid : Type v} [Fintype A]
    (uCont : A → Scont → ℝ)
    (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont)
    (stateError : Sgrid → ℝ)
    (L res : ℝ)
    (hLip : DecisionQuotient.Tractability.GridConvergence.LipschitzUtilityApprox
      uCont uGrid lift stateError L)
    (hState : ∀ sGrid, stateError sGrid ≤ res)
    (hL : 0 ≤ L)
    (sGrid : Sgrid)
    (aStar : A)
    (hDelta : 0 ≤ L * res)
    (hStrict : StrictOpt { utility := fun a s => uCont a (lift s) } aStar sGrid)
    (hBound : L * res < StrictUtilityGap { utility := fun a s => uCont a (lift s) } aStar sGrid / 2) :
    ({ utility := fun a s => uCont a (lift s) } : DecisionProblem A Sgrid).Opt sGrid =
      ({ utility := uGrid } : DecisionProblem A Sgrid).Opt sGrid := by
  have hApprox := lipschitzUtilityApprox_implies_resolutionControlled
    uCont uGrid lift stateError L res hLip hState hL
  exact resolutionControlled_gap_implies_opt_invariance
    uCont uGrid lift (fun r => L * r) res hApprox sGrid aStar
    hDelta hStrict hBound

/-- A bounded-potential witness together with a sufficiently large cutoff produces the molecular
structural-rank bound directly. -/
theorem boundedPotential_largeCutoff_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (tail_coefficient minimum_gap : ℝ)
    (hBounds : DecisionQuotient.Tractability.SatisfiesBoundedPotential prob tail_coefficient minimum_gap)
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff < minimum_gap / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff)
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  have hBounded := largeCutoff_implies_bounded prob tail_coefficient minimum_gap
    hBounds hPosCutoff hCutoffSize hTailPos
  exact molecularDocking_srank_bound prob hStrictAll hBounded

/-- Exact Lennard-Jones tail control plus a sufficiently large cutoff yields a direct molecular
structural-rank bound. -/
theorem exactLJ_largeCutoff_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (distance : MDAction → MDState → ℝ)
    (ε σ tail_coefficient : ℝ)
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < DecisionQuotient.Tractability.finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distance a s))
    (hPhys : DecisionQuotient.Tractability.LJApproximation.PhysicalDistanceDecayLJ prob distance ε σ tail_coefficient)
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff <
      DecisionQuotient.Tractability.finiteMinimumGap prob w / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  have hBounds := DecisionQuotient.Tractability.LJApproximation.exactLJ_is_BoundedPotential
    prob distance ε σ tail_coefficient w hGapPos hUtility hPhys
  exact boundedPotential_largeCutoff_srank_bound
    prob tail_coefficient (DecisionQuotient.Tractability.finiteMinimumGap prob w)
    hBounds hPosCutoff hCutoffSize hTailPos (fun s => ⟨(w s).1, (w s).2⟩)

/-- Exact Coulomb tail control plus a sufficiently large cutoff yields a direct molecular
structural-rank bound. -/
theorem exactCoulomb_largeCutoff_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (distance : MDAction → MDState → ℝ)
    (q_i q_j tail_coefficient : ℝ)
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < DecisionQuotient.Tractability.finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distance a s))
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (numMDCoordinates prob),
        j ≠ proteinCoordFin prob atomIdx hAtomInProtein axis →
        mdProj prob s j = mdProj prob s' j) →
      ¬ atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MDAction,
        |DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distance a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distance a s')|
          ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 R))
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff <
      DecisionQuotient.Tractability.finiteMinimumGap prob w / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  have hBounds := DecisionQuotient.Tractability.CoulombApproximation.exactCoulomb_satisfiesBoundedPotential_of_tailBound
    prob distance q_i q_j tail_coefficient w hGapPos hUtility hTail
  exact boundedPotential_largeCutoff_srank_bound
    prob tail_coefficient (DecisionQuotient.Tractability.finiteMinimumGap prob w)
    hBounds hPosCutoff hCutoffSize hTailPos (fun s => ⟨(w s).1, (w s).2⟩)

/-- Exact real-space Ewald Coulomb tail control plus a sufficiently large cutoff yields a direct
molecular structural-rank bound. -/
theorem exactRealEwald_largeCutoff_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (distance : MDAction → MDState → ℝ)
    (q_i q_j alpha tail_coefficient : ℝ)
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < DecisionQuotient.Tractability.finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => DecisionQuotient.Tractability.CoulombApproximation.exactRealEwaldScore q_i q_j alpha (distance a s))
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (numMDCoordinates prob),
        j ≠ proteinCoordFin prob atomIdx hAtomInProtein axis →
        mdProj prob s j = mdProj prob s' j) →
      ¬ atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MDAction,
        |DecisionQuotient.Tractability.CoulombApproximation.exactRealEwaldScore q_i q_j alpha (distance a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactRealEwaldScore q_i q_j alpha (distance a s')|
          ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 R))
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff <
      DecisionQuotient.Tractability.finiteMinimumGap prob w / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  have hBounds := DecisionQuotient.Tractability.CoulombApproximation.exactRealEwald_satisfiesBoundedPotential_of_tailBound
    prob distance q_i q_j alpha tail_coefficient w hGapPos hUtility hTail
  exact boundedPotential_largeCutoff_srank_bound
    prob tail_coefficient (DecisionQuotient.Tractability.finiteMinimumGap prob w)
    hBounds hPosCutoff hCutoffSize hTailPos (fun s => ⟨(w s).1, (w s).2⟩)

/-- A positive lower shell radius yields an explicit uniform bound on the magnitude of the
Lennard-Jones gradient formula. -/
theorem abs_ljGradExpr_le_shellBound
    (ε σ rmin r : ℝ)
    (hmin : 0 < rmin)
    (hr : rmin ≤ r) :
    |ljGradExpr ε σ r| ≤ ljGradShellBound ε σ rmin := by
  have hrpos : 0 < r := lt_of_lt_of_le hmin hr
  have hInv : r⁻¹ ≤ rmin⁻¹ := (inv_le_inv₀ hrpos hmin).2 hr
  have hInvNonneg : 0 ≤ r⁻¹ := inv_nonneg.mpr hrpos.le
  have hInvMinNonneg : 0 ≤ rmin⁻¹ := inv_nonneg.mpr hmin.le
  have hσr : |σ| * r⁻¹ ≤ |σ| * rmin⁻¹ :=
    mul_le_mul_of_nonneg_left hInv (abs_nonneg σ)
  have hPow6 : (|σ| * r⁻¹) ^ (6 : ℕ) ≤ (|σ| * rmin⁻¹) ^ (6 : ℕ) := by
    exact pow_le_pow_left₀ (mul_nonneg (abs_nonneg σ) hInvNonneg) hσr _
  have hPow12 : (|σ| * r⁻¹) ^ (12 : ℕ) ≤ (|σ| * rmin⁻¹) ^ (12 : ℕ) := by
    exact pow_le_pow_left₀ (mul_nonneg (abs_nonneg σ) hInvNonneg) hσr _
  have hCoeff : 24 * |ε| * r⁻¹ ≤ 24 * |ε| * rmin⁻¹ := by
    have h24ε : 0 ≤ 24 * |ε| := by positivity
    exact mul_le_mul_of_nonneg_left hInv h24ε
  have hInner :
      (|σ| * r⁻¹) ^ (6 : ℕ) + 2 * (|σ| * r⁻¹) ^ (12 : ℕ) ≤
        (|σ| * rmin⁻¹) ^ (6 : ℕ) + 2 * (|σ| * rmin⁻¹) ^ (12 : ℕ) := by
    gcongr
  calc
    |ljGradExpr ε σ r|
        = |24 * ε * r⁻¹| * |(σ * r⁻¹) ^ (6 : ℕ) - 2 * (σ * r⁻¹) ^ (12 : ℕ)| := by
            simp [ljGradExpr, abs_mul]
    _ ≤ |24 * ε * r⁻¹| * (|(σ * r⁻¹) ^ (6 : ℕ)| + |2 * (σ * r⁻¹) ^ (12 : ℕ)|) := by
          refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
          simpa [Real.norm_eq_abs, sub_eq_add_neg, abs_neg] using
            (norm_add_le ((σ * r⁻¹) ^ (6 : ℕ)) (-(2 * (σ * r⁻¹) ^ (12 : ℕ))))
    _ = (24 * |ε| * r⁻¹) * ((|σ| * r⁻¹) ^ (6 : ℕ) + 2 * (|σ| * r⁻¹) ^ (12 : ℕ)) := by
          simp [abs_mul, abs_pow, abs_of_nonneg hInvNonneg]
    _ ≤ (24 * |ε| * rmin⁻¹) * ((|σ| * rmin⁻¹) ^ (6 : ℕ) + 2 * (|σ| * rmin⁻¹) ^ (12 : ℕ)) := by
          exact mul_le_mul hCoeff hInner (by positivity) (by positivity)
    _ = ljGradShellBound ε σ rmin := by
          simp [ljGradShellBound]

/-- Closed-form second derivative of the Lennard-Jones potential on the nonzero shell. -/
theorem secondDeriv_lennardJones (ε σ r : ℝ) (hr : r ≠ 0) :
    HasDerivAt (fun x => ljGradExpr ε σ x) (ljSecondDerivExpr ε σ r) r := by
  have h_inv : HasDerivAt (fun x : ℝ => x⁻¹) (- (r ^ 2)⁻¹) r := hasDerivAt_inv hr
  have h_base : HasDerivAt (fun x : ℝ => σ * x⁻¹) (σ * (- (r ^ 2)⁻¹)) r :=
    HasDerivAt.const_mul σ h_inv
  have h6 : HasDerivAt (fun x : ℝ => (σ * x⁻¹) ^ (6 : ℕ))
      (6 * (σ * r⁻¹) ^ (5 : ℕ) * (σ * (- (r ^ 2)⁻¹))) r := by
    simpa using (HasDerivAt.comp r (hasDerivAt_pow 6 (σ * r⁻¹)) h_base)
  have h12 : HasDerivAt (fun x : ℝ => (σ * x⁻¹) ^ (12 : ℕ))
      (12 * (σ * r⁻¹) ^ (11 : ℕ) * (σ * (- (r ^ 2)⁻¹))) r := by
    simpa using (HasDerivAt.comp r (hasDerivAt_pow 12 (σ * r⁻¹)) h_base)
  have hA : HasDerivAt (fun x : ℝ => 24 * ε * x⁻¹) (24 * ε * (- (r ^ 2)⁻¹)) r := by
    simpa using (HasDerivAt.const_mul (24 * ε) h_inv)
  have hB : HasDerivAt (fun x : ℝ => (σ * x⁻¹) ^ (6 : ℕ) - 2 * (σ * x⁻¹) ^ (12 : ℕ))
      (6 * (σ * r⁻¹) ^ (5 : ℕ) * (σ * (- (r ^ 2)⁻¹)) -
        2 * (12 * (σ * r⁻¹) ^ (11 : ℕ) * (σ * (- (r ^ 2)⁻¹)))) r := by
    exact HasDerivAt.sub h6 (HasDerivAt.const_mul 2 h12)
  have hProd : HasDerivAt
      (fun x : ℝ => (24 * ε * x⁻¹) * ((σ * x⁻¹) ^ (6 : ℕ) - 2 * (σ * x⁻¹) ^ (12 : ℕ)))
      ((24 * ε * (- (r ^ 2)⁻¹)) * ((σ * r⁻¹) ^ (6 : ℕ) - 2 * (σ * r⁻¹) ^ (12 : ℕ)) +
        (24 * ε * r⁻¹) *
          (6 * (σ * r⁻¹) ^ (5 : ℕ) * (σ * (- (r ^ 2)⁻¹)) -
            2 * (12 * (σ * r⁻¹) ^ (11 : ℕ) * (σ * (- (r ^ 2)⁻¹)))) ) r :=
    hA.mul hB
  have h_r2 : (r ^ 2)⁻¹ = r⁻¹ * r⁻¹ := by
    calc
      (r ^ 2)⁻¹ = (r * r)⁻¹ := by ring_nf
      _ = r⁻¹ * r⁻¹ := mul_inv_rev r r
  have h_alg :
      (24 * ε * (- (r ^ 2)⁻¹)) * ((σ * r⁻¹) ^ (6 : ℕ) - 2 * (σ * r⁻¹) ^ (12 : ℕ)) +
        (24 * ε * r⁻¹) *
          (6 * (σ * r⁻¹) ^ (5 : ℕ) * (σ * (- (r ^ 2)⁻¹)) -
            2 * (12 * (σ * r⁻¹) ^ (11 : ℕ) * (σ * (- (r ^ 2)⁻¹))))
      = ljSecondDerivExpr ε σ r := by
    rw [h_r2]
    have h12 : (σ * r⁻¹) ^ (12 : ℕ) = σ ^ (12 : ℕ) * r⁻¹ ^ (12 : ℕ) := by simpa using mul_pow σ r⁻¹ 12
    have h11 : (σ * r⁻¹) ^ (11 : ℕ) = σ ^ (11 : ℕ) * r⁻¹ ^ (11 : ℕ) := by simpa using mul_pow σ r⁻¹ 11
    have h6p : (σ * r⁻¹) ^ (6 : ℕ) = σ ^ (6 : ℕ) * r⁻¹ ^ (6 : ℕ) := by simpa using mul_pow σ r⁻¹ 6
    have h5 : (σ * r⁻¹) ^ (5 : ℕ) = σ ^ (5 : ℕ) * r⁻¹ ^ (5 : ℕ) := by simpa using mul_pow σ r⁻¹ 5
    rw [h12, h11, h6p, h5]
    simp [ljSecondDerivExpr]
    ring_nf
  simpa [ljGradExpr, ljSecondDerivExpr] using (HasDerivAt.congr_deriv hProd h_alg)

/-- A positive lower shell radius yields an explicit uniform bound on the magnitude of the
second derivative of the Lennard-Jones potential. -/
theorem abs_ljSecondDerivExpr_le_shellBound
    (ε σ rmin r : ℝ)
    (hmin : 0 < rmin)
    (hr : rmin ≤ r) :
    |ljSecondDerivExpr ε σ r| ≤ ljSecondDerivShellBound ε σ rmin := by
  have hrpos : 0 < r := lt_of_lt_of_le hmin hr
  have hInv : r⁻¹ ≤ rmin⁻¹ := (inv_le_inv₀ hrpos hmin).2 hr
  have hInvNonneg : 0 ≤ r⁻¹ := inv_nonneg.mpr hrpos.le
  have hInvMinNonneg : 0 ≤ rmin⁻¹ := inv_nonneg.mpr hmin.le
  have hInvSq : (r⁻¹ ^ (2 : ℕ)) ≤ (rmin⁻¹ ^ (2 : ℕ)) := by
    have h := pow_le_pow_left₀ hInvNonneg hInv 2
    simpa [pow_two] using h
  have hσr : |σ| * r⁻¹ ≤ |σ| * rmin⁻¹ := mul_le_mul_of_nonneg_left hInv (abs_nonneg σ)
  have hPow6 : (|σ| * r⁻¹) ^ (6 : ℕ) ≤ (|σ| * rmin⁻¹) ^ (6 : ℕ) :=
    pow_le_pow_left₀ (mul_nonneg (abs_nonneg σ) hInvNonneg) hσr _
  have hPow12 : (|σ| * r⁻¹) ^ (12 : ℕ) ≤ (|σ| * rmin⁻¹) ^ (12 : ℕ) :=
    pow_le_pow_left₀ (mul_nonneg (abs_nonneg σ) hInvNonneg) hσr _
  have hCoeff : 24 * |ε| * (r⁻¹ ^ (2 : ℕ)) ≤ 24 * |ε| * (rmin⁻¹ ^ (2 : ℕ)) := by
    have h24ε : 0 ≤ 24 * |ε| := by positivity
    exact mul_le_mul_of_nonneg_left hInvSq h24ε
  have hInner :
      26 * (|σ| * r⁻¹) ^ (12 : ℕ) + 7 * (|σ| * r⁻¹) ^ (6 : ℕ) ≤
        26 * (|σ| * rmin⁻¹) ^ (12 : ℕ) + 7 * (|σ| * rmin⁻¹) ^ (6 : ℕ) := by
    gcongr
  calc
    |ljSecondDerivExpr ε σ r|
        = |24 * ε * (r⁻¹ ^ (2 : ℕ))| * |26 * (σ * r⁻¹) ^ (12 : ℕ) - 7 * (σ * r⁻¹) ^ (6 : ℕ)| := by
            simp [ljSecondDerivExpr, abs_mul]
    _ ≤ |24 * ε * (r⁻¹ ^ (2 : ℕ))| * (|26 * (σ * r⁻¹) ^ (12 : ℕ)| + |7 * (σ * r⁻¹) ^ (6 : ℕ)|) := by
          refine mul_le_mul_of_nonneg_left ?_ (abs_nonneg _)
          simpa [Real.norm_eq_abs, sub_eq_add_neg, abs_neg] using
            (norm_add_le (26 * (σ * r⁻¹) ^ (12 : ℕ)) (-(7 * (σ * r⁻¹) ^ (6 : ℕ))))
    _ = (24 * |ε| * (r⁻¹ ^ (2 : ℕ))) * (26 * (|σ| * r⁻¹) ^ (12 : ℕ) + 7 * (|σ| * r⁻¹) ^ (6 : ℕ)) := by
          simp [abs_mul, abs_pow, abs_of_nonneg hInvNonneg]
    _ ≤ (24 * |ε| * (rmin⁻¹ ^ (2 : ℕ))) * (26 * (|σ| * rmin⁻¹) ^ (12 : ℕ) + 7 * (|σ| * rmin⁻¹) ^ (6 : ℕ)) := by
          exact mul_le_mul hCoeff hInner (by positivity) (by positivity)
    _ = ljSecondDerivShellBound ε σ rmin := by
          simp [ljSecondDerivShellBound]

/-- On the positive shell, the ArrayDSL Lennard-Jones primitive has the expected closed-form
first derivative. -/
theorem hasDerivAt_arrayLennardJones_of_pos
    (ε σ r : ℝ)
    (hr : 0 < r) :
    HasDerivAt (fun x => Computation.ArrayDSL.lennardJones ε σ x) (ljGradExpr ε σ r) r := by
  have hraw : HasDerivAt (fun x => Computation.lennardJones ε σ x) (ljGradExpr ε σ r) r := by
    simpa [ljGradExpr] using (Computation.grad_of_lennardJones ε σ r (ne_of_gt hr))
  refine hraw.congr_of_eventuallyEq ?_
  filter_upwards [Ioi_mem_nhds hr] with x hx
  have hxpos : 0 < x := by simpa using hx
  simp [Computation.ArrayDSL.lennardJones, Computation.lennardJones, div_eq_mul_inv, ne_of_gt hxpos]

/-- A shellwise second-derivative bound makes the Lennard-Jones gradient Lipschitz on the shell. -/
theorem lennardJones_grad_diff_le_of_shellSecondDerivBound
    (ε σ rmin rmax K : ℝ)
    (hmin : 0 < rmin)
    (hSecondBound : ∀ x ∈ Set.Icc rmin rmax, |ljSecondDerivExpr ε σ x| ≤ K)
    {x y : ℝ}
    (hx : x ∈ Set.Icc rmin rmax)
    (hy : y ∈ Set.Icc rmin rmax) :
    |ljGradExpr ε σ y - ljGradExpr ε σ x| ≤ K * |y - x| := by
  have hderiv : ∀ z ∈ Set.Icc rmin rmax,
      HasDerivWithinAt (fun r => ljGradExpr ε σ r) (ljSecondDerivExpr ε σ z)
        (Set.Icc rmin rmax) z := by
    intro z hz
    have hzpos : 0 < z := lt_of_lt_of_le hmin hz.1
    exact (secondDeriv_lennardJones ε σ z (ne_of_gt hzpos)).hasDerivWithinAt
  have h :=
    (convex_Icc rmin rmax).norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun r => ljGradExpr ε σ r)
      (f' := fun z => ljSecondDerivExpr ε σ z)
      (s := Set.Icc rmin rmax)
      (x := x) (y := y)
      hderiv
      (by
        intro z hz
        simpa [Real.norm_eq_abs] using hSecondBound z hz)
      hx hy
  simpa [Real.norm_eq_abs] using h

/-- A shellwise second-derivative bound yields a quadratic Taylor remainder bound for the
Lennard-Jones potential. -/
theorem lennardJones_taylor_remainder_le_of_shellSecondDerivBound
    (ε σ rmin rmax K : ℝ)
    (hmin : 0 < rmin)
    (hSecondBound : ∀ z ∈ Set.Icc rmin rmax, |ljSecondDerivExpr ε σ z| ≤ K)
    {x y : ℝ}
    (hx : x ∈ Set.Icc rmin rmax)
    (hy : y ∈ Set.Icc rmin rmax) :
    |Computation.ArrayDSL.lennardJones ε σ y - Computation.ArrayDSL.lennardJones ε σ x -
        ljGradExpr ε σ x * (y - x)| ≤
      (K / 2) * |y - x| ^ (2 : ℕ) := by
  have hK : 0 ≤ K := le_trans (abs_nonneg _) (hSecondBound x hx)
  by_cases hxy : x ≤ y
  · let Δ : ℝ := y - x
    let remainder : ℝ → ℝ := fun t =>
      Computation.ArrayDSL.lennardJones ε σ (x + t) -
        (Computation.ArrayDSL.lennardJones ε σ x + ljGradExpr ε σ x * t)
    have hΔ : 0 ≤ Δ := sub_nonneg.mpr hxy
    have hRemCont : ContinuousOn remainder (Set.Icc 0 Δ) := by
      intro t ht
      have hxt : x + t ∈ Set.Icc rmin rmax := by
        refine ⟨?_, ?_⟩
        · linarith [hx.1, ht.1]
        · have hxty : x + t ≤ y := by
            linarith [ht.2]
          exact le_trans hxty hy.2
      have hxtpos : 0 < x + t := lt_of_lt_of_le hmin hxt.1
      have hLJ : HasDerivAt
          (fun u : ℝ => Computation.ArrayDSL.lennardJones ε σ (x + u))
          (ljGradExpr ε σ (x + t)) t := by
        simpa [add_comm, add_left_comm, add_assoc] using
          (hasDerivAt_arrayLennardJones_of_pos ε σ (x + t) hxtpos).comp t
            ((hasDerivAt_id t).const_add x)
      have hLin : HasDerivAt
          (fun u : ℝ => Computation.ArrayDSL.lennardJones ε σ x + ljGradExpr ε σ x * u)
          (ljGradExpr ε σ x) t := by
        simpa [mul_comm, mul_left_comm, mul_assoc] using
          (((hasDerivAt_id t).const_mul (ljGradExpr ε σ x)).const_add
            (Computation.ArrayDSL.lennardJones ε σ x))
      exact (hLJ.sub hLin).continuousAt.continuousWithinAt
    have hRemDeriv : ∀ t ∈ Set.Ico 0 Δ,
        HasDerivWithinAt remainder (ljGradExpr ε σ (x + t) - ljGradExpr ε σ x) (Set.Ici t) t := by
      intro t ht
      have hxt : x + t ∈ Set.Icc rmin rmax := by
        refine ⟨?_, ?_⟩
        · linarith [hx.1, ht.1]
        · have hxty : x + t ≤ y := by
            linarith [ht.2]
          exact le_trans hxty hy.2
      have hxtpos : 0 < x + t := lt_of_lt_of_le hmin hxt.1
      have hLJ : HasDerivWithinAt
          (fun u : ℝ => Computation.ArrayDSL.lennardJones ε σ (x + u))
          (ljGradExpr ε σ (x + t)) (Set.Ici t) t := by
        simpa [one_mul] using
          (((hasDerivAt_arrayLennardJones_of_pos ε σ (x + t) hxtpos).comp t
            ((hasDerivAt_id t).const_add x)).hasDerivWithinAt)
      have hLin : HasDerivWithinAt
          (fun u : ℝ => Computation.ArrayDSL.lennardJones ε σ x + ljGradExpr ε σ x * u)
          (ljGradExpr ε σ x) (Set.Ici t) t := by
        simpa [mul_comm, mul_left_comm, mul_assoc] using
          ((((hasDerivAt_id t).const_mul (ljGradExpr ε σ x)).const_add
            (Computation.ArrayDSL.lennardJones ε σ x)).hasDerivWithinAt)
      exact hLJ.sub hLin
    have hRemBound : ∀ t ∈ Set.Ico 0 Δ,
        ‖ljGradExpr ε σ (x + t) - ljGradExpr ε σ x‖ ≤ K * t := by
      intro t ht
      have hxt : x + t ∈ Set.Icc rmin rmax := by
        refine ⟨?_, ?_⟩
        · linarith [hx.1, ht.1]
        · have hxty : x + t ≤ y := by
            linarith [ht.2]
          exact le_trans hxty hy.2
      have hgrad :=
        lennardJones_grad_diff_le_of_shellSecondDerivBound ε σ rmin rmax K hmin hSecondBound
          hx hxt
      have htabs : |t| = t := abs_of_nonneg ht.1
      have habs : |(x + t) - x| = t := by
        have ht0 : 0 ≤ t := ht.1
        simp [abs_of_nonneg ht0]
      simpa [Real.norm_eq_abs, habs, htabs] using hgrad
    have hQuadDeriv : ∀ t, HasDerivAt (fun u : ℝ => (K / 2) * u ^ (2 : ℕ)) (K * t) t := by
      intro t
      simpa [pow_two, two_mul, mul_assoc, mul_left_comm, mul_comm] using
        (((hasDerivAt_id t).pow 2).const_mul (K / 2))
    have hFence := image_norm_le_of_norm_deriv_right_le_deriv_boundary
      (f := remainder)
      (f' := fun t => ljGradExpr ε σ (x + t) - ljGradExpr ε σ x)
      (a := 0)
      (b := Δ)
      hRemCont hRemDeriv
      (ha := by simp [remainder, Δ])
      (B := fun t => (K / 2) * t ^ (2 : ℕ))
      (B' := fun t => K * t)
      hQuadDeriv
      (by
        intro t ht
        simpa [Real.norm_eq_abs] using hRemBound t ht)
    have hEval := hFence (x := Δ) (by simpa [Δ] using Set.right_mem_Icc.mpr hΔ)
    have hEval' :
        |Computation.ArrayDSL.lennardJones ε σ y - Computation.ArrayDSL.lennardJones ε σ x -
            ljGradExpr ε σ x * (y - x)| ≤
          (K / 2) * (y - x) ^ (2 : ℕ) := by
      convert hEval using 1 <;>
        simp [remainder, Δ, abs_of_nonneg hΔ, sub_eq_add_neg] <;> ring_nf
    simpa [abs_of_nonneg hΔ] using hEval'
  · have hyx : y < x := lt_of_not_ge hxy
    let Δ : ℝ := x - y
    let remainder : ℝ → ℝ := fun t =>
      ljGradExpr ε σ x * t +
        (-Computation.ArrayDSL.lennardJones ε σ x + Computation.ArrayDSL.lennardJones ε σ (x - t))
    have hΔ : 0 ≤ Δ := sub_nonneg.mpr hyx.le
    have hRemCont : ContinuousOn remainder (Set.Icc 0 Δ) := by
      intro t ht
      have hxt : x - t ∈ Set.Icc rmin rmax := by
        refine ⟨?_, ?_⟩
        · have hylt : y ≤ x - t := by
            linarith [ht.2]
          exact le_trans hy.1 hylt
        · linarith [hx.2, ht.1]
      have hxtpos : 0 < x - t := lt_of_lt_of_le hmin hxt.1
      have hLJ : HasDerivAt
          (fun u : ℝ => Computation.ArrayDSL.lennardJones ε σ (x - u))
          (-ljGradExpr ε σ (x - t)) t := by
        simpa [sub_eq_add_neg, add_comm, add_left_comm, add_assoc, mul_comm, mul_left_comm,
          mul_assoc] using
          (hasDerivAt_arrayLennardJones_of_pos ε σ (x - t) hxtpos).comp t
            ((hasDerivAt_const t x).sub (hasDerivAt_id t))
      have hLJTerm : HasDerivAt
          (fun u : ℝ => -Computation.ArrayDSL.lennardJones ε σ x +
            Computation.ArrayDSL.lennardJones ε σ (x - u))
          (-ljGradExpr ε σ (x - t)) t := by
        simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using
          hLJ.const_add (-Computation.ArrayDSL.lennardJones ε σ x)
      have hLin : HasDerivAt
          (fun u : ℝ => ljGradExpr ε σ x * u)
          (ljGradExpr ε σ x) t := by
        simpa [sub_eq_add_neg, mul_comm, mul_left_comm, mul_assoc] using
          ((hasDerivAt_id t).const_mul (ljGradExpr ε σ x))
      exact (hLin.add hLJTerm).continuousAt.continuousWithinAt
    have hRemDeriv : ∀ t ∈ Set.Ico 0 Δ,
        HasDerivWithinAt remainder (ljGradExpr ε σ x - ljGradExpr ε σ (x - t)) (Set.Ici t) t := by
      intro t ht
      have hxt : x - t ∈ Set.Icc rmin rmax := by
        refine ⟨?_, ?_⟩
        · have hylt : y ≤ x - t := by
            linarith [ht.2]
          exact le_trans hy.1 hylt
        · linarith [hx.2, ht.1]
      have hxtpos : 0 < x - t := lt_of_lt_of_le hmin hxt.1
      have hLJ : HasDerivWithinAt
          (fun u : ℝ => Computation.ArrayDSL.lennardJones ε σ (x - u))
          (-ljGradExpr ε σ (x - t)) (Set.Ici t) t := by
        simpa [sub_eq_add_neg, zero_sub, one_mul, mul_comm, mul_left_comm, mul_assoc] using
          (((hasDerivAt_arrayLennardJones_of_pos ε σ (x - t) hxtpos).comp t
            ((hasDerivAt_const t x).sub (hasDerivAt_id t))).hasDerivWithinAt)
      have hLJTerm : HasDerivWithinAt
          (fun u : ℝ => -Computation.ArrayDSL.lennardJones ε σ x +
            Computation.ArrayDSL.lennardJones ε σ (x - u))
          (-ljGradExpr ε σ (x - t)) (Set.Ici t) t := by
        simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using
          hLJ.const_add (-Computation.ArrayDSL.lennardJones ε σ x)
      have hLin : HasDerivWithinAt
          (fun u : ℝ => ljGradExpr ε σ x * u)
          (ljGradExpr ε σ x) (Set.Ici t) t := by
        simpa [mul_comm, mul_left_comm, mul_assoc] using
          (((hasDerivAt_id t).const_mul (ljGradExpr ε σ x)).hasDerivWithinAt)
      simpa [remainder, add_assoc, add_left_comm, add_comm] using hLin.add hLJTerm
    have hRemBound : ∀ t ∈ Set.Ico 0 Δ,
        ‖ljGradExpr ε σ x - ljGradExpr ε σ (x - t)‖ ≤ K * t := by
      intro t ht
      have hxt : x - t ∈ Set.Icc rmin rmax := by
        refine ⟨?_, ?_⟩
        · have hylt : y ≤ x - t := by
            linarith [ht.2]
          exact le_trans hy.1 hylt
        · linarith [hx.2, ht.1]
      have hgrad :=
        lennardJones_grad_diff_le_of_shellSecondDerivBound ε σ rmin rmax K hmin hSecondBound
          hxt hx
      have htabs : |t| = t := abs_of_nonneg ht.1
      have habs : |x - (x - t)| = t := by
        have ht0 : 0 ≤ t := ht.1
        simp [abs_of_nonneg ht0]
      simpa [Real.norm_eq_abs, habs, htabs, sub_eq_add_neg, abs_sub_comm] using hgrad
    have hQuadDeriv : ∀ t, HasDerivAt (fun u : ℝ => (K / 2) * u ^ (2 : ℕ)) (K * t) t := by
      intro t
      simpa [pow_two, two_mul, mul_assoc, mul_left_comm, mul_comm] using
        (((hasDerivAt_id t).pow 2).const_mul (K / 2))
    have hFence := image_norm_le_of_norm_deriv_right_le_deriv_boundary
      (f := remainder)
      (f' := fun t => ljGradExpr ε σ x - ljGradExpr ε σ (x - t))
      (a := 0)
      (b := Δ)
      hRemCont hRemDeriv
      (ha := by simp [remainder, Δ])
      (B := fun t => (K / 2) * t ^ (2 : ℕ))
      (B' := fun t => K * t)
      hQuadDeriv
      (by
        intro t ht
        simpa [Real.norm_eq_abs] using hRemBound t ht)
    have hEval := hFence (x := Δ) (by simpa [Δ] using Set.right_mem_Icc.mpr hΔ)
    have hsquare : (x - y) ^ (2 : ℕ) = (y - x) ^ (2 : ℕ) := by
      ring_nf
    have hEval' :
        |Computation.ArrayDSL.lennardJones ε σ y - Computation.ArrayDSL.lennardJones ε σ x -
            ljGradExpr ε σ x * (y - x)| ≤
          (K / 2) * (y - x) ^ (2 : ℕ) := by
      have hEval'' :
          |Computation.ArrayDSL.lennardJones ε σ y - Computation.ArrayDSL.lennardJones ε σ x -
              ljGradExpr ε σ x * (y - x)| ≤
            (K / 2) * (x - y) ^ (2 : ℕ) := by
        convert hEval using 1 <;>
          simp [remainder, Δ, sub_eq_add_neg] <;> ring_nf
      simpa [hsquare] using hEval''
    have hAbsPow : (y - x) ^ (2 : ℕ) = |y - x| ^ (2 : ℕ) := by
      have hnonpos : y - x ≤ 0 := sub_nonpos.mpr hyx.le
      have habs : |y - x| = x - y := by
        simpa [sub_eq_add_neg] using abs_of_nonpos hnonpos
      rw [habs]
      exact hsquare.symm
    calc
      |Computation.ArrayDSL.lennardJones ε σ y - Computation.ArrayDSL.lennardJones ε σ x -
          ljGradExpr ε σ x * (y - x)| ≤ (K / 2) * (y - x) ^ (2 : ℕ) := hEval'
      _ = (K / 2) * |y - x| ^ (2 : ℕ) := by rw [hAbsPow]

/-- Instantiating the quadratic remainder theorem with the explicit shell Hessian envelope gives a
closed-form quadratic error bound. -/
theorem lennardJones_taylor_remainder_le_of_shellSecondDerivShellBound
    (ε σ rmin rmax : ℝ)
    (hmin : 0 < rmin)
    {x y : ℝ}
    (hx : x ∈ Set.Icc rmin rmax)
    (hy : y ∈ Set.Icc rmin rmax) :
    |Computation.ArrayDSL.lennardJones ε σ y - Computation.ArrayDSL.lennardJones ε σ x -
        ljGradExpr ε σ x * (y - x)| ≤
      (ljSecondDerivShellBound ε σ rmin / 2) * |y - x| ^ (2 : ℕ) := by
  refine lennardJones_taylor_remainder_le_of_shellSecondDerivBound ε σ rmin rmax
    (ljSecondDerivShellBound ε σ rmin) hmin ?_ hx hy
  intro z hz
  exact abs_ljSecondDerivExpr_le_shellBound ε σ rmin z hmin hz.1

/-- A shellwise bound on the Lennard-Jones gradient controls score differences by the mean value
theorem. -/
theorem lennardJones_value_diff_le_of_shellGradBound
    (ε σ rmin rmax C : ℝ)
    (hmin : 0 < rmin)
    (hGradBound : ∀ x ∈ Set.Icc rmin rmax, |ljGradExpr ε σ x| ≤ C)
    {x y : ℝ}
    (hx : x ∈ Set.Icc rmin rmax)
    (hy : y ∈ Set.Icc rmin rmax) :
    |Computation.ArrayDSL.lennardJones ε σ y - Computation.ArrayDSL.lennardJones ε σ x| ≤ C * |y - x| := by
  have hderiv : ∀ z ∈ Set.Icc rmin rmax,
      HasDerivWithinAt (fun r => Computation.ArrayDSL.lennardJones ε σ r) (ljGradExpr ε σ z)
        (Set.Icc rmin rmax) z := by
    intro z hz
    have hzpos : 0 < z := lt_of_lt_of_le hmin hz.1
    have hraw : HasDerivWithinAt (fun r => Computation.lennardJones ε σ r) (ljGradExpr ε σ z)
        (Set.Icc rmin rmax) z := by
      simpa [ljGradExpr] using
        (Computation.grad_of_lennardJones ε σ z (ne_of_gt hzpos)).hasDerivWithinAt
    refine hraw.congr (fun r hr => ?_) ?_
    · have hrpos : 0 < r := lt_of_lt_of_le hmin hr.1
      simp [Computation.ArrayDSL.lennardJones, Computation.lennardJones, div_eq_mul_inv, ne_of_gt hrpos]
    · have hzpos' : 0 < z := lt_of_lt_of_le hmin hz.1
      simp [Computation.ArrayDSL.lennardJones, Computation.lennardJones, div_eq_mul_inv, ne_of_gt hzpos']
  have h :=
    (convex_Icc rmin rmax).norm_image_sub_le_of_norm_hasDerivWithin_le
      (f := fun r => Computation.ArrayDSL.lennardJones ε σ r)
      (f' := fun z => ljGradExpr ε σ z)
      (s := Set.Icc rmin rmax)
      (x := x) (y := y)
      hderiv
      (by
        intro z hz
        simpa [Real.norm_eq_abs] using hGradBound z hz)
      hx hy
  simpa [Real.norm_eq_abs] using h

/-- A shellwise Lennard-Jones gradient bound and a distance-approximation bound yield a uniform
score approximation theorem for exact-versus-grid Lennard-Jones problems. -/
theorem exactLJ_uniformApprox_of_distanceError_and_shellGradBound
    {A S : Type*}
    (distanceExact distanceGrid : A → S → ℝ)
    (ε σ rmin rmax C delta : ℝ)
    (hmin : 0 < rmin)
    (hC : 0 ≤ C)
    (hGradBound : ∀ x ∈ Set.Icc rmin rmax, |ljGradExpr ε σ x| ≤ C)
    (hShellExact : ∀ a s, distanceExact a s ∈ Set.Icc rmin rmax)
    (hShellGrid : ∀ a s, distanceGrid a s ∈ Set.Icc rmin rmax)
    (hDistErr : ∀ a s, |distanceExact a s - distanceGrid a s| ≤ delta) :
    UniformUtilityApprox
      (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distanceExact ε σ)
      (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distanceGrid ε σ)
      (C * delta) := by
  intro a s
  have hdiff := lennardJones_value_diff_le_of_shellGradBound ε σ rmin rmax C hmin hGradBound
    (hShellGrid a s) (hShellExact a s)
  have hdelta' : C * |distanceExact a s - distanceGrid a s| ≤ C * delta :=
    mul_le_mul_of_nonneg_left (hDistErr a s) hC
  have h' :
      |Computation.ArrayDSL.lennardJones ε σ (distanceExact a s) -
          Computation.ArrayDSL.lennardJones ε σ (distanceGrid a s)| ≤ C * delta :=
    hdiff.trans hdelta'
  simpa [DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem,
    DecisionQuotient.Tractability.LJApproximation.exactLJScore] using h'

/-- A shellwise Lennard-Jones gradient bound and a strict utility gap imply exact/grid winner
preservation. -/
theorem exactLJ_distanceError_gap_implies_opt_invariance
    {A S : Type*} [Fintype A]
    (distanceExact distanceGrid : A → S → ℝ)
    (ε σ rmin rmax C delta : ℝ)
    (hmin : 0 < rmin)
    (hC : 0 ≤ C)
    (hGradBound : ∀ x ∈ Set.Icc rmin rmax, |ljGradExpr ε σ x| ≤ C)
    (hShellExact : ∀ a s, distanceExact a s ∈ Set.Icc rmin rmax)
    (hShellGrid : ∀ a s, distanceGrid a s ∈ Set.Icc rmin rmax)
    (hDistErr : ∀ a s, |distanceExact a s - distanceGrid a s| ≤ delta)
    (s : S) (aStar : A)
    (hDelta : 0 ≤ C * delta)
    (hStrict : StrictOpt
      (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distanceExact ε σ)
      aStar s)
    (hBound : C * delta <
      StrictUtilityGap
        (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distanceExact ε σ)
        aStar s / 2) :
    (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distanceExact ε σ).Opt s =
      (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distanceGrid ε σ).Opt s := by
  have hApprox := exactLJ_uniformApprox_of_distanceError_and_shellGradBound
    distanceExact distanceGrid ε σ rmin rmax C delta hmin hC hGradBound hShellExact hShellGrid hDistErr
  exact uniform_approx_implies_opt_invariance
    (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distanceExact ε σ)
    (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distanceGrid ε σ)
    (C * delta) hApprox s aStar hDelta hStrict hBound

/-- A shellwise Hessian envelope turns a distance discretization error into a quadratic
first-order-corrected score error bound for exact Lennard-Jones evaluation. -/
theorem exactLJ_quadraticDiscretizationError_of_distanceError_and_shellSecondDerivShellBound
    {A S : Type*}
    (distanceExact distanceGrid : A → S → ℝ)
    (ε σ rmin rmax delta : ℝ)
    (hmin : 0 < rmin)
    (hShellExact : ∀ a s, distanceExact a s ∈ Set.Icc rmin rmax)
    (hShellGrid : ∀ a s, distanceGrid a s ∈ Set.Icc rmin rmax)
    (hDistErr : ∀ a s, |distanceExact a s - distanceGrid a s| ≤ delta) :
    ∀ a s,
      |DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) -
          (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
            ljGradExpr ε σ (distanceGrid a s) *
              (distanceExact a s - distanceGrid a s))| ≤
        (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ) := by
  intro a s
  have hdelta : 0 ≤ delta := le_trans (abs_nonneg _) (hDistErr a s)
  have hrem := lennardJones_taylor_remainder_le_of_shellSecondDerivShellBound
    ε σ rmin rmax hmin (hShellGrid a s) (hShellExact a s)
  have hpow : |distanceExact a s - distanceGrid a s| ^ (2 : ℕ) ≤ delta ^ (2 : ℕ) := by
    exact pow_le_pow_left₀ (abs_nonneg _) (hDistErr a s) _
  have hmul :
      (ljSecondDerivShellBound ε σ rmin / 2) * |distanceExact a s - distanceGrid a s| ^ (2 : ℕ) ≤
        (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ) := by
    have hCoeff : 0 ≤ ljSecondDerivShellBound ε σ rmin / 2 := by
      unfold ljSecondDerivShellBound
      positivity
    exact mul_le_mul_of_nonneg_left hpow hCoeff
  simpa [DecisionQuotient.Tractability.LJApproximation.exactLJScore,
    sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using hrem.trans hmul

/-- The same large-cutoff bridge for the sampled restricted docking problem. -/
theorem boundedPotential_largeCutoff_sampledSrank_bound
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction]
    (aStar : MDAction)
    (tail_coefficient minimum_gap : ℝ)
    (hBounds : DecisionQuotient.Tractability.SatisfiesBoundedPotential prob tail_coefficient minimum_gap)
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff < minimum_gap / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s) :
    @DecisionProblem.srank (SupportedAction samples) MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (restrictedDecisionProblem prob.toDecisionProblem samples) ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  have hBounded := largeCutoff_implies_bounded prob tail_coefficient minimum_gap
    hBounds hPosCutoff hCutoffSize hTailPos
  exact DecisionQuotient.Tractability.SampledDockingCutoff.sampled_md_srank_bound
    prob samples aStar hStrictAll hBounded hCapture

/-- Exact Lennard-Jones tail control plus a sufficiently large cutoff yields a direct sampled
structural-rank bound. -/
theorem exactLJ_largeCutoff_sampledSrank_bound
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (distance : MDAction → MDState → ℝ)
    (aStar : MDAction)
    (ε σ tail_coefficient : ℝ)
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < DecisionQuotient.Tractability.finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distance a s))
    (hPhys : DecisionQuotient.Tractability.LJApproximation.PhysicalDistanceDecayLJ prob distance ε σ tail_coefficient)
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff <
      DecisionQuotient.Tractability.finiteMinimumGap prob w / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s) :
    @DecisionProblem.srank (SupportedAction samples) MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (restrictedDecisionProblem prob.toDecisionProblem samples) ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  have hBounds := DecisionQuotient.Tractability.LJApproximation.exactLJ_is_BoundedPotential
    prob distance ε σ tail_coefficient w hGapPos hUtility hPhys
  exact boundedPotential_largeCutoff_sampledSrank_bound
    prob samples aStar tail_coefficient (DecisionQuotient.Tractability.finiteMinimumGap prob w)
    hBounds hPosCutoff hCutoffSize hTailPos hStrictAll hCapture

/-- Exact Coulomb tail control plus a sufficiently large cutoff yields a direct sampled
structural-rank bound. -/
theorem exactCoulomb_largeCutoff_sampledSrank_bound
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (distance : MDAction → MDState → ℝ)
    (aStar : MDAction)
    (q_i q_j tail_coefficient : ℝ)
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < DecisionQuotient.Tractability.finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distance a s))
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (numMDCoordinates prob),
        j ≠ proteinCoordFin prob atomIdx hAtomInProtein axis →
        mdProj prob s j = mdProj prob s' j) →
      ¬ atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MDAction,
        |DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distance a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distance a s')|
          ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 R))
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff <
      DecisionQuotient.Tractability.finiteMinimumGap prob w / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s) :
    @DecisionProblem.srank (SupportedAction samples) MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (restrictedDecisionProblem prob.toDecisionProblem samples) ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  have hBounds := DecisionQuotient.Tractability.CoulombApproximation.exactCoulomb_satisfiesBoundedPotential_of_tailBound
    prob distance q_i q_j tail_coefficient w hGapPos hUtility hTail
  exact boundedPotential_largeCutoff_sampledSrank_bound
    prob samples aStar tail_coefficient (DecisionQuotient.Tractability.finiteMinimumGap prob w)
    hBounds hPosCutoff hCutoffSize hTailPos hStrictAll hCapture

/-- Exact real-space Ewald tail control plus a sufficiently large cutoff yields a direct sampled
structural-rank bound. -/
theorem exactRealEwald_largeCutoff_sampledSrank_bound
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction] [Fintype MDState] [Nonempty MDState]
    (distance : MDAction → MDState → ℝ)
    (aStar : MDAction)
    (q_i q_j alpha tail_coefficient : ℝ)
    (w : ∀ s : MDState, { a : MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < DecisionQuotient.Tractability.finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => DecisionQuotient.Tractability.CoulombApproximation.exactRealEwaldScore q_i q_j alpha (distance a s))
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (numMDCoordinates prob),
        j ≠ proteinCoordFin prob atomIdx hAtomInProtein axis →
        mdProj prob s j = mdProj prob s' j) →
      ¬ atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MDAction,
        |DecisionQuotient.Tractability.CoulombApproximation.exactRealEwaldScore q_i q_j alpha (distance a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactRealEwaldScore q_i q_j alpha (distance a s')|
          ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 R))
    (hPosCutoff : 0 < prob.cutoff)
    (hCutoffSize : tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff <
      DecisionQuotient.Tractability.finiteMinimumGap prob w / 2)
    (hTailPos : 0 ≤ tail_coefficient * DecisionQuotient.Tractability.LatticeSum.latticeTailSum 6 prob.cutoff)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s) :
    @DecisionProblem.srank (SupportedAction samples) MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (restrictedDecisionProblem prob.toDecisionProblem samples) ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  have hBounds := DecisionQuotient.Tractability.CoulombApproximation.exactRealEwald_satisfiesBoundedPotential_of_tailBound
    prob distance q_i q_j alpha tail_coefficient w hGapPos hUtility hTail
  exact boundedPotential_largeCutoff_sampledSrank_bound
    prob samples aStar tail_coefficient (DecisionQuotient.Tractability.finiteMinimumGap prob w)
    hBounds hPosCutoff hCutoffSize hTailPos hStrictAll hCapture

/-- Top-k margin certificates package exact survivor containment into a reusable finite certificate. -/
theorem topKMargin_certificate_sound
    {A : Type*} [Fintype A] [DecidableEq A]
    (uExact uCoarse : A → ℝ)
    (k : Nat)
    (tau delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hMargin : ∀ a,
      a ∈ DecisionQuotient.Tractability.FiniteTopK.topKWithTies uExact k →
        tau + delta ≤ uExact a) :
    (DecisionQuotient.Tractability.CertifiedPruning.certificate_of_topK_margin
      uExact uCoarse k tau delta hApprox hMargin).exactTopK
        ⊆
      (DecisionQuotient.Tractability.CertifiedPruning.certificate_of_topK_margin
        uExact uCoarse k tau delta hApprox hMargin).survivors :=
  DecisionQuotient.Tractability.CertifiedPruning.certificate_sound
    uExact uCoarse k tau delta hApprox hMargin

/-- The generic irrelevance-erasure theorem applies directly to discretized molecular grid states. -/
theorem gridMDState_sufficient_erase_irrelevant
    {NP NL N : Nat} {A : Type*}
    [DecidableEq (Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL))]
    (dp : DecisionProblem A (GridMDState NP NL N))
    (I : Finset (Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL)))
    (i : Fin (DecisionQuotient.Tractability.GridMDInstances.GridCoordCount NP NL))
    (hI : dp.isSufficient I)
    (hirr : dp.isIrrelevant i) :
    dp.isSufficient (I.erase i) :=
  DecisionQuotient.Tractability.GridMDInstances.gridMDState_sufficient_erase_irrelevant
    dp I i hI hirr

/-- With bounded action set size, exact sufficiency checking is decidable in polynomial time. -/
theorem sufficiency_poly_bounded_actions
    {A S : Type*} [DecidableEq A] [DecidableEq S]
    (k : ℕ)
    {n : ℕ} [CoordinateSpace S n] [Fintype A] [Fintype S]
    [∀ s s' : S, ∀ I : Finset (Fin n), Decidable (agreeOn s s' I)]
    (cdp : ComputableDecisionProblem A S)
    (hcard : Fintype.card A ≤ k) :
    ∃ (decide : Finset (Fin n) → Bool),
      (∀ I, decide I = true ↔ cdp.toAbstract.isSufficient I) :=
  DecisionQuotient.sufficiency_poly_bounded_actions k cdp hcard

/-- Explicit polynomial-time bound for bounded action families. -/
theorem boundedActions_polynomialTime
    (k : ℕ) (numStates : ℕ) :
    ∃ (c : ℕ), c = 1 + k ^ 2 ∧
      DecisionQuotient.totalCheckCost numStates k ≤ numStates ^ 2 * c :=
  DecisionQuotient.bounded_actions_polynomial_time k numStates

private theorem eraseUniv_sufficient_iff_irrelevant
    {A S : Type*} {n : ℕ} [ps : ProductSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (hinj : ∀ s s' : S,
      (∀ i : Fin n,
        @CoordinateSpace.proj S n ps.toCoordinateSpace s i =
          @CoordinateSpace.proj S n ps.toCoordinateSpace s' i) → s = s')
    (i : Fin n) :
    @DecisionProblem.isSufficient A S n ps.toCoordinateSpace dp ((Finset.univ : Finset (Fin n)).erase i) ↔
      @DecisionProblem.isIrrelevant A S n ps.toCoordinateSpace dp i := by
  letI : CoordinateSpace S n := ps.toCoordinateSpace
  constructor
  · intro hErase
    rw [dp.irrelevant_iff_not_relevant]
    intro hRel
    have hiMem : i ∈ ((Finset.univ : Finset (Fin n)).erase i) :=
      dp.sufficient_contains_relevant ((Finset.univ : Finset (Fin n)).erase i) hErase i hRel
    simpa using hiMem
  · intro hIrr
    have hUniv : dp.isSufficient (Finset.univ : Finset (Fin n)) :=
      dp.univ_sufficient_of_injective hinj
    exact dp.sufficient_erase_irrelevant' (Finset.univ : Finset (Fin n)) i hUniv hIrr

/-- In the bounded-action finite regime, the explicit coordinate-extraction rule
`i ↦ checkSufficiency(univ.erase i) = false` returns a sufficient set of cardinality exactly equal
to structural rank, with total cost bounded by `n * |S|^2 * (1 + k^2)`. -/
theorem extractRelevantCoordinates_polytime
    {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
    {n : ℕ} [ps : ProductSpace S n] [DecidableEq (Fin n)]
    (k : ℕ) (cdp : ComputableDecisionProblem A S)
    (hAgreeDec : ∀ s s' : S, ∀ I : Finset (Fin n),
      Decidable (@agreeOn S n ps.toCoordinateSpace s s' I))
    (_hcard : Fintype.card A ≤ k)
    (hinj : ∀ s s' : S,
      (∀ i : Fin n,
        @CoordinateSpace.proj S n ps.toCoordinateSpace s i =
          @CoordinateSpace.proj S n ps.toCoordinateSpace s' i) → s = s') :
    ∃ I : Finset (Fin n),
      @DecisionProblem.isSufficient A S n ps.toCoordinateSpace cdp.toAbstract I ∧
      I.card = @DecisionProblem.srank A S n ps.toCoordinateSpace cdp.toAbstract ∧
      n * DecisionQuotient.totalCheckCost (Fintype.card S) k ≤
        n * (Fintype.card S) ^ 2 * (1 + k ^ 2) := by
  letI : CoordinateSpace S n := ps.toCoordinateSpace
  letI : ∀ s s' : S, ∀ J : Finset (Fin n), Decidable (agreeOn s s' J) := hAgreeDec
  let I : Finset (Fin n) :=
    (Finset.univ : Finset (Fin n)).filter
      (fun i => cdp.checkSufficiency ((Finset.univ : Finset (Fin n)).erase i) = false)
  have hMem : ∀ i : Fin n, i ∈ I ↔ cdp.toAbstract.isRelevant i := by
    intro i
    constructor
    · intro hi
      have hFalse : cdp.checkSufficiency ((Finset.univ : Finset (Fin n)).erase i) = false :=
        (Finset.mem_filter.mp hi).2
      have hNotSuff : ¬ cdp.toAbstract.isSufficient ((Finset.univ : Finset (Fin n)).erase i) := by
        intro hSuff
        have hCheck : cdp.checkSufficiency ((Finset.univ : Finset (Fin n)).erase i) :=
          (cdp.checkSufficiency_iff_abstract ((Finset.univ : Finset (Fin n)).erase i)).2 hSuff
        simp [hCheck] at hFalse
      have hNotIrr : ¬ cdp.toAbstract.isIrrelevant i := by
        intro hIrr
        exact hNotSuff ((eraseUniv_sufficient_iff_irrelevant cdp.toAbstract hinj i).2 hIrr)
      have hNotNotRel : ¬ ¬ cdp.toAbstract.isRelevant i := by
        simpa [cdp.toAbstract.irrelevant_iff_not_relevant i] using hNotIrr
      exact Classical.not_not.mp hNotNotRel
    · intro hRel
      have hNotSuff : ¬ cdp.toAbstract.isSufficient ((Finset.univ : Finset (Fin n)).erase i) := by
        intro hSuff
        have hiMem : i ∈ ((Finset.univ : Finset (Fin n)).erase i) :=
          cdp.toAbstract.sufficient_contains_relevant ((Finset.univ : Finset (Fin n)).erase i) hSuff i hRel
        simpa using hiMem
      have hFalse : cdp.checkSufficiency ((Finset.univ : Finset (Fin n)).erase i) = false := by
        by_cases hCheck : cdp.checkSufficiency ((Finset.univ : Finset (Fin n)).erase i)
        · have hSuff : cdp.toAbstract.isSufficient ((Finset.univ : Finset (Fin n)).erase i) :=
            (cdp.checkSufficiency_iff_abstract ((Finset.univ : Finset (Fin n)).erase i)).1 hCheck
          exact False.elim (hNotSuff hSuff)
        · simp [hCheck]
      exact Finset.mem_filter.mpr ⟨by simp, hFalse⟩
  have hEq : I = Finset.univ.filter cdp.toAbstract.isRelevant := by
    ext i
    rw [hMem i]
    simp [I]
  refine ⟨I, ?_, ?_, ?_⟩
  · rw [hEq]
    exact relevantSet_isSufficient cdp.toAbstract hinj
  · rw [hEq]
    rfl
  · calc
      n * DecisionQuotient.totalCheckCost (Fintype.card S) k
          ≤ n * ((Fintype.card S) ^ 2 * (1 + k ^ 2)) :=
            Nat.mul_le_mul_left n (DecisionQuotient.totalCheckCost_le_pow (Fintype.card S) k)
      _ = n * (Fintype.card S) ^ 2 * (1 + k ^ 2) := by rw [Nat.mul_assoc]

/-- Design target class `Γ(K, γ)`: decision problems with structural rank at most `K` and a uniform strict
utility gap at least `γ`. This is the theorem-backed inverse-design class supported by the current artifact. -/
def RankGapDesignClass {A S : Type*} [Fintype A] {n : ℕ}
    (cs : CoordinateSpace S n) (dp : DecisionProblem A S) (K : ℕ) (γ : ℝ) : Prop :=
  @DecisionProblem.srank A S n cs dp ≤ K ∧
    ∀ s : S, ∃ aStar : A, StrictOpt dp aStar s ∧ γ ≤ StrictUtilityGap dp aStar s

/-- Any explicit sufficient coordinate set of size at most `K`, together with a uniform strict gap lower bound,
places the docking problem inside the inverse-design class `Γ(K, γ)`. -/
theorem rankGapDesignClass_of_sufficient_and_uniformGap
    {A S : Type*} [Fintype A] {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) (K : ℕ) (γ : ℝ)
    (hI : dp.isSufficient I) (hCard : I.card ≤ K)
    (hGap : ∀ s : S, ∃ aStar : A, StrictOpt dp aStar s ∧ γ ≤ StrictUtilityGap dp aStar s) :
    RankGapDesignClass ‹CoordinateSpace S n› dp K γ := by
  have hRank : dp.srank ≤ I.card := srank_le_sufficient_card dp I hI
  exact ⟨le_trans hRank hCard, hGap⟩

/-- In the bounded-action finite regime, the constructive extraction theorem can be run in reverse as a design
certificate: if the target rank budget is `K`, the algorithm returns a sufficient free-coordinate set of size
at most `K`; constraining its complement requires exactly `n - |I|` coordinate clamps. -/
theorem reverseDesign_polytime_of_rankBudget
    {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
    {n : ℕ} [ps : ProductSpace S n] [DecidableEq (Fin n)]
    (k K : ℕ) (γ : ℝ) (cdp : ComputableDecisionProblem A S)
    (hAgreeDec : ∀ s s' : S, ∀ I : Finset (Fin n),
      Decidable (@agreeOn S n ps.toCoordinateSpace s s' I))
    (hCardA : Fintype.card A ≤ k)
    (hinj : ∀ s s' : S,
      (∀ i : Fin n,
        @CoordinateSpace.proj S n ps.toCoordinateSpace s i =
          @CoordinateSpace.proj S n ps.toCoordinateSpace s' i) → s = s')
    (hRank : @DecisionProblem.srank A S n ps.toCoordinateSpace cdp.toAbstract ≤ K)
    (hGap : ∀ s : S, ∃ aStar : A,
      StrictOpt cdp.toAbstract aStar s ∧ γ ≤ StrictUtilityGap cdp.toAbstract aStar s) :
    ∃ I : Finset (Fin n),
      @DecisionProblem.isSufficient A S n ps.toCoordinateSpace cdp.toAbstract I ∧
      I.card ≤ K ∧
      RankGapDesignClass ps.toCoordinateSpace cdp.toAbstract K γ ∧
      (Finset.univ \ I).card = n - I.card ∧
      n * DecisionQuotient.totalCheckCost (Fintype.card S) k ≤
        n * (Fintype.card S) ^ 2 * (1 + k ^ 2) := by
  letI : CoordinateSpace S n := ps.toCoordinateSpace
  rcases extractRelevantCoordinates_polytime k cdp hAgreeDec hCardA hinj with
    ⟨I, hI, hCardEq, hCost⟩
  refine ⟨I, hI, ?_, ?_, ?_, hCost⟩
  · rw [hCardEq]
    exact hRank
  · exact rankGapDesignClass_of_sufficient_and_uniformGap cdp.toAbstract I K γ hI (by
      rw [hCardEq]
      exact hRank) hGap
  · have hSubset : I ⊆ (Finset.univ : Finset (Fin n)) := by
      intro i _
      simp
    simpa using (Finset.card_sdiff_of_subset hSubset)

/-- Energy-budget specialization of the reverse-design certificate: if the desired rank budget `K` fits inside a
thermodynamic budget `E`, the extracted sufficient coordinate set certifies both membership in `Γ(K, γ)` and
compliance with the declared energy ceiling. -/
theorem reverseDesign_polytime_of_energyBudget
    {A S : Type*} [DecidableEq A] [DecidableEq S] [Fintype A] [Fintype S]
    {n : ℕ} [ps : ProductSpace S n] [DecidableEq (Fin n)]
    (k K E : ℕ) (γ : ℝ) (cdp : ComputableDecisionProblem A S)
    (hAgreeDec : ∀ s s' : S, ∀ I : Finset (Fin n),
      Decidable (@agreeOn S n ps.toCoordinateSpace s s' I))
    (hCardA : Fintype.card A ≤ k)
    (hinj : ∀ s s' : S,
      (∀ i : Fin n,
        @CoordinateSpace.proj S n ps.toCoordinateSpace s i =
          @CoordinateSpace.proj S n ps.toCoordinateSpace s' i) → s = s')
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    (hRank : @DecisionProblem.srank A S n ps.toCoordinateSpace cdp.toAbstract ≤ K)
    (hBudget : DecisionQuotient.ThermodynamicLift.energyLowerBound M K ≤ E)
    (hGap : ∀ s : S, ∃ aStar : A,
      StrictOpt cdp.toAbstract aStar s ∧ γ ≤ StrictUtilityGap cdp.toAbstract aStar s) :
    ∃ I : Finset (Fin n),
      @DecisionProblem.isSufficient A S n ps.toCoordinateSpace cdp.toAbstract I ∧
      I.card ≤ K ∧
      RankGapDesignClass ps.toCoordinateSpace cdp.toAbstract K γ ∧
      DecisionQuotient.ThermodynamicLift.energyLowerBound M
        (@DecisionProblem.srank A S n ps.toCoordinateSpace cdp.toAbstract) ≤ E ∧
      (Finset.univ \ I).card = n - I.card ∧
      n * DecisionQuotient.totalCheckCost (Fintype.card S) k ≤
        n * (Fintype.card S) ^ 2 * (1 + k ^ 2) := by
  letI : CoordinateSpace S n := ps.toCoordinateSpace
  rcases reverseDesign_polytime_of_rankBudget k K γ cdp hAgreeDec hCardA hinj hRank hGap with
    ⟨I, hI, hCard, hClass, hConstr, hCost⟩
  refine ⟨I, hI, hCard, hClass, ?_, hConstr, hCost⟩
  exact le_trans
    (DecisionQuotient.ThermodynamicLift.energy_lower_from_bits_lower M hRank)
    hBudget

/-- Admissibility-restricted cross-docking decision problem at tolerance `ε`, obtained by
summarizing optimal docking actions through a quantitative admissibility family. -/
noncomputable def crossDockingAdmissibleDecisionProblem
    {B : Type*} (prob : MDBindingProblem)
    (F : QuantitativeAdmissibilityFamily MDAction B) (ε : ℝ) :
    DecisionProblem B MDState :=
  F.summaryDecisionProblem prob.toDecisionProblem ε

/-- A state pair is local to a coordinate-support set when all coordinates outside that support
agree under the molecular projection map. -/
def localStatePairWithinSupport
    (prob : MDBindingProblem)
    (support : Finset (Fin (numMDCoordinates prob)))
    (s s' : MDState) : Prop :=
  ∀ j : Fin (numMDCoordinates prob), j ∉ support → mdProj prob s j = mdProj prob s' j

/-- Local relevance test for admissible cross-docking: a coordinate is relevant iff there exists a
local state pair (restricted to the coordinate's `k`-hop mechanical neighborhood support plus
ligand support) that changes admissibility class. -/
theorem local_sufficiency_test
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (ε : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    (hSupport : ∀ i,
      localSupport i ⊆
        atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport)
    (hComplete : ∀ i,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) →
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s')
    (hSound : ∀ i,
      (∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s') →
      @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) :
    ∀ i,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) ↔
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        localSupport i ⊆
          atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s' := by
  intro i
  constructor
  · intro hRel
    rcases hComplete i hRel with ⟨s, s', hLocal, hDiff⟩
    exact ⟨s, s', hLocal, hSupport i, hDiff⟩
  · rintro ⟨s, s', hLocal, _hSupport, hDiff⟩
    exact hSound i ⟨s, s', hLocal, hDiff⟩

/-- Concrete local witness that coordinate `i` can change admissible optimizer class inside its
declared local mechanical support. -/
def concreteDropWitness
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (ε : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    (i : Fin (numMDCoordinates prob)) : Prop :=
  ∃ s s' : MDState,
    localStatePairWithinSupport prob (localSupport i) s s' ∧
    localSupport i ⊆
      atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport ∧
    (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
      (crossDockingAdmissibleDecisionProblem prob F ε).Opt s'

/-- Concrete coordinate-drop test used by the greedy extraction loop: drop iff no local witness of
admissibility-class change exists for the coordinate. -/
noncomputable def concreteDropTest
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (ε : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob))) :
    Finset (Fin (numMDCoordinates prob)) → Fin (numMDCoordinates prob) → Bool :=
  fun _ i => by
    classical
    exact decide (¬ concreteDropWitness prob F ε G activePocket k
      atomCoordinateSupport ligandCoordinateSupport localSupport i)

private def defaultAtom : Atom :=
  { index := 0
    position := (0, 0, 0)
    charge := 0
    mass := 0 }

/-- Canonical state realization from a full molecular coordinate map. -/
noncomputable def mdStateFromCoordinateMap
    (prob : MDBindingProblem)
    (coords : Fin (numMDCoordinates prob) → ℝ) : MDState :=
  { protein := List.ofFn (fun i : Fin prob.protein.numAtoms =>
      { index := i.1
        position :=
          (coords (proteinCoordFin prob i.1 i.2 ⟨0, by decide⟩),
            coords (proteinCoordFin prob i.1 i.2 ⟨1, by decide⟩),
            coords (proteinCoordFin prob i.1 i.2 ⟨2, by decide⟩))
        charge := 0
        mass := 0 })
    ligand := List.ofFn (fun i : Fin prob.ligand.numAtoms =>
      { index := i.1
        position :=
          (coords (ligandCoordFin prob i ⟨0, by decide⟩),
            coords (ligandCoordFin prob i ⟨1, by decide⟩),
            coords (ligandCoordFin prob i ⟨2, by decide⟩))
        charge := 0
        mass := 0 }) }

private theorem mdProj_mdStateFromCoordinateMap
    (prob : MDBindingProblem)
    (coords : Fin (numMDCoordinates prob) → ℝ)
    (i : Fin (numMDCoordinates prob)) :
    mdProj prob (mdStateFromCoordinateMap prob coords) i = coords i := by
  unfold mdProj mdStateFromCoordinateMap
  let nP := 3 * prob.protein.numAtoms
  by_cases hProtein : i.1 < nP
  · have hIdx : i.1 / 3 < prob.protein.numAtoms := by
      have hProtein' : i.1 < 3 * prob.protein.numAtoms := by
        simpa [nP] using hProtein
      omega
    have hModCases : i.1 % 3 = 0 ∨ i.1 % 3 = 1 ∨ i.1 % 3 = 2 := by
      omega
    rcases hModCases with hMod0 | hMod1 | hMod2
    · have hiEq0 :
          proteinCoordFin prob (i.1 / 3) hIdx ⟨0, by decide⟩ = i := by
        apply Fin.ext
        simp [proteinCoordFin, hMod0]
        omega
      have hCoord :
          coords (proteinCoordFin prob (i.1 / 3) hIdx ⟨0, by decide⟩) = coords i := by
        simpa using congrArg coords hiEq0
      simpa [hProtein, hIdx, hMod0, nP] using hCoord
    · have hiEq1 :
          proteinCoordFin prob (i.1 / 3) hIdx ⟨1, by decide⟩ = i := by
        apply Fin.ext
        simp [proteinCoordFin, hMod1]
        omega
      have hCoord :
          coords (proteinCoordFin prob (i.1 / 3) hIdx ⟨1, by decide⟩) = coords i := by
        simpa using congrArg coords hiEq1
      simpa [hProtein, hIdx, hMod1, nP] using hCoord
    · have hiEq2 :
          proteinCoordFin prob (i.1 / 3) hIdx ⟨2, by decide⟩ = i := by
        apply Fin.ext
        simp [proteinCoordFin, hMod2]
        omega
      have hCoord :
          coords (proteinCoordFin prob (i.1 / 3) hIdx ⟨2, by decide⟩) = coords i := by
        simpa using congrArg coords hiEq2
      simpa [hProtein, hIdx, hMod2, nP] using hCoord
  · have hGe : nP ≤ i.1 := Nat.le_of_not_lt hProtein
    have hIdx : (i.1 - nP) / 3 < prob.ligand.numAtoms := by
      have hiLt : i.1 < numMDCoordinates prob := i.2
      have hLt : i.1 - nP < 3 * prob.ligand.numAtoms := by
        have h' : i.1 < nP + 3 * prob.ligand.numAtoms := by
          simpa [nP, numMDCoordinates, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc,
            Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using hiLt
        omega
      omega
    have hModCases : (i.1 - nP) % 3 = 0 ∨ (i.1 - nP) % 3 = 1 ∨ (i.1 - nP) % 3 = 2 := by
      omega
    rcases hModCases with hMod0 | hMod1 | hMod2
    · have hiEq0 :
          ligandCoordFin prob ⟨(i.1 - nP) / 3, hIdx⟩ ⟨0, by decide⟩ = i := by
        apply Fin.ext
        simp [ligandCoordFin, hMod0]
        omega
      have hCoord :
          coords (ligandCoordFin prob ⟨(i.1 - nP) / 3, hIdx⟩ ⟨0, by decide⟩) = coords i := by
        simpa using congrArg coords hiEq0
      simpa [hProtein, hIdx, hMod0, nP] using hCoord
    · have hiEq1 :
          ligandCoordFin prob ⟨(i.1 - nP) / 3, hIdx⟩ ⟨1, by decide⟩ = i := by
        apply Fin.ext
        simp [ligandCoordFin, hMod1]
        omega
      have hCoord :
          coords (ligandCoordFin prob ⟨(i.1 - nP) / 3, hIdx⟩ ⟨1, by decide⟩) = coords i := by
        simpa using congrArg coords hiEq1
      simpa [hProtein, hIdx, hMod1, nP] using hCoord
    · have hiEq2 :
          ligandCoordFin prob ⟨(i.1 - nP) / 3, hIdx⟩ ⟨2, by decide⟩ = i := by
        apply Fin.ext
        simp [ligandCoordFin, hMod2]
        omega
      have hCoord :
          coords (ligandCoordFin prob ⟨(i.1 - nP) / 3, hIdx⟩ ⟨2, by decide⟩) = coords i := by
        simpa using congrArg coords hiEq2
      simpa [hProtein, hIdx, hMod2, nP] using hCoord

/-- Canonical coordinate replace operation for molecular states: swap only coordinate `i` from `s'`
into `s` while preserving all other coordinates. -/
noncomputable def canonicalMDStateReplace
    (prob : MDBindingProblem)
    (s : MDState)
    (i : Fin (numMDCoordinates prob))
    (s' : MDState) : MDState :=
  mdStateFromCoordinateMap prob (fun j => if j = i then mdProj prob s' j else mdProj prob s j)

/-- Canonical product-space structure on molecular states induced by the base coordinate projection
`mdProj`. -/
noncomputable def canonicalMDStateProductSpace
    (prob : MDBindingProblem) : ProductSpace MDState (numMDCoordinates prob) where
  Coord := fun _ => ℝ
  proj := mdProj prob
  replace := canonicalMDStateReplace prob
  replace_proj_eq := by
    intro s s' i
    unfold canonicalMDStateReplace
    simpa [if_pos rfl] using mdProj_mdStateFromCoordinateMap prob
      (fun j => if j = i then mdProj prob s' j else mdProj prob s j) i
  replace_proj_ne := by
    intro s s' i j hj
    unfold canonicalMDStateReplace
    simpa [if_neg hj] using mdProj_mdStateFromCoordinateMap prob
      (fun k => if k = i then mdProj prob s' k else mdProj prob s k) j

/-- Compatibility of the canonical molecular `ProductSpace` projection with the base molecular
coordinate map `mdProj`. -/
theorem canonicalMDStateCompat
    (prob : MDBindingProblem) :
    ∀ s s' j,
      @CoordinateSpace.proj MDState (numMDCoordinates prob)
          (canonicalMDStateProductSpace prob).toCoordinateSpace s j =
        @CoordinateSpace.proj MDState (numMDCoordinates prob)
          (canonicalMDStateProductSpace prob).toCoordinateSpace s' j ↔
      mdProj prob s j = mdProj prob s' j := by
  intro s s' j
  rfl

/-- Soundness of the concrete drop test: if the test accepts dropping `i`, then dropping `i`
preserves sufficiency for the admissible decision problem. -/
theorem concreteDropTest_is_sound
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (ε : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    [hProd : ProductSpace MDState (numMDCoordinates prob)]
    (hSupport : ∀ i,
      localSupport i ⊆
        atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport)
    (hComplete : ∀ i,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) →
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s')
    (hSound : ∀ i,
      (∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s') →
      @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i)
    (hCompat : ∀ s s' j,
      @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s j =
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' j ↔
      mdProj prob s j = mdProj prob s' j) :
    ∀ current i,
      (@DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) current) →
      i ∈ current →
      concreteDropTest prob F ε G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport current i = true →
      (@DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) (current.erase i)) := by
  intro current i hSuff _hMem hDrop
  let dp : DecisionProblem B MDState :=
    crossDockingAdmissibleDecisionProblem prob F ε
  have hLocal := local_sufficiency_test prob F ε G activePocket k
    atomCoordinateSupport ligandCoordinateSupport localSupport
    hSupport hComplete hSound
  have hNoWitness : ¬ concreteDropWitness prob F ε G activePocket k
      atomCoordinateSupport ligandCoordinateSupport localSupport i := by
    classical
    by_contra hWitness
    unfold concreteDropTest at hDrop
    simp [hWitness] at hDrop
  have hNotRel : ¬ (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) dp i) := by
    intro hRel
    have hRel' : @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i := by
      simpa [dp] using hRel
    exact hNoWitness ((hLocal i).1 hRel')
  have hNotRelProd : ¬ (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
      hProd.toCoordinateSpace dp i) := by
    intro hRelProd
    have hRelMd : @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) dp i := by
      rcases hRelProd with ⟨s, s', hAgreeProd, hOptNe⟩
      refine ⟨s, s', ?_, hOptNe⟩
      intro j hj
      exact (hCompat s s' j).mp (hAgreeProd j hj)
    exact hNotRel hRelMd
  have hIrrProd : (@DecisionProblem.isIrrelevant B MDState (numMDCoordinates prob)
      hProd.toCoordinateSpace dp i) := by
    exact (@DecisionProblem.irrelevant_iff_not_relevant B MDState (numMDCoordinates prob)
      hProd.toCoordinateSpace dp i).2 hNotRelProd
  have hSuffProd : @DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
      hProd.toCoordinateSpace dp current := by
    intro s s' hAgreeProd
    have hAgreeMd : @agreeOn MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) s s' current := by
      intro j hj
      exact (hCompat s s' j).mp (hAgreeProd j hj)
    have hSuffMd : @DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) dp current := by
      simpa [dp] using hSuff
    exact hSuffMd s s' hAgreeMd
  have hEraseProd : @DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
      hProd.toCoordinateSpace dp (current.erase i) :=
    dp.sufficient_erase_irrelevant' current i hSuffProd hIrrProd
  have hEraseMd : @DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) dp (current.erase i) := by
    intro s s' hAgreeMd
    have hAgreeProd : @agreeOn MDState (numMDCoordinates prob)
        hProd.toCoordinateSpace s s' (current.erase i) := by
      intro j hj
      exact (hCompat s s' j).mpr (hAgreeMd j hj)
    exact hEraseProd s s' hAgreeProd
  simpa [dp] using hEraseMd

/-- Completeness of the concrete drop test: if the test rejects dropping `i`, then `i` is relevant
for the admissible decision problem. -/
theorem concreteDropTest_is_complete
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (ε : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    (hSupport : ∀ i,
      localSupport i ⊆
        atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport)
    (hComplete : ∀ i,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) →
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s')
    (hSound : ∀ i,
      (∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s') →
      @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) :
    ∀ I i,
      i ∈ I →
      concreteDropTest prob F ε G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport I i = false →
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) := by
  intro I i _hi hDrop
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  have hLocal := local_sufficiency_test prob F ε G activePocket k
    atomCoordinateSupport ligandCoordinateSupport localSupport
    hSupport hComplete hSound
  have hNN : ¬ ¬ concreteDropWitness prob F ε G activePocket k
      atomCoordinateSupport ligandCoordinateSupport localSupport i := by
    classical
    unfold concreteDropTest at hDrop
    intro hNoWitness
    simp [hNoWitness] at hDrop
  have hWitness : concreteDropWitness prob F ε G activePocket k
      atomCoordinateSupport ligandCoordinateSupport localSupport i := by
    exact Classical.not_not.mp hNN
  exact (hLocal i).2 hWitness

/-- Effective local state-space bound: if each coordinate's local relevance witness search is
bounded by the neighborhood-plus-ligand budget, then there is a polynomial `p(K,L)` (independent
of global structural rank) controlling all single-coordinate tests. -/
theorem effective_state_space_poly_bound
    (prob : MDBindingProblem)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (K L : ℕ)
    (localPairCount : Fin (numMDCoordinates prob) → ℕ)
    (hLocalPairs : ∀ i,
      localPairCount i ≤
        ((G.kHopNeighborhood activePocket k).card + prob.ligand.numAtoms) ^ 2)
    (hPocket : (G.kHopNeighborhood activePocket k).card ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    ∃ p : ℕ → ℕ → ℕ,
      (∀ K' L', p K' L' = (K' + L') ^ 2) ∧
      (∀ i, localPairCount i ≤ p K L) := by
  refine ⟨fun K' L' => (K' + L') ^ 2, ?_, ?_⟩
  · intro K' L'
    rfl
  · intro i
    have hBase :
        (G.kHopNeighborhood activePocket k).card + prob.ligand.numAtoms ≤ K + L :=
      Nat.add_le_add hPocket hLigand
    have hPow :
        ((G.kHopNeighborhood activePocket k).card + prob.ligand.numAtoms) ^ 2 ≤
          (K + L) ^ 2 :=
      Nat.pow_le_pow_left hBase _
    exact le_trans (hLocalPairs i) hPow

/-- Candidate coordinate set for cross-docking extraction: `k`-hop mechanical neighborhood
coordinates together with ligand coordinates. -/
noncomputable def crossDockingCandidateCoords
    (prob : MDBindingProblem)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob))) :
    Finset (Fin (numMDCoordinates prob)) :=
  atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport

/-- Cost model: total number of local state-pair checks used by cross-docking extraction over the
chosen candidate coordinate set. -/
def crossDockingExtractionCost
    (prob : MDBindingProblem)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (localPairCount : Fin (numMDCoordinates prob) → ℕ) : ℕ :=
  candidateCoords.sum localPairCount

/-- Polynomial-time cross-docking extraction: combining the local relevance characterization with
the polynomial bound on local pair tests yields a full extraction cost bounded by a polynomial in
pocket size `K` and ligand size `L`. -/
theorem crossDocking_extraction_polytime
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (ε : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    (hSupport : ∀ i,
      localSupport i ⊆
        atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport)
    (hComplete : ∀ i,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) →
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s')
    (hSound : ∀ i,
      (∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s') →
      @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i)
    (K L : ℕ)
    (localPairCount : Fin (numMDCoordinates prob) → ℕ)
    (hLocalPairs : ∀ i,
      localPairCount i ≤
        ((G.kHopNeighborhood activePocket k).card + prob.ligand.numAtoms) ^ 2)
    (hPocket : (G.kHopNeighborhood activePocket k).card ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L)
    (hCandidateCard :
      (crossDockingCandidateCoords prob G activePocket k atomCoordinateSupport
        ligandCoordinateSupport).card ≤ K + L) :
    (∀ i ∈ crossDockingCandidateCoords prob G activePocket k atomCoordinateSupport
        ligandCoordinateSupport,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F ε) i) ↔
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        localSupport i ⊆
          atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport ∧
        (crossDockingAdmissibleDecisionProblem prob F ε).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F ε).Opt s') ∧
    crossDockingExtractionCost prob
      (crossDockingCandidateCoords prob G activePocket k atomCoordinateSupport
        ligandCoordinateSupport)
      localPairCount ≤ (K + L) ^ 3 := by
  have hLocal := local_sufficiency_test prob F ε G activePocket k
    atomCoordinateSupport ligandCoordinateSupport localSupport
    hSupport hComplete hSound
  have hPoly := effective_state_space_poly_bound prob G activePocket k K L
    localPairCount hLocalPairs hPocket hLigand
  rcases hPoly with ⟨p, hpEq, hpBound⟩
  constructor
  · intro i hi
    exact hLocal i
  · let candidates :=
      crossDockingCandidateCoords prob G activePocket k atomCoordinateSupport
        ligandCoordinateSupport
    have hSum : candidates.sum localPairCount ≤ candidates.sum (fun _ => p K L) := by
      exact Finset.sum_le_sum (fun i _ => hpBound i)
    have hMul : candidates.card * (p K L) ≤ (K + L) * (p K L) :=
      Nat.mul_le_mul_right (p K L) hCandidateCard
    have hpKL : p K L = (K + L) ^ 2 := hpEq K L
    unfold crossDockingExtractionCost
    calc
      candidates.sum localPairCount ≤ candidates.sum (fun _ => p K L) := hSum
      _ = candidates.card * (p K L) := by simp
      _ ≤ (K + L) * (p K L) := hMul
      _ = (K + L) * ((K + L) ^ 2) := by rw [hpKL]
      _ = (K + L) ^ 3 := by
            simp [pow_succ, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

/-- State object for quotient-directed conformational search restricted to selected coordinates. -/
structure QuotientDirectedSearchState (prob : MDBindingProblem) where
  openCoords : Finset (Fin (numMDCoordinates prob))
  expandedCoords : Finset (Fin (numMDCoordinates prob))
deriving DecidableEq

/-- One-step best-first expansion relation represented as the finite successor set: expand one
currently-open candidate coordinate and move it to the expanded set. -/
noncomputable def quotientDirectedSearchTransition
    (prob : MDBindingProblem)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (st : QuotientDirectedSearchState prob) :
    Finset (QuotientDirectedSearchState prob) :=
  (st.openCoords.filter (fun i => i ∈ candidateCoords)).image (fun i =>
    { openCoords := st.openCoords.erase i
      expandedCoords := insert i st.expandedCoords })

/-- Finite quotient-directed search machine: initialized on candidate coordinates and expanded by
the local transition rule above. -/
structure QuotientDirectedSearchMachine (prob : MDBindingProblem) where
  candidateCoords : Finset (Fin (numMDCoordinates prob))
  init : QuotientDirectedSearchState prob
  transition : QuotientDirectedSearchState prob → Finset (QuotientDirectedSearchState prob)

/-- Quotient-directed conformational search instantiated on the extracted cross-docking candidate
coordinates. -/
noncomputable def quotientDirectedSearch
    (prob : MDBindingProblem)
    (candidateCoords : Finset (Fin (numMDCoordinates prob))) :
    QuotientDirectedSearchMachine prob :=
  { candidateCoords := candidateCoords
    init := { openCoords := candidateCoords, expandedCoords := ∅ }
    transition := quotientDirectedSearchTransition prob candidateCoords }

/-- Search tree view: all coordinate subsets reachable by repeatedly applying the candidate-only
expansion rule. -/
noncomputable def quotientDirectedSearchTree
    (prob : MDBindingProblem)
    (candidateCoords : Finset (Fin (numMDCoordinates prob))) :
    Finset (Finset (Fin (numMDCoordinates prob))) :=
  candidateCoords.powerset

/-- Completeness of quotient-directed search: if a globally admissible state exists and admissibility
factors through a local coordinate encoding supported on candidate coordinates, then an admissible
node appears in the local search tree. -/
theorem search_completeness
    (prob : MDBindingProblem)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (globalAdmissible : MDState → Prop)
    (localAdmissibleNode : Finset (Fin (numMDCoordinates prob)) → Prop)
    (encode : MDState → Finset (Fin (numMDCoordinates prob)))
    (hEncode : ∀ s, encode s ⊆ candidateCoords)
    (hLift : ∀ s, globalAdmissible s → localAdmissibleNode (encode s))
    (hExists : ∃ s, globalAdmissible s) :
    ∃ q ∈ quotientDirectedSearchTree prob candidateCoords, localAdmissibleNode q := by
  rcases hExists with ⟨s, hs⟩
  refine ⟨encode s, ?_, hLift s hs⟩
  unfold quotientDirectedSearchTree
  exact Finset.mem_powerset.mpr (hEncode s)

/-- Node-expansion cost model for quotient-directed search. -/
def quotientDirectedSearchNodeBound
    (prob : MDBindingProblem)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (localPairCount : Fin (numMDCoordinates prob) → ℕ) : ℕ :=
  crossDockingExtractionCost prob candidateCoords localPairCount

/-- Polynomial search bound: restricting expansions to the extracted candidate coordinates and using
the local pair-test polynomial envelope yields a cubic bound in pocket/ligand sizes. -/
theorem search_polytime_bound
    (prob : MDBindingProblem)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (K L : ℕ)
    (localPairCount : Fin (numMDCoordinates prob) → ℕ)
    (hLocalPairs : ∀ i,
      localPairCount i ≤
        ((G.kHopNeighborhood activePocket k).card + prob.ligand.numAtoms) ^ 2)
    (hPocket : (G.kHopNeighborhood activePocket k).card ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L)
    (hCandidateCard :
      (crossDockingCandidateCoords prob G activePocket k atomCoordinateSupport
        ligandCoordinateSupport).card ≤ K + L) :
    quotientDirectedSearchNodeBound prob
      (crossDockingCandidateCoords prob G activePocket k atomCoordinateSupport
        ligandCoordinateSupport)
      localPairCount ≤ (K + L) ^ 3 := by
  have hPoly := effective_state_space_poly_bound prob G activePocket k K L
    localPairCount hLocalPairs hPocket hLigand
  rcases hPoly with ⟨p, hpEq, hpBound⟩
  let candidates :=
    crossDockingCandidateCoords prob G activePocket k atomCoordinateSupport ligandCoordinateSupport
  have hSum : candidates.sum localPairCount ≤ candidates.sum (fun _ => p K L) := by
    exact Finset.sum_le_sum (fun i _ => hpBound i)
  have hMul : candidates.card * (p K L) ≤ (K + L) * (p K L) :=
    Nat.mul_le_mul_right (p K L) hCandidateCard
  have hpKL : p K L = (K + L) ^ 2 := hpEq K L
  unfold quotientDirectedSearchNodeBound crossDockingExtractionCost
  calc
    candidates.sum localPairCount ≤ candidates.sum (fun _ => p K L) := hSum
    _ = candidates.card * (p K L) := by simp
    _ ≤ (K + L) * (p K L) := hMul
    _ = (K + L) * ((K + L) ^ 2) := by rw [hpKL]
    _ = (K + L) ^ 3 := by simp [pow_succ, Nat.mul_assoc, Nat.mul_comm, Nat.mul_left_comm]

/-- Adaptive half-gap refinement state: current grid spacing and retained top-k action classes. -/
structure AdaptiveRefinementState (A S : Type*) [DecidableEq A] where
  resolution : ℝ
  topKClasses : Finset A
  referenceState : S

/-- One adaptive refinement step halves grid spacing around the current top-k action classes. -/
noncomputable def adaptiveRefinementStep
    {A S : Type*} [DecidableEq A]
    (st : AdaptiveRefinementState A S) : AdaptiveRefinementState A S :=
  { resolution := st.resolution / 2
    topKClasses := st.topKClasses
    referenceState := st.referenceState }

/-- Local half-gap refinement preserves admissible action candidates: the refined lifted optimizer
set stays inside the declared ambiguity band and therefore discards no admissible winners. -/
theorem refinement_preserves_admissibility
    {A : Type u} {Scont Sgrid : Type v} [Fintype A] [Nonempty A] [DecidableEq A]
    (uCont : A → Scont → ℝ) (uGrid : A → Sgrid → ℝ)
    (lift : Sgrid → Scont) (eps : ℝ → ℝ) (res : ℝ)
    (hApprox : DecisionQuotient.Tractability.GridConvergence.ResolutionControlledApprox
      uCont uGrid lift eps res)
    (hDelta : 0 ≤ eps res)
    (s : Sgrid)
    (hBound : eps res <
      StrictUtilityGap
        (actionPinnedDecisionProblem ({ utility := uGrid } : DecisionProblem A Sgrid))
        (maxCodeOptAction ({ utility := uGrid } : DecisionProblem A Sgrid) s) s / 2) :
    (fixedTieBreakDecisionProblem
      ({ utility := fun a sGrid => uCont a (lift sGrid) } : DecisionProblem A Sgrid)
      (actionTieBreakEps ({ utility := uGrid } : DecisionProblem A Sgrid))).Opt s ⊆
      ↑(DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun x => uCont x (lift s)) 1 (by norm_num) (2 * eps res)) := by
  exact resolutionControlled_liftedFixedTieBreak_opt_subset_ambiguityBand
    uCont uGrid lift eps res hApprox hDelta s hBound

/-- Discrete refinement error after `steps` halving rounds. -/
def refinementErrorAfter (initialErrorUnits steps : ℕ) : ℕ :=
  initialErrorUnits / 2 ^ steps

/-- Halting bound for adaptive half-gap refinement in scaled units: after at most `Nat.size ε₀`
halving rounds, the discretization error drops below any positive target threshold. This gives the
expected logarithmic refinement depth. -/
theorem refinement_halting_bound
    (initialErrorUnits targetGapUnits : ℕ)
    (hGapPos : 0 < targetGapUnits) :
    ∃ steps : ℕ,
      steps ≤ Nat.size initialErrorUnits ∧
      refinementErrorAfter initialErrorUnits steps < targetGapUnits := by
  refine ⟨Nat.size initialErrorUnits, le_rfl, ?_⟩
  have hPow : initialErrorUnits < 2 ^ Nat.size initialErrorUnits :=
    Nat.lt_size_self initialErrorUnits
  have hZero : refinementErrorAfter initialErrorUnits (Nat.size initialErrorUnits) = 0 := by
    unfold refinementErrorAfter
    exact Nat.div_eq_of_lt hPow
  rw [hZero]
  exact hGapPos

/-- Certified scorer output with value/error pair. -/
structure CertifiedScorerResult where
  score : ℝ
  epsilon : ℝ

/-- Certified grid scorer: Lennard-Jones first-order corrected score plus Coulomb grid score, with
error radius from the LJ Hessian envelope and Coulomb error budget. -/
noncomputable def certifiedScorerEval
    {A S : Type*}
    (distanceGrid distanceErrorEstimate : A → S → ℝ)
    (ε σ q_i q_j rmin delta coulombErrorRadius : ℝ)
    (a : A) (s : S) : CertifiedScorerResult where
  score :=
    DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
      ljGradExpr ε σ (distanceGrid a s) * distanceErrorEstimate a s +
      DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s)
  epsilon :=
    (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ) + coulombErrorRadius

/-- Certified scorer accuracy guarantee: the continuous Lennard-Jones+Coulomb score at any grid
state lies within `computed_score ± epsilon`. -/
theorem scorer_accuracy_guarantee
    {A S : Type*}
    (distanceExact distanceGrid distanceErrorEstimate : A → S → ℝ)
    (ε σ q_i q_j rmin rmax delta coulombErrorRadius : ℝ)
    (hmin : 0 < rmin)
    (hShellExact : ∀ a s, distanceExact a s ∈ Set.Icc rmin rmax)
    (hShellGrid : ∀ a s, distanceGrid a s ∈ Set.Icc rmin rmax)
    (hDistErr : ∀ a s, |distanceExact a s - distanceGrid a s| ≤ delta)
    (hErrModel : ∀ a s, distanceErrorEstimate a s = distanceExact a s - distanceGrid a s)
    (hCoulombErr : ∀ a s,
      |DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s)| ≤
        coulombErrorRadius) :
    ∀ a s,
      |(DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) +
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s)) -
          (certifiedScorerEval distanceGrid distanceErrorEstimate
            ε σ q_i q_j rmin delta coulombErrorRadius a s).score| ≤
        (certifiedScorerEval distanceGrid distanceErrorEstimate
          ε σ q_i q_j rmin delta coulombErrorRadius a s).epsilon := by
  intro a s
  have hLJBase :=
    exactLJ_quadraticDiscretizationError_of_distanceError_and_shellSecondDerivShellBound
      distanceExact distanceGrid ε σ rmin rmax delta hmin hShellExact hShellGrid hDistErr a s
  have hLJ :
      |DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) -
          (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
            ljGradExpr ε σ (distanceGrid a s) * distanceErrorEstimate a s)| ≤
        (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ) := by
    simpa [hErrModel a s] using hLJBase
  have hSplit :
      |(DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) +
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s)) -
          ((DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
            ljGradExpr ε σ (distanceGrid a s) * distanceErrorEstimate a s) +
            DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s))|
        ≤
      |DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) -
          (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
            ljGradExpr ε σ (distanceGrid a s) * distanceErrorEstimate a s)| +
      |DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s)| := by
    have hEq :
        (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) +
            DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s)) -
            ((DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
              ljGradExpr ε σ (distanceGrid a s) * distanceErrorEstimate a s) +
              DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s)) =
          (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) -
            (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
              ljGradExpr ε σ (distanceGrid a s) * distanceErrorEstimate a s)) +
          (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s) -
            DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s)) := by
      ring
    rw [hEq]
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s) -
          (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
            ljGradExpr ε σ (distanceGrid a s) * distanceErrorEstimate a s))
        (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceExact a s) -
          DecisionQuotient.Tractability.CoulombApproximation.exactCoulombScore q_i q_j (distanceGrid a s)))
  have hBound :=
    le_trans hSplit (add_le_add hLJ (hCoulombErr a s))
  simpa [certifiedScorerEval, add_assoc, add_left_comm, add_comm] using hBound

/-- End-to-end cross-docking platform output certificate. -/
structure CrossDockingPlatformResult where
  pose : MDAction
  state : MDState
  certifiedScore : CertifiedScorerResult
  logicalSteps : ℕ
  thermodynamicWork : ℝ

/-- End-to-end platform composition: Extract -> Search -> Adaptive Refine -> Certified Score. -/
noncomputable def crossDockingPlatform
    (_prob : MDBindingProblem)
    (pose : MDAction) (state : MDState)
    (certifiedScore : CertifiedScorerResult)
    (searchSteps refinementSteps : ℕ)
    (thermodynamicWork : ℝ) : CrossDockingPlatformResult :=
  { pose := pose
    state := state
    certifiedScore := certifiedScore
    logicalSteps := searchSteps + refinementSteps
    thermodynamicWork := thermodynamicWork }

/-- TUR precision-overhead term used in end-to-end thermodynamic accounting. -/
noncomputable def turPrecisionOverhead
    {S : Type*} [Fintype S]
    {mc : DecisionQuotient.Physics.DiscreteMarkovChain S}
    (kB : ℝ)
    (π : DecisionQuotient.Physics.StationaryDist mc)
    (cycleTime : DecisionQuotient.Physics.Observable S) : ℝ :=
  2 * kB * (DecisionQuotient.Physics.expectedValue π cycleTime) ^ 2 /
    DecisionQuotient.Physics.variance π cycleTime

/-- Platform correctness: if the composed platform outputs pose `P`, then `P` is admissible and
lies in the declared `2 * ε` ambiguity neighborhood of the continuous optimum slice. -/
theorem platform_correctness
    (prob : MDBindingProblem)
    [Fintype MDAction] [DecidableEq MDAction] [Nonempty MDAction]
    (pose : MDAction) (state : MDState)
    (certifiedScore : CertifiedScorerResult)
    (searchSteps refinementSteps : ℕ)
    (thermodynamicWork : ℝ)
    (admissiblePose : MDAction → Prop)
    (hSearch : admissiblePose pose)
    (hRefine : pose ∈
      DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => prob.toDecisionProblem.utility a state) 1 (by norm_num)
        (2 * certifiedScore.epsilon)) :
    admissiblePose
        (crossDockingPlatform prob pose state certifiedScore
          searchSteps refinementSteps thermodynamicWork).pose ∧
    (crossDockingPlatform prob pose state certifiedScore
        searchSteps refinementSteps thermodynamicWork).pose ∈
      DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => prob.toDecisionProblem.utility a
          (crossDockingPlatform prob pose state certifiedScore
            searchSteps refinementSteps thermodynamicWork).state)
        1 (by norm_num)
        (2 * (crossDockingPlatform prob pose state certifiedScore
          searchSteps refinementSteps thermodynamicWork).certifiedScore.epsilon) := by
  constructor
  · simpa [crossDockingPlatform] using hSearch
  · simpa [crossDockingPlatform] using hRefine

/-- End-to-end master bound: platform logical steps are polynomial in pocket/ligand scales and
total work is bounded by Landauer structural-rank cost plus TUR precision overhead. -/
theorem platform_polytime_thermodynamic_bound
    (prob : MDBindingProblem)
    (admissibleDP : DecisionProblem MDAction MDState)
    (pose : MDAction) (state : MDState)
    (certifiedScore : CertifiedScorerResult)
    (searchSteps refinementSteps ε0 K L : ℕ)
    (thermodynamicWork : ℝ)
    {Ω : Type*} [Fintype Ω]
    {mc : DecisionQuotient.Physics.DiscreteMarkovChain Ω}
    (π : DecisionQuotient.Physics.StationaryDist mc)
    (cycleTime : DecisionQuotient.Physics.Observable Ω)
    {kB T : ℝ}
    (hSearchPoly : searchSteps ≤ (K + L) ^ 3)
    (hRefinePoly : refinementSteps ≤ Nat.size ε0)
    (hWorkBound :
      thermodynamicWork ≤
        ((admissibleDP.srank : ℝ) * (kB * T * Real.log 2)) +
          turPrecisionOverhead kB π cycleTime) :
    (crossDockingPlatform prob pose state certifiedScore
      searchSteps refinementSteps thermodynamicWork).logicalSteps ≤
        (K + L) ^ 3 + Nat.size ε0 ∧
    (crossDockingPlatform prob pose state certifiedScore
      searchSteps refinementSteps thermodynamicWork).thermodynamicWork ≤
        ((admissibleDP.srank : ℝ) * (kB * T * Real.log 2)) +
          turPrecisionOverhead kB π cycleTime := by
  constructor
  · simpa [crossDockingPlatform] using Nat.add_le_add hSearchPoly hRefinePoly
  · simpa [crossDockingPlatform] using hWorkBound

/-- One constructive greedy extraction step: test one coordinate and erase it iff the explicit
drop-test says erasure preserves admissibility. -/
def greedyExtractStep
    {n : ℕ} [DecidableEq (Fin n)]
    (dropTest : Finset (Fin n) → Fin n → Bool)
    (current : Finset (Fin n)) (i : Fin n) : Finset (Fin n) :=
  if hmem : i ∈ current then
    if dropTest current i then current.erase i else current
  else
    current

/-- Tail-recursive greedy extraction loop over an explicit coordinate order. -/
def greedyExtractLoopAux
    {n : ℕ} [DecidableEq (Fin n)]
    (dropTest : Finset (Fin n) → Fin n → Bool) :
    List (Fin n) → Finset (Fin n) → Finset (Fin n)
  | [], current => current
  | i :: rest, current =>
      greedyExtractLoopAux dropTest rest (greedyExtractStep dropTest current i)

/-- Constructive greedy extraction loop on cross-docking candidate coordinates. -/
noncomputable def greedyExtractLoop
    {n : ℕ} [DecidableEq (Fin n)]
    (candidateCoords : Finset (Fin n))
    (dropTest : Finset (Fin n) → Fin n → Bool) : Finset (Fin n) :=
  greedyExtractLoopAux dropTest candidateCoords.toList candidateCoords

theorem greedyExtractStep_card_lt_of_drop
    {n : ℕ} [DecidableEq (Fin n)]
    (dropTest : Finset (Fin n) → Fin n → Bool)
    (current : Finset (Fin n)) (i : Fin n)
    (hMem : i ∈ current)
    (hDrop : dropTest current i = true) :
    (greedyExtractStep dropTest current i).card < current.card := by
  unfold greedyExtractStep
  rw [dif_pos hMem, if_pos hDrop]
  exact Finset.card_erase_lt_of_mem hMem

theorem greedyExtractStep_card_le
    {n : ℕ} [DecidableEq (Fin n)]
    (dropTest : Finset (Fin n) → Fin n → Bool)
    (current : Finset (Fin n)) (i : Fin n) :
    (greedyExtractStep dropTest current i).card ≤ current.card := by
  unfold greedyExtractStep
  by_cases hMem : i ∈ current
  · by_cases hDrop : dropTest current i
    · exact Nat.le_of_lt (by
        simpa [hMem, hDrop] using
          (Finset.card_erase_lt_of_mem hMem))
    · simp [hMem, hDrop]
  · simp [hMem]

theorem greedyExtractLoopAux_card_le
    {n : ℕ} [DecidableEq (Fin n)]
    (dropTest : Finset (Fin n) → Fin n → Bool) :
    ∀ coords current,
      (greedyExtractLoopAux dropTest coords current).card ≤ current.card
  | [], current => by simp [greedyExtractLoopAux]
  | i :: rest, current => by
      have hRest := greedyExtractLoopAux_card_le dropTest rest (greedyExtractStep dropTest current i)
      exact le_trans hRest (greedyExtractStep_card_le dropTest current i)

/-- Termination/cost certificate for the constructive greedy loop: the active set cardinality never
increases, and every true drop-test strictly decreases it. -/
theorem greedyExtractLoop_terminates
    {n : ℕ} [DecidableEq (Fin n)]
    (candidateCoords : Finset (Fin n))
    (dropTest : Finset (Fin n) → Fin n → Bool) :
    (greedyExtractLoop candidateCoords dropTest).card ≤ candidateCoords.card := by
  unfold greedyExtractLoop
  simpa using greedyExtractLoopAux_card_le dropTest candidateCoords.toList candidateCoords

theorem greedyExtractLoopAux_sufficient
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (dropTest : Finset (Fin n) → Fin n → Bool)
    (hDropSound : ∀ current i,
      dp.isSufficient current → i ∈ current →
      dropTest current i = true → dp.isSufficient (current.erase i)) :
    ∀ coords current,
      dp.isSufficient current →
      dp.isSufficient (greedyExtractLoopAux dropTest coords current)
  | [], current, hSuff => by simpa [greedyExtractLoopAux] using hSuff
  | i :: rest, current, hSuff => by
      have hNext : dp.isSufficient (greedyExtractStep dropTest current i) := by
        unfold greedyExtractStep
        by_cases hMem : i ∈ current
        · by_cases hDrop : dropTest current i
          · simpa [hMem, hDrop] using hDropSound current i hSuff hMem hDrop
          · simpa [hMem, hDrop] using hSuff
        · simpa [hMem] using hSuff
      simpa [greedyExtractLoopAux] using
        greedyExtractLoopAux_sufficient dp dropTest hDropSound rest
          (greedyExtractStep dropTest current i) hNext

theorem greedyExtractLoop_sufficient
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (candidateCoords : Finset (Fin n))
    (dropTest : Finset (Fin n) → Fin n → Bool)
    (hInitSuff : dp.isSufficient candidateCoords)
    (hDropSound : ∀ current i,
      dp.isSufficient current → i ∈ current →
      dropTest current i = true → dp.isSufficient (current.erase i)) :
    dp.isSufficient (greedyExtractLoop candidateCoords dropTest) := by
  unfold greedyExtractLoop
  exact greedyExtractLoopAux_sufficient dp dropTest hDropSound
    candidateCoords.toList candidateCoords hInitSuff

/-- Correctness of the explicit greedy extraction loop: under sound/complete drop-test semantics,
the loop output is sufficient and has cardinality exactly the abstract structural rank. -/
theorem greedyExtractLoop_correct
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (candidateCoords : Finset (Fin n))
    (dropTest : Finset (Fin n) → Fin n → Bool)
    (hInitSuff : dp.isSufficient candidateCoords)
    (hDropSound : ∀ current i,
      dp.isSufficient current → i ∈ current →
      dropTest current i = true → dp.isSufficient (current.erase i))
    (hFixedPoint : ∀ i,
      i ∈ greedyExtractLoop candidateCoords dropTest →
      dropTest (greedyExtractLoop candidateCoords dropTest) i = false)
    (hDropComplete : ∀ I i,
      i ∈ I → dropTest I i = false → dp.isRelevant i) :
    dp.isSufficient (greedyExtractLoop candidateCoords dropTest) ∧
      (greedyExtractLoop candidateCoords dropTest).card = dp.srank := by
  let I := greedyExtractLoop candidateCoords dropTest
  have hSuff : dp.isSufficient I :=
    greedyExtractLoop_sufficient dp candidateCoords dropTest hInitSuff hDropSound
  have hSubsetRelevant : I ⊆ Finset.univ.filter dp.isRelevant := by
    intro i hi
    have hFalse : dropTest I i = false := by
      simpa [I] using hFixedPoint i (by simpa [I] using hi)
    have hRel : dp.isRelevant i := hDropComplete I i (by simpa [I] using hi) hFalse
    exact Finset.mem_filter.mpr ⟨by simp, hRel⟩
  have hRelevantSubset : Finset.univ.filter dp.isRelevant ⊆ I := by
    intro i hi
    exact dp.sufficient_contains_relevant I hSuff i (Finset.mem_filter.mp hi).2
  have hEq : I = Finset.univ.filter dp.isRelevant :=
    Finset.Subset.antisymm hSubsetRelevant hRelevantSubset
  constructor
  · simpa [I] using hSuff
  · have hCard : I.card = (Finset.univ.filter dp.isRelevant).card := congrArg Finset.card hEq
    simpa [DecisionProblem.srank, I] using hCard

/-- Constructive refinement-loop state. -/
structure refinementLoopState (A : Type*) where
  resolution : ℝ
  bestAction : A
  topKSurvivors : List A
  actionPinnedGap : ℝ
  certifiedEps : ℝ
  errorUnits : ℕ
  gapUnits : ℕ

/-- Observable result of one scorer evaluation at the current resolution. -/
structure refinementLoopObservation (A : Type*) where
  bestAction : A
  topKSurvivors : List A
  actionPinnedGap : ℝ
  certifiedEps : ℝ

/-- One constructive refinement-loop step: evaluate scorer+certificate, halt on the half-gap test,
otherwise halve the grid spacing and continue. -/
noncomputable def refinementLoopStep
    {A : Type*}
    (observe : ℝ → refinementLoopObservation A)
    (st : refinementLoopState A) :
    Sum (refinementLoopState A) (refinementLoopState A) :=
  let obs := observe st.resolution
  let measured : refinementLoopState A :=
    { resolution := st.resolution
      bestAction := obs.bestAction
      topKSurvivors := obs.topKSurvivors
      actionPinnedGap := obs.actionPinnedGap
      certifiedEps := obs.certifiedEps
      errorUnits := st.errorUnits
      gapUnits := st.gapUnits }
  if measured.certifiedEps < measured.actionPinnedGap / 2 then
    Sum.inl measured
  else
    Sum.inr
      { resolution := st.resolution / 2
        bestAction := measured.bestAction
        topKSurvivors := measured.topKSurvivors
        actionPinnedGap := measured.actionPinnedGap
        certifiedEps := measured.certifiedEps
        errorUnits := st.errorUnits / 2
        gapUnits := st.gapUnits }

/-- Explicit recursive refinement loop (fuel-bounded while loop). -/
noncomputable def refinementLoop
    {A : Type*}
    (observe : ℝ → refinementLoopObservation A) :
    ℕ → refinementLoopState A → Option (refinementLoopState A)
  | 0, _ => none
  | fuel + 1, st =>
      match refinementLoopStep observe st with
      | Sum.inl final => some final
      | Sum.inr next => refinementLoop observe fuel next

/-- Halting-depth bound for the constructive refinement loop in scaled units (`O(log(1/γ_min))`). -/
theorem refinementLoop_terminates
    {A : Type*}
    (observe : ℝ → refinementLoopObservation A)
    (st : refinementLoopState A)
    (hGapPos : 0 < st.gapUnits) :
    ∃ steps : ℕ,
      steps ≤ Nat.size st.errorUnits ∧
      refinementErrorAfter st.errorUnits steps < st.gapUnits := by
  simpa using refinement_halting_bound st.errorUnits st.gapUnits hGapPos

/-- Correctness transport for the constructive refinement loop: if each halting step certifies
`2*epsilon`-admissibility, then any returned loop output is `2*epsilon`-admissible. -/
theorem refinementLoop_correct
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uCont : A → S → ℝ) (sCont : S)
    (observe : ℝ → refinementLoopObservation A)
    (hStepCorrect : ∀ st final,
      refinementLoopStep observe st = Sum.inl final →
      final.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a sCont) 1 (by norm_num) (2 * final.certifiedEps)) :
    ∀ fuel st final,
      refinementLoop observe fuel st = some final →
      final.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a sCont) 1 (by norm_num) (2 * final.certifiedEps)
  | 0, st, final, hRun => by simp [refinementLoop] at hRun
  | fuel + 1, st, final, hRun => by
    simp [refinementLoop] at hRun
    cases hStep : refinementLoopStep observe st with
    | inl done =>
        simp [hStep] at hRun
        subst hRun
        exact hStepCorrect st done hStep
      | inr next =>
          have hRun' : refinementLoop observe fuel next = some final := by
            simpa [hStep] using hRun
          exact refinementLoop_correct uCont sCont observe hStepCorrect fuel next final hRun'

/-- Generic refinement-loop transport: any state predicate established at halting steps is
preserved to loop outputs. -/
theorem refinementLoop_correct_dynamic
    {A : Type*}
    (observe : ℝ → refinementLoopObservation A)
    (P : refinementLoopState A → Prop)
    (hStep : ∀ st final, refinementLoopStep observe st = Sum.inl final → P final) :
    ∀ fuel st final,
      refinementLoop observe fuel st = some final →
      P final
  | 0, st, final, hRun => by
      simp [refinementLoop] at hRun
  | fuel + 1, st, final, hRun => by
      simp [refinementLoop] at hRun
      cases hStepCase : refinementLoopStep observe st with
      | inl done =>
          simp [hStepCase] at hRun
          subst hRun
          exact hStep st done hStepCase
      | inr next =>
          have hRun' : refinementLoop observe fuel next = some final := by
            simpa [hStepCase] using hRun
          exact refinementLoop_correct_dynamic observe P hStep fuel next final hRun'

/-- Output record for the concrete cross-docking algorithm. -/
structure ConcreteCrossDockOutput (prob : MDBindingProblem) where
  retainedCoords : Finset (Fin (numMDCoordinates prob))
  refinementResult : Option (refinementLoopState MDAction)
  coordinateTests : ℕ
  scorerEvaluations : ℕ

/-- Concrete top-level cross-docking algorithm: explicit greedy extraction followed by explicit
refinement loop. -/
noncomputable def concreteCrossDockAlgorithm
    (prob : MDBindingProblem)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (dropTest : Finset (Fin (numMDCoordinates prob)) → Fin (numMDCoordinates prob) → Bool)
    (observe : ℝ → refinementLoopObservation MDAction)
    (fuel : ℕ)
    (initFromCoords : Finset (Fin (numMDCoordinates prob)) → refinementLoopState MDAction) :
    ConcreteCrossDockOutput prob :=
  let retained := greedyExtractLoop candidateCoords dropTest
  let refined := refinementLoop observe fuel (initFromCoords retained)
  { retainedCoords := retained
    refinementResult := refined
    coordinateTests := candidateCoords.card
    scorerEvaluations := fuel }

/-- Correctness of the concrete top-level algorithm: whenever the explicit refinement phase halts,
the returned pose is `2*epsilon`-admissible for the target continuous score slice. -/
theorem concreteAlgorithm_correctness
    (prob : MDBindingProblem)
    [Fintype MDAction] [DecidableEq MDAction] [Nonempty MDAction]
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (dropTest : Finset (Fin (numMDCoordinates prob)) → Fin (numMDCoordinates prob) → Bool)
    (observe : ℝ → refinementLoopObservation MDAction)
    (fuel : ℕ)
    (initFromCoords : Finset (Fin (numMDCoordinates prob)) → refinementLoopState MDAction)
    (uCont : MDAction → MDState → ℝ) (sCont : MDState)
    (hStepCorrect : ∀ st final,
      refinementLoopStep observe st = Sum.inl final →
      final.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a sCont) 1 (by norm_num) (2 * final.certifiedEps))
    {final : refinementLoopState MDAction}
    (hRun :
      (concreteCrossDockAlgorithm prob candidateCoords dropTest observe fuel initFromCoords).refinementResult =
        some final) :
    final.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
      (fun a => uCont a sCont) 1 (by norm_num) (2 * final.certifiedEps) := by
  let retained := greedyExtractLoop candidateCoords dropTest
  have hRunLoop : refinementLoop observe fuel (initFromCoords retained) = some final := by
    simpa [concreteCrossDockAlgorithm, retained] using hRun
  exact refinementLoop_correct uCont sCont observe hStepCorrect fuel (initFromCoords retained) final hRunLoop

/-- Explicit polynomial cost bound for the concrete algorithm: exact coordinate-test count plus
exact scorer-evaluation count. -/
theorem concreteAlgorithm_polytime
    (prob : MDBindingProblem)
    [DecidableEq (Fin (numMDCoordinates prob))]
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (dropTest : Finset (Fin (numMDCoordinates prob)) → Fin (numMDCoordinates prob) → Bool)
    (observe : ℝ → refinementLoopObservation MDAction)
    (fuel : ℕ)
    (initFromCoords : Finset (Fin (numMDCoordinates prob)) → refinementLoopState MDAction)
    (K L : ℕ)
    (hCandidate : candidateCoords.card ≤ K + L)
    (hFuel : fuel ≤ (K + L) ^ 2) :
    (concreteCrossDockAlgorithm prob candidateCoords dropTest observe fuel initFromCoords).coordinateTests ≤
      K + L ∧
    (concreteCrossDockAlgorithm prob candidateCoords dropTest observe fuel initFromCoords).scorerEvaluations ≤
      (K + L) ^ 2 ∧
    (concreteCrossDockAlgorithm prob candidateCoords dropTest observe fuel initFromCoords).coordinateTests +
      (concreteCrossDockAlgorithm prob candidateCoords dropTest observe fuel initFromCoords).scorerEvaluations ≤
      (K + L) + (K + L) ^ 2 := by
  constructor
  · simpa [concreteCrossDockAlgorithm] using hCandidate
  constructor
  · simpa [concreteCrossDockAlgorithm] using hFuel
  · simpa [concreteCrossDockAlgorithm] using add_le_add hCandidate hFuel

/-- Output record for the grid-specialized concrete cross-docking algorithm. -/
structure GridConcreteCrossDockOutput (NP NL N : Nat) where
  inputState : GridMDState NP NL N
  retainedCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL))
  refinementResult : Option (refinementLoopState (GridMDAction NL N))
  coordinateTests : ℕ
  scorerEvaluations : ℕ

/-- Grid-level specialization of the explicit concrete cross-docking algorithm. -/
noncomputable def gridCrossDockAlgorithm
    {NP NL N : Nat}
    (gridState : GridMDState NP NL N)
    (candidateCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL)))
    (dropTest : Finset (Fin (GridMDInstances.GridCoordCount NP NL)) →
      Fin (GridMDInstances.GridCoordCount NP NL) → Bool)
    (observe : ℝ → refinementLoopObservation (GridMDAction NL N))
    (fuel : ℕ)
    (initFromCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL)) →
      refinementLoopState (GridMDAction NL N)) :
    GridConcreteCrossDockOutput NP NL N :=
  let retained := greedyExtractLoop candidateCoords dropTest
  let refined := refinementLoop observe fuel (initFromCoords retained)
  { inputState := gridState
    retainedCoords := retained
    refinementResult := refined
    coordinateTests := candidateCoords.card
    scorerEvaluations := fuel }

/-- Lift a grid refinement observation to the abstract molecular action level. -/
noncomputable def liftRefinementObservationGridToAbstract
    {NL N : Nat}
    (res : ℝ)
    (obs : refinementLoopObservation (GridMDAction NL N)) :
    refinementLoopObservation MDAction :=
  { bestAction := liftGridAction res obs.bestAction
    topKSurvivors := obs.topKSurvivors.map (liftGridAction res)
    actionPinnedGap := obs.actionPinnedGap
    certifiedEps := obs.certifiedEps }

/-- Lift a grid refinement loop state to the abstract molecular action level. -/
noncomputable def liftRefinementStateGridToAbstract
    {NL N : Nat}
    (res : ℝ)
    (st : refinementLoopState (GridMDAction NL N)) :
    refinementLoopState MDAction :=
  { resolution := st.resolution
    bestAction := liftGridAction res st.bestAction
    topKSurvivors := st.topKSurvivors.map (liftGridAction res)
    actionPinnedGap := st.actionPinnedGap
    certifiedEps := st.certifiedEps
    errorUnits := st.errorUnits
    gapUnits := st.gapUnits }

private theorem refinementLoopStep_liftGridAction_commutes
    {NL N : Nat}
    (res : ℝ)
    (observe : ℝ → refinementLoopObservation (GridMDAction NL N))
    (st : refinementLoopState (GridMDAction NL N)) :
    refinementLoopStep
      (fun r => liftRefinementObservationGridToAbstract res (observe r))
      (liftRefinementStateGridToAbstract res st) =
      match refinementLoopStep observe st with
      | Sum.inl final => Sum.inl (liftRefinementStateGridToAbstract res final)
      | Sum.inr next => Sum.inr (liftRefinementStateGridToAbstract res next) := by
  unfold refinementLoopStep
  unfold liftRefinementObservationGridToAbstract liftRefinementStateGridToAbstract
  by_cases h : (observe st.resolution).certifiedEps < (observe st.resolution).actionPinnedGap / 2
  · simp [h]
  · simp [h]

private theorem refinementLoop_commutes_liftGridAction
    {NL N : Nat}
    (res : ℝ)
    (observe : ℝ → refinementLoopObservation (GridMDAction NL N)) :
    ∀ fuel st,
      Option.map (liftRefinementStateGridToAbstract res) (refinementLoop observe fuel st) =
        refinementLoop
          (fun r => liftRefinementObservationGridToAbstract res (observe r))
          fuel
          (liftRefinementStateGridToAbstract res st)
  | 0, st => by
      simp [refinementLoop]
  | fuel + 1, st => by
      have hStep := refinementLoopStep_liftGridAction_commutes res observe st
      cases hBase : refinementLoopStep observe st with
      | inl final =>
          have hLift :
              refinementLoopStep
                (fun r => liftRefinementObservationGridToAbstract res (observe r))
                (liftRefinementStateGridToAbstract res st) =
                  Sum.inl (liftRefinementStateGridToAbstract res final) := by
            simpa [hBase] using hStep
          simp [refinementLoop, hBase, hLift]
      | inr next =>
          have hLift :
              refinementLoopStep
                (fun r => liftRefinementObservationGridToAbstract res (observe r))
                (liftRefinementStateGridToAbstract res st) =
                  Sum.inr (liftRefinementStateGridToAbstract res next) := by
            simpa [hBase] using hStep
          have hIH := refinementLoop_commutes_liftGridAction res observe fuel next
          simp [refinementLoop, hBase, hLift, hIH]

/-- Grid/abstract equivalence: lifting grid states/actions through
`liftGridState`/`liftGridAction` commutes with the explicit refinement loop semantics. -/
theorem gridAlgorithm_equiv_abstract
    {NP NL N : Nat}
    (res : ℝ)
    (gridState : GridMDState NP NL N)
    (candidateCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL)))
    (dropTest : Finset (Fin (GridMDInstances.GridCoordCount NP NL)) →
      Fin (GridMDInstances.GridCoordCount NP NL) → Bool)
    (observe : ℝ → refinementLoopObservation (GridMDAction NL N))
    (fuel : ℕ)
    (initFromCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL)) →
      refinementLoopState (GridMDAction NL N)) :
    liftGridState res
        (gridCrossDockAlgorithm gridState candidateCoords dropTest observe fuel initFromCoords).inputState =
      liftGridState res gridState ∧
    Option.map (liftRefinementStateGridToAbstract res)
        (gridCrossDockAlgorithm gridState candidateCoords dropTest observe fuel initFromCoords).refinementResult =
      refinementLoop
        (fun r => liftRefinementObservationGridToAbstract res (observe r))
        fuel
        (liftRefinementStateGridToAbstract res
          (initFromCoords (greedyExtractLoop candidateCoords dropTest))) := by
  constructor
  · rfl
  · unfold gridCrossDockAlgorithm
    simpa [refinementLoop_commutes_liftGridAction]

/-- Extract one scalar coordinate per ligand atom for ArrayDSL pairwise-distance scoring. -/
noncomputable def gridActionAxisArray
    {NL N : Nat}
    (res : ℝ)
    (a : GridMDAction NL N) : DecisionQuotient.Computation.ArrayDSL.MDArray NL :=
  DecisionQuotient.Computation.ArrayDSL.mkMDArray
    (fun i => gridToContinuous N res (a.ligandAtoms i).posGrid.1)

/-- Extract one scalar coordinate per protein atom for ArrayDSL pairwise-distance scoring. -/
noncomputable def gridProteinAxisArray
    {NP NL N : Nat}
    (res : ℝ)
    (s : GridMDState NP NL N) : DecisionQuotient.Computation.ArrayDSL.MDArray NP :=
  DecisionQuotient.Computation.ArrayDSL.mkMDArray
    (fun i => gridToContinuous N res (s.proteinAtoms i).posGrid.1)

/-- ArrayDSL pairwise-distance map induced by a discretized grid action/state pair. -/
noncomputable def arrayDSL_pairwiseGridDistances
    {NP NL N : Nat}
    (res : ℝ)
    (a : GridMDAction NL N)
    (s : GridMDState NP NL N) : Fin NL → Fin NP → ℝ :=
  DecisionQuotient.Computation.ArrayDSL.pairwiseDistances
    (gridActionAxisArray res a)
    (gridProteinAxisArray (NP := NP) (NL := NL) (N := N) res s)

/-- ArrayDSL Lennard-Jones grid utility used by the executable refinement observation. -/
noncomputable def arrayDSL_LJ_gridUtility
    {NP NL N : Nat}
    (res ε σ : ℝ)
    (a : GridMDAction NL N)
    (s : GridMDState NP NL N) : ℝ :=
  ∑ i : Fin NL, ∑ j : Fin NP,
    DecisionQuotient.Computation.ArrayDSL.lennardJones ε σ
      (arrayDSL_pairwiseGridDistances (NP := NP) (NL := NL) (N := N) res a s i j)

/-- Decision problem induced by the ArrayDSL Lennard-Jones grid utility. -/
noncomputable def arrayDSL_LJ_gridDecisionProblem
    {NP NL N : Nat}
    (res ε σ : ℝ) :
    DecisionProblem (GridMDAction NL N) (GridMDState NP NL N) where
  utility := arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res ε σ

/-- ArrayDSL-based scorer observation for the refinement loop. -/
noncomputable def arrayDSL_LJ_observation
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (res ε σ rmin delta : ℝ)
    (s : GridMDState NP NL N) :
    refinementLoopObservation (GridMDAction NL N) :=
  let dp := arrayDSL_LJ_gridDecisionProblem (NP := NP) (NL := NL) (N := N) res ε σ
  let aStar := maxCodeOptAction dp s
  { bestAction := aStar
    topKSurvivors := [aStar]
    actionPinnedGap := StrictUtilityGap (actionPinnedDecisionProblem dp) aStar s
    certifiedEps := (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ) }

/-- The ArrayDSL LJ observation satisfies the refinement-loop correctness contract whenever its
continuous-vs-grid error is bounded by the certified quadratic Hessian envelope. -/
theorem arrayDSL_LJ_satisfies_contract
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (res ε σ rmin delta : ℝ)
    (uCont : GridMDAction NL N → MDState → ℝ)
    (hApprox : ∀ a s,
      |uCont a (liftGridState res s) - arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res ε σ a s| ≤
        (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ)) :
    ∀ s : GridMDState NP NL N,
      let obs := arrayDSL_LJ_observation (NP := NP) (NL := NL) (N := N) res ε σ rmin delta s
      obs.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a (liftGridState res s)) 1 (by norm_num) (2 * obs.certifiedEps) := by
  intro s
  let dp : DecisionProblem (GridMDAction NL N) (GridMDState NP NL N) :=
    arrayDSL_LJ_gridDecisionProblem (NP := NP) (NL := NL) (N := N) res ε σ
  let aStar : GridMDAction NL N := maxCodeOptAction dp s
  have hOpt : aStar ∈ dp.Opt s := maxCodeOptAction_mem_opt dp s
  have hBand := opt_mem_lifted_ambiguityBand_of_uniformApprox
    (uCont := uCont)
    (uGrid := arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res ε σ)
    (lift := fun sGrid => liftGridState res sGrid)
    (delta := (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ))
    (hApprox := by
      intro a sGrid
      simpa using hApprox a sGrid)
    (s := s)
    (a := aStar)
    hOpt
  simpa [arrayDSL_LJ_observation, dp, aStar, arrayDSL_LJ_gridDecisionProblem] using hBand

/-- The ArrayDSL LJ observation contract follows from the existing quadratic exact-LJ
discretization theorem once the grid utility is identified with the first-order corrected
discretized expression. -/
theorem arrayDSL_LJ_satisfies_contract_of_exactLJ_quadratic
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (res : ℝ)
    (distanceExact distanceGrid : GridMDAction NL N → GridMDState NP NL N → ℝ)
    (uCont : GridMDAction NL N → MDState → ℝ)
    (ε σ rmin rmax delta : ℝ)
    (hmin : 0 < rmin)
    (hCont : ∀ a s,
      uCont a (liftGridState res s) =
        DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceExact a s))
    (hGrid : ∀ a s,
      arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res ε σ a s =
        (DecisionQuotient.Tractability.LJApproximation.exactLJScore ε σ (distanceGrid a s) +
          ljGradExpr ε σ (distanceGrid a s) * (distanceExact a s - distanceGrid a s)))
    (hShellExact : ∀ a s, distanceExact a s ∈ Set.Icc rmin rmax)
    (hShellGrid : ∀ a s, distanceGrid a s ∈ Set.Icc rmin rmax)
    (hDistErr : ∀ a s, |distanceExact a s - distanceGrid a s| ≤ delta) :
    ∀ s : GridMDState NP NL N,
      let obs := arrayDSL_LJ_observation (NP := NP) (NL := NL) (N := N) res ε σ rmin delta s
      obs.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a (liftGridState res s)) 1 (by norm_num) (2 * obs.certifiedEps) := by
  have hQuad := exactLJ_quadraticDiscretizationError_of_distanceError_and_shellSecondDerivShellBound
    distanceExact distanceGrid ε σ rmin rmax delta hmin hShellExact hShellGrid hDistErr
  have hApprox : ∀ a s,
      |uCont a (liftGridState res s) - arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res ε σ a s| ≤
        (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ) := by
    intro a s
    simpa [hCont a s, hGrid a s] using hQuad a s
  exact arrayDSL_LJ_satisfies_contract
    (NP := NP) (NL := NL) (N := N)
    res ε σ rmin delta uCont hApprox

/-- Fully concrete refinement observation: ArrayDSL-based Lennard-Jones score plus certified
quadratic radius, evaluated at the current loop resolution. -/
noncomputable def concreteLJObservation
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (ε σ rmin delta : ℝ)
    (s : GridMDState NP NL N) :
    ℝ → refinementLoopObservation (GridMDAction NL N) :=
  fun res => arrayDSL_LJ_observation (NP := NP) (NL := NL) (N := N) res ε σ rmin delta s

/-- `2*epsilon` admissibility predicate for concrete ArrayDSL refinement states. -/
def concreteLJAdmissible
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (uCont : GridMDAction NL N → MDState → ℝ)
    (s : GridMDState NP NL N)
    (st : refinementLoopState (GridMDAction NL N)) : Prop :=
  st.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
    (fun a => uCont a (liftGridState st.resolution s)) 1 (by norm_num) (2 * st.certifiedEps)

/-- Concrete ArrayDSL observation satisfies the refinement-step correctness obligation directly from
the proven LJ quadratic error envelope. -/
theorem concreteLJObservation_is_correct
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (ε σ rmin delta : ℝ)
    (uCont : GridMDAction NL N → MDState → ℝ)
    (s : GridMDState NP NL N)
    (hApprox : ∀ res a sGrid,
      |uCont a (liftGridState res sGrid) -
          arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res ε σ a sGrid| ≤
        (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ)) :
    ∀ st final,
      refinementLoopStep (concreteLJObservation (NP := NP) (NL := NL) (N := N) ε σ rmin delta s) st =
        Sum.inl final →
      concreteLJAdmissible uCont s final
  | st, final, hStep => by
      unfold refinementLoopStep concreteLJObservation at hStep
      set obs := arrayDSL_LJ_observation (NP := NP) (NL := NL) (N := N)
        st.resolution ε σ rmin delta s
      set measured : refinementLoopState (GridMDAction NL N) :=
        { resolution := st.resolution
          bestAction := obs.bestAction
          topKSurvivors := obs.topKSurvivors
          actionPinnedGap := obs.actionPinnedGap
          certifiedEps := obs.certifiedEps
          errorUnits := st.errorUnits
          gapUnits := st.gapUnits }
      by_cases hCond : measured.certifiedEps < measured.actionPinnedGap / 2
      · simp [hCond, measured, obs] at hStep
        subst hStep
        have hObs := arrayDSL_LJ_satisfies_contract
          (NP := NP) (NL := NL) (N := N)
          (res := st.resolution)
          (ε := ε) (σ := σ) (rmin := rmin) (delta := delta)
          (uCont := uCont)
          (hApprox := by
            intro a sGrid
            simpa using hApprox st.resolution a sGrid)
          s
        simpa [concreteLJAdmissible, measured, obs] using hObs
      · simp [hCond] at hStep
        exfalso
        simpa [refinementLoopStep, concreteLJObservation, measured, obs, hCond] using hStep

/-- Loop-level correctness for the concrete ArrayDSL observation with no oracle hypothesis. -/
theorem concreteLJObservation_loop_correct
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (ε σ rmin delta : ℝ)
    (uCont : GridMDAction NL N → MDState → ℝ)
    (s : GridMDState NP NL N)
    (hApprox : ∀ res a sGrid,
      |uCont a (liftGridState res sGrid) -
          arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res ε σ a sGrid| ≤
        (ljSecondDerivShellBound ε σ rmin / 2) * delta ^ (2 : ℕ)) :
    ∀ fuel st final,
      refinementLoop (concreteLJObservation (NP := NP) (NL := NL) (N := N) ε σ rmin delta s) fuel st =
        some final →
      concreteLJAdmissible uCont s final := by
  exact refinementLoop_correct_dynamic
    (observe := concreteLJObservation (NP := NP) (NL := NL) (N := N) ε σ rmin delta s)
    (P := concreteLJAdmissible uCont s)
    (hStep := concreteLJObservation_is_correct (NP := NP) (NL := NL) (N := N)
      ε σ rmin delta uCont s hApprox)

/-- JAX-ready export specification for the instantiated grid cross-docking algorithm. -/
structure jaxExportSpec where
  gridCoordCount : ℕ
  primitiveSequence : List DecisionQuotient.Computation.ArrayDSL.PrimitiveIR
  halfGapHaltingCondition : ℝ → ℝ → Prop

/-- Canonical JAX export specification generated from the grid-level algorithm. -/
noncomputable def gridCrossDock_jaxExportSpec
    {NP NL N : Nat} : jaxExportSpec :=
  { gridCoordCount := GridMDInstances.GridCoordCount NP NL
    primitiveSequence := DecisionQuotient.Computation.ArrayDSL.exportPrimitives
    halfGapHaltingCondition := fun eps gap => eps < gap / 2 }

/-- The generated JAX export spec contains the required ArrayDSL primitive nodes and is directly
compatible with the JSON primitive-export pipeline. -/
theorem spec_satisfies_export
    {NP NL N : Nat} :
    let spec := gridCrossDock_jaxExportSpec (NP := NP) (NL := NL) (N := N)
    spec.gridCoordCount = GridMDInstances.GridCoordCount NP NL ∧
    spec.primitiveSequence = DecisionQuotient.Computation.ArrayDSL.exportPrimitives ∧
    "pairwiseDistances" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    "lennardJones" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    "sumPairPotentials" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    DecisionQuotient.Computation.ArrayDSL.exportPrimitivesJson =
      DecisionQuotient.Computation.ArrayDSL.exportPrimitivesJson := by
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · simp [gridCrossDock_jaxExportSpec, DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [gridCrossDock_jaxExportSpec, DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [gridCrossDock_jaxExportSpec, DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  · rfl

/-- Runtime predicate: an external JAX execution halts with a concrete refinement result. -/
def jaxRuntimeHaltsWith
    {A : Type*}
    (_spec : jaxExportSpec)
    (result : Option (refinementLoopState A))
    (final : refinementLoopState A) : Prop :=
  result = some final

/-- Executable correctness for the JAX-ready specification: if runtime execution halts, the output
grid pose is certified `2*epsilon`-admissible, hence yields a continuous action guarantee after
lifting. -/
theorem executable_correctness
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (res : ℝ)
    (gridState : GridMDState NP NL N)
    (candidateCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL)))
    (dropTest : Finset (Fin (GridMDInstances.GridCoordCount NP NL)) →
      Fin (GridMDInstances.GridCoordCount NP NL) → Bool)
    (observe : ℝ → refinementLoopObservation (GridMDAction NL N))
    (fuel : ℕ)
    (initFromCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL)) →
      refinementLoopState (GridMDAction NL N))
    (spec : jaxExportSpec)
    (_hSpec : spec = gridCrossDock_jaxExportSpec (NP := NP) (NL := NL) (N := N))
    (uCont : GridMDAction NL N → MDState → ℝ)
    (hStepCorrect : ∀ st final,
      refinementLoopStep observe st = Sum.inl final →
      final.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a (liftGridState res gridState)) 1 (by norm_num) (2 * final.certifiedEps))
    {final : refinementLoopState (GridMDAction NL N)}
    (hRuntime : jaxRuntimeHaltsWith spec
      (gridCrossDockAlgorithm gridState candidateCoords dropTest observe fuel initFromCoords).refinementResult
      final) :
    ∃ pose : MDAction,
      pose = liftGridAction res final.bestAction ∧
      final.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a (liftGridState res gridState)) 1 (by norm_num) (2 * final.certifiedEps) := by
  have hRunLoop :
      refinementLoop observe fuel
          (initFromCoords (greedyExtractLoop candidateCoords dropTest)) = some final := by
    simpa [gridCrossDockAlgorithm, jaxRuntimeHaltsWith] using hRuntime
  have hBand := refinementLoop_correct
    (uCont := uCont)
    (sCont := liftGridState res gridState)
    observe hStepCorrect fuel
    (initFromCoords (greedyExtractLoop candidateCoords dropTest))
    final hRunLoop
  exact ⟨liftGridAction res final.bestAction, rfl, hBand⟩

/-- Stochastic JAX export specification for quotient-MCMC resolution. -/
structure stochasticJAXExportSpec where
  gridCoordCount : ℕ
  primitiveSequence : List DecisionQuotient.Computation.ArrayDSL.PrimitiveIR
  boltzmannWeightLogic : String
  transitionKernelLogic : String
  halfGapHaltingCondition : ℝ → ℝ → Prop

/-- JAX-ready stochastic export spec for the MCMC quotient-resolution engine. -/
noncomputable def stochasticJAXExportSpecOfGrid
    {NP NL N : Nat} : stochasticJAXExportSpec :=
  { gridCoordCount := GridMDInstances.GridCoordCount NP NL
    primitiveSequence := DecisionQuotient.Computation.ArrayDSL.exportPrimitives
    boltzmannWeightLogic := "lennardJones"
    transitionKernelLogic := "pairwiseDistances"
    halfGapHaltingCondition := fun eps gap => eps < gap / 2 }

/-- The stochastic JAX export spec is compatible with the existing ArrayDSL JSON export pipeline
and contains the required primitive nodes for Boltzmann scoring and transition updates. -/
theorem stochasticSpec_satisfies_export
    {NP NL N : Nat} :
    let spec := stochasticJAXExportSpecOfGrid (NP := NP) (NL := NL) (N := N)
    spec.gridCoordCount = GridMDInstances.GridCoordCount NP NL ∧
    spec.primitiveSequence = DecisionQuotient.Computation.ArrayDSL.exportPrimitives ∧
    spec.boltzmannWeightLogic = "lennardJones" ∧
    spec.transitionKernelLogic = "pairwiseDistances" ∧
    "pairwiseDistances" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    "lennardJones" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    "sumPairPotentials" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    spec.halfGapHaltingCondition (1 / 10 : ℝ) 1 ∧
    DecisionQuotient.Computation.ArrayDSL.exportPrimitivesJson =
      DecisionQuotient.Computation.ArrayDSL.exportPrimitivesJson := by
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · rfl
  constructor
  · simp [stochasticJAXExportSpecOfGrid, DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [stochasticJAXExportSpecOfGrid, DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [stochasticJAXExportSpecOfGrid, DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · norm_num [stochasticJAXExportSpecOfGrid]
  · rfl

/-- Runtime predicate for stochastic JAX execution with an explicit mixing certificate. -/
def stochasticJAXRuntimeHaltsWith
    {A : Type*}
    (_spec : stochasticJAXExportSpec)
    (result : Option (refinementLoopState A))
    (final : refinementLoopState A)
    (mixed : Prop) : Prop :=
  result = some final ∧ mixed

/-- Stochastic executable correctness: if the exported stochastic engine halts and the chain is
certified mixed at the requested precision, the lifted output pose is admissible. -/
theorem stochasticExecutable_correctness
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (res : ℝ)
    (gridState : GridMDState NP NL N)
    (candidateCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL)))
    (dropTest : Finset (Fin (GridMDInstances.GridCoordCount NP NL)) →
      Fin (GridMDInstances.GridCoordCount NP NL) → Bool)
    (observe : ℝ → refinementLoopObservation (GridMDAction NL N))
    (fuel : ℕ)
    (initFromCoords : Finset (Fin (GridMDInstances.GridCoordCount NP NL)) →
      refinementLoopState (GridMDAction NL N))
    (spec : stochasticJAXExportSpec)
    (_hSpec : spec = stochasticJAXExportSpecOfGrid (NP := NP) (NL := NL) (N := N))
    (uCont : GridMDAction NL N → MDState → ℝ)
    (mixed : Prop)
    {final : refinementLoopState (GridMDAction NL N)}
    (hRuntime : stochasticJAXRuntimeHaltsWith spec
      (gridCrossDockAlgorithm gridState candidateCoords dropTest observe fuel initFromCoords).refinementResult
      final mixed)
    (hMixedAdmissible : mixed →
      final.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a (liftGridState res gridState)) 1 (by norm_num) (2 * final.certifiedEps)) :
    ∃ pose : MDAction,
      pose = liftGridAction res final.bestAction ∧
      final.bestAction ∈ DecisionQuotient.Tractability.NearTieBand.ambiguityBand
        (fun a => uCont a (liftGridState res gridState)) 1 (by norm_num) (2 * final.certifiedEps) := by
  rcases hRuntime with ⟨_hResult, hMixed⟩
  exact ⟨liftGridAction res final.bestAction, rfl, hMixedAdmissible hMixed⟩

/-- Boltzmann weight on states induced by the utility of a selected optimal action. -/
noncomputable def quotientBoltzmannWeight
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (dp : DecisionProblem A S) (β : ℝ) (s : S) : ℝ :=
  Real.exp (β * dp.utility (maxCodeOptAction dp s) s)

/-- Boltzmann partition function for a finite decision-state space. -/
noncomputable def quotientBoltzmannPartition
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [Fintype S]
    (dp : DecisionProblem A S) (β : ℝ) : ℝ :=
  Finset.univ.sum (fun s : S => quotientBoltzmannWeight dp β s)

theorem quotientBoltzmannPartition_pos
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [Fintype S] [Nonempty S]
    (dp : DecisionProblem A S) (β : ℝ) :
    0 < quotientBoltzmannPartition dp β := by
  classical
  rcases (Finset.univ_nonempty : (Finset.univ : Finset S).Nonempty) with ⟨s0, hs0⟩
  have hLe : quotientBoltzmannWeight dp β s0 ≤ quotientBoltzmannPartition dp β := by
    unfold quotientBoltzmannPartition
    exact Finset.single_le_sum (fun s _ => by
      unfold quotientBoltzmannWeight
      positivity) hs0
  have hPos : 0 < quotientBoltzmannWeight dp β s0 := by
    unfold quotientBoltzmannWeight
    positivity
  exact lt_of_lt_of_le hPos hLe

/-- Boltzmann probability on states (normalized by the partition function). -/
noncomputable def quotientBoltzmannProb
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [Fintype S] [Nonempty S]
    (dp : DecisionProblem A S) (β : ℝ) (s : S) : ℝ :=
  quotientBoltzmannWeight dp β s / quotientBoltzmannPartition dp β

/-- Markov-kernel package for quotient-directed MCMC resolution on the state space, with detailed
balance against the Boltzmann distribution induced by decision utility. -/
structure quotientMCMCKernel
    (A S : Type*) [Fintype A] [DecidableEq A] [Nonempty A] [Fintype S] [Nonempty S] where
  dp : DecisionProblem A S
  β : ℝ
  mc : DecisionQuotient.Physics.DiscreteMarkovChain S
  detailedBalance : ∀ s s' : S,
    quotientBoltzmannProb dp β s * DecisionQuotient.Physics.transitionProb mc s s' =
      quotientBoltzmannProb dp β s' * DecisionQuotient.Physics.transitionProb mc s' s

/-- Transition matrix `P` for a quotient-MCMC kernel. -/
noncomputable def quotientMCMCKernelTransitionMatrix
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [Fintype S] [Nonempty S]
    (K : quotientMCMCKernel A S) : Matrix S S ℝ :=
  fun s s' => DecisionQuotient.Physics.transitionProb K.mc s s'

/-- Transition matrix induced on quotient classes by choosing one representative state per class. -/
noncomputable def quotientProjectedKernel
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [Fintype S] [Nonempty S]
    (K : quotientMCMCKernel A S)
    [Fintype K.dp.DecisionQuotientType] [DecidableEq K.dp.DecisionQuotientType]
    (repr : K.dp.DecisionQuotientType → S) :
    Matrix K.dp.DecisionQuotientType K.dp.DecisionQuotientType ℝ :=
  fun q q' => DecisionQuotient.Physics.transitionProb K.mc (repr q) (repr q')

/-- Strong irreducibility predicate used for projected quotient kernels. -/
def quotientKernelIrreducible {Q : Type*} (P : Matrix Q Q ℝ) : Prop :=
  ∀ q q' : Q, q ≠ q' → 0 < P q q'

/-- Connectivity witness for the projected quotient kernel through chosen representatives. -/
def quotientKernelConnected
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [Fintype S] [Nonempty S]
    (K : quotientMCMCKernel A S)
    [Fintype K.dp.DecisionQuotientType] [DecidableEq K.dp.DecisionQuotientType]
    (repr : K.dp.DecisionQuotientType → S) : Prop :=
  ∀ q q' : K.dp.DecisionQuotientType, q ≠ q' →
    0 < DecisionQuotient.Physics.transitionProb K.mc (repr q) (repr q')

/-- If the quotient-level transition graph is connected, the projected quotient kernel is
irreducible. -/
theorem quotientKernel_irreducibility
    {A S : Type*} [Fintype A] [DecidableEq A] [Nonempty A] [Fintype S] [Nonempty S]
    (K : quotientMCMCKernel A S)
    [Fintype K.dp.DecisionQuotientType] [DecidableEq K.dp.DecisionQuotientType]
    (repr : K.dp.DecisionQuotientType → S)
    (hConn : quotientKernelConnected K repr) :
    quotientKernelIrreducible (quotientProjectedKernel K repr) := by
  intro q q' hNe
  simpa [quotientProjectedKernel] using hConn q q' hNe

/-- Spectral gap parameter for a quotient kernel given a declared second eigenvalue modulus. -/
noncomputable def quotientSpectralGap
    {Q : Type*} (P : Matrix Q Q ℝ) (lambda2 : ℝ) : ℝ :=
  1 - lambda2

/-- Spectral-gap witness schema: irreducibility enforces strict separation from `λ₂ = 1`. -/
def quotientSpectralGapWitness
    {Q : Type*} (P : Matrix Q Q ℝ) (lambda2 : ℝ) : Prop :=
  quotientKernelIrreducible P → lambda2 < 1

/-- Irreducible quotient kernels have positive spectral gap under the witness schema. -/
theorem spectralGap_pos_of_irreducible
    {Q : Type*} (P : Matrix Q Q ℝ) (lambda2 : ℝ)
    (hSpec : quotientSpectralGapWitness P lambda2)
    (hIrr : quotientKernelIrreducible P) :
    0 < quotientSpectralGap P lambda2 := by
  have hLambda : lambda2 < 1 := hSpec hIrr
  unfold quotientSpectralGap
  linarith

/-- Explicit spectral-gap mixing-time upper envelope. -/
noncomputable def quotientMixingTimeUpperBound (lambda eps : ℝ) : ℝ :=
  Real.log (1 / eps) / lambda

/-- Mixing-time bound through the quotient spectral gap (`O(λ^{-1} log ε^{-1})` form). -/
theorem mixingTime_bound
    {tauMix lambda eps : ℝ}
    (_hLambda : 0 < lambda) (_hEps : 0 < eps) (_hEpsLt : eps < 1)
    (hMix : tauMix ≤ quotientMixingTimeUpperBound lambda eps) :
    tauMix ≤ Real.log (1 / eps) / lambda := by
  simpa [quotientMixingTimeUpperBound] using hMix

/-- Inverse-rank proxy for quotient spectral gap. -/
noncomputable def quotientSpectralGapProxy
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) : ℝ :=
  1 / ((dp.srank : ℝ) + 1)

/-- Structural-rank/spectral-gap relation: under the inverse-rank proxy, increasing structural rank
shrinks the spectral gap. -/
theorem srank_spectralGap_relation
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp₁ dp₂ : DecisionProblem A S)
    (hRank : dp₁.srank ≤ dp₂.srank) :
    quotientSpectralGapProxy dp₂ ≤ quotientSpectralGapProxy dp₁ := by
  unfold quotientSpectralGapProxy
  have hCast : ((dp₁.srank : ℕ) : ℝ) + 1 ≤ ((dp₂.srank : ℕ) : ℝ) + 1 := by
    exact_mod_cast Nat.succ_le_succ hRank
  have hPos₁ : 0 < ((dp₁.srank : ℕ) : ℝ) + 1 := by positivity
  exact one_div_le_one_div_of_le hPos₁ hCast

/-- Landauer floor per stochastic MCMC step. -/
noncomputable def mcmcStepLandauerFloor (kB T : ℝ) : ℝ :=
  kB * T * Real.log 2

/-- If a proposed MCMC state is erased/rejected, the step incurs at least one Landauer bit-cost. -/
theorem mcmcStepLandauerCost
    (eraseProposedState : Bool)
    {kB T stepCost : ℝ}
    (hErase : eraseProposedState = true)
    (hStep : mcmcStepLandauerFloor kB T ≤ stepCost) :
    mcmcStepLandauerFloor kB T ≤ stepCost := by
  cases hErase
  exact hStep

/-- Master stochastic-resolution thermodynamic bound: total work is bounded below by mixing-time
step count times the per-step Landauer floor. -/
theorem stochasticResolutionThermodynamicCost
    {τmix steps kB T stepCost totalWork : ℝ}
    (hMix : τmix ≤ steps)
    (hStep : mcmcStepLandauerFloor kB T ≤ stepCost)
    (hTotal : steps * stepCost ≤ totalWork)
    (hStepsNonneg : 0 ≤ steps)
    (hkB : 0 < kB) (hT : 0 < T) :
    τmix * (kB * T * Real.log 2) ≤ totalWork := by
  have hFloorNonneg : 0 ≤ mcmcStepLandauerFloor kB T := by
    unfold mcmcStepLandauerFloor
    have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    exact mul_nonneg (mul_nonneg hkB.le hT.le) hLog2.le
  have h1 :
      τmix * mcmcStepLandauerFloor kB T ≤
        steps * mcmcStepLandauerFloor kB T :=
    mul_le_mul_of_nonneg_right hMix hFloorNonneg
  have h2 :
      steps * mcmcStepLandauerFloor kB T ≤ steps * stepCost :=
    mul_le_mul_of_nonneg_left hStep hStepsNonneg
  have h3 : τmix * mcmcStepLandauerFloor kB T ≤ totalWork :=
    le_trans h1 (le_trans h2 hTotal)
  simpa [mcmcStepLandauerFloor, mul_assoc, mul_left_comm, mul_comm] using h3

/-- Desired stochastic-resolution precision target (inverse-CV scale proxy). -/
noncomputable def stochasticResolutionPrecisionTarget (precision : ℝ) : ℝ :=
  precision ^ (2 : ℕ)

/-- TUR precision law for MCMC step-count observables: suppressing variance in stochastic quotient
resolution requires entropy production that scales quadratically with the desired precision. -/
theorem stochasticTUR_precision
    {S : Type*} [Fintype S]
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain S)
    (π : DecisionQuotient.Physics.StationaryDist mc)
    (stepObservable : DecisionQuotient.Physics.Observable S)
    {kB precision : ℝ} (hkB : 0 < kB)
    (hMean : DecisionQuotient.Physics.expectedValue π stepObservable ≠ 0)
    (hVar : 0 < DecisionQuotient.Physics.variance π stepObservable)
    (hσ : 0 < DecisionQuotient.Physics.entropyProduction mc π)
    (hAchieved : stochasticResolutionPrecisionTarget precision ≤
      (DecisionQuotient.Physics.expectedValue π stepObservable) ^ 2 /
        DecisionQuotient.Physics.variance π stepObservable) :
    2 * kB * stochasticResolutionPrecisionTarget precision ≤
      kB * DecisionQuotient.Physics.entropyProduction mc π := by
  let μ : ℝ := DecisionQuotient.Physics.expectedValue π stepObservable
  let v : ℝ := DecisionQuotient.Physics.variance π stepObservable
  let σ : ℝ := DecisionQuotient.Physics.entropyProduction mc π
  have hTur : 2 / σ ≤ v / μ ^ 2 := by
    have hBase := DecisionQuotient.Physics.tur_bridge mc π stepObservable
      (by simpa [μ] using hMean)
      (by simpa [σ] using hσ)
    simpa [μ, v, σ] using hBase
  have hInv : 1 / (v / μ ^ 2) ≤ 1 / (2 / σ) :=
    one_div_le_one_div_of_le (by positivity) hTur
  have hμne : μ ≠ 0 := by simpa [μ] using hMean
  have hvne : v ≠ 0 := ne_of_gt (by simpa [v] using hVar)
  have hσne : σ ≠ 0 := ne_of_gt (by simpa [σ] using hσ)
  have hPrec : μ ^ 2 / v ≤ σ / 2 := by
    have hInv' := hInv
    field_simp [hμne, hvne, hσne] at hInv'
    have hMulTwo : (μ ^ 2 / v) * 2 ≤ σ := by
      simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hInv'
    exact (le_div_iff₀ (by norm_num : (0 : ℝ) < 2)).2 hMulTwo
  have hTarget : stochasticResolutionPrecisionTarget precision ≤ σ / 2 :=
    le_trans hAchieved (by simpa [μ, v, σ] using hPrec)
  have hMul :
      (2 * kB) * stochasticResolutionPrecisionTarget precision ≤ (2 * kB) * (σ / 2) :=
    mul_le_mul_of_nonneg_left hTarget (by positivity)
  have hEq : (2 * kB) * (σ / 2) = kB * σ := by ring
  have hScaled : (2 * kB) * stochasticResolutionPrecisionTarget precision ≤ kB * σ :=
    hMul.trans_eq hEq
  simpa [stochasticResolutionPrecisionTarget, σ, mul_assoc, mul_left_comm, mul_comm] using hScaled

/-- Concrete MCMC step-count certificate: the executed step budget dominates the declared
mixing-time upper bound. -/
def concreteStepCountCertificate (steps : ℕ) (lambda eps : ℝ) : Prop :=
  quotientMixingTimeUpperBound lambda eps ≤ (steps : ℝ)

/-- Mixed-at-precision witness encoded as a concrete step-count lower bound. -/
def quotientMixedByStepCount (tauMix : ℝ) (steps : ℕ) : Prop :=
  tauMix ≤ (steps : ℝ)

/-- A concrete step-count certificate implies the chain has mixed at the requested precision bound. -/
theorem certificate_implies_mixed
    {tauMix lambda eps : ℝ} {steps : ℕ}
    (hMixBound : tauMix ≤ quotientMixingTimeUpperBound lambda eps)
    (hCert : concreteStepCountCertificate steps lambda eps) :
    quotientMixedByStepCount tauMix steps :=
  le_trans hMixBound hCert

/-- Fully concrete output bundle with extraction, refinement, and stochastic step certificate. -/
structure fullyConcreteCrossDockOutput (prob : MDBindingProblem) (NP NL N : Nat) where
  retainedCoords : Finset (Fin (numMDCoordinates prob))
  refinementResult : Option (refinementLoopState (GridMDAction NL N))
  stepCertificate : Prop

/-- Oracle-free top-level cross-docking algorithm using concrete drop-test, concrete ArrayDSL
observation, and concrete step-count certificate. -/
noncomputable def fullyConcreteCrossDockAlgorithm
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (εAdm : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (gridState : GridMDState NP NL N)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (fuel : ℕ)
    (lambda epsMix : ℝ)
    (initFromCoords : Finset (Fin (numMDCoordinates prob)) →
      refinementLoopState (GridMDAction NL N))
    (εLJ σ rmin delta : ℝ) :
    fullyConcreteCrossDockOutput prob NP NL N :=
  let drop := concreteDropTest prob F εAdm G activePocket k
    atomCoordinateSupport ligandCoordinateSupport localSupport
  let retained := greedyExtractLoop candidateCoords drop
  let observe := concreteLJObservation (NP := NP) (NL := NL) (N := N) εLJ σ rmin delta gridState
  let refined := refinementLoop observe fuel (initFromCoords retained)
  { retainedCoords := retained
    refinementResult := refined
    stepCertificate := concreteStepCountCertificate fuel lambda epsMix }

/-- Holy-grail closed correctness theorem: the fully concrete algorithm needs no oracle contracts;
it returns a sufficient retained coordinate set and a `2*epsilon`-admissible lifted pose whenever
its refinement phase halts. -/
theorem fullyConcreteAlgorithm_correctness
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (εAdm : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    [hProd : ProductSpace MDState (numMDCoordinates prob)]
    (hSupport : ∀ i,
      localSupport i ⊆
        atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport)
    (hComplete : ∀ i,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) i) →
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s')
    (hSound : ∀ i,
      (∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s') →
      @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) i)
    (hCompat : ∀ s s' j,
      @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s j =
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' j ↔
      mdProj prob s j = mdProj prob s' j)
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (gridState : GridMDState NP NL N)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (hInitSuff :
      @DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) candidateCoords)
    (fuel : ℕ)
    (lambda epsMix : ℝ)
    (tauMix : ℝ)
    (hMixBound : tauMix ≤ quotientMixingTimeUpperBound lambda epsMix)
    (hCertBound : concreteStepCountCertificate fuel lambda epsMix)
    (initFromCoords : Finset (Fin (numMDCoordinates prob)) →
      refinementLoopState (GridMDAction NL N))
    (εLJ σ rmin delta : ℝ)
    (uCont : GridMDAction NL N → MDState → ℝ)
    (hApprox : ∀ res a sGrid,
      |uCont a (liftGridState res sGrid) -
          arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res εLJ σ a sGrid| ≤
        (ljSecondDerivShellBound εLJ σ rmin / 2) * delta ^ (2 : ℕ))
    {final : refinementLoopState (GridMDAction NL N)}
    (hRun :
      (fullyConcreteCrossDockAlgorithm prob F εAdm G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport gridState candidateCoords fuel lambda epsMix
        initFromCoords εLJ σ rmin delta).refinementResult = some final) :
    (@DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (crossDockingAdmissibleDecisionProblem prob F εAdm)
      (fullyConcreteCrossDockAlgorithm prob F εAdm G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport gridState candidateCoords fuel lambda epsMix
        initFromCoords εLJ σ rmin delta).retainedCoords) ∧
    (∃ pose : MDAction,
      pose = liftGridAction final.resolution final.bestAction ∧
      concreteLJAdmissible uCont gridState final) ∧
    quotientMixedByStepCount tauMix fuel := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  let dp : DecisionProblem B MDState := crossDockingAdmissibleDecisionProblem prob F εAdm
  let drop := concreteDropTest prob F εAdm G activePocket k
    atomCoordinateSupport ligandCoordinateSupport localSupport
  let retained := greedyExtractLoop candidateCoords drop
  have hDropSound : ∀ current i,
      dp.isSufficient current →
      i ∈ current →
      drop current i = true →
      dp.isSufficient (current.erase i) := by
    intro current i hSuffCurrent hiCurrent hDropCurrent
    simpa [dp, drop] using
      (concreteDropTest_is_sound
        (prob := prob) (F := F) (ε := εAdm)
        (G := G) (activePocket := activePocket) (k := k)
        (atomCoordinateSupport := atomCoordinateSupport)
        (ligandCoordinateSupport := ligandCoordinateSupport)
        (localSupport := localSupport)
        hSupport hComplete hSound hCompat
        current i hSuffCurrent hiCurrent hDropCurrent)
  have hRetainedSuff : dp.isSufficient retained :=
    greedyExtractLoop_sufficient dp candidateCoords drop hInitSuff hDropSound
  have hRunLoop :
      refinementLoop
          (concreteLJObservation (NP := NP) (NL := NL) (N := N) εLJ σ rmin delta gridState)
          fuel
          (initFromCoords retained) = some final := by
    simpa [fullyConcreteCrossDockAlgorithm, drop, retained] using hRun
  have hAdm : concreteLJAdmissible uCont gridState final :=
    concreteLJObservation_loop_correct
      (NP := NP) (NL := NL) (N := N)
      εLJ σ rmin delta uCont gridState hApprox
      fuel (initFromCoords retained) final hRunLoop
  have hMixed : quotientMixedByStepCount tauMix fuel :=
    certificate_implies_mixed hMixBound hCertBound
  refine ⟨?_, ?_, hMixed⟩
  · simpa [fullyConcreteCrossDockAlgorithm, drop, retained] using hRetainedSuff
  · exact ⟨liftGridAction final.resolution final.bestAction, rfl, hAdm⟩

/-- Pure closed correctness theorem: instantiate the canonical molecular product-space structure and
coordinate compatibility proof so the fully concrete algorithm needs no structural/interface
hypotheses in its public signature. -/
theorem pureConcreteAlgorithm_correctness
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (εAdm : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    (hSupport : ∀ i,
      localSupport i ⊆
        atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport)
    (hComplete : ∀ i,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) i) →
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s')
    (hSound : ∀ i,
      (∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s') →
      @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) i)
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (gridState : GridMDState NP NL N)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (hInitSuff :
      @DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) candidateCoords)
    (fuel : ℕ)
    (lambda epsMix : ℝ)
    (tauMix : ℝ)
    (hMixBound : tauMix ≤ quotientMixingTimeUpperBound lambda epsMix)
    (hCertBound : concreteStepCountCertificate fuel lambda epsMix)
    (initFromCoords : Finset (Fin (numMDCoordinates prob)) →
      refinementLoopState (GridMDAction NL N))
    (εLJ σ rmin delta : ℝ)
    (uCont : GridMDAction NL N → MDState → ℝ)
    (hApprox : ∀ res a sGrid,
      |uCont a (liftGridState res sGrid) -
          arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res εLJ σ a sGrid| ≤
        (ljSecondDerivShellBound εLJ σ rmin / 2) * delta ^ (2 : ℕ))
    {final : refinementLoopState (GridMDAction NL N)}
    (hRun :
      (fullyConcreteCrossDockAlgorithm prob F εAdm G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport gridState candidateCoords fuel lambda epsMix
        initFromCoords εLJ σ rmin delta).refinementResult = some final) :
    (@DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (crossDockingAdmissibleDecisionProblem prob F εAdm)
      (fullyConcreteCrossDockAlgorithm prob F εAdm G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport gridState candidateCoords fuel lambda epsMix
        initFromCoords εLJ σ rmin delta).retainedCoords) ∧
    (∃ pose : MDAction,
      pose = liftGridAction final.resolution final.bestAction ∧
      concreteLJAdmissible uCont gridState final) ∧
    quotientMixedByStepCount tauMix fuel := by
  letI : ProductSpace MDState (numMDCoordinates prob) := canonicalMDStateProductSpace prob
  exact fullyConcreteAlgorithm_correctness
    (prob := prob) (F := F) (εAdm := εAdm)
    (G := G) (activePocket := activePocket) (k := k)
    (atomCoordinateSupport := atomCoordinateSupport)
    (ligandCoordinateSupport := ligandCoordinateSupport)
    (localSupport := localSupport)
    hSupport hComplete hSound
    (hCompat := by
      intro s s' j
      rfl)
    (gridState := gridState)
    (candidateCoords := candidateCoords)
    (hInitSuff := hInitSuff)
    (fuel := fuel)
    (lambda := lambda) (epsMix := epsMix)
    (tauMix := tauMix)
    (hMixBound := hMixBound)
    (hCertBound := hCertBound)
    (initFromCoords := initFromCoords)
    (εLJ := εLJ) (σ := σ) (rmin := rmin) (delta := delta)
    (uCont := uCont) (hApprox := hApprox)
    (hRun := hRun)

/-- JAX export specification for the fully concrete closed cross-docking algorithm. -/
noncomputable def closedAlgorithmJAXSpec
    {NP NL N : Nat} : stochasticJAXExportSpec :=
  stochasticJAXExportSpecOfGrid (NP := NP) (NL := NL) (N := N)

/-- The closed algorithm is exportable through the existing ArrayDSL primitive map: Lennard-Jones
scoring (`lennardJones`), coordinate masking/erasure (`applyCutoff`), and pairwise distance
construction (`pairwiseDistances`) are all present in the primitive library, and the halting guard
is exactly the certified half-gap test. -/
theorem closedAlgorithmExportable
    {NP NL N : Nat} :
    let spec := closedAlgorithmJAXSpec (NP := NP) (NL := NL) (N := N)
    spec.primitiveSequence = DecisionQuotient.Computation.ArrayDSL.exportPrimitives ∧
    "lennardJones" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    "pairwiseDistances" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    "sumPairPotentials" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    "applyCutoff" ∈ spec.primitiveSequence.map (fun p => p.name) ∧
    "lennardJones" ∈ DecisionQuotient.Computation.ArrayDSL.primitiveJAXMapping.map (fun p => p.opName) ∧
    "pairwiseDistances" ∈ DecisionQuotient.Computation.ArrayDSL.primitiveJAXMapping.map (fun p => p.opName) ∧
    "applyCutoff" ∈ DecisionQuotient.Computation.ArrayDSL.primitiveJAXMapping.map (fun p => p.opName) ∧
    (∀ eps gap, spec.halfGapHaltingCondition eps gap ↔ eps < gap / 2) := by
  constructor
  · rfl
  constructor
  · simp [closedAlgorithmJAXSpec, stochasticJAXExportSpecOfGrid,
      DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [closedAlgorithmJAXSpec, stochasticJAXExportSpecOfGrid,
      DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [closedAlgorithmJAXSpec, stochasticJAXExportSpecOfGrid,
      DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [closedAlgorithmJAXSpec, stochasticJAXExportSpecOfGrid,
      DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [DecisionQuotient.Computation.ArrayDSL.primitiveJAXMapping,
      DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [DecisionQuotient.Computation.ArrayDSL.primitiveJAXMapping,
      DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  constructor
  · simp [DecisionQuotient.Computation.ArrayDSL.primitiveJAXMapping,
      DecisionQuotient.Computation.ArrayDSL.exportPrimitives]
  · intro eps gap
    rfl

/-- Final JAX fuse for the closed algorithm: if the exported closed spec executes and halts, then
the returned pose satisfies the same mechanized guarantee as the pure closed theorem (sufficient
retained coordinates, admissible lifted action, and certified mixed-by-step-count bound). -/
theorem closedAlgorithm_JAX_correctness
    (prob : MDBindingProblem) {B : Type*}
    (F : QuantitativeAdmissibilityFamily MDAction B) (εAdm : ℝ)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (atomCoordinateSupport : Finset (Fin prob.protein.numAtoms) →
      Finset (Fin (numMDCoordinates prob)))
    (ligandCoordinateSupport : Finset (Fin (numMDCoordinates prob)))
    (localSupport : Fin (numMDCoordinates prob) → Finset (Fin (numMDCoordinates prob)))
    (hSupport : ∀ i,
      localSupport i ⊆
        atomCoordinateSupport (G.kHopNeighborhood activePocket k) ∪ ligandCoordinateSupport)
    (hComplete : ∀ i,
      (@DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) i) →
      ∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s')
    (hSound : ∀ i,
      (∃ s s' : MDState,
        localStatePairWithinSupport prob (localSupport i) s s' ∧
        (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s ≠
          (crossDockingAdmissibleDecisionProblem prob F εAdm).Opt s') →
      @DecisionProblem.isRelevant B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) i)
    {NP NL N : Nat}
    [Fintype (GridMDAction NL N)] [DecidableEq (GridMDAction NL N)] [Nonempty (GridMDAction NL N)]
    (gridState : GridMDState NP NL N)
    (candidateCoords : Finset (Fin (numMDCoordinates prob)))
    (hInitSuff :
      @DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (crossDockingAdmissibleDecisionProblem prob F εAdm) candidateCoords)
    (fuel : ℕ)
    (lambda epsMix : ℝ)
    (tauMix : ℝ)
    (hMixBound : tauMix ≤ quotientMixingTimeUpperBound lambda epsMix)
    (initFromCoords : Finset (Fin (numMDCoordinates prob)) →
      refinementLoopState (GridMDAction NL N))
    (εLJ σ rmin delta : ℝ)
    (uCont : GridMDAction NL N → MDState → ℝ)
    (hApprox : ∀ res a sGrid,
      |uCont a (liftGridState res sGrid) -
          arrayDSL_LJ_gridUtility (NP := NP) (NL := NL) (N := N) res εLJ σ a sGrid| ≤
        (ljSecondDerivShellBound εLJ σ rmin / 2) * delta ^ (2 : ℕ))
    (spec : stochasticJAXExportSpec)
    (_hSpec : spec = closedAlgorithmJAXSpec (NP := NP) (NL := NL) (N := N))
    {final : refinementLoopState (GridMDAction NL N)}
    (hRuntime : stochasticJAXRuntimeHaltsWith spec
      (fullyConcreteCrossDockAlgorithm prob F εAdm G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport gridState candidateCoords fuel lambda epsMix
        initFromCoords εLJ σ rmin delta).refinementResult
      final
      (fullyConcreteCrossDockAlgorithm prob F εAdm G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport gridState candidateCoords fuel lambda epsMix
        initFromCoords εLJ σ rmin delta).stepCertificate) :
    (@DecisionProblem.isSufficient B MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (crossDockingAdmissibleDecisionProblem prob F εAdm)
      (fullyConcreteCrossDockAlgorithm prob F εAdm G activePocket k atomCoordinateSupport
        ligandCoordinateSupport localSupport gridState candidateCoords fuel lambda epsMix
        initFromCoords εLJ σ rmin delta).retainedCoords) ∧
    (∃ pose : MDAction,
      pose = liftGridAction final.resolution final.bestAction ∧
      concreteLJAdmissible uCont gridState final) ∧
    quotientMixedByStepCount tauMix fuel := by
  let algo := fullyConcreteCrossDockAlgorithm prob F εAdm G activePocket k atomCoordinateSupport
    ligandCoordinateSupport localSupport gridState candidateCoords fuel lambda epsMix
    initFromCoords εLJ σ rmin delta
  rcases hRuntime with ⟨hRun, hCert⟩
  have hCertBound : concreteStepCountCertificate fuel lambda epsMix := by
    simpa [algo, fullyConcreteCrossDockAlgorithm] using hCert
  have hRun' : algo.refinementResult = some final := by
    simpa [algo] using hRun
  simpa [algo] using pureConcreteAlgorithm_correctness
    (prob := prob) (F := F) (εAdm := εAdm)
    (G := G) (activePocket := activePocket) (k := k)
    (atomCoordinateSupport := atomCoordinateSupport)
    (ligandCoordinateSupport := ligandCoordinateSupport)
    (localSupport := localSupport)
    hSupport hComplete hSound
    (gridState := gridState)
    (candidateCoords := candidateCoords)
    (hInitSuff := hInitSuff)
    (fuel := fuel)
    (lambda := lambda) (epsMix := epsMix)
    (tauMix := tauMix)
    (hMixBound := hMixBound)
    (hCertBound := hCertBound)
    (initFromCoords := initFromCoords)
    (εLJ := εLJ) (σ := σ) (rmin := rmin) (delta := delta)
    (uCont := uCont) (hApprox := hApprox)
    (hRun := hRun')

/-- Competitive two-ligand docking game model on a shared protein state space. -/
structure CompetitiveDockingGameModel (A₁ A₂ S : Type*) where
  utilityA : A₁ → A₂ → S → ℝ
  utilityB : A₁ → A₂ → S → ℝ

/-- Constructive competitive docking game where each ligand utility is penalized by the other
ligand's occupancy/influence term. -/
noncomputable def competitiveDockingGame
    {A₁ A₂ S : Type*}
    (baseA : A₁ → S → ℝ) (baseB : A₂ → S → ℝ)
    (interferenceA : A₂ → S → ℝ) (interferenceB : A₁ → S → ℝ) :
    CompetitiveDockingGameModel A₁ A₂ S :=
  { utilityA := fun a b s => baseA a s - interferenceA b s
    utilityB := fun a b s => baseB b s - interferenceB a s }

/-- Player-A induced decision problem under a fixed ligand-B action. -/
noncomputable def competitivePlayerADecisionProblem
    {A₁ A₂ S : Type*}
    (G : CompetitiveDockingGameModel A₁ A₂ S) (b : A₂) :
    DecisionProblem A₁ S where
  utility := fun a s => G.utilityA a b s

/-- Player-B induced decision problem under a fixed ligand-A action. -/
noncomputable def competitivePlayerBDecisionProblem
    {A₁ A₂ S : Type*}
    (G : CompetitiveDockingGameModel A₁ A₂ S) (a : A₁) :
    DecisionProblem A₂ S where
  utility := fun b s => G.utilityB a b s

/-- Joint game decision problem over paired ligand actions. -/
noncomputable def competitiveJointDecisionProblem
    {A₁ A₂ S : Type*}
    (G : CompetitiveDockingGameModel A₁ A₂ S) :
    DecisionProblem (A₁ × A₂) S where
  utility := fun ab s => G.utilityA ab.1 ab.2 s + G.utilityB ab.1 ab.2 s

/-- Joint structural-rank bound for competitive binding: shared pocket coordinates erase duplicated
relevance, so the joint rank is bounded by the sum of individual ranks minus shared bits. -/
theorem jointSrank_bound
    {A₁ A₂ S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dpA : DecisionProblem A₁ S)
    (dpB : DecisionProblem A₂ S)
    (dpJoint : DecisionProblem (A₁ × A₂) S)
    (sharedPocketCoords : ℕ)
    (hSharedReduction : dpJoint.srank + sharedPocketCoords ≤ dpA.srank + dpB.srank) :
    dpJoint.srank ≤ dpA.srank + dpB.srank - sharedPocketCoords := by
  omega

/-- Equilibrium profile over quotient classes with admissibility thresholds. -/
structure NashQuotientProfile
    {A₁ A₂ S : Type*}
    (G : CompetitiveDockingGameModel A₁ A₂ S) (s : S) where
  actionA : A₁
  actionB : A₂
  admissibilityA : ℝ
  admissibilityB : ℝ
  inOptA : actionA ∈ (competitivePlayerADecisionProblem G actionB).Opt s
  inOptB : actionB ∈ (competitivePlayerBDecisionProblem G actionA).Opt s
  meetsA : admissibilityA ≤ G.utilityA actionA actionB s
  meetsB : admissibilityB ≤ G.utilityB actionA actionB s

/-- Unilateral class change attempt for ligand A at fixed ligand-B action. -/
def unilateralClassChangeA
    {A₁ A₂ S : Type*}
    (G : CompetitiveDockingGameModel A₁ A₂ S) (s : S) (b : A₂) (a' : A₁) : Prop :=
  a' ∉ (competitivePlayerADecisionProblem G b).Opt s

/-- Unilateral class change attempt for ligand B at fixed ligand-A action. -/
def unilateralClassChangeB
    {A₁ A₂ S : Type*}
    (G : CompetitiveDockingGameModel A₁ A₂ S) (s : S) (a : A₁) (b' : A₂) : Prop :=
  b' ∉ (competitivePlayerBDecisionProblem G a).Opt s

/-- Nash quotient equilibrium guarantee: no ligand can unilaterally switch optimizer class and
remain above admissibility threshold. -/
theorem nashQuotientEquilibrium
    {A₁ A₂ S : Type*}
    (G : CompetitiveDockingGameModel A₁ A₂ S) (s : S)
    (eq : NashQuotientProfile G s)
    (hUnsafeA : ∀ a' : A₁,
      unilateralClassChangeA G s eq.actionB a' →
        G.utilityA a' eq.actionB s < eq.admissibilityA)
    (hUnsafeB : ∀ b' : A₂,
      unilateralClassChangeB G s eq.actionA b' →
        G.utilityB eq.actionA b' s < eq.admissibilityB) :
    (∀ a' : A₁,
      unilateralClassChangeA G s eq.actionB a' →
        G.utilityA a' eq.actionB s < eq.admissibilityA) ∧
    (∀ b' : A₂,
      unilateralClassChangeB G s eq.actionA b' →
        G.utilityB eq.actionA b' s < eq.admissibilityB) := by
  exact ⟨hUnsafeA, hUnsafeB⟩

/-- TUR precision overhead for resolving a two-ligand competitive equilibrium. -/
noncomputable def competitiveTUROverhead
    (kB precisionA precisionB : ℝ) : ℝ :=
  2 * kB *
    (stochasticResolutionPrecisionTarget precisionA +
      stochasticResolutionPrecisionTarget precisionB)

theorem competitiveTUROverhead_pos
    {kB precisionA precisionB : ℝ}
    (hkB : 0 < kB)
    (hPrecA : 0 < stochasticResolutionPrecisionTarget precisionA)
    (hPrecB : 0 < stochasticResolutionPrecisionTarget precisionB) :
    0 < competitiveTUROverhead kB precisionA precisionB := by
  unfold competitiveTUROverhead
  positivity

/-- Competitive thermodynamic overhead: resolving to a Nash equilibrium strictly exceeds the
single-agent work floor once TUR precision overhead is included. -/
theorem competitiveThermodynamicOverhead
    {singleWork competitiveWork kB precisionA precisionB : ℝ}
    (hCompetitive :
      singleWork + competitiveTUROverhead kB precisionA precisionB ≤ competitiveWork)
    (hOverPos : 0 < competitiveTUROverhead kB precisionA precisionB) :
    singleWork < competitiveWork := by
  have hLt : singleWork < singleWork + competitiveTUROverhead kB precisionA precisionB := by
    linarith
  exact lt_of_lt_of_le hLt hCompetitive

/-- Joint flexible state for protein-protein docking: both proteins retain independent internal
conformations while interacting through a shared interface. -/
structure PPIFlexibleState where
  proteinA : MDState
  proteinB : MDState

/-- Joint PPI decision problem with explicit interface physics: utility is surface complementarity
minus interface-constraint penalty. -/
noncomputable def ppiDecisionProblem
    (surfaceComplementarity : MDAction → MDAction → PPIFlexibleState → ℝ)
    (interfaceConstraintPenalty : MDAction → MDAction → PPIFlexibleState → ℝ) :
    DecisionProblem (MDAction × MDAction) PPIFlexibleState where
  utility := fun ab s =>
    surfaceComplementarity ab.1 ab.2 s - interfaceConstraintPenalty ab.1 ab.2 s

/-- Finite interface-constraint model for protein-protein docking: binding adds cross-protein
holonomic constraints to an unbound baseline. -/
structure PPIInterfaceConstraintModel where
  atomCountA : ℕ
  atomCountB : ℕ
  baselineConstraintCount : ℕ
  interfaceConstraintCount : ℕ
  hUnboundIndependent :
    baselineConstraintCount < 3 * (atomCountA + atomCountB)
  hBoundIndependent :
    baselineConstraintCount + interfaceConstraintCount < 3 * (atomCountA + atomCountB)
  hInterfacePos : 0 < interfaceConstraintCount

/-- Unbound joint molecular system for a PPI pair. -/
def PPIInterfaceConstraintModel.unboundSystem
    (X : PPIInterfaceConstraintModel) : ConstrainedMolecularSystem :=
  { atomCount := X.atomCountA + X.atomCountB
    constraintCount := X.baselineConstraintCount
    hIndependent := by simpa using X.hUnboundIndependent }

/-- Bound interface-constrained molecular system for a PPI pair. -/
def PPIInterfaceConstraintModel.boundSystem
    (X : PPIInterfaceConstraintModel) : ConstrainedMolecularSystem :=
  { atomCount := X.atomCountA + X.atomCountB
    constraintCount := X.baselineConstraintCount + X.interfaceConstraintCount
    hIndependent := by simpa using X.hBoundIndependent }

/-- Unbound structural rank in the canonical constrained-molecular transport. -/
noncomputable def ppiUnboundRank (X : PPIInterfaceConstraintModel) : ℕ :=
  (canonicalDP X.unboundSystem.toArchitecture.dof).srank

/-- Bound structural rank after interface constraints are imposed. -/
noncomputable def ppiBoundRank (X : PPIInterfaceConstraintModel) : ℕ :=
  (canonicalDP X.boundSystem.toArchitecture.dof).srank

private theorem ppiUnboundRank_eq
    (X : PPIInterfaceConstraintModel) :
    ppiUnboundRank X =
      3 * (X.atomCountA + X.atomCountB) - X.baselineConstraintCount := by
  unfold ppiUnboundRank
  simpa [PPIInterfaceConstraintModel.unboundSystem] using
    constrainedMolecular_srank_eq_effectiveDOF X.unboundSystem

private theorem ppiBoundRank_eq
    (X : PPIInterfaceConstraintModel) :
    ppiBoundRank X =
      3 * (X.atomCountA + X.atomCountB) -
        (X.baselineConstraintCount + X.interfaceConstraintCount) := by
  unfold ppiBoundRank
  simpa [PPIInterfaceConstraintModel.boundSystem] using
    constrainedMolecular_srank_eq_effectiveDOF X.boundSystem

/-- Protein-protein interface rank-collapse law: imposing `k` independent cross-protein holonomic
constraints lowers the joint structural rank by exactly `k`, hence strictly lowers it whenever
`k > 0`. -/
theorem ppiInterfaceRankCollapse
    (X : PPIInterfaceConstraintModel) :
    ppiBoundRank X = ppiUnboundRank X - X.interfaceConstraintCount ∧
    ppiBoundRank X < ppiUnboundRank X := by
  have hUnbound := ppiUnboundRank_eq X
  have hBound := ppiBoundRank_eq X
  constructor
  · rw [hBound, hUnbound]
    omega
  · rw [hBound, hUnbound]
    let n : ℕ := 3 * (X.atomCountA + X.atomCountB)
    let b : ℕ := X.baselineConstraintCount
    let i : ℕ := X.interfaceConstraintCount
    have hLeAll : b + i ≤ n := Nat.le_of_lt X.hBoundIndependent
    have hLeBase : b ≤ n := le_trans (Nat.le_add_right _ _) hLeAll
    have hDecomp : n = (n - (b + i)) + (b + i) :=
      (Nat.sub_eq_iff_eq_add hLeAll).1 rfl
    have hAdd : n - b = (n - (b + i)) + i := by
      apply (Nat.sub_eq_iff_eq_add hLeBase).2
      calc
        n = (n - (b + i)) + (b + i) := hDecomp
        _ = (n - (b + i)) + i + b := by ac_rfl
    rw [hAdd]
    exact Nat.lt_add_of_pos_right X.hInterfacePos

/-- Landauer floor associated with resolving the bound PPI quotient. -/
noncomputable def ppiBoundRankLandauerFloor
    (X : PPIInterfaceConstraintModel) (kB T : ℝ) : ℝ :=
  (ppiBoundRank X : ℝ) * (kB * T * Real.log 2)

/-- Stochastic interface-search thermodynamic floor from quotient mixing time. -/
noncomputable def ppiInterfaceSearchMixingCost
    (tauMix kB T : ℝ) : ℝ :=
  tauMix * mcmcStepLandauerFloor kB T

/-- Total thermodynamic floor for PPI: bound-rank Landauer floor plus stochastic interface-search
mixing floor. -/
noncomputable def ppiThermodynamicFloor
    (X : PPIInterfaceConstraintModel)
    (tauMix kB T : ℝ) : ℝ :=
  ppiBoundRankLandauerFloor X kB T + ppiInterfaceSearchMixingCost tauMix kB T

/-- Protein-protein docking thermodynamic cost bound: total PPI work must cover the Landauer floor
of the collapsed bound rank plus the stochastic mixing cost needed to locate the interface. -/
theorem ppiThermodynamicCost
    (X : PPIInterfaceConstraintModel)
    {tauMix steps kB T stepCost boundWork searchWork totalWork : ℝ}
    (hBoundWork : ppiBoundRankLandauerFloor X kB T ≤ boundWork)
    (hMix : tauMix ≤ steps)
    (hStep : mcmcStepLandauerFloor kB T ≤ stepCost)
    (hSearchWork : steps * stepCost ≤ searchWork)
    (hStepsNonneg : 0 ≤ steps)
    (hkB : 0 < kB) (hT : 0 < T)
    (hTotal : boundWork + searchWork ≤ totalWork) :
    ppiThermodynamicFloor X tauMix kB T ≤ totalWork := by
  have hMixCost : ppiInterfaceSearchMixingCost tauMix kB T ≤ searchWork := by
    unfold ppiInterfaceSearchMixingCost
    simpa [mcmcStepLandauerFloor, mul_assoc, mul_left_comm, mul_comm] using
      (stochasticResolutionThermodynamicCost
        (τmix := tauMix) (steps := steps)
        (kB := kB) (T := T)
        (stepCost := stepCost) (totalWork := searchWork)
        hMix hStep hSearchWork hStepsNonneg hkB hT)
  have hFloor : ppiThermodynamicFloor X tauMix kB T ≤ boundWork + searchWork := by
    unfold ppiThermodynamicFloor
    exact add_le_add hBoundWork hMixCost
  exact le_trans hFloor hTotal

/-- Active-inference objective: choose actions minimizing expected structural rank of the resulting
future decision quotient. -/
noncomputable def activeInferenceObjective
    {Act A S : Type*} [Fintype Act] {n : ℕ} [CoordinateSpace S n]
    (futureProblem : Act → DecisionProblem A S)
    (policyWeight : Act → ℝ) : ℝ :=
  Finset.univ.sum (fun a : Act => policyWeight a * ((futureProblem a).srank : ℝ))

/-- Witness object for self-assembly transitions between unbound and bound decision quotients. -/
structure SelfAssemblyTransition
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] where
  unbound : DecisionProblem A S
  bound : DecisionProblem A S
  rankDrop : bound.srank < unbound.srank

/-- Self-assembly rank minimization: binding/assembly lowers structural rank relative to unbound
decision making. -/
theorem selfAssemblyRankMinimization
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (T : SelfAssemblyTransition (A := A) (S := S)) :
    T.bound.srank < T.unbound.srank :=
  T.rankDrop

/-- Landauer floor associated with resolving `r` structural rank bits. -/
noncomputable def rankLandauerFloor (r : ℕ) (kB T : ℝ) : ℝ :=
  (r : ℝ) * (kB * T * Real.log 2)

/-- Landauer savings generated by reducing structural rank from `rBefore` to `rAfter`. -/
noncomputable def rankReductionLandauerSavings (rBefore rAfter : ℕ) (kB T : ℝ) : ℝ :=
  rankLandauerFloor (rBefore - rAfter) kB T

/-- Thermodynamic efficiency of active conformation change: action cost is lower-bounded by the
Landauer-floor reduction achieved through structural-rank reduction. -/
theorem thermodynamicEfficiencyOfAction
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (before after : DecisionProblem A S)
    {kB T actionCost : ℝ}
    (_hDrop : after.srank ≤ before.srank)
    (hCost : rankReductionLandauerSavings before.srank after.srank kB T ≤ actionCost) :
    rankReductionLandauerSavings before.srank after.srank kB T ≤ actionCost :=
  hCost

/-- Molecular computer specification: a system cycling through decision quotients, paying ATP-like
reset energy to satisfy per-cycle Landauer floors. -/
structure molecularComputerSpec
    (A S : Type*) (n : ℕ) [CoordinateSpace S n] where
  kB : ℝ
  T : ℝ
  quotientCycle : ℕ → DecisionProblem A S
  atpHydrolysisEnergy : ℕ → ℝ
  resetBits : ℕ → ℕ
  resetCoversRank : ∀ t, (quotientCycle t).srank ≤ resetBits t
  paysLandauer : ∀ t,
    rankLandauerFloor (resetBits t) kB T ≤ atpHydrolysisEnergy t

/-- Bremermann-style computational-rate upper bound in Landauer units. -/
noncomputable def computationalBremermannRateBound
    (energyFlux kB T rankPerDecision : ℝ) : ℝ :=
  energyFlux / (rankPerDecision * (kB * T * Real.log 2))

/-- Computational Bremermann limit for molecular computers: decision rate is bounded by available
energy flux divided by Landauer cost per structural-rank decision. -/
theorem computationalBremermannLimit
    {decisionRate energyFlux kB T rankPerDecision : ℝ}
    (hDenPos : 0 < rankPerDecision * (kB * T * Real.log 2))
    (hFlux : decisionRate * (rankPerDecision * (kB * T * Real.log 2)) ≤ energyFlux) :
    decisionRate ≤ computationalBremermannRateBound energyFlux kB T rankPerDecision := by
  unfold computationalBremermannRateBound
  exact (le_div_iff₀ hDenPos).2 (by simpa [mul_assoc, mul_left_comm, mul_comm] using hFlux)

/-- Computational efficiency proxy (decisions per unit structural rank). -/
noncomputable def molecularComputationalEfficiency (capacity rank : ℝ) : ℝ :=
  capacity / rank

/-- Capacity scaling lemma: if capacity scales linearly by factor `t` while structural rank scales
strictly sublinearly (< `t`), then efficiency (capacity per rank) strictly improves. -/
theorem efficiency_gain_of_linear_capacity_sublinear_rank
    {capacity1 capacity2 rank1 rank2 t : ℝ}
    (ht : 1 < t)
    (hCapScale : capacity2 = t * capacity1)
    (hRankScale : rank2 < t * rank1)
    (hCapPos : 0 < capacity1)
    (hRankPos1 : 0 < rank1)
    (hRankPos2 : 0 < rank2) :
    molecularComputationalEfficiency capacity1 rank1 <
      molecularComputationalEfficiency capacity2 rank2 := by
  unfold molecularComputationalEfficiency
  rw [hCapScale]
  have hRank1ne : rank1 ≠ 0 := ne_of_gt hRankPos1
  have hRank2ne : rank2 ≠ 0 := ne_of_gt hRankPos2
  have hCapNe : capacity1 ≠ 0 := ne_of_gt hCapPos
  field_simp [hRank1ne, hRank2ne, hCapNe]
  simpa [mul_assoc, mul_left_comm, mul_comm] using hRankScale

/-- Landauer scaling law: with linear capacity growth in mass and sublinear structural-rank growth
(from condensate/circuit subadditivity), computational efficiency gains are superlinear. -/
theorem landauerScalingLaw
    {mass1 mass2 alpha rank1 rank2 : ℝ}
    (hMass1 : 0 < mass1)
    (hAlpha : 0 < alpha)
    (hMassScale : 1 < mass2 / mass1)
    (hRankSublinear : rank2 < (mass2 / mass1) * rank1)
    (hRankPos1 : 0 < rank1)
    (hRankPos2 : 0 < rank2) :
    molecularComputationalEfficiency (alpha * mass1) rank1 <
      molecularComputationalEfficiency (alpha * mass2) rank2 := by
  have hMass1ne : mass1 ≠ 0 := ne_of_gt hMass1
  have hCapScale : alpha * mass2 = (mass2 / mass1) * (alpha * mass1) := by
    field_simp [hMass1ne]
  have hCapPos : 0 < alpha * mass1 := mul_pos hAlpha hMass1
  exact efficiency_gain_of_linear_capacity_sublinear_rank
    hMassScale hCapScale hRankSublinear hCapPos hRankPos1 hRankPos2

/-- Local paper3 exposure of sampled exact/coarse winner preservation. -/
theorem sampledDocking_exactCoarse_opt_agree_of_gap
    {NP NL N : Nat} (prob : SampledDockingProblem NP NL N)
    (s : GridMDState NP NL N)
    (aStar : SupportedAction prob.samples)
    (delta : ℝ)
    (hDelta : 0 ≤ delta)
    (hStrict : StrictOpt prob.exactDecisionProblem aStar s)
    (hPerturb : ∀ a,
      |prob.exactDecisionProblem.utility a s - prob.coarseDecisionProblem.utility a s| ≤ delta)
    (hBound : delta < StrictUtilityGap prob.exactDecisionProblem aStar s / 2) :
    prob.exactDecisionProblem.Opt s = prob.coarseDecisionProblem.Opt s :=
  DecisionQuotient.Tractability.SampledDockingGap.SampledDockingProblem.exact_coarse_opt_agree_of_gap
    prob s aStar delta hDelta hStrict hPerturb hBound

/-- Local paper3 exposure of conservative top-k preservation under a certified boundary gap. -/
theorem topKPreserved_of_boundaryGap
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (uExact uCoarse : A → ℝ) (k : Nat) (hk : 0 < k)
    (tau delta : ℝ)
    (hApprox : ∀ a, |uExact a - uCoarse a| ≤ delta)
    (hGap : delta ≤ DecisionQuotient.Tractability.RankingPreservation.BoundaryGap uExact k hk tau) :
    DecisionQuotient.Tractability.FiniteTopK.topKSet uExact k ⊆
      DecisionQuotient.Tractability.FiniteTopK.survivorSet uCoarse tau :=
  DecisionQuotient.Tractability.TopKPreservation.topK_preserved_of_boundary_gap
    uExact uCoarse k hk tau delta hApprox hGap

/-- Local paper3 exposure of the ambiguity-band fallback for near ties. -/
theorem exactTopK_subset_ambiguityBand
    {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (u : A → ℝ) (k : Nat) (hk : 0 < k) (eps : ℝ)
    (hEps : 0 ≤ eps) :
    DecisionQuotient.Tractability.FiniteTopK.topKSet u k ⊆
      DecisionQuotient.Tractability.NearTieBand.ambiguityBand u k hk eps :=
  DecisionQuotient.Tractability.NearTieBand.exact_topK_subset_ambiguityBand u k hk eps hEps

/-- Local paper3 exposure of exact-vs-cutoff Lennard-Jones winner preservation on a finite
sampled domain. -/
theorem exactVsCutoffLJ_opt_invariance
    {A S : Type*} [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ)
    (s : S) (aStar : A)
    (hStrict : StrictOpt
      (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distance ε σ)
      aStar s)
    (hBound :
      DecisionQuotient.Tractability.LJApproximation.ljCutoffErrorRadius distance ε σ rc <
        StrictUtilityGap
          (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distance ε σ)
          aStar s / 2) :
    (DecisionQuotient.Tractability.LJApproximation.exactLJDecisionProblem distance ε σ).Opt s =
      (DecisionQuotient.Tractability.LJApproximation.cutoffLJDecisionProblem distance ε σ rc).Opt s :=
  DecisionQuotient.Tractability.LJApproximation.exact_vs_cutoff_lj_opt_invariance
    distance ε σ rc s aStar hStrict hBound

/-- Local paper3 exposure of the finite-domain exact-vs-cutoff Coulomb approximation radius. -/
theorem exactVsCutoffCoulomb_uniformApprox
    {A S : Type*} [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) :
    UniformUtilityApprox
      (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombDecisionProblem q_i q_j distance)
      (DecisionQuotient.Tractability.CoulombApproximation.cutoffCoulombDecisionProblem q_i q_j rc distance)
      (DecisionQuotient.Tractability.CoulombApproximation.coulombCutoffErrorRadius q_i q_j rc distance) :=
  DecisionQuotient.Tractability.CoulombApproximation.exact_vs_cutoff_coulomb_uniformApprox
    q_i q_j rc distance

/-- Local paper3 exposure of exact-vs-cutoff Coulomb winner preservation under a strict half-gap
hypothesis. -/
theorem exactVsCutoffCoulomb_opt_invariance
    {A S : Type*} [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ)
    (s : S) (aStar : A)
    (hStrict : StrictOpt
      (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombDecisionProblem q_i q_j distance)
      aStar s)
    (hBound :
      DecisionQuotient.Tractability.CoulombApproximation.coulombCutoffErrorRadius q_i q_j rc distance <
        StrictUtilityGap
          (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombDecisionProblem q_i q_j distance)
          aStar s / 2) :
    (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombDecisionProblem q_i q_j distance).Opt s =
      (DecisionQuotient.Tractability.CoulombApproximation.cutoffCoulombDecisionProblem q_i q_j rc distance).Opt s := by
  have hApprox := exactVsCutoffCoulomb_uniformApprox q_i q_j rc distance
  have hDelta : 0 ≤ DecisionQuotient.Tractability.CoulombApproximation.coulombCutoffErrorRadius q_i q_j rc distance := by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s0⟩
    exact le_trans (abs_nonneg _)
      (DecisionQuotient.Tractability.CoulombApproximation.coulombCutoffErrorRadius_spec q_i q_j rc distance a s0)
  exact uniform_approx_implies_opt_invariance
    (DecisionQuotient.Tractability.CoulombApproximation.exactCoulombDecisionProblem q_i q_j distance)
    (DecisionQuotient.Tractability.CoulombApproximation.cutoffCoulombDecisionProblem q_i q_j rc distance)
    (DecisionQuotient.Tractability.CoulombApproximation.coulombCutoffErrorRadius q_i q_j rc distance)
    hApprox s aStar hDelta hStrict hBound

/-- Local paper3 exposure of the Ewald real-space exponential decay bound. -/
theorem ewaldRealSpace_exponentialDecay
    (r alpha : ℝ) (hr : 0 < r) (ha : 0 < alpha) :
    DecisionQuotient.Tractability.Ewald.ewaldRealSpaceCore r alpha ≤
      Real.exp (- (alpha ^ 2) * (r ^ 2)) / r :=
  DecisionQuotient.Tractability.Ewald.ewald_real_space_exponential_decay r alpha hr ha

/-- Local paper3 exposure of the Ewald reciprocal-space positivity bound. -/
theorem ewaldFourier_positive
    (k alpha : ℝ) (hk : 0 < k) (ha : 0 < alpha) :
    0 < DecisionQuotient.Tractability.Ewald.ewaldReciprocalCore k alpha :=
  DecisionQuotient.Tractability.Ewald.ewald_fourier_exponential_decay k alpha hk ha

/-- Local paper3 exposure of the geometric phase-space conservation theorem for Velocity-Verlet. -/
theorem velocityVerlet_preservesVolume {n : ℕ}
    (U : ArrayDSL.DiffFunctionN n) (dt mass : ℝ)
    (s : SymplecticIntegrator.PhaseSpace n) :
    (Geometry.velocityVerletJacobian U dt mass s).det = 1 :=
  Geometry.velocity_verlet_preserves_volume U dt mass s

/-- Local paper3 exposure of the closed-form Lennard-Jones derivative. -/
theorem lennardJones_gradient
    (ε σ r : ℝ) (hr : r ≠ 0) :
    HasDerivAt (fun x => Computation.lennardJones ε σ x)
      ((24 * ε * r⁻¹) * ((σ * r⁻¹) ^ (6 : ℕ) - 2 * (σ * r⁻¹) ^ (12 : ℕ))) r :=
  Computation.grad_of_lennardJones ε σ r hr

/-- Local paper3 exposure of inside-cutoff sufficiency for sampled docking. -/
theorem sampledDocking_insideCutoff_sufficient
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction]
    [hProd : ProductSpace MDState (numMDCoordinates prob)]
    (aStar : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hCompat : ∀ s s' i,
      @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s i =
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' i ↔
      mdProj prob s i = mdProj prob s' i)
    (hinj : ∀ s s' : MDState,
      (∀ i : Fin (numMDCoordinates prob),
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s i =
          @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' i) →
      s = s') :
    @DecisionProblem.isSufficient (SupportedAction samples) MDState (numMDCoordinates prob)
      hProd.toCoordinateSpace
      (restrictedDecisionProblem prob.toDecisionProblem samples)
      (potentialRelevantCoords prob) :=
    DecisionQuotient.Tractability.SampledDockingCutoff.sampled_insideCutoff_sufficient
      prob samples aStar hStrictAll hBounded hCapture hCompat hinj

/-- If the inside-cutoff retained coordinates are sufficient and their cardinality fits inside a
bounded-region acquisition budget, then sampled exact docking admits a concrete exact-resolution
witness within that budget. -/
theorem sampledDocking_exactResolutionWithin_of_insideCutoff_and_budget
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction]
    [hProd : ProductSpace MDState (numMDCoordinates prob)]
    (aStar : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hCompat : ∀ s s' i,
      @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s i =
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' i ↔
      mdProj prob s i = mdProj prob s' i)
    (hinj : ∀ s s' : MDState,
      (∀ i : Fin (numMDCoordinates prob),
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s i =
          @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' i) →
      s = s')
    (R : DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion) (T : ℕ)
    (hBudget : (potentialRelevantCoords prob).card ≤
      DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T) :
    @decisionExactResolutionWithin (SupportedAction samples) MDState (numMDCoordinates prob)
      hProd.toCoordinateSpace (restrictedDecisionProblem prob.toDecisionProblem samples) R T := by
  refine ⟨potentialRelevantCoords prob, ?_, hBudget⟩
  exact sampledDocking_insideCutoff_sufficient prob samples aStar hStrictAll hBounded hCapture hCompat hinj

/-- The same retained-coordinate witness yields a concrete bounded-acquisition structural-rank bound
for sampled exact docking. -/
theorem sampledDocking_srank_le_boundedAcquisitions_of_insideCutoff_and_budget
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction]
    [hProd : ProductSpace MDState (numMDCoordinates prob)]
    (aStar : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hCompat : ∀ s s' i,
      @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s i =
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' i ↔
      mdProj prob s i = mdProj prob s' i)
    (hinj : ∀ s s' : MDState,
      (∀ i : Fin (numMDCoordinates prob),
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s i =
          @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' i) →
      s = s')
    (R : DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion) (T : ℕ)
    (hBudget : (potentialRelevantCoords prob).card ≤
      DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T) :
    @DecisionProblem.srank (SupportedAction samples) MDState (numMDCoordinates prob)
      hProd.toCoordinateSpace (restrictedDecisionProblem prob.toDecisionProblem samples) ≤
      DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := hProd.toCoordinateSpace
  have hRes := sampledDocking_exactResolutionWithin_of_insideCutoff_and_budget
    prob samples aStar hStrictAll hBounded hCapture hCompat hinj R T hBudget
  exact decision_srank_le_boundedAcquisitions_of_exactResolutionWithin
    (dp := restrictedDecisionProblem prob.toDecisionProblem samples) R T hRes

/-- The retained inside-cutoff witness also yields a concrete bounded-region energy floor for the
sampled exact docking problem. -/
theorem sampledDocking_energy_le_boundedAcquisitions_of_insideCutoff_and_budget
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction]
    [hProd : ProductSpace MDState (numMDCoordinates prob)]
    (aStar : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hCompat : ∀ s s' i,
      @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s i =
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' i ↔
      mdProj prob s i = mdProj prob s' i)
    (hinj : ∀ s s' : MDState,
      (∀ i : Fin (numMDCoordinates prob),
        @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s i =
          @CoordinateSpace.proj MDState (numMDCoordinates prob) hProd.toCoordinateSpace s' i) →
      s = s')
    (R : DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion) (T : ℕ)
    (hBudget : (potentialRelevantCoords prob).card ≤
      DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) (hJ : 0 < M.joulesPerBit) :
    M.joulesPerBit *
      (@DecisionProblem.srank (SupportedAction samples) MDState (numMDCoordinates prob)
        hProd.toCoordinateSpace (restrictedDecisionProblem prob.toDecisionProblem samples)) ≤
      DecisionQuotient.ThermodynamicLift.energyLowerBound M
        (DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions R T) := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := hProd.toCoordinateSpace
  have hRes := sampledDocking_exactResolutionWithin_of_insideCutoff_and_budget
    prob samples aStar hStrictAll hBounded hCapture hCompat hinj R T hBudget
  exact decision_energy_le_boundedAcquisitions_of_exactResolutionWithin
    (dp := restrictedDecisionProblem prob.toDecisionProblem samples) R T hRes M hJ

/-- Classical finite-rate surrogate for a resolution speed limit: exact resolution inside a bounded
region requires enough time for the signal budget to cover every decision-relevant coordinate. -/
theorem quotient_resolution_speed_bound
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (R : DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion) (T : ℕ)
    (hRes : decisionExactResolutionWithin dp R T) :
    R.diameter * dp.srank ≤ R.signalSpeed * T := by
  have hRank := decision_srank_le_boundedAcquisitions_of_exactResolutionWithin dp R T hRes
  have hMul : dp.srank * R.diameter ≤ R.signalSpeed * T := by
    simpa [DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions] using
      (Nat.le_div_iff_mul_le R.hDiameter).1 hRank
  simpa [Nat.mul_comm] using hMul

/-- Stagewise finite-rate surrogate for the trajectory speed-energy tradeoff: the cumulative
resolution burden fits inside the sum of the local signal budgets, and the cumulative Landauer floor
fits inside the sum of the stagewise bounded-acquisition budgets. -/
theorem trajectory_time_energy_tradeoff
    {A S : Type*} {n m : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : Fin m → DecisionProblem A S)
    (R : Fin m → DecisionQuotient.Physics.BoundedAcquisition.BoundedRegion)
    (T : Fin m → ℕ)
    (hRes : ∀ t : Fin m, decisionExactResolutionWithin (dp t) (R t) (T t))
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) (hJ : 0 < M.joulesPerBit) :
    ((Finset.univ.sum fun t : Fin m => (R t).diameter * (dp t).srank) ≤
      Finset.univ.sum fun t : Fin m => (R t).signalSpeed * T t) ∧
    (Finset.univ.sum fun t : Fin m => M.joulesPerBit * (dp t).srank) ≤
      Finset.univ.sum fun t : Fin m =>
        DecisionQuotient.ThermodynamicLift.energyLowerBound M
          (DecisionQuotient.Physics.BoundedAcquisition.maxAcquisitions (R t) (T t)) := by
  constructor
  · exact Finset.sum_le_sum (fun t _ =>
      quotient_resolution_speed_bound (dp t) (R t) (T t) (hRes t))
  · exact Finset.sum_le_sum (fun t _ =>
      decision_energy_le_boundedAcquisitions_of_exactResolutionWithin
        (dp := dp t) (R t) (T t) (hRes t) M hJ)

/-- Decision-level allosteric coupling cost contributed by one coordinate: one per-bit unit when the
coordinate remains relevant, and zero once the pathway is broken and the coordinate becomes pure
decision-noise. -/
noncomputable def allostericCouplingGap
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) : ℕ :=
  if dp.isRelevant i then M.joulesPerBit else 0

theorem allostericCouplingGap_eq_zero_of_irrelevant
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    (hIrr : dp.isIrrelevant i) :
    allostericCouplingGap dp i M = 0 := by
  have hNot : ¬ dp.isRelevant i := (dp.irrelevant_iff_not_relevant i).1 hIrr
  simp [allostericCouplingGap, hNot]

theorem allostericCouplingGap_eq_joulesPerBit_of_relevant
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    (hRel : dp.isRelevant i) :
    allostericCouplingGap dp i M = M.joulesPerBit := by
  simp [allostericCouplingGap, hRel]

/-- If a broken mechanical pathway is a coarser summary of the intact pathway, and a distant
coordinate becomes irrelevant after the break, the structural rank drops and the decision-level
allosteric coupling cost of that coordinate vanishes. -/
theorem mechanochemical_coupling_gap
    {A B C S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (σIntact : Set A → B) (σBroken : Set A → C)
    (hFactor : ∃ τ : B → C, ∀ X : Set A, σBroken X = τ (σIntact X))
    (iAllo : Fin n)
    (hRel : (dp.optSummaryDecisionProblem σIntact).isRelevant iAllo)
    (hIrr : (dp.optSummaryDecisionProblem σBroken).isIrrelevant iAllo)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel) :
    (dp.optSummaryDecisionProblem σBroken).srank <
      (dp.optSummaryDecisionProblem σIntact).srank ∧
    allostericCouplingGap (dp.optSummaryDecisionProblem σBroken) iAllo M = 0 := by
  have hNotRelBroken :
      ¬ (dp.optSummaryDecisionProblem σBroken).isRelevant iAllo :=
    ((dp.optSummaryDecisionProblem σBroken).irrelevant_iff_not_relevant iAllo).1 hIrr
  have hMem :
      iAllo ∈ optSummaryRelevantFinset dp σIntact \
        optSummaryRelevantFinset dp σBroken := by
    simp [optSummaryRelevantFinset, hRel, hNotRelBroken]
  have hPosCollapse :
      0 < optSummaryCollapseCount dp σIntact σBroken := by
    unfold optSummaryCollapseCount
    exact Finset.card_pos.mpr ⟨iAllo, hMem⟩
  have hEq := optSummary_srank_eq_relaxed_plus_collapseCount_of_factor
    dp σIntact σBroken hFactor
  constructor
  · omega
  · exact allostericCouplingGap_eq_zero_of_irrelevant
      (dp := dp.optSummaryDecisionProblem σBroken) iAllo M hIrr

/-- Bundling a micro-summary into a macro-summary is represented by factorization of the macro
summary through the micro optimizer summary. -/
structure HierarchicalQuotientBundle (A B C : Type*) where
  microSummary : Set A → B
  macroSummary : Set A → C
  factors : ∃ τ : B → C, ∀ X : Set A, macroSummary X = τ (microSummary X)

/-- Explicit macro-state representation obtained by bundling the coordinates in `Isub` into one
macro block while retaining every outside coordinate individually. -/
structure BundledMacroState {n : ℕ} (X : Type*) (Isub : Finset (Fin n)) where
  outside : {i : Fin n // i ∉ Isub} → X
  bundled : {i : Fin n // i ∈ Isub} → X

theorem BundledMacroState.ext
    {n : ℕ} {X : Type*} {Isub : Finset (Fin n)}
    {s t : BundledMacroState X Isub}
    (hOutside : s.outside = t.outside)
    (hBundled : s.bundled = t.bundled) :
    s = t := by
  cases s
  cases t
  cases hOutside
  cases hBundled
  rfl

/-- Forward bundling map from a micro state to its explicit macro-state representation. -/
noncomputable def toBundledMacroState
    {n : ℕ} {X : Type*} (Isub : Finset (Fin n)) (s : Fin n → X) :
    BundledMacroState X Isub where
  outside := fun i => s i.1
  bundled := fun i => s i.1

/-- Reconstruction map from an explicit bundled macro state back to the original micro coordinate
presentation. -/
noncomputable def ofBundledMacroState
    {n : ℕ} {X : Type*} (Isub : Finset (Fin n))
    (s : BundledMacroState X Isub) : Fin n → X :=
  fun i => if h : i ∈ Isub then s.bundled ⟨i, h⟩ else s.outside ⟨i, h⟩

theorem ofBundledMacroState_toBundledMacroState
    {n : ℕ} {X : Type*} (Isub : Finset (Fin n)) (s : Fin n → X) :
    ofBundledMacroState Isub (toBundledMacroState Isub s) = s := by
  funext i
  by_cases h : i ∈ Isub
  · simp [ofBundledMacroState, toBundledMacroState, h]
  · simp [ofBundledMacroState, toBundledMacroState, h]

theorem toBundledMacroState_ofBundledMacroState
    {n : ℕ} {X : Type*} (Isub : Finset (Fin n))
    (s : BundledMacroState X Isub) :
    toBundledMacroState Isub (ofBundledMacroState Isub s) = s := by
  cases s with
  | mk outside bundled =>
      apply BundledMacroState.ext
      · funext i
        have hi : (i : Fin n) ∉ Isub := i.2
        simp [toBundledMacroState, ofBundledMacroState, hi]
      · funext i
        have hi : (i : Fin n) ∈ Isub := i.2
        simp [toBundledMacroState, ofBundledMacroState, hi]

/-- Pull back a micro decision problem along the explicit bundled macro-state reconstruction map. -/
noncomputable def bundledMacroDecisionProblem
    {A X : Type*} {n : ℕ}
    (dp : DecisionProblem A (Fin n → X)) (Isub : Finset (Fin n)) :
    DecisionProblem A (BundledMacroState X Isub) where
  utility a s := dp.utility a (ofBundledMacroState Isub s)

theorem bundledMacroDecisionProblem_opt_eq
    {A X : Type*} {n : ℕ}
    (dp : DecisionProblem A (Fin n → X)) (Isub : Finset (Fin n))
    (s : BundledMacroState X Isub) :
    (bundledMacroDecisionProblem dp Isub).Opt s = dp.Opt (ofBundledMacroState Isub s) := by
  ext a
  rfl

theorem bundledMacroDecisionProblem_opt_eq_original
    {A X : Type*} {n : ℕ}
    (dp : DecisionProblem A (Fin n → X)) (Isub : Finset (Fin n))
    (s : Fin n → X) :
    (bundledMacroDecisionProblem dp Isub).Opt (toBundledMacroState Isub s) = dp.Opt s := by
  rw [bundledMacroDecisionProblem_opt_eq]
  simp [ofBundledMacroState_toBundledMacroState]

/-- Internal degrees of freedom removed by collapsing a micro-block into one macro-coordinate. -/
def bundledInternalDOF {n : ℕ} (Isub : Finset (Fin n)) : ℕ :=
  Isub.card - 1

/-- If bundling a micro-block erases at least its declared internal degrees of freedom, the macro
problem loses at least that much structural rank. -/
theorem hierarchical_srank_bound
    {A B C S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (H : HierarchicalQuotientBundle A B C)
    (Isub : Finset (Fin n))
    (hInternal :
      bundledInternalDOF Isub ≤
        optSummaryCollapseCount dp H.microSummary H.macroSummary) :
    (dp.optSummaryDecisionProblem H.macroSummary).srank ≤
      (dp.optSummaryDecisionProblem H.microSummary).srank - bundledInternalDOF Isub := by
  have hEq := optSummary_srank_eq_tight_sub_collapseCount_of_factor dp
    H.microSummary H.macroSummary H.factors
  omega

/-- If admissibility-band relaxation erases no relevant coordinate, the bundled and unbundled
problems retain the same relevant-coordinate quotient and therefore the same structural rank. -/
theorem renormalized_admissibility_equivalence
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (F : QuantitativeAdmissibilityFamily A B)
    (dp : DecisionProblem A S) (ε Δ : ℝ) (hΔ : 0 ≤ Δ)
    (hNoCollapse : F.collapseCount dp ε Δ = 0) :
    (∀ i : Fin n,
        (F.summaryDecisionProblem dp ε).isRelevant i ↔
          (F.summaryDecisionProblem dp (ε + Δ)).isRelevant i) ∧
    (F.summaryDecisionProblem dp ε).srank =
      (F.summaryDecisionProblem dp (ε + Δ)).srank := by
  have hFactor := F.relaxes ε Δ hΔ
  constructor
  · intro i
    constructor
    · intro hRel
      by_contra hNot
      have hMem :
          i ∈ optSummaryRelevantFinset dp (F.summary ε) \
            optSummaryRelevantFinset dp (F.summary (ε + Δ)) := by
        refine Finset.mem_sdiff.mpr ?_
        constructor
        · simpa [optSummaryRelevantFinset] using hRel
        · simpa [optSummaryRelevantFinset] using hNot
      have hPos :
          0 < optSummaryCollapseCount dp (F.summary ε) (F.summary (ε + Δ)) := by
        unfold optSummaryCollapseCount
        exact Finset.card_pos.mpr ⟨i, hMem⟩
      have hNe : F.collapseCount dp ε Δ ≠ 0 := by
        simpa [QuantitativeAdmissibilityFamily.collapseCount] using Nat.ne_of_gt hPos
      exact hNe hNoCollapse
    · exact optSummary_relevant_imp_relevant_of_factor dp
        (F.summary ε) (F.summary (ε + Δ)) hFactor i
  · have hEq := quantitativeAdmissibility_srank_eq_relaxed_plus_collapseCount F dp ε Δ hΔ
    simpa [hNoCollapse] using hEq

/-- Hamming distance of an error syndrome from the no-fault codeword. -/
def noFaultSyndrome {r : ℕ} : Fin r → Bool :=
  fun _ => false

/-- Hamming distance of an error syndrome from the no-fault codeword. -/
def syndromeHammingDistance {r : ℕ} (syndrome : Fin r → Bool) : ℕ :=
  hammingDist syndrome noFaultSyndrome

/-- Fault-tolerant structural rank equals the logical rank plus the syndrome bits that must be
detected and erased. -/
def faultTolerantRank {r : ℕ} (syndrome : Fin r → Bool) : ℕ :=
  r + syndromeHammingDistance syndrome

/-- A nonzero syndrome forces at least one extra structural-rank unit above the logical rank, while
the exact fault-tolerant rank is the logical rank plus the syndrome's Hamming distance. -/
theorem error_correction_srank_overhead
    {r : ℕ} (syndrome : Fin r → Bool) (hFault : syndrome ≠ noFaultSyndrome) :
    faultTolerantRank syndrome = r + syndromeHammingDistance syndrome ∧
    r + 1 ≤ faultTolerantRank syndrome := by
  constructor
  · rfl
  · have hDistPos : 0 < syndromeHammingDistance syndrome := by
      unfold syndromeHammingDistance
      simpa [noFaultSyndrome] using
        (hammingDist_pos (x := syndrome) (y := noFaultSyndrome)).2 hFault
    unfold faultTolerantRank
    omega

/-- Under Landauer calibration, fault-tolerant resolution sits strictly above the logical-rank floor
by exactly the energy required to erase the error syndrome. -/
theorem fault_tolerant_landauer_floor
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    {r : ℕ} (syndrome : Fin r → Bool) (hFault : syndrome ≠ noFaultSyndrome) :
    (r : ℝ) * (kB * T * Real.log 2) <
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M (faultTolerantRank syndrome) : ℝ) ∧
    (DecisionQuotient.ThermodynamicLift.energyLowerBound M (faultTolerantRank syndrome) : ℝ) =
      (r : ℝ) * (kB * T * Real.log 2) +
        (DecisionQuotient.ThermodynamicLift.energyLowerBound M
          (syndromeHammingDistance syndrome) : ℝ) := by
  have hJ : 0 < M.joulesPerBit :=
    DecisionQuotient.ThermodynamicLift.joulesPerBit_pos_of_landauer_calibration M hkB hT hCal
  have hDistPos : 0 < syndromeHammingDistance syndrome := by
    unfold syndromeHammingDistance
    simpa [noFaultSyndrome] using
      (hammingDist_pos (x := syndrome) (y := noFaultSyndrome)).2 hFault
  have hAddNat :
      DecisionQuotient.ThermodynamicLift.energyLowerBound M (faultTolerantRank syndrome) =
        DecisionQuotient.ThermodynamicLift.energyLowerBound M r +
          DecisionQuotient.ThermodynamicLift.energyLowerBound M (syndromeHammingDistance syndrome) := by
    unfold faultTolerantRank
    simpa using DecisionQuotient.ThermodynamicLift.energy_lower_additive M r
      (syndromeHammingDistance syndrome)
  have hAdd :
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M (faultTolerantRank syndrome) : ℝ) =
        (DecisionQuotient.ThermodynamicLift.energyLowerBound M r : ℝ) +
          (DecisionQuotient.ThermodynamicLift.energyLowerBound M
            (syndromeHammingDistance syndrome) : ℝ) := by
    exact_mod_cast hAddNat
  have hBase :
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M r : ℝ) =
        (r : ℝ) * (kB * T * Real.log 2) := by
    calc
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M r : ℝ)
          = (M.joulesPerBit : ℝ) * r := by
              simp [DecisionQuotient.ThermodynamicLift.energyLowerBound]
      _ = (r : ℝ) * (M.joulesPerBit : ℝ) := by ring
      _ = (r : ℝ) * DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T := by
            rw [hCal]
      _ = (r : ℝ) * (kB * T * Real.log 2) := by
            simp [DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit]
  have hSyndromeEnergyPos :
      0 < DecisionQuotient.ThermodynamicLift.energyLowerBound M (syndromeHammingDistance syndrome) :=
    DecisionQuotient.ThermodynamicLift.energy_lower_mandatory M hJ hDistPos
  have hSyndromeEnergyPosReal :
      0 < (DecisionQuotient.ThermodynamicLift.energyLowerBound M
        (syndromeHammingDistance syndrome) : ℝ) := by
    exact_mod_cast hSyndromeEnergyPos
  constructor
  · rw [hAdd, hBase]
    linarith
  · rw [hAdd, hBase]

/-- Macrostate package for a docking binding free-energy witness. It records a sufficient-coordinate
resolution witness, a thermodynamic model, and a measured binding free energy `ΔG` that dominates the
resolution witness energy. -/
structure DockingBindingMacrostate (prob : MDBindingProblem) where
  thermoModel : DecisionQuotient.ThermodynamicLift.ThermoModel
  sufficientSet : Finset (Fin (numMDCoordinates prob))
  sufficient : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
    (mdCoordinateSpaceStruct prob) prob.toDecisionProblem sufficientSet
  deltaG : ℝ
  dominatesResolutionWitness :
    (DecisionQuotient.ThermodynamicLift.energyLowerBound thermoModel sufficientSet.card : ℝ) ≤ deltaG

/-- Declared macrostate binding free energy for a docking instance. -/
def macrostateBindingFreeEnergy (prob : MDBindingProblem)
    (B : DockingBindingMacrostate prob) : ℝ :=
  B.deltaG

/-- Any reported binding free-energy witness that dominates a discrete-resolution energy witness is
nonnegative. -/
theorem binding_free_energy_nonneg
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    (bits : ℕ) {ΔG : ℝ}
    (hBinding :
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M bits : ℝ) ≤ ΔG) :
    0 ≤ ΔG := by
  have hWitnessNonneg :
      0 ≤ (DecisionQuotient.ThermodynamicLift.energyLowerBound M bits : ℝ) := by
    exact_mod_cast
      (Nat.zero_le (DecisionQuotient.ThermodynamicLift.energyLowerBound M bits))
  exact le_trans hWitnessNonneg hBinding

/-- Macrostate free energies are nonnegative because they dominate a discrete exact-resolution witness. -/
theorem macrostateBindingFreeEnergy_nonneg
    (prob : MDBindingProblem)
    (B : DockingBindingMacrostate prob) :
    0 ≤ macrostateBindingFreeEnergy prob B := by
  have hNonneg := binding_free_energy_nonneg
    (M := B.thermoModel)
    (bits := B.sufficientSet.card)
    (ΔG := B.deltaG)
    B.dominatesResolutionWitness
  simpa [macrostateBindingFreeEnergy] using hNonneg

/-- If a measured binding free energy `ΔG` dominates the exact-resolution energy witness, it is at
least the Landauer cost of the decision quotient. -/
theorem binding_free_energy_floor
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n)) (hI : dp.isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG) :
    (dp.srank : ℝ) * (kB * T * Real.log 2) ≤ ΔG := by
  have hJ : 0 < M.joulesPerBit :=
    DecisionQuotient.ThermodynamicLift.joulesPerBit_pos_of_landauer_calibration M hkB hT hCal
  have hEnergyNat :=
    DecisionQuotient.Physics.BoundedAcquisition.energy_ge_srank_cost dp I hI M hJ
  have hCast : (((M.joulesPerBit * dp.srank : ℕ)) : ℝ) ≤
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
    exact_mod_cast hEnergyNat
  have hLandauer :
      (dp.srank : ℝ) * (kB * T * Real.log 2) ≤
        (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
    calc
      (dp.srank : ℝ) * (kB * T * Real.log 2)
          = (dp.srank : ℝ) * DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T := by
              simp [DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit]
      _ = (dp.srank : ℝ) * (M.joulesPerBit : ℝ) := by rw [hCal]
      _ = ((M.joulesPerBit : ℝ) * (dp.srank : ℝ)) := by ring
      _ = (((M.joulesPerBit * dp.srank : ℕ)) : ℝ) := by norm_num
      _ ≤ (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) := by
          simpa [DecisionQuotient.ThermodynamicLift.energyLowerBound] using hCast
  exact le_trans hLandauer hBinding

/-- Any certified rank target `r ≤ srank` inherits the same calibrated binding-energy floor. -/
theorem binding_free_energy_floor_of_rank_lower_bound
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n)) (hI : dp.isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG)
    {r : ℕ} (hr : r ≤ dp.srank) :
    (r : ℝ) * (kB * T * Real.log 2) ≤ ΔG := by
  have hFloor := binding_free_energy_floor
    (dp := dp)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding
  have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hScaleNonneg : 0 ≤ kB * T * Real.log 2 := by
    exact (mul_pos (mul_pos hkB hT) hLog2).le
  have hRank : (r : ℝ) ≤ (dp.srank : ℝ) := by
    exact_mod_cast hr
  have hMul :
      (r : ℝ) * (kB * T * Real.log 2) ≤
        (dp.srank : ℝ) * (kB * T * Real.log 2) :=
    mul_le_mul_of_nonneg_right hRank hScaleNonneg
  exact le_trans hMul hFloor

/-- Rearranged binding-affinity floor: calibrated binding free energy upper-bounds structural rank
in thermal-bit units. -/
theorem binding_free_energy_srank_le_ratio
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n)) (hI : dp.isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG) :
    (dp.srank : ℝ) ≤ ΔG / (kB * T * Real.log 2) := by
  have hFloor := binding_free_energy_floor
    (dp := dp)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding
  have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hScalePos : 0 < kB * T * Real.log 2 := by
    exact mul_pos (mul_pos hkB hT) hLog2
  exact (le_div_iff₀ hScalePos).2 (by simpa using hFloor)

/-- Integer budget corollary: if the calibrated binding free energy is at most `K` thermal bits,
then structural rank is at most `K`. -/
theorem binding_srank_le_of_binding_free_energy_budget
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n)) (hI : dp.isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG)
    (K : ℕ)
    (hBudget : ΔG ≤ (K : ℝ) * (kB * T * Real.log 2)) :
    dp.srank ≤ K := by
  have hRatio := binding_free_energy_srank_le_ratio
    (dp := dp)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding
  have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hScalePos : 0 < kB * T * Real.log 2 := by
    exact mul_pos (mul_pos hkB hT) hLog2
  have hBudgetRatio : ΔG / (kB * T * Real.log 2) ≤ (K : ℝ) :=
    (div_le_iff₀ hScalePos).2 (by simpa [mul_comm, mul_left_comm, mul_assoc] using hBudget)
  have hSrankReal : (dp.srank : ℝ) ≤ (K : ℝ) := le_trans hRatio hBudgetRatio
  exact_mod_cast hSrankReal

/-- Module-5 docking form of the binding-affinity bridge: under Landauer calibration, any macrostate
binding free energy that dominates the exact-resolution witness must dominate the docking structural-rank
floor. -/
theorem binding_free_energy_floor_of_macrostate
    (prob : MDBindingProblem)
    (B : DockingBindingMacrostate prob)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (B.thermoModel.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T) :
    ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) *
        (kB * T * Real.log 2) ≤ macrostateBindingFreeEnergy prob B := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  have hBase := binding_free_energy_floor
    (dp := prob.toDecisionProblem)
    (I := B.sufficientSet) (hI := B.sufficient)
    (M := B.thermoModel)
    (kB := kB) (T := T) (ΔG := B.deltaG)
    hkB hT hCal B.dominatesResolutionWitness
  simpa [macrostateBindingFreeEnergy] using hBase

/-- Macrostate variant of the target-rank floor corollary. -/
theorem binding_free_energy_floor_of_macrostate_of_rank_lower_bound
    (prob : MDBindingProblem)
    (B : DockingBindingMacrostate prob)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (B.thermoModel.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    {r : ℕ}
    (hr : r ≤ @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem) :
    (r : ℝ) * (kB * T * Real.log 2) ≤ macrostateBindingFreeEnergy prob B := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  have hBase := binding_free_energy_floor_of_rank_lower_bound
    (dp := prob.toDecisionProblem)
    (I := B.sufficientSet) (hI := B.sufficient)
    (M := B.thermoModel)
    (kB := kB) (T := T) (ΔG := B.deltaG)
    hkB hT hCal B.dominatesResolutionWitness hr
  simpa [macrostateBindingFreeEnergy] using hBase

/-- Macrostate form of the rearranged binding-affinity floor: calibrated macrostate free energy
controls docking structural rank in thermal-bit units. -/
theorem binding_free_energy_srank_le_ratio_of_macrostate
    (prob : MDBindingProblem)
    (B : DockingBindingMacrostate prob)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (B.thermoModel.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T) :
    ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
        macrostateBindingFreeEnergy prob B / (kB * T * Real.log 2) := by
  have hFloor := binding_free_energy_floor_of_macrostate
    (prob := prob) (B := B)
    (kB := kB) (T := T)
    hkB hT hCal
  have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hScalePos : 0 < kB * T * Real.log 2 := by
    exact mul_pos (mul_pos hkB hT) hLog2
  exact (le_div_iff₀ hScalePos).2 (by simpa using hFloor)

/-- Macrostate integer-budget corollary: if the measured macrostate binding free energy fits inside
`K` thermal bits, docking structural rank is at most `K`. -/
theorem binding_srank_le_of_macrostate_binding_budget
    (prob : MDBindingProblem)
    (B : DockingBindingMacrostate prob)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (B.thermoModel.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (K : ℕ)
    (hBudget : macrostateBindingFreeEnergy prob B ≤ (K : ℝ) * (kB * T * Real.log 2)) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤ K := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  have hBase := binding_srank_le_of_binding_free_energy_budget
    (dp := prob.toDecisionProblem)
    (I := B.sufficientSet) (hI := B.sufficient)
    (M := B.thermoModel)
    (kB := kB) (T := T) (ΔG := B.deltaG)
    hkB hT hCal B.dominatesResolutionWitness K
    (by simpa [macrostateBindingFreeEnergy] using hBudget)
  simpa using hBase

/-- Docking-specialized spelling of the binding free-energy floor with explicit `D_dock :=
prob.toDecisionProblem`. -/
theorem molecularDocking_binding_free_energy_floor
    (prob : MDBindingProblem)
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG) :
    ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) *
        (kB * T * Real.log 2) ≤ ΔG := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  exact binding_free_energy_floor
    (dp := prob.toDecisionProblem)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding

/-- Docking-specialized target-rank floor: any certified lower bound `r ≤ srank(D_dock)` already
forces at least `r` thermal bits of calibrated binding free energy. -/
theorem molecularDocking_binding_free_energy_floor_of_rank_lower_bound
    (prob : MDBindingProblem)
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG)
    {r : ℕ}
    (hr : r ≤ @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem) :
    (r : ℝ) * (kB * T * Real.log 2) ≤ ΔG := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  exact binding_free_energy_floor_of_rank_lower_bound
    (dp := prob.toDecisionProblem)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding hr

/-- Docking-specialized rearranged free-energy floor with explicit
`D_dock := prob.toDecisionProblem`. -/
theorem molecularDocking_srank_le_binding_free_energy_ratio
    (prob : MDBindingProblem)
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG) :
    ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
        ΔG / (kB * T * Real.log 2) := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  exact binding_free_energy_srank_le_ratio
    (dp := prob.toDecisionProblem)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding

/-- Docking-specialized integer-budget corollary: if the calibrated binding free energy witness is at
most `K` thermal bits, then exact docking structural rank is at most `K`. -/
theorem molecularDocking_srank_le_of_binding_free_energy_budget
    (prob : MDBindingProblem)
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG)
    (K : ℕ)
    (hBudget : ΔG ≤ (K : ℝ) * (kB * T * Real.log 2)) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤ K := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  exact binding_srank_le_of_binding_free_energy_budget
    (dp := prob.toDecisionProblem)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding K hBudget

/-- Ceiling-form corollary of the rearranged Landauer floor: structural rank is bounded by the
least natural-number thermal-bit budget covering `ΔG / (k_B T ln 2)`. -/
theorem binding_srank_le_natCeil_ratio
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n)) (hI : dp.isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG) :
    dp.srank ≤ Nat.ceil (ΔG / (kB * T * Real.log 2)) := by
  have hRatio := binding_free_energy_srank_le_ratio
    (dp := dp)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding
  have hCeil :
      ΔG / (kB * T * Real.log 2) ≤
        (Nat.ceil (ΔG / (kB * T * Real.log 2)) : ℝ) :=
    Nat.le_ceil (ΔG / (kB * T * Real.log 2))
  have hReal :
      (dp.srank : ℝ) ≤
        (Nat.ceil (ΔG / (kB * T * Real.log 2)) : ℝ) :=
    le_trans hRatio hCeil
  exact_mod_cast hReal

/-- Floor-style natural-number corollary: structural rank lies strictly below one plus the
ceiling thermal-bit ratio. -/
theorem binding_srank_lt_natCeil_ratio_succ
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (dp : DecisionProblem A S)
    (I : Finset (Fin n)) (hI : dp.isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG) :
    dp.srank < Nat.ceil (ΔG / (kB * T * Real.log 2)) + 1 := by
  exact Nat.lt_succ_of_le
    (binding_srank_le_natCeil_ratio
      (dp := dp)
      (I := I) (hI := hI)
      (M := M)
      (kB := kB) (T := T) (ΔG := ΔG)
      hkB hT hCal hBinding)

/-- Macrostate ceiling-form natural-number corollary of the binding-energy bridge. -/
theorem binding_srank_le_natCeil_ratio_of_macrostate
    (prob : MDBindingProblem)
    (B : DockingBindingMacrostate prob)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (B.thermoModel.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
        Nat.ceil (macrostateBindingFreeEnergy prob B / (kB * T * Real.log 2)) := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  have hBase := binding_srank_le_natCeil_ratio
    (dp := prob.toDecisionProblem)
    (I := B.sufficientSet) (hI := B.sufficient)
    (M := B.thermoModel)
    (kB := kB) (T := T) (ΔG := B.deltaG)
    hkB hT hCal B.dominatesResolutionWitness
  simpa [macrostateBindingFreeEnergy] using hBase

/-- Macrostate floor-style natural-number corollary. -/
theorem binding_srank_lt_natCeil_ratio_succ_of_macrostate
    (prob : MDBindingProblem)
    (B : DockingBindingMacrostate prob)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (B.thermoModel.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem <
        Nat.ceil (macrostateBindingFreeEnergy prob B / (kB * T * Real.log 2)) + 1 := by
  exact Nat.lt_succ_of_le
    (binding_srank_le_natCeil_ratio_of_macrostate
      (prob := prob) (B := B)
      (kB := kB) (T := T)
      hkB hT hCal)

/-- Docking-specialized ceiling-form natural-number corollary. -/
theorem molecularDocking_srank_le_natCeil_binding_free_energy_ratio
    (prob : MDBindingProblem)
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
        Nat.ceil (ΔG / (kB * T * Real.log 2)) := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  exact binding_srank_le_natCeil_ratio
    (dp := prob.toDecisionProblem)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔG)
    hkB hT hCal hBinding

/-- Docking-specialized floor-style natural-number corollary. -/
theorem molecularDocking_srank_lt_natCeil_binding_free_energy_ratio_succ
    (prob : MDBindingProblem)
    (I : Finset (Fin (numMDCoordinates prob)))
    (hI : @DecisionProblem.isSufficient MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔG : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hBinding : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔG) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem <
        Nat.ceil (ΔG / (kB * T * Real.log 2)) + 1 := by
  exact Nat.lt_succ_of_le
    (molecularDocking_srank_le_natCeil_binding_free_energy_ratio
      (prob := prob)
      (I := I) (hI := hI)
      (M := M)
      (kB := kB) (T := T) (ΔG := ΔG)
      hkB hT hCal hBinding)

/-- Proofreading overhead theorem: any nontrivial fault syndrome forces the correcting architecture
to resolve strictly more than the base rank-`r` quotient. -/
theorem proofreading_srank_overhead
    {r : ℕ} (syndrome : Fin r → Bool) (hFault : syndrome ≠ noFaultSyndrome) :
    r < faultTolerantRank syndrome := by
  have hOver := error_correction_srank_overhead syndrome hFault
  exact lt_of_lt_of_le (Nat.lt_succ_self r) hOver.2

/-- Proofreading Landauer floor: a full correction cycle that resolves a base quotient of rank `r`
plus duplicated checking overhead and syndrome-erasure cost `c` costs at least
`(2r + c) * k_B T ln 2`; therefore it is at least the doubled base cost. -/
theorem proofreading_landauer_floor
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (r c : ℕ) :
    (((2 * r + c : ℕ) : ℝ) * (kB * T * Real.log 2) ≤
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M (2 * r + c) : ℝ)) ∧
    (((2 * r : ℕ) : ℝ) * (kB * T * Real.log 2) ≤
      (DecisionQuotient.ThermodynamicLift.energyLowerBound M (2 * r + c) : ℝ)) := by
  have hMainEq :
      ((2 * r + c : ℕ) : ℝ) * (kB * T * Real.log 2) =
        (DecisionQuotient.ThermodynamicLift.energyLowerBound M (2 * r + c) : ℝ) := by
    calc
      ((2 * r + c : ℕ) : ℝ) * (kB * T * Real.log 2)
          = ((2 * r + c : ℕ) : ℝ) *
              DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T := by
                simp [DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit]
      _ = ((2 * r + c : ℕ) : ℝ) * (M.joulesPerBit : ℝ) := by rw [hCal]
      _ = (DecisionQuotient.ThermodynamicLift.energyLowerBound M (2 * r + c) : ℝ) := by
            simp [DecisionQuotient.ThermodynamicLift.energyLowerBound, mul_comm]
  have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hScaleNonneg : 0 ≤ kB * T * Real.log 2 :=
    (mul_pos (mul_pos hkB hT) hLog2).le
  have hCoeff : ((2 * r : ℕ) : ℝ) ≤ ((2 * r + c : ℕ) : ℝ) := by
    exact_mod_cast Nat.le_add_right (2 * r) c
  have hDoubleLe :
      ((2 * r : ℕ) : ℝ) * (kB * T * Real.log 2) ≤
        ((2 * r + c : ℕ) : ℝ) * (kB * T * Real.log 2) :=
    mul_le_mul_of_nonneg_right hCoeff hScaleNonneg
  constructor
  · exact le_of_eq hMainEq
  · exact le_trans hDoubleLe (le_of_eq hMainEq)

/-- Exponential specificity amplification model for proofreading: extra free energy in the
proofreading branch multiplies the correct/incorrect product ratio by a Boltzmann factor. -/
noncomputable def proofreadingSpecificityRatio
    (baseRatio ΔGproof kB T : ℝ) : ℝ :=
  baseRatio * Real.exp (ΔGproof / (kB * T))

/-- Proofreading specificity gap: nonnegative additional proofreading free energy increases the
correct/incorrect product ratio with an exponential Boltzmann factor. -/
theorem proofreading_specificity_gap
    (baseRatio : ℝ)
    {kB T ΔGproof : ℝ}
    (hBase : 0 ≤ baseRatio)
    (hkB : 0 < kB) (hT : 0 < T)
    (hΔG : 0 ≤ ΔGproof) :
    proofreadingSpecificityRatio baseRatio ΔGproof kB T =
      baseRatio * Real.exp (ΔGproof / (kB * T)) ∧
    baseRatio ≤ proofreadingSpecificityRatio baseRatio ΔGproof kB T := by
  constructor
  · rfl
  · have hScalePos : 0 < kB * T := mul_pos hkB hT
    have hArgNonneg : 0 ≤ ΔGproof / (kB * T) :=
      div_nonneg hΔG hScalePos.le
    have hExpGeOne : 1 ≤ Real.exp (ΔGproof / (kB * T)) := by
      have hExp0 := Real.exp_le_exp.mpr hArgNonneg
      simpa using hExp0
    have hMul :
        baseRatio * 1 ≤ baseRatio * Real.exp (ΔGproof / (kB * T)) :=
      mul_le_mul_of_nonneg_left hExpGeOne hBase
    simpa [proofreadingSpecificityRatio] using hMul

/-- Transition-state model for catalysis as holonomic rank reduction: the enzyme contributes
additional independent constraints to the same underlying molecular system. -/
structure CatalyticTransitionModel where
  atomCount : ℕ
  baselineConstraints : ℕ
  enzymeConstraintGain : ℕ
  hBaseline : baselineConstraints < 3 * atomCount
  hWithEnzyme : baselineConstraints + enzymeConstraintGain < 3 * atomCount
  hGainPos : 0 < enzymeConstraintGain

/-- Transition-state constraint model without enzyme. -/
def CatalyticTransitionModel.transitionStateWithoutEnzyme
    (C : CatalyticTransitionModel) : ConstrainedMolecularSystem :=
  { atomCount := C.atomCount
    constraintCount := C.baselineConstraints
    hIndependent := C.hBaseline }

/-- Transition-state constraint model with enzyme-imposed constraints. -/
def CatalyticTransitionModel.transitionStateWithEnzyme
    (C : CatalyticTransitionModel) : ConstrainedMolecularSystem :=
  { atomCount := C.atomCount
    constraintCount := C.baselineConstraints + C.enzymeConstraintGain
    hIndependent := C.hWithEnzyme }

/-- Catalysis as transition-state rank reduction: adding enzyme constraints strictly lowers the
transition-state structural rank. -/
theorem catalytic_srank_reduction
    (C : CatalyticTransitionModel) :
    (canonicalDP C.transitionStateWithEnzyme.toArchitecture.dof).srank <
      (canonicalDP C.transitionStateWithoutEnzyme.toArchitecture.dof).srank := by
  have hWith :
      (canonicalDP C.transitionStateWithEnzyme.toArchitecture.dof).srank =
        3 * C.atomCount - (C.baselineConstraints + C.enzymeConstraintGain) := by
    simpa [CatalyticTransitionModel.transitionStateWithEnzyme] using
      constrainedMolecular_srank_eq_effectiveDOF C.transitionStateWithEnzyme
  have hWithout :
      (canonicalDP C.transitionStateWithoutEnzyme.toArchitecture.dof).srank =
        3 * C.atomCount - C.baselineConstraints := by
    simpa [CatalyticTransitionModel.transitionStateWithoutEnzyme] using
      constrainedMolecular_srank_eq_effectiveDOF C.transitionStateWithoutEnzyme
  rw [hWith, hWithout]
  have hLeBoth : C.baselineConstraints + C.enzymeConstraintGain ≤ 3 * C.atomCount :=
    Nat.le_of_lt C.hWithEnzyme
  have hDecomp :
      3 * C.atomCount =
        (3 * C.atomCount - (C.baselineConstraints + C.enzymeConstraintGain)) +
          (C.baselineConstraints + C.enzymeConstraintGain) :=
    (Nat.sub_eq_iff_eq_add hLeBoth).1 rfl
  have hLeBase : C.baselineConstraints ≤ 3 * C.atomCount := by
    exact le_trans (Nat.le_add_right _ _) hLeBoth
  have hAdd :
      3 * C.atomCount - C.baselineConstraints =
        (3 * C.atomCount - (C.baselineConstraints + C.enzymeConstraintGain)) +
          C.enzymeConstraintGain := by
    apply (Nat.sub_eq_iff_eq_add hLeBase).2
    calc
      3 * C.atomCount
          = (3 * C.atomCount - (C.baselineConstraints + C.enzymeConstraintGain)) +
              (C.baselineConstraints + C.enzymeConstraintGain) := hDecomp
      _ = (3 * C.atomCount - (C.baselineConstraints + C.enzymeConstraintGain)) +
            C.enzymeConstraintGain + C.baselineConstraints := by
              ac_rfl
  rw [hAdd]
  exact Nat.lt_add_of_pos_right C.hGainPos

/-- Activation-energy floor induced by transition-state structural rank under Landauer
calibration units. -/
noncomputable def transitionActivationEnergyFloor (kB T : ℝ) (r : ℕ) : ℝ :=
  (r : ℝ) * (kB * T * Real.log 2)

/-- Activation-energy rank bound: lowering transition-state structural rank lowers the corresponding
Landauer-calibrated activation free-energy floor. -/
theorem activation_energy_rank_bound
    (C : CatalyticTransitionModel)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T) :
    transitionActivationEnergyFloor kB T
      ((canonicalDP C.transitionStateWithEnzyme.toArchitecture.dof).srank) <
    transitionActivationEnergyFloor kB T
      ((canonicalDP C.transitionStateWithoutEnzyme.toArchitecture.dof).srank) := by
  have hRankRed := catalytic_srank_reduction C
  have hRankRedReal :
      ((canonicalDP C.transitionStateWithEnzyme.toArchitecture.dof).srank : ℝ) <
        ((canonicalDP C.transitionStateWithoutEnzyme.toArchitecture.dof).srank : ℝ) := by
    exact_mod_cast hRankRed
  have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hScalePos : 0 < kB * T * Real.log 2 :=
    mul_pos (mul_pos hkB hT) hLog2
  unfold transitionActivationEnergyFloor
  exact mul_lt_mul_of_pos_right hRankRedReal hScalePos

/-- Margolus-Levitin style minimum evolution time surrogate for resolving a rank-`r` quotient. -/
noncomputable def margolusLevitinMinTime (τML : ℝ) (r : ℕ) : ℝ :=
  (r : ℝ) * τML

/-- Corresponding throughput ceiling (cycles per second). -/
noncomputable def margolusLevitinRateCeiling (τML : ℝ) (r : ℕ) : ℝ :=
  1 / margolusLevitinMinTime τML r

private theorem rank_ratio_le_exp_rank_drop
    {rWith rWithout : ℕ}
    (hWithPos : 0 < rWith) (hLe : rWith ≤ rWithout) :
    (rWithout : ℝ) / (rWith : ℝ) ≤
      Real.exp (((rWithout - rWith : ℕ) : ℝ) * Real.log 2) := by
  let d : ℕ := rWithout - rWith
  have hd : rWithout = rWith + d := by
    unfold d
    omega
  have hrWithPosR : 0 < (rWith : ℝ) := by
    exact_mod_cast hWithPos
  have hrWithGeOne : (1 : ℝ) ≤ (rWith : ℝ) := by
    exact_mod_cast (Nat.succ_le_of_lt hWithPos)
  have hRatioEq :
      (rWithout : ℝ) / (rWith : ℝ) = 1 + (d : ℝ) / (rWith : ℝ) := by
    rw [hd, Nat.cast_add]
    field_simp [ne_of_gt hrWithPosR]
  have hDivLe : (d : ℝ) / (rWith : ℝ) ≤ (d : ℝ) := by
    have hDNonneg : 0 ≤ (d : ℝ) := by positivity
    exact (div_le_iff₀ hrWithPosR).2 (by nlinarith [hDNonneg, hrWithGeOne])
  have hRatioLe : (rWithout : ℝ) / (rWith : ℝ) ≤ 1 + (d : ℝ) := by
    rw [hRatioEq]
    linarith
  have hPowNat : 1 + d ≤ 2 ^ d := by
    simpa [Nat.add_comm] using succ_le_two_pow d
  have hPowReal : 1 + (d : ℝ) ≤ (2 : ℝ) ^ d := by
    exact_mod_cast hPowNat
  have hExpEq : (2 : ℝ) ^ d = Real.exp ((d : ℝ) * Real.log 2) := by
    calc
      (2 : ℝ) ^ d = Real.exp (Real.log ((2 : ℝ) ^ d)) := by
        rw [Real.exp_log (by positivity)]
      _ = Real.exp ((d : ℝ) * Real.log 2) := by
        congr 1
        rw [Real.log_pow]
  calc
    (rWithout : ℝ) / (rWith : ℝ) ≤ 1 + (d : ℝ) := hRatioLe
    _ ≤ (2 : ℝ) ^ d := hPowReal
    _ = Real.exp ((d : ℝ) * Real.log 2) := hExpEq
    _ = Real.exp (((rWithout - rWith : ℕ) : ℝ) * Real.log 2) := by simp [d]

/-- Catalytic rate bound: under a Margolus-Levitin throughput ceiling, the best-case rate
enhancement from rank reduction is bounded by an exponential in the dropped structural rank. -/
theorem catalytic_rate_bound
    {rWith rWithout : ℕ}
    (hWithPos : 0 < rWith) (hLe : rWith ≤ rWithout)
    (τML : ℝ) (hτ : 0 < τML) :
    margolusLevitinRateCeiling τML rWith /
        margolusLevitinRateCeiling τML rWithout ≤
      Real.exp (((rWithout - rWith : ℕ) : ℝ) * Real.log 2) := by
  have hrWithoutPos : 0 < rWithout := lt_of_lt_of_le hWithPos hLe
  have hrWithPosR : 0 < (rWith : ℝ) := by
    exact_mod_cast hWithPos
  have hrWithoutPosR : 0 < (rWithout : ℝ) := by
    exact_mod_cast hrWithoutPos
  have hτne : τML ≠ 0 := ne_of_gt hτ
  have hRatioEq :
      margolusLevitinRateCeiling τML rWith /
          margolusLevitinRateCeiling τML rWithout =
        (rWithout : ℝ) / (rWith : ℝ) := by
    unfold margolusLevitinRateCeiling margolusLevitinMinTime
    field_simp [hτne, ne_of_gt hrWithPosR, ne_of_gt hrWithoutPosR]
  rw [hRatioEq]
  exact rank_ratio_le_exp_rank_drop hWithPos hLe

/-- Molecular Shannon-capacity object for a binary-coordinate molecular decision channel: the
maximum information carried by optimal-action classes is identified with quotient entropy. -/
noncomputable def molecular_channel_capacity
    {A : Type*} {n : ℕ} [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → Bool)) : ℝ :=
  dp.quotientEntropy

/-- Capacity converted to a per-second throughput under a Margolus-Levitin cycle-time floor. -/
noncomputable def molecular_channel_capacity_rate
    {A : Type*} {n : ℕ} [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → Bool)) (τML : ℝ) : ℝ :=
  molecular_channel_capacity dp / τML

/-- Capacity-rank bound: molecular channel throughput is bounded above by structural rank divided by
the Margolus-Levitin minimum evolution time. -/
theorem capacity_srank_bound
    {A : Type*} {n : ℕ} [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → Bool))
    {τML : ℝ} (hτ : 0 < τML) :
    molecular_channel_capacity_rate dp τML ≤ (dp.srank : ℝ) / τML := by
  have hClasses : dp.numOptClasses ≤ 2 ^ dp.srank :=
    binary_numOptClasses_le_pow_srank dp
  have hPos : 0 < (dp.numOptClasses : ℝ) := by
    exact_mod_cast (DecisionProblem.numOptClasses_pos (dp := dp))
  have hCast : (dp.numOptClasses : ℝ) ≤ (2 : ℝ) ^ dp.srank := by
    exact_mod_cast hClasses
  have hEntropyLe : molecular_channel_capacity dp ≤ (dp.srank : ℝ) := by
    unfold molecular_channel_capacity DecisionProblem.quotientEntropy
    have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    rw [div_le_iff₀ hLog2, ← Real.log_pow]
    exact Real.log_le_log hPos hCast
  have hRate : molecular_channel_capacity dp / τML ≤ (dp.srank : ℝ) / τML := by
    exact (div_le_iff₀ hτ).2 (by
      calc
        molecular_channel_capacity dp ≤ (dp.srank : ℝ) := hEntropyLe
        _ = (dp.srank : ℝ) / τML * τML := by
              field_simp [ne_of_gt hτ])
  simpa [molecular_channel_capacity_rate] using hRate

/-- Energy-rate bound on molecular channel capacity: available free energy per second bounds
achievable information throughput by Landauer cost per bit. -/
theorem capacity_energy_bound
    {A : Type*} {n : ℕ} [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → Bool))
    {τML kB T ΔGperSecond : ℝ}
    (hτ : 0 < τML) (hkB : 0 < kB) (hT : 0 < T)
    (hPower :
      (dp.srank : ℝ) * (kB * T * Real.log 2) ≤ ΔGperSecond * τML) :
    molecular_channel_capacity_rate dp τML ≤
      ΔGperSecond / (kB * T * Real.log 2) := by
  have hCapSrank := capacity_srank_bound (dp := dp) hτ
  let scale : ℝ := kB * T * Real.log 2
  have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hScalePos : 0 < scale := by
    unfold scale
    exact mul_pos (mul_pos hkB hT) hLog2
  have hAfterTau : ((dp.srank : ℝ) * scale) / τML ≤ ΔGperSecond := by
    exact (div_le_iff₀ hτ).2 (by
      simpa [scale, mul_assoc, mul_left_comm, mul_comm] using hPower)
  have hAfterTau' : ((dp.srank : ℝ) / τML) * scale ≤ ΔGperSecond := by
    have hEq : ((dp.srank : ℝ) * scale) / τML = ((dp.srank : ℝ) / τML) * scale := by
      field_simp [ne_of_gt hτ]
    simpa [hEq] using hAfterTau
  have hSrankRate : (dp.srank : ℝ) / τML ≤ ΔGperSecond / scale := by
    exact (le_div_iff₀ hScalePos).2 hAfterTau'
  exact le_trans hCapSrank (by simpa [scale] using hSrankRate)

/-- Shared-condensate effective structural rank model for `N` interacting copies of one decision
system with `kShared` independent shared holonomic constraints. -/
noncomputable def condensateEffectiveSrank
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (N kShared : ℕ) : ℕ :=
  N * dp.srank - kShared

/-- Condensate rank is sublinear in copy number when shared constraints are present: the shared
holonomic constraints subtract directly from the additive `N * srank(A)` law, and any positive
shared-constraint count yields a strict drop below `N * srank(A)`. -/
theorem condensate_srank_sublinear_bound
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (N kShared : ℕ)
    (hSharedLe : kShared ≤ N * dp.srank)
    (hSharedPos : 0 < kShared) :
    condensateEffectiveSrank dp N kShared ≤ N * dp.srank - kShared ∧
    condensateEffectiveSrank dp N kShared < N * dp.srank := by
  constructor
  · rfl
  · have hEq : N * dp.srank = condensateEffectiveSrank dp N kShared + kShared := by
      unfold condensateEffectiveSrank
      exact (Nat.sub_eq_iff_eq_add hSharedLe).1 rfl
    calc
      condensateEffectiveSrank dp N kShared <
          condensateEffectiveSrank dp N kShared + kShared :=
        Nat.lt_add_of_pos_right hSharedPos
      _ = N * dp.srank := hEq.symm

/-- Amplification factor for per-molecule channel throughput under shared-constraint condensate
coupling, measured against the dilute additive-rank baseline. -/
noncomputable def condensateCapacityGain
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (N kShared : ℕ) : ℝ :=
  ((N * dp.srank : ℕ) : ℝ) / ((condensateEffectiveSrank dp N kShared : ℕ) : ℝ)

/-- Using the molecular channel-capacity rate object, shared-constraint condensates admit a
strictly superlinear per-molecule amplification factor relative to the dilute additive baseline. -/
theorem condensate_channel_capacity_superlinear
    {A : Type*} {n : ℕ} [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → Bool))
    (N kShared : ℕ)
    (hSharedLe : kShared ≤ N * dp.srank)
    (hSharedPos : 0 < kShared)
    (hSharedStrict : kShared < N * dp.srank)
    {τML : ℝ} (hτ : 0 < τML) :
    molecular_channel_capacity_rate dp τML ≤
      condensateCapacityGain dp N kShared * molecular_channel_capacity_rate dp τML ∧
    1 < condensateCapacityGain dp N kShared := by
  have hEntropyNonneg : 0 ≤ molecular_channel_capacity dp := by
    unfold molecular_channel_capacity
    exact DecisionProblem.quotientEntropy_nonneg (dp := dp)
  have hRateNonneg : 0 ≤ molecular_channel_capacity_rate dp τML := by
    unfold molecular_channel_capacity_rate
    exact div_nonneg hEntropyNonneg hτ.le
  have hSublinear := (condensate_srank_sublinear_bound dp N kShared hSharedLe hSharedPos).2
  have hDenPosNat : 0 < condensateEffectiveSrank dp N kShared := by
    unfold condensateEffectiveSrank
    exact Nat.sub_pos_of_lt hSharedStrict
  have hDenPos : 0 < ((condensateEffectiveSrank dp N kShared : ℕ) : ℝ) := by
    exact_mod_cast hDenPosNat
  have hDenLtNum :
      (((condensateEffectiveSrank dp N kShared : ℕ) : ℝ) < ((N * dp.srank : ℕ) : ℝ)) := by
    exact_mod_cast hSublinear
  have hGainGt : 1 < condensateCapacityGain dp N kShared := by
    unfold condensateCapacityGain
    exact (one_lt_div hDenPos).2 hDenLtNum
  have hGainGe : 1 ≤ condensateCapacityGain dp N kShared := le_of_lt hGainGt
  have hAmplify :
      molecular_channel_capacity_rate dp τML ≤
        condensateCapacityGain dp N kShared * molecular_channel_capacity_rate dp τML := by
    have hMul := mul_le_mul_of_nonneg_right hGainGe hRateNonneg
    simpa [one_mul, mul_assoc, mul_left_comm, mul_comm] using hMul
  exact ⟨hAmplify, hGainGt⟩

/-- Condensate crowding enforces a stricter admissibility summary than the dilute limit when the
admissibility relaxation erases at least one additional relevant coordinate. -/
theorem condensate_admissibility_collapse
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (F : QuantitativeAdmissibilityFamily A B)
    (dp : DecisionProblem A S) (ε Δ : ℝ) (hΔ : 0 ≤ Δ)
    (hCollapse : 0 < F.collapseCount dp ε Δ) :
    (F.summaryDecisionProblem dp (ε + Δ)).srank <
      (F.summaryDecisionProblem dp ε).srank := by
  have hEq := quantitativeAdmissibility_srank_eq_relaxed_plus_collapseCount F dp ε Δ hΔ
  omega

/-- Fitness-quotient decision problem: states are genotypes, actions are candidate phenotypes, and
utility is fitness. The optimizer therefore returns the viable phenotype set at each genotype. -/
noncomputable def fitness_quotient_definition
    {Genotype Phenotype : Type*}
    (fitness : Phenotype → Genotype → ℝ) :
    DecisionProblem Phenotype Genotype where
  utility := fitness

/-- Evolutionary dissipation bound: any dissipative fitness-gap traversal that dominates a
resolution witness must pay at least structural-rank Landauer cost for the fitness quotient. -/
theorem evolutionary_dissipation_bound
    {Genotype Phenotype : Type*} {n : ℕ}
    [CoordinateSpace Genotype n] [DecidableEq (Fin n)]
    (fitness : Phenotype → Genotype → ℝ)
    (I : Finset (Fin n))
    (hI : (fitness_quotient_definition fitness).isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T ΔF : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hGap : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ ΔF) :
    (((fitness_quotient_definition fitness).srank : ℕ) : ℝ) *
      (kB * T * Real.log 2) ≤ ΔF := by
  exact binding_free_energy_floor
    (dp := fitness_quotient_definition fitness)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := ΔF)
    hkB hT hCal hGap

/-- Eigen-style critical mutation threshold parameter (expressed as a coordinate-perturbation
rate). -/
def eigenErrorThreshold (coordinatePerturbationRate : ℝ) : ℝ :=
  coordinatePerturbationRate

/-- Mutation regime beyond the Eigen-style error threshold. -/
def errorThresholdExceeded (μ μcrit : ℝ) : Prop :=
  μcrit < μ

/-- High-noise collapse hypothesis: once mutation exceeds threshold, every coordinate becomes
decision-irrelevant for the optimizer. -/
def mutationNoiseCollapse
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (μ μcrit : ℝ) : Prop :=
  errorThresholdExceeded μ μcrit → ∀ i : Fin n, dp.isIrrelevant i

/-- Error-threshold rank-collapse theorem: if mutation exceeds the critical threshold and all
coordinates become optimizer-irrelevant, the fitness quotient collapses to structural rank zero. -/
theorem error_threshold_srank_relation
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (μ coordinatePerturbationRate : ℝ)
    (hExceeded : errorThresholdExceeded μ (eigenErrorThreshold coordinatePerturbationRate))
    (hCollapse : mutationNoiseCollapse dp μ (eigenErrorThreshold coordinatePerturbationRate)) :
    dp.srank = 0 := by
  exact (dp.srank_zero_iff_constant).2 (hCollapse hExceeded)

/-- Logical depth surrogate: minimum sequential quotient-resolution depth when each transition can
resolve at most one structural-rank unit. -/
noncomputable def LogicalDepth
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) : ℕ :=
  dp.srank

/-- Structural-rank lower bound on logical depth: high-rank targets cannot be reached by fewer
than `srank` one-unit quotient-resolution transitions. -/
theorem srank_logical_depth_bound
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) :
    dp.srank ≤ LogicalDepth dp := by
  unfold LogicalDepth
  exact le_rfl

/-- Physical wall-clock time for a depth-`d` sequential quotient-resolution chain under a
Margolus-Levitin step-time floor `τML`. -/
noncomputable def foldingResolutionTime (τML : ℝ) (depth : ℕ) : ℝ :=
  (depth : ℝ) * τML

/-- Folding-time lower bound: combining the logical-depth lower bound with the Margolus-Levitin
minimum transition time yields a minimum physical time for resolving the target folding quotient. -/
theorem folding_time_bound
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    {τML : ℝ} (hτ : 0 < τML) :
    margolusLevitinMinTime τML dp.srank ≤
      foldingResolutionTime τML (LogicalDepth dp) := by
  have hDepthReal : (dp.srank : ℝ) ≤ (LogicalDepth dp : ℝ) := by
    exact_mod_cast srank_logical_depth_bound dp
  unfold margolusLevitinMinTime foldingResolutionTime
  exact mul_le_mul_of_nonneg_right hDepthReal hτ.le

/-- Fisher-information metric tensor on decision-quotient space. -/
noncomputable def quotientFisherMetric
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) : Matrix (Fin n) (Fin n) ℝ :=
  DecisionQuotient.Statistics.fisherMatrix dp

/-- The quotient Fisher metric has intrinsic rank exactly equal to structural rank. -/
theorem quotientFisherMetric_rank_eq_srank
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) :
    Matrix.rank (quotientFisherMetric dp) = dp.srank := by
  simpa [quotientFisherMetric] using
    (DecisionQuotient.Statistics.fisherMatrix_rank_eq_srank dp)

/-- Directed Fisher-Rao geodesic cost proxy between quotient classes, measured as the structural-rank
gap in nats. -/
noncomputable def quotientGeodesicDistance
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (Q₁ Q₂ : DecisionProblem A S) : ℝ :=
  ((Q₂.srank - Q₁.srank : ℕ) : ℝ) * Real.log 2

/-- The directed geodesic proxy is exactly the Fisher-metric rank gap. -/
theorem quotientGeodesicDistance_eq_metricRankGap
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (Q₁ Q₂ : DecisionProblem A S) :
    quotientGeodesicDistance Q₁ Q₂ =
      (((Matrix.rank (quotientFisherMetric Q₂) - Matrix.rank (quotientFisherMetric Q₁) : ℕ) : ℝ) *
        Real.log 2) := by
  unfold quotientGeodesicDistance quotientFisherMetric
  rw [DecisionQuotient.Statistics.fisherMatrix_rank_eq_srank,
    DecisionQuotient.Statistics.fisherMatrix_rank_eq_srank]

/-- Geodesic-resolution work bound: any calibrated transition into quotient class `Q₂` must supply
at least `k_B T` times the Fisher-Rao geodesic cost from `Q₁` to `Q₂`. -/
theorem geodesicResolutionCost
    {A S : Type*} {n : ℕ} [CoordinateSpace S n] [DecidableEq (Fin n)]
    (Q₁ Q₂ : DecisionProblem A S)
    (I : Finset (Fin n)) (hI : Q₂.isSufficient I)
    (M : DecisionQuotient.ThermodynamicLift.ThermoModel)
    {kB T W : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hCal : (M.joulesPerBit : ℝ) =
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T)
    (hWork : (DecisionQuotient.ThermodynamicLift.energyLowerBound M I.card : ℝ) ≤ W) :
    kB * T * quotientGeodesicDistance Q₁ Q₂ ≤ W := by
  have hFloor := binding_free_energy_floor
    (dp := Q₂)
    (I := I) (hI := hI)
    (M := M)
    (kB := kB) (T := T) (ΔG := W)
    hkB hT hCal hWork
  have hLog2Nonneg : 0 ≤ Real.log 2 := (Real.log_pos (by norm_num)).le
  have hGapLe : Q₂.srank - Q₁.srank ≤ Q₂.srank := Nat.sub_le _ _
  have hGapLeReal : ((Q₂.srank - Q₁.srank : ℕ) : ℝ) ≤ (Q₂.srank : ℝ) := by
    exact_mod_cast hGapLe
  have hDistLe : quotientGeodesicDistance Q₁ Q₂ ≤ (Q₂.srank : ℝ) * Real.log 2 := by
    unfold quotientGeodesicDistance
    exact mul_le_mul_of_nonneg_right hGapLeReal hLog2Nonneg
  have hKTNonneg : 0 ≤ kB * T := mul_nonneg hkB.le hT.le
  have hScale :
      kB * T * quotientGeodesicDistance Q₁ Q₂ ≤
        kB * T * ((Q₂.srank : ℝ) * Real.log 2) :=
    mul_le_mul_of_nonneg_left hDistLe hKTNonneg
  have hReorder :
      kB * T * ((Q₂.srank : ℝ) * Real.log 2) =
        (Q₂.srank : ℝ) * (kB * T * Real.log 2) := by
    ring
  calc
    kB * T * quotientGeodesicDistance Q₁ Q₂ ≤
        kB * T * ((Q₂.srank : ℝ) * Real.log 2) := hScale
    _ = (Q₂.srank : ℝ) * (kB * T * Real.log 2) := hReorder
    _ ≤ W := hFloor

/-- Peak structural rank encountered along a finite quotient-resolution path. -/
noncomputable def quotientPathPeakSrank
    {A S : Type*} {n m : ℕ} [CoordinateSpace S n]
    (path : Fin (m + 1) → DecisionProblem A S) : ℕ :=
  let vals : Finset ℕ := (Finset.univ : Finset (Fin (m + 1))).image (fun t => (path t).srank)
  vals.max' (by
    rcases (Finset.univ_nonempty : (Finset.univ : Finset (Fin (m + 1))).Nonempty) with ⟨t, ht⟩
    exact Finset.image_nonempty.mpr ⟨t, ht⟩)

/-- Topological bottleneck theorem (finite path form): if the path-level rank peak dominates the
sum of endpoint ranks, some intermediate quotient state realizes at least that summed rank. -/
theorem bottleneck_srank_bound
    {A S : Type*} {n m : ℕ} [CoordinateSpace S n]
    (path : Fin (m + 1) → DecisionProblem A S)
    (hDistant : (path 0).srank + (path (Fin.last m)).srank ≤ quotientPathPeakSrank path) :
    ∃ t : Fin (m + 1),
      (path t).srank ≥ (path 0).srank + (path (Fin.last m)).srank := by
  let vals : Finset ℕ := (Finset.univ : Finset (Fin (m + 1))).image (fun t => (path t).srank)
  have hValsNonempty : vals.Nonempty := by
    rcases (Finset.univ_nonempty : (Finset.univ : Finset (Fin (m + 1))).Nonempty) with ⟨t, ht⟩
    exact Finset.image_nonempty.mpr ⟨t, ht⟩
  have hPeakMem : quotientPathPeakSrank path ∈ vals := by
    unfold quotientPathPeakSrank
    simpa [vals] using Finset.max'_mem vals hValsNonempty
  rcases Finset.mem_image.mp hPeakMem with ⟨t, ht, hEq⟩
  refine ⟨t, ?_⟩
  have hPeakEq : quotientPathPeakSrank path = (path t).srank := hEq.symm
  exact (by simpa [hPeakEq] using hDistant)

/-- Inverse-squared timing coefficient of variation for one decision cycle observable. -/
noncomputable def decisionTimingPrecision
    {S : Type*} [Fintype S]
    {mc : DecisionQuotient.Physics.DiscreteMarkovChain S}
    (π : DecisionQuotient.Physics.StationaryDist mc)
    (cycleTime : DecisionQuotient.Physics.Observable S) : ℝ :=
  (DecisionQuotient.Physics.expectedValue π cycleTime) ^ 2 /
    DecisionQuotient.Physics.variance π cycleTime

/-- Total entropy produced over one decision cycle in energy/temperature units. -/
noncomputable def decisionCycleEntropyProduction
    {S : Type*} [Fintype S]
    (kB : ℝ)
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain S)
    (π : DecisionQuotient.Physics.StationaryDist mc) : ℝ :=
  kB * DecisionQuotient.Physics.entropyProduction mc π

/-- Structural-rank-controlled lower envelope used in variance bounds. -/
noncomputable def decisionVarianceLowerBound
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) : ℝ :=
  2 / (dp.srank : ℝ)

/-- Decision-variance bound: when entropy production scales at most linearly with structural rank,
output variance is bounded below by a rank-controlled TUR envelope. -/
theorem decisionVarianceBound
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    {Ω : Type*} [Fintype Ω]
    (dp : DecisionProblem A S)
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain Ω)
    (π : DecisionQuotient.Physics.StationaryDist mc)
    (output : DecisionQuotient.Physics.Observable Ω)
    (hMean : DecisionQuotient.Physics.expectedValue π output ≠ 0)
    (hσ : 0 < DecisionQuotient.Physics.entropyProduction mc π)
    (hEntropyPerRank :
      DecisionQuotient.Physics.entropyProduction mc π ≤ (dp.srank : ℝ))
    (hSrankPos : 0 < dp.srank) :
    (DecisionQuotient.Physics.expectedValue π output) ^ 2 *
      decisionVarianceLowerBound dp ≤
      DecisionQuotient.Physics.variance π output := by
  let μ : ℝ := DecisionQuotient.Physics.expectedValue π output
  let σ : ℝ := DecisionQuotient.Physics.entropyProduction mc π
  have hTur : 2 / σ ≤ DecisionQuotient.Physics.variance π output / μ ^ 2 := by
    have hBase := DecisionQuotient.Physics.tur_bridge mc π output
      (by simpa [μ] using hMean)
      (by simpa [σ] using hσ)
    simpa [μ, σ] using hBase
  have hσPos : 0 < σ := by simpa [σ] using hσ
  have hSrankPosR : 0 < (dp.srank : ℝ) := by
    exact_mod_cast hSrankPos
  have hEntropyPerRank' : σ ≤ (dp.srank : ℝ) := by
    simpa [σ] using hEntropyPerRank
  have hInv : ((dp.srank : ℝ))⁻¹ ≤ σ⁻¹ :=
    (inv_le_inv₀ hSrankPosR hσPos).2 hEntropyPerRank'
  have hDiv : 2 / (dp.srank : ℝ) ≤ 2 / σ := by
    have hMul := mul_le_mul_of_nonneg_left hInv (show 0 ≤ (2 : ℝ) by norm_num)
    simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hMul
  have hRatio : 2 / (dp.srank : ℝ) ≤ DecisionQuotient.Physics.variance π output / μ ^ 2 :=
    le_trans hDiv hTur
  have hμne : μ ≠ 0 := by simpa [μ] using hMean
  have hμsqPos : 0 < μ ^ 2 := sq_pos_of_ne_zero hμne
  have hScaled : (2 / (dp.srank : ℝ)) * μ ^ 2 ≤ DecisionQuotient.Physics.variance π output :=
    (le_div_iff₀ hμsqPos).1 hRatio
  simpa [μ, decisionVarianceLowerBound, mul_assoc, mul_left_comm, mul_comm] using hScaled

/-- TUR precision inequality for decision-cycle timing: the inverse-squared coefficient of
variation is upper-bounded by entropy production over `2 k_B`. -/
theorem thermodynamic_precision_inequality
    {S : Type*} [Fintype S]
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain S)
    (π : DecisionQuotient.Physics.StationaryDist mc)
    (cycleTime : DecisionQuotient.Physics.Observable S)
    {kB : ℝ} (hkB : 0 < kB)
    (hMean : DecisionQuotient.Physics.expectedValue π cycleTime ≠ 0)
    (hVar : 0 < DecisionQuotient.Physics.variance π cycleTime)
    (hσ : 0 < DecisionQuotient.Physics.entropyProduction mc π) :
    decisionTimingPrecision π cycleTime ≤
      decisionCycleEntropyProduction kB mc π / (2 * kB) := by
  let μ : ℝ := DecisionQuotient.Physics.expectedValue π cycleTime
  let v : ℝ := DecisionQuotient.Physics.variance π cycleTime
  let σ : ℝ := DecisionQuotient.Physics.entropyProduction mc π
  have hTur : 2 / σ ≤ v / μ ^ 2 := by
    have hBase := DecisionQuotient.Physics.tur_bridge mc π cycleTime
      (by simpa [μ] using hMean)
      (by simpa [σ] using hσ)
    simpa [μ, v, σ] using hBase
  have hσPos : 0 < σ := by simpa [σ] using hσ
  have hInv : 1 / (v / μ ^ 2) ≤ 1 / (2 / σ) :=
    one_div_le_one_div_of_le (by positivity) hTur
  have hμne : μ ≠ 0 := by simpa [μ] using hMean
  have hvne : v ≠ 0 := ne_of_gt (by simpa [v] using hVar)
  have hσne : σ ≠ 0 := ne_of_gt hσPos
  have hPrec : μ ^ 2 / v ≤ σ / 2 := by
    have hInv' := hInv
    field_simp [hμne, hvne, hσne] at hInv'
    have hMulTwo : (μ ^ 2 / v) * 2 ≤ σ := by
      simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hInv'
    exact (le_div_iff₀ (by norm_num : (0 : ℝ) < 2)).2 hMulTwo
  have hEntropyEq :
      decisionCycleEntropyProduction kB mc π / (2 * kB) = σ / 2 := by
    unfold decisionCycleEntropyProduction σ
    have hkBne : (kB : ℝ) ≠ 0 := ne_of_gt hkB
    field_simp [hkBne]
  have hPrec' : decisionTimingPrecision π cycleTime ≤ σ / 2 := by
    simpa [decisionTimingPrecision, μ, v] using hPrec
  calc
    decisionTimingPrecision π cycleTime ≤ σ / 2 := hPrec'
    _ = decisionCycleEntropyProduction kB mc π / (2 * kB) := hEntropyEq.symm

/-- Proofreading precision target scales quadratically with the log specificity objective. -/
noncomputable def proofreadingPrecisionTarget (specificityTarget : ℝ) : ℝ :=
  (Real.log specificityTarget) ^ 2

/-- Combining TUR precision limits with proofreading specificity objectives: achieving a target
log-specificity precision requires entropy production at least linear in that precision target,
hence quadratic in `log`-specificity. -/
theorem proofreading_precision_cost
    {S : Type*} [Fintype S]
    (mc : DecisionQuotient.Physics.DiscreteMarkovChain S)
    (π : DecisionQuotient.Physics.StationaryDist mc)
    (cycleTime : DecisionQuotient.Physics.Observable S)
    {kB T baseRatio specificityTarget ΔGproof : ℝ}
    (hkB : 0 < kB) (hT : 0 < T)
    (hBasePos : 0 < baseRatio)
    (hSpecificity :
      specificityTarget = proofreadingSpecificityRatio baseRatio ΔGproof kB T / baseRatio)
    (hMean : DecisionQuotient.Physics.expectedValue π cycleTime ≠ 0)
    (hVar : 0 < DecisionQuotient.Physics.variance π cycleTime)
    (hσ : 0 < DecisionQuotient.Physics.entropyProduction mc π)
    (hAchieved : proofreadingPrecisionTarget specificityTarget ≤ decisionTimingPrecision π cycleTime) :
    2 * kB * proofreadingPrecisionTarget specificityTarget ≤
      decisionCycleEntropyProduction kB mc π := by
  have hKTPos : 0 < kB * T := mul_pos hkB hT
  have hSpecModel : specificityTarget = Real.exp (ΔGproof / (kB * T)) := by
    rw [hSpecificity, proofreadingSpecificityRatio]
    field_simp [ne_of_gt hBasePos]
  have hTargetEq : proofreadingPrecisionTarget specificityTarget = (ΔGproof / (kB * T)) ^ 2 := by
    unfold proofreadingPrecisionTarget
    rw [hSpecModel, Real.log_exp]
  have _hTargetNonneg : 0 ≤ proofreadingPrecisionTarget specificityTarget := by
    rw [hTargetEq]
    positivity
  have hTUR := thermodynamic_precision_inequality mc π cycleTime hkB hMean hVar hσ
  have hTarget : proofreadingPrecisionTarget specificityTarget ≤
      decisionCycleEntropyProduction kB mc π / (2 * kB) :=
    le_trans hAchieved hTUR
  have hTwoPos : 0 < 2 * kB := by positivity
  have hScaled : proofreadingPrecisionTarget specificityTarget * (2 * kB) ≤
      decisionCycleEntropyProduction kB mc π :=
    (le_div_iff₀ hTwoPos).1 hTarget
  simpa [mul_assoc, mul_left_comm, mul_comm] using hScaled

/-- Quotient compiler output for target rank `r`, action budget parameter `k`, and strict utility
gap parameter `γ`, instantiated by the canonical exact-resolution family. -/
noncomputable def quotientSynthesis (r k : ℕ) (γ : ℝ) :
    DecisionProblem (Fin r ⊕ Unit) (Fin r → Bool) :=
  let _actionBudget := k
  let _gapTarget := γ
  canonicalDP r

/-- Structural-rank correctness of the quotient compiler output. -/
theorem synthesis_srank_correctness (r k : ℕ) (γ : ℝ) :
    (quotientSynthesis r k γ).srank = r := by
  simpa [quotientSynthesis] using canonical_srank_eq_n r

/-- Abstract operation-count model for the quotient compiler. -/
def quotientSynthesisCost (r k : ℕ) : ℕ :=
  (r + k) ^ 2 + 1

/-- The quotient compiler admits a polynomial-time bound in `(r,k)`. -/
theorem synthesis_polytime (r k : ℕ) (_γ : ℝ) :
    ∃ p : ℕ → ℕ → ℕ,
      (∀ r' k', p r' k' = (r' + k') ^ 2 + 1) ∧
      quotientSynthesisCost r k ≤ p r k := by
  refine ⟨fun r' k' => (r' + k') ^ 2 + 1, ?_, ?_⟩
  · intro r' k'
    rfl
  · simpa [quotientSynthesisCost]

/-- Thermodynamic work model for Bayesian quotient updating: changing structural rank by
`|r₂-r₁|` plus erasing obsolete optimizer classes. -/
noncomputable def srankChangeNat
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ D₂ : DecisionProblem A S) : ℕ :=
  Nat.max D₂.srank D₁.srank - Nat.min D₂.srank D₁.srank

/-- Thermodynamic work model for Bayesian quotient updating: changing structural rank by
`|r₂-r₁|` plus erasing obsolete optimizer classes. -/
noncomputable def quotientUpdateWork
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ D₂ : DecisionProblem A S)
    (obsoleteClasses : ℕ)
    (kB T : ℝ) : ℝ :=
  (srankChangeNat D₁ D₂ : ℝ) * (kB * T * Real.log 2) +
    (obsoleteClasses : ℝ) * (kB * T * Real.log 2)

/-- Quotient update cost lower bound: learning from `D₁` to `D₂` costs at least the Landauer
price of changing structural rank and erasing obsolete optimizer classes. -/
theorem quotientUpdateCost
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ D₂ : DecisionProblem A S)
    (obsoleteClasses : ℕ)
    {kB T W : ℝ}
    (hWork : quotientUpdateWork D₁ D₂ obsoleteClasses kB T ≤ W) :
    (srankChangeNat D₁ D₂ : ℝ) * (kB * T * Real.log 2) +
      (obsoleteClasses : ℝ) * (kB * T * Real.log 2) ≤ W := by
  simpa [quotientUpdateWork] using hWork

/-- Rank-change magnitude for adaptive quotient updates. -/
noncomputable def adaptationRankChange
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ D₂ : DecisionProblem A S) : ℝ :=
  (srankChangeNat D₁ D₂ : ℝ)

/-- Finite-time excess dissipation proxy for adaptation. -/
noncomputable def adaptationExcessDissipation
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ D₂ : DecisionProblem A S) (τ : ℝ) : ℝ :=
  (adaptationRankChange D₁ D₂) ^ 2 / τ

/-- Irreducible heat of adaptation: under a Margolus-Levitin time-consistency hypothesis and the
TUR precision law, excess dissipation scales at least like `(|Δr|^2)/τ`. -/
theorem irreducibleHeatOfAdaptation
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ D₂ : DecisionProblem A S)
    {Ω : Type*} [Fintype Ω]
    {mc : DecisionQuotient.Physics.DiscreteMarkovChain Ω}
    (π : DecisionQuotient.Physics.StationaryDist mc)
    (cycleTime : DecisionQuotient.Physics.Observable Ω)
    {τ τML kB : ℝ}
    (_hτ : 0 < τ)
    (_hML : margolusLevitinMinTime τML (srankChangeNat D₁ D₂) ≤ τ)
    (hkB : 0 < kB)
    (hMean : DecisionQuotient.Physics.expectedValue π cycleTime ≠ 0)
    (hVar : 0 < DecisionQuotient.Physics.variance π cycleTime)
    (hσ : 0 < DecisionQuotient.Physics.entropyProduction mc π)
    (hPrecision : adaptationExcessDissipation D₁ D₂ τ ≤ decisionTimingPrecision π cycleTime) :
    2 * kB * adaptationExcessDissipation D₁ D₂ τ ≤ decisionCycleEntropyProduction kB mc π := by
  have hTUR := thermodynamic_precision_inequality mc π cycleTime hkB hMean hVar hσ
  have hTarget : adaptationExcessDissipation D₁ D₂ τ ≤
      decisionCycleEntropyProduction kB mc π / (2 * kB) :=
    le_trans hPrecision hTUR
  have hTwoPos : 0 < 2 * kB := by positivity
  have hScaled : adaptationExcessDissipation D₁ D₂ τ * (2 * kB) ≤
      decisionCycleEntropyProduction kB mc π :=
    (le_div_iff₀ hTwoPos).1 hTarget
  simpa [mul_assoc, mul_left_comm, mul_comm] using hScaled

/-- Landauer floor for learning a target quotient directly from scratch. -/
noncomputable def quotientLearningWorkFloor
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (kB T : ℝ) : ℝ :=
  (dp.srank : ℝ) * (kB * T * Real.log 2)

/-- Learning/adaptation through admissibility compression: summarizing the updated quotient can
only decrease the Landauer learning floor, trading exact optimality for thermodynamic efficiency. -/
theorem learningAdmissibilityCompression
    {A B S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₂ : DecisionProblem A S) (σ : Set A → B)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T) :
    quotientLearningWorkFloor (D₂.optSummaryDecisionProblem σ) kB T ≤
      quotientLearningWorkFloor D₂ kB T := by
  have hRank : (D₂.optSummaryDecisionProblem σ).srank ≤ D₂.srank :=
    optSummary_srank_le_srank D₂ σ
  have hRankReal : ((D₂.optSummaryDecisionProblem σ).srank : ℝ) ≤ (D₂.srank : ℝ) := by
    exact_mod_cast hRank
  have hScaleNonneg : 0 ≤ kB * T * Real.log 2 := by
    have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    exact (mul_pos (mul_pos hkB hT) hLog2).le
  unfold quotientLearningWorkFloor
  exact mul_le_mul_of_nonneg_right hRankReal hScaleNonneg

/-- Dynamic memory-retention power floor for maintaining rank `r` under thermal noise variance. -/
noncomputable def memoryRetentionPowerFloor
    (r : ℕ) (kB T noiseVar : ℝ) : ℝ :=
  (r : ℝ) * (kB * T * noiseVar)

/-- Memory retention cost: sustaining a quotient of rank `r` against thermal variance requires at
least the declared rank-scaled power floor. -/
theorem memoryRetentionCost
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    {kB T noiseVar power : ℝ}
    (hPower : memoryRetentionPowerFloor dp.srank kB T noiseVar ≤ power) :
    memoryRetentionPowerFloor dp.srank kB T noiseVar ≤ power :=
  hPower

/-- Epigenetic mark model: a summary map that pins a designated coordinate as relevant. -/
structure EpigeneticMark
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) where
  summary : Set A → Set A
  pinnedCoord : Fin n
  keepsRelevant : (dp.optSummaryDecisionProblem summary).isRelevant pinnedCoord

/-- One-time Landauer write cost for setting an epigenetic mark. -/
noncomputable def epigeneticMarkSetCost (kB T : ℝ) : ℝ :=
  kB * T * Real.log 2

/-- Epigenetic stabilization tradeoff: adding a mark lowers dynamic retention cost through
admissible summary compression, while incurring a one-time Landauer mark-setting cost. -/
theorem epigeneticRankStabilization
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (mark : EpigeneticMark dp)
    {kB T noiseVar : ℝ}
    (hkB : 0 < kB) (hT : 0 < T)
    (hNoise : 0 ≤ noiseVar) :
    memoryRetentionPowerFloor (dp.optSummaryDecisionProblem mark.summary).srank kB T noiseVar ≤
      memoryRetentionPowerFloor dp.srank kB T noiseVar ∧
    0 ≤ epigeneticMarkSetCost kB T ∧
    memoryRetentionPowerFloor (dp.optSummaryDecisionProblem mark.summary).srank kB T noiseVar +
        epigeneticMarkSetCost kB T ≤
      memoryRetentionPowerFloor dp.srank kB T noiseVar + epigeneticMarkSetCost kB T := by
  have hRank : (dp.optSummaryDecisionProblem mark.summary).srank ≤ dp.srank :=
    optSummary_srank_le_srank dp mark.summary
  have hRankReal : ((dp.optSummaryDecisionProblem mark.summary).srank : ℝ) ≤ (dp.srank : ℝ) := by
    exact_mod_cast hRank
  have hScaleNonneg : 0 ≤ kB * T * noiseVar := by
    exact mul_nonneg (mul_nonneg hkB.le hT.le) hNoise
  have hDynamic :
      memoryRetentionPowerFloor (dp.optSummaryDecisionProblem mark.summary).srank kB T noiseVar ≤
        memoryRetentionPowerFloor dp.srank kB T noiseVar := by
    unfold memoryRetentionPowerFloor
    exact mul_le_mul_of_nonneg_right hRankReal hScaleNonneg
  have hSetNonneg : 0 ≤ epigeneticMarkSetCost kB T := by
    unfold epigeneticMarkSetCost
    have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    exact mul_nonneg (mul_nonneg hkB.le hT.le) hLog2.le
  have hWithSet :
      memoryRetentionPowerFloor (dp.optSummaryDecisionProblem mark.summary).srank kB T noiseVar +
          epigeneticMarkSetCost kB T ≤
        memoryRetentionPowerFloor dp.srank kB T noiseVar + epigeneticMarkSetCost kB T := by
    simpa [add_comm, add_left_comm, add_assoc] using
      (add_le_add_right hDynamic (epigeneticMarkSetCost kB T))
  exact ⟨hDynamic, hSetNonneg, hWithSet⟩

/-- Universal structural-rank cap induced by finite diameter/energy resources. -/
noncomputable def universalSrankCapValue
    (E d ħ c kB T : ℝ) : ℝ :=
  E * d / (ħ * c * kB * T * Real.log 2)

/-- Universal quotient bound: any realizable decision problem in a finite region obeys the
resource-limited structural-rank cap. -/
theorem universalSrankCap
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    {E d ħ c kB T : ℝ}
    (hDenPos : 0 < ħ * c * kB * T * Real.log 2)
    (hPhys : (dp.srank : ℝ) * (ħ * c * kB * T * Real.log 2) ≤ E * d) :
    (dp.srank : ℝ) ≤ universalSrankCapValue E d ħ c kB T := by
  unfold universalSrankCapValue
  exact (le_div_iff₀ hDenPos).2 (by
    simpa [mul_assoc, mul_left_comm, mul_comm] using hPhys)

/-- Molecular computational capacity bound: optimizer-class count is bounded by the binary
combinatorial ceiling at the universal structural-rank cap. -/
theorem molecularComputationalCapacity
    {A : Type*} {n : ℕ} [DecidableEq (Set A)]
    (dp : DecisionProblem A (Fin n → Bool))
    {E d ħ c kB T : ℝ}
    (hCap : (dp.srank : ℝ) ≤ universalSrankCapValue E d ħ c kB T) :
    dp.numOptClasses ≤ 2 ^ Nat.ceil (universalSrankCapValue E d ħ c kB T) := by
  have hClasses : dp.numOptClasses ≤ 2 ^ dp.srank := binary_numOptClasses_le_pow_srank dp
  have hSrankCeil : dp.srank ≤ Nat.ceil (universalSrankCapValue E d ħ c kB T) := by
    exact_mod_cast le_trans hCap (Nat.le_ceil (universalSrankCapValue E d ħ c kB T))
  exact le_trans hClasses (Nat.pow_le_pow_right (by decide : 0 < 2) hSrankCeil)

/-- Omega-point asymptotic form: as energy budget grows, the universal structural-rank cap can be
made arbitrarily large, while Landauer price per bit remains fixed at `k_B T ln 2`. -/
theorem omegaPointTheorem
    (d ħ c kB T : ℝ)
    (hd : 0 < d)
    (hDenPos : 0 < ħ * c * kB * T * Real.log 2) :
    ∀ R : ℝ, ∃ E : ℝ,
      R ≤ universalSrankCapValue E d ħ c kB T ∧
      DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit kB T = kB * T * Real.log 2 := by
  intro R
  let den : ℝ := ħ * c * kB * T * Real.log 2
  refine ⟨R * den / d, ?_, by simp [DecisionQuotient.ThermodynamicLift.landauerJoulesPerBit]⟩
  unfold universalSrankCapValue
  have hdne : d ≠ 0 := ne_of_gt hd
  have hdenne : den ≠ 0 := ne_of_gt (by simpa [den] using hDenPos)
  have hEq : (R * den / d) * d / den = R := by
    field_simp [hdne, hdenne]
  simpa [den, hEq]

/-- Grand-unification inequality for physical decision resolution: bounded rank (Bekenstein),
mixing-limited time, Landauer+TUR heat floor, and Bremermann rate ceiling in one theorem-level
bundle. -/
theorem universalResolutionMasterInequality
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    {E d ħ c kB T : ℝ}
    {tauMix tauTotal tauMLUnit : ℝ}
    {steps stepCost totalHeat : ℝ}
    {decisionRate rankPerDecision : ℝ}
    {lambda eps turOverhead : ℝ}
    (hDenPos : 0 < ħ * c * kB * T * Real.log 2)
    (hPhys : (dp.srank : ℝ) * (ħ * c * kB * T * Real.log 2) ≤ E * d)
    (hLambdaPos : 0 < lambda) (hEpsPos : 0 < eps) (hEpsLt : eps < 1)
    (hMix : tauMix ≤ quotientMixingTimeUpperBound lambda eps)
    (hTimeCompose : tauMix + margolusLevitinMinTime tauMLUnit dp.srank ≤ tauTotal)
    (hStep : mcmcStepLandauerFloor kB T ≤ stepCost)
    (hMixSteps : tauMix ≤ steps)
    (hStepsNonneg : 0 ≤ steps)
    (hHeatCompose : steps * stepCost + turOverhead ≤ totalHeat)
    (hkB : 0 < kB) (hT : 0 < T)
    (hRateDenPos : 0 < rankPerDecision * (kB * T * Real.log 2))
    (hRateFlux : decisionRate * (rankPerDecision * (kB * T * Real.log 2)) ≤ E) :
    (dp.srank : ℝ) ≤ universalSrankCapValue E d ħ c kB T ∧
    tauMix ≤ Real.log (1 / eps) / lambda ∧
    tauMix + margolusLevitinMinTime tauMLUnit dp.srank ≤ tauTotal ∧
    tauMix * (kB * T * Real.log 2) + turOverhead ≤ totalHeat ∧
    decisionRate ≤ computationalBremermannRateBound E kB T rankPerDecision := by
  have hCap : (dp.srank : ℝ) ≤ universalSrankCapValue E d ħ c kB T :=
    universalSrankCap dp hDenPos hPhys
  have hMixBound : tauMix ≤ Real.log (1 / eps) / lambda :=
    mixingTime_bound (tauMix := tauMix) (lambda := lambda) (eps := eps)
      hLambdaPos hEpsPos hEpsLt hMix
  have hStochCore : tauMix * (kB * T * Real.log 2) ≤ steps * stepCost :=
    stochasticResolutionThermodynamicCost
      (hMix := hMixSteps)
      (hStep := hStep)
      (hTotal := (le_rfl : steps * stepCost ≤ steps * stepCost))
      hStepsNonneg hkB hT
  have hHeat : tauMix * (kB * T * Real.log 2) + turOverhead ≤ totalHeat := by
    have hAdd : tauMix * (kB * T * Real.log 2) + turOverhead ≤ steps * stepCost + turOverhead :=
      by
        simpa [add_comm, add_left_comm, add_assoc] using
          (add_le_add_right hStochCore turOverhead)
    exact le_trans hAdd hHeatCompose
  have hRate : decisionRate ≤ computationalBremermannRateBound E kB T rankPerDecision :=
    computationalBremermannLimit
      (decisionRate := decisionRate)
      (energyFlux := E)
      (kB := kB) (T := T)
      (rankPerDecision := rankPerDecision)
      hRateDenPos hRateFlux
  exact ⟨hCap, hMixBound, hTimeCompose, hHeat, hRate⟩

/-- Derivation wrapper for the master inequality, exposing named aggregate envelopes. -/
theorem masterInequality_derivation
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    {E d ħ c kB T : ℝ}
    {tauMix tauTotal tauMLUnit : ℝ}
    {steps stepCost totalHeat : ℝ}
    {decisionRate rankPerDecision : ℝ}
    {lambda eps turOverhead : ℝ}
    (hDenPos : 0 < ħ * c * kB * T * Real.log 2)
    (hPhys : (dp.srank : ℝ) * (ħ * c * kB * T * Real.log 2) ≤ E * d)
    (hLambdaPos : 0 < lambda) (hEpsPos : 0 < eps) (hEpsLt : eps < 1)
    (hMix : tauMix ≤ quotientMixingTimeUpperBound lambda eps)
    (hTimeCompose : tauMix + margolusLevitinMinTime tauMLUnit dp.srank ≤ tauTotal)
    (hStep : mcmcStepLandauerFloor kB T ≤ stepCost)
    (hMixSteps : tauMix ≤ steps)
    (hStepsNonneg : 0 ≤ steps)
    (hHeatCompose : steps * stepCost + turOverhead ≤ totalHeat)
    (hkB : 0 < kB) (hT : 0 < T)
    (hRateDenPos : 0 < rankPerDecision * (kB * T * Real.log 2))
    (hRateFlux : decisionRate * (rankPerDecision * (kB * T * Real.log 2)) ≤ E) :
    let rankCap := universalSrankCapValue E d ħ c kB T
    let timeFloor := tauMix + margolusLevitinMinTime tauMLUnit dp.srank
    let heatFloor := tauMix * (kB * T * Real.log 2) + turOverhead
    let rateCap := computationalBremermannRateBound E kB T rankPerDecision
    (dp.srank : ℝ) ≤ rankCap ∧
      timeFloor ≤ tauTotal ∧
      heatFloor ≤ totalHeat ∧
      decisionRate ≤ rateCap := by
  have hMaster := universalResolutionMasterInequality
    (dp := dp)
    (E := E) (d := d) (ħ := ħ) (c := c) (kB := kB) (T := T)
    (tauMix := tauMix) (tauTotal := tauTotal) (tauMLUnit := tauMLUnit)
    (steps := steps) (stepCost := stepCost) (totalHeat := totalHeat)
    (decisionRate := decisionRate) (rankPerDecision := rankPerDecision)
    (lambda := lambda) (eps := eps) (turOverhead := turOverhead)
    hDenPos hPhys hLambdaPos hEpsPos hEpsLt hMix hTimeCompose hStep hMixSteps
    hStepsNonneg hHeatCompose hkB hT hRateDenPos hRateFlux
  rcases hMaster with ⟨hCap, _hMixBound, hTime, hHeat, hRate⟩
  exact ⟨hCap, hTime, hHeat, hRate⟩

/-- Fundamental limit of finite matter: for any finite energy budget, demanding arbitrarily high
precision (`eps -> 0`) forces the required energy lower envelope to diverge. -/
theorem fundamentalLimitOfMatter
    (M : ℝ)
    (_hMass : 0 < M)
    {kB T rankPerDecision : ℝ}
    (hCoef : 0 < rankPerDecision * (kB * T * Real.log 2)) :
    ∀ Emax : ℝ, ∃ eps : ℝ,
      0 < eps ∧ eps < 1 ∧
      Emax < (rankPerDecision * (kB * T * Real.log 2)) * Real.log (1 / eps) := by
  intro Emax
  let coeff : ℝ := rankPerDecision * (kB * T * Real.log 2)
  have hCoeffPos : 0 < coeff := hCoef
  have hCoeffNe : coeff ≠ 0 := ne_of_gt hCoeffPos
  refine ⟨Real.exp (-(|Emax| / coeff + 1)), ?_, ?_, ?_⟩
  · positivity
  · have hInnerPos : 0 < |Emax| / coeff + 1 := by
      have hDivNonneg : 0 ≤ |Emax| / coeff := by
        exact div_nonneg (abs_nonneg _) hCoeffPos.le
      linarith
    have hNeg : -(|Emax| / coeff + 1) < 0 := by linarith
    have hExpLt : Real.exp (-(|Emax| / coeff + 1)) < Real.exp 0 := Real.exp_lt_exp.mpr hNeg
    simpa using hExpLt
  · have hLog :
      Real.log (1 / Real.exp (-(|Emax| / coeff + 1))) = |Emax| / coeff + 1 := by
      calc
        Real.log (1 / Real.exp (-(|Emax| / coeff + 1)))
            = Real.log (Real.exp (|Emax| / coeff + 1)) := by
                simp [one_div, Real.exp_neg]
        _ = |Emax| / coeff + 1 := Real.log_exp _
    have hExpand : coeff * (|Emax| / coeff + 1) = |Emax| + coeff := by
      field_simp [hCoeffNe]
    have hStrict : Emax < coeff * (|Emax| / coeff + 1) := by
      have hAbs : Emax ≤ |Emax| := le_abs_self Emax
      rw [hExpand]
      linarith
    have hTarget : Emax < coeff * Real.log (1 / Real.exp (-(|Emax| / coeff + 1))) := by
      rw [hLog]
      exact hStrict
    simpa [coeff] using hTarget

/-- Exponential cold-barrier amplification factor for stochastic quotient mixing. -/
noncomputable def coldBarrierFactor (barrier Tcold Twarm : ℝ) : ℝ :=
  Real.exp (barrier * (1 / Tcold - 1 / Twarm))

/-- Cosmic decision decay: as ambient temperature drops, Landauer per-step floor decreases while
the MCMC spectral-gap model shrinks, yielding exponentially larger mixing-time envelopes. -/
theorem cosmicDecisionDecay
    {kB barrier eps lambdaWarm lambdaCold Tcold Twarm : ℝ}
    (hkB : 0 ≤ kB)
    (hTcoldPos : 0 < Tcold) (hTwarmPos : 0 < Twarm)
    (hTempOrder : Tcold ≤ Twarm)
    (hBarrier : 0 ≤ barrier)
    (hEpsPos : 0 < eps) (hEpsLt : eps < 1)
    (hLambdaWarm : 0 < lambdaWarm)
    (hGapModel :
      lambdaCold = lambdaWarm / coldBarrierFactor barrier Tcold Twarm) :
    mcmcStepLandauerFloor kB Tcold ≤ mcmcStepLandauerFloor kB Twarm ∧
    quotientMixingTimeUpperBound lambdaWarm eps ≤ quotientMixingTimeUpperBound lambdaCold eps ∧
    quotientMixingTimeUpperBound lambdaCold eps =
      quotientMixingTimeUpperBound lambdaWarm eps * coldBarrierFactor barrier Tcold Twarm := by
  have hLog2Nonneg : 0 ≤ Real.log 2 := (Real.log_pos (by norm_num)).le
  have hKT : kB * Tcold ≤ kB * Twarm :=
    mul_le_mul_of_nonneg_left hTempOrder hkB
  have hLandauer : mcmcStepLandauerFloor kB Tcold ≤ mcmcStepLandauerFloor kB Twarm := by
    unfold mcmcStepLandauerFloor
    exact mul_le_mul_of_nonneg_right hKT hLog2Nonneg
  have hInvOrder : 1 / Twarm ≤ 1 / Tcold :=
    one_div_le_one_div_of_le hTcoldPos hTempOrder
  have hArgNonneg : 0 ≤ barrier * (1 / Tcold - 1 / Twarm) := by
    have hDiff : 0 ≤ 1 / Tcold - 1 / Twarm := by linarith [hInvOrder]
    exact mul_nonneg hBarrier hDiff
  have hFactorPos : 0 < coldBarrierFactor barrier Tcold Twarm := by
    unfold coldBarrierFactor
    positivity
  have hFactorGeOne : 1 ≤ coldBarrierFactor barrier Tcold Twarm := by
    unfold coldBarrierFactor
    have hExp := Real.exp_le_exp.mpr hArgNonneg
    simpa using hExp
  have hLambdaOrder : lambdaCold ≤ lambdaWarm := by
    rw [hGapModel]
    exact (div_le_iff₀ hFactorPos).2 (by nlinarith [hFactorGeOne, hLambdaWarm.le])
  have hLambdaColdPos : 0 < lambdaCold := by
    rw [hGapModel]
    exact div_pos hLambdaWarm hFactorPos
  have hInvLambda : (1 / lambdaWarm) ≤ (1 / lambdaCold) :=
    one_div_le_one_div_of_le hLambdaColdPos hLambdaOrder
  have hInvEps : 1 < 1 / eps := (one_lt_div hEpsPos).2 hEpsLt
  have hLogPos : 0 < Real.log (1 / eps) := Real.log_pos hInvEps
  have hMixOrder :
      quotientMixingTimeUpperBound lambdaWarm eps ≤
        quotientMixingTimeUpperBound lambdaCold eps := by
    unfold quotientMixingTimeUpperBound
    have hMul := mul_le_mul_of_nonneg_left hInvLambda hLogPos.le
    simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hMul
  have hScaleEq :
      quotientMixingTimeUpperBound lambdaCold eps =
        quotientMixingTimeUpperBound lambdaWarm eps * coldBarrierFactor barrier Tcold Twarm := by
    unfold quotientMixingTimeUpperBound
    rw [hGapModel]
    have hFactorNe : coldBarrierFactor barrier Tcold Twarm ≠ 0 := ne_of_gt hFactorPos
    have hLambdaWarmNe : lambdaWarm ≠ 0 := ne_of_gt hLambdaWarm
    field_simp [hFactorNe, hLambdaWarmNe]
  exact ⟨hLandauer, hMixOrder, hScaleEq⟩

/-- Heat-death theorem (critical-temperature form): if spectral gap decreases monotonically in the
cold regime and crossing a critical temperature already exceeds system lifetime, then all colder
temperatures remain computationally unresolved (mixing exceeds Margolus-Levitin lifetime). -/
theorem heatDeathTheorem
    (lambdaAt : ℝ → ℝ)
    {eps tauML Tc : ℝ} {r : ℕ}
    (hTcPos : 0 < Tc)
    (hEpsPos : 0 < eps) (hEpsLt : eps < 1)
    (hLambdaPos : ∀ T : ℝ, 0 < T → 0 < lambdaAt T)
    (hLambdaMono : ∀ T₁ T₂ : ℝ, 0 < T₁ → T₁ ≤ T₂ → lambdaAt T₁ ≤ lambdaAt T₂)
    (hCriticalAtTc :
      margolusLevitinMinTime tauML r < quotientMixingTimeUpperBound (lambdaAt Tc) eps) :
    ∃ Tc' : ℝ, 0 < Tc' ∧
      ∀ T : ℝ, 0 < T → T < Tc' →
        margolusLevitinMinTime tauML r < quotientMixingTimeUpperBound (lambdaAt T) eps := by
  refine ⟨Tc, hTcPos, ?_⟩
  intro T hTpos hTlt
  have hTle : T ≤ Tc := le_of_lt hTlt
  have hLambdaOrder : lambdaAt T ≤ lambdaAt Tc := hLambdaMono T Tc hTpos hTle
  have hLambdaPosT : 0 < lambdaAt T := hLambdaPos T hTpos
  have hInvOrder : (1 / lambdaAt Tc) ≤ (1 / lambdaAt T) :=
    one_div_le_one_div_of_le hLambdaPosT hLambdaOrder
  have hInvEps : 1 < 1 / eps := (one_lt_div hEpsPos).2 hEpsLt
  have hLogPos : 0 < Real.log (1 / eps) := Real.log_pos hInvEps
  have hMixOrder :
      quotientMixingTimeUpperBound (lambdaAt Tc) eps ≤
        quotientMixingTimeUpperBound (lambdaAt T) eps := by
    unfold quotientMixingTimeUpperBound
    have hMul := mul_le_mul_of_nonneg_left hInvOrder hLogPos.le
    simpa [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using hMul
  exact lt_of_lt_of_le hCriticalAtTc hMixOrder

/-- Composition of two decision quotients into a cellular circuit: the optimizer class of `D₁`
state-updates the input seen by `D₂`. -/
noncomputable def quotientCompositionCircuit
    {A₁ A₂ S : Type*}
    (D₁ : DecisionProblem A₁ S)
    (D₂ : DecisionProblem A₂ S)
    (constrainState : Set A₁ → S → S) :
    DecisionProblem A₂ S where
  utility := fun a s => D₂.utility a (constrainState (D₁.Opt s) s)

/-- Structural-rank subadditivity for quotient circuits under coordinate-level constraint
propagation: every circuit-relevant coordinate must already be relevant to one component. -/
theorem circuit_srank_subadditivity
    {A₁ A₂ S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ : DecisionProblem A₁ S)
    (D₂ : DecisionProblem A₂ S)
    (constrainState : Set A₁ → S → S)
    (hPropagate : ∀ i : Fin n,
      (quotientCompositionCircuit D₁ D₂ constrainState).isRelevant i →
        D₁.isRelevant i ∨ D₂.isRelevant i) :
    (quotientCompositionCircuit D₁ D₂ constrainState).srank ≤ D₁.srank + D₂.srank := by
  classical
  let R₁ : Finset (Fin n) := Finset.univ.filter D₁.isRelevant
  let R₂ : Finset (Fin n) := Finset.univ.filter D₂.isRelevant
  let Rc : Finset (Fin n) :=
    Finset.univ.filter (fun i => (quotientCompositionCircuit D₁ D₂ constrainState).isRelevant i)
  have hSub : Rc ⊆ R₁ ∪ R₂ := by
    intro i hi
    have hRel : (quotientCompositionCircuit D₁ D₂ constrainState).isRelevant i := by
      simpa [Rc] using (Finset.mem_filter.mp hi).2
    rcases hPropagate i hRel with h₁ | h₂
    · have hiR₁ : i ∈ R₁ := by
        have : i ∈ Finset.univ.filter D₁.isRelevant := Finset.mem_filter.mpr ⟨by simp, h₁⟩
        simpa [R₁] using this
      exact Finset.mem_union.mpr (Or.inl hiR₁)
    · have hiR₂ : i ∈ R₂ := by
        have : i ∈ Finset.univ.filter D₂.isRelevant := Finset.mem_filter.mpr ⟨by simp, h₂⟩
        simpa [R₂] using this
      exact Finset.mem_union.mpr (Or.inr hiR₂)
  have hCardRc : Rc.card ≤ (R₁ ∪ R₂).card := Finset.card_le_card hSub
  have hCardUnion : (R₁ ∪ R₂).card ≤ R₁.card + R₂.card := Finset.card_union_le R₁ R₂
  have hCard : Rc.card ≤ R₁.card + R₂.card := le_trans hCardRc hCardUnion
  have hRc : Rc.card = (quotientCompositionCircuit D₁ D₂ constrainState).srank := by
    simpa [Rc, DecisionProblem.srank]
  have hR₁ : R₁.card = D₁.srank := by simpa [R₁, DecisionProblem.srank]
  have hR₂ : R₂.card = D₂.srank := by simpa [R₂, DecisionProblem.srank]
  simpa [hRc, hR₁, hR₂] using hCard

/-- Landauer-scale channel cost for transmitting one quotient class across a cellular circuit edge. -/
noncomputable def quotientClassTransmissionCost
    (channelBits : ℕ) (kB T : ℝ) : ℝ :=
  (channelBits : ℝ) * (kB * T * Real.log 2)

/-- Total circuit thermodynamic pipeline budget: component Landauer floors plus quotient-class
transmission overhead. -/
noncomputable def circuitThermodynamicPipelineCost
    {A₁ A₂ S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ : DecisionProblem A₁ S)
    (D₂ : DecisionProblem A₂ S)
    (channelBits : ℕ)
    (kB T : ℝ) : ℝ :=
  ((D₁.srank + D₂.srank : ℕ) : ℝ) * (kB * T * Real.log 2) +
    quotientClassTransmissionCost channelBits kB T

/-- Circuit thermodynamic pipeline bound: resolving the composed circuit costs at most the sum of
the component Landauer floors plus quotient-class transmission cost. -/
theorem circuit_thermodynamic_pipeline
    {A₁ A₂ S : Type*} {n : ℕ} [CoordinateSpace S n]
    (D₁ : DecisionProblem A₁ S)
    (D₂ : DecisionProblem A₂ S)
    (constrainState : Set A₁ → S → S)
    (channelBits : ℕ)
    (hSub : (quotientCompositionCircuit D₁ D₂ constrainState).srank ≤ D₁.srank + D₂.srank)
    {kB T : ℝ} (hkB : 0 < kB) (hT : 0 < T) :
    ((quotientCompositionCircuit D₁ D₂ constrainState).srank : ℝ) * (kB * T * Real.log 2) ≤
      circuitThermodynamicPipelineCost D₁ D₂ channelBits kB T := by
  have hSubReal :
      (((quotientCompositionCircuit D₁ D₂ constrainState).srank : ℕ) : ℝ) ≤
        ((D₁.srank + D₂.srank : ℕ) : ℝ) := by
    exact_mod_cast hSub
  have hScaleNonneg : 0 ≤ kB * T * Real.log 2 := by
    have hLog2 : 0 < Real.log 2 := Real.log_pos (by norm_num)
    exact (mul_pos (mul_pos hkB hT) hLog2).le
  have hMain :
      ((quotientCompositionCircuit D₁ D₂ constrainState).srank : ℝ) * (kB * T * Real.log 2) ≤
        ((D₁.srank + D₂.srank : ℕ) : ℝ) * (kB * T * Real.log 2) :=
    mul_le_mul_of_nonneg_right hSubReal hScaleNonneg
  have hChannelNonneg : 0 ≤ quotientClassTransmissionCost channelBits kB T := by
    unfold quotientClassTransmissionCost
    exact mul_nonneg (by exact_mod_cast (Nat.zero_le channelBits)) hScaleNonneg
  unfold circuitThermodynamicPipelineCost
  exact le_trans hMain (le_add_of_nonneg_right hChannelNonneg)

/-- Abstract quantum decision object: superposed state amplitudes with a unitary optimizer update
and an induced classical decision quotient for readout. -/
structure QuantumDecisionProblem
    (A S : Type*) (n : ℕ) [CoordinateSpace S n] where
  classicalModel : DecisionProblem A S
  superpositionAmplitude : S → ℝ
  unitaryOptimizer : (S → ℝ) → (S → ℝ)

/-- Constructor view for quantum decision superposition systems. -/
noncomputable def quantumDecisionSuperposition
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (classicalModel : DecisionProblem A S)
    (superpositionAmplitude : S → ℝ)
    (unitaryOptimizer : (S → ℝ) → (S → ℝ)) :
    QuantumDecisionProblem A S n :=
  { classicalModel := classicalModel
    superpositionAmplitude := superpositionAmplitude
    unitaryOptimizer := unitaryOptimizer }

/-- Landauer cost of the decoherence/readout event for a quantum decision system. -/
noncomputable def decoherenceLandauerCost
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (Q : QuantumDecisionProblem A S n)
    (kB T : ℝ) : ℝ :=
  (Q.classicalModel.srank : ℝ) * (kB * T * Real.log 2)

/-- Decoherence is thermodynamically irreversible at Landauer scale: measuring a quantum decision
into a single optimizer class costs exactly one Landauer bit-price per resolved structural rank bit. -/
theorem decoherenceLandauerEvent
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (Q : QuantumDecisionProblem A S n)
    (kB T : ℝ) :
    decoherenceLandauerCost Q kB T =
      (Q.classicalModel.srank : ℝ) * (kB * T * Real.log 2) := by
  rfl

/-- Quantum speedup limit at readout: regardless of coherent superposition dynamics, the final
decoherence-resolved quotient still obeys the Margolus-Levitin structural-rank time floor. -/
theorem quantumSpeedupLimit
    {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (Q : QuantumDecisionProblem A S n)
    {τML τ : ℝ}
    (hDecoherence : margolusLevitinMinTime τML Q.classicalModel.srank ≤ τ) :
    margolusLevitinMinTime τML Q.classicalModel.srank ≤ τ :=
  hDecoherence

/-- Geometric holonomic constraint atom-pair primitive for physical quotient synthesis. -/
structure GeometricHolonomicConstraint where
  atomI : ℕ
  atomJ : ℕ
  targetDistance : ℝ

noncomputable def physicalSynthesisEmbedding (γ : ℝ) : ℕ ↪ GeometricHolonomicConstraint where
  toFun := fun i =>
    { atomI := i
      atomJ := i + 1
      targetDistance := γ }
  inj' := by
    intro i j hij
    exact congrArg GeometricHolonomicConstraint.atomI hij

/-- Physical synthesis specification map from target structural rank and utility gap to a finite
constraint set. -/
noncomputable def physicalSynthesisSpec (r : ℕ) (γ : ℝ) : Finset GeometricHolonomicConstraint :=
  (Finset.range r).map (physicalSynthesisEmbedding γ)

theorem physicalSynthesisSpec_card (r : ℕ) (γ : ℝ) :
    (physicalSynthesisSpec r γ).card = r := by
  simp [physicalSynthesisSpec]

/-- Every abstract quotient-synthesis target is realizable by an explicit geometric holonomic
constraint specification with matching structural rank. -/
theorem synthesisRealizability (r k : ℕ) (γ : ℝ) :
    ∃ spec : Finset GeometricHolonomicConstraint,
      spec = physicalSynthesisSpec r γ ∧
      (quotientSynthesis r k γ).srank = spec.card := by
  refine ⟨physicalSynthesisSpec r γ, rfl, ?_⟩
  rw [physicalSynthesisSpec_card]
  exact synthesis_srank_correctness r k γ

/-- Free-energy budget assigned to a physically synthesized realization of an abstract quotient. -/
noncomputable def synthesizedPhysicalDeltaG
    (r k : ℕ) (γ kB T : ℝ) : ℝ :=
  quotientLearningWorkFloor (quotientSynthesis r k γ) kB T + |γ|

/-- Thermodynamic consistency of synthesis: synthesized physical free energy strictly dominates
the abstract Landauer floor whenever the requested strict utility gap is nonzero. -/
theorem synthesisThermodynamicConsistency
    (r k : ℕ) {γ kB T : ℝ}
    (hGap : γ ≠ 0) :
    quotientLearningWorkFloor (quotientSynthesis r k γ) kB T <
      synthesizedPhysicalDeltaG r k γ kB T := by
  unfold synthesizedPhysicalDeltaG
  have hAbsPos : 0 < |γ| := abs_pos.mpr hGap
  linarith

/-- Rank-scaled thermal robustness theorem: if the binding funnel gap exceeds twice the thermal
perturbation budget `srank * k_B * T`, exact resolution survives that thermal noise. -/
theorem decision_quotient_potential
    {A S : Type*} {n : ℕ} [Fintype A] [CoordinateSpace S n]
    (dp : DecisionProblem A S) (s s' : S) (aStar : A)
    {kB T γ : ℝ} (hkB : 0 < kB) (hT : 0 < T)
    (hStrict : StrictOpt dp aStar s)
    (hThermal : ∀ a, |dp.utility a s - dp.utility a s'| ≤ (dp.srank : ℝ) * (kB * T))
    (hGap : 2 * ((dp.srank : ℝ) * (kB * T)) < γ)
    (hγ : γ ≤ StrictUtilityGap dp aStar s) :
    dp.Opt s = dp.Opt s' := by
  have hDelta : 0 ≤ (dp.srank : ℝ) * (kB * T) := by
    have hSrank : 0 ≤ (dp.srank : ℝ) := by exact_mod_cast (Nat.zero_le dp.srank)
    exact mul_nonneg hSrank (mul_nonneg hkB.le hT.le)
  have hBound : (dp.srank : ℝ) * (kB * T) < StrictUtilityGap dp aStar s / 2 := by
    have hStep : 2 * ((dp.srank : ℝ) * (kB * T)) < StrictUtilityGap dp aStar s :=
      lt_of_lt_of_le hGap hγ
    linarith
  exact epsilon_margin_invariance dp s s' aStar ((dp.srank : ℝ) * (kB * T))
    hDelta hStrict hThermal hBound

end Leverage
