import Leverage.BridgeToDQ
import DecisionQuotient.AbstractionCollapse
import DecisionQuotient.Summary
import DecisionQuotient.Reduction_AllCoords
import DecisionQuotient.WitnessCheckingDuality
import DecisionQuotient.Statistics.FisherInformation
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.MolecularSrank
import DecisionQuotient.Tractability.DiscretizedState
import DecisionQuotient.Tractability.TopKPreservation
import DecisionQuotient.Tractability.NearTieBand
import DecisionQuotient.Tractability.LJApproximation
import DecisionQuotient.Tractability.CoulombApproximation
import DecisionQuotient.Tractability.SampledDockingGap
import DecisionQuotient.Tractability.SampledDockingCutoff

namespace Leverage

open Classical DecisionQuotient
open DecisionQuotient.Tractability
open DecisionQuotient.Tractability.DiscretizedState
open DecisionQuotient.Tractability.MolecularSrank
open DecisionQuotient.Tractability.SampledDocking
open DecisionQuotient.Tractability.CoarseApproximation

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

end Leverage
