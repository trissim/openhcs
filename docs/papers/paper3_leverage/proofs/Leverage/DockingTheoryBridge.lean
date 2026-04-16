import Leverage.BridgeToDQ
import DecisionQuotient.Summary
import DecisionQuotient.Reduction_AllCoords
import DecisionQuotient.Tractability.MolecularSrank
import DecisionQuotient.Tractability.DiscretizedState
import DecisionQuotient.Tractability.SampledDockingGap
import DecisionQuotient.Tractability.SampledDockingCutoff

namespace Leverage

open Classical DecisionQuotient
open DecisionQuotient.Tractability
open DecisionQuotient.Tractability.DiscretizedState
open DecisionQuotient.Tractability.MolecularSrank
open DecisionQuotient.Tractability.SampledDocking

/-- Local paper3 exposure of the general coNP-hardness reduction core for exact sufficiency. -/
theorem exactSufficiency_conp_core {n : ℕ} (φ : Formula n) :
    (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) ↔ φ.isTautology :=
  DecisionQuotient.Summary.conp_completeness φ

/-- Local paper3 exposure of the maximal-structural-rank hard family. -/
theorem exactSufficiency_hardFamily_srank_eq_n {n : ℕ} (hn : 0 < n)
    (φ : Formula n) (hnt : ¬ φ.isTautology) :
    (reductionProblemMany φ).srank = n :=
  DecisionQuotient.hard_family_srank_eq_n hn φ hnt

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
