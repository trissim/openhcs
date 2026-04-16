import Leverage.BridgeToDQ
import DecisionQuotient.Summary
import DecisionQuotient.Reduction_AllCoords
import DecisionQuotient.Tractability.MolecularSrank
import DecisionQuotient.Tractability.SampledDockingGap
import DecisionQuotient.Tractability.SampledDockingCutoff

namespace Leverage

open Classical DecisionQuotient
open DecisionQuotient.Tractability
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

end Leverage
