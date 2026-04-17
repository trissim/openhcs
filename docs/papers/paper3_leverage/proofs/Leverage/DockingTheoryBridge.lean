import Leverage.BridgeToDQ
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Data.Finset.Max
import Mathlib.Data.Nat.Bitwise
import Mathlib.Tactic
import DecisionQuotient.AbstractionCollapse
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
import DecisionQuotient.Tractability.TopKPreservation
import DecisionQuotient.Tractability.NearTieBand
import DecisionQuotient.Tractability.LJApproximation
import DecisionQuotient.Tractability.CoulombApproximation
import DecisionQuotient.Tractability.EwaldSummation
import DecisionQuotient.Computation.PhaseSpaceGeometry
import DecisionQuotient.Computation.LennardJonesDeriv
import DecisionQuotient.Tractability.SampledDockingGap
import DecisionQuotient.Tractability.SampledDockingCutoff

namespace Leverage

universe u v

open Classical DecisionQuotient
open DecisionQuotient.Tractability
open DecisionQuotient.Tractability.DiscretizedState
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

end Leverage
