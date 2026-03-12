import DecisionQuotient.BayesFromDQ
import DecisionQuotient.BayesFoundations
import DecisionQuotient.ThermodynamicLift
import DecisionQuotient.Physics.Instantiation
import DecisionQuotient.Physics.MolecularIntegrity
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

/-!
  Paper 4: Decision-Relevant Uncertainty

  GraphNontriviality.lean

  Layered formalization of observer-conditioned nontriviality on logic graphs:
  1) path/cycle structure on directed graphs;
  2) observer-conditioned path probability, surprisal, novelty score;
  3) bridge to forced nondegenerate prior mass and Bayes-update existence;
  4) DQ/entropy normalization bridge;
  5) thermodynamic and atomic/material lower-bound lift.

  No new axioms are introduced in this module.
-/

namespace DecisionQuotient
namespace LogicGraph

universe u v w

/-! ## Layer 1: Graph Paths and Cycles -/

/-- Directed graph over vertex type `V`. -/
structure Graph (V : Type u) where
  edge : V → V → Prop

/-- Path validity: every consecutive pair follows a graph edge. -/
def validPath (G : Graph V) : List V → Prop
  | [] => True
  | [_] => True
  | x :: y :: xs => G.edge x y ∧ validPath G (y :: xs)

/-- Cycle witness: valid path, length at least two, same start and end node. -/
def isCycle (G : Graph V) (p : List V) : Prop :=
  validPath G p ∧ 2 ≤ p.length ∧ p.head? = p.getLast?

/-- Bit witness for cycle certification in a path trace.
    We count one bit per path position when a cycle witness is present. -/
noncomputable def cycleWitnessBits (G : Graph V) (p : List V) : ℕ :=
  by
    classical
    exact if isCycle G p then p.length else 0

/-- Any certified cycle witness has strictly positive bit witness size. -/
theorem cycleWitnessBits_pos
    (G : Graph V) (p : List V) (hCycle : isCycle G p) :
    0 < cycleWitnessBits G p := by
  classical
  have hLenPos : 0 < p.length := lt_of_lt_of_le (by decide : 0 < 2) hCycle.2.1
  have hCycle' : isCycle G p := hCycle
  simp [cycleWitnessBits, hCycle', hLenPos]

/-! ## Layer 2: Observer Probability, Surprisal, Novelty -/

section Observer

variable {H : Type v} [Fintype H]
variable {V : Type u} [DecidableEq V]

/-- Observer path probability under hypothesis prior and decode map. -/
noncomputable def pathProb
    (decode : H → List V) (prior : ProbDist H) (p : List V) : ℝ :=
  Finset.univ.sum (fun h => if decode h = p then prior.prob h else 0)

/-- Path probability is nonnegative. -/
theorem pathProb_nonneg
    (decode : H → List V) (prior : ProbDist H) (p : List V) :
    0 ≤ pathProb decode prior p := by
  unfold pathProb
  refine Finset.sum_nonneg ?_
  intro h _hh
  by_cases hEq : decode h = p
  · simp [hEq, prior.nonneg h]
  · simp [hEq]

/-- Observer prior mass on any hypothesis is at most one. -/
theorem hypothesisProb_le_one
    (prior : ProbDist H) (h : H) :
    prior.prob h ≤ 1 := by
  have hLeSum : prior.prob h ≤ Finset.univ.sum prior.prob := by
    exact Finset.single_le_sum (fun x _hx => prior.nonneg x) (by simp)
  simpa [prior.sum_one] using hLeSum

/-- Path probability is at most one. -/
theorem pathProb_le_one
    (decode : H → List V) (prior : ProbDist H) (p : List V) :
    pathProb decode prior p ≤ 1 := by
  have hLe :
      pathProb decode prior p ≤ Finset.univ.sum prior.prob := by
    unfold pathProb
    refine Finset.sum_le_sum ?_
    intro h _hh
    by_cases hEq : decode h = p
    · simp [hEq]
    · simp [hEq, prior.nonneg h]
  simpa [prior.sum_one] using hLe

/-- Surprisal of a positive mass value. -/
noncomputable def surprisal (x : ℝ) : ℝ := -Real.log x

/-- Path surprisal under observer prior and decode map. -/
noncomputable def pathSurprisal
    (decode : H → List V) (prior : ProbDist H) (p : List V) : ℝ :=
  surprisal (pathProb decode prior p)

/-- Lower probability means higher surprisal (antitone). -/
theorem surprisal_antitone
    {x y : ℝ} (hx : 0 < x) (hxy : x ≤ y) :
    surprisal y ≤ surprisal x := by
  unfold surprisal
  have hlog : Real.log x ≤ Real.log y := Real.log_le_log hx hxy
  linarith

/-- Positive-probability path surprisal is nonnegative. -/
theorem pathSurprisal_nonneg_of_positive_mass
    (decode : H → List V) (prior : ProbDist H) (p : List V)
    (hPos : 0 < pathProb decode prior p) :
    0 ≤ pathSurprisal decode prior p := by
  unfold pathSurprisal surprisal
  have hLeOne : pathProb decode prior p ≤ 1 := pathProb_le_one decode prior p
  have hLogNonpos : Real.log (pathProb decode prior p) ≤ 0 :=
    Real.log_nonpos (le_of_lt hPos) hLeOne
  linarith

/-- Observer-known path set. -/
abbrev KnownPaths (V : Type u) := Finset (List V)

/-- Binary novelty distance to known-path support. -/
def noveltyDistance (known : KnownPaths V) (p : List V) : ℕ :=
  if p ∈ known then 0 else 1

/-- Known path has zero novelty distance. -/
theorem noveltyDistance_eq_zero_of_mem
    (known : KnownPaths V) (p : List V) (hMem : p ∈ known) :
    noveltyDistance known p = 0 := by
  simp [noveltyDistance, hMem]

/-- Unknown path has unit novelty distance. -/
theorem noveltyDistance_eq_one_of_not_mem
    (known : KnownPaths V) (p : List V) (hNotMem : p ∉ known) :
    noveltyDistance known p = 1 := by
  simp [noveltyDistance, hNotMem]

/-- Observer-level nontriviality score:
    surprisal + novelty distance. -/
noncomputable def nontrivialityScore
    (decode : H → List V) (prior : ProbDist H)
    (known : KnownPaths V) (p : List V) : ℝ :=
  pathSurprisal decode prior p + noveltyDistance known p

/-- Unknown-path score decomposes as surprisal + 1. -/
theorem nontrivialityScore_unknown
    (decode : H → List V) (prior : ProbDist H)
    (known : KnownPaths V) (p : List V)
    (hNotMem : p ∉ known) :
    nontrivialityScore decode prior known p = pathSurprisal decode prior p + 1 := by
  simp [nontrivialityScore, noveltyDistance, hNotMem]

/-- Known-path score decomposes as surprisal + 0. -/
theorem nontrivialityScore_known
    (decode : H → List V) (prior : ProbDist H)
    (known : KnownPaths V) (p : List V)
    (hMem : p ∈ known) :
    nontrivialityScore decode prior known p = pathSurprisal decode prior p := by
  simp [nontrivialityScore, noveltyDistance, hMem]

/-- Observer entropy over finite hypothesis support (nats). -/
noncomputable def observerEntropy (prior : ProbDist H) : ℝ :=
  Finset.univ.sum
    (fun h => if prior.prob h > 0 then -(prior.prob h * Real.log (prior.prob h)) else 0)

/-- Observer entropy is nonnegative. -/
theorem observerEntropy_nonneg (prior : ProbDist H) :
    0 ≤ observerEntropy prior := by
  unfold observerEntropy
  refine Finset.sum_nonneg ?_
  intro h _hh
  by_cases hPos : prior.prob h > 0
  · have hLeOne : prior.prob h ≤ 1 := hypothesisProb_le_one prior h
    have hLogNonpos : Real.log (prior.prob h) ≤ 0 :=
      Real.log_nonpos (le_of_lt hPos) hLeOne
    have hMulNonpos : prior.prob h * Real.log (prior.prob h) ≤ 0 :=
      mul_nonpos_of_nonneg_of_nonpos (prior.nonneg h) hLogNonpos
    have hTerm : 0 ≤ -(prior.prob h * Real.log (prior.prob h)) := by linarith
    simpa [hPos] using hTerm
  · simp [hPos]

/-- DQ-from-entropy normalization reusing the BayesFoundations DQ definition. -/
noncomputable def dqFromEntropy (priorH posteriorH : ℝ) : ℝ :=
  Foundations.dqBayes priorH posteriorH

/-- Entropy-contraction assumptions place DQ-from-entropy in [0,1]. -/
theorem dqFromEntropy_in_unit_interval
    (priorH posteriorH : ℝ)
    (hPriorPos : priorH > 0)
    (hPosteriorNonneg : 0 ≤ posteriorH)
    (hContraction : posteriorH ≤ priorH) :
    0 ≤ dqFromEntropy priorH posteriorH ∧ dqFromEntropy priorH posteriorH ≤ 1 := by
  exact Foundations.dq_in_unit_interval priorH posteriorH hPriorPos hPosteriorNonneg hContraction

end Observer

/-! ## Layer 3: Bayes / Nondegenerate-Prior Bridge -/

section PriorBridge

variable {H : Type v} [Fintype H]
variable {E : Type u}
variable {V : Type w}

/-- Forced action + provable uncertainty forces nondegenerate prior mass
    (independent of Bayesian uniqueness). -/
theorem path_belief_forced_under_uncertainty
    (A : Type*) (hAction : ActionForced A)
    (prior : ProbDist H) (hUnc : UncertaintyForced prior) :
    ∃ _ : A, NondegenerateBelief prior := by
  exact forced_action_under_uncertainty hAction prior hUnc

/-- Under nondegenerate prior mass and strictly informative likelihood support,
    Bayes update exists for observer path hypotheses. -/
theorem bayes_update_exists_for_observer_paths
    (decode : H → List V)
    (likelihood : H → E → ℝ)
    (hLikeNonneg : ∀ h e, 0 ≤ likelihood h e)
    (hLikePosWitness : ∃ e0 : E, ∀ h, 0 < likelihood h e0)
    (prior : ProbDist H)
    (hBelief : NondegenerateBelief prior) :
    ∃ e : E, ∃ posterior : ProbDist H,
      ∀ h, posterior.prob h =
        prior.prob h * likelihood h e /
          Finset.univ.sum (fun h' => prior.prob h' * likelihood h' e) := by
  have _ := decode
  exact bayes_update_exists_of_nondegenerateBelief
    likelihood hLikeNonneg hLikePosWitness prior hBelief

end PriorBridge

/-! ## Layer 4: Physical Lift (Landauer + Atomic/Material) -/

section PhysicalBridge

variable {V : Type u}

/-- Cycle witness implies strictly positive Landauer lower-bound energy
    under positive joules-per-bit conversion. -/
theorem cycle_witness_implies_positive_landauer
    (G : Graph V) (p : List V)
    (M : ThermodynamicLift.ThermoModel)
    (hJ : 0 < M.joulesPerBit)
    (hCycle : isCycle G p) :
    0 < ThermodynamicLift.energyLowerBound M (cycleWitnessBits G p) := by
  have hBits : 0 < cycleWitnessBits G p := cycleWitnessBits_pos G p hCycle
  exact ThermodynamicLift.energy_lower_mandatory M hJ hBits

/-- Same cycle witness lower bound in Neukart--Vinokur `λ·dC` form. -/
theorem cycle_witness_implies_positive_nv_work
    (G : Graph V) (p : List V)
    (M : ThermodynamicLift.ThermoModel)
    (hLam : 0 < ThermodynamicLift.nvLambda M)
    (hCycle : isCycle G p) :
    0 < ThermodynamicLift.nvLambda M * cycleWitnessBits G p := by
  have hPos :
      0 < ThermodynamicLift.energyLowerBound M (cycleWitnessBits G p) :=
    cycle_witness_implies_positive_landauer G p M hLam hCycle
  simpa [ThermodynamicLift.energyLowerBound, ThermodynamicLift.nvLambda] using hPos

/-- Atomic/material anchor: canonical DNA-like configuration has positive erasure bits. -/
theorem dna_erasure_bits_pos :
    0 < Physics.MolecularIntegrity.erasureCost Physics.MolecularIntegrity.dnaConfiguration := by
  simp [Physics.MolecularIntegrity.erasureCost, Physics.MolecularIntegrity.dnaConfiguration]

/-- Atomic/material bridge: positive erasure bits imply positive Landauer lower bound
    under positive joules-per-bit conversion. -/
theorem dna_erasure_implies_positive_landauer
    (M : ThermodynamicLift.ThermoModel)
    (hJ : 0 < M.joulesPerBit) :
    0 < ThermodynamicLift.energyLowerBound M
      (Physics.MolecularIntegrity.erasureCost Physics.MolecularIntegrity.dnaConfiguration) := by
  exact ThermodynamicLift.energy_lower_mandatory M hJ dna_erasure_bits_pos

/-- Atomic/material persistence witness at room-temperature competence model. -/
theorem dna_room_temp_environmental_stability :
    Physics.MolecularIntegrity.isEnvironmentallyStable
      Physics.MolecularIntegrity.roomTempEnvironment
      Physics.MolecularIntegrity.dnaConfiguration := by
  exact Physics.MolecularIntegrity.dna_persists_room_temp

end PhysicalBridge

end LogicGraph
end DecisionQuotient
