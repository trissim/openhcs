import DecisionQuotient.UniverseObjective
import DecisionQuotient.Physics.DecisionTime
import DecisionQuotient.ThermodynamicLift
import DecisionQuotient.BayesFromDQ
import Mathlib.Tactic

/-!
  Paper 4: Decision-Relevant Uncertainty

  ConstraintForcing.lean

  Rigorous layered formalization:
  - logic/time scaffold (consistency + deadline),
  - physical law parameterization (timed allowed-action families),
  - forcing consequences (deadline compels action),
  - thermodynamic consequence (forced decision has positive Landauer lower bound),
  - prior-mass consequence under uncertainty (nondegenerate prior is forced).

  No new axioms are introduced.
-/

namespace DecisionQuotient
namespace Physics
namespace ConstraintForcing

universe u v w z

/-! ## Layer 1: Logic/Time Scaffold -/

/-- Logic/time scaffold: consistency predicate and deadline index. -/
structure LogicTimeScaffold (S : Type u) where
  consistent : S → Prop
  deadline : Nat

/-- Timed family of admissible-action laws, parameterized by `Θ`. -/
structure TimedLawFamily (Θ : Type w) (S : Type u) (A : Type v) where
  allowed : Θ → Nat → S → A → Prop

/-- Time-indexed law utility induced by allowed/blocked partition. -/
noncomputable def timedLawUtility
    (L : TimedLawFamily Θ S A)
    (uAllowed uBlocked : ℝ)
    (θ : Θ) (t : Nat) (a : A) (s : S) : ℝ :=
  by
    classical
    exact if L.allowed θ t s a then uAllowed else uBlocked

/-- Law uniqueness from scaffold alone: one law partition for all parameters. -/
def lawsDeterminedByScaffold (L : TimedLawFamily Θ S A) : Prop :=
  ∃ allowed₀ : Nat → S → A → Prop,
    ∀ θ t s a, L.allowed θ t s a ↔ allowed₀ t s a

/-- Objective uniqueness from scaffold alone: one utility for all parameters. -/
def objectiveDeterminedByScaffold
    (L : TimedLawFamily Θ S A)
    (uAllowed uBlocked : ℝ) : Prop :=
  ∃ U₀ : Nat → A → S → ℝ,
    ∀ θ t a s, timedLawUtility L uAllowed uBlocked θ t a s = U₀ t a s

/-- If parameter values separate the timed allowed/blocked partition at some point,
    law uniqueness from scaffold alone fails. -/
theorem laws_not_determined_of_parameter_separation
    (L : TimedLawFamily Θ S A)
    (hSep : ∃ θ₁ θ₂ t s a, L.allowed θ₁ t s a ∧ ¬ L.allowed θ₂ t s a) :
    ¬ lawsDeterminedByScaffold L := by
  intro hDet
  rcases hDet with ⟨allowed₀, hEq⟩
  rcases hSep with ⟨θ₁, θ₂, t, s, a, hAllow, hNotAllow⟩
  have h1 : allowed₀ t s a := (hEq θ₁ t s a).1 hAllow
  have h2 : L.allowed θ₂ t s a := (hEq θ₂ t s a).2 h1
  exact hNotAllow h2

/-- Therefore logic/time scaffold alone is not sufficient for a unique timed law
    whenever the parameterization is separating. -/
theorem logic_time_not_sufficient_for_unique_law
    (F : LogicTimeScaffold S)
    (L : TimedLawFamily Θ S A)
    (hSep : ∃ θ₁ θ₂ t s a, L.allowed θ₁ t s a ∧ ¬ L.allowed θ₂ t s a) :
    ¬ lawsDeterminedByScaffold L := by
  have _ := F
  exact laws_not_determined_of_parameter_separation L hSep

/-- If a unique timed law partition is fixed, then the timed objective is unique
    for fixed utility levels `uAllowed`, `uBlocked`. -/
theorem laws_determined_implies_objective_determined
    (L : TimedLawFamily Θ S A)
    {uAllowed uBlocked : ℝ}
    (hDet : lawsDeterminedByScaffold L) :
    objectiveDeterminedByScaffold L uAllowed uBlocked := by
  classical
  rcases hDet with ⟨allowed₀, hEq⟩
  refine ⟨fun t a s => if allowed₀ t s a then uAllowed else uBlocked, ?_⟩
  intro θ t a s
  have hiff : L.allowed θ t s a ↔ allowed₀ t s a := hEq θ t s a
  by_cases h : L.allowed θ t s a
  · have h0 : allowed₀ t s a := hiff.mp h
    simp [timedLawUtility, h, h0]
  · have h0 : ¬ allowed₀ t s a := by
      intro h0
      exact h (hiff.mpr h0)
    simp [timedLawUtility, h, h0]

/-- If utility levels are separated and parameter values separate allowed/blocked
    at a timed state-action point, objective uniqueness from scaffold alone fails. -/
theorem objective_not_determined_of_parameter_separation
    (L : TimedLawFamily Θ S A)
    {uAllowed uBlocked : ℝ}
    (hGap : uAllowed ≠ uBlocked)
    (hSep : ∃ θ₁ θ₂ t s a, L.allowed θ₁ t s a ∧ ¬ L.allowed θ₂ t s a) :
    ¬ objectiveDeterminedByScaffold L uAllowed uBlocked := by
  classical
  intro hObj
  rcases hObj with ⟨U₀, hU⟩
  rcases hSep with ⟨θ₁, θ₂, t, s, a, hAllow, hNotAllow⟩
  have hEq : timedLawUtility L uAllowed uBlocked θ₁ t a s =
      timedLawUtility L uAllowed uBlocked θ₂ t a s := by
    calc
      timedLawUtility L uAllowed uBlocked θ₁ t a s = U₀ t a s := hU θ₁ t a s
      _ = timedLawUtility L uAllowed uBlocked θ₂ t a s := (hU θ₂ t a s).symm
  have hLeft : timedLawUtility L uAllowed uBlocked θ₁ t a s = uAllowed := by
    simp [timedLawUtility, hAllow]
  have hRight : timedLawUtility L uAllowed uBlocked θ₂ t a s = uBlocked := by
    simp [timedLawUtility, hNotAllow]
  have : uAllowed = uBlocked := by
    calc
      uAllowed = timedLawUtility L uAllowed uBlocked θ₁ t a s := hLeft.symm
      _ = timedLawUtility L uAllowed uBlocked θ₂ t a s := hEq
      _ = uBlocked := hRight
  exact hGap this

/-! ## Layer 2: Deadline Forcing -/

/-- At deadline, every consistent state has at least one admissible action. -/
def deadlineForcesAction
    (F : LogicTimeScaffold S)
    (L : TimedLawFamily Θ S A)
    (θ : Θ) : Prop :=
  ∀ s, F.consistent s → ∃ a, L.allowed θ F.deadline s a

/-- Bit lower bound for a forced one-shot decision at deadline:
    one irreversible decision bit when an action must be emitted. -/
noncomputable def forcedDecisionBits
    (F : LogicTimeScaffold S)
    (L : TimedLawFamily Θ S A)
    (θ : Θ) (s : S) : ℕ :=
  by
    classical
    exact if ∃ a, L.allowed θ F.deadline s a then 1 else 0

/-- Deadline forcing yields a positive one-shot decision-bit lower bound. -/
theorem forcedDecisionBits_pos_of_deadline
    (F : LogicTimeScaffold S)
    (L : TimedLawFamily Θ S A)
    (θ : Θ) {s : S}
    (hDeadline : deadlineForcesAction F L θ)
    (hCons : F.consistent s) :
    0 < forcedDecisionBits F L θ s := by
  classical
  have hAct : ∃ a, L.allowed θ F.deadline s a := hDeadline s hCons
  simp [forcedDecisionBits, hAct]

/-- If at least one consistent state exists and deadline forcing holds, action
    availability is nonempty (paper notion: `ActionForced`). -/
theorem actionForced_of_deadline
    (F : LogicTimeScaffold S)
    (L : TimedLawFamily Θ S A)
    (θ : Θ)
    (hState : ∃ s, F.consistent s)
    (hDeadline : deadlineForcesAction F L θ) :
    ActionForced A := by
  rcases hState with ⟨s, hs⟩
  rcases hDeadline s hs with ⟨a, _ha⟩
  exact ⟨a⟩

/-- Bridge to nondegenerate-prior theorem: deadline-forced action plus provable
    uncertainty implies nondegenerate prior mass. -/
theorem nondegenerateBelief_of_deadline_and_uncertainty
    (F : LogicTimeScaffold S)
    (L : TimedLawFamily Θ S A)
    (θ : Θ)
    (hState : ∃ s, F.consistent s)
    (hDeadline : deadlineForcesAction F L θ)
    {H : Type z} [Fintype H]
    (prior : ProbDist H)
    (hUnc : UncertaintyForced prior) :
    NondegenerateBelief prior := by
  have hAction : ActionForced A := actionForced_of_deadline F L θ hState hDeadline
  rcases forced_action_under_uncertainty (A := A) hAction prior hUnc with ⟨_a, hBelief⟩
  exact hBelief

/-! ## Layer 3: Thermodynamic Consequence -/

/-- Forced deadline decision has positive Landauer lower-bound energy cost
    under positive joules-per-bit conversion constant. -/
theorem forced_decision_implies_positive_landauer_cost
    (F : LogicTimeScaffold S)
    (L : TimedLawFamily Θ S A)
    (θ : Θ) {s : S}
    (M : ThermodynamicLift.ThermoModel)
    (hJ : 0 < M.joulesPerBit)
    (hDeadline : deadlineForcesAction F L θ)
    (hCons : F.consistent s) :
    0 < ThermodynamicLift.energyLowerBound M (forcedDecisionBits F L θ s) := by
  apply ThermodynamicLift.energy_lower_mandatory M hJ
  exact forcedDecisionBits_pos_of_deadline F L θ hDeadline hCons

/-- Same forced-decision lower bound in Neukart--Vinokur `λ·dC` form. -/
theorem forced_decision_implies_positive_nv_work
    (F : LogicTimeScaffold S)
    (L : TimedLawFamily Θ S A)
    (θ : Θ) {s : S}
    (M : ThermodynamicLift.ThermoModel)
    (hLam : 0 < ThermodynamicLift.nvLambda M)
    (hDeadline : deadlineForcesAction F L θ)
    (hCons : F.consistent s) :
    0 < ThermodynamicLift.nvLambda M * forcedDecisionBits F L θ s := by
  have hPos :
      0 < ThermodynamicLift.energyLowerBound M (forcedDecisionBits F L θ s) :=
    forced_decision_implies_positive_landauer_cost F L θ M hLam hDeadline hCons
  simpa [ThermodynamicLift.energyLowerBound, ThermodynamicLift.nvLambda] using hPos

end ConstraintForcing
end Physics
end DecisionQuotient
