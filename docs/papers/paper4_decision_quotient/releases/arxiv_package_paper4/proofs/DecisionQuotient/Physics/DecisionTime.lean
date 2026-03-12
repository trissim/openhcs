import Mathlib.Data.Nat.Basic
import Mathlib.Data.List.Basic
import Mathlib.Tactic
import DecisionQuotient.Physics.ClaimTransport

/-!
# Decision-Time Atomicity

## Dependency Graph
- **Nontrivial source:** ClaimTransport.substrate_consistency (the actual proof that
  substrate dynamics preserves interface behavior)
- **This module:** Defines the interface-level time model. The "time = decision atom"
  claim is definitional at the interface level; the nontrivial proof is that
  substrate models realize this interface (ClaimTransport)

## Role
This module provides the definitional framework: tick increments time by 1.
The connection to physical substrates is in ClaimTransport.

## Triviality Level
TRIVIAL — interface-level discrete-time model and lemmas. Closest nontrivial
proof: `ClaimTransport.substrate_consistency` (DecisionQuotient.Physics.ClaimTransport).
-/

namespace DecisionQuotient
namespace Physics
namespace DecisionTime

universe u v w

/-- Interface state paired with a discrete time index. -/
structure TimedState (S : Type u) where
  state : S
  time : Nat
  deriving Repr

/-- Deterministic decision process at the interface level. -/
structure DecisionProcess (S : Type u) (A : Type v) where
  decide : S → A
  transition : S → A → S

/-- One interface tick: emit one decision and advance time by one. -/
def tick (P : DecisionProcess S A) (x : TimedState S) : TimedState S :=
  let a := P.decide x.state
  { state := P.transition x.state a
    time := x.time + 1 }

/-- One decision event from `x` to `y` under process `P`. -/
def DecisionEvent (P : DecisionProcess S A) (x y : TimedState S) : Prop :=
  ∃ a : A,
    a = P.decide x.state ∧
    y = { state := P.transition x.state a, time := x.time + 1 }

/-- One unit time step between two timed states. -/
def TimeUnitStep (x y : TimedState S) : Prop :=
  y.time = x.time + 1

/-- Time is discrete in the model: every time value is a natural number witness. -/
theorem time_is_discrete (x : TimedState S) :
    ∃ t : Nat, x.time = t := by
  exact ⟨x.time, rfl⟩

/-- Discrete-time coordinate is decidable pointwise by numeric equality. -/
theorem time_coordinate_falsifiable (x : TimedState S) (t : Nat) :
    x.time = t ∨ x.time ≠ t := by
  exact em (x.time = t)

/-- Every process tick increases time by exactly one. -/
theorem tick_increments_time (P : DecisionProcess S A) (x : TimedState S) :
    (tick P x).time = x.time + 1 := by
  simp [tick]

/-- Every process tick has an explicit decision witness. -/
theorem tick_decision_witness (P : DecisionProcess S A) (x : TimedState S) :
    ∃ a : A,
      a = P.decide x.state ∧
      (tick P x) = { state := P.transition x.state a, time := x.time + 1 } := by
  refine ⟨P.decide x.state, rfl, ?_⟩
  simp [tick]

/-- A process tick is always a valid decision event. -/
theorem tick_is_decision_event (P : DecisionProcess S A) (x : TimedState S) :
    DecisionEvent P x (tick P x) := by
  rcases tick_decision_witness (P := P) x with ⟨a, ha, hcfg⟩
  exact ⟨a, ha, hcfg⟩

/-- Decision events force exactly one unit of time. -/
theorem decision_event_implies_time_unit
    (P : DecisionProcess S A) {x y : TimedState S}
    (h : DecisionEvent P x y) :
    TimeUnitStep x y := by
  rcases h with ⟨a, ha, hy⟩
  subst ha
  simp [TimeUnitStep] at hy ⊢
  simpa [hy]

/-- Paper-facing theorem name: a decision taking place is a unit of time. -/
theorem decision_taking_place_is_unit_of_time
    (P : DecisionProcess S A) {x y : TimedState S}
    (h : DecisionEvent P x y) :
    TimeUnitStep x y :=
  decision_event_implies_time_unit (P := P) h

/-- Event relation and process tick coincide for deterministic interface dynamics. -/
theorem decision_event_iff_eq_tick
    (P : DecisionProcess S A) (x y : TimedState S) :
    DecisionEvent P x y ↔ y = tick P x := by
  constructor
  · intro h
    rcases h with ⟨a, ha, hy⟩
    subst ha
    simpa [tick] using hy
  · intro hy
    subst hy
    exact tick_is_decision_event (P := P) x

/-- Run `n` ticks from an initial timed state. -/
def run (P : DecisionProcess S A) : Nat → TimedState S → TimedState S
  | 0, x => x
  | n + 1, x => run P n (tick P x)

/-- The time coordinate after `n` ticks is exactly `start + n`. -/
theorem run_time_exact (P : DecisionProcess S A) (n : Nat) (x : TimedState S) :
    (run P n x).time = x.time + n := by
  induction n generalizing x with
  | zero =>
      simp [run]
  | succ n ih =>
      calc
        (run P (n + 1) x).time
            = (run P n (tick P x)).time := by simp [run]
        _ = (tick P x).time + n := ih (tick P x)
        _ = (x.time + 1) + n := by simp [tick]
        _ = x.time + (n + 1) := by omega

/-- The elapsed time in a run is exactly the number of ticks. -/
theorem run_elapsed_time_eq_ticks (P : DecisionProcess S A) (n : Nat) (x : TimedState S) :
    (run P n x).time - x.time = n := by
  rw [run_time_exact]
  omega

/-- Trace of decisions emitted during `n` ticks. -/
def decisionTrace (P : DecisionProcess S A) : Nat → TimedState S → List A
  | 0, _ => []
  | n + 1, x => P.decide x.state :: decisionTrace P n (tick P x)

/-- Exactly one decision is emitted per tick. -/
theorem decisionTrace_length_eq_ticks (P : DecisionProcess S A) (n : Nat) (x : TimedState S) :
    (decisionTrace P n x).length = n := by
  induction n generalizing x with
  | zero =>
      simp [decisionTrace]
  | succ n ih =>
      simpa [decisionTrace] using congrArg Nat.succ (ih (tick P x))

/-- Decision-count equals elapsed time in the run semantics. -/
theorem decision_count_equals_elapsed_time
    (P : DecisionProcess S A) (n : Nat) (x : TimedState S) :
    (decisionTrace P n x).length = (run P n x).time - x.time := by
  rw [decisionTrace_length_eq_ticks, run_elapsed_time_eq_ticks]

/-! ## Substrate Realization (classical/quantum/chemical kernels) -/

/-- Substrate tag used for regime-invariant statements. -/
inductive SubstrateKind
  | classical
  | quantum
  | chemical
  | hybrid
  deriving DecidableEq, Repr

/-- A substrate-level realization whose observed interface obeys decision ticks. -/
structure SubstrateModel (Micro : Type w) (S : Type u) (A : Type v) where
  kind : SubstrateKind
  microStep : Micro → Micro
  observe : Micro → S
  interface : DecisionProcess S A
  consistency :
    ∀ μ : Micro,
      observe (microStep μ) =
        interface.transition (observe μ) (interface.decide (observe μ))

/-- Any one-step substrate evolution yields a decision event at the interface. -/
theorem substrate_step_realizes_decision_event
    (M : SubstrateModel Micro S A) (μ : Micro) (t : Nat) :
    DecisionEvent M.interface
      { state := M.observe μ, time := t }
      { state := M.observe (M.microStep μ), time := t + 1 } := by
  refine ⟨M.interface.decide (M.observe μ), rfl, ?_⟩
  change
    ({ state := M.observe (M.microStep μ), time := t + 1 } : TimedState S) =
      ({ state := M.interface.transition (M.observe μ) (M.interface.decide (M.observe μ)),
         time := t + 1 } : TimedState S)
  simp [M.consistency]

/-- Therefore every substrate step advances interface time by one unit. -/
theorem substrate_step_is_time_unit
    (M : SubstrateModel Micro S A) (μ : Micro) (t : Nat) :
    TimeUnitStep
      { state := M.observe μ, time := t }
      { state := M.observe (M.microStep μ), time := t + 1 } := by
  exact decision_taking_place_is_unit_of_time
    (P := M.interface)
    (h := substrate_step_realizes_decision_event (M := M) μ t)

/-- Time-unit law is independent of substrate tag. -/
theorem time_unit_law_substrate_invariant
    (M₁ : SubstrateModel Micro₁ S A) (M₂ : SubstrateModel Micro₂ S A)
    (x : TimedState S) :
    (tick M₁.interface x).time = (tick M₂.interface x).time := by
  simp [tick]

end DecisionTime
end Physics
end DecisionQuotient
