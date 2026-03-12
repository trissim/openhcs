/-
  EditDynamics: Proving inconsistency arises from edits

  This addresses the critique: "So what if inconsistent states exist?
  Maybe no edit sequence produces them."

  We prove: If DOF > 1 and edits don't propagate, some edit trace
  causes inconsistency starting from any consistent configuration.
-/

import Ssot.SSOTGrounded

namespace EditDynamics

open SSOTGrounded

/-- An edit operation modifies a single location -/
structure Edit where
  targetId : Nat
  newValue : Nat
  deriving DecidableEq, Repr

/-- A trace is a sequence of edits -/
def EditTrace := List Edit

/-- Apply an edit: update location with matching id to new value -/
def apply_edit (locs : List EncodingLocation) (e : Edit) : List EncodingLocation :=
  locs.map fun loc => if loc.id = e.targetId then ⟨loc.id, e.newValue⟩ else loc

/-- Apply a trace to locations -/
def apply_trace (locs : List EncodingLocation) (trace : EditTrace) : List EncodingLocation :=
  trace.foldl apply_edit locs

/-- All locations have the same value -/
def all_same_value (locs : List EncodingLocation) : Prop :=
  ∀ l1 l2, l1 ∈ locs → l2 ∈ locs → l1.value = l2.value

/-- Some locations have different values -/
def has_different_values (locs : List EncodingLocation) : Prop :=
  ∃ l1 l2, l1 ∈ locs ∧ l2 ∈ locs ∧ l1.value ≠ l2.value

/-- KEY THEOREM: Two distinct-ID locations, one edit causes inconsistency -/
theorem edit_causes_inconsistency_two_locs :
    ∀ (id1 id2 val : Nat),
      id1 ≠ id2 →
      has_different_values (apply_edit [⟨id1, val⟩, ⟨id2, val⟩] ⟨id1, val + 1⟩) := by
  intro id1 id2 val hne
  unfold has_different_values apply_edit
  simp only [List.map_cons, List.map_nil]
  use ⟨id1, val + 1⟩, ⟨id2, val⟩
  refine ⟨?_, ?_, ?_⟩
  · simp
  · -- loc2 unchanged because id2 ≠ id1
    simp only [List.mem_cons]
    right
    left
    split_ifs with h
    · exact absurd h.symm hne
    · rfl
  · -- val + 1 ≠ val
    simp only [ne_eq]
    exact Nat.succ_ne_self val

/-- Constructive: given any two distinct locations, we can break consistency -/
theorem two_locations_breakable :
    ∀ (id1 id2 val : Nat),
      id1 ≠ id2 →
      ∃ trace : EditTrace,
        has_different_values (apply_trace [⟨id1, val⟩, ⟨id2, val⟩] trace) := by
  intro id1 id2 val hne
  use [⟨id1, val + 1⟩]
  unfold apply_trace
  simp only [List.foldl_cons, List.foldl_nil]
  exact edit_causes_inconsistency_two_locs id1 id2 val hne

/-- Single location: any edit preserves consistency -/
theorem single_location_consistent :
    ∀ (loc : EncodingLocation) (e : Edit),
      all_same_value (apply_edit [loc] e) := by
  intro loc e
  unfold all_same_value apply_edit
  simp only [List.map_cons, List.map_nil]
  intro l1 l2 h1 h2
  simp only [List.mem_singleton] at h1 h2
  rw [h1, h2]

/-- Single location: any trace preserves consistency -/
theorem single_location_trace_consistent :
    ∀ (loc : EncodingLocation) (trace : EditTrace),
      all_same_value (apply_trace [loc] trace) := by
  intro loc trace
  induction trace generalizing loc with
  | nil =>
    unfold apply_trace all_same_value
    simp
  | cons e rest ih =>
    unfold apply_trace
    simp only [List.foldl_cons]
    -- apply_edit [loc] e is a singleton
    unfold apply_edit
    simp only [List.map_cons, List.map_nil]
    let loc' := if loc.id = e.targetId then ⟨loc.id, e.newValue⟩ else loc
    exact ih loc'

/-- The asymmetry: DOF=1 is immune, DOF≥2 is vulnerable -/
theorem ssot_asymmetry :
    -- DOF = 1: immune
    (∀ loc trace, all_same_value (apply_trace [loc] trace)) ∧
    -- DOF ≥ 2: vulnerable (if IDs distinct)
    (∀ id1 id2 val, id1 ≠ id2 →
      ∃ trace, has_different_values (apply_trace [⟨id1, val⟩, ⟨id2, val⟩] trace)) := by
  constructor
  · exact single_location_trace_consistent
  · exact two_locations_breakable

/-- Non-propagating edit affects only target -/
theorem edit_preserves_other_values :
    ∀ (locs : List EncodingLocation) (e : Edit) (loc : EncodingLocation),
      loc ∈ locs →
      loc.id ≠ e.targetId →
      loc ∈ apply_edit locs e := by
  intro locs e loc hmem hne
  unfold apply_edit
  simp only [List.mem_map]
  use loc
  constructor
  · exact hmem
  · simp [hne]

/-- Edit changes target value -/
theorem edit_changes_target :
    ∀ (locs : List EncodingLocation) (e : Edit) (loc : EncodingLocation),
      loc ∈ locs →
      loc.id = e.targetId →
      ⟨loc.id, e.newValue⟩ ∈ apply_edit locs e := by
  intro locs e loc hmem heq
  unfold apply_edit
  simp only [List.mem_map]
  use loc
  constructor
  · exact hmem
  · simp [heq]

end EditDynamics

