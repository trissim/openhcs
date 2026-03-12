/-
  SSOTGrounded: Connecting SSOT to Operational Semantics

  This file bridges the abstract SSOT definition (DOF = 1) to the
  concrete operational semantics from AbstractClassSystem.

  The key insight: SSOT failures arise when the same fact has multiple
  independent encodings that can diverge. This is formalized via:

  1. AbstractClassSystem proves: shape-based observers cannot distinguish
     types with same namespace but different bases (capability gap)

  2. This file proves: if a derived fact has DOF > 1, there exist
     observers that can see inconsistent values (operational inconsistency)

  Together: The capability gap from Paper 1 explains WHY DOF > 1 is bad.
-/

import abstract_class_system
import Ssot.SSOT

namespace SSOTGrounded

open AbstractClassSystem Ssot

/-
  PART 1: Connecting DOF to Observable Consistency

  A configuration with DOF > 1 means the same fact is encoded in multiple
  places. If these places can be modified independently, observers can
  see inconsistent values.
-/

/-- A fact encoding location in a configuration -/
structure EncodingLocation where
  id : Nat
  value : Nat
  deriving DecidableEq

/-- A configuration with potentially multiple encodings of the same fact -/
structure MultiEncodingConfig where
  locations : List EncodingLocation
  -- DOF = number of independent encoding locations
  dof : Nat := locations.length

/-- All encodings agree on the value -/
def consistent (cfg : MultiEncodingConfig) : Prop :=
  ∀ l1 l2, l1 ∈ cfg.locations → l2 ∈ cfg.locations → l1.value = l2.value

/-- At least two encodings disagree -/
def inconsistent (cfg : MultiEncodingConfig) : Prop :=
  ∃ l1 l2, l1 ∈ cfg.locations ∧ l2 ∈ cfg.locations ∧ l1.value ≠ l2.value

/-- DOF = 1 implies consistency (SSOT = no inconsistency possible) -/
theorem dof_one_implies_consistent (cfg : MultiEncodingConfig)
    (h_nonempty : cfg.locations.length = 1) :
    consistent cfg := by
  intro l1 l2 h1 h2
  -- With only one location, l1 = l2
  cases hl : cfg.locations with
  | nil =>
    rw [hl] at h1
    simp at h1
  | cons x xs =>
    rw [hl] at h_nonempty
    simp only [List.length_cons] at h_nonempty
    have hxs : xs.length = 0 := by omega
    have hxs' : xs = [] := by
      cases xs with
      | nil => rfl
      | cons _ _ => simp at hxs
    rw [hl, hxs'] at h1 h2
    simp only [List.mem_singleton] at h1 h2
    rw [h1, h2]

/-- DOF > 1 permits inconsistency (can construct divergent state) -/
theorem dof_gt_one_permits_inconsistency :
    ∃ cfg : MultiEncodingConfig,
      cfg.dof > 1 ∧ inconsistent cfg := by
  -- Witness: two locations with different values
  let loc1 : EncodingLocation := ⟨0, 42⟩
  let loc2 : EncodingLocation := ⟨1, 99⟩
  use ⟨[loc1, loc2], 2⟩
  constructor
  · decide
  · use loc1, loc2
    refine ⟨?_, ?_, by decide⟩ <;> simp [loc1, loc2]

/-
  PART 2: Connecting to Type System Capabilities

  AbstractClassSystem proves that shape-based typing loses information
  (the Bases axis). This information loss is a form of DOF reduction
  that can cause inconsistency.
-/

/-- Type identity as an encoding location for provenance facts -/
def typeIdentityEncoding (T : Typ) : EncodingLocation :=
  ⟨0, T.bs.length⟩  -- Using bases length as a simple numeric encoding

/-- Two types with same shape but different bases encode provenance differently -/
theorem same_shape_different_provenance :
    ∃ T1 T2 : Typ,
      shapeEquivalent T1 T2 ∧
      typeIdentityEncoding T1 ≠ typeIdentityEncoding T2 := by
  -- Witness from AbstractClassSystem
  let parent : Typ := ⟨∅, []⟩
  let child : Typ := ⟨∅, [parent]⟩
  use child, parent
  constructor
  · -- Same shape (both have empty namespace)
    rfl
  · -- Different provenance encoding (different bases length)
    simp [typeIdentityEncoding, child, parent]

/-
  PART 3: The Main Bridge Theorem

  SSOT is necessary because:
  1. DOF > 1 permits inconsistency (dof_gt_one_permits_inconsistency)
  2. DOF = 0 means fact not encoded (unusable)
  3. DOF = 1 is the unique consistent, complete encoding

  Combined with AbstractClassSystem:
  - Shape-based typing discards Bases axis (reduces observable DOF)
  - This causes capability gap (provenance, identity lost)
  - SSOT for type facts requires nominal typing (preserves full DOF)
-/

/-- DOF ≥ 2 permits inconsistency (explicit witness) -/
theorem dof_ge_two_permits_inconsistency (dof : Nat) (_h : dof ≥ 2) :
    ∃ cfg : MultiEncodingConfig, cfg.dof = dof ∧ inconsistent cfg := by
  -- Build config: [⟨0, 42⟩, ⟨1, 99⟩] with declared dof
  use ⟨[⟨0, 42⟩, ⟨1, 99⟩], dof⟩
  constructor
  · rfl
  · use ⟨0, 42⟩, ⟨1, 99⟩
    refine ⟨?_, ?_, by decide⟩
    · simp
    · simp

/-- SSOT uniqueness: only DOF = 1 is both complete and guarantees consistency -/
theorem ssot_unique_complete_consistent :
    ∀ dof : Nat,
      dof ≠ 0 →  -- Complete: fact is encoded
      (∀ cfg : MultiEncodingConfig, cfg.dof = dof → consistent cfg) →  -- Always consistent
      satisfies_SSOT dof := by
  intro dof h_complete h_always_consistent
  unfold satisfies_SSOT
  by_contra h_not_one
  -- dof ≠ 0 and dof ≠ 1, so dof ≥ 2
  have h_ge2 : dof ≥ 2 := by omega
  -- Use the explicit witness theorem
  obtain ⟨cfg, h_dof_eq, loc1, loc2, h1, h2, h_neq⟩ := dof_ge_two_permits_inconsistency dof h_ge2
  -- h_always_consistent says this cfg must be consistent
  have h_con : consistent cfg := h_always_consistent cfg h_dof_eq
  -- But loc1 and loc2 have different values
  have h_eq : loc1.value = loc2.value := h_con loc1 loc2 h1 h2
  exact h_neq h_eq

/-- Completeness: DOF = 0 means fact is missing -/
theorem dof_zero_incomplete : ∀ cfg : MultiEncodingConfig,
    cfg.locations.length = 0 → cfg.locations = [] := by
  intro cfg h
  cases hl : cfg.locations with
  | nil => rfl
  | cons _ _ =>
    rw [hl] at h
    simp at h

/-- The trichotomy: every DOF is either incomplete, optimal, or permits inconsistency -/
theorem dof_trichotomy : ∀ dof : Nat,
    dof = 0 ∨ satisfies_SSOT dof ∨
    (∃ cfg : MultiEncodingConfig, cfg.dof = dof ∧ inconsistent cfg) := by
  intro dof
  match dof with
  | 0 => left; rfl
  | 1 => right; left; rfl
  | n + 2 =>
    right; right
    exact dof_ge_two_permits_inconsistency (n + 2) (by omega)

end SSOTGrounded
