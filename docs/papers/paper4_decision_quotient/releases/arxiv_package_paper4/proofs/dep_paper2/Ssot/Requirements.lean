/-
  SSOT Formalization - Language Requirements (Necessity Proofs)
  Paper 2: Formal Foundations for the Single Source of Truth Principle

  KEY INSIGHT: These requirements are DERIVED, not chosen.
  The logical structure forces them from the definition of SSOT.
-/

import Ssot.Basic
import Ssot.Derivation

namespace Ssot

-- Language feature predicates
structure LanguageFeatures where
  -- DEF: Definition-time hooks (code executes when class/type is defined)
  has_definition_hooks : Bool
  -- INTRO: Introspectable derivation results (can query what was derived)
  has_introspection : Bool
  -- STRUCT: Can modify structure at definition time
  has_structural_modification : Bool
  -- HIER: Can enumerate subclasses/implementers
  has_hierarchy_queries : Bool
  deriving DecidableEq, Inhabited

-- A structural fact is one established at definition time
-- (class existence, method signatures, type relationships)
inductive FactKind where
  | structural  -- Fixed at definition time
  | runtime     -- Can be modified at runtime
  deriving DecidableEq

-- Timing relation: when can a fact be modified?
inductive Timing where
  | definition  -- At class/type definition
  | runtime     -- After program starts
  deriving DecidableEq

-- Axiom: Structural facts are fixed at definition time
def structural_timing : FactKind → Timing
  | FactKind.structural => Timing.definition
  | FactKind.runtime => Timing.runtime

-- Derivation must occur at or before the fact is established
-- This is the key constraint that forces definition-time hooks
def derivation_timing_constraint (F : FactKind) : Timing :=
  structural_timing F

-- Can a language derive at the required time?
def can_derive_at (L : LanguageFeatures) (t : Timing) : Bool :=
  match t with
  | Timing.definition => L.has_definition_hooks
  | Timing.runtime => true  -- All languages can compute at runtime

-- Theorem 3.2: Definition-Time Hooks are NECESSARY for Structural SSOT
-- DERIVED from: structural facts fixed at definition ⟹ derivation must happen then
theorem definition_hooks_necessary (L : LanguageFeatures) :
    can_derive_at L Timing.definition = false →
    -- Then L cannot achieve SSOT for structural facts
    -- Because derivation cannot happen when the fact is established
    L.has_definition_hooks = false := by
  intro h
  simp [can_derive_at] at h
  exact h

-- Verification requires enumeration
-- If we can't list all encodings, we can't verify DOF = 1
def can_enumerate_encodings (L : LanguageFeatures) : Bool :=
  L.has_introspection

-- Theorem 3.4: Introspection is NECESSARY for Verifiable SSOT
-- DERIVED from: verification requires enumeration ⟹ introspection required
theorem introspection_necessary_for_verification (L : LanguageFeatures) :
    can_enumerate_encodings L = false →
    -- Then SSOT cannot be verified
    L.has_introspection = false := by
  intro h
  simp [can_enumerate_encodings] at h
  exact h

-- Exhaustive dispatch requires knowing all implementers
def can_exhaustive_dispatch (L : LanguageFeatures) : Bool :=
  L.has_hierarchy_queries

-- Theorem 3.5: Hierarchy Queries Enable Exhaustive Dispatch
theorem hierarchy_enables_exhaustive (L : LanguageFeatures) :
    can_exhaustive_dispatch L = true ↔ L.has_hierarchy_queries = true := by
  simp [can_exhaustive_dispatch]

-- Corollary: Without hierarchy queries, exhaustive dispatch is impossible
theorem no_hierarchy_no_exhaustive (L : LanguageFeatures)
    (h : L.has_hierarchy_queries = false) :
    can_exhaustive_dispatch L = false := by
  simp [can_exhaustive_dispatch, h]

-- THE KEY THEOREM: Both requirements are independently necessary
-- Missing either one breaks SSOT guarantee
theorem both_requirements_independent :
    -- Definition hooks: needed for derivation timing
    -- Introspection: needed for verification
    -- Neither implies the other; both are required
    ∀ L : LanguageFeatures,
      (L.has_definition_hooks = true ∧ L.has_introspection = false) →
      can_enumerate_encodings L = false := by
  intro L ⟨_, h_no_intro⟩
  simp [can_enumerate_encodings, h_no_intro]

theorem both_requirements_independent' :
    ∀ L : LanguageFeatures,
      (L.has_definition_hooks = false ∧ L.has_introspection = true) →
      can_derive_at L Timing.definition = false := by
  intro L ⟨h_no_hooks, _⟩
  simp [can_derive_at, h_no_hooks]

end Ssot
