import Mathlib.Tactic
import abstract_class_system

/-
  Rust Instantiation: axes and observers factorization

  Axes:
  - B : implemented trait IDs (vtable set) + concrete type_id (identity)
  - S : provided method names (for objects with known vtable)

  NOTE: There is no N axis. The typeId is part of B (identity in trait hierarchy).
  See paper Theorem 2.2: names/tags are derived from B, not independent.

  Observers:
  - implementsTrait (vtable membership)
  - hasMethod (method presence via vtable)
  - typeIdIs (downcast identity from B)
-/

namespace RustInstantiation
open AbstractClassSystem

abbrev Typ := Nat
abbrev AttrName := String

structure RustType where
  traits : List Typ      -- implemented traits (vtable IDs)
  methods : List AttrName -- method names exposed via trait objects
  typeId : Typ           -- concrete type id (as in Any::type_id)
  deriving DecidableEq, Repr

def rtAxes (r : RustType) : List Typ × List AttrName × Typ :=
  (r.traits, r.methods, r.typeId)

def sameAxes (a b : RustType) : Prop := rtAxes a = rtAxes b

-- Observers
def implementsTrait (tr : Typ) : RustType → Bool :=
  fun r => decide (tr ∈ r.traits)

def hasMethod (m : AttrName) : RustType → Bool :=
  fun r => decide (m ∈ r.methods)

def typeIdIs (t : Typ) : RustType → Bool :=
  fun r => decide (r.typeId = t)

-- Factorization
lemma implementsTrait_factors {a b : RustType} (h : sameAxes a b) (tr : Typ) :
    implementsTrait tr a = implementsTrait tr b := by
  cases a; cases b; cases h; simp [implementsTrait]

lemma hasMethod_factors {a b : RustType} (h : sameAxes a b) (m : AttrName) :
    hasMethod m a = hasMethod m b := by
  cases a; cases b; cases h; simp [hasMethod]

lemma typeId_factors {a b : RustType} (h : sameAxes a b) (t : Typ) :
    typeIdIs t a = typeIdIs t b := by
  cases a; cases b; cases h; simp [typeIdIs]

lemma observables_factor_through_axes {a b : RustType} (h : sameAxes a b) (tr : Typ) (m : AttrName) (t : Typ) :
    implementsTrait tr a = implementsTrait tr b ∧ hasMethod m a = hasMethod m b ∧ typeIdIs t a = typeIdIs t b := by
  exact ⟨implementsTrait_factors h tr, hasMethod_factors h m, typeId_factors h t⟩

end RustInstantiation
