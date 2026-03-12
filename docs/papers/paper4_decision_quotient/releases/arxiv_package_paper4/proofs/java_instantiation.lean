import Mathlib.Tactic
import abstract_class_system

/-
  Java Instantiation: axes and observers factorization

  Axes:
  - B : superclass + interfaces list, plus class identity (tag)
  - S : field/method names (names only)

  NOTE: There is no N axis. The class tag is part of B (identity in hierarchy).
  See paper Theorem 2.2: names are derived from B, not independent.

  Observers:
  - hasMember (reflection presence)
  - instanceOf (inherits/implements)
  - getClassTag (identity from B)
-/

namespace JavaInstantiation
open AbstractClassSystem

abbrev Typ := Nat
abbrev AttrName := String

structure JavaType where
  bases : List Typ        -- superclass first, then interfaces
  members : List AttrName -- field/method names
  tag : Typ               -- runtime class tag
  deriving DecidableEq, Repr

def jtAxes (j : JavaType) : List Typ × List AttrName × Typ :=
  (j.bases, j.members, j.tag)

def sameAxes (a b : JavaType) : Prop := jtAxes a = jtAxes b

-- Observers
def hasMember (attr : AttrName) : JavaType → Bool :=
  fun j => decide (attr ∈ j.members)

def instanceOf (T : Typ) : JavaType → Bool :=
  fun j => decide (T ∈ j.bases ∨ T = j.tag)

def getClassTag : JavaType → Typ := fun j => j.tag

-- Factorization
lemma hasMember_factors {a b : JavaType} (h : sameAxes a b) (attr : AttrName) :
    hasMember attr a = hasMember attr b := by
  cases a; cases b; cases h; simp [hasMember]

lemma instanceOf_factors {a b : JavaType} (h : sameAxes a b) (T : Typ) :
    instanceOf T a = instanceOf T b := by
  cases a; cases b; cases h; simp [instanceOf]

lemma tag_factors {a b : JavaType} (h : sameAxes a b) :
    getClassTag a = getClassTag b := by
  cases a; cases b; cases h; simp [getClassTag]

lemma observables_factor_through_axes {a b : JavaType} (h : sameAxes a b) (attr : AttrName) (T : Typ) :
    hasMember attr a = hasMember attr b ∧ instanceOf T a = instanceOf T b ∧ getClassTag a = getClassTag b := by
  exact ⟨hasMember_factors h attr, instanceOf_factors h T, tag_factors h⟩

end JavaInstantiation
