import Mathlib.Tactic
import abstract_class_system

/-
  TypeScript Instantiation: axes and observers factorization

  We model TS types with:
  - B : heritage (extends/implements) list + optional brand (identity)
  - S : property signatures (names only, for shape)

  NOTE: There is no N axis. The brand is part of B (nominal identity).
  See paper Theorem 2.2: names/brands are derived from B, not independent.

  Observers:
  - hasProp (structural)
  - instanceOf (heritage / brand from B)
  - brandIs (nominal identity from B)
-/

namespace TypeScriptInstantiation
open AbstractClassSystem

abbrev Typ := Nat
abbrev AttrName := String

structure TSType where
  bases : List Typ   -- heritage
  props : List AttrName -- structural members
  brand : Typ        -- nominal brand/tag (0 = no brand)
  deriving DecidableEq, Repr

def tsAxes (t : TSType) : List Typ × List AttrName × Typ :=
  (t.bases, t.props, t.brand)

def sameAxes (a b : TSType) : Prop := tsAxes a = tsAxes b

-- Observers
def tsHasProp (attr : AttrName) : TSType → Bool :=
  fun t => decide (attr ∈ t.props)

def tsInstanceOf (T : Typ) : TSType → Bool :=
  fun t => decide (T ∈ t.bases ∨ T = t.brand)

def tsBrandIs (T : Typ) : TSType → Bool :=
  fun t => decide (t.brand = T)

-- Factorization lemmas
lemma hasProp_factors {a b : TSType} (h : sameAxes a b) (attr : AttrName) :
    tsHasProp attr a = tsHasProp attr b := by
  cases a; cases b; cases h; simp [tsHasProp]

lemma instanceOf_factors {a b : TSType} (h : sameAxes a b) (T : Typ) :
    tsInstanceOf T a = tsInstanceOf T b := by
  cases a; cases b; cases h; simp [tsInstanceOf]

lemma brandIs_factors {a b : TSType} (h : sameAxes a b) (T : Typ) :
    tsBrandIs T a = tsBrandIs T b := by
  cases a; cases b; cases h; simp [tsBrandIs]

-- Aggregate: all observers above factor through the axes projection
lemma observables_factor_through_axes {a b : TSType} (h : sameAxes a b) (attr : AttrName) (T : Typ) :
    tsHasProp attr a = tsHasProp attr b ∧ tsInstanceOf T a = tsInstanceOf T b ∧ tsBrandIs T a = tsBrandIs T b := by
  exact ⟨hasProp_factors h attr, instanceOf_factors h T, brandIs_factors h T⟩

end TypeScriptInstantiation
