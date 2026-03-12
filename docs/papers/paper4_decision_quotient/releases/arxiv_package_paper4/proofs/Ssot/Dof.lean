import Ssot.Derivation

/-!
  DOF Foundation - Minimal Semantics for Paper 2

  This file replaces the trivial "dof = Nat" formulation with a minimal,
  semantics-backed notion of degrees of freedom. It connects directly to the
  derivation relation from Paper 1 (via `DerivationSystem`).

  The intent is lightweight: no full semantics, just enough structure to make
  the SSOT claims meaningful.
-/

open Classical Ssot

namespace Dof

/-!
## Encodings

An encoding is a concrete location that stores a value for a fact.
Two encodings are independent unless one is derivable from the other.
-/

structure Encoding (Fact Value : Type) where
  fact : Fact
  location : String
  value : Value

/-!
## Derivability and Independence

`Derives e1 e2` means e2 is derived from e1 (changing e1 forces e2 to change).
This is parameterized by a `DerivationSystem` on encodings.
-/

def Derives {F V : Type} (D : DerivationSystem (Encoding F V))
    (e1 e2 : Encoding F V) : Prop :=
  D.derived_from e1 e2

def Independent {F V : Type} (D : DerivationSystem (Encoding F V))
    (e1 e2 : Encoding F V) : Prop :=
  ¬Derives D e1 e2 ∧ ¬Derives D e2 e1

/-!
## Minimal Independent Core

We conservatively model the "independent core" as the subset of encodings
that are not derivable from any other encoding in the list.
-/

def redundant {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) (e : Encoding F V) : Prop :=
  ∃ e' ∈ encodings, e' ≠ e ∧ Derives D e' e

noncomputable def minimalIndependentCore {F V : Type}
    (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) : List (Encoding F V) :=
  encodings.filter (fun e => decide (¬ redundant D encodings e))

theorem core_subset {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) :
    minimalIndependentCore D encodings ⊆ encodings := by
  intro e h
  exact (List.mem_filter.mp h).left

/-!
## DOF and SSOT

DOF is the size of the minimal independent core.
SSOT holds iff DOF = 1.
-/

noncomputable def dof {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) : Nat :=
  (minimalIndependentCore D encodings).length

def SSOT {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) : Prop :=
  dof D encodings = 1

end Dof
