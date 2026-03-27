import Ssot.Derivation

open Classical Ssot

namespace Dof

structure Encoding (Fact Value : Type) where
  fact : Fact
  location : String
  value : Value

def Derives {F V : Type} (D : DerivationSystem (Encoding F V))
    (e1 e2 : Encoding F V) : Prop :=
  D.derived_from e1 e2

def Independent {F V : Type} (D : DerivationSystem (Encoding F V))
    (e1 e2 : Encoding F V) : Prop :=
  ¬Derives D e1 e2 ∧ ¬Derives D e2 e1

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

noncomputable def dof {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) : Nat :=
  (minimalIndependentCore D encodings).length

def SSOT {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) : Prop :=
  dof D encodings = 1

end Dof
