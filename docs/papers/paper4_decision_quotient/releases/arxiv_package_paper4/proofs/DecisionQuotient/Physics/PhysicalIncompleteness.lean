import Mathlib.Data.Fintype.Card
import Mathlib.Data.Set.Basic
import Mathlib.Tactic
import DecisionQuotient.Dichotomy

/-!
# Physical Incompleteness of Instantiation

## Dependency Graph
- **Nontrivial source:** Dichotomy theorem (encoding-regime separation)
- **This module:** Cardinality triviality - if logical > physical, some logical
  is not physical. This is a consequence of the dichotomy, not a core proof.

## Role
This is a trivial cardinality argument. The nontrivial content is the dichotomy
theorem in Dichotomy.lean which establishes the encoding regimes.

## Triviality Level
TRIVIAL — cardinality and finiteness arguments. Closest nontrivial source: `Dichotomy.dichotomy`
(DecisionQuotient.Dichotomy).
-/

namespace DecisionQuotient
namespace Physics
namespace PhysicalIncompleteness

universe u v

/-- Universe-level interface for physical instantiation of logical possibilities. -/
structure UniverseModel where
  PhysicalState : Type u
  LogicalPossibility : Type v
  instantiates : PhysicalState → LogicalPossibility

/-- Logical possibility `ℓ` is physically instantiated iff it is in the range. -/
def PhysicallyInstantiated (U : UniverseModel) (ℓ : U.LogicalPossibility) : Prop :=
  ℓ ∈ Set.range U.instantiates

/-- Strict cardinality gap blocks surjective physical instantiation. -/
theorem no_surjective_instantiation_of_card_gap
    (U : UniverseModel)
    [Fintype U.PhysicalState] [Fintype U.LogicalPossibility]
    (hGap : Fintype.card U.PhysicalState < Fintype.card U.LogicalPossibility) :
    ¬ Function.Surjective U.instantiates := by
  intro hSurj
  have hLe : Fintype.card U.LogicalPossibility ≤ Fintype.card U.PhysicalState :=
    Fintype.card_le_of_surjective U.instantiates hSurj
  exact (Nat.not_lt_of_ge hLe) hGap

/-- Physical incompleteness theorem:
there exists a logical possibility that cannot be physically instantiated. -/
theorem physical_incompleteness_of_card_gap
    (U : UniverseModel)
    [Fintype U.PhysicalState] [Fintype U.LogicalPossibility]
    (hGap : Fintype.card U.PhysicalState < Fintype.card U.LogicalPossibility) :
    ∃ ℓ : U.LogicalPossibility, ¬ PhysicallyInstantiated U ℓ := by
  by_contra hNone
  have hAll : ∀ ℓ : U.LogicalPossibility, PhysicallyInstantiated U ℓ := by
    intro ℓ
    by_contra hNot
    exact hNone ⟨ℓ, hNot⟩
  have hSurj : Function.Surjective U.instantiates := by
    intro ℓ
    rcases hAll ℓ with ⟨p, hp⟩
    exact ⟨p, hp⟩
  exact (no_surjective_instantiation_of_card_gap U hGap) hSurj

/-- Bound-form corollary:
if physical capacity is bounded above by `M` and logical possibility count
is bounded below by `L` with `M < L`, physical incompleteness follows. -/
theorem physical_incompleteness_of_bounds
    (U : UniverseModel)
    [Fintype U.PhysicalState] [Fintype U.LogicalPossibility]
    (M L : Nat)
    (hPhys : Fintype.card U.PhysicalState ≤ M)
    (hLog : L ≤ Fintype.card U.LogicalPossibility)
    (hGap : M < L) :
    ∃ ℓ : U.LogicalPossibility, ¬ PhysicallyInstantiated U ℓ := by
  have hCardGap : Fintype.card U.PhysicalState < Fintype.card U.LogicalPossibility := by
    exact lt_of_le_of_lt hPhys (lt_of_lt_of_le hGap hLog)
  exact physical_incompleteness_of_card_gap U hCardGap

/-- Under-resolution collision theorem:
if the representation alphabet is strictly smaller than the logical space,
some two distinct logical possibilities must share the same code. -/
theorem under_resolution_implies_collision
    {Logical Code : Type*}
    [Fintype Logical] [Fintype Code]
    (encode : Logical → Code)
    (hUnder : Fintype.card Code < Fintype.card Logical) :
    ∃ ℓ₁ ℓ₂ : Logical, ℓ₁ ≠ ℓ₂ ∧ encode ℓ₁ = encode ℓ₂ := by
  have hNotInj : ¬ Function.Injective encode := by
    intro hInj
    have hLe : Fintype.card Logical ≤ Fintype.card Code :=
      Fintype.card_le_of_injective encode hInj
    exact (Nat.not_lt_of_ge hLe) hUnder
  by_contra hNoCollision
  have hInj : Function.Injective encode := by
    intro a b hab
    by_contra hne
    exact hNoCollision ⟨a, b, hne, hab⟩
  exact hNotInj hInj

/-- Decision-facing alias:
if a decision representation is under-resolved, distinct logical states are
forced to collide at the representation level. -/
theorem under_resolution_implies_decision_collision
    {Logical Code : Type*}
    [Fintype Logical] [Fintype Code]
    (repr : Logical → Code)
    (hUnder : Fintype.card Code < Fintype.card Logical) :
    ∃ s s' : Logical, s ≠ s' ∧ repr s = repr s' :=
  under_resolution_implies_collision repr hUnder

end PhysicalIncompleteness
end Physics
end DecisionQuotient
