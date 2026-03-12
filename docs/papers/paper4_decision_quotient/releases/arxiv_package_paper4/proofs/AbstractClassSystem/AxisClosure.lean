import Mathlib.Logic.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Tactic

/-!
# Axis Closure Formalization

This module provides the closure operator for axis systems, formalizing
when adding an axis yields strictly new observables.

The closure system is used by the axis framework to establish matroid
structure and basis equicardinality theorems.
-/

namespace AbstractClassSystem

namespace AxisClosure

universe u v

variable {W : Type u} {ι : Type v}
variable (Val : ι → Type) (π : ∀ i : ι, W → Val i)

/-- Indistinguishability using only the axes in `X`. -/
def AxisEq (X : Set ι) (x y : W) : Prop :=
  ∀ i : ι, i ∈ X → π i x = π i y

/-- Axis `a` is determined by axis-set `X` if `X`-equality forces agreement on `a`. -/
def Determines (X : Set ι) (a : ι) : Prop :=
  ∀ x y : W, AxisEq (Val := Val) (π := π) X x y → π a x = π a y

/-- Closure of `X`: all axes determined by `X`. -/
def closure (X : Set ι) : Set ι :=
  {a | Determines (Val := Val) (π := π) X a}

theorem closure_extensive (X : Set ι) :
    X ⊆ closure (Val := Val) (π := π) X := by
  intro a ha x y hX
  exact hX a ha

theorem closure_mono {X Y : Set ι} (hXY : X ⊆ Y) :
    closure (Val := Val) (π := π) X ⊆ closure (Val := Val) (π := π) Y := by
  intro a ha x y hY
  have hX : AxisEq (Val := Val) (π := π) X x y := by
    intro i hiX; exact hY i (hXY hiX)
  exact ha x y hX

theorem closure_idem (X : Set ι) :
    closure (Val := Val) (π := π) (closure (Val := Val) (π := π) X)
      = closure (Val := Val) (π := π) X := by
  classical
  ext a; constructor
  · intro ha x y hX
    have hcl : AxisEq (Val := Val) (π := π) (closure (Val := Val) (π := π) X) x y := by
      intro i hi; exact hi x y hX
    exact ha x y hcl
  · intro ha x y hcl
    have hX : AxisEq (Val := Val) (π := π) X x y := by
      intro i hiX
      exact hcl i ((closure_extensive (Val := Val) (π := π) X) hiX)
    exact ha x y hX

/-- If an axis is *not* in the closure of `X`, there exists a witness distinguishing it. -/
theorem gain_of_not_in_closure {a : ι} {X : Set ι}
    (h : a ∉ closure (Val := Val) (π := π) X) :
    ∃ x y : W, AxisEq (Val := Val) (π := π) X x y ∧ π a x ≠ π a y := by
  classical
  have h' : ¬ Determines (Val := Val) (π := π) X a := by
    simpa [closure] using h
  unfold Determines at h'
  rcases not_forall.mp h' with ⟨x, hx⟩
  rcases not_forall.mp hx with ⟨y, hy⟩
  have hcontra : AxisEq (Val := Val) (π := π) X x y ∧ π a x ≠ π a y :=
    Classical.not_imp.mp hy
  exact ⟨x, y, hcontra.1, hcontra.2⟩

end AxisClosure

end AbstractClassSystem
