import Mathlib.Computability.Halting
import AbstractClassSystem.Core

open Nat.Partrec.Code

namespace AbstractClassSystem
namespace OpenWorld

/-- A finite world is barrier-free when in-world shape collisions imply equality. -/
def BarrierFreeWorld (W : Finset Typ) : Prop :=
  ∀ ⦃A B : Typ⦄, A ∈ W → B ∈ W → shapeEquivalent A B → A = B

/-- Any finite world can be extended to a super-world with a guaranteed barrier. -/
theorem extend_to_force_barrier (W : Finset Typ) :
    ∃ W' : Finset Typ, W ⊆ W' ∧ ¬ BarrierFreeWorld W' := by
  let W' : Finset Typ := insert ConfigType (insert StepConfigType W)
  refine ⟨W', ?_, ?_⟩
  · intro T hT
    exact Finset.mem_insert_of_mem (Finset.mem_insert_of_mem hT)
  · intro hfree
    have hA : ConfigType ∈ W' := Finset.mem_insert_self _ _
    have hB : StepConfigType ∈ W' := Finset.mem_insert_of_mem (Finset.mem_insert_self _ _)
    have hEq : ConfigType = StepConfigType := hfree hA hB (by rfl)
    exact types_nominally_distinct hEq

/-- Barrier-freedom is not extension-stable in an open-world model. -/
theorem barrierFreedom_not_extension_stable :
    ¬ (∀ W W' : Finset Typ, W ⊆ W' → BarrierFreeWorld W → BarrierFreeWorld W') := by
  intro hstable
  let W0 : Finset Typ := ∅
  have hfree0 : BarrierFreeWorld W0 := by
    intro A B hA
    have hempty : A ∉ (∅ : Finset Typ) := by simp
    exact False.elim (hempty hA)
  rcases extend_to_force_barrier W0 with ⟨W1, hsub, hbar⟩
  exact hbar (hstable W0 W1 hsub hfree0)

/-- Function-level barrier event: two distinct inputs collapse to the same observed output. -/
def HasBarrier (f : ℕ →. ℕ) : Prop :=
  ∃ x y z : ℕ, x ≠ y ∧ z ∈ f x ∧ z ∈ f y

/-- Absence of barriers for a partial observer. -/
def BarrierFree (f : ℕ →. ℕ) : Prop := ¬ HasBarrier f

lemma zero_has_barrier : HasBarrier (pure 0 : ℕ →. ℕ) := by
  refine ⟨0, 1, 0, by decide, ?_, ?_⟩
  · exact Part.mem_some _
  · exact Part.mem_some _

lemma succ_barrier_free : BarrierFree ((↑Nat.succ) : ℕ →. ℕ) := by
  intro h
  rcases h with ⟨x, y, z, hxy, hz1, hz2⟩
  have hx : z = x.succ := by simpa using hz1
  have hy : z = y.succ := by simpa using hz2
  have hxy' : x = y := Nat.succ.inj (hx.symm.trans hy)
  exact hxy hxy'

/-- Rice-style impossibility: no computable decider can certify barrier existence for arbitrary generators. -/
theorem hasBarrier_not_computable :
    ¬ ComputablePred (fun c : Nat.Partrec.Code => HasBarrier (c.eval)) := by
  intro h
  have hsucc : HasBarrier (((↑Nat.succ) : ℕ →. ℕ)) := by
    exact ComputablePred.rice {f : ℕ →. ℕ | HasBarrier f} h
      Nat.Partrec.zero Nat.Partrec.succ zero_has_barrier
  exact succ_barrier_free hsucc

/-- Equivalent Rice-style impossibility for barrier-freedom certification. -/
theorem barrierFree_not_computable :
    ¬ ComputablePred (fun c : Nat.Partrec.Code => BarrierFree (c.eval)) := by
  intro h
  have hHas : ComputablePred (fun c : Nat.Partrec.Code => ¬ BarrierFree (c.eval)) :=
    ComputablePred.not h
  have hHas' : ComputablePred (fun c : Nat.Partrec.Code => HasBarrier (c.eval)) := by
    exact ComputablePred.of_eq hHas (by
      intro c
      simp [BarrierFree])
  exact hasBarrier_not_computable hHas'

end OpenWorld
end AbstractClassSystem
