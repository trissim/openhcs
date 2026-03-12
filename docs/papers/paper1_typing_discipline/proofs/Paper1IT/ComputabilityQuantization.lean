import Mathlib.Algebra.Order.Floor.Div
import Paper1IT.GraphEntropy

namespace Ssot
namespace GraphEntropy

variable {α β : Type*} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]

/-- `A_π` is the finite supremum of the fiber cardinalities. This is the finite-enumeration
formula cited in the paper's computability remark. -/
theorem maxFiberCard_eq_finiteSup (observe : α → β) :
    maxFiberCard observe =
      Finset.sup Finset.univ
        (fun b : β => (Finset.univ.filter (fun a : α => observe a = b)).card) := rfl

/-- For finite alphabets, the collision multiplicity is obtained by enumerating the codomain,
counting each fiber, and taking the finite supremum. -/
theorem maxFiberCard_computable_by_fiber_enumeration (observe : α → β) :
    ∃ compute : β → ℕ,
      (∀ b, compute b = (Finset.univ.filter (fun a : α => observe a = b)).card) ∧
      maxFiberCard observe = Finset.sup Finset.univ compute := by
  refine ⟨fun b => (Finset.univ.filter (fun a : α => observe a = b)).card, ?_, ?_⟩
  · intro b
    rfl
  · rfl

/-- Global counting bound induced by the worst fiber size. This is derived from the existing
zero-error tagging machinery rather than reproved directly. -/
theorem card_le_card_mul_maxFiberCard (observe : α → β) [Nonempty β] :
    Fintype.card α ≤ Fintype.card β * maxFiberCard observe := by
  rcases tagFeasible_of_maxFiberCard_le (observe := observe) (n := maxFiberCard observe) le_rfl with
    ⟨tag, htag⟩
  exact card_le_mul_tag_alphabet (observe := observe) tag htag

/-- Quantization lower bound in ceil-div form: some fiber must have size at least the average
occupancy rounded up. -/
theorem ceilDiv_card_le_maxFiberCard (observe : α → β) [Nonempty β] :
    Fintype.card α ⌈/⌉ Fintype.card β ≤ maxFiberCard observe := by
  have hβ : 0 < Fintype.card β := Fintype.card_pos_iff.mpr ‹Nonempty β›
  exact (ceilDiv_le_iff_le_mul hβ).2 <| by
    simpa [Nat.mul_comm] using card_le_card_mul_maxFiberCard (observe := observe)

/-- `Fin`-specialized quantization lower bound. If a representation uses `m` bins, the maximal
collision fiber is at least `⌈n/m⌉`. -/
theorem quantization_lower_bound_fin {n m : ℕ} (observe : Fin n → Fin m) (hm : 0 < m) :
    n ⌈/⌉ m ≤ maxFiberCard observe := by
  letI : Nonempty (Fin m) := ⟨⟨0, hm⟩⟩
  simpa [Fintype.card_fin] using ceilDiv_card_le_maxFiberCard (observe := observe)

/-- If every fiber has size at most `k`, then the total class set fits inside `|β| * k`. -/
theorem card_le_card_mul_of_maxFiberCard_le (observe : α → β) [Nonempty β] {k : ℕ}
    (hk : maxFiberCard observe ≤ k) :
    Fintype.card α ≤ Fintype.card β * k := by
  rcases tagFeasible_of_maxFiberCard_le (observe := observe) (n := k) hk with ⟨tag, htag⟩
  exact card_le_mul_tag_alphabet (observe := observe) tag htag

/-- Precision requirement in ceil-div form: if all collision fibers are bounded by `k`, then the
representation needs at least `⌈|α|/k⌉` distinct output values. -/
theorem precision_requirement (observe : α → β) [Nonempty β] {k : ℕ} (hk : 0 < k)
    (hmax : maxFiberCard observe ≤ k) :
    Fintype.card α ⌈/⌉ k ≤ Fintype.card β := by
  exact (ceilDiv_le_iff_le_mul hk).2 <| by
    simpa [Nat.mul_comm] using
      card_le_card_mul_of_maxFiberCard_le (observe := observe) hmax

/-- `Fin`-specialized precision requirement. -/
theorem precision_requirement_fin {n m k : ℕ} (observe : Fin n → Fin m)
    (hm : 0 < m) (hk : 0 < k) (hmax : maxFiberCard observe ≤ k) :
    n ⌈/⌉ k ≤ m := by
  letI : Nonempty (Fin m) := ⟨⟨0, hm⟩⟩
  simpa [Fintype.card_fin] using
    precision_requirement (observe := observe) hk hmax

end GraphEntropy
end Ssot
