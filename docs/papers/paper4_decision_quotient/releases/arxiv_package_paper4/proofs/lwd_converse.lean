import Mathlib.Data.Fintype.Card

/-!
  LWD Converse Counting Lemmas

  These lemmas formalize the finite counting core and the factorization step
  behind the paper's converse:
  if all classes in a collision block induce the same query transcript, then
  any zero-error decoder must separate the block using tag outcomes alone.
  Hence a tag channel with `L` bits must satisfy `a ≤ 2^L`.
-/

namespace LWDConverse

/-- Any zero-error decoder for `a` colliding classes needs at least `a` outcomes. -/
theorem collision_block_requires_outcomes {a m : Nat}
    (encode : Fin a → Fin m) (hinj : Function.Injective encode) :
    a ≤ m := by
  simpa [Fintype.card_fin] using Fintype.card_le_of_injective encode hinj

/-- Collision-block converse:
if transcript is constant on a block and decoding is zero-error on that block,
tags must be injective on the block, so `a ≤ 2^L`. -/
theorem collision_block_requires_bits {a L : Nat} {Transcript : Type _}
    (tag : Fin a → Fin (2 ^ L))
    (transcript : Fin a → Transcript)
    (decode : Fin (2 ^ L) → Transcript → Fin a)
    (htranscript : ∀ c₁ c₂, transcript c₁ = transcript c₂)
    (hzero : ∀ c, decode (tag c) (transcript c) = c) :
    a ≤ 2 ^ L := by
  have htag_inj : Function.Injective tag := by
    intro c₁ c₂ htag
    have htr : transcript c₁ = transcript c₂ := htranscript c₁ c₂
    calc
      c₁ = decode (tag c₁) (transcript c₁) := by symm; exact hzero c₁
      _ = decode (tag c₂) (transcript c₂) := by simpa [htag, htr]
      _ = c₂ := hzero c₂
  exact collision_block_requires_outcomes tag htag_inj

/-- Maximal-barrier special case: `k` colliding classes force `k ≤ 2^L`. -/
theorem maximal_barrier_requires_bits {k L : Nat} {Transcript : Type _}
    (tag : Fin k → Fin (2 ^ L))
    (transcript : Fin k → Transcript)
    (decode : Fin (2 ^ L) → Transcript → Fin k)
    (htranscript : ∀ c₁ c₂, transcript c₁ = transcript c₂)
    (hzero : ∀ c, decode (tag c) (transcript c) = c) :
    k ≤ 2 ^ L := by
  exact collision_block_requires_bits tag transcript decode htranscript hzero

/-- Impossibility form: too few tag bits rules out zero-error separation. -/
theorem impossible_when_bits_too_small {a L : Nat} {Transcript : Type _}
    (hsmall : 2 ^ L < a) :
    ¬ ∃ (tag : Fin a → Fin (2 ^ L))
        (transcript : Fin a → Transcript)
        (decode : Fin (2 ^ L) → Transcript → Fin a),
        (∀ c₁ c₂, transcript c₁ = transcript c₂) ∧
        (∀ c, decode (tag c) (transcript c) = c) := by
  intro h
  rcases h with ⟨tag, transcript, decode, htranscript, hzero⟩
  have hle : a ≤ 2 ^ L :=
    collision_block_requires_bits tag transcript decode htranscript hzero
  exact Nat.not_lt_of_ge hle hsmall

end LWDConverse
