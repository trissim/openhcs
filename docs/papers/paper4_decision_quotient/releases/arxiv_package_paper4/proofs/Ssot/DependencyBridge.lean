import lwd_converse

/-!
  Dependency bridge: Paper 2 -> Paper 1

  This module imports a Paper 1 mechanized result and re-exposes it under
  `Ssot` so the Paper 2 active graph uses Paper 1 proofs directly.
-/

namespace Ssot

/-- Paper 1 finite counting converse reused in Paper 2. -/
theorem paper1_collision_block_requires_bits
    {a L : Nat} {Transcript : Type _}
    (tag : Fin a → Fin (2 ^ L))
    (transcript : Fin a → Transcript)
    (decode : Fin (2 ^ L) → Transcript → Fin a)
    (htranscript : ∀ c₁ c₂, transcript c₁ = transcript c₂)
    (hzero : ∀ c, decode (tag c) (transcript c) = c) :
    a ≤ 2 ^ L :=
  LWDConverse.collision_block_requires_bits tag transcript decode htranscript hzero

end Ssot
