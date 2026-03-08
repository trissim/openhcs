import DecisionQuotient.Information
import DecisionQuotient.Physics.TemporalCountingGap

open Classical

namespace DecisionQuotient.DryRun

/-
Dry-run scaffold for the inflation->entropy bridge.
This file is intentionally NOT imported by DecisionQuotient.lean.
-/

theorem numOptClasses_mono_of_embedding
    {A S1 S2 : Type*}
    [Fintype S1] [Fintype S2]
    [DecidableEq (Set A)]
    (dp1 : DecisionProblem A S1)
    (dp2 : DecisionProblem A S2)
    (emb : S1 → S2)
    (hOptCompat : ∀ s : S1, dp2.Opt (emb s) = dp1.Opt s) :
    dp1.numOptClasses ≤ dp2.numOptClasses := by
  classical
  unfold DecisionProblem.numOptClasses
  have hsubset : (Finset.univ.image dp1.Opt) ⊆ (Finset.univ.image dp2.Opt) := by
    intro O hO
    rcases Finset.mem_image.mp hO with ⟨s, _, hs⟩
    apply Finset.mem_image.mpr
    refine ⟨emb s, Finset.mem_univ _, ?_⟩
    simpa [hOptCompat s] using hs
  exact Finset.card_le_card hsubset

theorem quotientEntropy_mono_of_embedding
    {A S1 S2 : Type*}
    [Fintype S1] [Fintype S2] [Nonempty S1] [Nonempty S2]
    [DecidableEq (Set A)]
    (dp1 : DecisionProblem A S1)
    (dp2 : DecisionProblem A S2)
    (emb : S1 → S2)
    (hOptCompat : ∀ s : S1, dp2.Opt (emb s) = dp1.Opt s) :
    dp1.quotientEntropy ≤ dp2.quotientEntropy := by
  have hClasses : dp1.numOptClasses ≤ dp2.numOptClasses :=
    numOptClasses_mono_of_embedding dp1 dp2 emb hOptCompat
  have hPos1 : 0 < dp1.numOptClasses := dp1.numOptClasses_pos
  have hPos2 : 0 < dp2.numOptClasses := dp2.numOptClasses_pos
  have hLog : Real.log (dp1.numOptClasses : ℝ) ≤ Real.log (dp2.numOptClasses : ℝ) :=
    Real.log_le_log (by exact_mod_cast hPos1) (by exact_mod_cast hClasses)
  have hLog2Pos : 0 < Real.log 2 := Real.log_pos (by norm_num)
  unfold DecisionProblem.quotientEntropy
  have hInvNonneg : 0 ≤ (Real.log 2)⁻¹ := by positivity
  simpa [div_eq_mul_inv] using mul_le_mul_of_nonneg_right hLog hInvNonneg

namespace TemporalBridge

open Physics.TemporalCountingGap

def StateAt (psf : PhysicalScaleFactor) (ρ : ℕ) (t : ℕ) : Type :=
  Fin (StateSpaceCardinality psf.a ρ t)

def embedState
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t1 t2 : ℕ} (h : t1 ≤ t2) :
    StateAt psf ρ t1 → StateAt psf ρ t2 := by
  intro x
  refine ⟨x.1, ?_⟩
  exact lt_of_lt_of_le x.2 (states_nondecreasing psf ρ hρ t1 t2 h)

theorem embedState_injective
    (psf : PhysicalScaleFactor) (ρ : ℕ) (hρ : 0 < ρ)
    {t1 t2 : ℕ} (h : t1 ≤ t2) :
    Function.Injective (embedState psf ρ hρ h) := by
  intro x y hxy
  exact Fin.ext (by simpa [embedState] using congrArg Fin.val hxy)

/-
Key dry-run observation:

At this point we can build state embeddings from temporal growth.
But we still cannot derive numOptClasses monotonicity from TemporalCountingGap alone,
because we need a time-indexed decision family and an Opt-compatibility hypothesis:

  ∀ s, (dp t2).Opt (embed s) = (dp t1).Opt s

TemporalCountingGap currently proves state-cardinality/cost growth, not this optimizer bridge.
-/

end TemporalBridge
end DecisionQuotient.DryRun
