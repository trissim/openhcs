/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/SupportExpansion.lean

  Finite support-shell semantics for the local certified action family. This
  models the round-wise union of shells used by the runtime support-expansion
  strategy.
-/
import Mathlib.Data.Finset.Max
import Mathlib.Tactic

namespace DecisionQuotient
namespace Tractability
namespace SupportExpansion

open Classical

/-- Number of coarser shells added to the active round. -/
def coarserShellCount (roundIndex supportExpansionLevel : ℕ) : ℕ :=
  min roundIndex supportExpansionLevel

/-- Shell levels carried by the runtime support-expansion family. -/
noncomputable def supportShellLevels (roundIndex supportExpansionLevel : ℕ) : Finset ℕ :=
  (Finset.range (coarserShellCount roundIndex supportExpansionLevel + 1)).image
    (fun offset => roundIndex - offset)

/-- The merged runtime family keeps one noop plus 12 local actions per shell. -/
noncomputable def mergedActionCount (roundIndex supportExpansionLevel : ℕ) : ℕ :=
  1 + 12 * (supportShellLevels roundIndex supportExpansionLevel).card

theorem roundIndex_mem_supportShellLevels (roundIndex supportExpansionLevel : ℕ) :
    roundIndex ∈ supportShellLevels roundIndex supportExpansionLevel := by
  unfold supportShellLevels
  refine Finset.mem_image.mpr ?_
  refine ⟨0, by simp [coarserShellCount], by simp⟩

theorem coarsestShell_mem_supportShellLevels (roundIndex supportExpansionLevel : ℕ) :
    roundIndex - coarserShellCount roundIndex supportExpansionLevel
      ∈ supportShellLevels roundIndex supportExpansionLevel := by
  unfold supportShellLevels
  refine Finset.mem_image.mpr ?_
  refine ⟨coarserShellCount roundIndex supportExpansionLevel, by simp, rfl⟩

theorem supportShellLevels_card (roundIndex supportExpansionLevel : ℕ) :
    (supportShellLevels roundIndex supportExpansionLevel).card =
      coarserShellCount roundIndex supportExpansionLevel + 1 := by
  classical
  unfold supportShellLevels
  calc
    (Finset.image (fun offset => roundIndex - offset)
        (Finset.range (coarserShellCount roundIndex supportExpansionLevel + 1))).card
        = (Finset.range (coarserShellCount roundIndex supportExpansionLevel + 1)).card := by
            refine Finset.card_image_of_injOn ?_
            intro a ha
            intro b hb
            intro hEq
            have haRange : a < coarserShellCount roundIndex supportExpansionLevel + 1 := by
              simpa using ha
            have hbRange : b < coarserShellCount roundIndex supportExpansionLevel + 1 := by
              simpa using hb
            have ha' : a ≤ roundIndex := by
              have hMin : coarserShellCount roundIndex supportExpansionLevel ≤ roundIndex := by
                unfold coarserShellCount
                exact min_le_left _ _
              omega
            have hb' : b ≤ roundIndex := by
              have hMin : coarserShellCount roundIndex supportExpansionLevel ≤ roundIndex := by
                unfold coarserShellCount
                exact min_le_left _ _
              omega
            have hAdd : roundIndex = (roundIndex - b) + a := by
              simpa [Nat.sub_add_cancel ha'] using congrArg (fun t => t + a) hEq
            have hRound : roundIndex = (roundIndex - b) + b := by
              exact (Nat.sub_add_cancel hb').symm
            have hSame : (roundIndex - b) + a = (roundIndex - b) + b := by
              calc
                (roundIndex - b) + a = roundIndex := by exact hAdd.symm
                _ = (roundIndex - b) + b := by exact hRound
            exact Nat.add_left_cancel hSame
    _ = coarserShellCount roundIndex supportExpansionLevel + 1 := by
          simp

theorem supportShellLevels_monotone {roundIndex e1 e2 : ℕ}
    (hExp : e1 ≤ e2) :
    supportShellLevels roundIndex e1 ⊆ supportShellLevels roundIndex e2 := by
  intro shell hs
  unfold supportShellLevels at hs ⊢
  rcases Finset.mem_image.mp hs with ⟨offset, hOffset, rfl⟩
  refine Finset.mem_image.mpr ?_
  refine ⟨offset, ?_, rfl⟩
  simp only [Finset.mem_range] at hOffset ⊢
  unfold coarserShellCount at hOffset ⊢
  omega

theorem supportShellLevels_max' (roundIndex supportExpansionLevel : ℕ) :
    (supportShellLevels roundIndex supportExpansionLevel).max'
      ⟨roundIndex, roundIndex_mem_supportShellLevels roundIndex supportExpansionLevel⟩ = roundIndex := by
  refine le_antisymm ?_ ?_
  · have hMaxMem :
        (supportShellLevels roundIndex supportExpansionLevel).max'
          ⟨roundIndex, roundIndex_mem_supportShellLevels roundIndex supportExpansionLevel⟩
          ∈ supportShellLevels roundIndex supportExpansionLevel :=
      Finset.max'_mem _ _
    rcases Finset.mem_image.mp hMaxMem with ⟨offset, hOffset, hEq⟩
    have hLe : roundIndex - offset ≤ roundIndex := by
      omega
    simpa [hEq] using hLe
  · exact Finset.le_max' _ _ (roundIndex_mem_supportShellLevels roundIndex supportExpansionLevel)

theorem mergedActionCount_eq (roundIndex supportExpansionLevel : ℕ) :
    mergedActionCount roundIndex supportExpansionLevel =
      1 + 12 * (coarserShellCount roundIndex supportExpansionLevel + 1) := by
  unfold mergedActionCount
  rw [supportShellLevels_card]

end SupportExpansion
end Tractability
end DecisionQuotient
