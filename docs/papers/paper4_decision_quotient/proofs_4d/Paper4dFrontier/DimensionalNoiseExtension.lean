import Paper4dFrontier.FamilyAxioms
import DecisionQuotient.Tractability.Dimensional
import Mathlib.Data.Fin.Embedding
import Mathlib.Data.Fin.SuccPred

namespace Paper4dFrontier

open DecisionQuotient

def liftCoordSet {d : ℕ} (I : Finset (Fin d)) : Finset (Fin (d + 1)) :=
  I.map Fin.castSuccEmb

def truncateNoise {k d : ℕ} (s : DimensionalStateSpace k (d + 1)) : DimensionalStateSpace k d where
  state := fun i => s.state i.castSucc

def padZero {k d : ℕ} [NeZero k] (s : DimensionalStateSpace k d) : DimensionalStateSpace k (d + 1) where
  state := Fin.lastCases 0 s.state

def noiseExtendedProblem {A : Type*} {k d : ℕ}
    (dp : DecisionProblem A (DimensionalStateSpace k d)) :
    DecisionProblem A (DimensionalStateSpace k (d + 1)) where
  utility a s := dp.utility a (truncateNoise s)

theorem opt_noiseExtended_eq {A : Type*} {k d : ℕ}
    (dp : DecisionProblem A (DimensionalStateSpace k d)) (s : DimensionalStateSpace k (d + 1)) :
    (noiseExtendedProblem dp).Opt s = dp.Opt (truncateNoise s) := by
  rfl

theorem agreeOn_liftCoordSet_iff {k d : ℕ}
    (s s' : DimensionalStateSpace k (d + 1)) (I : Finset (Fin d)) :
    agreeOn s s' (liftCoordSet I) ↔ agreeOn (truncateNoise s) (truncateNoise s') I := by
  constructor
  · intro h i hi
    exact h i.castSucc (by
      exact Finset.mem_map.mpr ⟨i, hi, rfl⟩)
  · intro h j hj
    rcases Finset.mem_map.mp hj with ⟨i, hi, rfl⟩
    exact h i hi

theorem liftCoordSet_univ_eq_erase_last {d : ℕ} :
    liftCoordSet (Finset.univ : Finset (Fin d)) = Finset.univ.erase (Fin.last d) := by
  ext j
  constructor
  · intro hj
    simp
    intro hlast
    rcases Finset.mem_map.mp hj with ⟨i, _, hi⟩
    exact Fin.castSucc_ne_last i (by simpa using hi.trans hlast)
  · intro hj
    simp at hj
    rcases Fin.eq_castSucc_or_eq_last j with ⟨i, rfl⟩ | hlast
    · exact Finset.mem_map.mpr ⟨i, Finset.mem_univ i, rfl⟩
    · exact False.elim (hj hlast)

theorem isSufficient_noiseExtended_iff {A : Type*} {k d : ℕ} [NeZero k]
    (dp : DecisionProblem A (DimensionalStateSpace k d)) (I : Finset (Fin d)) :
    (noiseExtendedProblem dp).isSufficient (liftCoordSet I) ↔ dp.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    have hagree' : agreeOn (truncateNoise (padZero s)) (truncateNoise (padZero s')) I := by
      simpa [truncateNoise, padZero] using hagree
    have hEq := h (padZero s) (padZero s') ((agreeOn_liftCoordSet_iff (padZero s) (padZero s') I).2 hagree')
    simpa [padZero, truncateNoise, opt_noiseExtended_eq] using hEq
  · intro h s s' hagree
    have hEq := h (truncateNoise s) (truncateNoise s') ((agreeOn_liftCoordSet_iff s s' I).1 hagree)
    simpa [opt_noiseExtended_eq] using hEq

theorem isRelevant_noiseExtended_iff {A : Type*} {k d : ℕ} [NeZero k]
    (dp : DecisionProblem A (DimensionalStateSpace k d)) (i : Fin d) :
    (noiseExtendedProblem dp).isRelevant i.castSucc ↔ dp.isRelevant i := by
  unfold DecisionProblem.isRelevant
  constructor
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨truncateNoise s, truncateNoise s', ?_, ?_⟩
    · intro j hj
      exact hcoord j.castSucc (by
        intro h
        apply hj
        exact Fin.castSucc_injective _ h)
    · intro hEq
      exact hneq (by simpa [opt_noiseExtended_eq] using hEq)
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨padZero s, padZero s', ?_, ?_⟩
    · intro j hj
      rcases Fin.eq_castSucc_or_eq_last j with ⟨j', rfl⟩ | hlast
      · have hj' : j' ≠ i := by
          intro h
          apply hj
          exact congrArg Fin.castSucc h
        show (padZero s).state j'.castSucc = (padZero s').state j'.castSucc
        simpa [padZero] using hcoord j' hj'
      · subst j
        change (padZero s).state (Fin.last d) = (padZero s').state (Fin.last d)
        simp [padZero]
    · intro hEq
      exact hneq (by simpa [padZero, truncateNoise, opt_noiseExtended_eq] using hEq)

theorem lastCoord_irrelevant_noiseExtended {A : Type*} {k d : ℕ} [NeZero k]
    (dp : DecisionProblem A (DimensionalStateSpace k d)) :
    (noiseExtendedProblem dp).isIrrelevant (Fin.last d) := by
  rw [isIrrelevant_iff_sufficient_erase]
  have huniv : dp.isSufficient Finset.univ := by
    intro x y hagree
    have hxy : x = y := by
      ext i
      cases x
      cases y
      exact congrArg Fin.val (hagree i (by simp : i ∈ (Finset.univ : Finset (Fin d))))
    simpa [hxy]
  simpa [liftCoordSet_univ_eq_erase_last] using
    (isSufficient_noiseExtended_iff dp Finset.univ).2 huniv

end Paper4dFrontier
