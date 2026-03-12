import DecisionQuotient.Sufficiency
import DecisionQuotient.AlgorithmComplexity
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

namespace DecisionQuotient

open Classical

inductive SetCoverAction where
  | accept
  | reject
  deriving DecidableEq

structure SetCoverInstance where
  numElems : ℕ
  numSets : ℕ
  covers : Fin numElems → Fin numSets → Bool

namespace SetCoverInstance

abbrev State (inst : SetCoverInstance) := Bool × Fin inst.numElems

instance (inst : SetCoverInstance) : CoordinateSpace inst.State inst.numSets where
  Coord := fun _ => Bool
  proj := fun s i => s.1 && inst.covers s.2 i

noncomputable def toDecisionProblem (inst : SetCoverInstance) :
    DecisionProblem SetCoverAction inst.State where
  utility
    | SetCoverAction.accept, (b, _) => if b then 0 else 1
    | SetCoverAction.reject, _ => 0

def IsCover (inst : SetCoverInstance) (I : Finset (Fin inst.numSets)) : Prop :=
  ∀ u : Fin inst.numElems, ∃ i ∈ I, inst.covers u i = true

/-- Explicit encoding size for the finite set-cover instance represented by the
cover-incidence matrix. This is the right size notion for the local counted
runtime model; it avoids relying on `sizeOf` for function-valued fields. -/
def inputSize (inst : SetCoverInstance) : ℕ :=
  inst.numElems * inst.numSets + inst.numElems + inst.numSets + 1

theorem opt_false_tag (inst : SetCoverInstance) (u : Fin inst.numElems) :
    (inst.toDecisionProblem).Opt (false, u) = {SetCoverAction.accept} := by
  ext a
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq, Set.mem_singleton_iff]
  constructor
  · intro h
    cases a with
    | accept => rfl
    | reject =>
        have h1 := h SetCoverAction.accept
        simp [toDecisionProblem] at h1
        linarith
  · intro h
    subst h
    intro a'
    cases a' <;> simp [toDecisionProblem]

theorem opt_true_tag (inst : SetCoverInstance) (u : Fin inst.numElems) :
    (inst.toDecisionProblem).Opt (true, u) = {SetCoverAction.accept, SetCoverAction.reject} := by
  ext a
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq,
    Set.mem_insert_iff, Set.mem_singleton_iff]
  constructor
  · intro _
    cases a <;> simp
  · intro _
    intro a'
    cases a <;> cases a' <;> simp [toDecisionProblem]

theorem sufficiency_iff_cover (inst : SetCoverInstance)
    (I : Finset (Fin inst.numSets)) :
    (inst.toDecisionProblem).isSufficient I ↔ inst.IsCover I := by
  constructor
  · intro hI u
    by_contra hNoCover
    push_neg at hNoCover
    have hagree : agreeOn (false, u) (true, u) I := by
      intro i hi
      have hfalse : inst.covers u i = false := by
        cases hcov : inst.covers u i with
        | true => exact False.elim (hNoCover i hi hcov)
        | false => simpa using hcov
      simp [CoordinateSpace.proj, hfalse]
    have heq := hI (false, u) (true, u) hagree
    have hmem : SetCoverAction.reject ∈ (inst.toDecisionProblem).Opt (true, u) := by
      simp [opt_true_tag]
    have hcontra : SetCoverAction.reject ∈ (inst.toDecisionProblem).Opt (false, u) := by
      simpa [heq] using hmem
    simp [opt_false_tag] at hcontra
  · intro hCover s s' hagree
    rcases s with ⟨b, u⟩
    rcases s' with ⟨b', u'⟩
    by_cases hb : b = b'
    · subst hb
      cases b <;> simp [opt_false_tag, opt_true_tag]
    · cases b <;> cases b' <;> try contradiction
      · exfalso
        rcases hCover u' with ⟨i, hiI, hcov⟩
        have hEq := hagree i hiI
        simp [CoordinateSpace.proj, hcov] at hEq
      · exfalso
        rcases hCover u with ⟨i, hiI, hcov⟩
        have hEq := hagree i hiI
        simp [CoordinateSpace.proj, hcov] at hEq

theorem min_sufficient_iff_set_cover (inst : SetCoverInstance) (k : ℕ) :
    (∃ I : Finset (Fin inst.numSets), I.card ≤ k ∧ (inst.toDecisionProblem).isSufficient I) ↔
      (∃ I : Finset (Fin inst.numSets), I.card ≤ k ∧ inst.IsCover I) := by
  constructor
  · rintro ⟨I, hcard, hI⟩
    exact ⟨I, hcard, (sufficiency_iff_cover inst I).1 hI⟩
  · rintro ⟨I, hcard, hI⟩
    exact ⟨I, hcard, (sufficiency_iff_cover inst I).2 hI⟩

def FactorApproxSetCoverSolver := ∀ inst : SetCoverInstance, Finset (Fin inst.numSets)

def IsFactorApproxSetCoverSolver (ρ : ℕ) (solver : FactorApproxSetCoverSolver) : Prop :=
  ∀ inst : SetCoverInstance,
    inst.IsCover (solver inst) ∧
      ∀ J : Finset (Fin inst.numSets), inst.IsCover J → (solver inst).card ≤ ρ * J.card

def FactorApproxMinSuffSolver := ∀ inst : SetCoverInstance, Finset (Fin inst.numSets)

def IsFactorApproxMinSuffSolver (ρ : ℕ) (solver : FactorApproxMinSuffSolver) : Prop :=
  ∀ inst : SetCoverInstance,
    (inst.toDecisionProblem).isSufficient (solver inst) ∧
      ∀ J : Finset (Fin inst.numSets),
        (inst.toDecisionProblem).isSufficient J → (solver inst).card ≤ ρ * J.card

def RatioSetCoverSolver := ∀ inst : SetCoverInstance, Finset (Fin inst.numSets)

def IsRatioSetCoverSolver (ρ : SetCoverInstance → ℕ) (solver : RatioSetCoverSolver) : Prop :=
  ∀ inst : SetCoverInstance,
    inst.IsCover (solver inst) ∧
      ∀ J : Finset (Fin inst.numSets), inst.IsCover J → (solver inst).card ≤ ρ inst * J.card

def RatioMinSuffSolver := ∀ inst : SetCoverInstance, Finset (Fin inst.numSets)

def IsRatioMinSuffSolver (ρ : SetCoverInstance → ℕ) (solver : RatioMinSuffSolver) : Prop :=
  ∀ inst : SetCoverInstance,
    (inst.toDecisionProblem).isSufficient (solver inst) ∧
      ∀ J : Finset (Fin inst.numSets),
        (inst.toDecisionProblem).isSufficient J → (solver inst).card ≤ ρ inst * J.card

theorem setCoverSolver_of_minSuffSolver (ρ : ℕ)
    (solver : FactorApproxMinSuffSolver)
    (hsolver : IsFactorApproxMinSuffSolver ρ solver) :
    IsFactorApproxSetCoverSolver ρ solver := by
  intro inst
  rcases hsolver inst with ⟨hout, happrox⟩
  refine ⟨(sufficiency_iff_cover inst (solver inst)).1 hout, ?_⟩
  intro J hJ
  exact happrox J ((sufficiency_iff_cover inst J).2 hJ)

theorem minSuffSolver_of_setCoverSolver (ρ : ℕ)
    (solver : FactorApproxSetCoverSolver)
    (hsolver : IsFactorApproxSetCoverSolver ρ solver) :
    IsFactorApproxMinSuffSolver ρ solver := by
  intro inst
  rcases hsolver inst with ⟨hout, happrox⟩
  refine ⟨(sufficiency_iff_cover inst (solver inst)).2 hout, ?_⟩
  intro J hJ
  exact happrox J ((sufficiency_iff_cover inst J).1 hJ)

theorem ratioSetCoverSolver_of_ratioMinSuffSolver
    (ρ : SetCoverInstance → ℕ)
    (solver : RatioMinSuffSolver)
    (hsolver : IsRatioMinSuffSolver ρ solver) :
    IsRatioSetCoverSolver ρ solver := by
  intro inst
  rcases hsolver inst with ⟨hout, happrox⟩
  refine ⟨(sufficiency_iff_cover inst (solver inst)).1 hout, ?_⟩
  intro J hJ
  exact happrox J ((sufficiency_iff_cover inst J).2 hJ)

theorem ratioMinSuffSolver_of_ratioSetCoverSolver
    (ρ : SetCoverInstance → ℕ)
    (solver : RatioSetCoverSolver)
    (hsolver : IsRatioSetCoverSolver ρ solver) :
    IsRatioMinSuffSolver ρ solver := by
  intro inst
  rcases hsolver inst with ⟨hout, happrox⟩
  refine ⟨(sufficiency_iff_cover inst (solver inst)).2 hout, ?_⟩
  intro J hJ
  exact happrox J ((sufficiency_iff_cover inst J).1 hJ)

/-! ## Counted Polynomial-Time Approximation Interfaces

These structures upgrade the semantic approximation interfaces above with an
explicit counted runtime bound. This is weaker than a TM witness but strong
enough to state honest algorithmic transfer theorems inside the current library.
-/

structure CountedFactorApproxSetCoverSolver (ρ : ℕ) where
  run : (inst : SetCoverInstance) → Counted (Finset (Fin inst.numSets))
  cover_ok : ∀ inst, IsCover inst (run inst).result
  approx_ok :
    ∀ inst (J : Finset (Fin inst.numSets)),
      IsCover inst J → (run inst).result.card ≤ ρ * J.card
  polytime :
    ∃ c k : ℕ, ∀ inst, (run inst).steps ≤ c * (inputSize inst) ^ k + c

structure CountedFactorApproxMinSuffSolver (ρ : ℕ) where
  run : (inst : SetCoverInstance) → Counted (Finset (Fin inst.numSets))
  sufficient_ok : ∀ inst, (toDecisionProblem inst).isSufficient (run inst).result
  approx_ok :
    ∀ inst (J : Finset (Fin inst.numSets)),
      (toDecisionProblem inst).isSufficient J → (run inst).result.card ≤ ρ * J.card
  polytime :
    ∃ c k : ℕ, ∀ inst, (run inst).steps ≤ c * (inputSize inst) ^ k + c

structure CountedRatioSetCoverSolver (ρ : SetCoverInstance → ℕ) where
  run : (inst : SetCoverInstance) → Counted (Finset (Fin inst.numSets))
  cover_ok : ∀ inst, IsCover inst (run inst).result
  approx_ok :
    ∀ inst (J : Finset (Fin inst.numSets)),
      IsCover inst J → (run inst).result.card ≤ ρ inst * J.card
  polytime :
    ∃ c k : ℕ, ∀ inst, (run inst).steps ≤ c * (inputSize inst) ^ k + c

structure CountedRatioMinSuffSolver (ρ : SetCoverInstance → ℕ) where
  run : (inst : SetCoverInstance) → Counted (Finset (Fin inst.numSets))
  sufficient_ok : ∀ inst, (toDecisionProblem inst).isSufficient (run inst).result
  approx_ok :
    ∀ inst (J : Finset (Fin inst.numSets)),
      (toDecisionProblem inst).isSufficient J → (run inst).result.card ≤ ρ inst * J.card
  polytime :
    ∃ c k : ℕ, ∀ inst, (run inst).steps ≤ c * (inputSize inst) ^ k + c

def countedSetCoverSolver_of_countedMinSuffSolver
    (ρ : ℕ)
    (solver : CountedFactorApproxMinSuffSolver ρ) :
    CountedFactorApproxSetCoverSolver ρ where
  run := solver.run
  cover_ok := by
    intro inst
    exact (sufficiency_iff_cover inst (solver.run inst).result).1 (solver.sufficient_ok inst)
  approx_ok := by
    intro inst J hJ
    exact solver.approx_ok inst J ((sufficiency_iff_cover inst J).2 hJ)
  polytime := solver.polytime

def countedMinSuffSolver_of_countedSetCoverSolver
    (ρ : ℕ)
    (solver : CountedFactorApproxSetCoverSolver ρ) :
    CountedFactorApproxMinSuffSolver ρ where
  run := solver.run
  sufficient_ok := by
    intro inst
    exact (sufficiency_iff_cover inst (solver.run inst).result).2 (solver.cover_ok inst)
  approx_ok := by
    intro inst J hJ
    exact solver.approx_ok inst J ((sufficiency_iff_cover inst J).1 hJ)
  polytime := solver.polytime

def countedRatioSetCoverSolver_of_countedMinSuffSolver
    (ρ : SetCoverInstance → ℕ)
    (solver : CountedRatioMinSuffSolver ρ) :
    CountedRatioSetCoverSolver ρ where
  run := solver.run
  cover_ok := by
    intro inst
    exact (sufficiency_iff_cover inst (solver.run inst).result).1 (solver.sufficient_ok inst)
  approx_ok := by
    intro inst J hJ
    exact solver.approx_ok inst J ((sufficiency_iff_cover inst J).2 hJ)
  polytime := solver.polytime

def countedRatioMinSuffSolver_of_countedSetCoverSolver
    (ρ : SetCoverInstance → ℕ)
    (solver : CountedRatioSetCoverSolver ρ) :
    CountedRatioMinSuffSolver ρ where
  run := solver.run
  sufficient_ok := by
    intro inst
    exact (sufficiency_iff_cover inst (solver.run inst).result).2 (solver.cover_ok inst)
  approx_ok := by
    intro inst J hJ
    exact solver.approx_ok inst J ((sufficiency_iff_cover inst J).1 hJ)
  polytime := solver.polytime

end SetCoverInstance

end DecisionQuotient
