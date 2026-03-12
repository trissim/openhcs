import Paper1IT.FiberRateDistortion

namespace ObserverModel

theorem fiberCandidates_mono {K O : Nat} (obs : Fin K → Fin O) {T T' : Nat} (hTT' : T ≤ T')
    (o : Fin O) :
    fiberCandidates obs T o ⊆ fiberCandidates obs T' o := by
  intro S hS
  simp [fiberCandidates] at hS ⊢
  exact ⟨hS.1, hS.2.trans hTT'⟩

theorem fiberTopMass_mono {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    {T T' : Nat} (hTT' : T ≤ T') (o : Fin O) :
    fiberTopMass μ obs T o ≤ fiberTopMass μ obs T' o := by
  unfold fiberTopMass
  have hmem := Finset.max'_mem ((fiberCandidates obs T o).image (subsetMass μ))
    ((fiberCandidates_nonempty obs T o).image _)
  rcases Finset.mem_image.mp hmem with ⟨S, hS, hEq⟩
  rw [← hEq]
  exact Finset.le_max' _ _ <| by
    apply Finset.mem_image.mpr
    exact ⟨S, fiberCandidates_mono obs hTT' o hS, rfl⟩

/-- Total recoverable source mass under a fiberwise bit allocation. -/
noncomputable def allocatedRecoverableMass {K O : Nat} (μ : FiniteSource K)
    (obs : Fin K → Fin O) (ℓ : Fin O → Nat) : ℝ :=
  Finset.univ.sum (fun o : Fin O => fiberTopMass μ obs (2 ^ ℓ o) o)

/-- Distortion induced by the optimal exact-on subset under a fiberwise bit allocation. -/
noncomputable def allocatedDistortion {K O : Nat} (μ : FiniteSource K)
    (obs : Fin K → Fin O) (ℓ : Fin O → Nat) : ℝ :=
  1 - allocatedRecoverableMass μ obs ℓ

theorem allocatedRecoverableMass_mono {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    {ℓ ℓ' : Fin O → Nat} (hℓ : ∀ o, ℓ o ≤ ℓ' o) :
    allocatedRecoverableMass μ obs ℓ ≤ allocatedRecoverableMass μ obs ℓ' := by
  unfold allocatedRecoverableMass
  refine Finset.sum_le_sum ?_
  intro o ho
  exact fiberTopMass_mono μ obs (pow_le_pow_right₀ (by norm_num : 1 ≤ 2) (hℓ o)) o

theorem allocatedDistortion_anti {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    {ℓ ℓ' : Fin O → Nat} (hℓ : ∀ o, ℓ o ≤ ℓ' o) :
    allocatedDistortion μ obs ℓ' ≤ allocatedDistortion μ obs ℓ := by
  unfold allocatedDistortion
  have hmass := allocatedRecoverableMass_mono μ obs hℓ
  linarith

theorem allocatedDistortion_eq_one_sub_allocatedRecoverableMass
    {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O) (ℓ : Fin O → Nat) :
    allocatedDistortion μ obs ℓ = 1 - allocatedRecoverableMass μ obs ℓ := rfl

theorem zero_allocation_distortion_eq_one_sub_zero_recoverableMass
    {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O) :
    allocatedDistortion μ obs (fun _ => 0) =
      1 - Finset.univ.sum (fun o : Fin O => fiberTopMass μ obs 1 o) := by
  simp [allocatedDistortion, allocatedRecoverableMass]

end ObserverModel
