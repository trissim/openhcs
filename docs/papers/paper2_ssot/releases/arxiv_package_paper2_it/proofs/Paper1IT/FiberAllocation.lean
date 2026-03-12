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

/-- The subset obtained by taking the fiberwise maximizing subset at each fiber under the given
allocation. -/
noncomputable def allocatedOptimalSubset {K O : Nat} (μ : FiniteSource K)
    (obs : Fin K → Fin O) (ℓ : Fin O → Nat) : Finset (Fin K) :=
  Finset.univ.filter (fun v => v ∈ optimalFiberSubset μ obs (2 ^ ℓ (obs v)) (obs v))

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

theorem allocatedOptimalSubset_feasible {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (ℓ : Fin O → Nat) :
    ∀ o, (fiberSlice obs (allocatedOptimalSubset μ obs ℓ) o).card ≤ 2 ^ ℓ o := by
  intro o
  have hslice : fiberSlice obs (allocatedOptimalSubset μ obs ℓ) o = optimalFiberSubset μ obs (2 ^ ℓ o) o := by
    ext v
    constructor
    · intro hv
      simp [fiberSlice, allocatedOptimalSubset] at hv
      rcases hv with ⟨hmem, hobs⟩
      simpa [hobs] using hmem
    · intro hv
      have hsource : v ∈ sourceFiber obs o := optimalFiberSubset_subset_sourceFiber μ obs (2 ^ ℓ o) o hv
      have hobs : obs v = o := by simpa [sourceFiber] using hsource
      simp [fiberSlice, allocatedOptimalSubset, hv, hobs]
  rw [hslice]
  exact optimalFiberSubset_card_le μ obs (2 ^ ℓ o) o

theorem subsetMass_allocatedOptimalSubset_eq_allocatedRecoverableMass {K O : Nat}
    (μ : FiniteSource K) (obs : Fin K → Fin O) (ℓ : Fin O → Nat) :
    subsetMass μ (allocatedOptimalSubset μ obs ℓ) = allocatedRecoverableMass μ obs ℓ := by
  rw [← sum_fiberSlice_subsetMass_eq_subsetMass μ obs (allocatedOptimalSubset μ obs ℓ)]
  unfold allocatedRecoverableMass
  refine Finset.sum_congr rfl ?_
  intro o ho
  have hslice : fiberSlice obs (allocatedOptimalSubset μ obs ℓ) o = optimalFiberSubset μ obs (2 ^ ℓ o) o := by
    ext v
    constructor
    · intro hv
      simp [fiberSlice, allocatedOptimalSubset] at hv
      rcases hv with ⟨hmem, hobs⟩
      simpa [hobs] using hmem
    · intro hv
      have hsource : v ∈ sourceFiber obs o := optimalFiberSubset_subset_sourceFiber μ obs (2 ^ ℓ o) o hv
      have hobs : obs v = o := by simpa [sourceFiber] using hsource
      simp [fiberSlice, allocatedOptimalSubset, hv, hobs]
  rw [hslice]
  exact subsetMass_optimalFiberSubset_eq_fiberTopMass μ obs (2 ^ ℓ o) o

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

theorem allocatedOptimalSubset_attains_allocatedRecoverableMass {K O : Nat}
    (μ : FiniteSource K) (obs : Fin K → Fin O) (ℓ : Fin O → Nat) :
    (∀ o, (fiberSlice obs (allocatedOptimalSubset μ obs ℓ) o).card ≤ 2 ^ ℓ o)
    ∧ subsetMass μ (allocatedOptimalSubset μ obs ℓ) = allocatedRecoverableMass μ obs ℓ := by
  exact ⟨
    allocatedOptimalSubset_feasible μ obs ℓ,
    subsetMass_allocatedOptimalSubset_eq_allocatedRecoverableMass μ obs ℓ
  ⟩

end ObserverModel
