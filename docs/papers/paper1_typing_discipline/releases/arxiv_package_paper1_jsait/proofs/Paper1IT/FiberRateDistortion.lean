import Paper1IT.ProbabilisticFinite
import Paper1IT.ObserverTagModel
import Mathlib.Data.Finset.Powerset

namespace ObserverModel

/-- Source mass carried by a finite subset of states. -/
noncomputable def subsetMass {K : Nat} (μ : FiniteSource K) (S : Finset (Fin K)) : ℝ :=
  S.sum μ.pmf

/-- The slice of a subset inside one observation fiber. -/
def fiberSlice {K O : Nat} (obs : Fin K → Fin O) (S : Finset (Fin K)) (o : Fin O) : Finset (Fin K) :=
  S.filter (fun v => obs v = o)

/-- A subset is feasible under a uniform per-fiber tag alphabet of size `T` when every fiber slice
has cardinality at most `T`. -/
def FiberBudgetFeasible {K O : Nat} (obs : Fin K → Fin O) (T : Nat) (S : Finset (Fin K)) : Prop :=
  ∀ o, (fiberSlice obs S o).card ≤ T

/-- Candidate exactly recoverable subsets inside one observation fiber. -/
def fiberCandidates {K O : Nat} (obs : Fin K → Fin O) (T : Nat) (o : Fin O) : Finset (Finset (Fin K)) :=
  ((sourceFiber obs o).powerset.filter fun S => S.card ≤ T)

theorem empty_mem_fiberCandidates {K O : Nat} (obs : Fin K → Fin O) (T : Nat) (o : Fin O) :
    (∅ : Finset (Fin K)) ∈ fiberCandidates obs T o := by
  simp [fiberCandidates]

theorem fiberCandidates_nonempty {K O : Nat} (obs : Fin K → Fin O) (T : Nat) (o : Fin O) :
    (fiberCandidates obs T o).Nonempty := by
  exact ⟨∅, empty_mem_fiberCandidates obs T o⟩

/-- The optimal source mass recoverable inside one fiber with at most `T` tags. -/
noncomputable def fiberTopMass {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (T : Nat) (o : Fin O) : ℝ :=
  ((fiberCandidates obs T o).image (subsetMass μ)).max' ((fiberCandidates_nonempty obs T o).image _)

/-- A maximizing subset inside one fiber under tag budget `T`. -/
noncomputable def optimalFiberSubset {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (T : Nat) (o : Fin O) : Finset (Fin K) :=
  Classical.choose <| Finset.mem_image.mp <|
    Finset.max'_mem ((fiberCandidates obs T o).image (subsetMass μ))
      ((fiberCandidates_nonempty obs T o).image _)

/-- The chosen maximizing subset is a valid candidate. -/
theorem optimalFiberSubset_mem_candidates {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (T : Nat) (o : Fin O) :
    optimalFiberSubset μ obs T o ∈ fiberCandidates obs T o := by
  unfold optimalFiberSubset
  exact (Classical.choose_spec <| Finset.mem_image.mp <|
    Finset.max'_mem ((fiberCandidates obs T o).image (subsetMass μ))
      ((fiberCandidates_nonempty obs T o).image _)).1

/-- The chosen maximizing subset attains the fiberwise optimum. -/
theorem subsetMass_optimalFiberSubset_eq_fiberTopMass {K O : Nat} (μ : FiniteSource K)
    (obs : Fin K → Fin O) (T : Nat) (o : Fin O) :
    subsetMass μ (optimalFiberSubset μ obs T o) = fiberTopMass μ obs T o := by
  unfold optimalFiberSubset fiberTopMass
  exact (Classical.choose_spec <| Finset.mem_image.mp <|
    Finset.max'_mem ((fiberCandidates obs T o).image (subsetMass μ))
      ((fiberCandidates_nonempty obs T o).image _)).2

/-- The chosen maximizing subset lies inside the target fiber. -/
theorem optimalFiberSubset_subset_sourceFiber {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (T : Nat) (o : Fin O) :
    optimalFiberSubset μ obs T o ⊆ sourceFiber obs o := by
  have hmem := optimalFiberSubset_mem_candidates μ obs T o
  simp [fiberCandidates] at hmem
  exact hmem.1

/-- The chosen maximizing subset has cardinality at most `T`. -/
theorem optimalFiberSubset_card_le {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (T : Nat) (o : Fin O) :
    (optimalFiberSubset μ obs T o).card ≤ T := by
  have hmem := optimalFiberSubset_mem_candidates μ obs T o
  simp [fiberCandidates] at hmem
  exact hmem.2

theorem subsetMass_nonneg {K : Nat} (μ : FiniteSource K) (S : Finset (Fin K)) :
    0 ≤ subsetMass μ S := by
  unfold subsetMass
  exact Finset.sum_nonneg (by intro v hv; exact μ.nonneg v)

theorem fiberSlice_subset_sourceFiber {K O : Nat} (obs : Fin K → Fin O) (S : Finset (Fin K)) (o : Fin O) :
    fiberSlice obs S o ⊆ sourceFiber obs o := by
  intro v hv
  simp [fiberSlice, sourceFiber] at hv ⊢
  exact hv.2

theorem fiberSlice_mem_fiberCandidates {K O : Nat} (obs : Fin K → Fin O) (T : Nat)
    (S : Finset (Fin K)) (hfeas : FiberBudgetFeasible obs T S) (o : Fin O) :
    fiberSlice obs S o ∈ fiberCandidates obs T o := by
  simp [fiberCandidates, fiberSlice_subset_sourceFiber, hfeas o]

theorem subsetMass_fiberSlice_le_fiberTopMass {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (T : Nat) (S : Finset (Fin K)) (hfeas : FiberBudgetFeasible obs T S) (o : Fin O) :
    subsetMass μ (fiberSlice obs S o) ≤ fiberTopMass μ obs T o := by
  unfold fiberTopMass
  exact Finset.le_max' _ _ <| by
    apply Finset.mem_image.mpr
    exact ⟨fiberSlice obs S o, fiberSlice_mem_fiberCandidates obs T S hfeas o, rfl⟩

theorem subsetMass_fiberSlice_eq_filter_indicator
    {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O) (S : Finset (Fin K)) (o : Fin O) :
    subsetMass μ (fiberSlice obs S o) = ∑ a ∈ S, if obs a = o then μ.pmf a else 0 := by
  unfold subsetMass fiberSlice
  simp [Finset.sum_filter]

theorem sum_fiberSlice_subsetMass_eq_subsetMass
    {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O) (S : Finset (Fin K)) :
    Finset.univ.sum (fun o : Fin O => subsetMass μ (fiberSlice obs S o)) = subsetMass μ S := by
  calc
    Finset.univ.sum (fun o : Fin O => subsetMass μ (fiberSlice obs S o))
        = Finset.univ.sum (fun o : Fin O =>
            ∑ a ∈ S, if obs a = o then μ.pmf a else 0) := by
              refine Finset.sum_congr rfl ?_
              intro o ho
              exact subsetMass_fiberSlice_eq_filter_indicator μ obs S o
    _ = ∑ a ∈ S, ∑ o : Fin O, if obs a = o then μ.pmf a else 0 := by
          rw [Finset.sum_comm]
    _ = ∑ a ∈ S, μ.pmf a := by
          refine Finset.sum_congr rfl ?_
          intro a ha
          rw [Finset.sum_eq_single (obs a)]
          · simp
          · intro o ho hne
            simp
            intro hEq
            exact (hne hEq.symm).elim
          · intro hmem
            simp at hmem
    _ = subsetMass μ S := by
          rfl

/-- The exact finite optimum under a uniform per-fiber tag alphabet is the sum of the fiberwise
optima. -/
noncomputable def optimalFeasibleMass {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (T : Nat) : ℝ :=
  Finset.univ.sum (fun o : Fin O => fiberTopMass μ obs T o)

/-- The subset obtained by taking each fiberwise maximizing subset at its observed fiber. -/
noncomputable def optimalSubset {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (T : Nat) : Finset (Fin K) :=
  Finset.univ.filter (fun v => v ∈ optimalFiberSubset μ obs T (obs v))

theorem fiberSlice_optimalSubset_eq_optimalFiberSubset {K O : Nat} (μ : FiniteSource K)
    (obs : Fin K → Fin O) (T : Nat) (o : Fin O) :
    fiberSlice obs (optimalSubset μ obs T) o = optimalFiberSubset μ obs T o := by
  ext v
  constructor
  · intro hv
    simp [fiberSlice, optimalSubset] at hv
    rcases hv with ⟨hmem, hobs⟩
    simpa [hobs] using hmem
  · intro hv
    have hsource : v ∈ sourceFiber obs o := optimalFiberSubset_subset_sourceFiber μ obs T o hv
    have hobs : obs v = o := by simpa [sourceFiber] using hsource
    simp [fiberSlice, optimalSubset, hv, hobs]

theorem optimalSubset_feasible {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O) (T : Nat) :
    FiberBudgetFeasible obs T (optimalSubset μ obs T) := by
  intro o
  rw [fiberSlice_optimalSubset_eq_optimalFiberSubset μ obs T o]
  exact optimalFiberSubset_card_le μ obs T o

theorem subsetMass_optimalSubset_eq_optimalFeasibleMass {K O : Nat} (μ : FiniteSource K)
    (obs : Fin K → Fin O) (T : Nat) :
    subsetMass μ (optimalSubset μ obs T) = optimalFeasibleMass μ obs T := by
  rw [← sum_fiberSlice_subsetMass_eq_subsetMass μ obs (optimalSubset μ obs T)]
  unfold optimalFeasibleMass
  refine Finset.sum_congr rfl ?_
  intro o ho
  rw [fiberSlice_optimalSubset_eq_optimalFiberSubset μ obs T o]
  exact subsetMass_optimalFiberSubset_eq_fiberTopMass μ obs T o

theorem optimalSubset_attains_optimalFeasibleMass {K O : Nat} (μ : FiniteSource K)
    (obs : Fin K → Fin O) (T : Nat) :
    FiberBudgetFeasible obs T (optimalSubset μ obs T)
    ∧ subsetMass μ (optimalSubset μ obs T) = optimalFeasibleMass μ obs T := by
  exact ⟨optimalSubset_feasible μ obs T, subsetMass_optimalSubset_eq_optimalFeasibleMass μ obs T⟩

theorem feasible_subsetMass_le_optimalFeasibleMass
    {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O) (T : Nat)
    (S : Finset (Fin K)) (hfeas : FiberBudgetFeasible obs T S) :
    subsetMass μ S ≤ optimalFeasibleMass μ obs T := by
  rw [← sum_fiberSlice_subsetMass_eq_subsetMass μ obs S]
  unfold optimalFeasibleMass
  refine Finset.sum_le_sum ?_
  intro o ho
  exact subsetMass_fiberSlice_le_fiberTopMass μ obs T S hfeas o

/-- Distortion induced by exact recovery on a designated subset of states. -/
noncomputable def subsetDistortion {K : Nat} (μ : FiniteSource K) (S : Finset (Fin K)) : ℝ :=
  1 - subsetMass μ S

theorem optimal_distortion_lower_bound
    {K O : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O) (T : Nat)
    (S : Finset (Fin K)) (hfeas : FiberBudgetFeasible obs T S) :
    1 - optimalFeasibleMass μ obs T ≤ subsetDistortion μ S := by
  unfold subsetDistortion
  have hmass : subsetMass μ S ≤ optimalFeasibleMass μ obs T :=
    feasible_subsetMass_le_optimalFeasibleMass μ obs T S hfeas
  linarith

theorem exactOn_implies_fiberBudgetFeasible
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {S : Finset (Fin K)}
    (hexact : ExactOn obs tag decode S) :
    FiberBudgetFeasible obs T S := by
  intro o
  have hexactSlice : ExactOn obs tag decode (fiberSlice obs S o) := by
    intro v hv
    have hv' : v ∈ S ∧ obs v = o := by simpa [fiberSlice] using hv
    have hvS : v ∈ S := hv'.1
    exact hexact v hvS
  have hclique : IsClique obs (fiberSlice obs S o) := by
    intro v w hv hw hne
    simp [fiberSlice] at hv hw
    exact hv.2.trans hw.2.symm
  exact exactOn_clique_card_le_tag_alphabet obs tag decode hexactSlice hclique

theorem exactOn_distortion_lower_bound
    {K O T : Nat} (μ : FiniteSource K) (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) (decode : Fin O → Fin T → Option (Fin K))
    {S : Finset (Fin K)} (hexact : ExactOn obs tag decode S) :
    1 - optimalFeasibleMass μ obs T ≤ subsetDistortion μ S := by
  exact optimal_distortion_lower_bound μ obs T S
    (exactOn_implies_fiberBudgetFeasible obs tag decode hexact)

end ObserverModel
