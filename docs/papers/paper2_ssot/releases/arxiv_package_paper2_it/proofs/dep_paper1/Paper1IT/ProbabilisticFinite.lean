import Paper1IT.FiniteSource
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Data.Real.Basic

namespace ObserverModel

def sourceFiber {K Z : Nat} (κ : Fin K → Fin Z) (z : Fin Z) : Finset (Fin K) :=
  Finset.univ.filter (fun v => κ v = z)

theorem sum_sourceFiber_sum_eq_univ_sum
    {K Z : Nat}
    (κ : Fin K → Fin Z)
    (f : Fin K → ℝ) :
    Finset.univ.sum (fun z : Fin Z => (sourceFiber κ z).sum f) = Finset.univ.sum f := by
  classical
  unfold sourceFiber
  calc
    Finset.univ.sum (fun z : Fin Z => (Finset.univ.filter (fun v : Fin K => κ v = z)).sum f)
        = Finset.univ.sum (fun z : Fin Z =>
            Finset.univ.sum (fun v : Fin K => if κ v = z then f v else 0)) := by
              simp [Finset.sum_filter]
    _ = Finset.univ.sum (fun v : Fin K =>
          Finset.univ.sum (fun z : Fin Z => if κ v = z then f v else 0)) := by
            rw [Finset.sum_comm]
    _ = Finset.univ.sum f := by
          apply Finset.sum_congr rfl
          intro v hv
          rw [Finset.sum_eq_single (κ v)]
          · simp
          · intro z hz hne
            simp [hne.symm]
          · intro hz
            simp at hz

noncomputable def pushforwardFiniteSource {K Z : Nat}
    (μ : FiniteSource K)
    (κ : Fin K → Fin Z) : FiniteSource Z where
  pmf := fun z => (sourceFiber κ z).sum μ.pmf
  nonneg := by
    intro z
    exact Finset.sum_nonneg (by intro v hv; exact μ.nonneg v)
  sum_one := by
    rw [sum_sourceFiber_sum_eq_univ_sum κ μ.pmf, μ.sum_one]

theorem pushforwardEntropy_le_sourceEntropy
    {K Z : Nat}
    (μ : FiniteSource K)
    (κ : Fin K → Fin Z) :
    sourceEntropy (pushforwardFiniteSource μ κ) ≤ sourceEntropy μ := by
  unfold sourceEntropy pushforwardFiniteSource
  calc
    Finset.univ.sum (fun z : Fin Z => (sourceFiber κ z).sum μ.pmf * Real.log ((sourceFiber κ z).sum μ.pmf)⁻¹)
        = Finset.univ.sum (fun z : Fin Z => Real.negMulLog ((sourceFiber κ z).sum μ.pmf)) := by
            apply Finset.sum_congr rfl
            intro z hz
            rw [Real.negMulLog, Real.log_inv]
            ring
    _ ≤ Finset.univ.sum (fun z : Fin Z => (sourceFiber κ z).sum (fun v => Real.negMulLog (μ.pmf v))) := by
          apply Finset.sum_le_sum
          intro z hz
          simpa [sourceFiber] using negMulLog_sum_le (sourceFiber κ z) μ.pmf (by
            intro v hv
            exact μ.nonneg v)
    _ = Finset.univ.sum (fun v : Fin K => Real.negMulLog (μ.pmf v)) := by
          rw [sum_sourceFiber_sum_eq_univ_sum κ (fun v => Real.negMulLog (μ.pmf v))]
    _ = sourceEntropy μ := by
          unfold sourceEntropy
          apply Finset.sum_congr rfl
          intro v hv
          rw [Real.negMulLog, Real.log_inv]
          ring

noncomputable def failureKernelFin {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : Fin K → Fin 2 :=
  fun v => if decode (obs v) (tag v) = some v then 0 else 1

noncomputable def successFailureFiniteSource {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : FiniteSource 2 :=
  pushforwardFiniteSource μ (failureKernelFin obs tag decode)

theorem successFailureFiniteSource_successProb
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    (successFailureFiniteSource μ obs tag decode).pmf 0 = successProb μ obs tag decode := by
  unfold successFailureFiniteSource pushforwardFiniteSource failureKernelFin sourceFiber successProb successSet
  simp [Finset.sum_filter]

theorem successFailureFiniteSource_errorProb
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    (successFailureFiniteSource μ obs tag decode).pmf 1 = errorProb μ obs tag decode := by
  let succ : ℝ := (successSet obs tag decode).sum μ.pmf
  let fail : ℝ := (Finset.univ.filter (fun x : Fin K => ¬ decode (obs x) (tag x) = some x)).sum μ.pmf
  have hsplit := Finset.sum_filter_add_sum_filter_not
      (s := Finset.univ)
      (p := fun v : Fin K => decode (obs v) (tag v) = some v)
      (f := μ.pmf)
  rw [μ.sum_one] at hsplit
  have hsplit' : succ + fail = 1 := by
    simpa [succ, fail, successSet] using hsplit
  have hfail : fail = 1 - succ := by
    linarith
  simpa [succ, fail, successProb, errorProb, successSet, successFailureFiniteSource,
    pushforwardFiniteSource, failureKernelFin, sourceFiber]
    using hfail

theorem successFailureFiniteSource_entropy_eq_binEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    sourceEntropy (successFailureFiniteSource μ obs tag decode) = Real.binEntropy (errorProb μ obs tag decode) := by
  unfold sourceEntropy
  rw [Fin.sum_univ_two]
  rw [successFailureFiniteSource_successProb, successFailureFiniteSource_errorProb]
  have hs : successProb μ obs tag decode * Real.log (successProb μ obs tag decode)⁻¹
      = Real.negMulLog (successProb μ obs tag decode) := by
    rw [Real.negMulLog, Real.log_inv]
    ring_nf
  have he : errorProb μ obs tag decode * Real.log (errorProb μ obs tag decode)⁻¹
      = Real.negMulLog (errorProb μ obs tag decode) := by
    rw [Real.negMulLog, Real.log_inv]
    ring_nf
  rw [hs, he, Real.binEntropy_eq_negMulLog_add_negMulLog_one_sub]
  unfold errorProb
  ring_nf

theorem successFailureEntropy_le_sourceEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    sourceEntropy (successFailureFiniteSource μ obs tag decode) ≤ sourceEntropy μ := by
  exact pushforwardEntropy_le_sourceEntropy μ (failureKernelFin obs tag decode)

structure FiniteRandomVariable (K Z : Nat) where
  map : Fin K → Fin Z

@[ext] theorem FiniteRandomVariable.ext {K Z : Nat} {X Y : FiniteRandomVariable K Z}
    (hmap : X.map = Y.map) : X = Y := by
  cases X
  cases Y
  cases hmap
  rfl

namespace FiniteRandomVariable

noncomputable def comp {K Y Z : Nat}
    (X : FiniteRandomVariable K Y)
    (g : Fin Y → Fin Z) : FiniteRandomVariable K Z where
  map := fun v => g (X.map v)

end FiniteRandomVariable

noncomputable def rvPushforward {K Z : Nat}
    (μ : FiniteSource K)
    (X : FiniteRandomVariable K Z) : FiniteSource Z :=
  pushforwardFiniteSource μ X.map

noncomputable def rvEntropy {K Z : Nat}
    (μ : FiniteSource K)
    (X : FiniteRandomVariable K Z) : ℝ :=
  sourceEntropy (rvPushforward μ X)

theorem rvEntropy_le_sourceEntropy {K Z : Nat}
    (μ : FiniteSource K)
    (X : FiniteRandomVariable K Z) :
    rvEntropy μ X ≤ sourceEntropy μ := by
  exact pushforwardEntropy_le_sourceEntropy μ X.map

noncomputable def failureRVFin {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : FiniteRandomVariable K 2 where
  map := failureKernelFin obs tag decode

theorem rvEntropy_failureRVFin_eq_binEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    rvEntropy μ (failureRVFin obs tag decode) = Real.binEntropy (errorProb μ obs tag decode) := by
  exact successFailureFiniteSource_entropy_eq_binEntropy μ obs tag decode

theorem rvEntropy_failureRVFin_le_sourceEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    rvEntropy μ (failureRVFin obs tag decode) ≤ sourceEntropy μ := by
  exact successFailureEntropy_le_sourceEntropy μ obs tag decode

structure DeterministicObservable (K Z : Nat) where
  rv : FiniteRandomVariable K Z

@[ext] theorem DeterministicObservable.ext {K Z : Nat} {X Y : DeterministicObservable K Z}
    (hrv : X.rv = Y.rv) : X = Y := by
  cases X
  cases Y
  cases hrv
  rfl

noncomputable def observableEntropy {K Z : Nat}
    (μ : FiniteSource K)
    (X : DeterministicObservable K Z) : ℝ :=
  rvEntropy μ X.rv

theorem observableEntropy_le_sourceEntropy {K Z : Nat}
    (μ : FiniteSource K)
    (X : DeterministicObservable K Z) :
    observableEntropy μ X ≤ sourceEntropy μ := by
  exact rvEntropy_le_sourceEntropy μ X.rv

namespace DeterministicObservable

noncomputable def coarsen {K Y Z : Nat}
    (X : DeterministicObservable K Y)
    (g : Fin Y → Fin Z) : DeterministicObservable K Z where
  rv := X.rv.comp g

theorem entropy_coarsen_le_sourceEntropy {K Y Z : Nat}
    (μ : FiniteSource K)
    (X : DeterministicObservable K Y)
    (g : Fin Y → Fin Z) :
    observableEntropy μ (X.coarsen g) ≤ sourceEntropy μ := by
  exact observableEntropy_le_sourceEntropy μ (X.coarsen g)

end DeterministicObservable

noncomputable def failureObservableFin {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : DeterministicObservable K 2 where
  rv := failureRVFin obs tag decode

theorem observableEntropy_failureObservable_eq_binEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    observableEntropy μ (failureObservableFin obs tag decode) = Real.binEntropy (errorProb μ obs tag decode) := by
  exact rvEntropy_failureRVFin_eq_binEntropy μ obs tag decode

theorem observableEntropy_failureObservable_le_sourceEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    observableEntropy μ (failureObservableFin obs tag decode) ≤ sourceEntropy μ := by
  exact observableEntropy_le_sourceEntropy μ (failureObservableFin obs tag decode)

noncomputable def optionEncodeFin {K : Nat} : Option (Fin K) → Fin (K + 1)
  | none => 0
  | some v => ⟨v.1.succ, Nat.succ_lt_succ v.2⟩

noncomputable def decodedOutputKernelFin {K O T : Nat}
    (decode : Fin O → Fin T → Option (Fin K)) : Fin (O * T) → Fin (K + 1) :=
  fun i => optionEncodeFin (decode ((finProdFinEquiv : Fin O × Fin T ≃ Fin (O * T)).symm i).1
    ((finProdFinEquiv : Fin O × Fin T ≃ Fin (O * T)).symm i).2)

noncomputable def decodedOutputRVOnPair {K O T : Nat}
    (decode : Fin O → Fin T → Option (Fin K)) : FiniteRandomVariable (O * T) (K + 1) where
  map := decodedOutputKernelFin decode

noncomputable def decodedOutputEntropy {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : ℝ :=
  rvEntropy (pairFiniteSource μ obs tag) (decodedOutputRVOnPair decode)

noncomputable def inducedPairPMFFin {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) : PMF (Fin (O * T)) :=
  finiteSourcePMF (pairFiniteSource μ obs tag)

noncomputable def inducedDecodedOutputPMFFin {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : PMF (Fin (K + 1)) :=
  finiteSourcePMF (pushforwardFiniteSource (pairFiniteSource μ obs tag) (decodedOutputKernelFin decode))

theorem observationTagEntropy_eq_pairFiniteSource_entropy {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    observationTagEntropy μ obs tag = sourceEntropy (pairFiniteSource μ obs tag) := by
  let e : Fin O × Fin T ≃ Fin (O * T) := finProdFinEquiv
  calc
    observationTagEntropy μ obs tag
        = Finset.univ.sum (fun i : Fin (O * T) => Real.negMulLog (pairMass μ obs tag (e.symm i))) := by
            unfold observationTagEntropy
            exact (Fintype.sum_equiv e.symm _ _ (fun y => rfl)).symm
    _ = Finset.univ.sum (fun i : Fin (O * T) =>
          pairMass μ obs tag (e.symm i) * Real.log (pairMass μ obs tag (e.symm i))⁻¹) := by
            apply Finset.sum_congr rfl
            intro i hi
            rw [Real.negMulLog, Real.log_inv]
            ring
    _ = sourceEntropy (pairFiniteSource μ obs tag) := by
          rfl

theorem pmfEntropy_inducedPairPMFFin_eq_observationTagEntropy {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    pmfEntropy (inducedPairPMFFin μ obs tag) = observationTagEntropy μ obs tag := by
  unfold inducedPairPMFFin
  rw [pmfEntropy_finiteSourcePMF_eq_sourceEntropy, observationTagEntropy_eq_pairFiniteSource_entropy]

theorem inducedPairPMFFin_eq_uniform_of_klDiv_zero
    {K O T : Nat} [Nonempty (Fin (O * T))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hkl : InformationTheory.klDiv (inducedPairPMFFin μ obs tag).toMeasure
        (PMF.uniformOfFintype (Fin (O * T))).toMeasure = 0) :
    inducedPairPMFFin μ obs tag = PMF.uniformOfFintype (Fin (O * T)) := by
  exact pmf_eq_uniform_fin_of_klDiv_zero_nonempty (inducedPairPMFFin μ obs tag) hkl

theorem observationTagEntropy_eq_log_budget_of_klDiv_zero_uniform
    {K O T : Nat} [Nonempty (Fin (O * T))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hkl : InformationTheory.klDiv (inducedPairPMFFin μ obs tag).toMeasure
        (PMF.uniformOfFintype (Fin (O * T))).toMeasure = 0) :
    observationTagEntropy μ obs tag = Real.log ((O * T : Nat) : ℝ) := by
  rw [← pmfEntropy_inducedPairPMFFin_eq_observationTagEntropy]
  exact pmfEntropy_eq_log_card_of_klDiv_zero_uniform_fin_nonempty (inducedPairPMFFin μ obs tag) hkl

theorem mutualInfoDeterministic_eq_log_budget_of_klDiv_zero_uniform
    {K O T : Nat} [Nonempty (Fin (O * T))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hkl : InformationTheory.klDiv (inducedPairPMFFin μ obs tag).toMeasure
        (PMF.uniformOfFintype (Fin (O * T))).toMeasure = 0) :
    mutualInfoDeterministic μ obs tag = Real.log ((O * T : Nat) : ℝ) := by
  unfold mutualInfoDeterministic jointEntropySourcePair
  rw [observationTagEntropy_eq_log_budget_of_klDiv_zero_uniform μ obs tag hkl]
  ring

theorem nonuniform_inducedPairPMFFin_implies_klDiv_ne_zero
    {K O T : Nat} [Nonempty (Fin (O * T))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hne : inducedPairPMFFin μ obs tag ≠ PMF.uniformOfFintype (Fin (O * T))) :
    InformationTheory.klDiv (inducedPairPMFFin μ obs tag).toMeasure
      (PMF.uniformOfFintype (Fin (O * T))).toMeasure ≠ 0 := by
  intro hkl
  apply hne
  exact inducedPairPMFFin_eq_uniform_of_klDiv_zero μ obs tag hkl

theorem pmfEntropy_inducedDecodedOutputPMFFin_eq_decodedOutputEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    pmfEntropy (inducedDecodedOutputPMFFin μ obs tag decode) = decodedOutputEntropy μ obs tag decode := by
  unfold inducedDecodedOutputPMFFin decodedOutputEntropy decodedOutputRVOnPair rvEntropy rvPushforward
  rw [pmfEntropy_finiteSourcePMF_eq_sourceEntropy]

theorem inducedDecodedOutputPMFFin_eq_uniform_of_klDiv_zero
    {K O T : Nat} [Nonempty (Fin (K + 1))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hkl : InformationTheory.klDiv (inducedDecodedOutputPMFFin μ obs tag decode).toMeasure
        (PMF.uniformOfFintype (Fin (K + 1))).toMeasure = 0) :
    inducedDecodedOutputPMFFin μ obs tag decode = PMF.uniformOfFintype (Fin (K + 1)) := by
  exact pmf_eq_uniform_fin_of_klDiv_zero_nonempty (inducedDecodedOutputPMFFin μ obs tag decode) hkl

theorem decodedOutputEntropy_eq_log_outputAlphabet_of_klDiv_zero_uniform
    {K O T : Nat} [Nonempty (Fin (K + 1))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hkl : InformationTheory.klDiv (inducedDecodedOutputPMFFin μ obs tag decode).toMeasure
        (PMF.uniformOfFintype (Fin (K + 1))).toMeasure = 0) :
    decodedOutputEntropy μ obs tag decode = Real.log ((K + 1 : Nat) : ℝ) := by
  rw [← pmfEntropy_inducedDecodedOutputPMFFin_eq_decodedOutputEntropy]
  exact pmfEntropy_eq_log_card_of_klDiv_zero_uniform_fin_nonempty (inducedDecodedOutputPMFFin μ obs tag decode) hkl

theorem decodedOutputEntropy_gap_eq_zero_of_klDiv_zero_uniform
    {K O T : Nat} [Nonempty (Fin (K + 1))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hkl : InformationTheory.klDiv (inducedDecodedOutputPMFFin μ obs tag decode).toMeasure
        (PMF.uniformOfFintype (Fin (K + 1))).toMeasure = 0) :
    Real.log ((K + 1 : Nat) : ℝ) - decodedOutputEntropy μ obs tag decode = 0 := by
  rw [decodedOutputEntropy_eq_log_outputAlphabet_of_klDiv_zero_uniform μ obs tag decode hkl]
  ring

theorem decodedOutputEntropy_gap_pos_implies_klDiv_ne_zero
    {K O T : Nat} [Nonempty (Fin (K + 1))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hgap : 0 < Real.log ((K + 1 : Nat) : ℝ) - decodedOutputEntropy μ obs tag decode) :
    InformationTheory.klDiv (inducedDecodedOutputPMFFin μ obs tag decode).toMeasure
      (PMF.uniformOfFintype (Fin (K + 1))).toMeasure ≠ 0 := by
  intro hkl
  have hzero := decodedOutputEntropy_gap_eq_zero_of_klDiv_zero_uniform μ obs tag decode hkl
  linarith

noncomputable def coarsenedObservationEntropy {K O T Z : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (g : Fin O × Fin T → Fin Z) : ℝ :=
  observationTagEntropy (pairFiniteSource μ obs tag)
    (fun i => g ((finProdFinEquiv : Fin O × Fin T ≃ Fin (O * T)).symm i))
    (fun _ => (0 : Fin 1))

noncomputable def coarsenedMutualInfoDeterministic {K O T Z : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (g : Fin O × Fin T → Fin Z) : ℝ :=
  mutualInfoDeterministic (pairFiniteSource μ obs tag)
    (fun i => g ((finProdFinEquiv : Fin O × Fin T ≃ Fin (O * T)).symm i))
    (fun _ => (0 : Fin 1))

abbrev DeterministicKernel (O T Z : Nat) := Fin O × Fin T → Fin Z

theorem observationTagEntropy_le_log_budget {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    observationTagEntropy μ obs tag ≤ Real.log ((O * T : Nat) : ℝ) := by
  have hO : 0 < O := by
    have ho := (obs ⟨0, hK⟩).2
    omega
  have hT : 0 < T := by
    have ht := (tag ⟨0, hK⟩).2
    omega
  have hOT : 0 < O * T := Nat.mul_pos hO hT
  rw [observationTagEntropy_eq_pairFiniteSource_entropy]
  exact sourceEntropy_le_log_card (pairFiniteSource μ obs tag) hOT

theorem sourceEntropy_eq_observationTagEntropy_add_conditionalEntropyGivenPair
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    sourceEntropy μ = observationTagEntropy μ obs tag + conditionalEntropyGivenPair μ obs tag := by
  symm
  calc
    observationTagEntropy μ obs tag + conditionalEntropyGivenPair μ obs tag
        = Finset.univ.sum (fun y : Fin O × Fin T =>
            (pairFiber obs tag y).sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)) := by
              unfold observationTagEntropy conditionalEntropyGivenPair
              rw [← Finset.sum_add_distrib]
              apply Finset.sum_congr rfl
              intro y hy
              ring
    _ = sourceEntropy μ := by
          rw [sum_pairFiber_sum_eq_univ_sum obs tag (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)]
          rfl

theorem conditionalEntropyGivenPair_le_sourceEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    conditionalEntropyGivenPair μ obs tag ≤ sourceEntropy μ := by
  rw [sourceEntropy_eq_observationTagEntropy_add_conditionalEntropyGivenPair]
  have hnonneg := observationTagEntropy_nonneg μ obs tag
  linarith

theorem conditionalEntropyGivenPair_nonneg
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    0 ≤ conditionalEntropyGivenPair μ obs tag := by
  unfold conditionalEntropyGivenPair
  apply Finset.sum_nonneg
  intro y hy
  have hmain := negMulLog_sum_le (pairFiber obs tag y) μ.pmf (by
    intro v hv
    exact μ.nonneg v)
  have hmass : Real.negMulLog (pairMass μ obs tag y)
      ≤ (pairFiber obs tag y).sum (fun v => Real.negMulLog (μ.pmf v)) := by
    simpa [pairMass] using hmain
  have hrew : (pairFiber obs tag y).sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
      = (pairFiber obs tag y).sum (fun v => Real.negMulLog (μ.pmf v)) := by
    apply Finset.sum_congr rfl
    intro v hv
    rw [Real.negMulLog, Real.log_inv]
    ring
  rw [hrew]
  linarith

theorem observationTagEntropy_le_sourceEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    observationTagEntropy μ obs tag ≤ sourceEntropy μ := by
  rw [sourceEntropy_eq_observationTagEntropy_add_conditionalEntropyGivenPair]
  have hnonneg := conditionalEntropyGivenPair_nonneg μ obs tag
  linarith

theorem mutualInfoSurrogate_eq_observationTagEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    mutualInfoSurrogate μ obs tag = observationTagEntropy μ obs tag := by
  unfold mutualInfoSurrogate
  rw [sourceEntropy_eq_observationTagEntropy_add_conditionalEntropyGivenPair]
  ring

theorem jointEntropySourcePair_eq_sourceEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    jointEntropySourcePair μ obs tag = sourceEntropy μ := rfl

theorem mutualInfoDeterministic_eq_observationTagEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    mutualInfoDeterministic μ obs tag = observationTagEntropy μ obs tag := by
  unfold mutualInfoDeterministic jointEntropySourcePair
  ring

theorem mutualInfoDeterministic_eq_source_minus_conditional
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    mutualInfoDeterministic μ obs tag = sourceEntropy μ - conditionalEntropyGivenPair μ obs tag := by
  rw [mutualInfoDeterministic_eq_observationTagEntropy]
  have h := sourceEntropy_eq_observationTagEntropy_add_conditionalEntropyGivenPair μ obs tag
  linarith

theorem mutualInfoDeterministic_eq_mutualInfoSurrogate
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    mutualInfoDeterministic μ obs tag = mutualInfoSurrogate μ obs tag := by
  rw [mutualInfoDeterministic_eq_observationTagEntropy, mutualInfoSurrogate_eq_observationTagEntropy]

theorem mutualInfoDeterministic_nonneg
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    0 ≤ mutualInfoDeterministic μ obs tag := by
  rw [mutualInfoDeterministic_eq_observationTagEntropy]
  exact observationTagEntropy_nonneg μ obs tag

theorem mutualInfoDeterministic_le_sourceEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    mutualInfoDeterministic μ obs tag ≤ sourceEntropy μ := by
  rw [mutualInfoDeterministic_eq_observationTagEntropy]
  exact observationTagEntropy_le_sourceEntropy μ obs tag

theorem decodedOutputEntropy_le_mutualInfoDeterministic
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    decodedOutputEntropy μ obs tag decode ≤ mutualInfoDeterministic μ obs tag := by
  unfold decodedOutputEntropy rvEntropy rvPushforward
  have h := pushforwardEntropy_le_sourceEntropy (pairFiniteSource μ obs tag) (decodedOutputKernelFin decode)
  rw [mutualInfoDeterministic_eq_observationTagEntropy, observationTagEntropy_eq_pairFiniteSource_entropy]
  exact h

theorem decodedOutputEntropy_le_observationTagEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    decodedOutputEntropy μ obs tag decode ≤ observationTagEntropy μ obs tag := by
  calc
    decodedOutputEntropy μ obs tag decode ≤ mutualInfoDeterministic μ obs tag :=
      decodedOutputEntropy_le_mutualInfoDeterministic μ obs tag decode
    _ = observationTagEntropy μ obs tag := by rw [mutualInfoDeterministic_eq_observationTagEntropy]

theorem decodedOutputEntropy_gap_nonneg
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    0 ≤ mutualInfoDeterministic μ obs tag - decodedOutputEntropy μ obs tag decode := by
  linarith [decodedOutputEntropy_le_mutualInfoDeterministic μ obs tag decode]

theorem decodedOutputEntropy_source_gap_nonneg
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    0 ≤ sourceEntropy μ - decodedOutputEntropy μ obs tag decode := by
  linarith [decodedOutputEntropy_le_mutualInfoDeterministic μ obs tag decode,
    mutualInfoDeterministic_le_sourceEntropy μ obs tag]

theorem decodedOutputEntropy_le_log_outputAlphabet
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    decodedOutputEntropy μ obs tag decode ≤ Real.log ((K + 1 : Nat) : ℝ) := by
  unfold decodedOutputEntropy rvEntropy rvPushforward
  exact sourceEntropy_le_log_card (pushforwardFiniteSource (pairFiniteSource μ obs tag) (decodedOutputKernelFin decode))
    (Nat.succ_pos K)

theorem decodedOutputEntropy_log_gap_nonneg
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    0 ≤ Real.log ((K + 1 : Nat) : ℝ) - decodedOutputEntropy μ obs tag decode := by
  linarith [decodedOutputEntropy_le_log_outputAlphabet μ obs tag decode]

noncomputable def observationTagObservableFin {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) : DeterministicObservable K (O * T) where
  rv := {
    map := fun v => (finProdFinEquiv : Fin O × Fin T ≃ Fin (O * T)) (obs v, tag v)
  }

noncomputable def decodedOutputObservableFin {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : DeterministicObservable K (K + 1) where
  rv := {
    map := fun v => optionEncodeFin (decode (obs v) (tag v))
  }

theorem decodedOutputObservable_eq_coarsenedObservationObservable
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    decodedOutputObservableFin obs tag decode
      = (observationTagObservableFin obs tag).coarsen (decodedOutputKernelFin decode) := by
  apply DeterministicObservable.ext
  apply FiniteRandomVariable.ext
  funext v
  simp [decodedOutputObservableFin, observationTagObservableFin,
    DeterministicObservable.coarsen, FiniteRandomVariable.comp, decodedOutputKernelFin]

theorem decodedOutputObservable_entropy_le_sourceEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    observableEntropy μ (decodedOutputObservableFin obs tag decode) ≤ sourceEntropy μ := by
  exact observableEntropy_le_sourceEntropy μ (decodedOutputObservableFin obs tag decode)

theorem mutualInfoSurrogate_le_log_budget {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    mutualInfoSurrogate μ obs tag ≤ Real.log ((O * T : Nat) : ℝ) := by
  rw [mutualInfoSurrogate_eq_observationTagEntropy]
  exact observationTagEntropy_le_log_budget μ hK obs tag

theorem mutualInfoDeterministic_le_log_budget {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    mutualInfoDeterministic μ obs tag ≤ Real.log ((O * T : Nat) : ℝ) := by
  rw [mutualInfoDeterministic_eq_observationTagEntropy]
  exact observationTagEntropy_le_log_budget μ hK obs tag

theorem observationTagEntropy_gap_nonneg
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    0 ≤ Real.log ((O * T : Nat) : ℝ) - observationTagEntropy μ obs tag := by
  linarith [observationTagEntropy_le_log_budget μ hK obs tag]

theorem observationTagEntropy_gap_eq_zero_of_klDiv_zero_uniform
    {K O T : Nat} [Nonempty (Fin (O * T))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hkl : InformationTheory.klDiv (inducedPairPMFFin μ obs tag).toMeasure
        (PMF.uniformOfFintype (Fin (O * T))).toMeasure = 0) :
    Real.log ((O * T : Nat) : ℝ) - observationTagEntropy μ obs tag = 0 := by
  rw [observationTagEntropy_eq_log_budget_of_klDiv_zero_uniform μ obs tag hkl]
  ring

theorem observationTagEntropy_gap_pos_implies_klDiv_ne_zero
    {K O T : Nat} [Nonempty (Fin (O * T))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hgap : 0 < Real.log ((O * T : Nat) : ℝ) - observationTagEntropy μ obs tag) :
    InformationTheory.klDiv (inducedPairPMFFin μ obs tag).toMeasure
      (PMF.uniformOfFintype (Fin (O * T))).toMeasure ≠ 0 := by
  intro hkl
  have hzero := observationTagEntropy_gap_eq_zero_of_klDiv_zero_uniform μ obs tag hkl
  linarith

theorem mutualInfoDeterministic_gap_pos_implies_klDiv_ne_zero
    {K O T : Nat} [Nonempty (Fin (O * T))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hgap : 0 < Real.log ((O * T : Nat) : ℝ) - mutualInfoDeterministic μ obs tag) :
    InformationTheory.klDiv (inducedPairPMFFin μ obs tag).toMeasure
      (PMF.uniformOfFintype (Fin (O * T))).toMeasure ≠ 0 := by
  rw [mutualInfoDeterministic_eq_observationTagEntropy] at hgap
  exact observationTagEntropy_gap_pos_implies_klDiv_ne_zero μ obs tag hgap

theorem coarsenedObservationEntropy_le_observationTagEntropy
    {K O T Z : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (g : Fin O × Fin T → Fin Z) :
    coarsenedObservationEntropy μ obs tag g ≤ observationTagEntropy μ obs tag := by
  unfold coarsenedObservationEntropy
  have h := observationTagEntropy_le_sourceEntropy (pairFiniteSource μ obs tag)
    (fun i => g ((finProdFinEquiv : Fin O × Fin T ≃ Fin (O * T)).symm i))
    (fun _ => (0 : Fin 1))
  rw [(observationTagEntropy_eq_pairFiniteSource_entropy μ obs tag).symm] at h
  exact h

theorem coarsenedMutualInfoDeterministic_eq_coarsenedObservationEntropy
    {K O T Z : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (g : Fin O × Fin T → Fin Z) :
    coarsenedMutualInfoDeterministic μ obs tag g = coarsenedObservationEntropy μ obs tag g := by
  unfold coarsenedMutualInfoDeterministic coarsenedObservationEntropy
  rw [mutualInfoDeterministic_eq_observationTagEntropy]

theorem coarsenedMutualInfoDeterministic_le_original
    {K O T Z : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (g : Fin O × Fin T → Fin Z) :
    coarsenedMutualInfoDeterministic μ obs tag g ≤ mutualInfoDeterministic μ obs tag := by
  rw [coarsenedMutualInfoDeterministic_eq_coarsenedObservationEntropy,
    mutualInfoDeterministic_eq_observationTagEntropy]
  exact coarsenedObservationEntropy_le_observationTagEntropy μ obs tag g

theorem deterministicKernel_data_processing
    {K O T Z : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (κ : DeterministicKernel O T Z) :
    coarsenedMutualInfoDeterministic μ obs tag κ ≤ mutualInfoDeterministic μ obs tag :=
  coarsenedMutualInfoDeterministic_le_original μ obs tag κ

theorem deterministicKernel_entropy_data_processing
    {K O T Z : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (κ : DeterministicKernel O T Z) :
    coarsenedObservationEntropy μ obs tag κ ≤ observationTagEntropy μ obs tag :=
  coarsenedObservationEntropy_le_observationTagEntropy μ obs tag κ

theorem errorProb_eq_failureSet_sum
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    errorProb μ obs tag decode = (failureSet obs tag decode).sum μ.pmf := by
  unfold errorProb successProb successSet failureSet
  have hsplit := Finset.sum_filter_add_sum_filter_not
      (s := Finset.univ)
      (p := fun v : Fin K => decode (obs v) (tag v) = some v)
      (f := μ.pmf)
  rw [μ.sum_one] at hsplit
  linarith

theorem successSet_card_add_failureSet_card
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    (successSet obs tag decode).card + (failureSet obs tag decode).card = K := by
  unfold successSet failureSet
  simpa using Finset.card_filter_add_card_filter_not
      (s := Finset.univ)
      (p := fun v : Fin K => decode (obs v) (tag v) = some v)


end ObserverModel
