import Mathlib.Analysis.SpecialFunctions.BinaryEntropy
import Mathlib.InformationTheory.KullbackLeibler.Basic
import Mathlib.Probability.Distributions.Uniform
import Mathlib.Probability.ProbabilityMassFunction.Basic
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Probability.ProbabilityMassFunction.Integrals

/-!
  Long-horizon entropy/information-theory scaffold.

  This module verifies that the current Paper 2 proof environment can import the
  available mathlib entropy / KL / PMF libraries without introducing any new
  axioms. It is the natural landing point for future work toward fuller Fano-style
  probabilistic converses.
-/

namespace ObserverModel

open scoped ENNReal

/-- Finite Shannon entropy in nats for a `PMF` on a fintype. -/
noncomputable def pmfEntropy {α : Type*} [Fintype α] (p : PMF α) : ℝ :=
  ∑ a, (p a).toReal * Real.log ((p a).toReal)⁻¹

/-- Binary entropy is available in the source proof environment. -/
theorem binEntropy_half : Real.binEntropy ((1 : ℝ) / 2) = Real.log 2 := by
  simp [show ((1 : ℝ) / 2) = (2 : ℝ)⁻¹ by norm_num]

/-- Kullback-Leibler divergence is importable in the source proof environment. -/
theorem klDiv_self_eq_zero
    {α : Type*} [MeasurableSpace α] (p : PMF α) :
    InformationTheory.klDiv p.toMeasure p.toMeasure = 0 := by
  exact InformationTheory.klDiv_self p.toMeasure

/-- PMFs are importable and can be turned into probability measures. -/
theorem pmf_toMeasure_isProbabilityMeasure
    {α : Type*} [MeasurableSpace α] (p : PMF α) :
    MeasureTheory.IsProbabilityMeasure p.toMeasure := by
  infer_instance

/-- The PMF-level entropy of a Bernoulli law matches mathlib's binary entropy. -/
theorem pmfEntropy_bernoulli
    (p : NNReal) (h : p ≤ 1) :
    pmfEntropy (PMF.bernoulli p h) = Real.binEntropy (p : ℝ) := by
  unfold pmfEntropy Real.binEntropy
  simp [PMF.bernoulli_apply, h]
  ring_nf

/-- Bernoulli PMF entropy is bounded by `log 2`. -/
theorem pmfEntropy_bernoulli_le_log_two
    (p : NNReal) (h : p ≤ 1) :
    pmfEntropy (PMF.bernoulli p h) ≤ Real.log 2 := by
  rw [pmfEntropy_bernoulli p h]
  exact Real.binEntropy_le_log_two

theorem pmfEntropy_bernoulli_ofReal
    {p : ℝ}
    (hp0 : 0 ≤ p)
    (hp1 : p ≤ 1) :
    pmfEntropy
        (PMF.bernoulli (Real.toNNReal p)
          (by
            change (Real.toNNReal p : ℝ) ≤ 1
            rw [Real.toNNReal_of_nonneg hp0]
            exact hp1))
      = Real.binEntropy p := by
  let q : NNReal := Real.toNNReal p
  have hq : q ≤ 1 := by
    change (Real.toNNReal p : ℝ) ≤ 1
    rw [Real.toNNReal_of_nonneg hp0]
    exact hp1
  simpa [q, Real.toNNReal_of_nonneg hp0] using pmfEntropy_bernoulli q hq

/-- Uniform entropy on a finite alphabet `Fin (n+1)` is `log (n+1)`. -/
theorem pmfEntropy_uniform_fin
    (n : Nat) :
    pmfEntropy (PMF.uniformOfFintype (Fin (n + 1))) = Real.log (n + 1) := by
  unfold pmfEntropy
  simp [PMF.uniformOfFintype_apply, Fintype.card_fin]
  set x : ℝ := ((↑n + 1 : ℝ≥0∞).toReal)
  have hx : x = (n : ℝ) + 1 := by
    unfold x
    simpa using ENNReal.toReal_natCast (n + 1)
  have hxne : x ≠ 0 := by
    rw [hx]
    positivity
  change x * (x⁻¹ * Real.log x) = Real.log x
  field_simp [hxne]

theorem pmf_eq_uniform_fin_of_klDiv_zero
    (n : Nat)
    (p : PMF (Fin (n + 1)))
    (hkl : InformationTheory.klDiv p.toMeasure (PMF.uniformOfFintype (Fin (n + 1))).toMeasure = 0) :
    p = PMF.uniformOfFintype (Fin (n + 1)) := by
  have hmeas : p.toMeasure = (PMF.uniformOfFintype (Fin (n + 1))).toMeasure :=
    (InformationTheory.klDiv_eq_zero_iff).1 hkl
  ext a
  have happly := congrArg (fun μ : MeasureTheory.Measure (Fin (n + 1)) => μ {a}) hmeas
  calc
    p a = p.toMeasure {a} := by
      symm
      exact PMF.toMeasure_apply_singleton p a (MeasurableSet.singleton a)
    _ = (PMF.uniformOfFintype (Fin (n + 1))).toMeasure {a} := happly
    _ = (PMF.uniformOfFintype (Fin (n + 1))) a := by
      exact PMF.toMeasure_apply_singleton (PMF.uniformOfFintype (Fin (n + 1))) a
        (MeasurableSet.singleton a)

theorem pmfEntropy_eq_log_card_of_klDiv_zero_uniform_fin
    (n : Nat)
    (p : PMF (Fin (n + 1)))
    (hkl : InformationTheory.klDiv p.toMeasure (PMF.uniformOfFintype (Fin (n + 1))).toMeasure = 0) :
    pmfEntropy p = Real.log (n + 1) := by
  rw [pmf_eq_uniform_fin_of_klDiv_zero n p hkl]
  exact pmfEntropy_uniform_fin n

theorem pmf_eq_uniform_fin_of_klDiv_zero_nonempty
    {n : Nat} [Nonempty (Fin n)]
    (p : PMF (Fin n))
    (hkl : InformationTheory.klDiv p.toMeasure (PMF.uniformOfFintype (Fin n)).toMeasure = 0) :
    p = PMF.uniformOfFintype (Fin n) := by
  have hmeas : p.toMeasure = (PMF.uniformOfFintype (Fin n)).toMeasure :=
    (InformationTheory.klDiv_eq_zero_iff).1 hkl
  ext a
  have happly := congrArg (fun μ : MeasureTheory.Measure (Fin n) => μ {a}) hmeas
  calc
    p a = p.toMeasure {a} := by
      symm
      exact PMF.toMeasure_apply_singleton p a (MeasurableSet.singleton a)
    _ = (PMF.uniformOfFintype (Fin n)).toMeasure {a} := happly
    _ = (PMF.uniformOfFintype (Fin n)) a := by
      exact PMF.toMeasure_apply_singleton (PMF.uniformOfFintype (Fin n)) a
        (MeasurableSet.singleton a)

theorem pmfEntropy_uniform_fin_nonempty
    {n : Nat} [Nonempty (Fin n)] :
    pmfEntropy (PMF.uniformOfFintype (Fin n)) = Real.log n := by
  have hn : 0 < n := Nat.zero_lt_of_lt (Classical.choice (inferInstance : Nonempty (Fin n))).2
  unfold pmfEntropy
  simp [PMF.uniformOfFintype_apply, Fintype.card_fin]
  set x : ℝ := ((n : ℝ≥0∞).toReal)
  have hx : x = (n : ℝ) := by
    unfold x
    exact ENNReal.toReal_natCast n
  have hxne : x ≠ 0 := by
    rw [hx]
    exact_mod_cast hn.ne'
  change x * (x⁻¹ * Real.log x) = Real.log x
  have hinv : x * x⁻¹ = 1 := by field_simp [hxne]
  calc
    x * (x⁻¹ * Real.log x) = (x * x⁻¹) * Real.log x := by ring
    _ = Real.log x := by rw [hinv, one_mul]
    _ = Real.log n := by rw [hx]

theorem pmfEntropy_eq_log_card_of_klDiv_zero_uniform_fin_nonempty
    {n : Nat} [Nonempty (Fin n)]
    (p : PMF (Fin n))
    (hkl : InformationTheory.klDiv p.toMeasure (PMF.uniformOfFintype (Fin n)).toMeasure = 0) :
    pmfEntropy p = Real.log n := by
  rw [pmf_eq_uniform_fin_of_klDiv_zero_nonempty p hkl]
  exact pmfEntropy_uniform_fin_nonempty

/-- Binary-entropy partition identity for splitting a uniform finite source into success/failure blocks. -/
theorem binEntropy_partition_identity
    {s f : ℝ}
    (hs : 0 < s)
    (hf : 0 < f) :
    Real.binEntropy (f / (s + f))
      + (s / (s + f)) * Real.log s
      + (f / (s + f)) * Real.log f
      = Real.log (s + f) := by
  have hsf : 0 < s + f := add_pos hs hf
  have hsf_ne : s + f ≠ 0 := ne_of_gt hsf
  let y : ℝ := (s + f)⁻¹
  have hy : y = (s + f)⁻¹ := rfl
  have hy_mul : (s + f) * y = 1 := by
    unfold y
    field_simp [hsf_ne]
  have hsplit : f / (s + f) = f * y := by
    unfold y
    field_simp [hsf_ne]
  have hsplit' : s / (s + f) = s * y := by
    unfold y
    field_simp [hsf_ne]
  rw [Real.binEntropy_eq_negMulLog_add_negMulLog_one_sub]
  have hone : 1 - f / (s + f) = s / (s + f) := by
    rw [hsplit, hsplit']
    nlinarith [hy_mul]
  rw [hone, hsplit, hsplit']
  rw [Real.negMulLog_mul, Real.negMulLog_mul]
  unfold Real.negMulLog
  have hy_pos : 0 < y := by
    unfold y
    positivity
  have hlogy : Real.log y = - Real.log (s + f) := by
    unfold y
    rw [Real.log_inv]
  rw [hlogy]
  ring_nf
  have hy_sum : y * f + y * s = 1 := by nlinarith [hy_mul]
  calc
    y * f * Real.log (f + s) + y * s * Real.log (f + s)
      = (y * f + y * s) * Real.log (f + s) := by ring
    _ = Real.log (f + s) := by rw [hy_sum, one_mul]

end ObserverModel
