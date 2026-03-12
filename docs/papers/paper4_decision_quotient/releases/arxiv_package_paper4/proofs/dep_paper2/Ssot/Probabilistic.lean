import Ssot.ObserverModel
import Ssot.EntropyGeneral
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Data.Real.Basic

namespace ObserverModel

/-- A finite source distribution on `Fin K`. -/
structure FiniteSource (K : Nat) where
  pmf : Fin K → ℝ
  nonneg : ∀ v, 0 ≤ pmf v
  sum_one : (Finset.univ.sum pmf) = 1

@[ext] theorem FiniteSource.ext {K : Nat} {μ ν : FiniteSource K}
    (hpmf : μ.pmf = ν.pmf) : μ = ν := by
  cases μ
  cases ν
  cases hpmf
  rfl

/-- Maximum atom mass for a nonempty finite source. -/
noncomputable def maxMass {K : Nat} (μ : FiniteSource K) (hK : 0 < K) : ℝ := by
  classical
  letI : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  have himage : (Finset.univ.image μ.pmf).Nonempty := by
    exact Finset.univ_nonempty.image μ.pmf
  exact (Finset.univ.image μ.pmf).max' himage

theorem pmf_le_maxMass {K : Nat} (μ : FiniteSource K) (hK : 0 < K) (v : Fin K) :
    μ.pmf v ≤ maxMass μ hK := by
  classical
  unfold maxMass
  letI : Nonempty (Fin K) := ⟨⟨0, hK⟩⟩
  exact Finset.le_max' _ _ (by simp)

/-- Min-entropy of a finite source. -/
noncomputable def minEntropy {K : Nat} (μ : FiniteSource K) (hK : 0 < K) : ℝ :=
  - Real.log (maxMass μ hK)

theorem maxMass_nonneg {K : Nat} (μ : FiniteSource K) (hK : 0 < K) :
    0 ≤ maxMass μ hK := by
  have h0 := μ.nonneg ⟨0, hK⟩
  exact le_trans h0 (pmf_le_maxMass μ hK ⟨0, hK⟩)

theorem maxMass_pos {K : Nat} (μ : FiniteSource K) (hK : 0 < K) :
    0 < maxMass μ hK := by
  by_contra hnot
  have hnonpos : maxMass μ hK ≤ 0 := le_of_not_gt hnot
  have hzero : maxMass μ hK = 0 := le_antisymm hnonpos (maxMass_nonneg μ hK)
  have hallzero : ∀ v : Fin K, μ.pmf v = 0 := by
    intro v
    have hle := pmf_le_maxMass μ hK v
    linarith [μ.nonneg v, hzero, hle]
  have : (Finset.univ.sum μ.pmf) = 0 := by simp [hallzero]
  linarith [μ.sum_one, this]

theorem inv_card_le_maxMass {K : Nat} (μ : FiniteSource K) (hK : 0 < K) :
    ((K : ℝ))⁻¹ ≤ maxMass μ hK := by
  have hsum_le : (Finset.univ.sum μ.pmf) ≤ Finset.univ.sum (fun _ : Fin K => maxMass μ hK) := by
    apply Finset.sum_le_sum
    intro v hv
    exact pmf_le_maxMass μ hK v
  have hcard : Finset.univ.sum (fun _ : Fin K => maxMass μ hK) = (K : ℝ) * maxMass μ hK := by
    simp
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  have hmain : 1 ≤ (K : ℝ) * maxMass μ hK := by
    rw [μ.sum_one] at hsum_le
    rwa [hcard] at hsum_le
  have hdiv : 1 / (K : ℝ) ≤ maxMass μ hK :=
    (div_le_iff₀ hKpos).2 <| by simpa [mul_comm] using hmain
  simpa [one_div] using hdiv

theorem minEntropy_eq_neg_log_maxMass {K : Nat} (μ : FiniteSource K) (hK : 0 < K) :
    minEntropy μ hK = - Real.log (maxMass μ hK) := rfl

theorem minEntropy_le_log_card {K : Nat} (μ : FiniteSource K) (hK : 0 < K) :
    minEntropy μ hK ≤ Real.log K := by
  rw [minEntropy_eq_neg_log_maxMass]
  have hKpos : (0 : ℝ) < K := by exact_mod_cast hK
  have hmaxpos : 0 < maxMass μ hK := maxMass_pos μ hK
  have hsum_le : (Finset.univ.sum μ.pmf) ≤ Finset.univ.sum (fun _ : Fin K => maxMass μ hK) := by
    apply Finset.sum_le_sum
    intro v hv
    exact pmf_le_maxMass μ hK v
  have hcard : Finset.univ.sum (fun _ : Fin K => maxMass μ hK) = (K : ℝ) * maxMass μ hK := by
    simp
  have hmain : 1 ≤ (K : ℝ) * maxMass μ hK := by
    rw [μ.sum_one] at hsum_le
    rwa [hcard] at hsum_le
  have hinv' : 1 / maxMass μ hK ≤ (K : ℝ) :=
    (div_le_iff₀ hmaxpos).2 <| by simpa [mul_comm] using hmain
  have hinv : (maxMass μ hK)⁻¹ ≤ (K : ℝ) := by
    simpa [one_div] using hinv'
  have hlog : Real.log ((maxMass μ hK)⁻¹) ≤ Real.log K := by
    apply Real.log_le_log
    · positivity
    · exact hinv
  simpa [Real.log_inv] using hlog

theorem pmf_le_one {K : Nat} (μ : FiniteSource K) (v : Fin K) :
    μ.pmf v ≤ 1 := by
  have hle : μ.pmf v ≤ Finset.univ.sum μ.pmf := by
    exact Finset.single_le_sum (fun w _ => μ.nonneg w) (by simp)
  simpa [μ.sum_one] using hle

/-- Shannon-style finite entropy of the source. -/
noncomputable def sourceEntropy {K : Nat} (μ : FiniteSource K) : ℝ :=
  Finset.univ.sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)

theorem sourceEntropy_nonneg {K : Nat} (μ : FiniteSource K) :
    0 ≤ sourceEntropy μ := by
  unfold sourceEntropy
  apply Finset.sum_nonneg
  intro v hv
  have hterm : 0 ≤ Real.negMulLog (μ.pmf v) :=
    Real.negMulLog_nonneg (μ.nonneg v) (pmf_le_one μ v)
  simpa [Real.negMulLog, Real.log_inv, mul_comm, mul_left_comm, mul_assoc] using hterm

theorem sourceEntropy_le_log_card {K : Nat} (μ : FiniteSource K) (hK : 0 < K) :
    sourceEntropy μ ≤ Real.log K := by
  have hKpos : (0 : ℝ) < K := by
    exact_mod_cast hK
  have hKne : (K : ℝ) ≠ 0 := ne_of_gt hKpos
  have hJ :
      (Finset.univ.sum (fun v : Fin K => ((K : ℝ)⁻¹) * Real.negMulLog (μ.pmf v)))
        ≤ Real.negMulLog (Finset.univ.sum (fun v : Fin K => ((K : ℝ)⁻¹) * μ.pmf v)) := by
    simpa [smul_eq_mul] using
      (Real.concaveOn_negMulLog.le_map_sum
        (t := Finset.univ)
        (w := fun _ : Fin K => (K : ℝ)⁻¹)
        (p := μ.pmf)
        (h₀ := by intro v hv; positivity)
        (h₁ := by
          calc
            (Finset.univ.sum fun _ : Fin K => (K : ℝ)⁻¹) = (K : ℝ) * (K : ℝ)⁻¹ := by simp
            _ = 1 := by field_simp [hKne])
        (hmem := by intro v hv; exact μ.nonneg v))
  have hleft :
      Finset.univ.sum (fun v : Fin K => ((K : ℝ)⁻¹) * Real.negMulLog (μ.pmf v)) =
        (K : ℝ)⁻¹ * sourceEntropy μ := by
    unfold sourceEntropy
    calc
      Finset.univ.sum (fun v : Fin K => ((K : ℝ)⁻¹) * Real.negMulLog (μ.pmf v))
          = (K : ℝ)⁻¹ * Finset.univ.sum (fun v : Fin K => Real.negMulLog (μ.pmf v)) := by
              simpa [mul_comm, mul_left_comm, mul_assoc] using
                (Finset.mul_sum (s := Finset.univ)
                  (a := (K : ℝ)⁻¹)
                  (f := fun v : Fin K => Real.negMulLog (μ.pmf v))).symm
      _ = (K : ℝ)⁻¹ * sourceEntropy μ := by
            congr 1
            apply Finset.sum_congr rfl
            intro v hv
            rw [Real.negMulLog, Real.log_inv]
            ring
  have hright :
      Real.negMulLog (Finset.univ.sum (fun v : Fin K => ((K : ℝ)⁻¹) * μ.pmf v)) =
        Real.negMulLog ((K : ℝ)⁻¹) := by
    calc
      Real.negMulLog (Finset.univ.sum (fun v : Fin K => ((K : ℝ)⁻¹) * μ.pmf v))
          = Real.negMulLog ((K : ℝ)⁻¹ * Finset.univ.sum μ.pmf) := by
              congr 1
              simpa [mul_comm, mul_left_comm, mul_assoc] using
                (Finset.mul_sum (s := Finset.univ) (a := (K : ℝ)⁻¹) (f := μ.pmf)).symm
      _ = Real.negMulLog ((K : ℝ)⁻¹) := by simp [μ.sum_one]
  have hscaled : (K : ℝ)⁻¹ * sourceEntropy μ ≤ Real.negMulLog ((K : ℝ)⁻¹) := by
    simpa [hleft, hright] using hJ
  have hmul := mul_le_mul_of_nonneg_left hscaled hKpos.le
  calc
    sourceEntropy μ = (K : ℝ) * ((K : ℝ)⁻¹ * sourceEntropy μ) := by
      field_simp [hKne]
    _ ≤ (K : ℝ) * Real.negMulLog ((K : ℝ)⁻¹) := hmul
    _ = (K : ℝ) * (((K : ℝ)⁻¹) * Real.log K) := by
      rw [Real.negMulLog, Real.log_inv]
      ring
    _ = Real.log K := by
      field_simp [hKne]

theorem subset_sourceEntropy_le_card
    {K : Nat}
    (μ : FiniteSource K)
    (s : Finset (Fin K))
    (hs : 0 < s.card) :
    s.sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
      ≤ Real.negMulLog (s.sum μ.pmf) + (s.sum μ.pmf) * Real.log s.card := by
  let m : ℝ := s.card
  have hmpos : 0 < m := by
    unfold m
    exact_mod_cast hs
  have hmne : m ≠ 0 := ne_of_gt hmpos
  have hJ :
      s.sum (fun v => m⁻¹ * Real.negMulLog (μ.pmf v))
        ≤ Real.negMulLog (s.sum (fun v => m⁻¹ * μ.pmf v)) := by
    simpa [m, smul_eq_mul] using
      (Real.concaveOn_negMulLog.le_map_sum
        (t := s)
        (w := fun _ : Fin K => m⁻¹)
        (p := μ.pmf)
        (h₀ := by intro v hv; positivity)
        (h₁ := by
          calc
            s.sum (fun _ : Fin K => m⁻¹) = (s.card : ℝ) * m⁻¹ := by simp [m]
            _ = 1 := by
              unfold m
              field_simp [show ((s.card : ℝ)) ≠ 0 by exact_mod_cast hs.ne'])
        (hmem := by intro v hv; exact μ.nonneg v))
  have hleft :
      s.sum (fun v => m⁻¹ * Real.negMulLog (μ.pmf v)) =
        m⁻¹ * s.sum (fun v => Real.negMulLog (μ.pmf v)) := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using
      (Finset.mul_sum (s := s) (a := m⁻¹) (f := fun v => Real.negMulLog (μ.pmf v))).symm
  have hright :
      s.sum (fun v => m⁻¹ * μ.pmf v) = m⁻¹ * s.sum μ.pmf := by
    simpa [mul_comm, mul_left_comm, mul_assoc] using
      (Finset.mul_sum (s := s) (a := m⁻¹) (f := μ.pmf)).symm
  have hscaled := mul_le_mul_of_nonneg_left hJ hmpos.le
  have hscaled' :
      s.sum (fun v => Real.negMulLog (μ.pmf v))
        ≤ m * Real.negMulLog (m⁻¹ * s.sum μ.pmf) := by
    simpa [hleft, hright, hmne, mul_comm, mul_left_comm, mul_assoc] using hscaled
  have hident :
      m * Real.negMulLog (m⁻¹ * s.sum μ.pmf) =
        Real.negMulLog (s.sum μ.pmf) + (s.sum μ.pmf) * Real.log m := by
    have hinv : Real.negMulLog m⁻¹ = m⁻¹ * Real.log m := by
      rw [Real.negMulLog, Real.log_inv]
      ring
    calc
      m * Real.negMulLog (m⁻¹ * s.sum μ.pmf)
          = m * ((s.sum μ.pmf) * Real.negMulLog m⁻¹ + m⁻¹ * Real.negMulLog (s.sum μ.pmf)) := by
              rw [Real.negMulLog_mul]
      _ = m * ((s.sum μ.pmf) * (m⁻¹ * Real.log m) + m⁻¹ * Real.negMulLog (s.sum μ.pmf)) := by
            rw [hinv]
      _ = Real.negMulLog (s.sum μ.pmf) + (s.sum μ.pmf) * Real.log m := by
            field_simp [hmne]
            ring
  calc
    s.sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
        = s.sum (fun v => Real.negMulLog (μ.pmf v)) := by
            apply Finset.sum_congr rfl
            intro v hv
            rw [Real.negMulLog, Real.log_inv]
            ring
    _ ≤ m * Real.negMulLog (m⁻¹ * s.sum μ.pmf) := hscaled'
    _ = Real.negMulLog (s.sum μ.pmf) + (s.sum μ.pmf) * Real.log m := hident
    _ = Real.negMulLog (s.sum μ.pmf) + (s.sum μ.pmf) * Real.log s.card := by
          simp [m]

theorem negMulLog_add_le (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
    Real.negMulLog (a + b) ≤ Real.negMulLog a + Real.negMulLog b := by
  rcases eq_or_lt_of_le ha with rfl | ha'
  · simp [Real.negMulLog]
  rcases eq_or_lt_of_le hb with rfl | hb'
  · simp [Real.negMulLog]
  have hbin_nonneg : 0 ≤ Real.binEntropy (b / (a + b)) := by
    apply Real.binEntropy_nonneg
    · positivity
    · exact div_le_one_of_le₀ (by linarith) (by linarith)
  have h := binEntropy_partition_identity ha' hb'
  have hmul := mul_nonneg (show 0 ≤ a + b by linarith) hbin_nonneg
  have hscaled : (a + b) * Real.binEntropy (b / (a + b)) + a * Real.log a + b * Real.log b
      = (a + b) * Real.log (a + b) := by
    have hab : (a + b : ℝ) ≠ 0 := by linarith
    field_simp [hab] at h
    linarith
  rw [Real.negMulLog, Real.negMulLog, Real.negMulLog]
  nlinarith [hscaled, hmul]

theorem negMulLog_sum_le {α : Type*} (s : Finset α) (f : α → ℝ)
    (hf : ∀ x, x ∈ s → 0 ≤ f x) :
    Real.negMulLog (s.sum f) ≤ s.sum (fun x => Real.negMulLog (f x)) := by
  classical
  revert f
  induction s using Finset.induction_on with
  | empty =>
      intro f hf
      simp [Real.negMulLog]
  | @insert a s ha ih =>
      intro f hf
      have hfa : 0 ≤ f a := hf a (by simp [ha])
      have hfs : 0 ≤ s.sum f := Finset.sum_nonneg (by
        intro x hx
        exact hf x (by simp [hx]))
      have hs' : ∀ x, x ∈ s → 0 ≤ f x := by
        intro x hx
        exact hf x (by simp [hx])
      calc
        Real.negMulLog ((insert a s).sum f)
            = Real.negMulLog (f a + s.sum f) := by simp [ha]
        _ ≤ Real.negMulLog (f a) + Real.negMulLog (s.sum f) := negMulLog_add_le (f a) (s.sum f) hfa hfs
        _ ≤ Real.negMulLog (f a) + s.sum (fun x => Real.negMulLog (f x)) := by
              gcongr
              exact ih f hs'
        _ = (insert a s).sum (fun x => Real.negMulLog (f x)) := by simp [ha]

lemma term_ge_scaled_minEntropy {K : Nat} (μ : FiniteSource K) (hK : 0 < K) (v : Fin K) :
    μ.pmf v * Real.log (μ.pmf v)⁻¹ ≥ μ.pmf v * minEntropy μ hK := by
  rw [minEntropy_eq_neg_log_maxMass]
  by_cases hv : μ.pmf v = 0
  · simp [hv]
  · have hμpos : 0 < μ.pmf v := lt_of_le_of_ne (μ.nonneg v) (Ne.symm hv)
    have hmaxpos : 0 < maxMass μ hK := maxMass_pos μ hK
    have hinv : (maxMass μ hK)⁻¹ ≤ (μ.pmf v)⁻¹ := by
      simpa [one_div] using one_div_le_one_div_of_le hμpos (pmf_le_maxMass μ hK v)
    have hlog : Real.log ((maxMass μ hK)⁻¹) ≤ Real.log ((μ.pmf v)⁻¹) := by
      apply Real.log_le_log
      · positivity
      · exact hinv
    have hmul := mul_le_mul_of_nonneg_left hlog (μ.nonneg v)
    simpa [neg_mul, mul_comm, mul_left_comm, mul_assoc] using hmul

theorem minEntropy_le_sourceEntropy {K : Nat} (μ : FiniteSource K) (hK : 0 < K) :
    minEntropy μ hK ≤ sourceEntropy μ := by
  unfold sourceEntropy
  have hsum : Finset.univ.sum (fun v : Fin K => μ.pmf v * minEntropy μ hK) ≤
      Finset.univ.sum (fun v : Fin K => μ.pmf v * Real.log (μ.pmf v)⁻¹) := by
    apply Finset.sum_le_sum
    intro v hv
    exact term_ge_scaled_minEntropy μ hK v
  have hconst : Finset.univ.sum (fun v : Fin K => μ.pmf v * minEntropy μ hK) = minEntropy μ hK := by
    calc
      Finset.univ.sum (fun v : Fin K => μ.pmf v * minEntropy μ hK)
          = (Finset.univ.sum fun v : Fin K => μ.pmf v) * minEntropy μ hK := by
              simpa [mul_comm, mul_left_comm, mul_assoc] using
                (Finset.sum_mul (s := Finset.univ) (f := fun v : Fin K => μ.pmf v) (a := minEntropy μ hK)).symm
      _ = minEntropy μ hK := by simp [μ.sum_one]
  linarith

/-- Uniform finite source on `Fin (n+1)`. -/
noncomputable def uniformFiniteSource (n : Nat) : FiniteSource (n + 1) where
  pmf := fun _ => ((n : ℝ) + 1)⁻¹
  nonneg := by intro _; positivity
  sum_one := by
    simp
    field_simp

theorem uniformFiniteSource_massBound (n : Nat) (v : Fin (n + 1)) :
    (uniformFiniteSource n).pmf v ≤ ((n : ℝ) + 1)⁻¹ := by
  simp [uniformFiniteSource]

theorem sourceEntropy_uniformFiniteSource (n : Nat) :
    sourceEntropy (uniformFiniteSource n) = Real.log (n + 1) := by
  unfold sourceEntropy uniformFiniteSource
  simp
  have hk : ((n : ℝ) + 1) ≠ 0 := by positivity
  field_simp [hk]

/-- States decoded exactly by a given observer/tag decoder. -/
def successSet {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : Finset (Fin K) :=
  Finset.univ.filter (fun v => decode (obs v) (tag v) = some v)

/-- Success probability under a finite source. -/
def successProb {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : ℝ :=
  (successSet obs tag decode).sum μ.pmf

/-- Error probability under a finite source. -/
def errorProb {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : ℝ :=
  1 - successProb μ obs tag decode

/-- States decoded incorrectly by a given observer/tag decoder. -/
def failureSet {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : Finset (Fin K) :=
  Finset.univ.filter (fun v => decode (obs v) (tag v) ≠ some v)

theorem successProb_nonneg
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    0 ≤ successProb μ obs tag decode := by
  unfold successProb
  exact Finset.sum_nonneg (by intro v hv; exact μ.nonneg v)

theorem successProb_le_one
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    successProb μ obs tag decode ≤ 1 := by
  unfold successProb successSet
  have hle : (Finset.univ.filter (fun v : Fin K => decode (obs v) (tag v) = some v)).sum μ.pmf
      ≤ Finset.univ.sum μ.pmf :=
    Finset.sum_le_univ_sum_of_nonneg (by intro v; exact μ.nonneg v)
  simpa [μ.sum_one] using hle

theorem errorProb_nonneg
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    0 ≤ errorProb μ obs tag decode := by
  unfold errorProb
  linarith [successProb_le_one μ obs tag decode]

theorem errorProb_le_one
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    errorProb μ obs tag decode ≤ 1 := by
  unfold errorProb
  linarith [successProb_nonneg μ obs tag decode]

noncomputable def successFailurePMF {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) : PMF Bool :=
  PMF.bernoulli (Real.toNNReal (errorProb μ obs tag decode))
    (by
      change (Real.toNNReal (errorProb μ obs tag decode) : ℝ) ≤ 1
      rw [Real.toNNReal_of_nonneg (errorProb_nonneg μ obs tag decode)]
      exact errorProb_le_one μ obs tag decode)

theorem pmfEntropy_successFailurePMF
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    pmfEntropy (successFailurePMF μ obs tag decode) = Real.binEntropy (errorProb μ obs tag decode) := by
  exact pmfEntropy_bernoulli_ofReal
    (errorProb_nonneg μ obs tag decode)
    (errorProb_le_one μ obs tag decode)

def pairFiber {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (y : Fin O × Fin T) : Finset (Fin K) :=
  Finset.univ.filter (fun v => (obs v, tag v) = y)

def pairMass {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (y : Fin O × Fin T) : ℝ :=
  (pairFiber obs tag y).sum μ.pmf

noncomputable def observationTagEntropy {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) : ℝ :=
  Finset.univ.sum (fun y : Fin O × Fin T => Real.negMulLog (pairMass μ obs tag y))

noncomputable def conditionalEntropyGivenPair {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) : ℝ :=
  Finset.univ.sum (fun y : Fin O × Fin T =>
    (pairFiber obs tag y).sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
      - Real.negMulLog (pairMass μ obs tag y))

noncomputable def mutualInfoSurrogate {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) : ℝ :=
  sourceEntropy μ - conditionalEntropyGivenPair μ obs tag

noncomputable def jointEntropySourcePair {K O T : Nat}
    (μ : FiniteSource K)
    (_obs : Fin K → Fin O)
    (_tag : Fin K → Fin T) : ℝ :=
  sourceEntropy μ

noncomputable def mutualInfoDeterministic {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) : ℝ :=
  sourceEntropy μ + observationTagEntropy μ obs tag - jointEntropySourcePair μ obs tag

theorem pairMass_nonneg {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (y : Fin O × Fin T) :
    0 ≤ pairMass μ obs tag y := by
  unfold pairMass
  exact Finset.sum_nonneg (by intro v hv; exact μ.nonneg v)

theorem pairMass_le_one {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (y : Fin O × Fin T) :
    pairMass μ obs tag y ≤ 1 := by
  unfold pairMass pairFiber
  have hle : (Finset.univ.filter (fun v : Fin K => (obs v, tag v) = y)).sum μ.pmf
      ≤ Finset.univ.sum μ.pmf :=
    Finset.sum_le_univ_sum_of_nonneg (by intro v; exact μ.nonneg v)
  simpa [μ.sum_one] using hle

theorem observationTagEntropy_nonneg {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    0 ≤ observationTagEntropy μ obs tag := by
  unfold observationTagEntropy
  apply Finset.sum_nonneg
  intro y hy
  exact Real.negMulLog_nonneg (pairMass_nonneg μ obs tag y) (pairMass_le_one μ obs tag y)

theorem sum_pairFiber_sum_eq_univ_sum
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (f : Fin K → ℝ) :
    Finset.univ.sum (fun y : Fin O × Fin T => (pairFiber obs tag y).sum f) = Finset.univ.sum f := by
  classical
  unfold pairFiber
  calc
    Finset.univ.sum (fun y : Fin O × Fin T =>
        (Finset.univ.filter (fun v : Fin K => (obs v, tag v) = y)).sum f)
        = Finset.univ.sum (fun y : Fin O × Fin T =>
            Finset.univ.sum (fun v : Fin K => if (obs v, tag v) = y then f v else 0)) := by
              simp [Finset.sum_filter]
    _ = Finset.univ.sum (fun v : Fin K =>
          Finset.univ.sum (fun y : Fin O × Fin T => if (obs v, tag v) = y then f v else 0)) := by
          rw [Finset.sum_comm]
    _ = Finset.univ.sum f := by
          apply Finset.sum_congr rfl
          intro v hv
          rw [Finset.sum_eq_single (obs v, tag v)]
          · simp
          · intro y hy hne
            simp [hne.symm]
          · intro hmem
            simp at hmem

theorem pairMass_sum_one {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    Finset.univ.sum (fun y : Fin O × Fin T => pairMass μ obs tag y) = 1 := by
  unfold pairMass
  rw [sum_pairFiber_sum_eq_univ_sum obs tag μ.pmf, μ.sum_one]

noncomputable def inducedPairPMF {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) : PMF (Fin O × Fin T) :=
  PMF.ofFintype (fun y => ENNReal.ofReal (pairMass μ obs tag y)) (by
    change Finset.univ.sum (fun y : Fin O × Fin T => ENNReal.ofReal (pairMass μ obs tag y)) = 1
    rw [← ENNReal.ofReal_sum_of_nonneg]
    · rw [pairMass_sum_one]
      norm_num
    · intro y hy
      exact pairMass_nonneg μ obs tag y)

theorem pmfEntropy_inducedPairPMF_eq_observationTagEntropy {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    pmfEntropy (inducedPairPMF μ obs tag) = observationTagEntropy μ obs tag := by
  unfold pmfEntropy inducedPairPMF observationTagEntropy
  simp [PMF.ofFintype_apply, pairMass_nonneg, Real.negMulLog]

noncomputable def pairFiniteSource {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) : FiniteSource (O * T) where
  pmf := fun i => pairMass μ obs tag ((finProdFinEquiv : Fin O × Fin T ≃ Fin (O * T)).symm i)
  nonneg := by
    intro i
    exact pairMass_nonneg μ obs tag _
  sum_one := by
    let e : Fin O × Fin T ≃ Fin (O * T) := finProdFinEquiv
    calc
      Finset.univ.sum (fun i : Fin (O * T) => pairMass μ obs tag (e.symm i))
          = Finset.univ.sum (fun y : Fin O × Fin T => pairMass μ obs tag y) := by
              exact Fintype.sum_equiv e.symm _ _ (fun y => rfl)
      _ = 1 := pairMass_sum_one μ obs tag

noncomputable def finiteSourcePMF {K : Nat} (μ : FiniteSource K) : PMF (Fin K) :=
  PMF.ofFintype (fun i => ENNReal.ofReal (μ.pmf i)) (by
    change Finset.univ.sum (fun i : Fin K => ENNReal.ofReal (μ.pmf i)) = 1
    rw [← ENNReal.ofReal_sum_of_nonneg]
    · rw [μ.sum_one]
      norm_num
    · intro i hi
      exact μ.nonneg i)

theorem pmfEntropy_finiteSourcePMF_eq_sourceEntropy {K : Nat} (μ : FiniteSource K) :
    pmfEntropy (finiteSourcePMF μ) = sourceEntropy μ := by
  unfold pmfEntropy finiteSourcePMF sourceEntropy
  simp [PMF.ofFintype_apply, μ.nonneg]

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

theorem successSet_exactOn
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    ExactOn obs tag decode (successSet obs tag decode) := by
  intro v hv
  exact (Finset.mem_filter.mp hv).2

theorem exactOn_clique_subsetEntropy_le_log_tags
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {S : Finset (Fin K)}
    (hexact : ExactOn obs tag decode S)
    (hs : IsClique obs S)
    (hScard : 0 < S.card) :
    S.sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
      ≤ Real.negMulLog (S.sum μ.pmf) + (S.sum μ.pmf) * Real.log T := by
  have hbase := subset_sourceEntropy_le_card μ S hScard
  have hcard : S.card ≤ T := exactOn_clique_card_le_tag_alphabet obs tag decode hexact hs
  have hTpos : 0 < T := by
    omega
  have hmass_nonneg : 0 ≤ S.sum μ.pmf := by
    exact Finset.sum_nonneg (by intro v hv; exact μ.nonneg v)
  have hlog : Real.log S.card ≤ Real.log T := by
    apply Real.log_le_log
    · exact_mod_cast hScard
    · exact_mod_cast hcard
  have hmul : (S.sum μ.pmf) * Real.log S.card ≤ (S.sum μ.pmf) * Real.log T := by
    exact mul_le_mul_of_nonneg_left hlog hmass_nonneg
  linarith

theorem successSet_clique_entropy_le_log_tags
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : IsClique obs (successSet obs tag decode))
    (hscard : 0 < (successSet obs tag decode).card) :
    (successSet obs tag decode).sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
      ≤ Real.negMulLog (successProb μ obs tag decode) + successProb μ obs tag decode * Real.log T := by
  have h := exactOn_clique_subsetEntropy_le_log_tags μ obs tag decode
      (successSet_exactOn obs tag decode) hs hscard
  simpa [successProb] using h

/-- Successful states must map injectively into observation/tag pairs. -/
theorem successSet_pair_injOn
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    Set.InjOn (fun v => (obs v, tag v)) {v | v ∈ successSet obs tag decode} := by
  intro v hv w hw hpair
  have hv' : decode (obs v) (tag v) = some v := by
    exact (Finset.mem_filter.mp hv).2
  have hw' : decode (obs w) (tag w) = some w := by
    exact (Finset.mem_filter.mp hw).2
  have hdecode : some v = some w := by
    calc
      some v = decode (obs v) (tag v) := by simpa using hv'.symm
      _ = decode (obs w) (tag w) := by simpa using congrArg (fun p : Fin O × Fin T => decode p.1 p.2) hpair
      _ = some w := hw'
  simpa using hdecode

/-- Number of exactly recovered states is bounded by the observation/tag budget. -/
theorem successSet_card_le_budget
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    (successSet obs tag decode).card ≤ O * T := by
  classical
  let s := successSet obs tag decode
  let pairMap : Fin K → Fin O × Fin T := fun v => (obs v, tag v)
  have hinjOn : Set.InjOn pairMap {v | v ∈ s} := successSet_pair_injOn obs tag decode
  have hcardImg : s.card = (s.image pairMap).card := by
    symm
    exact Finset.card_image_of_injOn (by
      intro a ha b hb hab
      exact hinjOn ha hb hab)
  calc
    s.card = (s.image pairMap).card := hcardImg
    _ ≤ (Finset.univ : Finset (Fin O × Fin T)).card := by
      apply Finset.card_le_card
      intro x hx
      simp
    _ = O * T := by simp

/-- Success probability is bounded by a point-mass ceiling times the budget. -/
theorem successProb_le_budget_mul_massBound
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {m : ℝ}
    (hm_nonneg : 0 ≤ m)
    (hm : ∀ v, μ.pmf v ≤ m) :
    successProb μ obs tag decode ≤ ((O * T : Nat) : ℝ) * m := by
  have hsum : successProb μ obs tag decode ≤
      (successSet obs tag decode).sum (fun _ => m) := by
    unfold successProb
    apply Finset.sum_le_sum
    intro v hv
    exact hm v
  have hcard : ((successSet obs tag decode).card : ℝ) ≤ ((O * T : Nat) : ℝ) := by
    exact_mod_cast successSet_card_le_budget obs tag decode
  calc
    successProb μ obs tag decode ≤ (successSet obs tag decode).sum (fun _ => m) := hsum
    _ = ((successSet obs tag decode).card : ℝ) * m := by simp
    _ ≤ ((O * T : Nat) : ℝ) * m := by
      exact mul_le_mul_of_nonneg_right hcard hm_nonneg

/-- Error probability lower bound from a finite budget and point-mass ceiling. -/
theorem errorProb_ge_one_sub_budget_mul_massBound
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {m : ℝ}
    (hm_nonneg : 0 ≤ m)
    (hm : ∀ v, μ.pmf v ≤ m) :
    1 - (((O * T : Nat) : ℝ) * m) ≤ errorProb μ obs tag decode := by
  unfold errorProb
  have hsucc := successProb_le_budget_mul_massBound μ obs tag decode hm_nonneg hm
  linarith

/-- Intrinsic success bound using the source's own maximum atom mass. -/
theorem successProb_le_budget_mul_maxMass
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    successProb μ obs tag decode ≤ ((O * T : Nat) : ℝ) * maxMass μ hK := by
  apply successProb_le_budget_mul_massBound
  · have h0 := μ.nonneg ⟨0, hK⟩
    exact le_trans h0 (pmf_le_maxMass μ hK ⟨0, hK⟩)
  · intro v
    exact pmf_le_maxMass μ hK v

theorem successProb_le_successSetCard_mul_maxMass
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    successProb μ obs tag decode ≤ ((successSet obs tag decode).card : ℝ) * maxMass μ hK := by
  have hsum : successProb μ obs tag decode ≤
      (successSet obs tag decode).sum (fun _ => maxMass μ hK) := by
    unfold successProb
    apply Finset.sum_le_sum
    intro v hv
    exact pmf_le_maxMass μ hK v
  calc
    successProb μ obs tag decode ≤ (successSet obs tag decode).sum (fun _ => maxMass μ hK) := hsum
    _ = ((successSet obs tag decode).card : ℝ) * maxMass μ hK := by simp

theorem successSetCard_ge_successProb_div_maxMass
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    successProb μ obs tag decode / maxMass μ hK ≤ ((successSet obs tag decode).card : ℝ) := by
  have hmasspos : 0 < maxMass μ hK := maxMass_pos μ hK
  have hbound := successProb_le_successSetCard_mul_maxMass μ hK obs tag decode
  exact (div_le_iff₀ hmasspos).2 <| by
    simpa [mul_comm, mul_left_comm, mul_assoc] using hbound

/-- Intrinsic error lower bound using the source's own maximum atom mass. -/
theorem errorProb_ge_one_sub_budget_mul_maxMass
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    1 - (((O * T : Nat) : ℝ) * maxMass μ hK) ≤ errorProb μ obs tag decode := by
  apply errorProb_ge_one_sub_budget_mul_massBound
  · have h0 := μ.nonneg ⟨0, hK⟩
    exact le_trans h0 (pmf_le_maxMass μ hK ⟨0, hK⟩)
  · intro v
    exact pmf_le_maxMass μ hK v

/-- Weak-Fano-style arbitrary-source converse via intrinsic max atom mass. -/
theorem weak_fano_maxMass_lower_bound
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {ε : ℝ}
    (herr : errorProb μ obs tag decode ≤ ε) :
    1 - ε ≤ ((O * T : Nat) : ℝ) * maxMass μ hK := by
  have hsucc : successProb μ obs tag decode ≤ ((O * T : Nat) : ℝ) * maxMass μ hK :=
    successProb_le_budget_mul_maxMass μ hK obs tag decode
  unfold errorProb at herr
  linarith

theorem successProb_le_budget_times_exp_neg_minEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    successProb μ obs tag decode
      ≤ ((O * T : Nat) : ℝ) * Real.exp (- minEntropy μ hK) := by
  rw [minEntropy_eq_neg_log_maxMass]
  have hbase := successProb_le_budget_mul_maxMass μ hK obs tag decode
  have hmaxpos : 0 < maxMass μ hK := maxMass_pos μ hK
  have hexp : Real.exp (Real.log (maxMass μ hK)) = maxMass μ hK := by
    exact Real.exp_log hmaxpos
  simpa [hexp] using hbase

theorem errorProb_ge_one_sub_budget_times_exp_neg_minEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    1 - (((O * T : Nat) : ℝ) * Real.exp (- minEntropy μ hK)) ≤ errorProb μ obs tag decode := by
  rw [minEntropy_eq_neg_log_maxMass]
  have hbase := errorProb_ge_one_sub_budget_mul_maxMass μ hK obs tag decode
  have hmaxpos : 0 < maxMass μ hK := maxMass_pos μ hK
  have hexp : Real.exp (Real.log (maxMass μ hK)) = maxMass μ hK := by
    exact Real.exp_log hmaxpos
  simpa [hexp] using hbase

theorem successSetCard_ge_one_sub_error_times_exp_minEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {ε : ℝ}
    (herr : errorProb μ obs tag decode ≤ ε) :
    (1 - ε) * Real.exp (minEntropy μ hK) ≤ ((successSet obs tag decode).card : ℝ) := by
  have hsucc_ge : 1 - ε ≤ successProb μ obs tag decode := by
    unfold errorProb at herr
    linarith
  have hbase : (1 - ε) / maxMass μ hK ≤ ((successSet obs tag decode).card : ℝ) := by
    calc
      (1 - ε) / maxMass μ hK ≤ successProb μ obs tag decode / maxMass μ hK := by
        exact div_le_div_of_nonneg_right hsucc_ge (le_of_lt (maxMass_pos μ hK))
      _ ≤ ((successSet obs tag decode).card : ℝ) :=
        successSetCard_ge_successProb_div_maxMass μ hK obs tag decode
  have hmaxpos : 0 < maxMass μ hK := maxMass_pos μ hK
  have hexp : Real.exp (minEntropy μ hK) = (maxMass μ hK)⁻¹ := by
    rw [minEntropy_eq_neg_log_maxMass, Real.exp_neg, Real.exp_log hmaxpos]
  simpa [div_eq_mul_inv, hexp, mul_comm, mul_left_comm, mul_assoc] using hbase

theorem budget_ge_one_sub_error_times_exp_minEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {ε : ℝ}
    (herr : errorProb μ obs tag decode ≤ ε) :
    (1 - ε) * Real.exp (minEntropy μ hK) ≤ ((O * T : Nat) : ℝ) := by
  have hlow : (1 - ε) * Real.exp (minEntropy μ hK) ≤ ((successSet obs tag decode).card : ℝ) :=
    successSetCard_ge_one_sub_error_times_exp_minEntropy μ hK obs tag decode herr
  have hupp : ((successSet obs tag decode).card : ℝ) ≤ ((O * T : Nat) : ℝ) := by
    exact_mod_cast successSet_card_le_budget obs tag decode
  linarith

theorem minEntropy_le_log_budget_sub_log_one_sub_error
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {ε : ℝ}
    (herr : errorProb μ obs tag decode ≤ ε)
    (hε : ε < 1) :
    minEntropy μ hK ≤ Real.log ((O * T : Nat) : ℝ) - Real.log (1 - ε) := by
  have hbudget := budget_ge_one_sub_error_times_exp_minEntropy μ hK obs tag decode herr
  have honepos : 0 < 1 - ε := by
    linarith
  have hleftpos : 0 < (1 - ε) * Real.exp (minEntropy μ hK) := by
    exact mul_pos honepos (Real.exp_pos _)
  have hBpos : 0 < ((O * T : Nat) : ℝ) := lt_of_lt_of_le hleftpos hbudget
  have hlog : Real.log ((1 - ε) * Real.exp (minEntropy μ hK)) ≤ Real.log ((O * T : Nat) : ℝ) := by
    exact Real.log_le_log hleftpos hbudget
  rw [Real.log_mul honepos.ne' (Real.exp_pos _).ne', Real.log_exp] at hlog
  linarith

theorem successSetCard_ge_one_sub_error_over_maxMass
    {K O T : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {ε : ℝ}
    (herr : errorProb μ obs tag decode ≤ ε) :
    (1 - ε) / maxMass μ hK ≤ ((successSet obs tag decode).card : ℝ) := by
  have hsucc_ge : 1 - ε ≤ successProb μ obs tag decode := by
    unfold errorProb at herr
    linarith
  calc
    (1 - ε) / maxMass μ hK ≤ successProb μ obs tag decode / maxMass μ hK := by
      exact div_le_div_of_nonneg_right hsucc_ge (le_of_lt (maxMass_pos μ hK))
    _ ≤ ((successSet obs tag decode).card : ℝ) :=
      successSetCard_ge_successProb_div_maxMass μ hK obs tag decode

theorem sourceEntropy_le_success_failure_partition
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : 0 < (failureSet obs tag decode).card) :
    sourceEntropy μ
      ≤ Real.binEntropy (errorProb μ obs tag decode)
        + successProb μ obs tag decode * Real.log ((successSet obs tag decode).card)
        + errorProb μ obs tag decode * Real.log ((failureSet obs tag decode).card) := by
  have hS := subset_sourceEntropy_le_card μ (successSet obs tag decode) hs
  have hF := subset_sourceEntropy_le_card μ (failureSet obs tag decode) hf
  have hsplit : sourceEntropy μ
      = (successSet obs tag decode).sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
        + (failureSet obs tag decode).sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹) := by
    unfold sourceEntropy successSet failureSet
    simpa using (Finset.sum_filter_add_sum_filter_not
      (s := Finset.univ)
      (p := fun v : Fin K => decode (obs v) (tag v) = some v)
      (f := fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)).symm
  have hbin :
      Real.negMulLog (successProb μ obs tag decode) + Real.negMulLog (errorProb μ obs tag decode)
        = Real.binEntropy (errorProb μ obs tag decode) := by
    unfold errorProb
    rw [Real.binEntropy_eq_negMulLog_add_negMulLog_one_sub]
    ring_nf
  rw [hsplit]
  have hS' :
      (successSet obs tag decode).sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
        ≤ Real.negMulLog (successProb μ obs tag decode)
          + successProb μ obs tag decode * Real.log ((successSet obs tag decode).card) := by
    simpa [successProb] using hS
  have hfeq : (failureSet obs tag decode).sum μ.pmf = errorProb μ obs tag decode :=
    (errorProb_eq_failureSet_sum μ obs tag decode).symm
  have hF' :
      (failureSet obs tag decode).sum (fun v => μ.pmf v * Real.log (μ.pmf v)⁻¹)
        ≤ Real.negMulLog (errorProb μ obs tag decode)
          + errorProb μ obs tag decode * Real.log ((failureSet obs tag decode).card) := by
    simpa [hfeq] using hF
  linarith

theorem fano_arbitrary_budgeted
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K) :
    sourceEntropy μ
      ≤ Real.binEntropy (errorProb μ obs tag decode)
        + successProb μ obs tag decode * Real.log (O * T)
        + errorProb μ obs tag decode * Real.log (K - 1) := by
  have hsumcard := successSet_card_add_failureSet_card obs tag decode
  have hfailpos : 0 < (failureSet obs tag decode).card := by
    omega
  have hpart := sourceEntropy_le_success_failure_partition μ obs tag decode hs hfailpos
  have hS : Real.log ((successSet obs tag decode).card) ≤ Real.log (O * T) := by
    apply Real.log_le_log
    · exact_mod_cast hs
    · exact_mod_cast successSet_card_le_budget obs tag decode
  have hFcard : (failureSet obs tag decode).card ≤ K - 1 := by
    omega
  have hK1pos : 0 < K - 1 := by
    omega
  have hF : Real.log ((failureSet obs tag decode).card) ≤ Real.log (K - 1) := by
    apply Real.log_le_log
    · exact_mod_cast hfailpos
    · have hsR : (1 : ℝ) ≤ ((successSet obs tag decode).card : ℝ) := by
        exact_mod_cast hs
      have hsumR : ((successSet obs tag decode).card : ℝ) + ((failureSet obs tag decode).card : ℝ) = K := by
        exact_mod_cast hsumcard
      have hFcardR : ((failureSet obs tag decode).card : ℝ) ≤ (K : ℝ) - 1 := by
        linarith
      exact hFcardR
  have hsucc_nonneg : 0 ≤ successProb μ obs tag decode := by
    unfold successProb
    exact Finset.sum_nonneg (by intro v hv; exact μ.nonneg v)
  have herr_nonneg : 0 ≤ errorProb μ obs tag decode := by
    rw [errorProb_eq_failureSet_sum μ obs tag decode]
    exact Finset.sum_nonneg (by intro v hv; exact μ.nonneg v)
  have hmulS :
      successProb μ obs tag decode * Real.log ((successSet obs tag decode).card)
        ≤ successProb μ obs tag decode * Real.log (O * T) := by
    exact mul_le_mul_of_nonneg_left hS hsucc_nonneg
  have hmulF :
      errorProb μ obs tag decode * Real.log ((failureSet obs tag decode).card)
        ≤ errorProb μ obs tag decode * Real.log (K - 1) := by
    exact mul_le_mul_of_nonneg_left hF herr_nonneg
  linarith

theorem fano_arbitrary_observation_only
    {K : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    (hs : 0 < (successSet obs (fun _ => (0 : Fin 1)) decode).card)
    (hf : (successSet obs (fun _ => (0 : Fin 1)) decode).card < K) :
    sourceEntropy μ
      ≤ Real.qaryEntropy K (errorProb μ obs (fun _ => (0 : Fin 1)) decode) := by
  have hbase := fano_arbitrary_budgeted μ obs (fun _ => (0 : Fin 1)) decode hs hf
  simpa [Real.qaryEntropy, add_comm, add_left_comm, add_assoc, one_mul, zero_mul] using hbase

theorem fano_arbitrary_conditional_style
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K) :
    sourceEntropy μ - successProb μ obs tag decode * Real.log (O * T)
      ≤ Real.binEntropy (errorProb μ obs tag decode)
        + errorProb μ obs tag decode * Real.log (K - 1) := by
  have h := fano_arbitrary_budgeted μ obs tag decode hs hf
  linarith

theorem fano_arbitrary_conditional_observation_only
    {K : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    (hs : 0 < (successSet obs (fun _ => (0 : Fin 1)) decode).card)
    (hf : (successSet obs (fun _ => (0 : Fin 1)) decode).card < K) :
    sourceEntropy μ
      ≤ Real.binEntropy (errorProb μ obs (fun _ => (0 : Fin 1)) decode)
        + errorProb μ obs (fun _ => (0 : Fin 1)) decode * Real.log (K - 1) := by
  have h := fano_arbitrary_conditional_style μ obs (fun _ => (0 : Fin 1)) decode hs hf
  simpa using h

theorem conditionalEntropyGivenPair_le_fano_arbitrary
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K) :
    conditionalEntropyGivenPair μ obs tag
      ≤ Real.binEntropy (errorProb μ obs tag decode)
        + successProb μ obs tag decode * Real.log (O * T)
        + errorProb μ obs tag decode * Real.log (K - 1) := by
  exact (conditionalEntropyGivenPair_le_sourceEntropy μ obs tag).trans
    (fano_arbitrary_budgeted μ obs tag decode hs hf)

theorem conditionalEntropyGivenPair_le_fano_observation_only
    {K : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    (hs : 0 < (successSet obs (fun _ => (0 : Fin 1)) decode).card)
    (hf : (successSet obs (fun _ => (0 : Fin 1)) decode).card < K) :
    conditionalEntropyGivenPair μ obs (fun _ => (0 : Fin 1))
      ≤ Real.binEntropy (errorProb μ obs (fun _ => (0 : Fin 1)) decode)
        + errorProb μ obs (fun _ => (0 : Fin 1)) decode * Real.log (K - 1) := by
  exact (conditionalEntropyGivenPair_le_sourceEntropy μ obs (fun _ => (0 : Fin 1))).trans
    (fano_arbitrary_conditional_observation_only μ obs decode hs hf)

theorem decodedOutputEntropy_le_success_failure_partition
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : 0 < (failureSet obs tag decode).card) :
    decodedOutputEntropy μ obs tag decode
      ≤ Real.binEntropy (errorProb μ obs tag decode)
        + successProb μ obs tag decode * Real.log ((successSet obs tag decode).card)
        + errorProb μ obs tag decode * Real.log ((failureSet obs tag decode).card) := by
  calc
    decodedOutputEntropy μ obs tag decode ≤ sourceEntropy μ := by
      linarith [decodedOutputEntropy_le_mutualInfoDeterministic μ obs tag decode,
        mutualInfoDeterministic_le_sourceEntropy μ obs tag]
    _ ≤ Real.binEntropy (errorProb μ obs tag decode)
          + successProb μ obs tag decode * Real.log ((successSet obs tag decode).card)
          + errorProb μ obs tag decode * Real.log ((failureSet obs tag decode).card) :=
        sourceEntropy_le_success_failure_partition μ obs tag decode hs hf

theorem decodedOutputEntropy_fano_budgeted
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K) :
    decodedOutputEntropy μ obs tag decode
      ≤ Real.binEntropy (errorProb μ obs tag decode)
        + successProb μ obs tag decode * Real.log (O * T)
        + errorProb μ obs tag decode * Real.log (K - 1) := by
  calc
    decodedOutputEntropy μ obs tag decode ≤ sourceEntropy μ := by
      linarith [decodedOutputEntropy_le_mutualInfoDeterministic μ obs tag decode,
        mutualInfoDeterministic_le_sourceEntropy μ obs tag]
    _ ≤ Real.binEntropy (errorProb μ obs tag decode)
          + successProb μ obs tag decode * Real.log (O * T)
          + errorProb μ obs tag decode * Real.log (K - 1) :=
        fano_arbitrary_budgeted μ obs tag decode hs hf

theorem decodedOutputEntropy_fano_observation_only
    {K : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    (hs : 0 < (successSet obs (fun _ => (0 : Fin 1)) decode).card)
    (hf : (successSet obs (fun _ => (0 : Fin 1)) decode).card < K) :
    decodedOutputEntropy μ obs (fun _ => (0 : Fin 1)) decode
      ≤ Real.binEntropy (errorProb μ obs (fun _ => (0 : Fin 1)) decode)
        + errorProb μ obs (fun _ => (0 : Fin 1)) decode * Real.log (K - 1) := by
  have h := decodedOutputEntropy_fano_budgeted μ obs (fun _ => (0 : Fin 1)) decode hs hf
  simpa [Real.log_one, add_comm, add_left_comm, add_assoc] using h

/-- Uniform-source error lower bound from observation/tag budget. -/
theorem errorProb_uniform_ge_one_sub_budget_div_card
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1))) :
    1 - (((O * T : Nat) : ℝ) * (((n : ℝ) + 1)⁻¹))
      ≤ errorProb (uniformFiniteSource n) obs tag decode := by
  apply errorProb_ge_one_sub_budget_mul_massBound
  · positivity
  · intro v
    exact uniformFiniteSource_massBound n v

/-- Under the uniform source, success probability is exactly the success-set fraction. -/
theorem successProb_uniform_eq_card_div_card
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1))) :
    successProb (uniformFiniteSource n) obs tag decode =
      ((successSet obs tag decode).card : ℝ) * (((n : ℝ) + 1)⁻¹) := by
  unfold successProb successSet uniformFiniteSource
  simp

/-- Under the uniform source, error probability is one minus the success-set fraction. -/
theorem errorProb_uniform_eq_one_sub_card_div_card
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1))) :
    errorProb (uniformFiniteSource n) obs tag decode =
      1 - ((successSet obs tag decode).card : ℝ) * (((n : ℝ) + 1)⁻¹) := by
  unfold errorProb
  rw [successProb_uniform_eq_card_div_card]

/-- Uniform-source success probability is bounded by the observation/tag budget fraction. -/
theorem successProb_uniform_le_budget_div_card
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1))) :
    successProb (uniformFiniteSource n) obs tag decode
      ≤ ((O * T : Nat) : ℝ) * (((n : ℝ) + 1)⁻¹) := by
  apply successProb_le_budget_mul_massBound
  · positivity
  · intro v
    exact uniformFiniteSource_massBound n v

/-- Weak Fano-style converse for a uniform finite source. -/
theorem weak_fano_uniform_budget_lower_bound
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1)))
    {ε : ℝ}
    (herr : errorProb (uniformFiniteSource n) obs tag decode ≤ ε) :
    1 - ε ≤ ((O * T : Nat) : ℝ) * (((n : ℝ) + 1)⁻¹) := by
  have hsucc : successProb (uniformFiniteSource n) obs tag decode
      ≤ ((O * T : Nat) : ℝ) * (((n : ℝ) + 1)⁻¹) :=
    successProb_uniform_le_budget_div_card n obs tag decode
  unfold errorProb at herr
  linarith

/-- Weak-Fano cardinality form: low error forces a large exactly recovered subset. -/
theorem weak_fano_uniform_successSet_lower_bound
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1)))
    {ε : ℝ}
    (herr : errorProb (uniformFiniteSource n) obs tag decode ≤ ε) :
    1 - ε ≤ ((successSet obs tag decode).card : ℝ) * (((n : ℝ) + 1)⁻¹) := by
  rw [errorProb_uniform_eq_one_sub_card_div_card] at herr
  linarith

/-- Weak-Fano budget form via exact success-set cardinality and the budget bound. -/
theorem weak_fano_uniform_via_successSet
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1)))
    {ε : ℝ}
    (herr : errorProb (uniformFiniteSource n) obs tag decode ≤ ε) :
    1 - ε ≤ ((O * T : Nat) : ℝ) * (((n : ℝ) + 1)⁻¹) := by
  have hsucc : 1 - ε ≤ ((successSet obs tag decode).card : ℝ) * (((n : ℝ) + 1)⁻¹) :=
    weak_fano_uniform_successSet_lower_bound n obs tag decode herr
  have hcard : ((successSet obs tag decode).card : ℝ) ≤ ((O * T : Nat) : ℝ) := by
    exact_mod_cast successSet_card_le_budget obs tag decode
  have hmul : ((successSet obs tag decode).card : ℝ) * (((n : ℝ) + 1)⁻¹)
      ≤ ((O * T : Nat) : ℝ) * (((n : ℝ) + 1)⁻¹) := by
    apply mul_le_mul_of_nonneg_right hcard
    positivity
  linarith

/-- Entropy identity for the success/failure partition under a uniform source. -/
theorem uniform_success_failure_partition_entropy
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1)))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < n + 1) :
    Real.binEntropy (errorProb (uniformFiniteSource n) obs tag decode)
      + successProb (uniformFiniteSource n) obs tag decode
          * Real.log ((successSet obs tag decode).card)
      + errorProb (uniformFiniteSource n) obs tag decode
          * Real.log ((n + 1) - (successSet obs tag decode).card)
      = Real.log (n + 1) := by
  let s : ℝ := ((successSet obs tag decode).card : ℝ)
  let f : ℝ := (((n + 1) - (successSet obs tag decode).card : Nat) : ℝ)
  have hsR : 0 < s := by
    unfold s
    exact_mod_cast hs
  have hfR : 0 < f := by
    unfold f
    exact_mod_cast Nat.sub_pos_of_lt hf
  have hs_prob : successProb (uniformFiniteSource n) obs tag decode = s * (((n : ℝ) + 1)⁻¹) := by
    unfold s
    simpa using successProb_uniform_eq_card_div_card n obs tag decode
  have hsum : s + f = (n : ℝ) + 1 := by
    have hcast : ((((n + 1) - (successSet obs tag decode).card : Nat) : ℝ))
        = ((n + 1 : Nat) : ℝ) - ((successSet obs tag decode).card : ℝ) := by
      exact_mod_cast (Nat.cast_sub (Nat.le_of_lt hf))
    unfold s f
    rw [hcast]
    norm_num
  have hf_prob : errorProb (uniformFiniteSource n) obs tag decode = f * (((n : ℝ) + 1)⁻¹) := by
    unfold errorProb
    rw [hs_prob]
    have hk : (n : ℝ) + 1 ≠ 0 := by positivity
    have hf_eq : f = (n : ℝ) + 1 - s := by linarith [hsum]
    rw [hf_eq]
    field_simp [hk]
  have hpart := binEntropy_partition_identity hsR hfR
  rw [hf_prob, hs_prob]
  have hcast : ((((n + 1) - (successSet obs tag decode).card : Nat) : ℝ))
      = ((n + 1 : Nat) : ℝ) - ((successSet obs tag decode).card : ℝ) := by
    exact_mod_cast (Nat.cast_sub (Nat.le_of_lt hf))
  simpa [hsum, hcast, s, f, div_eq_mul_inv, mul_comm, mul_left_comm, mul_assoc] using hpart

/-- Finite Fano-style inequality for a uniform source with observation/tag budget. -/
theorem fano_uniform_budgeted
    {O T : Nat}
    (n : Nat)
    (obs : Fin (n + 1) → Fin O)
    (tag : Fin (n + 1) → Fin T)
    (decode : Fin O → Fin T → Option (Fin (n + 1)))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < n + 1) :
    Real.log (n + 1)
      ≤ Real.binEntropy (errorProb (uniformFiniteSource n) obs tag decode)
        + successProb (uniformFiniteSource n) obs tag decode * Real.log (O * T)
        + errorProb (uniformFiniteSource n) obs tag decode * Real.log n := by
  have hpart := uniform_success_failure_partition_entropy n obs tag decode hs hf
  have hS : Real.log ((successSet obs tag decode).card) ≤ Real.log (O * T) := by
    apply Real.log_le_log
    · exact_mod_cast hs
    · exact_mod_cast successSet_card_le_budget obs tag decode
  have hFcard : (n + 1) - (successSet obs tag decode).card ≤ n := by
    omega
  have hF : Real.log ((n + 1) - (successSet obs tag decode).card) ≤ Real.log n := by
    have hcastF : ((((n + 1) - (successSet obs tag decode).card : Nat) : ℝ))
        = (n : ℝ) + 1 - ((successSet obs tag decode).card : ℝ) := by
      exact_mod_cast (Nat.cast_sub (Nat.le_of_lt hf))
    have hFpos : (0 : ℝ) < (n : ℝ) + 1 - ((successSet obs tag decode).card : ℝ) := by
      rw [← hcastF]
      exact_mod_cast Nat.sub_pos_of_lt hf
    have hcard_ge1 : (1 : ℝ) ≤ ((successSet obs tag decode).card : ℝ) := by
      exact_mod_cast hs
    have hFle : (n : ℝ) + 1 - ((successSet obs tag decode).card : ℝ) ≤ (n : ℝ) := by
      linarith
    apply Real.log_le_log
    · exact hFpos
    · exact hFle
  have hsucc_nonneg : 0 ≤ successProb (uniformFiniteSource n) obs tag decode := by
    rw [successProb_uniform_eq_card_div_card]
    positivity
  have herr_nonneg : 0 ≤ errorProb (uniformFiniteSource n) obs tag decode := by
    rw [errorProb_uniform_eq_one_sub_card_div_card]
    have hcard : (((successSet obs tag decode).card : ℝ)) ≤ (n : ℝ) + 1 := by
      exact_mod_cast Nat.le_of_lt hf
    have hkpos : 0 < (n : ℝ) + 1 := by positivity
    have hratio : ((successSet obs tag decode).card : ℝ) * ((n : ℝ) + 1)⁻¹ ≤ 1 := by
      have hmul : ((successSet obs tag decode).card : ℝ) * ((n : ℝ) + 1)⁻¹
          ≤ ((n : ℝ) + 1) * ((n : ℝ) + 1)⁻¹ := by
        apply mul_le_mul_of_nonneg_right hcard
        positivity
      have hkne : (n : ℝ) + 1 ≠ 0 := ne_of_gt hkpos
      simpa [hkne] using hmul
    linarith
  have hmulS : successProb (uniformFiniteSource n) obs tag decode * Real.log ((successSet obs tag decode).card)
      ≤ successProb (uniformFiniteSource n) obs tag decode * Real.log (O * T) := by
    exact mul_le_mul_of_nonneg_left hS hsucc_nonneg
  have hmulF : errorProb (uniformFiniteSource n) obs tag decode * Real.log ((n + 1) - (successSet obs tag decode).card)
      ≤ errorProb (uniformFiniteSource n) obs tag decode * Real.log n := by
    exact mul_le_mul_of_nonneg_left hF herr_nonneg
  linarith

/-- Observation-only specialization of the finite Fano-style inequality. -/
theorem fano_uniform_observation_only
    (n : Nat)
    (obs : Fin (n + 1) → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin (n + 1)))
    (hs : 0 < (successSet obs (fun _ => (0 : Fin 1)) decode).card)
    (hf : (successSet obs (fun _ => (0 : Fin 1)) decode).card < n + 1) :
    Real.log (n + 1)
      ≤ Real.qaryEntropy (n + 1)
          (errorProb (uniformFiniteSource n) obs (fun _ => (0 : Fin 1)) decode) := by
  have hbase := fano_uniform_budgeted n obs (fun _ => (0 : Fin 1)) decode hs hf
  simpa [Real.qaryEntropy, add_comm, add_left_comm, add_assoc, one_mul, zero_mul] using hbase

/-- Observation-only success probability is at most `1 / K` for a uniform `K`-state source. -/
theorem observation_only_successProb_uniform_le_one_div_card
    (n : Nat)
    (obs : Fin (n + 1) → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin (n + 1))) :
    successProb (uniformFiniteSource n) obs (fun _ => (0 : Fin 1)) decode ≤ ((n : ℝ) + 1)⁻¹ := by
  have h := successProb_uniform_le_budget_div_card n obs (fun _ => (0 : Fin 1)) decode
  simpa using h

/-- Observation-only error probability is at least `1 - 1/K` for a uniform `K`-state source. -/
theorem observation_only_errorProb_uniform_ge_one_sub_one_div_card
    (n : Nat)
    (obs : Fin (n + 1) → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin (n + 1))) :
    1 - ((n : ℝ) + 1)⁻¹ ≤ errorProb (uniformFiniteSource n) obs (fun _ => (0 : Fin 1)) decode := by
  have h := errorProb_uniform_ge_one_sub_budget_div_card n obs (fun _ => (0 : Fin 1)) decode
  simpa using h

/-- Vanishing error as an elementary quantifier definition. -/
def vanishingError (err : Nat → ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, err n < ε

/-- A uniform positive lower bound rules out vanishing error. -/
theorem not_vanishingError_of_lower_bound
    (err : Nat → ℝ)
    {ε : ℝ}
    (hε : 0 < ε)
    (hlb : ∀ n, ε ≤ err n) :
    ¬ vanishingError err := by
  intro hvan
  obtain ⟨N, hN⟩ := hvan ε hε
  have hlt := hN N le_rfl
  have hge := hlb N
  linarith

/-- Uniform-source finite-family nonvanishing theorem under a budget-ratio gap. -/
theorem uniform_family_not_vanishing_of_budget_ratio_gap
    (O T : Nat → Nat)
    (obs : ∀ n, Fin (n + 1) → Fin (O n))
    (tag : ∀ n, Fin (n + 1) → Fin (T n))
    (decode : ∀ n, Fin (O n) → Fin (T n) → Option (Fin (n + 1)))
    {ε : ℝ}
    (hε : 0 < ε)
    (hgap : ∀ n, (((O n) * (T n) : Nat) : ℝ) * (((n : ℝ) + 1)⁻¹) ≤ 1 - ε) :
    ¬ vanishingError
      (fun n => errorProb (uniformFiniteSource n) (obs n) (tag n) (decode n)) := by
  apply not_vanishingError_of_lower_bound _ hε
  intro n
  have herr : 1 - (((O n) * (T n) : Nat) : ℝ) * (((n : ℝ) + 1)⁻¹)
      ≤ errorProb (uniformFiniteSource n) (obs n) (tag n) (decode n) :=
    errorProb_uniform_ge_one_sub_budget_div_card n (obs n) (tag n) (decode n)
  linarith [hgap n, herr]

/-- Finite-family nonvanishing theorem under a uniform budget/mass deficit. -/
theorem family_not_vanishing_of_budget_mass_gap
    (K O T : Nat → Nat)
    (μ : ∀ n, FiniteSource (K n))
    (obs : ∀ n, Fin (K n) → Fin (O n))
    (tag : ∀ n, Fin (K n) → Fin (T n))
    (decode : ∀ n, Fin (O n) → Fin (T n) → Option (Fin (K n)))
    (m : Nat → ℝ)
    {ε : ℝ}
    (hε : 0 < ε)
    (hm_nonneg : ∀ n, 0 ≤ m n)
    (hm : ∀ n v, (μ n).pmf v ≤ m n)
    (hgap : ∀ n, (((O n) * (T n) : Nat) : ℝ) * m n ≤ 1 - ε) :
    ¬ vanishingError (fun n => errorProb (μ n) (obs n) (tag n) (decode n)) := by
  apply not_vanishingError_of_lower_bound _ hε
  intro n
  have herr : 1 - ((((O n) * (T n) : Nat) : ℝ) * m n)
      ≤ errorProb (μ n) (obs n) (tag n) (decode n) :=
    errorProb_ge_one_sub_budget_mul_massBound (μ n) (obs n) (tag n) (decode n)
      (hm_nonneg n) (hm n)
  linarith [hgap n, herr]

/-- Finite-family nonvanishing theorem using intrinsic maximum atom mass. -/
theorem family_not_vanishing_of_budget_maxMass_gap
    (K O T : Nat → Nat)
    (μ : ∀ n, FiniteSource (K n))
    (hK : ∀ n, 0 < K n)
    (obs : ∀ n, Fin (K n) → Fin (O n))
    (tag : ∀ n, Fin (K n) → Fin (T n))
    (decode : ∀ n, Fin (O n) → Fin (T n) → Option (Fin (K n)))
    {ε : ℝ}
    (hε : 0 < ε)
    (hgap : ∀ n, (((O n) * (T n) : Nat) : ℝ) * maxMass (μ n) (hK n) ≤ 1 - ε) :
    ¬ vanishingError (fun n => errorProb (μ n) (obs n) (tag n) (decode n)) := by
  apply not_vanishingError_of_lower_bound _ hε
  intro n
  have herr : 1 - ((((O n) * (T n) : Nat) : ℝ) * maxMass (μ n) (hK n))
      ≤ errorProb (μ n) (obs n) (tag n) (decode n) :=
    errorProb_ge_one_sub_budget_mul_maxMass (μ n) (hK n) (obs n) (tag n) (decode n)
  linarith [hgap n, herr]

theorem observation_only_successProb_le_maxMass
    {K : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K)) :
    successProb μ obs (fun _ => (0 : Fin 1)) decode ≤ maxMass μ hK := by
  have h := successProb_le_budget_mul_maxMass μ hK obs (fun _ => (0 : Fin 1)) decode
  simpa using h

theorem observation_only_errorProb_ge_one_sub_maxMass
    {K : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K)) :
    1 - maxMass μ hK ≤ errorProb μ obs (fun _ => (0 : Fin 1)) decode := by
  have h := errorProb_ge_one_sub_budget_mul_maxMass μ hK obs (fun _ => (0 : Fin 1)) decode
  simpa using h

theorem observation_only_successProb_le_exp_neg_minEntropy
    {K : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K)) :
    successProb μ obs (fun _ => (0 : Fin 1)) decode ≤ Real.exp (- minEntropy μ hK) := by
  have h := successProb_le_budget_times_exp_neg_minEntropy μ hK obs (fun _ => (0 : Fin 1)) decode
  simpa using h

theorem observation_only_errorProb_ge_one_sub_exp_neg_minEntropy
    {K : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K)) :
    1 - Real.exp (- minEntropy μ hK) ≤ errorProb μ obs (fun _ => (0 : Fin 1)) decode := by
  have h := errorProb_ge_one_sub_budget_times_exp_neg_minEntropy μ hK obs (fun _ => (0 : Fin 1)) decode
  simpa using h

theorem observation_only_minEntropy_le_neg_log_one_sub_error
    {K : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    {ε : ℝ}
    (herr : errorProb μ obs (fun _ => (0 : Fin 1)) decode ≤ ε)
    (hε : ε < 1) :
    minEntropy μ hK ≤ - Real.log (1 - ε) := by
  have h := minEntropy_le_log_budget_sub_log_one_sub_error μ hK obs (fun _ => (0 : Fin 1)) decode herr hε
  simpa using h

end ObserverModel
