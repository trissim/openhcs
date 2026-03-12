import Paper1IT.EntropyGeneral
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

end ObserverModel
