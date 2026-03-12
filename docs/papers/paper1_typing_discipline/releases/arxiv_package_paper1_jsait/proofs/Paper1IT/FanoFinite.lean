import Paper1IT.ProbabilisticFinite
import Paper1IT.ObserverTagModel

namespace ObserverModel

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

end ObserverModel
