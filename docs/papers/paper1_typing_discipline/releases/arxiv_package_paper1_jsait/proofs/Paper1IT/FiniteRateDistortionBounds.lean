import Paper1IT.FiniteRateDistortionConverse

namespace ObserverModel

/-- Entropy-sensitive finite converse with the success-probability term absorbed into the budget
whenever the observation/tag budget has at least one codeword. -/
theorem finiteRateDistortionBound
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K)
    (hOT : 1 ≤ O * T) :
    sourceEntropy μ
      ≤ Real.log (O * T)
        + Real.binEntropy (errorProb μ obs tag decode)
        + errorProb μ obs tag decode * Real.log (K - 1) := by
  have hbase := finiteRateDistortionConverse μ obs tag decode hs hf
  have hsucc_nonneg := successProb_nonneg μ obs tag decode
  have hsucc_le_one := successProb_le_one μ obs tag decode
  have hlog_nonneg : 0 ≤ Real.log (O * T) := by
    exact Real.log_nonneg (show (1 : ℝ) ≤ O * T by exact_mod_cast hOT)
  have hmul : successProb μ obs tag decode * Real.log (O * T) ≤ Real.log (O * T) := by
    have := mul_le_mul_of_nonneg_right hsucc_le_one hlog_nonneg
    simpa [one_mul] using this
  linarith

/-- Rearranged budget lower bound in logarithmic form. -/
theorem logBudgetLowerBoundFromError
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K)
    (hOT : 1 ≤ O * T) :
    sourceEntropy μ
      - Real.binEntropy (errorProb μ obs tag decode)
      - errorProb μ obs tag decode * Real.log (K - 1)
      ≤ Real.log (O * T) := by
  have h := finiteRateDistortionBound μ obs tag decode hs hf hOT
  linarith

/-- Observation-only specialization of the finite entropy-sensitive converse. -/
theorem finiteObservationOnlyBound
    {K : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    (hs : 0 < (successSet obs (fun _ => (0 : Fin 1)) decode).card)
    (hf : (successSet obs (fun _ => (0 : Fin 1)) decode).card < K) :
    sourceEntropy μ
      ≤ Real.binEntropy (errorProb μ obs (fun _ => (0 : Fin 1)) decode)
        + errorProb μ obs (fun _ => (0 : Fin 1)) decode * Real.log (K - 1) := by
  exact fano_arbitrary_conditional_observation_only μ obs decode hs hf

end ObserverModel
