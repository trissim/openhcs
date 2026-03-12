import Paper1IT.FanoFinite

namespace ObserverModel

/-- Entropy-sensitive converse in the finite budgeted setting: low error under an observation/tag
budget forces the source entropy below the corresponding Fano bound. -/
theorem finiteRateDistortionConverse
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
  exact fano_arbitrary_budgeted μ obs tag decode hs hf

/-- Conditional-entropy form of the finite budgeted converse. -/
theorem finiteConditionalRateDistortionConverse
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
  exact conditionalEntropyGivenPair_le_fano_arbitrary μ obs tag decode hs hf

/-- Observation-only specialization of the finite converse. -/
theorem finiteObservationOnlyRateDistortionConverse
    {K : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    (hs : 0 < (successSet obs (fun _ => (0 : Fin 1)) decode).card)
    (hf : (successSet obs (fun _ => (0 : Fin 1)) decode).card < K) :
    sourceEntropy μ
      ≤ Real.qaryEntropy K (errorProb μ obs (fun _ => (0 : Fin 1)) decode) := by
  exact fano_arbitrary_observation_only μ obs decode hs hf

/-- Min-entropy version of the finite budgeted converse. -/
theorem finiteMinEntropyBudgetConverse
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
  exact minEntropy_le_log_budget_sub_log_one_sub_error μ hK obs tag decode herr hε

/-- Uniform-source semantic identity rate-distortion converse. -/
theorem uniformFiniteRateDistortionConverse
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
  exact fano_uniform_budgeted n obs tag decode hs hf

end ObserverModel
