import Paper1IT.FiniteRateDistortionBounds
import Paper1IT.PMFEntropy

namespace ObserverModel

/-- Joint observation/tag alphabet size in the finite deterministic setting. -/
def jointBudgetAlphabet (O T : Nat) : Nat := O * T

/-- Log-budget of the joint observation/tag alphabet. -/
noncomputable def jointBudgetRate (O T : Nat) : ℝ := Real.log ((jointBudgetAlphabet O T : Nat) : ℝ)

theorem jointBudgetRate_eq_log_prod (O T : Nat) :
    jointBudgetRate O T = Real.log ((O * T : Nat) : ℝ) := rfl

/-- Finite budgeted rate-distortion converse packaged in Paper 1 language. -/
theorem finiteRateDistortionConverse_logBudget
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K) :
    sourceEntropy μ
      ≤ Real.binEntropy (errorProb μ obs tag decode)
        + successProb μ obs tag decode * jointBudgetRate O T
        + errorProb μ obs tag decode * Real.log (K - 1) := by
  unfold jointBudgetRate jointBudgetAlphabet
  simpa [Nat.cast_mul] using finiteRateDistortionConverse μ obs tag decode hs hf

/-- Coarser budget-only converse with the success term absorbed into the budget. -/
theorem finiteRateDistortionConverse_absorbed
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K)
    (hOT : 1 ≤ jointBudgetAlphabet O T) :
    sourceEntropy μ
      ≤ jointBudgetRate O T
        + Real.binEntropy (errorProb μ obs tag decode)
        + errorProb μ obs tag decode * Real.log (K - 1) := by
  unfold jointBudgetRate jointBudgetAlphabet
  simpa [Nat.cast_mul] using
    finiteRateDistortionBound μ obs tag decode hs hf hOT

/-- Rearranged logarithmic lower bound on the joint budget. -/
theorem finiteJointBudgetLowerBoundFromError
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hs : 0 < (successSet obs tag decode).card)
    (hf : (successSet obs tag decode).card < K)
    (hOT : 1 ≤ jointBudgetAlphabet O T) :
    sourceEntropy μ
      - Real.binEntropy (errorProb μ obs tag decode)
      - errorProb μ obs tag decode * Real.log (K - 1)
      ≤ jointBudgetRate O T := by
  unfold jointBudgetRate jointBudgetAlphabet
  simpa [Nat.cast_mul] using
    logBudgetLowerBoundFromError μ obs tag decode hs hf hOT

/-- Observation-only entropy-sensitive converse. -/
theorem observationOnlyRateDistortionConverse
    {K : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    (hs : 0 < (successSet obs (fun _ => (0 : Fin 1)) decode).card)
    (hf : (successSet obs (fun _ => (0 : Fin 1)) decode).card < K) :
    sourceEntropy μ
      ≤ Real.binEntropy (errorProb μ obs (fun _ => (0 : Fin 1)) decode)
        + errorProb μ obs (fun _ => (0 : Fin 1)) decode * Real.log (K - 1) := by
  exact finiteObservationOnlyBound μ obs decode hs hf

/-- Min-entropy observation-only lower bound against the error floor. -/
theorem observationOnlyMinEntropyBound
    {K : Nat}
    (μ : FiniteSource K)
    (hK : 0 < K)
    (obs : Fin K → Fin 1)
    (decode : Fin 1 → Fin 1 → Option (Fin K))
    {ε : ℝ}
    (herr : errorProb μ obs (fun _ => (0 : Fin 1)) decode ≤ ε)
    (hε : ε < 1) :
    minEntropy μ hK ≤ - Real.log (1 - ε) := by
  exact observation_only_minEntropy_le_neg_log_one_sub_error μ hK obs decode herr hε

end ObserverModel
