import Paper1IT.ProbabilisticFinite

namespace ObserverModel

/-- Bridge theorem: source entropy agrees with PMF entropy after coercion to a `PMF`. -/
theorem pmfEntropy_source_eq_sourceEntropy
    {K : Nat} (μ : FiniteSource K) :
    pmfEntropy (finiteSourcePMF μ) = sourceEntropy μ := by
  exact pmfEntropy_finiteSourcePMF_eq_sourceEntropy μ

/-- Bridge theorem for the joint observation/tag source. -/
theorem pmfEntropy_pair_eq_observationTagEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    pmfEntropy (inducedPairPMFFin μ obs tag) = observationTagEntropy μ obs tag := by
  exact pmfEntropy_inducedPairPMFFin_eq_observationTagEntropy μ obs tag

/-- Bridge theorem for the decoded output distribution. -/
theorem pmfEntropy_decodedOutput_eq_decodedOutputEntropy
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K)) :
    pmfEntropy (inducedDecodedOutputPMFFin μ obs tag decode) = decodedOutputEntropy μ obs tag decode := by
  exact pmfEntropy_inducedDecodedOutputPMFFin_eq_decodedOutputEntropy μ obs tag decode

/-- Conditional entropy can be read as the entropy gap between the source and the joint
observation/tag PMF. -/
theorem conditionalEntropyGivenPair_eq_source_minus_pmfEntropy_pair
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    conditionalEntropyGivenPair μ obs tag = sourceEntropy μ - pmfEntropy (inducedPairPMFFin μ obs tag) := by
  rw [pmfEntropy_inducedPairPMFFin_eq_observationTagEntropy]
  have h := sourceEntropy_eq_observationTagEntropy_add_conditionalEntropyGivenPair μ obs tag
  linarith

/-- Deterministic mutual information coincides with the PMF entropy of the induced pair. -/
theorem mutualInfoDeterministic_eq_pmfEntropy_pair
    {K O T : Nat}
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    mutualInfoDeterministic μ obs tag = pmfEntropy (inducedPairPMFFin μ obs tag) := by
  rw [mutualInfoDeterministic_eq_observationTagEntropy, pmfEntropy_inducedPairPMFFin_eq_observationTagEntropy]

/-- Zero KL gap to the uniform pair distribution forces the deterministic mutual information to equal
the full joint log-budget. -/
theorem mutualInfoDeterministic_eq_log_budget_of_uniform_pair
    {K O T : Nat} [Nonempty (Fin (O * T))]
    (μ : FiniteSource K)
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hkl : InformationTheory.klDiv (inducedPairPMFFin μ obs tag).toMeasure
        (PMF.uniformOfFintype (Fin (O * T))).toMeasure = 0) :
    mutualInfoDeterministic μ obs tag = Real.log ((O * T : Nat) : ℝ) := by
  exact mutualInfoDeterministic_eq_log_budget_of_klDiv_zero_uniform μ obs tag hkl

end ObserverModel
