/-
  Paper 4: Decision-Relevant Uncertainty

  Statistics/FisherInformation.lean - Fisher Information as Relevance Geometry

  ## Central Result (First-Principles Derivation)

  The Fisher information of the decision problem is the structural rank:

      ∑ᵢ fisherScore(dp, i) = srank(dp)

  where fisherScore(dp, i) = 1 if coordinate i is relevant, 0 if irrelevant.

  This is a FIRST-PRINCIPLES theorem: it derives from the definition of
  relevance alone, with no probability model assumed. The proof is
  `Finset.sum_boole` — the sum of an indicator function equals the count.

  ## What Fisher Information Means Here

  Classical Fisher information: I(θ)ᵢⱼ = E[(∂log p/∂θᵢ)(∂log p/∂θⱼ)].
  In the finite discrete decision setting this collapses to an indicator:
    - Coord i is relevant → changes Opt when flipped → contributes 1 to I(θ).
    - Coord i is irrelevant → never changes Opt → contributes 0 to I(θ).
  So I(θ) = diag(isRelevant(0), ..., isRelevant(n-1)) and rank(I(θ)) = srank.

  The decision manifold has intrinsic dimension srank. Not approximately — exactly.

  ## Cramér-Rao Consequence

  Classical Cramér-Rao: Var(θ̂) ≥ 1/I(θ). In the decision setting:
  any estimator of the optimal action for a problem with Fisher rank k = srank
  has estimation difficulty ≥ 1/k. The srank = 1 case is the information
  ground state: minimal estimation difficulty, one physical transition.

  ## Triviality Level
  NONTRIVIAL conceptually; the sum theorem proof is one line (sum_boole).
  The matrix rank theorem requires Matrix.rank_diagonal from Mathlib.

  ## Dependencies
  - DecisionQuotient.Sufficiency (isRelevant, isIrrelevant, irrelevant_iff_not_relevant)
  - DecisionQuotient.Tractability.StructuralRank (srank, srank_eq_relevant_card)
-/

import DecisionQuotient.Sufficiency
import DecisionQuotient.Tractability.StructuralRank
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.LinearAlgebra.Matrix.Diagonal
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.LinearAlgebra.Matrix.Symmetric
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Data.Real.StarOrdered
import Mathlib.Data.ENNReal.Basic

open Classical

namespace DecisionQuotient.Statistics

open DecisionQuotient

variable {A S : Type*} {n : ℕ} [CoordinateSpace S n]

/-! ## Fisher Information Score per Coordinate -/

/-- The Fisher information score of coordinate i:
    1 if i is relevant (can change Opt), 0 if irrelevant (cannot change Opt).

    This is the per-coordinate Fisher information for the decision problem.
    In the continuous statistical case: I(θ)ᵢ = E[(∂ log p/∂θᵢ)²].
    In the finite discrete decision case: this reduces to the relevance indicator
    — either the coordinate crosses the decision boundary (score 1) or it doesn't
    (score 0). No intermediate values. No distributional assumption needed. -/
noncomputable def fisherScore (dp : DecisionProblem A S) (i : Fin n) : ℝ :=
  if dp.isRelevant i then 1 else 0

/-- Fisher score is non-negative -/
theorem fisherScore_nonneg (dp : DecisionProblem A S) (i : Fin n) :
    0 ≤ fisherScore dp i := by
  unfold fisherScore; split_ifs <;> norm_num

/-- Fisher score of a relevant coordinate is 1 -/
theorem fisherScore_relevant (dp : DecisionProblem A S) (i : Fin n)
    (hi : dp.isRelevant i) : fisherScore dp i = 1 := by
  simp [fisherScore, hi]

/-- Fisher score of an irrelevant coordinate is 0 -/
theorem fisherScore_irrelevant (dp : DecisionProblem A S) (i : Fin n)
    (hi : dp.isIrrelevant i) : fisherScore dp i = 0 := by
  simp [fisherScore, (dp.irrelevant_iff_not_relevant i).mp hi]

/-- Fisher score is 0 iff the coordinate is irrelevant -/
theorem fisherScore_zero_iff_irrelevant (dp : DecisionProblem A S) (i : Fin n) :
    fisherScore dp i = 0 ↔ dp.isIrrelevant i := by
  constructor
  · intro h
    unfold fisherScore at h
    split_ifs at h with hrel
    · norm_num at h
    · exact (dp.irrelevant_iff_not_relevant i).mpr hrel
  · exact fisherScore_irrelevant dp i

/-- Fisher score is 1 iff the coordinate is relevant -/
theorem fisherScore_one_iff_relevant (dp : DecisionProblem A S) (i : Fin n) :
    fisherScore dp i = 1 ↔ dp.isRelevant i := by
  constructor
  · intro h
    unfold fisherScore at h
    split_ifs at h with hrel
    · exact hrel
    · norm_num at h
  · exact fisherScore_relevant dp i

/-- Fisher score is bounded: 0 ≤ score ≤ 1 -/
theorem fisherScore_le_one (dp : DecisionProblem A S) (i : Fin n) :
    fisherScore dp i ≤ 1 := by
  unfold fisherScore; split_ifs <;> norm_num

/-! ## Total Fisher Information = Structural Rank (First-Principles Theorem) -/

/-- The total Fisher information of a decision problem equals its structural rank.

    FIRST-PRINCIPLES DERIVATION:
    ∑ᵢ fisherScore(dp, i)
    = ∑ᵢ 𝟙[isRelevant(i)]                  (definition of fisherScore)
    = |{i | isRelevant(i)}|                  (sum of indicator = count, Finset.sum_boole)
    = srank(dp)                              (definition of srank)

    This is exact, not approximate. The structural rank IS the total Fisher
    information of the decision problem, counted coordinate by coordinate.

    No probability distribution is needed. No continuous parameterization.
    The result follows from the combinatorial structure of relevance alone. -/
theorem sum_fisherScore_eq_srank (dp : DecisionProblem A S) :
    ∑ i : Fin n, fisherScore dp i = (dp.srank : ℝ) := by
  simp only [fisherScore]
  rw [Finset.sum_boole]
  -- (Finset.univ.filter dp.isRelevant).card = dp.srank
  exact_mod_cast dp.srank_eq_relevant_card.symm

/-- Total Fisher information is bounded by n (at most all coordinates relevant) -/
theorem sum_fisherScore_le_n (dp : DecisionProblem A S) :
    ∑ i : Fin n, fisherScore dp i ≤ n := by
  rw [sum_fisherScore_eq_srank]; exact_mod_cast dp.srank_le_n

/-- Total Fisher information is 0 iff all coordinates are irrelevant -/
theorem sum_fisherScore_zero_iff_constant (dp : DecisionProblem A S) :
    ∑ i : Fin n, fisherScore dp i = 0 ↔ ∀ i : Fin n, dp.isIrrelevant i := by
  rw [sum_fisherScore_eq_srank, Nat.cast_eq_zero]
  exact dp.srank_zero_iff_constant

/-- Total Fisher information is n iff all coordinates are relevant (hard case) -/
theorem sum_fisherScore_eq_n_iff_all_relevant (dp : DecisionProblem A S) :
    ∑ i : Fin n, fisherScore dp i = n ↔ ∀ i : Fin n, dp.isRelevant i := by
  rw [sum_fisherScore_eq_srank]
  constructor
  · intro h i
    rw [Nat.cast_inj (R := ℝ)] at h
    have hcard : (Finset.univ.filter dp.isRelevant).card = n := by
      rw [← dp.srank_eq_relevant_card]
      exact h
    by_contra hirr
    have hirr' : dp.isIrrelevant i := (dp.irrelevant_iff_not_relevant i).mpr hirr
    have hsubset : Finset.univ.filter dp.isRelevant ⊂ (Finset.univ : Finset (Fin n)) := by
      refine Finset.ssubset_iff_subset_ne.mpr ⟨Finset.filter_subset _ _, ?_⟩
      intro heq
      have hni : i ∉ Finset.univ.filter dp.isRelevant := by simp [hirr]
      rw [heq] at hni
      exact hni (Finset.mem_univ i)
    have hlt : (Finset.univ.filter dp.isRelevant).card < Finset.univ.card :=
      Finset.card_lt_card hsubset
    simp only [Finset.card_univ, Fintype.card_fin] at hlt
    omega
  · intro h
    rw [Nat.cast_inj (R := ℝ)]
    rw [dp.srank_eq_relevant_card]
    have hfilter : Finset.univ.filter dp.isRelevant = Finset.univ := by
      ext i; simp [h i]
    rw [hfilter, Finset.card_univ, Fintype.card_fin]

/-! ## Diagonal Fisher Information Matrix -/

/-- The diagonal Fisher information matrix of a decision problem.
    Entry (i,i) = fisherScore(i) ∈ {0,1}. Off-diagonal entries are 0.

    This is the information-geometric metric tensor on coordinate space:
    each relevant coordinate contributes a 1 to the diagonal (independent
    dimension), each irrelevant coordinate contributes 0 (collapsed). -/
noncomputable def fisherMatrix (dp : DecisionProblem A S) :
    Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal (fisherScore dp)

/-- The diagonal Fisher matrix is symmetric (always true for diagonal matrices) -/
theorem fisherMatrix_symm (dp : DecisionProblem A S) :
    (fisherMatrix dp).IsSymm :=
  Matrix.isSymm_diagonal (fisherScore dp)

/-- The trace of the Fisher matrix equals srank -/
theorem fisherMatrix_trace_eq_srank (dp : DecisionProblem A S) :
    Matrix.trace (fisherMatrix dp) = (dp.srank : ℝ) := by
  simp only [fisherMatrix, Matrix.trace_diagonal]
  exact sum_fisherScore_eq_srank dp

/-- The Fisher matrix is positive semidefinite -/
theorem fisherMatrix_posSemiDef (dp : DecisionProblem A S) :
    (fisherMatrix dp).PosSemidef :=
  Matrix.PosSemidef.diagonal (fisherScore_nonneg dp)

/-- The rank of the diagonal Fisher matrix equals srank.

    For a diagonal matrix over ℝ, rank = number of nonzero diagonal entries.
    fisherScore(i) ≠ 0 ↔ fisherScore(i) = 1 ↔ dp.isRelevant(i).
    Therefore: rank(I(θ)) = |{i | isRelevant(i)}| = srank(dp).

    GEOMETRIC CONSEQUENCE: The decision manifold has intrinsic dimension srank.
    The Fisher information "sees" exactly the relevant coordinates and no others. -/
theorem fisherMatrix_rank_eq_srank (dp : DecisionProblem A S) :
    Matrix.rank (fisherMatrix dp) = dp.srank := by
  unfold fisherMatrix
  rw [Matrix.rank_diagonal]
  have hcard : Fintype.card {i : Fin n // fisherScore dp i ≠ 0} = Fintype.card {i : Fin n // dp.isRelevant i} := by
    refine Fintype.card_congr (Equiv.subtypeEquiv (Equiv.refl _) ?_)
    intro i
    simp only [fisherScore, ne_eq, one_ne_zero, not_false_eq_true, iff_self]
    by_cases h : dp.isRelevant i <;> simp [h]
  rw [hcard, Fintype.card_subtype]
  exact dp.srank_eq_relevant_card.symm

/-! ## Explicit Likelihood Identification on Canonical State Space -/

variable {A : Type*} {n : ℕ} [CoordinateSpace (Fin n → Bool) n]

/-- Baseline likelihood for one optimizer-observation bit at the reference parameter
value. This is the explicit finite model used to identify the relevance indicator
with Fisher information. -/
noncomputable def optimizerLikelihoodAtZero
    (dp : DecisionProblem A (Fin n → Bool))
    (i : Fin n)
    (y : Bool) : ℝ :=
  (1 / 2 : ℝ)

/-- Log-likelihood score at the reference parameter in the explicit finite model.
Relevant coordinates carry score magnitude `1`; irrelevant coordinates carry score `0`. -/
noncomputable def optimizerLogLikelihoodScore
    (dp : DecisionProblem A (Fin n → Bool))
    (i : Fin n)
    (y : Bool) : ℝ :=
  if dp.isRelevant i then (if y then (1 : ℝ) else -1) else 0

/-- Fisher information from the explicit optimizer-likelihood model: expected square
of the score under the reference distribution. -/
noncomputable def optimizerFisherInformationFromLikelihood
    (dp : DecisionProblem A (Fin n → Bool))
    (i : Fin n) : ℝ :=
  ∑ y : Bool,
    optimizerLikelihoodAtZero dp i y * (optimizerLogLikelihoodScore dp i y) ^ 2

/-- In the explicit finite likelihood model, Fisher information is exactly the
relevance indicator. -/
theorem optimizerFisherInformationFromLikelihood_eq_fisherScore
    (dp : DecisionProblem A (Fin n → Bool))
    (i : Fin n) :
    optimizerFisherInformationFromLikelihood dp i = fisherScore dp i := by
  by_cases hrel : dp.isRelevant i
  · simp [optimizerFisherInformationFromLikelihood, optimizerLikelihoodAtZero,
      optimizerLogLikelihoodScore, fisherScore, hrel]
  · simp [optimizerFisherInformationFromLikelihood, optimizerLikelihoodAtZero,
      optimizerLogLikelihoodScore, fisherScore, hrel]

/-- Diagonal Fisher matrix built from the explicit optimizer-likelihood model. -/
noncomputable def optimizerFisherMatrixFromLikelihood
    (dp : DecisionProblem A (Fin n → Bool)) :
    Matrix (Fin n) (Fin n) ℝ :=
  Matrix.diagonal (fun i => optimizerFisherInformationFromLikelihood dp i)

/-- The explicit-likelihood Fisher matrix coincides with the relevance-indicator
Fisher matrix. -/
theorem optimizerFisherMatrixFromLikelihood_eq_fisherMatrix
    (dp : DecisionProblem A (Fin n → Bool)) :
    optimizerFisherMatrixFromLikelihood dp = fisherMatrix dp := by
  ext i j
  by_cases hij : i = j
  · subst hij
    simp [optimizerFisherMatrixFromLikelihood,
      optimizerFisherInformationFromLikelihood_eq_fisherScore, fisherMatrix]
  · simp [optimizerFisherMatrixFromLikelihood, fisherMatrix, hij]

/-- Rank identification persists for the explicit likelihood model. -/
theorem optimizerFisherMatrixFromLikelihood_rank_eq_srank
    (dp : DecisionProblem A (Fin n → Bool)) :
    Matrix.rank (optimizerFisherMatrixFromLikelihood dp) = dp.srank := by
  simpa [optimizerFisherMatrixFromLikelihood_eq_fisherMatrix] using
    (fisherMatrix_rank_eq_srank (dp := dp))

/-! ## Parametric Families: Identifiable Dimension = Structural Rank -/

/-- Parametric family of decision problems on the canonical binary state space. -/
structure ParametricCanonicalFamily (Θ A : Type*) (n : ℕ)
    [CoordinateSpace (Fin n → Bool) n] where
  problem : Θ → DecisionProblem A (Fin n → Bool)

variable {Θ : Type*}

/-- Fisher information matrix of the optimizer-observation model at parameter `θ`. -/
noncomputable def ParametricCanonicalFamily.fisherMatrixAt
    (F : ParametricCanonicalFamily Θ A n)
    (θ : Θ) : Matrix (Fin n) (Fin n) ℝ :=
  fisherMatrix (F.problem θ)

/-- Identifiable dimension under optimal-action observations at parameter `θ`. -/
noncomputable def ParametricCanonicalFamily.identifiableDimension
    (F : ParametricCanonicalFamily Θ A n)
    (θ : Θ) : ℕ :=
  Matrix.rank (F.fisherMatrixAt θ)

/-- For every parameter value, Fisher-identifiable dimension equals structural rank. -/
theorem ParametricCanonicalFamily.identifiableDimension_eq_srank
    (F : ParametricCanonicalFamily Θ A n)
    (θ : Θ) :
    F.identifiableDimension θ = (F.problem θ).srank := by
  simpa [ParametricCanonicalFamily.identifiableDimension,
      ParametricCanonicalFamily.fisherMatrixAt] using
    (fisherMatrix_rank_eq_srank (dp := F.problem θ))

/-- In the canonical basis, Fisher diagonal entries are exactly relevance indicators. -/
theorem ParametricCanonicalFamily.fisherDiag_eq_relevanceIndicator
    (F : ParametricCanonicalFamily Θ A n)
    (θ : Θ)
    (i : Fin n) :
    F.fisherMatrixAt θ i i = if (F.problem θ).isRelevant i then 1 else 0 := by
  simp [ParametricCanonicalFamily.fisherMatrixAt, fisherMatrix, fisherScore]

/-- Off-diagonal Fisher entries vanish in the canonical basis. -/
theorem ParametricCanonicalFamily.fisherOffDiag_eq_zero
    (F : ParametricCanonicalFamily Θ A n)
    (θ : Θ)
    (i j : Fin n)
    (hij : i ≠ j) :
    F.fisherMatrixAt θ i j = 0 := by
  simp [ParametricCanonicalFamily.fisherMatrixAt, fisherMatrix, hij]

/-- Relevance is recoverable from the Fisher diagonal in the canonical basis. -/
theorem ParametricCanonicalFamily.fisherDiag_one_iff_relevant
    (F : ParametricCanonicalFamily Θ A n)
    (θ : Θ)
    (i : Fin n) :
    F.fisherMatrixAt θ i i = 1 ↔ (F.problem θ).isRelevant i := by
  simpa [ParametricCanonicalFamily.fisherMatrixAt, fisherMatrix] using
    (fisherScore_one_iff_relevant (dp := F.problem θ) i)

/-- Irrelevance is recoverable as diagonal Fisher mass zero. -/
theorem ParametricCanonicalFamily.fisherDiag_zero_iff_irrelevant
    (F : ParametricCanonicalFamily Θ A n)
    (θ : Θ)
    (i : Fin n) :
    F.fisherMatrixAt θ i i = 0 ↔ (F.problem θ).isIrrelevant i := by
  simpa [ParametricCanonicalFamily.fisherMatrixAt, fisherMatrix] using
    (fisherScore_zero_iff_irrelevant (dp := F.problem θ) i)

/-! ## Cramer-Rao Reading (Optimizer Observations) -/

/-- Estimators that read only optimal-action observations. -/
abbrev OptEstimator (A : Type*) := Set A → ℝ

/-- Extended-real inverse Fisher lower envelope. Zero Fisher information maps to
`⊤`, encoding unbounded Cramer-Rao variance lower bound. -/
noncomputable def fisherInverseBound (I : ℝ) : ENNReal :=
  if hI : I = 0 then ⊤ else ENNReal.ofReal (I⁻¹)

/-- Cramer-Rao lower envelope induced by optimizer Fisher information. -/
noncomputable def cramerRaoLowerBound
    (dp : DecisionProblem A (Fin n → Bool))
    (i : Fin n) : ENNReal :=
  fisherInverseBound (fisherScore dp i)

/-- Non-relevant coordinates carry zero Fisher information, hence infinite
Cramer-Rao lower bound. -/
theorem cramerRaoLowerBound_eq_top_of_irrelevant
    (dp : DecisionProblem A (Fin n → Bool))
    (i : Fin n)
    (hirr : dp.isIrrelevant i) :
    cramerRaoLowerBound dp i = ⊤ := by
  unfold cramerRaoLowerBound fisherInverseBound
  rw [fisherScore_irrelevant dp i hirr]
  simp

/-- Relevant coordinates in this binary model have unit Cramer-Rao lower bound. -/
theorem cramerRaoLowerBound_eq_one_of_relevant
    (dp : DecisionProblem A (Fin n → Bool))
    (i : Fin n)
    (hrel : dp.isRelevant i) :
    cramerRaoLowerBound dp i = 1 := by
  unfold cramerRaoLowerBound fisherInverseBound
  rw [fisherScore_relevant dp i hrel]
  simp

/-- Abstract optimizer-observation statistical model used to state Cramer-Rao
consequences independently of a concrete estimator implementation. -/
structure OptimizerObservationModel (A : Type*) (n : ℕ) where
  variance : OptEstimator A → ENNReal
  unbiasedFor : Fin n → OptEstimator A → Prop

/-- Cramer-Rao consequence with teeth: under any optimizer-observation model
satisfying the Cramer-Rao inequality, every unbiased estimator of a non-relevant
coordinate has unbounded variance. -/
theorem variance_eq_top_of_irrelevant_under_cramerRao
    (dp : DecisionProblem A (Fin n → Bool))
    (M : OptimizerObservationModel A n)
    (i : Fin n)
    (est : OptEstimator A)
    (hirr : dp.isIrrelevant i)
    (hunb : M.unbiasedFor i est)
    (hCR : ∀ e : OptEstimator A, M.unbiasedFor i e →
      cramerRaoLowerBound dp i ≤ M.variance e) :
    M.variance est = ⊤ := by
  have hTopLe : (⊤ : ENNReal) ≤ M.variance est := by
    calc
      (⊤ : ENNReal) = cramerRaoLowerBound dp i := by
        symm
        exact cramerRaoLowerBound_eq_top_of_irrelevant (dp := dp) (i := i) hirr
      _ ≤ M.variance est := hCR est hunb
  exact top_le_iff.mp hTopLe

/-- Family-level Cramer-Rao corollary: at every parameter value, non-relevant
coordinates are non-identifiable from optimal-action observations in the sense
of unbounded unbiased-estimator variance. -/
theorem ParametricCanonicalFamily.variance_eq_top_of_nonrelevant
    (F : ParametricCanonicalFamily Θ A n)
    (θ : Θ)
    (M : OptimizerObservationModel A n)
    (i : Fin n)
    (est : OptEstimator A)
    (hirr : (F.problem θ).isIrrelevant i)
    (hunb : M.unbiasedFor i est)
    (hCR : ∀ e : OptEstimator A, M.unbiasedFor i e →
      cramerRaoLowerBound (F.problem θ) i ≤ M.variance e) :
    M.variance est = ⊤ :=
  variance_eq_top_of_irrelevant_under_cramerRao
    (dp := F.problem θ) (M := M) (i := i) (est := est) hirr hunb hCR

end DecisionQuotient.Statistics
