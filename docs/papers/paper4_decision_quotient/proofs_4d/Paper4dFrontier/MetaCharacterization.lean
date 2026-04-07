import Paper4dFrontier.DecisionRelevantPairwiseDichotomy
import Paper4dFrontier.RealTreewidth

namespace Paper4dFrontier

open DecisionQuotient

/-- A single finite binary-pairwise utility slice. The meta-characterization layer
bundles the action type and arity so that predicates can range over a uniform
ambient object. -/
structure BinaryPairwiseSlice where
  Action : Type
  instFintypeAction : Fintype Action
  instDecidableEqAction : DecidableEq Action
  arity : ℕ
  utility : Action → (Fin arity → Fin 2) → ℤ
  pairwise : PairwiseUtility utility

attribute [instance] BinaryPairwiseSlice.instFintypeAction
attribute [instance] BinaryPairwiseSlice.instDecidableEqAction

/-- The impossibility statement is asymptotic, so the ambient object is an indexed
family of binary-pairwise slices rather than a single finite utility. -/
abbrev BinaryPairwiseLandscape : Type 1 := ℕ → BinaryPairwiseSlice

def BinaryPairwiseSlice.dimensionalUtility (U : BinaryPairwiseSlice) :
    U.Action → DimensionalStateSpace 2 U.arity → ℤ :=
  fun a s => U.utility a s.state

def CoordinateSymmetricLandscape (U : BinaryPairwiseLandscape) : Prop :=
  ∀ t : ℕ, SymmetricUtility (BinaryPairwiseSlice.dimensionalUtility (U t))

abbrev ExtractedGraphFamily (U : BinaryPairwiseLandscape) : Type :=
  ∀ t : ℕ, SimpleGraph (Fin ((U t).arity))

def UniformlyBoundedRealTreewidth {U : BinaryPairwiseLandscape}
    (G : ExtractedGraphFamily U) : Prop :=
  ∃ w : ℕ, ∀ t : ℕ, realTreewidth_le (G t) w

/-- A normalization predicate consists of a family-level predicate together with a
graph extractor defined on landscapes satisfying that predicate. -/
structure NormalizationPredicate where
  holds : BinaryPairwiseLandscape → Prop
  graph : ∀ U : BinaryPairwiseLandscape, holds U → ExtractedGraphFamily U

/-- A proposed frontier characterization by a finite list of normalizations. -/
structure TractabilityCharacterization where
  predicates : List NormalizationPredicate

def singletonCharacterization (P : NormalizationPredicate) :
    TractabilityCharacterization where
  predicates := [P]

def TractabilityCharacterization.ClaimsTractable
    (C : TractabilityCharacterization) (U : BinaryPairwiseLandscape) : Prop :=
  CoordinateSymmetricLandscape U ∨
    ∃ P ∈ C.predicates, ∃ hP : P.holds U,
      UniformlyBoundedRealTreewidth (P.graph U hP)

def TractabilityCharacterization.ClaimsHard
    (C : TractabilityCharacterization) (U : BinaryPairwiseLandscape) : Prop :=
  ¬ C.ClaimsTractable U

/-- The meta-layer needs an explicit oracle for the actual complexity behavior of
exact relevance certification on the chosen landscape class. -/
structure ExactRelevanceComplexityModel where
  Polynomial : BinaryPairwiseLandscape → Prop
  Hard : BinaryPairwiseLandscape → Prop

/-- A landscape defeats a proposed characterization when the classification it
returns disagrees with the supplied complexity model. -/
def Defeats (M : ExactRelevanceComplexityModel)
    (U : BinaryPairwiseLandscape) (C : TractabilityCharacterization) : Prop :=
  (C.ClaimsTractable U ∧ M.Hard U) ∨
    (C.ClaimsHard U ∧ M.Polynomial U)

def collapseLandscapeInfinity (M : ExactRelevanceComplexityModel) : Prop :=
  ∀ C : TractabilityCharacterization, ∃ U : BinaryPairwiseLandscape, Defeats M U C

theorem singletonClaimsTractable_iff
    (P : NormalizationPredicate) (U : BinaryPairwiseLandscape) :
    (singletonCharacterization P).ClaimsTractable U ↔
      CoordinateSymmetricLandscape U ∨
        ∃ hP : P.holds U, UniformlyBoundedRealTreewidth (P.graph U hP) := by
  constructor
  · intro h
    rcases h with hsymm | ⟨Q, hQ, hQholds, htw⟩
    · exact Or.inl hsymm
    · have hEq : Q = P := List.mem_singleton.mp hQ
      subst hEq
      exact Or.inr ⟨hQholds, htw⟩
  · intro h
    rcases h with hsymm | ⟨hP, htw⟩
    · exact Or.inl hsymm
    · exact Or.inr ⟨P, List.mem_singleton.2 rfl, hP, htw⟩

theorem singleton_oracle_characterization
    (M : ExactRelevanceComplexityModel) (P : NormalizationPredicate)
    (hOracle : ∀ U : BinaryPairwiseLandscape, P.holds U ↔ M.Polynomial U)
    (hBound : ∀ U : BinaryPairwiseLandscape, ∀ hP : P.holds U,
      UniformlyBoundedRealTreewidth (P.graph U hP))
    (hSymmPoly : ∀ U : BinaryPairwiseLandscape,
      CoordinateSymmetricLandscape U → M.Polynomial U)
    (U : BinaryPairwiseLandscape) :
    (singletonCharacterization P).ClaimsTractable U ↔ M.Polynomial U := by
  rw [singletonClaimsTractable_iff]
  constructor
  · intro h
    rcases h with hsymm | ⟨hP, _⟩
    · exact hSymmPoly U hsymm
    · exact (hOracle U).1 hP
  · intro hpoly
    let hP : P.holds U := (hOracle U).2 hpoly
    exact Or.inr ⟨hP, hBound U hP⟩

theorem not_collapseLandscapeInfinity_of_oracle_predicate
    (M : ExactRelevanceComplexityModel) (P : NormalizationPredicate)
    (hOracle : ∀ U : BinaryPairwiseLandscape, P.holds U ↔ M.Polynomial U)
    (hBound : ∀ U : BinaryPairwiseLandscape, ∀ hP : P.holds U,
      UniformlyBoundedRealTreewidth (P.graph U hP))
    (hSymmPoly : ∀ U : BinaryPairwiseLandscape,
      CoordinateSymmetricLandscape U → M.Polynomial U)
    (hDisjoint : ∀ U : BinaryPairwiseLandscape, M.Hard U → ¬ M.Polynomial U) :
    ¬ collapseLandscapeInfinity M := by
  intro hCollapse
  let C := singletonCharacterization P
  obtain ⟨U, hDefeat⟩ := hCollapse C
  have hChar : C.ClaimsTractable U ↔ M.Polynomial U := by
    simpa [C] using singleton_oracle_characterization M P hOracle hBound hSymmPoly U
  rcases hDefeat with ⟨hClaim, hHard⟩ | ⟨hClaimHard, hPoly⟩
  · exact hDisjoint U hHard (hChar.mp hClaim)
  · exact hClaimHard (hChar.mpr hPoly)

end Paper4dFrontier
