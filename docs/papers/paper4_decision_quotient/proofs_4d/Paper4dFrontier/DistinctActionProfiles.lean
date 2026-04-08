import Paper4dFrontier.AdmissibleCharacterization

namespace Paper4dFrontier

open Classical
open DecisionQuotient

abbrev SliceState (U : BinaryPairwiseSlice) : Type :=
  Fin U.arity → Fin 2

abbrev ActionProfile (U : BinaryPairwiseSlice) : Type :=
  SliceState U → ℤ

def actionProfile (U : BinaryPairwiseSlice) (a : U.Action) : ActionProfile U :=
  U.utility a

noncomputable def actionProfileFinset (U : BinaryPairwiseSlice) : Finset (ActionProfile U) :=
  Finset.univ.image (actionProfile U)

def ProfileAction (U : BinaryPairwiseSlice) : Type :=
  { p : ActionProfile U // p ∈ actionProfileFinset U }

noncomputable instance profileActionFintype (U : BinaryPairwiseSlice) : Fintype (ProfileAction U) := by
  classical
  exact (Finset.finite_toSet (actionProfileFinset U)).fintype

noncomputable def distinctActionCount (U : BinaryPairwiseSlice) : Nat :=
  Fintype.card (ProfileAction U)

noncomputable def profileActionOf (U : BinaryPairwiseSlice) (a : U.Action) : ProfileAction U :=
  ⟨actionProfile U a, Finset.mem_image.mpr ⟨a, Finset.mem_univ _, rfl⟩⟩

noncomputable def profileRepresentative {U : BinaryPairwiseSlice} (p : ProfileAction U) : U.Action :=
  Classical.choose <| by
    rcases Finset.mem_image.mp p.property with ⟨a, _ha, hEq⟩
    exact ⟨a, hEq⟩

theorem profileRepresentative_spec {U : BinaryPairwiseSlice} (p : ProfileAction U) :
    actionProfile U (profileRepresentative p) = p.1 :=
  Classical.choose_spec <| by
    rcases Finset.mem_image.mp p.property with ⟨a, _ha, hEq⟩
    exact ⟨a, hEq⟩

def BinaryPairwiseSlice.toDecisionProblem (U : BinaryPairwiseSlice) :
    DecisionProblem U.Action (SliceState U) where
  utility a s := (U.utility a s : ℝ)

noncomputable def profileCompressedSlice (U : BinaryPairwiseSlice) : BinaryPairwiseSlice where
  Action := ProfileAction U
  instFintypeAction := inferInstance
  instDecidableEqAction := inferInstance
  arity := U.arity
  utility p s := p.1 s
  pairwise := by
    refine
      { unary := fun i p x => U.pairwise.unary i (profileRepresentative p) x
        binary := fun i j p x y => U.pairwise.binary i j (profileRepresentative p) x y
        interacts := U.pairwise.interacts
        interacts_symm := U.pairwise.interacts_symm
        decomp := ?_ }
    intro p s
    have hEq := congrArg (fun f => f s) (profileRepresentative_spec p)
    calc
      p.1 s = U.utility (profileRepresentative p) s := by simpa using hEq.symm
      _ = (∑ i : Fin U.arity, U.pairwise.unary i (profileRepresentative p) (s i)) +
            (∑ i : Fin U.arity, ∑ j : Fin U.arity,
              if U.pairwise.interacts i j ∧ i < j
              then U.pairwise.binary i j (profileRepresentative p) (s i) (s j)
              else 0) := by
            simpa using U.pairwise.decomp (profileRepresentative p) s

theorem actionCount_profileCompressedSlice (U : BinaryPairwiseSlice) :
    (profileCompressedSlice U).actionCount = distinctActionCount U := by
  simp [BinaryPairwiseSlice.actionCount, distinctActionCount, profileCompressedSlice]

theorem profileActionOf_utility_eq {U : BinaryPairwiseSlice}
    (a : U.Action) (s : SliceState U) :
    (profileCompressedSlice U).utility (profileActionOf U a) s = U.utility a s := by
  rfl

theorem profileCompressedSlice_isOptimal_iff {U : BinaryPairwiseSlice}
    (p : ProfileAction U) (s : SliceState U) :
    (profileCompressedSlice U).toDecisionProblem.isOptimal p s ↔
      U.toDecisionProblem.isOptimal (profileRepresentative p) s := by
  constructor
  · intro h a
    have hOpt := h (profileActionOf U a)
    have hp : (profileCompressedSlice U).utility p s = U.utility (profileRepresentative p) s := by
      simpa using (congrArg (fun f => f s) (profileRepresentative_spec p)).symm
    change ((U.utility a s : ℝ) ≤ ((profileCompressedSlice U).utility p s : ℝ)) at hOpt
    simpa [BinaryPairwiseSlice.toDecisionProblem, hp] using hOpt
  · intro h q
    have hRep := h (profileRepresentative q)
    have hp : (profileCompressedSlice U).utility p s = U.utility (profileRepresentative p) s := by
      simpa using (congrArg (fun f => f s) (profileRepresentative_spec p)).symm
    have hq : (profileCompressedSlice U).utility q s = U.utility (profileRepresentative q) s := by
      simpa using (congrArg (fun f => f s) (profileRepresentative_spec q)).symm
    change (((profileCompressedSlice U).utility q s : ℝ) ≤ ((profileCompressedSlice U).utility p s : ℝ))
    simpa [BinaryPairwiseSlice.toDecisionProblem, hp, hq] using hRep

theorem toDecisionProblem_isOptimal_iff_of_same_profile {U : BinaryPairwiseSlice}
    {a b : U.Action} (hprof : actionProfile U a = actionProfile U b) (s : SliceState U) :
    U.toDecisionProblem.isOptimal a s ↔ U.toDecisionProblem.isOptimal b s := by
  constructor
  · intro h a'
    have hb : U.utility b s = U.utility a s := by
      simpa [actionProfile] using congrArg (fun f => f s) hprof.symm
    simpa [BinaryPairwiseSlice.toDecisionProblem, hb] using h a'
  · intro h a'
    have hb : U.utility a s = U.utility b s := by
      simpa [actionProfile] using congrArg (fun f => f s) hprof
    simpa [BinaryPairwiseSlice.toDecisionProblem, hb] using h a'

theorem profileActionOf_mem_opt_iff {U : BinaryPairwiseSlice}
    (a : U.Action) (s : SliceState U) :
    profileActionOf U a ∈ (profileCompressedSlice U).toDecisionProblem.Opt s ↔
      a ∈ U.toDecisionProblem.Opt s := by
  have hprof : actionProfile U (profileRepresentative (profileActionOf U a)) = actionProfile U a := by
    exact (profileRepresentative_spec (profileActionOf U a)).trans rfl
  constructor
  · intro hOpt
    exact (toDecisionProblem_isOptimal_iff_of_same_profile hprof s).1
      ((profileCompressedSlice_isOptimal_iff (profileActionOf U a) s).1 hOpt)
  · intro hOpt
    exact (profileCompressedSlice_isOptimal_iff (profileActionOf U a) s).2
      ((toDecisionProblem_isOptimal_iff_of_same_profile hprof s).2 hOpt)

theorem decisionEquiv_profileCompressedSlice_iff {U : BinaryPairwiseSlice}
    (s s' : SliceState U) :
    (profileCompressedSlice U).toDecisionProblem.DecisionEquiv s s' ↔
      U.toDecisionProblem.DecisionEquiv s s' := by
  unfold DecisionProblem.DecisionEquiv
  constructor
  · intro h
    ext a
    constructor
    · intro ha
      have hcomp : profileActionOf U a ∈ (profileCompressedSlice U).toDecisionProblem.Opt s :=
        (profileActionOf_mem_opt_iff a s).2 ha
      have hcomp' : profileActionOf U a ∈ (profileCompressedSlice U).toDecisionProblem.Opt s' := by
        simpa [h] using hcomp
      exact (profileActionOf_mem_opt_iff a s').1 hcomp'
    · intro ha
      have hcomp : profileActionOf U a ∈ (profileCompressedSlice U).toDecisionProblem.Opt s' :=
        (profileActionOf_mem_opt_iff a s').2 ha
      have hcomp' : profileActionOf U a ∈ (profileCompressedSlice U).toDecisionProblem.Opt s := by
        simpa [h] using hcomp
      exact (profileActionOf_mem_opt_iff a s).1 hcomp'
  · intro h
    ext p
    have hh := congrArg (fun t : Set U.Action => profileRepresentative p ∈ t) h
    simpa [DecisionProblem.Opt, profileCompressedSlice_isOptimal_iff] using hh

theorem isSufficient_profileCompressedSlice_iff {U : BinaryPairwiseSlice}
    (I : Finset (Fin U.arity)) :
    (profileCompressedSlice U).toDecisionProblem.isSufficient I ↔
      U.toDecisionProblem.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    exact (decisionEquiv_profileCompressedSlice_iff s s').1 (h s s' hagree)
  · intro h s s' hagree
    ext p
    have hh := congrArg (fun t : Set U.Action => profileRepresentative p ∈ t) (h s s' hagree)
    simpa [DecisionProblem.Opt, profileCompressedSlice_isOptimal_iff] using hh

theorem isRelevant_profileCompressedSlice_iff {U : BinaryPairwiseSlice}
    (i : Fin U.arity) :
    (profileCompressedSlice U).toDecisionProblem.isRelevant i ↔
      U.toDecisionProblem.isRelevant i := by
  unfold DecisionProblem.isRelevant
  constructor
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨s, s', hcoord, ?_⟩
    intro hEq
    have hComp : (profileCompressedSlice U).toDecisionProblem.DecisionEquiv s s' := by
      ext p
      have hh := congrArg (fun t : Set U.Action => profileRepresentative p ∈ t) hEq
      simpa [DecisionProblem.Opt, profileCompressedSlice_isOptimal_iff] using hh
    exact hneq hComp
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨s, s', hcoord, ?_⟩
    intro hEq
    exact hneq ((decisionEquiv_profileCompressedSlice_iff s s').1 hEq)

theorem profileCompressedSlice_preserves_exactCertification {U : BinaryPairwiseSlice} :
    ((∀ I : Finset (Fin U.arity),
        (profileCompressedSlice U).toDecisionProblem.isSufficient I ↔
          U.toDecisionProblem.isSufficient I) ∧
      (∀ i : Fin U.arity,
        (profileCompressedSlice U).toDecisionProblem.isRelevant i ↔
          U.toDecisionProblem.isRelevant i)) := by
  exact ⟨isSufficient_profileCompressedSlice_iff, isRelevant_profileCompressedSlice_iff⟩

theorem profileCompressedSlice_bounded_actions {U : BinaryPairwiseSlice} {k : ℕ}
    (hU : distinctActionCount U ≤ k) :
    (profileCompressedSlice U).actionCount ≤ k := by
  simpa [actionCount_profileCompressedSlice U] using hU

end Paper4dFrontier
