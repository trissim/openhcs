import Paper4dFrontier.FamilyAxioms

namespace Paper4dFrontier

open DecisionQuotient

/-- If the optimizer set is constant across states, the empty coordinate set is sufficient. -/
theorem empty_sufficient_of_constant_opt {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (hConst : ∀ s s' : S, dp.Opt s = dp.Opt s') :
    dp.isSufficient ∅ := by
  intro s s' _
  simpa [DecisionProblem.DecisionEquiv] using hConst s s'

theorem irrelevant_of_constant_opt {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (hConst : ∀ s s' : S, dp.Opt s = dp.Opt s') (i : Fin n) :
    dp.isIrrelevant i := by
  rw [dp.irrelevant_iff_not_relevant]
  intro hRel
  rcases hRel with ⟨s, s', _, hneq⟩
  exact hneq (by simpa [DecisionProblem.DecisionEquiv] using hConst s s')

/-- Duplicate every action by a Boolean tag while keeping utility unchanged. -/
def duplicateActionProblem {A S : Type*} (dp : DecisionProblem A S) : DecisionProblem (Bool × A) S where
  utility ba s := dp.utility ba.2 s

theorem isOptimal_duplicateAction_iff {A S : Type*}
    (dp : DecisionProblem A S) (ba : Bool × A) (s : S) :
    (duplicateActionProblem dp).isOptimal ba s ↔ dp.isOptimal ba.2 s := by
  constructor
  · intro h a'
    have hh := h (false, a')
    simpa [DecisionProblem.isOptimal, duplicateActionProblem] using hh
  · intro h ba'
    have hh := h ba'.2
    simpa [DecisionProblem.isOptimal, duplicateActionProblem] using hh

theorem decisionEquiv_duplicateAction_iff {A S : Type*}
    (dp : DecisionProblem A S) (s s' : S) :
    (duplicateActionProblem dp).DecisionEquiv s s' ↔ dp.DecisionEquiv s s' := by
  unfold DecisionProblem.DecisionEquiv
  constructor
  · intro h
    ext a
    have hh := congrArg (fun t : Set (Bool × A) => (false, a) ∈ t) h
    simpa [DecisionProblem.Opt, isOptimal_duplicateAction_iff] using hh
  · intro h
    ext ba
    have hh := congrArg (fun t : Set A => ba.2 ∈ t) h
    simpa [DecisionProblem.Opt, isOptimal_duplicateAction_iff] using hh

theorem isSufficient_duplicateAction_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    (duplicateActionProblem dp).isSufficient I ↔ dp.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    exact (decisionEquiv_duplicateAction_iff dp s s').1 (h s s' hagree)
  · intro h s s' hagree
    exact (decisionEquiv_duplicateAction_iff dp s s').2 (h s s' hagree)

theorem isRelevant_duplicateAction_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n) :
    (duplicateActionProblem dp).isRelevant i ↔ dp.isRelevant i := by
  unfold DecisionProblem.isRelevant
  constructor
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨s, s', hcoord, ?_⟩
    intro hEq
    exact hneq ((decisionEquiv_duplicateAction_iff dp s s').2 hEq)
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨s, s', hcoord, ?_⟩
    intro hEq
    exact hneq ((decisionEquiv_duplicateAction_iff dp s s').1 hEq)

theorem isIrrelevant_duplicateAction_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n) :
    (duplicateActionProblem dp).isIrrelevant i ↔ dp.isIrrelevant i := by
  rw [DecisionProblem.irrelevant_iff_not_relevant, DecisionProblem.irrelevant_iff_not_relevant,
    isRelevant_duplicateAction_iff dp i]

/-- Duplicate every state by a Boolean tag while keeping utility unchanged. -/
def duplicateStateProblem {A S : Type*} (dp : DecisionProblem A S) : DecisionProblem A (Bool × S) where
  utility a bs := dp.utility a bs.2

instance duplicateStateCoordinateSpace {S : Type*} {n : ℕ} [CoordinateSpace S n] :
    CoordinateSpace (Bool × S) n where
  Coord := fun i => CoordinateSpace.Coord (S := S) i
  proj := fun bs i => CoordinateSpace.proj (S := S) bs.2 i

theorem agreeOn_duplicateState_iff {S : Type*} {n : ℕ} [CoordinateSpace S n]
    (x y : Bool × S) (I : Finset (Fin n)) :
    agreeOn x y I ↔ agreeOn x.2 y.2 I := by
  rfl

theorem isOptimal_duplicateState_iff {A S : Type*}
    (dp : DecisionProblem A S) (a : A) (bs : Bool × S) :
    (duplicateStateProblem dp).isOptimal a bs ↔ dp.isOptimal a bs.2 := by
  rfl

theorem decisionEquiv_duplicateState_iff {A S : Type*}
    (dp : DecisionProblem A S) (x y : Bool × S) :
    (duplicateStateProblem dp).DecisionEquiv x y ↔ dp.DecisionEquiv x.2 y.2 := by
  simp [DecisionProblem.DecisionEquiv, DecisionProblem.Opt, isOptimal_duplicateState_iff]

theorem isSufficient_duplicateState_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    @DecisionProblem.isSufficient A (Bool × S) n (duplicateStateCoordinateSpace) (duplicateStateProblem dp) I
      ↔ dp.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    have hEq := h (false, s) (false, s') ((agreeOn_duplicateState_iff (false, s) (false, s') I).2 hagree)
    simpa using (decisionEquiv_duplicateState_iff dp (false, s) (false, s')).1 hEq
  · intro h x y hagree
    have hEq := h x.2 y.2 ((agreeOn_duplicateState_iff x y I).1 hagree)
    exact (decisionEquiv_duplicateState_iff dp x y).2 hEq

theorem isRelevant_duplicateState_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n) :
    @DecisionProblem.isRelevant A (Bool × S) n (duplicateStateCoordinateSpace) (duplicateStateProblem dp) i
      ↔ dp.isRelevant i := by
  unfold DecisionProblem.isRelevant
  constructor
  · rintro ⟨x, y, hcoord, hneq⟩
    refine ⟨x.2, y.2, ?_, ?_⟩
    · intro j hj
      exact hcoord j hj
    · intro hEq
      exact hneq ((decisionEquiv_duplicateState_iff dp x y).2 hEq)
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨(false, s), (false, s'), ?_, ?_⟩
    · intro j hj
      exact hcoord j hj
    · intro hEq
      exact hneq ((decisionEquiv_duplicateState_iff dp (false, s) (false, s')).1 hEq)

theorem isIrrelevant_duplicateState_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (i : Fin n) :
    @DecisionProblem.isIrrelevant A (Bool × S) n (duplicateStateCoordinateSpace) (duplicateStateProblem dp) i
      ↔ dp.isIrrelevant i := by
  rw [DecisionProblem.irrelevant_iff_not_relevant, DecisionProblem.irrelevant_iff_not_relevant,
    isRelevant_duplicateState_iff dp i]

noncomputable def duplicateStateQuotientToOriginal {A S : Type*}
    (dp : DecisionProblem A S) :
    (duplicateStateProblem dp).DecisionQuotientType → dp.DecisionQuotientType :=
  Quotient.lift
    (fun x => dp.quotientMap x.2)
    (fun x y h =>
      (dp.quotient_represents_opt_equiv x.2 y.2).2
        (by simpa [DecisionProblem.DecisionEquiv] using (decisionEquiv_duplicateState_iff dp x y).1 h))

theorem duplicateStateQuotientToOriginal_surjective {A S : Type*}
    (dp : DecisionProblem A S) :
    Function.Surjective (duplicateStateQuotientToOriginal dp) := by
  intro q
  refine Quotient.inductionOn q ?_
  intro s
  exact ⟨(duplicateStateProblem dp).quotientMap (false, s), rfl⟩

theorem duplicateStateQuotientToOriginal_injective {A S : Type*}
    (dp : DecisionProblem A S) :
    Function.Injective (duplicateStateQuotientToOriginal dp) := by
  intro q q' h
  refine Quotient.inductionOn₂ q q' ?_ h
  intro x y hxy
  apply Quotient.sound
  exact (decisionEquiv_duplicateState_iff dp x y).2
    (by simpa [DecisionProblem.DecisionEquiv] using (dp.quotient_represents_opt_equiv x.2 y.2).1 hxy)

noncomputable def duplicateStateQuotientEquivOriginal {A S : Type*}
    (dp : DecisionProblem A S) :
    (duplicateStateProblem dp).DecisionQuotientType ≃ dp.DecisionQuotientType :=
  Equiv.ofBijective (duplicateStateQuotientToOriginal dp)
    ⟨duplicateStateQuotientToOriginal_injective dp,
      duplicateStateQuotientToOriginal_surjective dp⟩

theorem duplicateStateQuotientEquivOriginal_apply_quotientMap {A S : Type*}
    (dp : DecisionProblem A S) (x : Bool × S) :
    duplicateStateQuotientEquivOriginal dp ((duplicateStateProblem dp).quotientMap x) = dp.quotientMap x.2 := rfl

/-- Add one distinguished duplicate action. `none` duplicates `a0`, while `some a`
keeps the original action `a`. -/
def addDuplicateActionProblem {A S : Type*} (dp : DecisionProblem A S) (a0 : A) :
    DecisionProblem (Option A) S where
  utility oa s := match oa with | none => dp.utility a0 s | some a => dp.utility a s

theorem isOptimal_addDuplicateAction_iff {A S : Type*}
    (dp : DecisionProblem A S) (a0 : A) (oa : Option A) (s : S) :
    (addDuplicateActionProblem dp a0).isOptimal oa s ↔
      match oa with
      | none => dp.isOptimal a0 s
      | some a => dp.isOptimal a s := by
  cases oa with
  | none =>
      constructor
      · intro h a'
        have hh := h (some a')
        simpa [DecisionProblem.isOptimal, addDuplicateActionProblem] using hh
      · intro h oa'
        cases oa' with
        | none => simpa [DecisionProblem.isOptimal, addDuplicateActionProblem] using h a0
        | some a' => simpa [DecisionProblem.isOptimal, addDuplicateActionProblem] using h a'
  | some a =>
      constructor
      · intro h a'
        have hh := h (some a')
        simpa [DecisionProblem.isOptimal, addDuplicateActionProblem] using hh
      · intro h oa'
        cases oa' with
        | none => simpa [DecisionProblem.isOptimal, addDuplicateActionProblem] using h a0
        | some a' => simpa [DecisionProblem.isOptimal, addDuplicateActionProblem] using h a'

theorem decisionEquiv_addDuplicateAction_iff {A S : Type*}
    (dp : DecisionProblem A S) (a0 : A) (s s' : S) :
    (addDuplicateActionProblem dp a0).DecisionEquiv s s' ↔ dp.DecisionEquiv s s' := by
  unfold DecisionProblem.DecisionEquiv
  constructor
  · intro h
    ext a
    have hh := congrArg (fun t : Set (Option A) => some a ∈ t) h
    simpa [DecisionProblem.Opt, isOptimal_addDuplicateAction_iff] using hh
  · intro h
    ext oa
    cases oa with
    | none =>
        have hh := congrArg (fun t : Set A => a0 ∈ t) h
        simpa [DecisionProblem.Opt, isOptimal_addDuplicateAction_iff] using hh
    | some a =>
        have hh := congrArg (fun t : Set A => a ∈ t) h
        simpa [DecisionProblem.Opt, isOptimal_addDuplicateAction_iff] using hh

theorem some_mem_opt_addDuplicateAction_iff {A S : Type*}
    (dp : DecisionProblem A S) (a0 a : A) (s : S) :
    some a ∈ (addDuplicateActionProblem dp a0).Opt s ↔ a ∈ dp.Opt s := by
  simp [DecisionProblem.Opt, isOptimal_addDuplicateAction_iff]

theorem none_mem_opt_addDuplicateAction_iff {A S : Type*}
    (dp : DecisionProblem A S) (a0 : A) (s : S) :
    none ∈ (addDuplicateActionProblem dp a0).Opt s ↔ a0 ∈ dp.Opt s := by
  simp [DecisionProblem.Opt, isOptimal_addDuplicateAction_iff]

theorem isSufficient_addDuplicateAction_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (a0 : A) (I : Finset (Fin n)) :
    (addDuplicateActionProblem dp a0).isSufficient I ↔ dp.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    exact (decisionEquiv_addDuplicateAction_iff dp a0 s s').1 (h s s' hagree)
  · intro h s s' hagree
    exact (decisionEquiv_addDuplicateAction_iff dp a0 s s').2 (h s s' hagree)

theorem isRelevant_addDuplicateAction_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (a0 : A) (i : Fin n) :
    (addDuplicateActionProblem dp a0).isRelevant i ↔ dp.isRelevant i := by
  unfold DecisionProblem.isRelevant
  constructor
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨s, s', hcoord, ?_⟩
    intro hEq
    exact hneq ((decisionEquiv_addDuplicateAction_iff dp a0 s s').2 hEq)
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨s, s', hcoord, ?_⟩
    intro hEq
    exact hneq ((decisionEquiv_addDuplicateAction_iff dp a0 s s').1 hEq)

/-- Add one distinguished duplicate state. `none` duplicates the fixed state `s0`. -/
def addDuplicateStateRep {S : Type*} (s0 : S) : Option S → S
  | none => s0
  | some s => s

def addDuplicateStateProblem {A S : Type*} (dp : DecisionProblem A S) (s0 : S) :
    DecisionProblem A (Option S) where
  utility a os := dp.utility a (addDuplicateStateRep s0 os)

instance addDuplicateStateCoordinateSpace {S : Type*} {n : ℕ} [CoordinateSpace S n] (s0 : S) :
    CoordinateSpace (Option S) n where
  Coord := fun i => CoordinateSpace.Coord (S := S) i
  proj := fun os i => CoordinateSpace.proj (S := S) (addDuplicateStateRep s0 os) i

theorem decisionEquiv_addDuplicateState_iff {A S : Type*}
    (dp : DecisionProblem A S) (s0 : S) (x y : Option S) :
    (addDuplicateStateProblem dp s0).DecisionEquiv x y ↔
      dp.DecisionEquiv (addDuplicateStateRep s0 x) (addDuplicateStateRep s0 y) := by
  unfold DecisionProblem.DecisionEquiv
  constructor
  · intro h
    ext a
    have hh := congrArg (fun t : Set A => a ∈ t) h
    simpa [DecisionProblem.Opt, addDuplicateStateProblem, addDuplicateStateRep] using hh
  · intro h
    ext a
    have hh := congrArg (fun t : Set A => a ∈ t) h
    simpa [DecisionProblem.Opt, addDuplicateStateProblem, addDuplicateStateRep] using hh

theorem isSufficient_addDuplicateState_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (s0 : S) (I : Finset (Fin n)) :
    @DecisionProblem.isSufficient A (Option S) n (addDuplicateStateCoordinateSpace s0)
      (addDuplicateStateProblem dp s0) I ↔ dp.isSufficient I := by
  unfold DecisionProblem.isSufficient
  constructor
  · intro h s s' hagree
    have hEq := h (some s) (some s') hagree
    simpa [addDuplicateStateRep] using
      (decisionEquiv_addDuplicateState_iff dp s0 (some s) (some s')).1 hEq
  · intro h x y hagree
    have hEq := h (addDuplicateStateRep s0 x) (addDuplicateStateRep s0 y) hagree
    exact (decisionEquiv_addDuplicateState_iff dp s0 x y).2 hEq

theorem isRelevant_addDuplicateState_iff {A S : Type*} {n : ℕ} [CoordinateSpace S n]
    (dp : DecisionProblem A S) (s0 : S) (i : Fin n) :
    @DecisionProblem.isRelevant A (Option S) n (addDuplicateStateCoordinateSpace s0)
      (addDuplicateStateProblem dp s0) i ↔ dp.isRelevant i := by
  unfold DecisionProblem.isRelevant
  constructor
  · rintro ⟨x, y, hcoord, hneq⟩
    refine ⟨addDuplicateStateRep s0 x, addDuplicateStateRep s0 y, hcoord, ?_⟩
    intro hEq
    exact hneq ((decisionEquiv_addDuplicateState_iff dp s0 x y).2 hEq)
  · rintro ⟨s, s', hcoord, hneq⟩
    refine ⟨some s, some s', hcoord, ?_⟩
    intro hEq
    exact hneq ((decisionEquiv_addDuplicateState_iff dp s0 (some s) (some s')).1 hEq)

noncomputable def addDuplicateStateQuotientToOriginal {A S : Type*}
    (dp : DecisionProblem A S) (s0 : S) :
    (addDuplicateStateProblem dp s0).DecisionQuotientType → dp.DecisionQuotientType :=
  Quotient.lift
    (fun x => dp.quotientMap (addDuplicateStateRep s0 x))
    (fun x y h =>
      (dp.quotient_represents_opt_equiv _ _).2
        ((decisionEquiv_addDuplicateState_iff dp s0 x y).1 h))

theorem addDuplicateStateQuotientToOriginal_surjective {A S : Type*}
    (dp : DecisionProblem A S) (s0 : S) :
    Function.Surjective (addDuplicateStateQuotientToOriginal dp s0) := by
  intro q
  refine Quotient.inductionOn q ?_
  intro s
  exact ⟨(addDuplicateStateProblem dp s0).quotientMap (some s), rfl⟩

theorem addDuplicateStateQuotientToOriginal_injective {A S : Type*}
    (dp : DecisionProblem A S) (s0 : S) :
    Function.Injective (addDuplicateStateQuotientToOriginal dp s0) := by
  intro q q' h
  refine Quotient.inductionOn₂ q q' ?_ h
  intro x y hxy
  apply Quotient.sound
  exact (decisionEquiv_addDuplicateState_iff dp s0 x y).2
    ((dp.quotient_represents_opt_equiv _ _).1 hxy)

noncomputable def addDuplicateStateQuotientEquivOriginal {A S : Type*}
    (dp : DecisionProblem A S) (s0 : S) :
    (addDuplicateStateProblem dp s0).DecisionQuotientType ≃ dp.DecisionQuotientType :=
  Equiv.ofBijective (addDuplicateStateQuotientToOriginal dp s0)
    ⟨addDuplicateStateQuotientToOriginal_injective dp s0,
      addDuplicateStateQuotientToOriginal_surjective dp s0⟩

theorem addDuplicateStateQuotientEquivOriginal_apply_quotientMap {A S : Type*}
    (dp : DecisionProblem A S) (s0 : S) (x : Option S) :
    addDuplicateStateQuotientEquivOriginal dp s0 ((addDuplicateStateProblem dp s0).quotientMap x) =
      dp.quotientMap (addDuplicateStateRep s0 x) := by
  rfl

end Paper4dFrontier
