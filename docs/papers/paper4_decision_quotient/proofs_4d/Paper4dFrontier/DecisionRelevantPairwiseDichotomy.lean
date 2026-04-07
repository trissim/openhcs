import Paper4dFrontier.BinaryPairwiseDichotomy
import Mathlib.Tactic

namespace Paper4dFrontier

open DecisionQuotient
open Classical

/-- The mixed difference of the action gap `u a - u b`. This is the canonical
decision-relevant binary interaction witness: action-independent pair terms cancel. -/
def actionGapCrossDifference {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (a b : A) (i j : Fin n) : ℤ :=
  pairCrossDifference (fun _ s => u a s - u b s) () i j

def addActionOffset {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (c : A → ℤ) : A → (Fin n → Fin 2) → ℤ :=
  fun a s => u a s + c a

/-- Utilities are offset-equivalent when they differ only by action-dependent,
state-independent constants. -/
def ActionOffsetEquivalent {A : Type*} {n : ℕ}
    (u v : A → (Fin n → Fin 2) → ℤ) : Prop :=
  ∃ c : A → ℤ, v = addActionOffset u c

/-- A pair carries decision-relevant binary interaction if some action gap has
nonzero mixed difference on that pair. -/
def HasDecisionRelevantBinaryPairInteraction {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (i j : Fin n) : Prop :=
  ∃ a b : A, actionGapCrossDifference u a b i j ≠ 0

/-- All action dependence is unary after discarding an action-independent base
state term. Such a base term does not affect the optimizer. -/
def DecisionRelevantUnaryReduction {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) : Prop :=
  ∃ base : (Fin n → Fin 2) → ℤ,
    ∃ unary : Fin n → A → Fin 2 → ℤ,
      ∀ a s, u a s = base s + ∑ i : Fin n, unary i a (s i)

theorem actionGapCrossDifference_comm {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (a b : A) (i j : Fin n) :
    actionGapCrossDifference u a b i j = actionGapCrossDifference u a b j i := by
  unfold actionGapCrossDifference
  simpa using pairCrossDifference_comm (u := fun _ s => u a s - u b s) () i j

theorem actionGapCrossDifference_self_action {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (a : A) (i j : Fin n) :
    actionGapCrossDifference u a a i j = 0 := by
  unfold actionGapCrossDifference pairCrossDifference
  ring

theorem actionGapCrossDifference_swap_actions {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (a b : A) (i j : Fin n) :
    actionGapCrossDifference u a b i j = - actionGapCrossDifference u b a i j := by
  unfold actionGapCrossDifference pairCrossDifference
  ring

theorem actionGapCrossDifference_addActionOffset {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (c : A → ℤ) (a b : A) (i j : Fin n) :
    actionGapCrossDifference (addActionOffset u c) a b i j = actionGapCrossDifference u a b i j := by
  unfold actionGapCrossDifference addActionOffset pairCrossDifference
  ring

theorem actionOffsetEquivalent_refl {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) : ActionOffsetEquivalent u u := by
  refine ⟨fun _ => 0, ?_⟩
  funext a s
  simp [addActionOffset]

theorem actionOffsetEquivalent_symm {A : Type*} {n : ℕ}
    {u v : A → (Fin n → Fin 2) → ℤ} :
    ActionOffsetEquivalent u v → ActionOffsetEquivalent v u := by
  rintro ⟨c, rfl⟩
  refine ⟨fun a => -c a, ?_⟩
  funext a s
  simp [addActionOffset]

theorem actionOffsetEquivalent_trans {A : Type*} {n : ℕ}
    {u v w : A → (Fin n → Fin 2) → ℤ} :
    ActionOffsetEquivalent u v → ActionOffsetEquivalent v w → ActionOffsetEquivalent u w := by
  rintro ⟨c, rfl⟩ ⟨d, rfl⟩
  refine ⟨fun a => c a + d a, ?_⟩
  funext a s
  simp [addActionOffset]
  ring

theorem HasDecisionRelevantBinaryPairInteraction_symm {A : Type*} {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ} :
    ∀ i j, HasDecisionRelevantBinaryPairInteraction u i j →
      HasDecisionRelevantBinaryPairInteraction u j i := by
  intro i j h
  rcases h with ⟨a, b, hab⟩
  refine ⟨a, b, ?_⟩
  rw [actionGapCrossDifference_comm u a b j i]
  exact hab

def decisionRelevantInteractionGraph {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) : SimpleGraph (Fin n) :=
  InteractionGraph (HasDecisionRelevantBinaryPairInteraction u)
    (HasDecisionRelevantBinaryPairInteraction_symm (u := u))

/-- The decision-relevant interaction graph is already well-defined on action-offset
equivalence classes, so the normalized graph is represented by the same object. -/
def offsetNormalizedDecisionRelevantInteractionGraph {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) : SimpleGraph (Fin n) :=
  decisionRelevantInteractionGraph u

theorem decisionRelevantInteractionGraph_addActionOffset {A : Type*} {n : ℕ}
    (u : A → (Fin n → Fin 2) → ℤ) (c : A → ℤ) :
    decisionRelevantInteractionGraph (addActionOffset u c) = decisionRelevantInteractionGraph u := by
  ext i j
  by_cases h : i = j
  · subst h
    simp [decisionRelevantInteractionGraph, InteractionGraph]
  · simp [decisionRelevantInteractionGraph, InteractionGraph, h,
      HasDecisionRelevantBinaryPairInteraction, actionGapCrossDifference_addActionOffset]

theorem offsetNormalizedDecisionRelevantInteractionGraph_wellDefined {A : Type*} {n : ℕ}
    {u v : A → (Fin n → Fin 2) → ℤ} :
    ActionOffsetEquivalent u v →
      offsetNormalizedDecisionRelevantInteractionGraph u =
        offsetNormalizedDecisionRelevantInteractionGraph v := by
  rintro ⟨c, rfl⟩
  simp [offsetNormalizedDecisionRelevantInteractionGraph,
    decisionRelevantInteractionGraph_addActionOffset]

/-- Decision-relevant interaction restricted to a designated support of actions.
This is the natural next normalization after the ghost-action obstruction. -/
def SupportedDecisionRelevantBinaryPairInteraction {A : Type*} {n : ℕ}
    (support : A → Prop) (u : A → (Fin n → Fin 2) → ℤ) (i j : Fin n) : Prop :=
  ∃ a b : A, support a ∧ support b ∧ actionGapCrossDifference u a b i j ≠ 0

theorem SupportedDecisionRelevantBinaryPairInteraction_symm {A : Type*} {n : ℕ}
    (support : A → Prop) {u : A → (Fin n → Fin 2) → ℤ} :
    ∀ i j, SupportedDecisionRelevantBinaryPairInteraction support u i j →
      SupportedDecisionRelevantBinaryPairInteraction support u j i := by
  intro i j h
  rcases h with ⟨a, b, ha, hb, hab⟩
  exact ⟨a, b, ha, hb, by rw [actionGapCrossDifference_comm u a b j i]; exact hab⟩

def supportedDecisionRelevantInteractionGraph {A : Type*} {n : ℕ}
    (support : A → Prop) (u : A → (Fin n → Fin 2) → ℤ) : SimpleGraph (Fin n) :=
  InteractionGraph (SupportedDecisionRelevantBinaryPairInteraction support u)
    (SupportedDecisionRelevantBinaryPairInteraction_symm support (u := u))

def OptimizerSupported {A S : Type*} (dp : DecisionProblem A S) (a : A) : Prop :=
  ∃ s : S, a ∈ dp.Opt s

noncomputable def maxPairInteractionMagnitude {A : Type*} [Fintype A] {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ} (pw : PairwiseUtility u) : Nat :=
  Finset.univ.sup fun i : Fin n =>
    Finset.univ.sup fun j : Fin n =>
      Finset.univ.sup fun a : A =>
        Int.natAbs (binaryCrossDifference (pw.binary i j a))

/-- A binary pairwise utility is margin-bounded when every unary term is at most
twice the largest pairwise mixed-difference magnitude. -/
def MarginBounded {A : Type*} [Fintype A] {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ} (pw : PairwiseUtility u) : Prop :=
  ∀ i : Fin n, ∀ a : A, ∀ x : Fin 2,
    Int.natAbs (pw.unary i a x) ≤ 2 * maxPairInteractionMagnitude pw

theorem marginBounded_of_unary_zero {A : Type*} [Fintype A] {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ} (pw : PairwiseUtility u)
    (hzero : ∀ i : Fin n, ∀ a : A, ∀ x : Fin 2, pw.unary i a x = 0) :
    MarginBounded pw := by
  intro i a x
  rw [hzero i a x]
  simpa using (Nat.zero_le (2 * maxPairInteractionMagnitude pw))

/-- Pairwise decomposition of an action gap. -/
noncomputable def actionGapPairwise {A : Type*} {n : ℕ}
    {u : A → (Fin n → Fin 2) → ℤ}
    (pw : PairwiseUtility u) (a b : A) :
    PairwiseUtility (fun _ : Unit => fun s => u a s - u b s) where
  unary i _ x := pw.unary i a x - pw.unary i b x
  binary i j _ x y := pw.binary i j a x y - pw.binary i j b x y
  interacts := pw.interacts
  interacts_symm := pw.interacts_symm
  decomp := by
    intro _ s
    let fa : Fin n → Fin n → ℤ := fun i j =>
      if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) else 0
    let fb : Fin n → Fin n → ℤ := fun i j =>
      if pw.interacts i j ∧ i < j then pw.binary i j b (s i) (s j) else 0
    have hinner : ∀ i : Fin n,
        (∑ j : Fin n, fa i j) - (∑ j : Fin n, fb i j) =
          ∑ j : Fin n, (fa i j - fb i j) := by
      intro i
      simpa using
        (Finset.sum_sub_distrib (s := Finset.univ)
          (f := fun j : Fin n => fa i j) (g := fun j : Fin n => fb i j)).symm
    have hbinary :
        (∑ i : Fin n, ∑ j : Fin n, fa i j) - (∑ i : Fin n, ∑ j : Fin n, fb i j) =
          ∑ i : Fin n, ∑ j : Fin n, (fa i j - fb i j) := by
      calc
        (∑ i : Fin n, ∑ j : Fin n, fa i j) - (∑ i : Fin n, ∑ j : Fin n, fb i j)
            = ∑ i : Fin n, ((∑ j : Fin n, fa i j) - (∑ j : Fin n, fb i j)) := by
                simpa using
                  (Finset.sum_sub_distrib (s := Finset.univ)
                    (f := fun i : Fin n => ∑ j : Fin n, fa i j)
                    (g := fun i : Fin n => ∑ j : Fin n, fb i j)).symm
        _ = ∑ i : Fin n, ∑ j : Fin n, (fa i j - fb i j) := by
              refine Finset.sum_congr rfl ?_
              intro i hi
              exact hinner i
    have hpoint : ∀ i j : Fin n,
        fa i j - fb i j =
          if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) - pw.binary i j b (s i) (s j) else 0 := by
      intro i j
      by_cases h : pw.interacts i j ∧ i < j
      · simp [fa, fb, h]
      · simp [fa, fb, h]
    have hunary :
        (∑ i : Fin n, pw.unary i a (s i)) - (∑ i : Fin n, pw.unary i b (s i)) =
          ∑ i : Fin n, (pw.unary i a (s i) - pw.unary i b (s i)) := by
      simpa using
        (Finset.sum_sub_distrib (s := Finset.univ)
          (f := fun i : Fin n => pw.unary i a (s i))
          (g := fun i : Fin n => pw.unary i b (s i))).symm
    have hbinary' :
        (∑ i : Fin n, ∑ j : Fin n, fa i j) - (∑ i : Fin n, ∑ j : Fin n, fb i j) =
          ∑ i : Fin n, ∑ j : Fin n,
            if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) - pw.binary i j b (s i) (s j) else 0 := by
      calc
        (∑ i : Fin n, ∑ j : Fin n, fa i j) - (∑ i : Fin n, ∑ j : Fin n, fb i j)
            = ∑ i : Fin n, ∑ j : Fin n, (fa i j - fb i j) := hbinary
        _ = ∑ i : Fin n, ∑ j : Fin n,
              if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) - pw.binary i j b (s i) (s j) else 0 := by
              refine Finset.sum_congr rfl ?_
              intro i hi
              refine Finset.sum_congr rfl ?_
              intro j hj
              exact hpoint i j
    rw [pw.decomp a s, pw.decomp b s]
    calc
      ((∑ i : Fin n, pw.unary i a (s i)) + ∑ i : Fin n, ∑ j : Fin n, fa i j) -
          ((∑ i : Fin n, pw.unary i b (s i)) + ∑ i : Fin n, ∑ j : Fin n, fb i j)
        = ((∑ i : Fin n, pw.unary i a (s i)) - (∑ i : Fin n, pw.unary i b (s i))) +
            ((∑ i : Fin n, ∑ j : Fin n, fa i j) - (∑ i : Fin n, ∑ j : Fin n, fb i j)) := by
              ring
      _ = (∑ i : Fin n, (pw.unary i a (s i) - pw.unary i b (s i))) +
            (∑ i : Fin n, ∑ j : Fin n,
              if pw.interacts i j ∧ i < j then pw.binary i j a (s i) (s j) - pw.binary i j b (s i) (s j) else 0) := by
              rw [hunary, hbinary']

theorem actionGapCrossDifference_eq_binaryCrossDifference_of_lt
    {A : Type*} {n : ℕ} {u : A → (Fin n → Fin 2) → ℤ}
    (pw : PairwiseUtility u) (a b : A) {i j : Fin n} (hij : i < j) :
    actionGapCrossDifference u a b i j =
      if pw.interacts i j then
        binaryCrossDifference (fun x y => pw.binary i j a x y - pw.binary i j b x y)
      else 0 := by
  simpa [actionGapCrossDifference] using
    (pairCrossDifference_eq_binaryCrossDifference_of_lt (pw := actionGapPairwise pw a b) (a := ()) hij)

theorem actionGapCrossDifference_eq_of_symmetry
    {A : Type*} {n : ℕ} {u : A → (Fin n → Fin 2) → ℤ}
    (hsym : SymmetricUtility (fun a s => u a s.state))
    (a b : A) (i j p q : Fin n) (hij : i ≠ j) (hpq : p ≠ q) :
    actionGapCrossDifference u a b p q = actionGapCrossDifference u a b i j := by
  unfold actionGapCrossDifference
  have hai := pairCrossDifference_eq_of_symmetry (u := u) hsym a i j p q hij hpq
  have hbi := pairCrossDifference_eq_of_symmetry (u := u) hsym b i j p q hij hpq
  unfold pairCrossDifference at hai hbi ⊢
  linarith

theorem decisionRelevant_zero_actionGap_implies_unaryReduction
    {A : Type*} {n : ℕ} {u : A → (Fin n → Fin 2) → ℤ}
    (pw : PairwiseUtility u) (a0 : A)
    (hzero : ∀ a : A, ∀ i j : Fin n, i < j → actionGapCrossDifference u a a0 i j = 0) :
    DecisionRelevantUnaryReduction u := by
  classical
  have hcollapse :
      ∀ a : A,
        ∃ unary : Fin n → Unit → Fin 2 → ℤ,
          ∀ _ : Unit, ∀ s,
            (u a s - u a0 s) = ∑ i : Fin n, unary i () (s i) := by
    intro a
    rcases pairwise_zero_crossDifference_unaryDecomposition
      (pw := actionGapPairwise pw a a0)
      (hzero := by
        intro _ i j hij
        simpa [actionGapCrossDifference] using hzero a i j hij) with ⟨unary, hunary⟩
    exact ⟨unary, hunary⟩
  choose unaryGap hunary using hcollapse
  refine ⟨u a0, fun i a x => unaryGap a i () x, ?_⟩
  intro a s
  have h := hunary a () s
  linarith

theorem binary_pairwise_symmetry_decision_relevant_dichotomy
    {A : Type*} {n : ℕ} {u : A → (Fin n → Fin 2) → ℤ}
    (pw : PairwiseUtility u)
    (hsym : SymmetricUtility (fun a s => u a s.state))
    (a0 : A) :
    DecisionRelevantUnaryReduction u ∨
      ∀ i j : Fin n, i ≠ j → HasDecisionRelevantBinaryPairInteraction u i j := by
  classical
  by_cases hzero : ∀ a : A, ∀ i j : Fin n, i < j → actionGapCrossDifference u a a0 i j = 0
  · exact Or.inl (decisionRelevant_zero_actionGap_implies_unaryReduction (pw := pw) a0 hzero)
  · push_neg at hzero
    rcases hzero with ⟨a, i, j, hij, hneq⟩
    refine Or.inr ?_
    intro p q hpq
    refine ⟨a, a0, ?_⟩
    have heq := actionGapCrossDifference_eq_of_symmetry (u := u) hsym a a0 i j p q hij.ne hpq
    rw [heq]
    exact hneq

end Paper4dFrontier
