import DecisionQuotient.Instances
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

/-!
  Paper 4: Decision-Relevant Uncertainty

  DecisionGeometry.lean

  Geometry machinery for product decision spaces:
  - anchored coordinate slices (fibers),
  - exact finite cardinalities of fibers,
  - geometric decay under coordinate fixation,
  - explicit node-count vs edge-structure separation.

  ## Triviality Level
  TRIVIAL: This provides geometric infrastructure. The nontrivial work is
  in using this geometry to prove hardness results.

  ## Dependencies
  - Chain: Instances.lean → here → Instantiation.lean (physics)
-/

namespace DecisionQuotient

open Classical

section FiberGeometry

variable {n : ℕ}
variable {X : Type*} [Fintype X]

/-- Outside-index type for a coordinate set `I`. -/
abbrev Outside (I : Finset (Fin n)) := { i : Fin n // i ∉ I }

/-- Anchored slice: all states that agree with `anchor` on coordinates in `I`. -/
def anchoredSlice (I : Finset (Fin n)) (anchor : Fin n → X) : Set (Fin n → X) :=
  { s | ∀ i ∈ I, s i = anchor i }

/-- Fiber equivalence: anchored slices are exactly functions on the unfixed coordinates. -/
noncomputable def anchoredSliceEquivOutside
    (I : Finset (Fin n)) (anchor : Fin n → X) :
    { s : Fin n → X // s ∈ anchoredSlice I anchor } ≃ (Outside I → X) where
  toFun := fun s i => s.1 i.1
  invFun := fun g =>
    ⟨
      fun i =>
        if hi : i ∈ I then anchor i else g ⟨i, hi⟩,
      by
        intro i hi
        simp [hi]
    ⟩
  left_inv := by
    intro s
    apply Subtype.ext
    funext i
    by_cases hi : i ∈ I
    · have hs : s.1 i = anchor i := s.2 i hi
      simp [hi, hs]
    · simp [hi]
  right_inv := by
    intro g
    funext i
    simp [i.2]

/-- Exact number of free coordinates after fixing `I`. -/
theorem card_outside_eq_sub (I : Finset (Fin n)) :
    Fintype.card (Outside I) = n - I.card := by
  simpa [Outside, Fintype.card_fin] using
    (Fintype.card_subtype_compl (fun i : Fin n => i ∈ I))

/-- Exact fiber cardinality in product space `X^n`. -/
theorem card_anchoredSlice
    (I : Finset (Fin n)) (anchor : Fin n → X) :
    Fintype.card { s : Fin n → X // s ∈ anchoredSlice I anchor } =
      (Fintype.card X) ^ (Fintype.card (Outside I)) := by
  simpa [Fintype.card_fun] using
    (Fintype.card_congr (anchoredSliceEquivOutside (I := I) (anchor := anchor)))

/-- Geometric form: fixing `|I|` coordinates leaves `|X|^(n-|I|)` states. -/
theorem card_anchoredSlice_eq_pow_sub
    (I : Finset (Fin n)) (anchor : Fin n → X) :
    Fintype.card { s : Fin n → X // s ∈ anchoredSlice I anchor } =
      (Fintype.card X) ^ (n - I.card) := by
  rw [card_anchoredSlice (I := I) (anchor := anchor), card_outside_eq_sub (I := I)]

/-- Uniform-domain specialization with explicit base cardinal `k`. -/
theorem card_anchoredSlice_eq_uniform
    (I : Finset (Fin n)) (anchor : Fin n → X) (k : ℕ)
    (hk : Fintype.card X = k) :
    Fintype.card { s : Fin n → X // s ∈ anchoredSlice I anchor } =
      k ^ (n - I.card) := by
  rw [card_anchoredSlice_eq_pow_sub (I := I) (anchor := anchor), hk]

/-- Multiplicative conservation identity for slice size. -/
theorem anchoredSlice_mul_fixed_eq_full
    (I : Finset (Fin n)) (anchor : Fin n → X) :
    Fintype.card { s : Fin n → X // s ∈ anchoredSlice I anchor } * (Fintype.card X) ^ I.card =
      (Fintype.card X) ^ n := by
  have hle : I.card ≤ n := by
    simpa [Fintype.card_fin] using (Finset.card_le_univ (s := I))
  rw [card_anchoredSlice_eq_pow_sub (I := I) (anchor := anchor)]
  rw [← Nat.pow_add, Nat.sub_add_cancel hle]

end FiberGeometry

section EdgeGeometry

variable {A S : Type*} {n : ℕ} [CoordinateSpace S n]

/-- Edge-structure witness relative to fixed coordinates `I`:
    there exists a decision-boundary difference hidden outside `I`. -/
def DecisionProblem.edgeOnComplement
    (dp : DecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  ∃ s s' : S, agreeOn s s' I ∧ dp.Opt s ≠ dp.Opt s'

/-- Edge witness is exactly non-sufficiency. -/
theorem DecisionProblem.edgeOnComplement_iff_not_sufficient
    (dp : DecisionProblem A S) (I : Finset (Fin n)) :
    dp.edgeOnComplement I ↔ ¬ dp.isSufficient I := by
  simpa [DecisionProblem.edgeOnComplement] using
    (DecisionProblem.not_sufficient_iff_exists_counterexample (dp := dp) (I := I)).symm

end EdgeGeometry

section NodeEdgeSeparation

variable {n : ℕ}

/-- Constant decision problem on Boolean hypercube: no edge structure. -/
def constantBoolDP (n : ℕ) : DecisionProblem Bool (Fin n → Bool) where
  utility := fun _ _ => 0

/-- First-coordinate decision problem: edge structure depends on coordinate `0`. -/
def firstCoordDP (n : ℕ) (h : 0 < n) : DecisionProblem Bool (Fin n → Bool) where
  utility := fun a s => if a = s ⟨0, h⟩ then 1 else 0

/-- Constant Boolean problem has full optimal set at every state. -/
theorem constantBoolDP_opt (n : ℕ) (s : Fin n → Bool) :
    (constantBoolDP n).Opt s = (Set.univ : Set Bool) := by
  ext a
  constructor
  · intro _
    simp
  · intro _ a'
    simp [constantBoolDP]

/-- First-coordinate problem picks exactly the state's first-coordinate bit. -/
theorem firstCoordDP_opt (n : ℕ) (h : 0 < n) (s : Fin n → Bool) :
    (firstCoordDP n h).Opt s = ({s ⟨0, h⟩} : Set Bool) := by
  ext a
  constructor
  · intro hopt
    by_cases ha : a = s ⟨0, h⟩
    · simp [ha]
    · have hineq : (1 : ℝ) ≤ 0 := by
        have := hopt (s ⟨0, h⟩)
        simpa [firstCoordDP, ha] using this
      linarith
  · intro ha
    have haeq : a = s ⟨0, h⟩ := by simpa using ha
    intro a'
    by_cases ha' : a' = s ⟨0, h⟩
    · simp [firstCoordDP, haeq, ha']
    · simp [firstCoordDP, haeq, ha']

/-- Constant Boolean problem has empty-set sufficiency. -/
theorem constantBoolDP_empty_sufficient (n : ℕ) :
    (constantBoolDP n).isSufficient (∅ : Finset (Fin n)) := by
  apply (DecisionProblem.emptySet_sufficient_iff_constant (dp := constantBoolDP n)).2
  intro s s'
  rw [constantBoolDP_opt (n := n) (s := s), constantBoolDP_opt (n := n) (s := s')]

/-- First-coordinate Boolean problem fails empty-set sufficiency. -/
theorem firstCoordDP_empty_not_sufficient (n : ℕ) (h : 0 < n) :
    ¬ (firstCoordDP n h).isSufficient (∅ : Finset (Fin n)) := by
  apply (DecisionProblem.emptySet_not_sufficient_iff_exists_opt_difference (dp := firstCoordDP n h)).2
  let sTrue : Fin n → Bool := fun _ => true
  let sFlip : Fin n → Bool := fun j => if j = (⟨0, h⟩ : Fin n) then false else true
  refine ⟨sTrue, sFlip, ?_⟩
  rw [firstCoordDP_opt (n := n) (h := h) (s := sTrue), firstCoordDP_opt (n := n) (h := h) (s := sFlip)]
  simp [sTrue, sFlip]

/-- Hypercube node count (`2^n`) depends only on `n`, not on decision edges. -/
theorem boolHypercube_node_count (n : ℕ) :
    Fintype.card (Fin n → Bool) = 2 ^ n := by
  simp [Fintype.card_fin, Fintype.card_bool]

/-- Node count alone does not determine decision-edge geometry. -/
theorem node_count_does_not_determine_edge_geometry (n : ℕ) (h : 0 < n) :
    ∃ dp₁ dp₂ : DecisionProblem Bool (Fin n → Bool),
      (Fintype.card (Fin n → Bool) = 2 ^ n) ∧
      (¬ dp₁.edgeOnComplement (n := n) (∅ : Finset (Fin n))) ∧
      (dp₂.edgeOnComplement (n := n) (∅ : Finset (Fin n))) := by
  refine ⟨constantBoolDP n, firstCoordDP n h, boolHypercube_node_count n, ?_, ?_⟩
  · intro hEdge
    have hsuff : (constantBoolDP n).isSufficient (∅ : Finset (Fin n)) :=
      constantBoolDP_empty_sufficient n
    exact (DecisionProblem.edgeOnComplement_iff_not_sufficient
      (dp := constantBoolDP n) (I := (∅ : Finset (Fin n)))).1 hEdge hsuff
  · exact (DecisionProblem.edgeOnComplement_iff_not_sufficient
      (dp := firstCoordDP n h) (I := (∅ : Finset (Fin n)))).2
      (firstCoordDP_empty_not_sufficient n h)

end NodeEdgeSeparation

end DecisionQuotient
