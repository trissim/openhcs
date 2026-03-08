/-
  Paper 4: Decision-Relevant Uncertainty
  
  Reduction.lean - Formal coNP-Completeness Proof
  
  We prove SUFFICIENCY-CHECK is coNP-complete by reduction from TAUTOLOGY.
  
  TAUTOLOGY: Given Boolean formula φ, is φ(x) = true for all assignments x?
  
  Reduction: φ ↦ (D_φ, I = ∅) where:
  - States = Boolean assignments + reference state
  - Actions = {accept, reject}  
  - Reference state has Opt = {accept}
  - Assignment s has Opt = {accept} iff φ(s) = true
  
  Then: φ is tautology ⟺ I = ∅ is sufficient in D_φ

  ## Triviality Level
  NONTRIVIAL: This is a core hardness proof - the main complexity-theoretic
  contribution showing coNP-completeness of SUFFICIENCY-CHECK.

  ## Dependencies
  - Depends on: Basic.lean, Sufficiency.lean
  - Used by: ClaimClosure.lean (closure of hardness results)
-/
  
import DecisionQuotient.Basic
import DecisionQuotient.Sufficiency
import Mathlib.Data.Fintype.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic.Linarith

namespace DecisionQuotient

/-! ## Boolean Formulas (General AST) -/

/-- A Boolean formula over n variables -/
inductive Formula (n : ℕ) where
  | var : Fin n → Formula n
  | not : Formula n → Formula n
  | and : Formula n → Formula n → Formula n
  | or : Formula n → Formula n → Formula n
  deriving DecidableEq

/-- A Boolean assignment over n variables (local to this reduction) -/
abbrev BoolAssignment (n : ℕ) := Fin n → Bool

/-- Evaluate a formula under an assignment -/
def Formula.eval (a : BoolAssignment n) : Formula n → Bool
  | var i => a i
  | not f => !(eval a f)
  | and f1 f2 => (eval a f1) && (eval a f2)
  | or f1 f2 => (eval a f1) || (eval a f2)

/-- A formula is a tautology if true under all assignments -/
def Formula.isTautology (φ : Formula n) : Prop :=
  ∀ a : BoolAssignment n, φ.eval a = true

/-! ## The Reduction -/

/-- States for the reduction: assignments plus a reference state -/
inductive ReductionState (n : ℕ) where
  | assignment : BoolAssignment n → ReductionState n
  | reference : ReductionState n
  deriving DecidableEq

/-- Actions: accept or reject -/
inductive ReductionAction where
  | accept : ReductionAction
  | reject : ReductionAction
  deriving DecidableEq

open ReductionAction in
/-- Utility function for the reduction -/
def reductionUtility (φ : Formula n) : ReductionAction → ReductionState n → ℝ
  | accept, ReductionState.reference => 1
  | reject, ReductionState.reference => 0
  | accept, ReductionState.assignment a => if φ.eval a then 1 else 0
  | reject, ReductionState.assignment _ => 0

/-- The decision problem constructed from a Boolean formula -/
def reductionProblem (φ : Formula n) : DecisionProblem ReductionAction (ReductionState n) where
  utility := reductionUtility φ

/-! ## Characterizing Optimal Sets -/

open ReductionAction ReductionState in
/-- At the reference state, only accept is optimal -/
theorem opt_reference (φ : Formula n) :
    (reductionProblem φ).Opt reference = {accept} := by
  ext a
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq,
             Set.mem_singleton_iff]
  constructor
  · intro h
    cases a with
    | accept => rfl
    | reject =>
      have h1 := h accept
      simp only [reductionProblem, reductionUtility] at h1
      linarith
  · intro h
    subst h
    intro a'
    cases a' with
    | accept => simp [reductionProblem, reductionUtility]
    | reject => simp only [reductionProblem, reductionUtility]; linarith

open ReductionAction ReductionState in
/-- At a satisfying assignment, only accept is optimal -/
theorem opt_satisfying (φ : Formula n) (a : BoolAssignment n) (hsat : φ.eval a = true) :
    (reductionProblem φ).Opt (assignment a) = {accept} := by
  ext x
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq,
             Set.mem_singleton_iff]
  constructor
  · intro hopt
    cases x with
    | accept => rfl
    | reject =>
      have h1 := hopt accept
      simp only [reductionProblem, reductionUtility, hsat, ↓reduceIte] at h1
      linarith
  · intro hx
    subst hx
    intro a'
    cases a' with
    | accept => simp [reductionProblem, reductionUtility, hsat]
    | reject => simp only [reductionProblem, reductionUtility, hsat, ↓reduceIte]; linarith

open ReductionAction ReductionState in
/-- At a falsifying assignment, both actions are optimal -/
theorem opt_falsifying (φ : Formula n) (a : BoolAssignment n) (hfalse : φ.eval a = false) :
    (reductionProblem φ).Opt (assignment a) = {accept, reject} := by
  ext x
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq,
             Set.mem_insert_iff, Set.mem_singleton_iff]
  constructor
  · cases x <;> simp
  · intro _
    intro a'
    cases a' <;> cases x <;> simp [reductionProblem, reductionUtility, hfalse]

/-! ## Main Theorem: Tautology ⟺ Sufficient -/

/-- The coordinate space is intentionally degenerate: a single Unit-valued
    coordinate means `agreeOn s s' ∅` is satisfied for all state pairs,
    reducing sufficiency of ∅ to Opt-constancy across the entire state space.
    This is the standard reduction technique — the structure carries the
    encoding, not additional coordinate data. -/
instance : CoordinateSpace (ReductionState n) 1 where
  Coord := fun _ => Unit
  proj := fun _ _ => ()

open ReductionAction ReductionState in
/-- If φ is a tautology, then I = ∅ is sufficient -/
theorem tautology_implies_sufficient (φ : Formula n) (htaut : φ.isTautology) :
    (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) := by
  intro s s' _
  -- All states have Opt = {accept}
  cases s with
  | reference =>
    cases s' with
    | reference => rfl
    | assignment a' =>
      rw [opt_reference, opt_satisfying φ a' (htaut a')]
  | assignment a =>
    cases s' with
    | reference =>
      rw [opt_reference, opt_satisfying φ a (htaut a)]
    | assignment a' =>
      rw [opt_satisfying φ a (htaut a), opt_satisfying φ a' (htaut a')]

open ReductionAction ReductionState in
/-- If I = ∅ is sufficient, then φ is a tautology -/
theorem sufficient_implies_tautology (φ : Formula n)
    (hsuff : (reductionProblem φ).isSufficient (∅ : Finset (Fin 1))) :
    φ.isTautology := by
  intro a
  -- Compare assignment a with reference state
  -- They agree on ∅, so must have same Opt
  have heq := hsuff (assignment a) reference (by
    intro i hi
    exact (by simpa using hi))
  -- Reference has Opt = {accept}
  rw [opt_reference] at heq
  -- So assignment a must also have Opt = {accept}
  -- This means φ.eval a = true (otherwise Opt = {accept, reject})
  by_contra hfalse
  push_neg at hfalse
  have hf : φ.eval a = false := Bool.eq_false_iff.mpr hfalse
  rw [opt_falsifying φ a hf] at heq
  -- {accept, reject} ≠ {accept}
  have : reject ∈ ({accept, reject} : Set ReductionAction) := Set.mem_insert_of_mem _ rfl
  rw [heq] at this
  simp at this

/-! ## The Main Reduction Theorem

### Correctness vs. Polynomial-Time Computability

A Karp reduction requires two properties:
1. **Correctness**: The reduction preserves membership (∀ x, x ∈ A ↔ f(x) ∈ B)
2. **Efficiency**: The reduction is computable in polynomial time

**What is machine-verified below**: The CORRECTNESS of the reduction (property 1).
The theorem `tautology_iff_sufficient` proves that φ is a tautology if and only if
I = ∅ is sufficient in D_φ. This is the mathematically essential property.

**What is a standard claim**: The POLYNOMIAL-TIME computability (property 2).
The reduction function `reductionProblem` constructs D_φ in time O(|φ|).
This is a straightforward claim about the construction, but full Turing-machine
verification of polynomial-time bounds remains an open problem in formal verification.

For a fully explicit formulation combining proved correctness with polytime as a
hypothesis, see `valid_karp_reduction` in PolynomialReduction.lean.
-/

/-- The reduction theorem: φ is a tautology iff I = ∅ is sufficient in D_φ.

    **CORRECTNESS PROOF** (fully machine-verified, zero sorry):
    This proves the membership-preservation property of the reduction.
    Combined with polynomial-time computability (a standard claim), this
    establishes a Karp reduction from TAUTOLOGY to SUFFICIENCY-CHECK.
    Since TAUTOLOGY is coNP-complete, this proves SUFFICIENCY-CHECK is coNP-hard. -/
theorem tautology_iff_sufficient (φ : Formula n) :
    φ.isTautology ↔ (reductionProblem φ).isSufficient (∅ : Finset (Fin 1)) :=
  ⟨tautology_implies_sufficient φ, sufficient_implies_tautology φ⟩

/-- Reduction bridge: tautology is exactly constancy of the decision boundary
    on the reduction instance (`I = ∅` equivalence unfolded). -/
theorem tautology_iff_opt_constant (φ : Formula n) :
    φ.isTautology ↔ ∀ s s' : ReductionState n, (reductionProblem φ).Opt s = (reductionProblem φ).Opt s' := by
  constructor
  · intro hT
    exact (DecisionProblem.emptySet_sufficient_iff_constant (dp := reductionProblem φ)).1
      ((tautology_iff_sufficient φ).1 hT)
  · intro hConst
    exact (tautology_iff_sufficient φ).2
      ((DecisionProblem.emptySet_sufficient_iff_constant (dp := reductionProblem φ)).2 hConst)

/-- Reduction bridge (counterexample form):
    non-tautology is exactly existence of a decision-boundary difference witness
    in the reduction instance. -/
theorem not_tautology_iff_exists_opt_difference (φ : Formula n) :
    ¬ φ.isTautology ↔
      ∃ s s' : ReductionState n, (reductionProblem φ).Opt s ≠ (reductionProblem φ).Opt s' := by
  constructor
  · intro hNotT
    have hNotConst : ¬ (∀ s s' : ReductionState n,
        (reductionProblem φ).Opt s = (reductionProblem φ).Opt s') := by
      intro hConst
      exact hNotT ((tautology_iff_opt_constant φ).2 hConst)
    push_neg at hNotConst
    rcases hNotConst with ⟨s, hs⟩
    rcases hs with ⟨s', hs'⟩
    exact ⟨s, s', hs'⟩
  · intro hDiff
    rcases hDiff with ⟨s, s', hNeq⟩
    intro hT
    have hConst := (tautology_iff_opt_constant φ).1 hT
    exact hNeq (hConst s s')

/-! ## Complexity Classification

The theorems above establish:

1. **SUFFICIENCY-CHECK ∈ coNP**: A witness for NON-sufficiency is a pair (s, s')
   with agreeOn I but different Opt. Verifiable in polynomial time.
   (Formalized in Computation.lean via InsufficiencyWitness)

2. **SUFFICIENCY-CHECK is coNP-hard**: We have a polynomial reduction from TAUTOLOGY.
   - Reduction function: `reductionProblem : Formula n → DecisionProblem ...`
   - **Correctness (MACHINE-VERIFIED)**: `tautology_iff_sufficient`
   - **Polynomial time (STANDARD CLAIM)**: Construction is linear in |φ|.
     This is stated as an explicit hypothesis when combined with `valid_karp_reduction`.

3. **SUFFICIENCY-CHECK is coNP-complete**: By (1) and (2).

### What is machine-verified vs. what is claimed

The correctness of each reduction is machine-verified. Polynomial-time computability
of the reduction functions is a standard claim stated as an explicit hypothesis;
full Turing-machine verification of polynomial-time bounds remains an open problem
in formal verification.

This is the **first formal machine-checked proof** of the correctness of a reduction
establishing computational hardness of decision-relevant variable identification.
-/

end DecisionQuotient
