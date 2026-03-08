/-
  Paper 4: Decision-Relevant Uncertainty

  Quotient.lean - The Universal Property (Theorem A)

  THEOREM A (Optimizer Factorization / Minimality):
  The decision quotient Q = S/∼ is the MINIMAL state abstraction that preserves Opt.
  Any other abstraction φ: S → T that preserves Opt must factor through π: S → Q.

  In the category Set, this is the coimage construction for the optimizer map
  `Opt : S → Set A`, canonically equivalent to the image/range of `Opt`.
  The artifact's novelty is not the existence of coimages in Set; it is the
  role this specific quotient plays in the later complexity and physical
  collapse arguments.

  ## Triviality Level
  NONTRIVIAL: This is Theorem A - the universal property of the quotient.
  This is a key mathematical result of the paper.

  ## Dependencies
  - Chain: Basic.lean (Opt definition) → here (proves universal property)
-/

import DecisionQuotient.Basic
import Mathlib.Logic.Function.Basic

namespace DecisionQuotient

open Classical

variable {A S : Type*}

/-! ## State Abstractions -/

/-- A state abstraction is a function φ: S → T for some type T -/
def StateAbstraction (S T : Type*) := S → T

/-- An abstraction φ preserves Opt if Opt factors through φ:
    s and s' map to same abstract state ⟹ Opt(s) = Opt(s') -/
def DecisionProblem.preservesOpt (dp : DecisionProblem A S) {T : Type*} (φ : S → T) : Prop :=
  ∀ s s' : S, φ s = φ s' → dp.Opt s = dp.Opt s'

/-! ## Theorem A: Universal Property of the Decision Quotient -/

/-- The quotient map preserves Opt (by construction) -/
theorem DecisionProblem.quotientMap_preservesOpt (dp : DecisionProblem A S) :
    dp.preservesOpt dp.quotientMap := by
  intro s s' h
  have : dp.DecisionEquiv s s' := Quotient.exact h
  exact this

/-!
  NOTE: The universal property requires a STRONGER condition.

  The correct statement is:

  If there exists F: T → Set A such that Opt = F ∘ φ, then φ factors through π.

  This is because "preserves Opt" in the sense of "Opt factors through φ"
  implies that φ respects the decision equivalence.
-/

/-- Correct version of preservesOpt: Opt factors through φ -/
def DecisionProblem.preservesOptStrong (dp : DecisionProblem A S) {T : Type*} (φ : S → T) : Prop :=
  ∃ F : T → Set A, ∀ s : S, dp.Opt s = F (φ s)

/-- The key lemma: if Opt = F ∘ φ, then φ(s) = φ(s') implies Opt(s) = Opt(s') -/
theorem DecisionProblem.factors_implies_respects (dp : DecisionProblem A S) {T : Type*}
    (φ : S → T) (F : T → Set A) (hF : ∀ s, dp.Opt s = F (φ s)) :
    ∀ s s' : S, φ s = φ s' → dp.Opt s = dp.Opt s' := by
  intro s s' hφ
  calc dp.Opt s = F (φ s) := hF s
    _ = F (φ s') := by rw [hφ]
    _ = dp.Opt s' := (hF s').symm

/-!
  The correct universal property is:

  Q is the COARSEST abstraction through which Opt factors.

  For any abstraction T with Opt = F ∘ φ, there exists ψ: T → Q (not Q → T!)
  such that π = ψ ∘ φ.
-/

/-- Q represents the equivalence relation induced by Opt -/
theorem DecisionProblem.quotient_represents_opt_equiv (dp : DecisionProblem A S) (s s' : S) :
    dp.quotientMap s = dp.quotientMap s' ↔ dp.Opt s = dp.Opt s' := by
  constructor
  · intro h
    exact Quotient.exact h
  · intro h
    exact Quotient.sound h

/-- Q is coarsest: if Opt factors through φ (with surjectivity), then π factors through φ -/
theorem DecisionProblem.quotient_is_coarsest (dp : DecisionProblem A S) {T : Type*}
    (φ : S → T) (hφ : ∀ s s', φ s = φ s' → dp.Opt s = dp.Opt s') (hSurj : Function.Surjective φ) :
    ∃ ψ : T → dp.DecisionQuotientType, ∀ s : S, dp.quotientMap s = ψ (φ s) := by
  -- For each t ∈ T, pick some s with φ s = t (using surjectivity), then map to π(s)
  choose inv hinv using hSurj
  use fun t => dp.quotientMap (inv t)
  intro s
  -- Need to show: dp.quotientMap s = dp.quotientMap (inv (φ s))
  -- We have: φ (inv (φ s)) = φ s by hinv
  have heq : φ (inv (φ s)) = φ s := hinv (φ s)
  -- By hφ: Opt (inv (φ s)) = Opt s
  have hopt : dp.Opt (inv (φ s)) = dp.Opt s := hφ _ _ heq
  -- By quotient_represents_opt_equiv: quotientMap agrees
  rw [dp.quotient_represents_opt_equiv]
  exact hopt.symm

/-- The factorization through the optimizer quotient is unique once the
abstraction map is surjective. -/
theorem DecisionProblem.quotient_factorization_unique (dp : DecisionProblem A S) {T : Type*}
    (φ : S → T) (hSurj : Function.Surjective φ)
    {ψ ψ' : T → dp.DecisionQuotientType}
    (hψ : ∀ s : S, dp.quotientMap s = ψ (φ s))
    (hψ' : ∀ s : S, dp.quotientMap s = ψ' (φ s)) :
    ψ = ψ' := by
  funext t
  rcases hSurj t with ⟨s, rfl⟩
  rw [← hψ s, ← hψ' s]

/-- In `Set`, the optimizer quotient satisfies the usual coimage universal
property: any surjective abstraction preserving `Opt` factors uniquely through
the quotient map. -/
theorem DecisionProblem.quotient_has_unique_factorization (dp : DecisionProblem A S) {T : Type*}
    (φ : S → T) (hφ : ∀ s s', φ s = φ s' → dp.Opt s = dp.Opt s') (hSurj : Function.Surjective φ) :
    ∃! ψ : T → dp.DecisionQuotientType, ∀ s : S, dp.quotientMap s = ψ (φ s) := by
  rcases dp.quotient_is_coarsest φ hφ hSurj with ⟨ψ, hψ⟩
  refine ⟨ψ, hψ, ?_⟩
  intro ψ' hψ'
  exact (dp.quotient_factorization_unique φ hSurj hψ hψ').symm

/-- Canonical map from the quotient object to the range of `Opt`.
In `Set`, this is the standard coimage-to-image comparison map. -/
noncomputable def DecisionProblem.quotientToOptRange (dp : DecisionProblem A S) :
    dp.DecisionQuotientType → ↥(Set.range dp.Opt) :=
  Quotient.lift
    (fun s => ⟨dp.Opt s, ⟨s, rfl⟩⟩)
    (fun _ _ h => Subtype.ext h)

/-- The canonical map from the quotient object to the range of `Opt` is
surjective. -/
theorem DecisionProblem.quotientToOptRange_surjective (dp : DecisionProblem A S) :
    Function.Surjective dp.quotientToOptRange := by
  intro y
  rcases Set.mem_range.mp y.property with ⟨s, hs⟩
  refine ⟨dp.quotientMap s, ?_⟩
  apply Subtype.ext
  simpa [DecisionProblem.quotientToOptRange] using hs

/-- The canonical map from the quotient object to the range of `Opt` is
injective. -/
theorem DecisionProblem.quotientToOptRange_injective (dp : DecisionProblem A S) :
    Function.Injective dp.quotientToOptRange := by
  intro q₁ q₂
  refine Quotient.inductionOn₂ q₁ q₂ ?_
  intro s s' hEq
  apply Quotient.sound
  exact congrArg Subtype.val hEq

/-- In `Set`, the decision quotient is canonically equivalent to the image/range
of the optimizer map. This is the familiar coimage-image identification for the
map `Opt : S → Set A`. -/
noncomputable def DecisionProblem.quotientEquivOptRange (dp : DecisionProblem A S) :
    dp.DecisionQuotientType ≃ ↥(Set.range dp.Opt) :=
  Equiv.ofBijective dp.quotientToOptRange
    ⟨dp.quotientToOptRange_injective, dp.quotientToOptRange_surjective⟩

/-- On representatives, the quotient-to-range equivalence sends `s` to the
value `Opt s` together with the obvious witness. -/
theorem DecisionProblem.quotientEquivOptRange_apply_quotientMap
    (dp : DecisionProblem A S) (s : S) :
    dp.quotientEquivOptRange (dp.quotientMap s) = ⟨dp.Opt s, ⟨s, rfl⟩⟩ := rfl

end DecisionQuotient
