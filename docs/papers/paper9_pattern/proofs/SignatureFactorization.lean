/-
  Signature factorization lemma linking Paper1 (information barrier)
  and Paper2 (derivation) to the software-design rule: identical
  parameter sequences should be factored into a single abstract
  interface to achieve zero-error coherence.
-/ 
/-
  Signature factorization: basic mechanization tying the Paper 1
  information-barrier viewpoint to a simple code-signature factoring
  statement.  This file intentionally keeps the statements minimal and
  directly mechanizable: it shows that injectivity of the semantics map
  is equivalent to fiber-level subsingleness, and it records the usual
  factorization corollary used in the text.
-/

import Mathlib.Tactic
import Mathlib.Data.Set.Basic

namespace SignatureFactorization

variable {Sig Sem : Type*}

/-! ## Representation fibers -/

/-- Observable semantics projection: a function from declared signatures
    to their observable semantics. In the paper this is written
    \(\pi:\mathrm{Sig}\to\mathrm{Sem}\). -/
variable (sem : Sig → Sem)

/-- The fiber of a semantic value: all signatures projecting to `s`. -/
def fiber (s : Sem) : Set Sig := { σ | sem σ = s }

/-- The information-barrier style equivalence: `sem` is injective iff
    every semantic fiber is a subsingleton (has at most one element). -/
theorem injective_iff_all_fibers_subsingleton :
    Function.Injective sem ↔ ∀ s, Set.Subsingleton (fiber sem s) := by
  constructor
  · intro h s
    -- if sem is injective then any two elements of the same fiber are equal
    intro x hx y hy
    have : sem x = sem y := by simpa [fiber] using hx.trans hy.symm
    exact h this
  · intro h x y hxy
    -- if every fiber is subsingleton then two elements with equal image
    -- must be equal (they lie in the same fiber).
    have H : Set.Subsingleton (fiber sem (sem x)) := h (sem x)
    have hx : x ∈ fiber sem (sem x) := by simp [fiber]
    have hy : y ∈ fiber sem (sem x) := by simp [fiber, hxy]
    exact H hx hy

/-- Factorization corollary (interface/ABC view): if there exists a
    canonical interface assignment `choose : Sem → Interface` and every
    signature `σ` implements `choose (sem σ)` then all signatures in a
    semantic fiber are implementations of the same interface. This is
    the mechanized form of the refactoring suggestion in the text. -/
variable {Interface : Type*} (choose : Sem → Interface)
variable (implements : Sig → Interface → Prop)

theorem factorization_respects_interface
    (himpl : ∀ σ, implements σ (choose (sem σ)))
    (s : Sem) (σ : Sig) (hσ : σ ∈ fiber sem s) :
    implements σ (choose s) := by
  -- `hσ` is `sem σ = s`; rewrite the implementation witness accordingly
  have h := himpl σ
  rw [Set.mem_def, fiber] at hσ
  rw [hσ] at h
  exact h

/-!
  Note: the paper-level derivation/SSOT lemma (Paper 2) interfaces with this
  statement as follows: making every signature in a fiber `derived` from a
  single declared interface enforces coherence (derivation_excludes_DO F),
  which removes the auxiliary information burden quantified by Paper 1.
  The mechanized ambient already contains `Derivation` and information-barrier
  lemmas which can be instantiated against the concrete `choose`/`implements`
  objects used in a real codebase.
-/

end SignatureFactorization
