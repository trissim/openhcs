import Ssot.Derivation

/-!
  DOF Foundation - Minimal Semantics for Paper 2

  This file replaces the trivial "dof = Nat" formulation with a minimal,
  semantics-backed notion of degrees of freedom. It connects directly to the
  derivation relation from Paper 1 (via `DerivationSystem`).

  The intent is lightweight: no full semantics, just enough structure to make
  the SSOT claims meaningful.
-/

open Classical Ssot

namespace Dof

/-!
## Encodings

An encoding is a concrete location that stores a value for a fact.
Two encodings are independent unless one is derivable from the other.
-/

structure Encoding (Fact Value : Type) where
  fact : Fact
  location : String
  value : Value

/-!
## Derivability and Independence

`Derives e1 e2` means e2 is derived from e1 (changing e1 forces e2 to change).
This is parameterized by a `DerivationSystem` on encodings.
-/

def Derives {F V : Type} (D : DerivationSystem (Encoding F V))
    (e1 e2 : Encoding F V) : Prop :=
  D.derived_from e1 e2

def Independent {F V : Type} (D : DerivationSystem (Encoding F V))
    (e1 e2 : Encoding F V) : Prop :=
  ¬Derives D e1 e2 ∧ ¬Derives D e2 e1

/-!
## Minimal Independent Core

We conservatively model the "independent core" as the subset of encodings
that are not derivable from any other encoding in the list.
-/

def redundant {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) (e : Encoding F V) : Prop :=
  ∃ e' ∈ encodings, e' ≠ e ∧ Derives D e' e

noncomputable def minimalIndependentCore {F V : Type}
    (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) : List (Encoding F V) :=
  encodings.filter (fun e => decide (¬ redundant D encodings e))

theorem core_subset {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) :
    minimalIndependentCore D encodings ⊆ encodings := by
  intro e h
  exact (List.mem_filter.mp h).left

/-!
## DOF and SSOT

DOF is the size of the minimal independent core.
SSOT holds iff DOF = 1.
-/

noncomputable def dof {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) : Nat :=
  (minimalIndependentCore D encodings).length

def SSOT {F V : Type} (D : DerivationSystem (Encoding F V))
    (encodings : List (Encoding F V)) : Prop :=
  dof D encodings = 1

end Dof

/-!
  Useful monotonicity and helper lemmas for DOF reasoning.
  These are small, local facts used by higher-level paper lemmas.
/-

namespace Dof

variable {F V : Type}

/- Helper: generic filter sublist when predicate implication holds.
   Copied/adapted from the axis-framework helper used elsewhere. -/
lemma filter_sublist_of_imp {l : List (Encoding F V)} {p q : (Encoding F V) → Bool}
    (h : ∀ x, p x → q x) : List.Sublist (l.filter p) (l.filter q) := by
  induction l with
  | nil => exact List.Sublist.refl []
  | cons x xs ih =>
    simp only [List.filter_cons]
    by_cases hp : p x
    · have hq : q x = true := h x hp
      simp only [hp, hq, ↓reduceIte]
      exact ih.cons₂ x
    · by_cases hq : q x
      · simp only [hp, hq, Bool.false_eq_true, ↓reduceIte]
        exact ih.cons x
      · simp only [hp, hq, Bool.false_eq_true, ↓reduceIte]
        exact ih

/- DOF monotonicity under refinement of the derivation relation.

   If every derivation in `D1` also holds in `D2` (i.e. `D2` is a refinement
   with possibly more derived edges), then redundancy can only increase and the
   minimal independent core can only shrink; hence DOF is nonincreasing.
-/
theorem dof_monotone_of_derivation_refinement {D1 D2 : DerivationSystem (Encoding F V)}
    (href : ∀ a b, D1.derived_from a b → D2.derived_from a b)
    (encodings : List (Encoding F V)) :
    dof D2 encodings ≤ dof D1 encodings := by
  -- abbreviations for the two filter predicates
  let p1 := fun e => decide (¬ redundant D1 encodings e)
  let p2 := fun e => decide (¬ redundant D2 encodings e)
  -- show pointwise implication p2 -> p1
  have himp : ∀ e, p2 e → p1 e := by
    intro e he2
    -- `he2` gives ¬ (∃ e' ∈ encodings, e' ≠ e ∧ D2.derived_from e' e)
    have hnred2 : ¬ (∃ e' ∈ encodings, e' ≠ e ∧ D2.derived_from e' e) := by
      simpa [decide_eq_true] using he2
    -- if `e` were redundant under D1 we'd get a contradiction via `href`
    by_contra hred1
    rcases hred1 with ⟨e', he', hne, hder1⟩
    have hder2 := href e' e hder1
    exact hnred2 ⟨e', he', hne, hder2⟩
  -- now the minimal independent cores are filters with pointwise implication
  have hsub : List.Sublist (encodings.filter p2) (encodings.filter p1) :=
    filter_sublist_of_imp (l:=encodings) (p:=p2) (q:=p1) himp
  -- lengths respect the sublist ordering
  have hlen := hsub.length_le
  simp [dof] at hlen
  exact hlen

/- Equality of filters when predicates agree on list elements. -/
lemma filter_eq_of_forall {l : List (Encoding F V)} {p q : (Encoding F V) → Bool}
    (h : ∀ x, x ∈ l → p x = q x) : l.filter p = l.filter q := by
  induction l with
  | nil => rfl
  | cons x xs ih =>
    simp only [List.filter_cons]
    have heq : p x = q x := h x (by simp)
    by_cases hp : p x
    · -- p x = true => q x = true as well
      have hq : q x = true := by simpa [heq] using hp
      simp [hp, hq]
      congr
      apply ih
      intro y hy
      exact h y (by simp [hy])
    · by_cases hq : q x
      · -- q x = true but p x ≠ true leads to contradiction via heq
        have : p x = true := by simpa [heq.symm] using hq
        simp [this] at hp
      · -- both false
        simp [hp, hq]
        congr
        apply ih
        intro y hy
        exact h y (by simp [hy])

/- DOF additivity across disjoint facts.

If derivation never links encodings of different `fact` values, then DOF
is additive across lists whose facts are pairwise disjoint.
-/
theorem dof_additive_on_disjoint_facts {D : DerivationSystem (Encoding F V)}
    (encs1 encs2 : List (Encoding F V))
    (hdisjoint : ∀ e1 ∈ encs1, ∀ e2 ∈ encs2, e1.fact ≠ e2.fact)
    (hrespect : ∀ a b, D.derived_from a b → a.fact = b.fact) :
    dof D (encs1 ++ encs2) = dof D encs1 + dof D encs2 := by
  -- predicates used for minimalIndependentCore
  let p_concat := fun e => decide (¬ redundant D (encs1 ++ encs2) e)
  let p1 := fun e => decide (¬ redundant D encs1 e)
  let p2 := fun e => decide (¬ redundant D encs2 e)
  -- show predicates agree on elements of encs1 and encs2 respectively
  have h1 : ∀ e ∈ encs1, p_concat e = p1 e := by
    intro e he
    -- show redundant w.r.t concatenation iff redundant w.r.t encs1
    have : (¬ ∃ e' ∈ (encs1 ++ encs2), e' ≠ e ∧ D.derived_from e' e) ↔ (¬ ∃ e' ∈ encs1, e' ≠ e ∧ D.derived_from e' e) := by
      constructor
      · intro h
        by_contra h2
        rcases h2 with ⟨e', he', hne, hder⟩
        -- e' ∈ encs1 yields witness in concat
        refine h ⟨e', List.mem_append.mpr (Or.inl he'), hne, hder⟩
      · intro h
        by_contra h2
        rcases h2 with ⟨e', he', hne, hder⟩
        rcases List.mem_append.mp he' with he1 | he2
        · -- witness in encs1 already shows redundancy in encs1
          exact h ⟨e', he1, hne, hder⟩
        · -- witness in encs2 impossible because facts are disjoint
          have hf := hrespect e' e hder
          have contra := hdisjoint e (List.mem_of_mem_append.mpr (Or.inr he2)) e' (by simpa using he2)
          -- simplify contradictions by rewriting facts equality; derive False
          have : e.fact = e'.fact := hf.symm
          have neq := contra
          contradiction
    -- now convert the propositional equivalence to boolean equality via `decide`
    simp [p_concat, p1]
    have : (¬ ∃ e' ∈ (encs1 ++ encs2), e' ≠ e ∧ D.derived_from e' e) ↔ (¬ ∃ e' ∈ encs1, e' ≠ e ∧ D.derived_from e' e) := this
    -- `decide` respects logical equivalence on finite decidable propositions
    by_cases hdec : decide ((¬ ∃ e' ∈ (encs1 ++ encs2), e' ≠ e ∧ D.derived_from e' e))
    · have htrue : decide ((¬ ∃ e' ∈ encs1, e' ≠ e ∧ D.derived_from e' e)) := by
        simp [← this] at hdec
        exact hdec
      simp [hdec, htrue]
    · by_cases hdec2 : decide ((¬ ∃ e' ∈ encs1, e' ≠ e ∧ D.derived_from e' e))
      · have : decide ((¬ ∃ e' ∈ (encs1 ++ encs2), e' ≠ e ∧ D.derived_from e' e)) := by
          simp [this] at hdec2
          exact hdec2
        simp [this, hdec2]
      · simp [hdec, hdec2]
  -- symmetric argument for encs2
  have h2 : ∀ e ∈ encs2, p_concat e = p2 e := by
    intro e he
    have : (¬ ∃ e' ∈ (encs1 ++ encs2), e' ≠ e ∧ D.derived_from e' e) ↔ (¬ ∃ e' ∈ encs2, e' ≠ e ∧ D.derived_from e' e) := by
      constructor
      · intro h
        by_contra h2
        rcases h2 with ⟨e', he', hne, hder⟩
        refine h ⟨e', List.mem_append.mpr (Or.inr he'), hne, hder⟩
      · intro h
        by_contra h2
        rcases h2 with ⟨e', he', hne, hder⟩
        rcases List.mem_append.mp he' with he1 | he2
        · -- witness in encs1 impossible due to disjoint facts
          have hf := hrespect e' e hder
          have contra := hdisjoint e' (by simpa [he1] using he1) e (List.mem_of_mem_append.mpr (Or.inr he))
          have : e'.fact = e.fact := hf
          contradiction
        · exact h ⟨e', he2, hne, hder⟩
    simp [p_concat, p2]
    have : (¬ ∃ e' ∈ (encs1 ++ encs2), e' ≠ e ∧ D.derived_from e' e) ↔ (¬ ∃ e' ∈ encs2, e' ≠ e ∧ D.derived_from e' e) := this
    by_cases hdec : decide ((¬ ∃ e' ∈ (encs1 ++ encs2), e' ≠ e ∧ D.derived_from e' e))
    · have htrue : decide ((¬ ∃ e' ∈ encs2, e' ≠ e ∧ D.derived_from e' e)) := by simp [← this] at hdec; exact hdec
      simp [hdec, htrue]
    · by_cases hdec2 : decide ((¬ ∃ e' ∈ encs2, e' ≠ e ∧ D.derived_from e' e))
      · have : decide ((¬ ∃ e' ∈ (encs1 ++ encs2), e' ≠ e ∧ D.derived_from e' e)) := by simp [this] at hdec2; exact hdec2
        simp [this, hdec2]
      · simp [hdec, hdec2]
  -- now use filter equality helper to rewrite filters
  have hfilter1 := filter_eq_of_forall (l:=encs1) (p:=p_concat) (q:=p1) fun x hx => (h1 x hx)
  have hfilter2 := filter_eq_of_forall (l:=encs2) (p:=p_concat) (q:=p2) fun x hx => (h2 x hx)
  -- split the concatenation filter and use equalities
  calc
    (minimalIndependentCore D (encs1 ++ encs2)).length
        = ((encs1 ++ encs2).filter p_concat).length := rfl
    _ = (encs1.filter p_concat ++ encs2.filter p_concat).length := by simp [List.filter_append]
    _ = (encs1.filter p1 ++ encs2.filter p2).length := by rw [hfilter1, hfilter2]
    _ = (minimalIndependentCore D encs1).length + (minimalIndependentCore D encs2).length := by simp [minimalIndependentCore]
    _ = dof D encs1 + dof D encs2 := rfl

end Dof
