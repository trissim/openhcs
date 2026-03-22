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
import Mathlib.Data.Fintype.Card
import Mathlib.Data.List.Basic

-- Reuse the SSOT / derivation and claim-closure machinery from the repo
import Ssot.Dof
import Ssot.Derivation
import Ssot.ClaimClosure

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
 -/

/-!
  Simple non-injectivity witnesses.

  We record a small, fully mechanized equivalence: `sem` fails to be
  injective exactly when there exist two distinct signatures with the
  same observable semantics. This is the finite/fiber witness used in
  the paper when applying counting / information arguments (Paper 1).
 -/

theorem noninjective_iff_exists_pair :
    (¬ Function.Injective sem) ↔ ∃ x y : Sig, x ≠ y ∧ sem x = sem y := by
  constructor
  · intro h
    -- ¬(∀ x y, sem x = sem y → x = y) so there exist x,y breaking it
    have : ¬(∀ x y, sem x = sem y → x = y) := h
    -- move to an explicit existential using classical reasoning
    by_contra contra
    apply this
    intro x y heq
    by_contra ne
    apply contra
    use x, y
    exact ⟨ne, heq⟩
  · intro ⟨x, y, hne, heq⟩ hinj
    apply hne
    apply hinj
    exact heq

 
/-!
  Fiber-card and factorization witnesses (paper-level corollaries).

  These are small bridge lemmas stated here so the paper text can point
  to mechanized handles. The proofs are trivial in this scaffold and are
  intended to be instantiated with DecisionQuotient / Derivation lemmas
  from the repository when doing a deeper mechanization pass.
/-

theorem fiber_card_gt_one_implies_noninjective {s : Sem}
    [Fintype (fiber sem s)] (hcard : 1 < Fintype.card (fiber sem s)) :
    ∃ x y : Sig, x ≠ y ∧ sem x = sem y := by
  -- obtain two distinct elements of the finite fiber (as a subtype)
  obtain ⟨a, b, ab_ne⟩ := Fintype.exists_pair_of_one_lt_card (fiber sem s) hcard
  -- coerce the subtype elements back to `Sig` and show they share the same semantics
  use (a : Sig), (b : Sig)
  refine ⟨fun h => ab_ne _, _⟩
  · -- if the coerced elements were equal as `Sig` then the subtype elements are equal
    have : (a : Sig) = (b : Sig) := h
    have : a = b := Subtype.ext this
    exact this
  · simp [Subtype.val_eq_coe] at a b
    -- both project to `s` by membership in the fiber, so their semantics coincide
    calc
      sem (a : Sig) = s := a.2
      _ = sem (b : Sig) := (b.2).symm

/-
  High-level witness: factoring repeated signatures into a single
  canonical interface and making the others *derived* collapses the
  independent core to size 1 (SSOT). The mechanized bridge below
  gives a small, checkable lemma that instantiates this pattern.
/-

/-!
  Bridge lemma: if a finite list of concrete signatures `enc_locs` is
  such that every element is either the chosen `source` or is derived
  from `source` under a `DerivationSystem` on `Sig`, then the derived
  encoding system (one `Dof.Encoding` per signature) has DOF = 1.

  Intuitively: making concrete signatures derived from a single
  canonical interface implements the Template Method payoff — the
  independent core collapses and the side-information burden vanishes.
/-

open Dof Ssot

/- Helper: lift a DerivationSystem on `Sig` to one on the Dof.Encoding
   records we use to compute DOF. -/
def liftDerivationOnEncodings {Fact Value : Type} (D : DerivationSystem Value) :
    DerivationSystem (Dof.Encoding Fact Value) where
  derived_from e1 e2 := D.derived_from e1.value e2.value
  transitive a b c h1 h2 := D.transitive a.value b.value c.value h1 h2
  irrefl a h := D.irrefl a.value h

theorem all_locations_derived_from_source_implies_dof_one
    {Dsig : DerivationSystem Sig} {s : Sem}
    (enc_locs : List Sig) (source : Sig)
    (hsource : source ∈ enc_locs)
    (hall : ∀ x ∈ enc_locs, x = source ∨ Dsig.derived_from source x) :
    Dof.dof (liftDerivationOnEncodings (D := Dsig))
      (enc_locs.map fun σ => { Dof.Encoding.fact := s, location := "", value := σ }) = 1 := by
  -- abbreviations
  let encs := enc_locs.map fun σ => { Dof.Encoding.fact := s, location := "", value := σ }
  let enc_src := { Dof.Encoding.fact := s, location := "", value := source }
  -- show `enc_src ∈ encs`
  have hmem_src : enc_src ∈ encs := by
    apply List.mem_map.mpr
    use source
    constructor; · exact hsource
    rfl
  -- show every element of `encs` is either the source encoding or redundant
  have h_all : ∀ e, e ∈ encs → (e = enc_src) ∨ (∃ e' ∈ encs, e' ≠ e ∧
    (liftDerivationOnEncodings (D := Dsig)).derived_from e' e) := by
    intro e he
    rcases List.mem_map.mp he with ⟨x, hx, heq⟩
    -- `heq : {.. value := x} = e` (so e.value = x)
    cases (hall x hx) with heq_x hder
    · -- x = source → e = enc_src
      have : e = enc_src := by
        -- e = f x and f x = f source because x = source
        have fx : e = { Dof.Encoding.fact := s, location := "", value := x } := heq.symm
        subst heq_x
        exact fx
      left; exact this
    · -- source derives x, so `enc_src` witnesses redundancy for `e`
      right
      refine ⟨enc_src, hmem_src, ?_, ?_⟩
      · -- enc_src ≠ e because otherwise we'd have Dsig.derived_from source source
        intro h
        -- take values of the equal encodings to contradict irrefl
        have hv : source = x := by
          have : (enc_src : Dof.Encoding s Sig).value = (e : Dof.Encoding s Sig).value := by
            congr
          -- `enc_src.value = source` and `e.value = x`
          simp_all [Dof.Encoding.value] at this
          exact this
        have : Dsig.derived_from source source := by
          rw [← hv]
          exact hder
        exact Dsig.irrefl source this
      · -- derivation witness lifts to the encoding level
        dsimp [liftDerivationOnEncodings]
        exact hder
  -- minimalIndependentCore filters out exactly the redundant elements,
  -- leaving only `enc_src` (hence DOF = 1)
  have : Dof.minimalIndependentCore (liftDerivationOnEncodings (D := Dsig)) encs = [enc_src] := by
    -- show both inclusions / equality of lists
    apply List.eq_of_mem_map_eq_singleton
    · -- every element of the filter is `enc_src`
      intro e he
      simp [Dof.minimalIndependentCore, Dof.redundant] at he
      -- `he` is `decide (¬ redundant ...) = true` hence `¬ redundant ...` holds
      have hnred : ¬ (∃ e' ∈ encs, e' ≠ e ∧ (liftDerivationOnEncodings (D := Dsig)).derived_from e' e) :=
        by simpa [decide_eq_true] using he
      -- but from `h_all` we know `(e = enc_src) ∨ redundant`, so it must be the former
      cases (h_all e (List.mem_of_mem_filter he)) with heq _;
      exact heq
    · -- `enc_src` is in the filter (not redundant)
      simp [Dof.minimalIndependentCore]
      refine List.mem_map.mpr ⟨source, hsource, ?_⟩
      simp [Dof.redundant]
      -- show `¬ ∃ e' ∈ encs, e' ≠ enc_src ∧ derived_from e' enc_src`
      intro h
      rcases h with ⟨e', he', hne, hder'⟩
      -- e' corresponds to some x in enc_locs, so `e'.value` ∈ enc_locs and is derived_from e'.value source
      have x := e'.value
      have hin : x ∈ enc_locs := by
        rcases List.mem_map.mp he' with ⟨y, hy, hyf⟩
        injection hyf with h
        exact h ▸ hy
      -- by `hall`, x = source ∨ Dsig.derived_from source x; can't be source because then e' = enc_src
      cases hall x hin with hx_eq hx_der
      · -- x = source leads to e' = enc_src contradicting hne
        have : e' = enc_src := by
          rcases List.mem_map.mp he' with ⟨y, hy, hyf⟩
          have : y = source := by injection hyf; exact hx_eq
          subst this
          rfl
        exact (hne this)
      · -- otherwise we have both `Dsig.derived_from source x` (hx_der) and `Dsig.derived_from x source` (from hder'), transitivity gives a contradiction with irrefl
        have : Dsig.derived_from source source := Dsig.transitive source x source hx_der (by simpa [liftDerivationOnEncodings] using hder')
        exact Dsig.irrefl source this
  -- conclude DOF = length of minimalIndependentCore = 1
  simp [Dof.dof, this]

/-
  Combining the previous lemma with the paper-level side-info handle
  shows the Template Method payoff: if all concrete encodings are the
  source or derived from the source then the side-information burden
  (measured in bits) collapses to zero.
/-

theorem factorization_removes_auxiliary_burden
    {Dsig : DerivationSystem Sig} {s : Sem}
    (enc_locs : List Sig) (source : Sig)
    (hsource : source ∈ enc_locs)
    (hall : ∀ x ∈ enc_locs, x = source ∨ Dsig.derived_from source x) :
    ClaimClosure.sideInfoBits
      (Dof.dof (liftDerivationOnEncodings (D := Dsig))
        (enc_locs.map fun σ => { Dof.Encoding.fact := s, location := "", value := σ })) = 0 := by
  have h := all_locations_derived_from_source_implies_dof_one enc_locs source hsource hall
  -- rewrite the DOF to `1` and use the closure lemma `dof1_zero_side_information`
  rw [h]
  exact ClaimClosure.dof1_zero_side_information

/-!
  Small arithmetic corollary: a nontrivial fiber has positive side-information
  requirement under the DecisionQuotient side-info proxy. This is the finite
  counting lower bound of Paper 1 instantiated on the fiber cardinality.
/-

theorem fiber_card_implies_positive_sideinfo {s : Sem} [Fintype (fiber sem s)]
    (hcard : 1 < Fintype.card (fiber sem s)) :
    ClaimClosure.sideInfoBits (Fintype.card (fiber sem s)) > 0 := by
  -- sideInfoBits k = log k / log 2; if k > 1 then numerator > 0
  have hkR : (1 : ℝ) < (Fintype.card (fiber sem s) : ℝ) := by exact_mod_cast hcard
  have hlog_pos : 0 < Real.log (Fintype.card (fiber sem s) : ℝ) := Real.log_pos hkR
  have hlog2_pos : 0 < Real.log 2 := Real.log_pos (by norm_num : (1 : ℝ) < 2)
  unfold ClaimClosure.sideInfoBits
  apply (div_pos hlog_pos hlog2_pos)


end SignatureFactorization
