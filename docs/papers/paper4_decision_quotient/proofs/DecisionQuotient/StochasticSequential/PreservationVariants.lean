/-
  Paper 4: Minimum and Anchor Variants of Stochastic Preservation

  This file completes the bifurcation picture for the stochastic regime:
  - Preservation: does coarse conditioning preserve the full-information optimizer?
  - Decisiveness: does each fiber admit a unique conditional optimum?

  The decisiveness variants (base, minimum, anchor) already exist in Basic.lean.
  This file adds the minimum and anchor variants for preservation.

  Key results (under full support):
  - Minimum stochastic preservation: coNP-complete (inherits from static)
  - Anchor stochastic preservation: Σ₂ᴾ-complete (inherits from static)
-/

import DecisionQuotient.StochasticSequential.Basic
import DecisionQuotient.StochasticSequential.SetValued
import DecisionQuotient.StochasticSequential.Computation
import DecisionQuotient.Hardness.Sigma2PHardness
import DecisionQuotient.Hardness.Sigma2PExhaustive.AnchorSufficiency
import DecisionQuotient.ClaimClosure

namespace DecisionQuotient.StochasticSequential

open DecisionQuotient
open DecisionQuotient.ClaimClosure
open Classical

/-! ## Minimum Stochastic Preservation -/

/-- Minimum stochastic preservation: does there exist a coordinate set of size ≤ k
    such that the conditional fiber optimizer agrees with the full-information optimizer
    at every state?

    This is the preservation analogue of MINIMUM-SUFFICIENT-SET for stochastic problems.
    Under full support, it inherits coNP-completeness from the static regime. -/
def StochasticMinimumPreservation
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) : Prop :=
  ∃ I : Finset (Fin n), I.card ≤ k ∧ StochasticPreservationSufficient P I

/-- Decision problem form for complexity classification -/
def StochasticMinimumPreservationCheck
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) : Prop :=
  StochasticMinimumPreservation P k

/-! ## Anchor Stochastic Preservation -/

/-- Anchor stochastic preservation: does there exist an anchor state s₀ such that
    the conditional fiber optimizer through s₀ agrees with the full-information optimizer
    at every state in the I-fiber of s₀?

    This is the preservation analogue of ANCHOR-SUFFICIENCY for stochastic problems.
    Under full support, it inherits Σ₂ᴾ-completeness from the static regime. -/
def StochasticAnchorPreservation
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  ∃ s₀ : S, ∀ s : S, agreeOn s s₀ I → fiberOpt P I s = P.toDecisionProblem.Opt s

/-- Decision problem form for complexity classification -/
def StochasticAnchorPreservationCheck
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  StochasticAnchorPreservation P I

/-! ## Unconditional Obstruction Theorems -/

/-- Any stochastically preserving coordinate set must contain every static
    relevant coordinate of the underlying decision problem. This gives an
    unconditional necessary condition for preservation, even without full
    support. -/
theorem stochastic_preservation_contains_static_relevant
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hpres : StochasticPreservationSufficient P I) (i : Fin n)
    (hrel : P.toDecisionProblem.isRelevant i) :
    i ∈ I := by
  exact P.toDecisionProblem.sufficient_contains_relevant I
    (stochastic_preservation_implies_static_sufficiency P I hpres) i hrel

/-- Contrapositive form of `stochastic_preservation_contains_static_relevant`. If
    a static relevant coordinate is missing, preservation fails. -/
theorem not_stochastic_preservation_of_missing_static_relevant
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) (i : Fin n)
    (hrel : P.toDecisionProblem.isRelevant i) (hmiss : i ∉ I) :
    ¬ StochasticPreservationSufficient P I := by
  intro hpres
  exact hmiss (stochastic_preservation_contains_static_relevant P I hpres i hrel)

/-- On Boolean product spaces, any witness for minimum stochastic preservation
    already forces the static relevant-coordinate lower bound. -/
theorem stochastic_minimum_preservation_static_relevant_card_le
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A]
    (P : StochasticDecisionProblem A (Fin n → Bool)) (k : ℕ)
    (hmin : StochasticMinimumPreservation P k) :
    (Sigma2PHardness.relevantFinset P.toDecisionProblem).card ≤ k := by
  rcases hmin with ⟨I, hcard, hpres⟩
  have hsub : Sigma2PHardness.relevantFinset P.toDecisionProblem ⊆ I := by
    exact (Sigma2PHardness.sufficient_iff_relevantFinset_subset P.toDecisionProblem I).1
      (stochastic_preservation_implies_static_sufficiency P I hpres)
  exact le_trans (Finset.card_le_card hsub) hcard

/-! ## Full-Support Bridges -/

/-- Under full support, minimum stochastic preservation is equivalent to
    minimum static preservation for the underlying decision problem.

    Proof: By the full-support equivalence, StochasticPreservationSufficient P I ⟺
    P.toDecisionProblem.isSufficient I, so the minimization problems coincide. -/
theorem stochastic_minimum_preservation_iff_static_of_full_support
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S) (k : ℕ)
    (hpos : ∀ s : S, 0 < P.distribution s) :
    StochasticMinimumPreservation P k ↔
    ∃ I : Finset (Fin n), I.card ≤ k ∧ P.toDecisionProblem.isSufficient I := by
  constructor
  · intro ⟨I, hcard, hpres⟩
    exact ⟨I, hcard, stochastic_preservation_implies_static_sufficiency P I hpres⟩
  · intro ⟨I, hcard, hstat⟩
    refine ⟨I, hcard, ?_⟩
    exact static_sufficiency_implies_stochastic_preservation_of_full_support P I hpos hstat

/-- Anchor stochastic preservation implies static anchor sufficiency.

    This direction does not need full support: if the fiber optimizer agrees with
    the full-information optimizer throughout one anchor fiber, then the original
    optimizer is already constant on that fiber. -/
theorem stochastic_anchor_preservation_implies_static_anchor
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    StochasticAnchorPreservation P I → P.toDecisionProblem.anchorSufficient I := by
  intro ⟨s₀, hpres⟩
  unfold DecisionProblem.anchorSufficient
  use s₀
  intro s hs
  calc
    P.toDecisionProblem.Opt s = fiberOpt P I s := by
      symm
      exact hpres s hs
    _ = fiberOpt P I s₀ := fiberOpt_eq_of_agreeOn (P := P) (I := I) hs
    _ = P.toDecisionProblem.Opt s₀ := hpres s₀ (agreeOn_refl s₀ I)

/-- Under full support, static anchor sufficiency lifts to anchor stochastic preservation.

    This is the anchor-local analogue of
    `static_sufficiency_implies_stochastic_preservation_of_full_support`: if the
    full-information optimizer is constant on one observed fiber, then the
    conditional fiber optimizer agrees with it throughout that fiber. -/
theorem static_anchor_implies_stochastic_anchor_preservation_of_full_support
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hpos : ∀ s : S, 0 < P.distribution s) :
    P.toDecisionProblem.anchorSufficient I → StochasticAnchorPreservation P I := by
  rintro ⟨s₀, hstat₀⟩
  use s₀
  have hpres₀ : fiberOpt P I s₀ = P.toDecisionProblem.Opt s₀ := by
    ext a
    constructor
    · intro ha
      classical
      by_contra hnot
      let vals : Finset ℝ := Finset.univ.image (fun x : A => P.utility x s₀)
      have hvals_nonempty : vals.Nonempty := by
        rcases ‹Nonempty A› with ⟨a0⟩
        refine ⟨P.utility a0 s₀, ?_⟩
        exact Finset.mem_image.mpr ⟨a0, by simp, rfl⟩
      let cVal : ℝ := Finset.max' vals hvals_nonempty
      have hcVal_mem : cVal ∈ vals := Finset.max'_mem vals hvals_nonempty
      rcases Finset.mem_image.mp hcVal_mem with ⟨c, -, hcVal_eq⟩
      have hc_opt_s₀ : c ∈ P.toDecisionProblem.Opt s₀ := by
        intro a'
        have ha'mem : P.utility a' s₀ ∈ vals := by
          exact Finset.mem_image.mpr ⟨a', by simp, rfl⟩
        have hle : P.utility a' s₀ ≤ cVal := Finset.le_max' vals (P.utility a' s₀) ha'mem
        simpa [cVal, hcVal_eq] using hle
      have hstrict_s₀ : P.utility a s₀ < P.utility c s₀ := by
        have hnot_le : ¬ P.utility c s₀ ≤ P.utility a s₀ := by
          intro hca
          apply hnot
          intro b
          calc
            P.utility b s₀ ≤ P.utility c s₀ := hc_opt_s₀ b
            _ ≤ P.utility a s₀ := hca
        exact lt_of_not_ge hnot_le
      have hle_term :
          ∀ t : S,
            (if agreeOn t s₀ I then P.distribution t * P.utility a t else 0)
              ≤ (if agreeOn t s₀ I then P.distribution t * P.utility c t else 0) := by
        intro t
        by_cases hts : agreeOn t s₀ I
        · have hc_opt_t : c ∈ P.toDecisionProblem.Opt t := by
            rw [hstat₀ t hts]
            exact hc_opt_s₀
          have hopt := hc_opt_t a
          have hmul : P.distribution t * P.utility a t ≤ P.distribution t * P.utility c t := by
            exact mul_le_mul_of_nonneg_left hopt (le_of_lt (hpos t))
          simp [hts, hmul]
        · simp [hts]
      have hstrict_term_s₀ :
          (if agreeOn s₀ s₀ I then P.distribution s₀ * P.utility a s₀ else 0)
            < (if agreeOn s₀ s₀ I then P.distribution s₀ * P.utility c s₀ else 0) := by
        have hss : agreeOn s₀ s₀ I := agreeOn_refl s₀ I
        have hmul : P.distribution s₀ * P.utility a s₀ < P.distribution s₀ * P.utility c s₀ := by
          exact mul_lt_mul_of_pos_left hstrict_s₀ (hpos s₀)
        simp [hss, hmul]
      have hsum_lt : fiberExpectedUtility P I s₀ a < fiberExpectedUtility P I s₀ c := by
        unfold fiberExpectedUtility
        refine Finset.sum_lt_sum ?_ ?_
        · intro t _
          exact hle_term t
        · refine ⟨s₀, by simp, ?_⟩
          exact hstrict_term_s₀
      have hca : fiberExpectedUtility P I s₀ c ≤ fiberExpectedUtility P I s₀ a := ha c
      exact (not_lt_of_ge hca) hsum_lt
    · intro ha
      intro a'
      have hle_term :
          ∀ t : S,
            (if agreeOn t s₀ I then P.distribution t * P.utility a' t else 0)
              ≤ (if agreeOn t s₀ I then P.distribution t * P.utility a t else 0) := by
        intro t
        by_cases hts : agreeOn t s₀ I
        · have hat : a ∈ P.toDecisionProblem.Opt t := by
            rw [hstat₀ t hts]
            exact ha
          have hopt := hat a'
          have hmul : P.distribution t * P.utility a' t ≤ P.distribution t * P.utility a t := by
            exact mul_le_mul_of_nonneg_left hopt (le_of_lt (hpos t))
          simp [hts, hmul]
        · simp [hts]
      unfold fiberExpectedUtility
      exact Finset.sum_le_sum (by intro t _; exact hle_term t)
  intro s hs
  calc
    fiberOpt P I s = fiberOpt P I s₀ := fiberOpt_eq_of_agreeOn (P := P) (I := I) hs
    _ = P.toDecisionProblem.Opt s₀ := hpres₀
    _ = P.toDecisionProblem.Opt s := by symm; exact hstat₀ s hs

/-- Under full support, anchor stochastic preservation and static anchor sufficiency
    are equivalent. -/
theorem stochastic_anchor_preservation_iff_static_anchor_of_full_support
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hpos : ∀ s : S, 0 < P.distribution s) :
    StochasticAnchorPreservation P I ↔ P.toDecisionProblem.anchorSufficient I := by
  constructor
  · exact stochastic_anchor_preservation_implies_static_anchor P I
  · exact static_anchor_implies_stochastic_anchor_preservation_of_full_support P I hpos

/-! ## Support-Restricted Preservation -/

/-- Support-restricted stochastic preservation: the conditional fiber optimizer
    agrees with the pointwise optimizer on every positive-probability state. This
    isolates the genuine obstruction to the unrestricted general-distribution
    theory, namely zero-probability fibers. -/
def SupportRestrictedPreservation
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  ∀ s : S, 0 < P.distribution s → fiberOpt P I s = P.toDecisionProblem.Opt s

/-- Every `I`-fiber contains a positive-probability representative. This is a
    support-sensitive weakening of full support tailored to preservation. -/
def PositiveFiberSupport
    {S : Type*} {n : ℕ} [Fintype S] [CoordinateSpace S n]
    (dist : S → ℝ) (I : Finset (Fin n)) : Prop :=
  ∀ s : S, ∃ t : S, 0 < dist t ∧ agreeOn t s I

/-- Static sufficiency implies support-restricted stochastic preservation under
    global nonnegativity of the distribution. Unlike the full-support theorem,
    positivity is only needed at the queried state. -/
theorem static_sufficiency_implies_support_restricted_preservation
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hnonneg : ∀ s : S, 0 ≤ P.distribution s)
    (hstat : P.toDecisionProblem.isSufficient I) :
    SupportRestrictedPreservation P I := by
  intro s hspos
  ext a
  constructor
  · intro ha
    classical
    by_contra hnot
    let vals : Finset ℝ := Finset.univ.image (fun x : A => P.utility x s)
    have hvals_nonempty : vals.Nonempty := by
      rcases ‹Nonempty A› with ⟨a0⟩
      refine ⟨P.utility a0 s, ?_⟩
      exact Finset.mem_image.mpr ⟨a0, by simp, rfl⟩
    let cVal : ℝ := Finset.max' vals hvals_nonempty
    have hcVal_mem : cVal ∈ vals := Finset.max'_mem vals hvals_nonempty
    rcases Finset.mem_image.mp hcVal_mem with ⟨c, -, hcVal_eq⟩
    have hc_opt_s : c ∈ P.toDecisionProblem.Opt s := by
      intro a'
      have ha'mem : P.utility a' s ∈ vals := by
        exact Finset.mem_image.mpr ⟨a', by simp, rfl⟩
      have hle : P.utility a' s ≤ cVal := Finset.le_max' vals (P.utility a' s) ha'mem
      simpa [cVal, hcVal_eq] using hle
    have hstrict_s : P.utility a s < P.utility c s := by
      have hnot_le : ¬ P.utility c s ≤ P.utility a s := by
        intro hca
        apply hnot
        intro b
        calc
          P.utility b s ≤ P.utility c s := hc_opt_s b
          _ ≤ P.utility a s := hca
      exact lt_of_not_ge hnot_le
    have hfiber : ∀ t : S, agreeOn t s I → P.toDecisionProblem.Opt t = P.toDecisionProblem.Opt s := by
      intro t ht
      exact hstat t s ht
    have hle_term :
        ∀ t : S,
          (if agreeOn t s I then P.distribution t * P.utility a t else 0)
            ≤ (if agreeOn t s I then P.distribution t * P.utility c t else 0) := by
      intro t
      by_cases hts : agreeOn t s I
      · have hc_opt_t : c ∈ P.toDecisionProblem.Opt t := by
          rw [hfiber t hts]
          exact hc_opt_s
        have hopt := hc_opt_t a
        have hmul : P.distribution t * P.utility a t ≤ P.distribution t * P.utility c t := by
          exact mul_le_mul_of_nonneg_left hopt (hnonneg t)
        simp [hts, hmul]
      · simp [hts]
    have hstrict_term_s :
        (if agreeOn s s I then P.distribution s * P.utility a s else 0)
          < (if agreeOn s s I then P.distribution s * P.utility c s else 0) := by
      have hss : agreeOn s s I := agreeOn_refl s I
      have hmul : P.distribution s * P.utility a s < P.distribution s * P.utility c s := by
        exact mul_lt_mul_of_pos_left hstrict_s hspos
      simp [hss, hmul]
    have hsum_lt : fiberExpectedUtility P I s a < fiberExpectedUtility P I s c := by
      unfold fiberExpectedUtility
      refine Finset.sum_lt_sum ?_ ?_
      · intro t _
        exact hle_term t
      · refine ⟨s, by simp, ?_⟩
        exact hstrict_term_s
    have hca : fiberExpectedUtility P I s c ≤ fiberExpectedUtility P I s a := ha c
    exact (not_lt_of_ge hca) hsum_lt
  · intro ha
    intro a'
    have hfiber : ∀ t : S, agreeOn t s I → P.toDecisionProblem.Opt t = P.toDecisionProblem.Opt s := by
      intro t ht
      exact hstat t s ht
    have hle_term :
        ∀ t : S,
          (if agreeOn t s I then P.distribution t * P.utility a' t else 0)
            ≤ (if agreeOn t s I then P.distribution t * P.utility a t else 0) := by
      intro t
      by_cases hts : agreeOn t s I
      · have hat : a ∈ P.toDecisionProblem.Opt t := by
          rw [hfiber t hts]
          exact ha
        have hopt := hat a'
        have hmul : P.distribution t * P.utility a' t ≤ P.distribution t * P.utility a t := by
          exact mul_le_mul_of_nonneg_left hopt (hnonneg t)
        simp [hts, hmul]
      · simp [hts]
    unfold fiberExpectedUtility
    exact Finset.sum_le_sum (by intro t _; exact hle_term t)

/-- Support-restricted preservation implies static sufficiency on the positive
    support slice. -/
theorem support_restricted_preservation_implies_static_sufficiency_on_support
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hpres : SupportRestrictedPreservation P I) :
    ∀ s s' : S,
      0 < P.distribution s → 0 < P.distribution s' → agreeOn s s' I →
      P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt s' := by
  intro s s' hs hs' hagree
  calc
    P.toDecisionProblem.Opt s = fiberOpt P I s := by
      symm
      exact hpres s hs
    _ = fiberOpt P I s' := fiberOpt_eq_of_agreeOn (P := P) (I := I) hagree
    _ = P.toDecisionProblem.Opt s' := hpres s' hs'

/-- If static sufficiency holds and every observed fiber contains a positive-mass
    representative, then full stochastic preservation follows. This is a
    support-sensitive weakening of the full-support bridge. -/
theorem static_sufficiency_implies_stochastic_preservation_of_positive_fiber_support
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hnonneg : ∀ s : S, 0 ≤ P.distribution s)
    (hcover : PositiveFiberSupport P.distribution I)
    (hstat : P.toDecisionProblem.isSufficient I) :
    StochasticPreservationSufficient P I := by
  have hsupport : SupportRestrictedPreservation P I :=
    static_sufficiency_implies_support_restricted_preservation P I hnonneg hstat
  intro s
  rcases hcover s with ⟨t, htpos, hts⟩
  calc
    fiberOpt P I s = fiberOpt P I t := by
      exact fiberOpt_eq_of_agreeOn (P := P) (I := I) (fun i hi => (hts i hi).symm)
    _ = P.toDecisionProblem.Opt t := hsupport t htpos
    _ = P.toDecisionProblem.Opt s := hstat t s hts

/-- Anchor-local version of the positive-fiber-support bridge. If a static anchor
    witness has a positive-mass representative in its fiber, then anchor
    preservation follows. -/
theorem static_anchor_implies_stochastic_anchor_preservation_of_positive_anchor_fiber
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hnonneg : ∀ s : S, 0 ≤ P.distribution s)
    {s₀ : S}
    (hstat₀ : ∀ s : S, agreeOn s s₀ I → P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt s₀)
    (hsup : ∃ t : S, 0 < P.distribution t ∧ agreeOn t s₀ I) :
    StochasticAnchorPreservation P I := by
  rcases hsup with ⟨t, htpos, hts0⟩
  have hconst_t : ∀ s : S, agreeOn s t I → P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt t := by
    intro s hs
    calc
      P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt s₀ :=
        hstat₀ s (fun i hi => (hs i hi).trans (hts0 i hi))
      _ = P.toDecisionProblem.Opt t := by
        symm
        exact hstat₀ t hts0
  have hpres_t : fiberOpt P I t = P.toDecisionProblem.Opt t := by
    ext a
    constructor
    · intro ha
      classical
      by_contra hnot
      let vals : Finset ℝ := Finset.univ.image (fun x : A => P.utility x t)
      have hvals_nonempty : vals.Nonempty := by
        rcases ‹Nonempty A› with ⟨a0⟩
        refine ⟨P.utility a0 t, ?_⟩
        exact Finset.mem_image.mpr ⟨a0, by simp, rfl⟩
      let cVal : ℝ := Finset.max' vals hvals_nonempty
      have hcVal_mem : cVal ∈ vals := Finset.max'_mem vals hvals_nonempty
      rcases Finset.mem_image.mp hcVal_mem with ⟨c, -, hcVal_eq⟩
      have hc_opt_t : c ∈ P.toDecisionProblem.Opt t := by
        intro a'
        have ha'mem : P.utility a' t ∈ vals := by
          exact Finset.mem_image.mpr ⟨a', by simp, rfl⟩
        have hle : P.utility a' t ≤ cVal := Finset.le_max' vals (P.utility a' t) ha'mem
        simpa [cVal, hcVal_eq] using hle
      have hstrict_t : P.utility a t < P.utility c t := by
        have hnot_le : ¬ P.utility c t ≤ P.utility a t := by
          intro hca
          apply hnot
          intro b
          calc
            P.utility b t ≤ P.utility c t := hc_opt_t b
            _ ≤ P.utility a t := hca
        exact lt_of_not_ge hnot_le
      have hle_term :
          ∀ u : S,
            (if agreeOn u t I then P.distribution u * P.utility a u else 0)
              ≤ (if agreeOn u t I then P.distribution u * P.utility c u else 0) := by
        intro u
        by_cases hut : agreeOn u t I
        · have hc_opt_u : c ∈ P.toDecisionProblem.Opt u := by
            rw [hconst_t u hut]
            exact hc_opt_t
          have hopt := hc_opt_u a
          have hmul : P.distribution u * P.utility a u ≤ P.distribution u * P.utility c u := by
            exact mul_le_mul_of_nonneg_left hopt (hnonneg u)
          simp [hut, hmul]
        · simp [hut]
      have hstrict_term_t :
          (if agreeOn t t I then P.distribution t * P.utility a t else 0)
            < (if agreeOn t t I then P.distribution t * P.utility c t else 0) := by
        have htt : agreeOn t t I := agreeOn_refl t I
        have hmul : P.distribution t * P.utility a t < P.distribution t * P.utility c t := by
          exact mul_lt_mul_of_pos_left hstrict_t htpos
        simp [htt, hmul]
      have hsum_lt : fiberExpectedUtility P I t a < fiberExpectedUtility P I t c := by
        unfold fiberExpectedUtility
        refine Finset.sum_lt_sum ?_ ?_
        · intro u _
          exact hle_term u
        · refine ⟨t, by simp, ?_⟩
          exact hstrict_term_t
      have hca : fiberExpectedUtility P I t c ≤ fiberExpectedUtility P I t a := ha c
      exact (not_lt_of_ge hca) hsum_lt
    · intro ha
      intro a'
      have hle_term :
          ∀ u : S,
            (if agreeOn u t I then P.distribution u * P.utility a' u else 0)
              ≤ (if agreeOn u t I then P.distribution u * P.utility a u else 0) := by
        intro u
        by_cases hut : agreeOn u t I
        · have ha_u : a ∈ P.toDecisionProblem.Opt u := by
            rw [hconst_t u hut]
            exact ha
          have hopt := ha_u a'
          have hmul : P.distribution u * P.utility a' u ≤ P.distribution u * P.utility a u := by
            exact mul_le_mul_of_nonneg_left hopt (hnonneg u)
          simp [hut, hmul]
        · simp [hut]
      unfold fiberExpectedUtility
      exact Finset.sum_le_sum (by intro u _; exact hle_term u)
  use s₀
  intro s hs
  calc
    fiberOpt P I s = fiberOpt P I t := by
      exact fiberOpt_eq_of_agreeOn (P := P) (I := I) (fun i hi => (hs i hi).trans ((hts0 i hi).symm))
    _ = P.toDecisionProblem.Opt t := hpres_t
    _ = P.toDecisionProblem.Opt s := by
      symm
      exact hconst_t s (fun i hi => (hs i hi).trans ((hts0 i hi).symm))

/-! ## Explicit Finite Search for Preservation Variants -/

/-- Explicit counted search for minimum stochastic preservation.
    It scans the subset lattice looking for a coordinate set of size at most `k`
    that satisfies the preservation predicate. -/
noncomputable def countedStochasticMinimumPreservationSearch
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) : Counted Bool :=
  countedAnyList (Finset.powerset (Finset.univ : Finset (Fin n))).toList
    (fun I => decide (I.card ≤ k) && stochasticPreservationBool P I)

theorem countedStochasticMinimumPreservationSearch_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) :
    (countedStochasticMinimumPreservationSearch P k).result = true ↔
      StochasticMinimumPreservation P k := by
  constructor
  · intro h
    rcases (countedAnyList_result_true_iff _ _).mp h with ⟨I, hI, hPred⟩
    have hPred' : decide (I.card ≤ k) = true ∧ stochasticPreservationBool P I = true := by
      simpa using hPred
    exact ⟨I, by
      constructor
      · simpa using hPred'.1
      · exact (stochasticPreservationBool_spec P I).mp hPred'.2⟩
  · rintro ⟨I, hCard, hPres⟩
    have hmemPow : I ∈ Finset.powerset (Finset.univ : Finset (Fin n)) := by
      apply Finset.mem_powerset.mpr
      exact Finset.subset_univ I
    have hmemList : I ∈ (Finset.powerset (Finset.univ : Finset (Fin n))).toList := by
      simpa using hmemPow
    apply (countedAnyList_result_true_iff _ _).mpr
    refine ⟨I, hmemList, ?_⟩
    simp [hCard, stochasticPreservationBool_spec P I, hPres]

theorem countedStochasticMinimumPreservationSearch_steps
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) :
    (countedStochasticMinimumPreservationSearch P k).steps ≤ 2 ^ n := by
  have hlen : (Finset.powerset (Finset.univ : Finset (Fin n))).toList.length = 2 ^ n := by
    simp
  calc
    (countedStochasticMinimumPreservationSearch P k).steps
        ≤ (Finset.powerset (Finset.univ : Finset (Fin n))).toList.length :=
          countedAnyList_steps_le_length _ _
    _ = 2 ^ n := hlen

/-- Explicit counted witness search for anchor stochastic preservation. -/
noncomputable def countedStochasticAnchorPreservationSearch
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Counted Bool :=
  countedAnyList (Finset.univ.toList : List S)
    (fun s0 => decide (∀ s : S, agreeOn s s0 I → fiberOpt P I s = P.toDecisionProblem.Opt s))

theorem countedStochasticAnchorPreservationSearch_spec
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    (countedStochasticAnchorPreservationSearch P I).result = true ↔
      StochasticAnchorPreservation P I := by
  constructor
  · intro h
    rcases (countedAnyList_result_true_iff _ _).mp h with ⟨s0, _, hs0⟩
    exact ⟨s0, by simpa [countedStochasticAnchorPreservationSearch] using hs0⟩
  · rintro ⟨s0, hs0⟩
    apply (countedAnyList_result_true_iff _ _).mpr
    refine ⟨s0, ?_, ?_⟩
    · simpa using (Finset.mem_toList.mpr (Finset.mem_univ s0))
    · simpa [countedStochasticAnchorPreservationSearch] using hs0

theorem countedStochasticAnchorPreservationSearch_steps
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    (countedStochasticAnchorPreservationSearch P I).steps ≤ Fintype.card S := by
  calc
    (countedStochasticAnchorPreservationSearch P I).steps ≤ (Finset.univ.toList : List S).length :=
      countedAnyList_steps_le_length _ _
    _ = Fintype.card S := by simp

theorem stochasticMinimumPreservation_counted_search_witness
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (k : ℕ) :
    HasCountedSearchWitness
      (StochasticMinimumPreservation P k)
      (countedStochasticMinimumPreservationSearch P k)
      (2 ^ n) := by
  constructor
  · exact countedStochasticMinimumPreservationSearch_spec P k
  · exact countedStochasticMinimumPreservationSearch_steps P k

theorem stochasticAnchorPreservation_counted_search_witness
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    HasCountedSearchWitness
      (StochasticAnchorPreservation P I)
      (countedStochasticAnchorPreservationSearch P I)
      (Fintype.card S) := by
  constructor
  · exact countedStochasticAnchorPreservationSearch_spec P I
  · exact countedStochasticAnchorPreservationSearch_steps P I

/-! ## General Distribution Case (Exploratory) -/

/-- Attempted definition: stochastically-relevant-for-preservation.
    A coordinate i is stochastically-relevant-for-preservation if changing it
    can affect the conditional optimizer on some positive-probability fiber. -/
def stochasticallyRelevantForPreservation
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (i : Fin n) : Prop :=
  ∃ s : S, 0 < P.distribution s ∧
    fiberOpt P (Finset.univ.erase i) s ≠ P.toDecisionProblem.Opt s

/-- Conjecture: Under general distributions, stochastic preservation is equivalent
    to containing all stochastically-relevant coordinates.

    Status: OPEN. This may or may not hold. If it holds, minimum stochastic
    preservation is coNP-complete in general. If it fails, the general case
    may be harder. -/
axiom stochastic_preservation_iff_contains_stochastically_relevant_conjecture
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) :
    StochasticPreservationSufficient P I ↔
    ∀ i : Fin n, stochasticallyRelevantForPreservation P i → i ∈ I

end DecisionQuotient.StochasticSequential
