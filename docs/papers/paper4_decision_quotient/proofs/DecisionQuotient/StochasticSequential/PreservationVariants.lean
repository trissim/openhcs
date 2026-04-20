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
def stochastic_preservation_iff_contains_stochastically_relevant_conjecture
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
    StochasticPreservationSufficient P I ↔
    ∀ i : Fin n, stochasticallyRelevantForPreservation P i → i ∈ I

/-- On Boolean product state spaces with full support, any statically relevant
coordinate is stochastically relevant for preservation (tested by erasing that
coordinate from the observed set). -/
theorem stochasticallyRelevantForPreservation_of_static_relevant_of_full_support
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool))
    (hpos : ∀ s : Fin n → Bool, 0 < P.distribution s)
    (i : Fin n)
    (hrel : P.toDecisionProblem.isRelevant i) :
    stochasticallyRelevantForPreservation P i := by
  have hNotSuffErase : ¬ P.toDecisionProblem.isSufficient (Finset.univ.erase i) := by
    intro hSuffErase
    have hContain :=
      (Sigma2PHardness.sufficient_iff_relevant_subset P.toDecisionProblem
        (Finset.univ.erase i)).1 hSuffErase
    have hiMem : i ∈ (Finset.univ.erase i) := hContain i hrel
    have hiNotMem : i ∉ (Finset.univ.erase i) := by simp
    exact hiNotMem hiMem
  have hNotPresErase : ¬ StochasticPreservationSufficient P (Finset.univ.erase i) := by
    intro hPresErase
    exact hNotSuffErase
      ((static_sufficiency_iff_stochastic_preservation_of_full_support P
        (Finset.univ.erase i) hpos).2 hPresErase)
  have hWitness :
      ∃ s : Fin n → Bool,
        fiberOpt P (Finset.univ.erase i) s ≠ P.toDecisionProblem.Opt s := by
    by_contra hNo
    apply hNotPresErase
    intro s
    by_contra hs
    exact hNo ⟨s, hs⟩
  rcases hWitness with ⟨s, hs⟩
  exact ⟨s, hpos s, hs⟩

/-- Full-support Boolean-state resolution of the exploratory general-distribution
conjecture: stochastic preservation is exactly containment of all
stochastically-relevant-for-preservation coordinates. -/
theorem stochastic_preservation_iff_contains_stochastically_relevant_of_full_support
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool)) (I : Finset (Fin n))
    (hpos : ∀ s : Fin n → Bool, 0 < P.distribution s) :
    StochasticPreservationSufficient P I ↔
      ∀ i : Fin n, stochasticallyRelevantForPreservation P i → i ∈ I := by
  constructor
  · intro hPres i hStochRel
    rcases hStochRel with ⟨s, _hspos, hsneq⟩
    have hNotPresErase : ¬ StochasticPreservationSufficient P (Finset.univ.erase i) := by
      intro hErase
      exact hsneq (hErase s)
    have hNotSuffErase : ¬ P.toDecisionProblem.isSufficient (Finset.univ.erase i) := by
      intro hSuffErase
      exact hNotPresErase
        ((static_sufficiency_iff_stochastic_preservation_of_full_support P
          (Finset.univ.erase i) hpos).1 hSuffErase)
    have hRel : P.toDecisionProblem.isRelevant i := by
      by_contra hNotRel
      have hContainErase :
          ∀ j : Fin n, P.toDecisionProblem.isRelevant j → j ∈ Finset.univ.erase i := by
        intro j hj
        by_cases hji : j = i
        · exact False.elim (hNotRel (hji ▸ hj))
        · simp [hji]
      have hSuffErase : P.toDecisionProblem.isSufficient (Finset.univ.erase i) :=
        (Sigma2PHardness.sufficient_iff_relevant_subset P.toDecisionProblem
          (Finset.univ.erase i)).2 hContainErase
      exact hNotSuffErase hSuffErase
    have hSuffI : P.toDecisionProblem.isSufficient I :=
      stochastic_preservation_implies_static_sufficiency P I hPres
    exact P.toDecisionProblem.sufficient_contains_relevant I hSuffI i hRel
  · intro hContain
    have hSuffI : P.toDecisionProblem.isSufficient I := by
      refine (Sigma2PHardness.sufficient_iff_relevant_subset P.toDecisionProblem I).2 ?_
      intro i hRel
      exact hContain i
        (stochasticallyRelevantForPreservation_of_static_relevant_of_full_support
          P hpos i hRel)
    exact (static_sufficiency_iff_stochastic_preservation_of_full_support P I hpos).1 hSuffI

/-- The exploratory conjecture is discharged for Boolean product spaces under
full-support hypotheses. -/
theorem stochastic_preservation_iff_contains_stochastically_relevant_conjecture_of_full_support
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool)) (I : Finset (Fin n))
    (hpos : ∀ s : Fin n → Bool, 0 < P.distribution s) :
    stochastic_preservation_iff_contains_stochastically_relevant_conjecture P I :=
  stochastic_preservation_iff_contains_stochastically_relevant_of_full_support P I hpos

/-- Under nonnegative distributions, stochastic preservation implies containment
of all stochastically-relevant-for-preservation coordinates (necessary
direction of the exploratory conjecture). -/
theorem stochastic_preservation_contains_stochastically_relevant_of_nonneg
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hnonneg : ∀ s : S, 0 ≤ P.distribution s)
    (hpres : StochasticPreservationSufficient P I) :
    ∀ i : Fin n, stochasticallyRelevantForPreservation P i → i ∈ I := by
  intro i hRel
  rcases hRel with ⟨s, hspos, hsneq⟩
  by_contra hiNot
  have hstatI : P.toDecisionProblem.isSufficient I :=
    stochastic_preservation_implies_static_sufficiency P I hpres
  have hsubset : I ⊆ Finset.univ.erase i := by
    intro j hj
    refine Finset.mem_erase.mpr ?_
    refine ⟨?_, Finset.mem_univ j⟩
    intro hji
    exact hiNot (hji ▸ hj)
  have hstatErase : P.toDecisionProblem.isSufficient (Finset.univ.erase i) :=
    P.toDecisionProblem.sufficient_superset I (Finset.univ.erase i) hstatI hsubset
  have hsupportErase : SupportRestrictedPreservation P (Finset.univ.erase i) :=
    static_sufficiency_implies_support_restricted_preservation P
      (Finset.univ.erase i) hnonneg hstatErase
  exact hsneq (hsupportErase s hspos)

/-- Every full-support distribution provides positive-fiber support for every
coordinate set. -/
theorem positiveFiberSupport_of_full_support
    {S : Type*} {n : ℕ} [Fintype S] [CoordinateSpace S n]
    (dist : S → ℝ) (I : Finset (Fin n))
    (hpos : ∀ s : S, 0 < dist s) :
    PositiveFiberSupport dist I := by
  intro s
  exact ⟨s, hpos s, agreeOn_refl s I⟩

/-- Boolean-state reduction of the exploratory conjecture under nonnegative
distributions and positive-fiber support: the equivalence follows once static
relevance is known to imply stochastic relevance. This isolates the remaining
general-distribution obstruction to one bridge premise. -/
theorem stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_support_bridge
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool)) (I : Finset (Fin n))
    (hnonneg : ∀ s : Fin n → Bool, 0 ≤ P.distribution s)
    (hcover : PositiveFiberSupport P.distribution I)
    (hBridge : ∀ i : Fin n,
      P.toDecisionProblem.isRelevant i → stochasticallyRelevantForPreservation P i) :
    StochasticPreservationSufficient P I ↔
      ∀ i : Fin n, stochasticallyRelevantForPreservation P i → i ∈ I := by
  constructor
  · intro hPres
    exact stochastic_preservation_contains_stochastically_relevant_of_nonneg
      P I hnonneg hPres
  · intro hContain
    have hContainStatic :
        ∀ i : Fin n, P.toDecisionProblem.isRelevant i → i ∈ I := by
      intro i hRel
      exact hContain i (hBridge i hRel)
    have hstatI : P.toDecisionProblem.isSufficient I :=
      (Sigma2PHardness.sufficient_iff_relevant_subset P.toDecisionProblem I).2
        hContainStatic
    exact static_sufficiency_implies_stochastic_preservation_of_positive_fiber_support
      P I hnonneg hcover hstatI

/-- Support-transport witness: every observed fiber admits a positive-probability
representative whose optimizer set matches the queried state optimizer set. This
explicitly controls zero-mass fibers in the unrestricted-distribution regime. -/
def SupportRepresentativeOptTransport
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) : Prop :=
  ∀ s : S, ∃ t : S,
    0 < P.distribution t ∧
    agreeOn t s I ∧
    P.toDecisionProblem.Opt t = P.toDecisionProblem.Opt s

/-- Primitive finite-time dynamics interface for deriving support transport from
state evolution rather than assuming support representatives directly. The
evolution must preserve queried coordinates and optimizer sets, and it must hit
positive-probability states after a fixed burn-in time. -/
structure PrimitiveSupportTransportDynamics
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) where
  evolution : ℕ → S → S
  evolution_zero : ∀ s : S, evolution 0 s = s
  evolution_add : ∀ k l : ℕ, ∀ s : S,
    evolution (k + l) s = evolution k (evolution l s)
  burnIn : ℕ
  positive_after_burnIn : ∀ s : S, 0 < P.distribution (evolution burnIn s)
  fiber_invariant : ∀ k : ℕ, ∀ s : S, agreeOn (evolution k s) s I
  optimizer_invariant :
    ∀ k : ℕ, ∀ s : S,
      P.toDecisionProblem.Opt (evolution k s) = P.toDecisionProblem.Opt s

/-- Explicit one-step deterministic dynamics model used to derive primitive
support transport assumptions constructively from iterated state evolution. -/
structure ExplicitStepSupportTransportDynamics
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n)) where
  step : S → S
  burnIn : ℕ
  positive_after_burnIn :
    ∀ s : S, 0 < P.distribution ((step^[burnIn]) s)
  step_fiber_invariant : ∀ s : S, agreeOn (step s) s I
  step_optimizer_invariant :
    ∀ s : S, P.toDecisionProblem.Opt (step s) = P.toDecisionProblem.Opt s

theorem ExplicitStepSupportTransportDynamics.fiber_invariant_iterate
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    {P : StochasticDecisionProblem A S} {I : Finset (Fin n)}
    (D : ExplicitStepSupportTransportDynamics P I) :
    ∀ k : ℕ, ∀ s : S, agreeOn ((D.step^[k]) s) s I := by
  intro k
  induction k with
  | zero =>
      intro s
      simpa using agreeOn_refl s I
  | succ k ih =>
      intro s
      have hStep : agreeOn (D.step ((D.step^[k]) s))
          ((D.step^[k]) s) I :=
        D.step_fiber_invariant ((D.step^[k]) s)
      have hPrev : agreeOn ((D.step^[k]) s) s I := ih s
      intro i hi
      calc
        CoordinateSpace.proj ((D.step^[k + 1]) s) i
            = CoordinateSpace.proj (D.step ((D.step^[k]) s)) i := by
                simp [Function.iterate_succ_apply']
        _ = CoordinateSpace.proj ((D.step^[k]) s) i := hStep i hi
        _ = CoordinateSpace.proj s i := hPrev i hi

theorem ExplicitStepSupportTransportDynamics.optimizer_invariant_iterate
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    {P : StochasticDecisionProblem A S} {I : Finset (Fin n)}
    (D : ExplicitStepSupportTransportDynamics P I) :
    ∀ k : ℕ, ∀ s : S,
      P.toDecisionProblem.Opt ((D.step^[k]) s) =
        P.toDecisionProblem.Opt s := by
  intro k
  induction k with
  | zero =>
      intro s
      simp
  | succ k ih =>
      intro s
      calc
        P.toDecisionProblem.Opt ((D.step^[k + 1]) s)
            = P.toDecisionProblem.Opt (D.step ((D.step^[k]) s)) := by
                simp [Function.iterate_succ_apply']
        _ = P.toDecisionProblem.Opt ((D.step^[k]) s) :=
              D.step_optimizer_invariant ((D.step^[k]) s)
        _ = P.toDecisionProblem.Opt s := ih s

/-- Any explicit one-step deterministic dynamics model induces a primitive
finite-time support-transport dynamics package by iteration. -/
noncomputable def ExplicitStepSupportTransportDynamics.toPrimitiveSupportTransportDynamics
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    {P : StochasticDecisionProblem A S} {I : Finset (Fin n)}
    (D : ExplicitStepSupportTransportDynamics P I) :
    PrimitiveSupportTransportDynamics P I where
  evolution := fun k s => (D.step^[k]) s
  evolution_zero := by
    intro s
    simp
  evolution_add := by
    intro k l s
    simpa using Function.iterate_add_apply D.step k l s
  burnIn := D.burnIn
  positive_after_burnIn := D.positive_after_burnIn
  fiber_invariant := D.fiber_invariant_iterate
  optimizer_invariant := D.optimizer_invariant_iterate

/-- Primitive finite-time dynamics imply support-transport representatives by
choosing the burn-in image of each queried state. -/
theorem supportRepresentativeOptTransport_of_primitive_dynamics
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (D : PrimitiveSupportTransportDynamics P I) :
    SupportRepresentativeOptTransport P I := by
  intro s
  refine ⟨D.evolution D.burnIn s, D.positive_after_burnIn s, ?_, ?_⟩
  · exact D.fiber_invariant D.burnIn s
  · exact D.optimizer_invariant D.burnIn s

/-- Explicit-step deterministic dynamics directly yield support-transport
representatives through their primitive-dynamics conversion. -/
theorem supportRepresentativeOptTransport_of_explicit_step_dynamics
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (D : ExplicitStepSupportTransportDynamics P I) :
    SupportRepresentativeOptTransport P I := by
  exact supportRepresentativeOptTransport_of_primitive_dynamics P I
    D.toPrimitiveSupportTransportDynamics

/-- Full support trivially implies support-transport witnesses (choose `t = s`).
-/
theorem supportRepresentativeOptTransport_of_full_support
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hpos : ∀ s : S, 0 < P.distribution s) :
    SupportRepresentativeOptTransport P I := by
  intro s
  exact ⟨s, hpos s, agreeOn_refl s I, rfl⟩

/-- Support-restricted preservation lifts to full stochastic preservation once
support-transport witnesses are provided for the queried fibers. -/
theorem support_restricted_preservation_implies_stochastic_preservation_of_support_transport
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hpres : SupportRestrictedPreservation P I)
    (htransport : SupportRepresentativeOptTransport P I) :
    StochasticPreservationSufficient P I := by
  intro s
  rcases htransport s with ⟨t, htpos, hts, hopt⟩
  calc
    fiberOpt P I s = fiberOpt P I t := by
      exact fiberOpt_eq_of_agreeOn (P := P) (I := I) (fun i hi => (hts i hi).symm)
    _ = P.toDecisionProblem.Opt t := hpres t htpos
    _ = P.toDecisionProblem.Opt s := hopt

/-- Support-restricted preservation yields static sufficiency if each queried
fiber admits a positive representative with matching optimizer set. -/
theorem support_restricted_preservation_implies_static_sufficiency_of_support_transport
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n]
    (P : StochasticDecisionProblem A S) (I : Finset (Fin n))
    (hpres : SupportRestrictedPreservation P I)
    (htransport : SupportRepresentativeOptTransport P I) :
    P.toDecisionProblem.isSufficient I := by
  intro s s' hss'
  rcases htransport s with ⟨t, htpos, hts, hopt_t⟩
  rcases htransport s' with ⟨u, hupos, hus, hopt_u⟩
  have htu : agreeOn t u I := by
    intro i hi
    calc
      CoordinateSpace.proj t i = CoordinateSpace.proj s i := hts i hi
      _ = CoordinateSpace.proj s' i := hss' i hi
      _ = CoordinateSpace.proj u i := (hus i hi).symm
  have hOpt_tu : P.toDecisionProblem.Opt t = P.toDecisionProblem.Opt u := by
    calc
      P.toDecisionProblem.Opt t = fiberOpt P I t := by
        symm
        exact hpres t htpos
      _ = fiberOpt P I u := fiberOpt_eq_of_agreeOn (P := P) (I := I) htu
      _ = P.toDecisionProblem.Opt u := hpres u hupos
  calc
    P.toDecisionProblem.Opt s = P.toDecisionProblem.Opt t := by
      symm
      exact hopt_t
    _ = P.toDecisionProblem.Opt u := hOpt_tu
    _ = P.toDecisionProblem.Opt s' := hopt_u

/-- Static relevance implies stochastic relevance for one-coordinate erasure
fibers once zero-mass fibers are controlled by support-transport witnesses. -/
theorem stochasticallyRelevantForPreservation_of_static_relevant_of_support_transport
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S)
    (hnonneg : ∀ s : S, 0 ≤ P.distribution s)
    (i : Fin n)
    (htransportErase :
      SupportRepresentativeOptTransport P (Finset.univ.erase i))
    (hrel : P.toDecisionProblem.isRelevant i) :
    stochasticallyRelevantForPreservation P i := by
  by_contra hnot
  have hpresErase : SupportRestrictedPreservation P (Finset.univ.erase i) := by
    intro s hs
    by_contra hsneq
    exact hnot ⟨s, hs, hsneq⟩
  have hstatErase : P.toDecisionProblem.isSufficient (Finset.univ.erase i) :=
    support_restricted_preservation_implies_static_sufficiency_of_support_transport
      P (Finset.univ.erase i) hpresErase htransportErase
  have hiMem : i ∈ Finset.univ.erase i :=
    P.toDecisionProblem.sufficient_contains_relevant (Finset.univ.erase i)
      hstatErase i hrel
  simpa using hiMem

/-- Unrestricted-distribution closure theorem on Boolean product states: under
nonnegative distributions and explicit support-transport witnesses for queried
and one-coordinate-erasure fibers, stochastic preservation is equivalent to
containment of all stochastically-relevant-for-preservation coordinates. -/
theorem stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_support_transport
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool))
    (I : Finset (Fin n))
    (hnonneg : ∀ s : Fin n → Bool, 0 ≤ P.distribution s)
    (htransportI : SupportRepresentativeOptTransport P I)
    (htransportErase :
      ∀ i : Fin n,
        SupportRepresentativeOptTransport P (Finset.univ.erase i)) :
    StochasticPreservationSufficient P I ↔
      ∀ i : Fin n, stochasticallyRelevantForPreservation P i → i ∈ I := by
  constructor
  · intro hPres
    exact stochastic_preservation_contains_stochastically_relevant_of_nonneg
      P I hnonneg hPres
  · intro hContain
    have hContainStatic :
        ∀ i : Fin n, P.toDecisionProblem.isRelevant i → i ∈ I := by
      intro i hrel
      exact hContain i
        (stochasticallyRelevantForPreservation_of_static_relevant_of_support_transport
          P hnonneg i (htransportErase i) hrel)
    have hstatI : P.toDecisionProblem.isSufficient I :=
      (Sigma2PHardness.sufficient_iff_relevant_subset P.toDecisionProblem I).2
        hContainStatic
    have hsupportI : SupportRestrictedPreservation P I :=
      static_sufficiency_implies_support_restricted_preservation P I hnonneg hstatI
    exact
      support_restricted_preservation_implies_stochastic_preservation_of_support_transport
        P I hsupportI htransportI

/-- The exploratory conjecture is discharged on Boolean product states under the
explicit unrestricted-distribution support-transport assumptions. -/
theorem stochastic_preservation_iff_contains_stochastically_relevant_conjecture_of_nonneg_support_transport
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool))
    (I : Finset (Fin n))
    (hnonneg : ∀ s : Fin n → Bool, 0 ≤ P.distribution s)
    (htransportI : SupportRepresentativeOptTransport P I)
    (htransportErase :
      ∀ i : Fin n,
        SupportRepresentativeOptTransport P (Finset.univ.erase i)) :
    stochastic_preservation_iff_contains_stochastically_relevant_conjecture P I :=
  stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_support_transport
    P I hnonneg htransportI htransportErase

/-- Primitive finite-time dynamics on one-coordinate erasure fibers are enough to
derive static-to-stochastic relevance transport without assuming support
representatives directly. -/
theorem stochasticallyRelevantForPreservation_of_static_relevant_of_primitive_dynamics
    {A S : Type*} {n : ℕ} [Fintype A] [Fintype S] [DecidableEq A]
    [CoordinateSpace S n] [Nonempty A]
    (P : StochasticDecisionProblem A S)
    (hnonneg : ∀ s : S, 0 ≤ P.distribution s)
    (i : Fin n)
    (hdynErase : PrimitiveSupportTransportDynamics P (Finset.univ.erase i))
    (hrel : P.toDecisionProblem.isRelevant i) :
    stochasticallyRelevantForPreservation P i := by
  exact stochasticallyRelevantForPreservation_of_static_relevant_of_support_transport
    P hnonneg i
    (supportRepresentativeOptTransport_of_primitive_dynamics
      (P := P) (I := Finset.univ.erase i) hdynErase)
    hrel

/-- Boolean-state unrestricted-distribution closure derived from primitive
finite-time dynamics, instead of postulating support-transport representatives.
-/
theorem stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_primitive_dynamics
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool))
    (I : Finset (Fin n))
    (hnonneg : ∀ s : Fin n → Bool, 0 ≤ P.distribution s)
    (hdynI : PrimitiveSupportTransportDynamics P I)
    (hdynErase :
      ∀ i : Fin n,
        PrimitiveSupportTransportDynamics P (Finset.univ.erase i)) :
    StochasticPreservationSufficient P I ↔
      ∀ i : Fin n, stochasticallyRelevantForPreservation P i → i ∈ I := by
  exact
    stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_support_transport
      P I hnonneg
      (supportRepresentativeOptTransport_of_primitive_dynamics (P := P) (I := I) hdynI)
      (fun i =>
        supportRepresentativeOptTransport_of_primitive_dynamics
          (P := P)
          (I := Finset.univ.erase i)
          (hdynErase i))

/-- Conjecture discharge under nonnegative distributions on Boolean product
states from primitive finite-time dynamics assumptions. -/
theorem stochastic_preservation_iff_contains_stochastically_relevant_conjecture_of_nonneg_primitive_dynamics
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool))
    (I : Finset (Fin n))
    (hnonneg : ∀ s : Fin n → Bool, 0 ≤ P.distribution s)
    (hdynI : PrimitiveSupportTransportDynamics P I)
    (hdynErase :
      ∀ i : Fin n,
        PrimitiveSupportTransportDynamics P (Finset.univ.erase i)) :
    stochastic_preservation_iff_contains_stochastically_relevant_conjecture P I :=
  stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_primitive_dynamics
    P I hnonneg hdynI hdynErase

/-- Boolean-state unrestricted-distribution closure under explicit one-step
deterministic dynamics on queried and erasure fibers. -/
theorem stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_explicit_step_dynamics
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool))
    (I : Finset (Fin n))
    (hnonneg : ∀ s : Fin n → Bool, 0 ≤ P.distribution s)
    (hdynI : ExplicitStepSupportTransportDynamics P I)
    (hdynErase :
      ∀ i : Fin n,
        ExplicitStepSupportTransportDynamics P (Finset.univ.erase i)) :
    StochasticPreservationSufficient P I ↔
      ∀ i : Fin n, stochasticallyRelevantForPreservation P i → i ∈ I :=
  stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_primitive_dynamics
    P I hnonneg
    hdynI.toPrimitiveSupportTransportDynamics
    (fun i => (hdynErase i).toPrimitiveSupportTransportDynamics)

/-- Conjecture discharge under nonnegative distributions on Boolean product
states from explicit one-step deterministic dynamics assumptions. -/
theorem stochastic_preservation_iff_contains_stochastically_relevant_conjecture_of_nonneg_explicit_step_dynamics
    {A : Type*} {n : ℕ} [Fintype A] [DecidableEq A] [Nonempty A]
    (P : StochasticDecisionProblem A (Fin n → Bool))
    (I : Finset (Fin n))
    (hnonneg : ∀ s : Fin n → Bool, 0 ≤ P.distribution s)
    (hdynI : ExplicitStepSupportTransportDynamics P I)
    (hdynErase :
      ∀ i : Fin n,
        ExplicitStepSupportTransportDynamics P (Finset.univ.erase i)) :
    stochastic_preservation_iff_contains_stochastically_relevant_conjecture P I :=
  stochastic_preservation_iff_contains_stochastically_relevant_of_nonneg_explicit_step_dynamics
    P I hnonneg hdynI hdynErase

end DecisionQuotient.StochasticSequential
