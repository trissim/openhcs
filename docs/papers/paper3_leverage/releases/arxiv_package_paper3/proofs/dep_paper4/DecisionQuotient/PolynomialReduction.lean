/-
  Paper 4: Decision-Relevant Uncertainty

  PolynomialReduction.lean - Reduction Correctness Framework

  This file formalizes the CORRECTNESS of reductions between decision problems.

  ## Important Note on Polynomial-Time Computability

  A Karp reduction has two parts:
  1. **Correctness**: ∀ x, x ∈ A ↔ f(x) ∈ B (membership preservation)
  2. **Efficiency**: f is computable in polynomial time on a Turing machine

  Part (1) can be fully machine-verified in Lean.

  Part (2) — proving that a specific function is polynomial-time computable on
  an actual Turing machine — is an **open problem in formal verification**.
  Mathlib defines `TM2ComputableInPolyTime`, but actually proving that concrete
  functions satisfy this definition requires building explicit TM implementations
  and proving their runtime bounds. No major formalization project has done this
  end-to-end for NP-completeness proofs.

  This file takes the honest approach:
  - **Correctness is fully proved** (zero sorry)
  - **Polynomial-time computability is an explicit hypothesis** in theorems about
    complete Karp reductions

  The gap is declared, not hidden.

  ## Key Definitions
  - `ReductionCorrectness`: The fully-verifiable correctness condition
  - `valid_karp_reduction`: Theorem schema combining correctness + polytime hypothesis
  - `SizeBoundedReduction`: Size bounds (weaker than polytime, but provable)

  ## Triviality Level
  TRIVIAL for correctness definitions. The nontrivial correctness PROOFS are
  in Reduction.lean (TAUTOLOGY reduction).

  ## Dependencies
  - Chain: AlgorithmComplexity.lean (complexity bounds) → here
  - The nontrivial work is in Reduction.lean (TAUTOLOGY reduction)
-/

import DecisionQuotient.AlgorithmComplexity
import DecisionQuotient.Complexity
import Mathlib.Computability.Reduce
import Mathlib.Tactic

namespace DecisionQuotient

/-! ## Reduction Correctness

The correctness condition for a many-one reduction: f preserves membership.
This is the part that CAN be fully verified in Lean. -/

/-- Correctness of a many-one reduction: f preserves membership.
    This is fully machine-verifiable. -/
structure ReductionCorrectness {A B : Type*} (f : A → B) (LA : Set A) (LB : Set B) : Prop where
  /-- The reduction preserves membership -/
  preserves_membership : ∀ x, x ∈ LA ↔ f x ∈ LB

/-- Convenience constructor for ReductionCorrectness -/
theorem ReductionCorrectness.mk' {A B : Type*} {f : A → B} {LA : Set A} {LB : Set B}
    (h : ∀ x, x ∈ LA ↔ f x ∈ LB) : ReductionCorrectness f LA LB :=
  ⟨h⟩

/-! ## Valid Karp Reductions (Honest Formulation)

A valid Karp reduction requires BOTH correctness AND polynomial-time computability.
This section provides the honest formulation where polytime is an explicit hypothesis. -/

/-- A valid Karp reduction combines proved correctness with polytime as a hypothesis.

    This theorem captures: "If f preserves membership (proved) AND f is polytime
    (assumed as hypothesis), then A ≤ₘᵖ B."

    The correctness condition can be fully machine-verified.
    The polytime condition is stated as an explicit hypothesis because TM verification
    is an open problem. This makes the gap visible rather than hiding it.

    In papers, state: "The correctness of each reduction is machine-verified.
    Polynomial-time computability of the reduction functions is a standard claim
    stated as an explicit hypothesis; full Turing-machine verification of
    polynomial-time bounds remains an open problem in formal verification." -/
theorem valid_karp_reduction
    {α β : Type} (ea : Computability.FinEncoding α) (eb : Computability.FinEncoding β)
    (A : Set α) (B : Set β) (f : α → β)
    (hcorrect : ∀ x, x ∈ A ↔ f x ∈ B)  -- Proved: correctness
    (hpolytime : Complexity.PolyTime ea eb f) :  -- Hypothesis: polytime
    Complexity.ManyOneReducesPoly ea eb A B :=
  ⟨f, hpolytime, hcorrect⟩

/-- Variant using `ReductionCorrectness` structure -/
theorem valid_karp_reduction' {α β : Type}
    (ea : Computability.FinEncoding α) (eb : Computability.FinEncoding β)
    (A : Set α) (B : Set β) (f : α → β)
    (hcorrect : ReductionCorrectness f A B)
    (hpolytime : Complexity.PolyTime ea eb f) :
    Complexity.ManyOneReducesPoly ea eb A B :=
  valid_karp_reduction ea eb A B f hcorrect.preserves_membership hpolytime

/-! ## Size-Bounded Reductions

Size bounds on output are weaker than polynomial-time computability on TMs,
but they are provable within Lean's type theory. These are useful for:
- Ensuring the output encoding doesn't blow up exponentially
- Composing reductions (size bounds compose polynomially)

**Important**: Size bounds are NOT the same as polynomial-time computability.
A function could have polynomial size bounds but require exponential time to
compute, or vice versa. However, for most "natural" reductions used in complexity
theory, polynomial size bounds are a reasonable proxy. -/

/-- A size-bounded reduction: output size is polynomially bounded in input size.

    NOTE: This captures output SIZE bounds, not computation TIME.
    For actual polynomial-time computability, use `Complexity.PolyTime` which
    requires a Turing machine witness. -/
structure SizeBoundedReduction (A B : Type*) [SizeOf A] [SizeOf B] where
  /-- The reduction function -/
  f : A → B
  /-- Output size is polynomially bounded in input size -/
  size_bound : ∃ (c k : ℕ), ∀ a : A, sizeOf (f a) ≤ c * (sizeOf a) ^ k + c

/-- Helper: c * n^k + c ≤ 2c * (n+1)^k for all c, n, k -/
private lemma poly_inner_bound (c n k : ℕ) : c * n ^ k + c ≤ 2 * c * (n + 1) ^ k := by
  have hpow_mono : n ^ k ≤ (n + 1) ^ k := Nat.pow_le_pow_left (by omega) k
  have hpow_ge1 : 1 ≤ (n + 1) ^ k := Nat.one_le_pow k (n + 1) (by omega)
  have heq : c * (n + 1) ^ k + c * (n + 1) ^ k = 2 * c * (n + 1) ^ k := by
    calc
      c * (n + 1) ^ k + c * (n + 1) ^ k
          = (c + c) * (n + 1) ^ k := by rw [Nat.add_mul]
      _ = (2 * c) * (n + 1) ^ k := by rw [Nat.two_mul]
      _ = 2 * c * (n + 1) ^ k := by rw [Nat.mul_assoc]
  have hstep : c * (n + 1) ^ k + c ≤ c * (n + 1) ^ k + c * (n + 1) ^ k := by
    have hc_le : c ≤ c * (n + 1) ^ k := by
      calc
        c = c * 1 := (Nat.mul_one c).symm
        _ ≤ c * (n + 1) ^ k := Nat.mul_le_mul_left c hpow_ge1
    omega
  have hmul_mono : c * n ^ k ≤ c * (n + 1) ^ k := Nat.mul_le_mul_left c hpow_mono
  calc c * n ^ k + c
      ≤ c * (n + 1) ^ k + c := by omega
    _ ≤ c * (n + 1) ^ k + c * (n + 1) ^ k := hstep
    _ = 2 * c * (n + 1) ^ k := heq

/-- Helper lemma: polynomials are closed under composition.
    This is a standard result from algebra: if f(n) ≤ c₁n^k₁ + c₁ and g(m) ≤ c₂m^k₂ + c₂,
    then g(f(n)) ≤ c₂(c₁n^k₁ + c₁)^k₂ + c₂ which is bounded by a polynomial in n.

    We use a generously large constant to simplify the proof. -/
theorem poly_compose_bound (c1 k1 c2 k2 n : ℕ) :
    c2 * (c1 * n ^ k1 + c1) ^ k2 + c2 ≤
    (c1 + c2 + 2) ^ (2 * (k1 + 1) * (k2 + 1) + 1) * n ^ ((k1 + 1) * (k2 + 1)) +
    (c1 + c2 + 2) ^ (2 * (k1 + 1) * (k2 + 1) + 1) := by
  set K := (k1 + 1) * (k2 + 1) with hK_def
  set B := c1 + c2 + 2 with hB_def
  have hB_ge2 : 2 ≤ B := by omega
  have hB_pos : 0 < B := by omega
  have hc1_le : c1 ≤ B := by omega
  have hc2_le : c2 ≤ B := by omega
  have hK_pos : 0 < K := Nat.mul_pos (Nat.succ_pos k1) (Nat.succ_pos k2)
  have hK_ne : K ≠ 0 := Nat.ne_of_gt hK_pos
  -- Key bound: c1 * n^k1 + c1 ≤ 2*c1 * (n + 1)^k1 ≤ B * (n + 1)^k1
  -- Since B = c1 + c2 + 2 ≥ 2*c1 when c2 ≥ c1 - 2, but we can't assume that.
  -- Instead use: c1 * n^k1 + c1 ≤ 2*c1 * (n+1)^k1 ≤ 2*B * (n+1)^k1 ≤ B^2 * (n+1)^k1
  have h2c1_le : 2 * c1 ≤ B ^ 2 := by
    have : c1 ≤ B := hc1_le
    have : 2 ≤ B := hB_ge2
    calc 2 * c1 ≤ B * c1 := Nat.mul_le_mul_right c1 hB_ge2
      _ ≤ B * B := Nat.mul_le_mul_left B hc1_le
      _ = B ^ 2 := (Nat.pow_two B).symm
  have hinner : c1 * n ^ k1 + c1 ≤ B ^ 2 * (n + 1) ^ k1 := by
    calc c1 * n ^ k1 + c1
        ≤ 2 * c1 * (n + 1) ^ k1 := poly_inner_bound c1 n k1
      _ ≤ B ^ 2 * (n + 1) ^ k1 := Nat.mul_le_mul_right _ h2c1_le
  -- (c1 * n^k1 + c1)^k2 ≤ B^(2*k2) * (n + 1)^(k1*k2)
  have hpower : (c1 * n ^ k1 + c1) ^ k2 ≤ B ^ (2 * k2) * (n + 1) ^ (k1 * k2) := by
    calc (c1 * n ^ k1 + c1) ^ k2
        ≤ (B ^ 2 * (n + 1) ^ k1) ^ k2 := Nat.pow_le_pow_left hinner k2
      _ = (B ^ 2) ^ k2 * ((n + 1) ^ k1) ^ k2 := Nat.mul_pow (B ^ 2) _ k2
      _ = B ^ (2 * k2) * (n + 1) ^ (k1 * k2) := by rw [← Nat.pow_mul, ← Nat.pow_mul]
  -- c2 * (...) + c2 ≤ B^(2*k2+1) * (n+1)^(k1*k2) + B
  have hstep1 : c2 * (c1 * n ^ k1 + c1) ^ k2 + c2 ≤ B ^ (2 * k2 + 1) * (n + 1) ^ (k1 * k2) + B := by
    have hc2B2k2 : c2 * B ^ (2 * k2) ≤ B * B ^ (2 * k2) := Nat.mul_le_mul_right _ hc2_le
    have hpow_ge1' : 1 ≤ (n + 1) ^ (k1 * k2) := Nat.one_le_pow (k1*k2) (n+1) (by omega)
    have hBpow_eq : B * B ^ (2 * k2) = B ^ (2 * k2 + 1) := by
      -- B^(n+1) = B^n * B; commute factors to match
      calc B * B ^ (2 * k2) = B ^ (2 * k2) * B := by simp [Nat.mul_comm]
        _ = B ^ (2 * k2 + 1) := by rw [Nat.pow_succ]
    calc c2 * (c1 * n ^ k1 + c1) ^ k2 + c2
        ≤ c2 * (B ^ (2 * k2) * (n + 1) ^ (k1 * k2)) + c2 := by
          have hm : c2 * (c1 * n ^ k1 + c1) ^ k2 ≤ c2 * (B ^ (2 * k2) * (n + 1) ^ (k1 * k2)) :=
            Nat.mul_le_mul_left c2 hpower
          omega
      _ = c2 * B ^ (2 * k2) * (n + 1) ^ (k1 * k2) + c2 := by rw [Nat.mul_assoc]
      _ ≤ B * B ^ (2 * k2) * (n + 1) ^ (k1 * k2) + c2 := by
          have hm : c2 * B ^ (2 * k2) * (n + 1) ^ (k1 * k2) ≤
              B * B ^ (2 * k2) * (n + 1) ^ (k1 * k2) := by
            have hm' : c2 * B ^ (2 * k2) ≤ B * B ^ (2 * k2) := hc2B2k2
            exact Nat.mul_le_mul_right ((n + 1) ^ (k1 * k2)) hm'
          omega
      _ ≤ B * B ^ (2 * k2) * (n + 1) ^ (k1 * k2) + B := by
          omega
      _ = B ^ (2 * k2 + 1) * (n + 1) ^ (k1 * k2) + B := by rw [hBpow_eq]
  -- Key: 2*k2 + 1 ≤ 2*K + 1 and k1*k2 ≤ K
  have hkkK : k1 * k2 ≤ K := by
    -- K = (k1+1)(k2+1) so product is bounded
    simp [hK_def]; nlinarith
  have h2k2K : 2 * k2 + 1 ≤ 2 * K + 1 := by
    simp [hK_def]; nlinarith
  have hB_le_pow : B ≤ B ^ (2 * K + 1) := by
    have hpos : 1 ≤ 2 * K + 1 := by omega
    calc B = B ^ 1 := (Nat.pow_one B).symm
      _ ≤ B ^ (2 * K + 1) := Nat.pow_le_pow_right hB_pos hpos
  by_cases hn0 : n = 0
  · -- n = 0
    subst hn0
    simp only [zero_pow hK_ne, mul_zero, zero_add]
    have hk2lt : k2 < K := by
      have hk2_succ_le : k2 + 1 ≤ K := by
        have hone_le : 1 ≤ k1 + 1 := by omega
        calc
          k2 + 1 = 1 * (k2 + 1) := by ring
          _ ≤ (k1 + 1) * (k2 + 1) := Nat.mul_le_mul_right (k2 + 1) hone_le
          _ = K := by simp [hK_def, Nat.mul_comm]
      exact Nat.lt_of_lt_of_le (Nat.lt_succ_self k2) hk2_succ_le
    have htwo_le : 2 ≤ B ^ (2 * (K - k2)) := by
      have hpos : 1 ≤ 2 * (K - k2) := by
        have : 0 < K - k2 := Nat.sub_pos_of_lt hk2lt
        exact Nat.succ_le_of_lt (Nat.mul_pos (by decide) this)
      have hmono : B ^ 1 ≤ B ^ (2 * (K - k2)) :=
        Nat.pow_le_pow_right hB_pos hpos
      have hBpow : B ≤ B ^ (2 * (K - k2)) := by
        simpa [Nat.pow_one] using hmono
      exact le_trans hB_ge2 hBpow
    have hstep1' : c2 * (c1 * 0 ^ k1 + c1) ^ k2 + c2 ≤ B ^ (2 * k2 + 1) + B := by
      simpa [pow_zero] using hstep1
    calc
      c2 * (c1 * 0 ^ k1 + c1) ^ k2 + c2
          ≤ B ^ (2 * k2 + 1) + B := hstep1'
      _ ≤ B ^ (2 * k2 + 1) + B ^ (2 * k2 + 1) := by
          have hpos : 1 ≤ 2 * k2 + 1 := by omega
          have hBpow : B ≤ B ^ (2 * k2 + 1) := by
            have hmono := Nat.pow_le_pow_right hB_pos hpos
            simpa [Nat.pow_one] using hmono
          omega
      _ = 2 * B ^ (2 * k2 + 1) := (Nat.two_mul _).symm
      _ ≤ B ^ (2 * (K - k2)) * B ^ (2 * k2 + 1) := by
          have := htwo_le
          exact Nat.mul_le_mul_right _ this
      _ = B ^ (2 * K + 1) := by
          have hk2_leK : k2 ≤ K := Nat.le_of_lt_succ (Nat.lt_succ_of_lt hk2lt)
          have hmul_sub : 2 * (K - k2) = 2 * K - 2 * k2 := by
            simpa [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc] using (Nat.mul_sub_left_distrib 2 K k2)
          have hpow :
              2 * (K - k2) + (2 * k2 + 1) = 2 * K + 1 := by
            calc
              2 * (K - k2) + (2 * k2 + 1)
                  = (2 * K - 2 * k2) + (2 * k2 + 1) := by simp [hmul_sub]
              _ = 2 * K + 1 := by
                  have hcancel : 2 * K - 2 * k2 + 2 * k2 = 2 * K :=
                    Nat.sub_add_cancel (Nat.mul_le_mul_left 2 hk2_leK)
                  linarith
          calc
            B ^ (2 * (K - k2)) * B ^ (2 * k2 + 1)
                = B ^ (2 * (K - k2)) * (B ^ (2 * k2) * B) := by simp [Nat.pow_succ]
            _ = B ^ (2 * (K - k2) + 2 * k2 + 1) := by
              calc
                B ^ (2 * (K - k2)) * (B ^ (2 * k2) * B)
                    = (B ^ (2 * (K - k2)) * B ^ (2 * k2)) * B := by ac_rfl
                _ = B ^ (2 * (K - k2) + 2 * k2) * B := by rw [Nat.pow_add]
                _ = B ^ (2 * (K - k2) + 2 * k2 + 1) := by
                      simp [Nat.pow_succ, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
            _ = B ^ (2 * K + 1) := by
              have hcancel2 : 2 * (K - k2) + 2 * k2 = 2 * K := by
                calc
                  2 * (K - k2) + 2 * k2 = 2 * ((K - k2) + k2) := by rw [← Nat.mul_add]
                  _ = 2 * K := by simp [Nat.sub_add_cancel hk2_leK]
              have h : 2 * (K - k2) + 2 * k2 + 1 = 2 * K + 1 := by linarith
              simp [h]
      _ ≤ B ^ (2 * (k1 + 1) * (k2 + 1) + 1) := by
          -- rewrite K back to its definition
          simp [hK_def, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]
  · -- n ≥ 1
    have hn_pos : 1 ≤ n := Nat.one_le_iff_ne_zero.mpr hn0
    -- (n+1)^(k1*k2) ≤ 2^(k1*k2) * n^(k1*k2) ≤ B^(k1*k2) * n^(k1*k2)
    have hnp1_bound : (n + 1) ^ (k1 * k2) ≤ B ^ (k1 * k2) * n ^ (k1 * k2) := by
      have hnp1 : n + 1 ≤ 2 * n := by omega
      calc
        (n + 1) ^ (k1 * k2) ≤ (2 * n) ^ (k1 * k2) := Nat.pow_le_pow_left hnp1 _
        _ = 2 ^ (k1 * k2) * n ^ (k1 * k2) := Nat.mul_pow 2 n _
        _ ≤ B ^ (k1 * k2) * n ^ (k1 * k2) := by
          apply Nat.mul_le_mul_right
          exact Nat.pow_le_pow_left hB_ge2 _
    have hBexp_le : B ^ (2 * k2 + 1 + k1 * k2) ≤ B ^ (2 * K + 1) := by
      apply Nat.pow_le_pow_right hB_pos
      have : 2 * k2 + 1 + k1 * k2 ≤ 2 * K + 1 := by nlinarith [hK_def]
      exact this
    have hnpow_le : n ^ (k1 * k2) ≤ n ^ K := Nat.pow_le_pow_right hn_pos (by nlinarith [hK_def])
    calc c2 * (c1 * n ^ k1 + c1) ^ k2 + c2
        ≤ B ^ (2 * k2 + 1) * (n + 1) ^ (k1 * k2) + B := hstep1
      _ ≤ B ^ (2 * k2 + 1) * (B ^ (k1 * k2) * n ^ (k1 * k2)) + B := by
          have hm : B ^ (2 * k2 + 1) * (n + 1) ^ (k1 * k2) ≤
              B ^ (2 * k2 + 1) * (B ^ (k1 * k2) * n ^ (k1 * k2)) :=
            Nat.mul_le_mul_left (B ^ (2 * k2 + 1)) hnp1_bound
          omega
      _ = B ^ (2 * k2 + 1 + k1 * k2) * n ^ (k1 * k2) + B := by
          -- combine the B powers
          simp [Nat.mul_assoc, Nat.pow_add, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc, Nat.mul_add]
      _ ≤ B ^ (2 * K + 1) * n ^ (k1 * k2) + B := by
          have hm : B ^ (2 * k2 + 1 + k1 * k2) * n ^ (k1 * k2) ≤
              B ^ (2 * K + 1) * n ^ (k1 * k2) :=
            Nat.mul_le_mul_right (n ^ (k1 * k2)) hBexp_le
          omega
      _ ≤ B ^ (2 * K + 1) * n ^ K + B := by
          have h := Nat.mul_le_mul_left (B ^ (2 * K + 1)) hnpow_le
          omega
      _ ≤ B ^ (2 * K + 1) * n ^ K + B ^ (2 * K + 1) := by
          omega
      _ ≤ B ^ (2 * (k1 + 1) * (k2 + 1) + 1) * n ^ K + B ^ (2 * (k1 + 1) * (k2 + 1) + 1) := by
          simp [hK_def, Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]

/-- Size-bounded reductions compose: composition preserves polynomial size bounds.

    NOTE: This is about SIZE bounds, not computation TIME. The analogous result
    for polynomial-time TM computability requires proving TM composition, which
    is significantly harder in Lean. -/
theorem SizeBoundedReduction.comp_exists {A B C : Type*} [SizeOf A] [SizeOf B] [SizeOf C]
    (r1 : SizeBoundedReduction A B) (r2 : SizeBoundedReduction B C) :
    ∃ (r : SizeBoundedReduction A C), r.f = r2.f ∘ r1.f := by
  obtain ⟨c1, k1, h1⟩ := r1.size_bound
  obtain ⟨c2, k2, h2⟩ := r2.size_bound
  refine ⟨⟨r2.f ∘ r1.f, ?_⟩, rfl⟩
  -- Use the larger constants from poly_compose_bound
  use (c1 + c2 + 2) ^ (2 * (k1 + 1) * (k2 + 1) + 1), (k1 + 1) * (k2 + 1)
  intro a
  calc sizeOf (r2.f (r1.f a))
      ≤ c2 * sizeOf (r1.f a) ^ k2 + c2 := h2 (r1.f a)
    _ ≤ c2 * (c1 * sizeOf a ^ k1 + c1) ^ k2 + c2 := by
        have hpow : sizeOf (r1.f a) ^ k2 ≤ (c1 * sizeOf a ^ k1 + c1) ^ k2 :=
          Nat.pow_le_pow_left (h1 a) k2
        have hmul : c2 * sizeOf (r1.f a) ^ k2 ≤ c2 * (c1 * sizeOf a ^ k1 + c1) ^ k2 :=
          Nat.mul_le_mul_left c2 hpow
        omega
    _ ≤ _ := poly_compose_bound c1 k1 c2 k2 (sizeOf a)

/-- Identity has polynomial size bounds -/
def SizeBoundedReduction.id (A : Type*) [SizeOf A] : SizeBoundedReduction A A where
  f := fun a => a
  size_bound := ⟨2, 1, fun a => by simp only [ge_iff_le, pow_one]; omega⟩

/-! ## Reduction from Sufficiency to Set Comparison

The sufficiency-checking problem reduces to comparing sets of optimal actions. -/

/-- The sufficiency checking problem instance -/
structure SufficiencyInstance (A S : Type*) where
  /-- The decision problem -/
  dp : DecisionProblem A S
  /-- The coordinate set to check -/
  coords : Set S
  /-- The equivalence relation induced by coords -/
  equiv : S → S → Prop

/-- The set comparison problem instance -/
structure SetComparisonInstance (A S : Type*) where
  /-- Pairs of sets to compare -/
  pairs : List (Set A × Set A)

/-- Reduction from sufficiency to set comparison (noncomputable due to Finset.toList) -/
noncomputable def sufficiencyToSetComparison {A S : Type*} [DecidableEq S] [Fintype S]
    (inst : SufficiencyInstance A S) : SetComparisonInstance A S where
  pairs := (Fintype.elems.toList.product Fintype.elems.toList).map
    fun (s, s') => (inst.dp.Opt s, inst.dp.Opt s')

/-! ## Connection to Mathlib's Computability

We connect our reduction correctness to Mathlib's reduction framework. -/

/-- A size-bounded reduction with correctness implies a many-one reduction exists
    (the full Mathlib ≤₁ requires Computable and Injective which need more setup) -/
theorem size_bounded_reduction_implies_many_one_exists {A B : Type*} [SizeOf A] [SizeOf B]
    (pA : A → Prop) (pB : B → Prop)
    (r : SizeBoundedReduction A B)
    (h : ∀ a, pA a ↔ pB (r.f a)) :
    ∃ f : A → B, ∀ a, pA a ↔ pB (f a) :=
  ⟨r.f, h⟩

/-! ## Complexity Classes

Abstract definitions connecting to complexity theory.

NOTE: The `InP` definition below uses step counting on a `Counted` monad,
which is an abstract model of computation. This is different from (and simpler than)
full Turing machine polynomial time as defined in `Complexity.PolyTime`. -/

/-- A problem is in P (using abstract step-counting model).

    NOTE: This uses the `Counted` monad for step counting, which is an
    abstract computational model. For the TM-based definition, see
    `Complexity.P` in Complexity.lean. -/
def InP {α : Type*} [SizeOf α] (p : α → Prop) : Prop :=
  ∃ (decide : α → Counted Bool) (c k : ℕ),
    (∀ a, (decide a).steps ≤ c * (sizeOf a) ^ k + c) ∧
    (∀ a, (decide a).result = true ↔ p a)

/-- Size-bounded reductions preserve membership in P (abstract step-counting model).

    NOTE: This theorem uses SIZE bounds as a proxy for TIME bounds. For natural
    reductions this is reasonable, but it's not formally equivalent to TM polytime.
    The theorem shows: if f has polynomial output SIZE and pB is in P, then pA is in P.

    The reasoning is: if computing f produces polynomial-sized output, and the decision
    algorithm for pB runs in time polynomial in its input size, then composing them
    gives polynomial time overall. This is sound but informal. -/
theorem size_bounded_reduction_preserves_P {A B : Type*} [SizeOf A] [SizeOf B]
    (pA : A → Prop) (pB : B → Prop)
    (r : SizeBoundedReduction A B)
    (h : ∀ a, pA a ↔ pB (r.f a))
    (hB : InP pB) : InP pA := by
  obtain ⟨decideB, c, k, htime, hcorrect⟩ := hB
  obtain ⟨cr, kr, hr⟩ := r.size_bound
  use fun a => decideB (r.f a)
  -- The composed algorithm is polynomial; use the generous bound from poly_compose_bound
  use (cr + c + 2) ^ (2 * (kr + 1) * (k + 1) + 1), (kr + 1) * (k + 1)
  constructor
  · intro a
    have hsteps := htime (r.f a)
    have hsize := hr a
    calc (decideB (r.f a)).steps
        ≤ c * (sizeOf (r.f a)) ^ k + c := hsteps
      _ ≤ c * (cr * (sizeOf a) ^ kr + cr) ^ k + c := by
          have hpow : (sizeOf (r.f a)) ^ k ≤ (cr * (sizeOf a) ^ kr + cr) ^ k :=
            Nat.pow_le_pow_left hsize _
          have hmul : c * (sizeOf (r.f a)) ^ k ≤ c * (cr * (sizeOf a) ^ kr + cr) ^ k :=
            Nat.mul_le_mul_left _ hpow
          omega
      _ ≤ (cr + c + 2) ^ (2 * (kr + 1) * (k + 1) + 1) * (sizeOf a) ^ ((kr + 1) * (k + 1)) +
          (cr + c + 2) ^ (2 * (kr + 1) * (k + 1) + 1) := by
            exact poly_compose_bound cr kr c k (sizeOf a)
  · intro a
    calc
      (decideB (r.f a)).result = true
          ↔ pB (r.f a) := hcorrect (r.f a)
      _ ↔ pA a := (h a).symm

end DecisionQuotient
