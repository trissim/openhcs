# Plan: Additional Tractable Subcases for Paper 4

**CRITERIA FOR INCLUSION:**
- Must be mathematically correct (not just plausible)
- Must give a real polynomial-time algorithm for sufficiency checking
- Must be explainable in 1-2 sentences in paper
- Mechanization must add to the artifact's theorem count

---

## PHASE 1: EXECUTE - Paper-Strength Additions

### 1. Multiplicative Separable (Sign-Constant)

**File:** `Tractability/MultiplicativeSeparable.lean`

**Restriction:** `U(a,s) = f(a) · g(s)` with constant sign on g(s)

**Theorems needed:**
- `opt_eq_argmax_f_of_pos_factor`: If g(s) > 0 for all s, then Opt(s) = argmax_a f(a) for all s
- `opt_eq_argmin_f_of_neg_factor`: If g(s) < 0 for all s, then Opt(s) = argmin_a f(a) for all s  
- `opt_all_tie_of_zero_factor`: If g(s) = 0 for all s, then all actions tie
- `empty_sufficient_of_sign_constant_factor`: Combined theorem

**Complexity:** O(1) — empty set is sufficient

**Lean Strategy:**
```lean
structure MultiplicativeSeparable (dp : FiniteDecisionProblem) where
  actionFactor : A → ℝ
  stateFactor : S → ℝ
  pos_sign : ∀ s, stateFactor s > 0  -- or neg_sign, or zero_sign
  utility_eq : ∀ a s, dp.utility a s = actionFactor a * stateFactor s
```

---

### 2. Strict Dominance / Constant Optimal Set

**File:** `Tractability/Dominance.lean`

**Restriction A (Strict Global Dominance):** ∃a* such that U(a*, s) > U(a, s) for all s, all a ≠ a*

**Theorems:**
- `strict_global_dominance_opt_singleton`: If strict dominance, Opt(s) = {a*} for all s
- `empty_sufficient_of_strict_dominance`: Therefore ∅ is sufficient

**Restriction B (Constant Optimal Set):** ∃D ⊆ A such that Opt(s) = D for all s

**Theorems:**
- `empty_sufficient_of_constant_optset`: If Opt is constant, ∅ is sufficient

**Complexity:** O(|A|·|S|) to verify dominance, then O(1) sufficiency check

**Lean Strategy:**
```lean
structure StrictGlobalDominance (dp : FiniteDecisionProblem) where
  dominant : A
  prove_strict : ∀ s a, a ≠ dominant → dp.utility dominant s > dp.utility a s
```

---

### 3. Additive-Linear as Corollary

**NOT a new island** — add as short corollary in existing separable file

**Theorem:** If U(a,s) = w·s + f(a), then Opt(s) = argmax_a f(a) for all s

**Proof:** The w·s term cancels in argmax (same for all actions), so all coordinates are irrelevant

**Lean placement:** Add to `Tractability/SeparableUtility.lean` as corollary

```lean
-- In SeparableUtility.lean, add:
theorem additive_linear_as_separable {A S n} [CoordinateSpace S n]
    (dp : FiniteDecisionProblem (A := A) (S := S))
    (f : A → ℝ) (w : S → ℝ) :
    (∀ a s, dp.utility a s = w s + f a) →
    ∀ s, dp.optimalActions s = dp.optimalActions (Classical.arbitrary S)
```

---

## PHASE 2: Artifact Warmups (Appendix/Optional)

### 4. Single Action

**File:** `Tractability/SingleAction.lean`

**Restriction:** |A| = 1

**Theorem:** If only one action, any I is sufficient (trivial)

**Complexity:** O(1)

**Note:** Too trivial for main paper — use as warmup or skip

---

### 5. Bounded State Space

**File:** `Tractability/BoundedStateSpace.lean` (already created)

**Restriction:** |S| ≤ k for constant k

**Theorem:** Brute-force check is O(k²) = O(1)

**Note:** More of a size-bound observation than structural case — appendix material

---

## PHASE 3: CUT (Do Not Execute As Written)

| Case | Reason for Cut |
|------|----------------|
| Binary/singleton domains | Does not make sufficiency polynomial |
| Monotone utility | Crossings can happen in interior — check-all-pairs not avoided |
| Independent coordinates | Sum across coordinates — doesn't reduce to per-coordinate check |
| Lexicographic utility | "No ties" is not sufficient structural condition |
| Bounded relevant coordinates | Promise condition, not verifiable input property |
| Deterministic mapping | Just re-encodes original problem |
| Idempotent utility | Thresholding changes optimizer — not preservation |

---

## IMPLEMENTATION ORDER

1. ✅ BoundedStateSpace.lean (done)
2. MultiplicativeSeparable.lean (Phase 1)
3. Dominance.lean (Phase 1)  
4. Add additive-linear corollary to SeparableUtility.lean (Phase 1)
5. SingleAction.lean (Phase 2, optional)
6. Stop and reassess

---

## SUMMARY: What Actually Gets Added

| File | Complexity | Paper Slot? |
|------|------------|--------------|
| MultiplicativeSeparable.lean | O(1) | Yes |
| Dominance.lean | O(\|A\|·\|S\|) → O(1) | Yes |
| SeparableUtility.lean (corollary) | O(1) | Yes (corollary) |
| SingleAction.lean | O(1) | Optional |
| BoundedStateSpace.lean | O(k²) | Appendix only |
