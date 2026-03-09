# MultiFact Lean Proof Roadmap

**Last updated**: 2026-03-09
**Current commit**: `b2bc608d` - PSD asymptotic upper theory
**File**: `docs/papers/paper2_ssot/proofs/Ssot/MultiFact.lean` (4069 lines, no sorries)

---

## Current State

### Completed Theorems

1. **PSD Upper Formulations** (isomorphic):
   - `graphThetaLogUpper G` - inf of log bounds from `ThetaWitness`
   - `graphThetaMatrixLogUpper G` - from `ThetaMatrixFeasible`
   - `graphThetaSDPLogUpper G` - from `ThetaSDPFeasible`
   - `graphThetaPSDLogUpper G` - from `ThetaPSDFeasible` (pure PSD, no Gram factorization)
   - All equal: `graphThetaPSDLogUpper = graphThetaSDPLogUpper = graphThetaMatrixLogUpper = graphThetaLogUpper`

2. **Asymptotic Upper Sequence**:
   - `graphThetaPSDLogSeq G N = graphThetaPSDLogUpper (strongPow G N)`
   - `graphThetaPSDLogSeq_subadditive`: Subadditive sequence (uses `strongPow_addIsoStrongProd`)
   - `graphThetaPSDAsymptoticUpper G`: Fekete limit of the subadditive sequence
   - `tendsto_graphThetaPSDPowerRateNat_graphThetaPSDAsymptoticUpper`: Convergence theorem
   - `iInf_graphThetaPSDPowerRate_eq_graphThetaPSDAsymptoticUpper`: Exact infimum characterization
   - `graphThetaPSDAsymptoticUpper_le_graphThetaPSDLogUpper`: One-shot comparison (Fekete gives ≤)

3. **Shannon Capacity**:
   - `graphShannonCapacityReal G`: Fekete limit of independence growth rates
   - `graphShannonCapacityReal_le_graphThetaLogUpper`: Capacity ≤ theta upper (one-shot)
   - `graphShannonCapacityReal_le_graphThetaPSDLogUpper`: Same via equality chain

4. **Key Isomorphisms**:
   - `blockConfusabilityGraph_eq_strongPow`: Block confusability = strong power of base graph
   - `strongPow_addIsoStrongProd G M N`: `strongPow G (M+N) ≃g strongProd (strongPow G M) (strongPow G N)`

---

## Next Target: Multiplicativity → Asymptotic Collapse

### Goal
Prove that `graphThetaPSDAsymptoticUpper = graphThetaPSDLogUpper`, meaning the asymptotic bound collapses to the one-shot bound. This is the key property that makes Lovász theta computationally tractable.

### Key Lemma Needed
**Multiplicativity on strong product**:
```
graphThetaLogUpper (strongProd G H) = graphThetaLogUpper G + graphThetaLogUpper H
```

Currently have: `≤` direction (subadditivity from `ThetaWitness.prod`)
Missing: `≥` direction (reverse inequality)

### Proof Strategy for Reverse Inequality

**Approach**: For any `ε > 0`, construct near-optimal witnesses for G and H from a near-optimal witness for `strongProd G H`.

1. Given: `WGH : ThetaWitness (strongProd G H)` with `Real.log WGH.bound < graphThetaLogUpper (strongProd G H) + ε`
2. Extract from `WGH`:
   - `WG : ThetaWitness G` with `Real.log WG.bound ≤ Real.log WGH.bound / 2 + ε`
   - `WH : ThetaWitness H` with `Real.log WH.bound ≤ Real.log WGH.bound / 2 + ε`
3. This gives: `graphThetaLogUpper G + graphThetaLogUpper H ≤ Real.log WGH.bound + 2ε`
4. Take inf over witnesses → reverse inequality

**Technical challenge**: The extraction requires projecting the product representation onto factor representations. May need:
- Marginalization of orthonormal representations
- Or: use the PSD matrix formulation and Schur complement / projection techniques

### Alternative: Direct Multiplicativity via PSD

The PSD formulation may be cleaner:
- `ThetaPSDFeasible` has matrix `M : Matrix (Option V) (Option V) ℝ`
- For product, can use Kronecker product structure
- `M_G ⊗ M_H` gives feasible object for `G ⊠ H` with `bound = bound_G * bound_H`

**Key theorem to prove**:
```
theorem graphThetaPSDLogUpper_strongProd_eq
    (G : SimpleGraph α) (H : SimpleGraph γ) :
    graphThetaPSDLogUpper (strongProd G H) = 
      graphThetaPSDLogUpper G + graphThetaPSDLogUpper H
```

---

## Implementation Plan

### Pass 1: Multiplicativity Infrastructure

1. **Tensor product of PSD matrices**:
   ```lean
   def ThetaPSDFeasible.tensor {G H} (TG : ThetaPSDFeasible G) (TH : ThetaPSDFeasible H) :
     ThetaPSDFeasible (strongProd G H)
   ```
   - Matrix: `M = TG.M ⊗ TH.M` (Kronecker product on `Option (α × γ)`)
   - Bound: `bound = TG.bound * TH.bound`
   - Prove PSD via Kronecker product properties

2. **Upper bound from factors**:
   ```lean
   theorem graphThetaPSDLogUpper_strongProd_le_add :
     graphThetaPSDLogUpper (strongProd G H) ≤ 
       graphThetaPSDLogUpper G + graphThetaPSDLogUpper H
   ```
   (Already have this via witness formulation)

3. **Lower bound from product to factors**:
   - Given `TGH : ThetaPSDFeasible (strongProd G H)`, extract `TG`, `TH`
   - Use principal submatrices or projection
   - Show bounds satisfy `log(TG.bound) + log(TH.bound) ≤ log(TGH.bound) + ε`

### Pass 2: Asymptotic Collapse

With multiplicativity:
```lean
theorem graphThetaPSDLogSeq_mul :
    graphThetaPSDLogSeq G (M * N) = graphThetaPSDLogSeq G M + graphThetaPSDLogSeq G N
```

This implies the subadditive sequence is actually *additive* on multiples, which forces:
```lean
theorem graphThetaPSDAsymptoticUpper_eq_graphThetaPSDLogUpper :
    graphThetaPSDAsymptoticUpper G = graphThetaPSDLogUpper G
```

### Pass 3: Complete Shannon Capacity Sandwich

Final chain:
```
graphShannonCapacityReal G 
  ≤ graphThetaPSDAsymptoticUpper G 
  = graphThetaPSDLogUpper G
  ≤ log (chromaticNumber (Gᶜ))
```

---

## Key Definitions Reference

```lean
-- Core graph structures
def strongProd (G : SimpleGraph α) (H : SimpleGraph β) : SimpleGraph (α × β)
def strongPow (G : SimpleGraph α) (N : Nat) : SimpleGraph (Fin N → α)

-- Theta formulations
structure ThetaWitness (G : SimpleGraph V) where
  repr : V → EuclideanSpace ℝ β
  handle : EuclideanSpace ℝ β
  repr_unit : ∀ v, ‖repr v‖ = 1
  handle_unit : ‖handle‖ = 1
  ortho_of_nonadj : ∀ v w, v ≠ w → ¬G.Adj v w → inner (repr v) (repr w) = 0
  bound : ℝ
  one_le_bound : 1 ≤ bound
  one_le_bound_inner_sq : ∀ v, 1 ≤ bound * (inner (repr v) handle)^2

structure ThetaPSDFeasible (G : SimpleGraph V) where
  M : Matrix (Option V) (Option V) ℝ
  posSemidef : M.PosSemidef
  diag_none : M none none = 1
  diag_some : ∀ v, M (some v) (some v) = 1
  zero_of_nonadj : ∀ v w, v ≠ w → ¬G.Adj v w → M (some v) (some w) = 0
  bound : ℝ
  one_le_bound : 1 ≤ bound
  one_le_bound_entry_sq : ∀ v, 1 ≤ bound * (M none (some v))^2

-- Upper bounds
noncomputable def graphThetaLogUpper (G : SimpleGraph α) : ℝ :=
  sInf { r | ∃ W : ThetaWitness G, r = Real.log W.bound }

noncomputable def graphThetaPSDLogUpper (G : SimpleGraph α) : ℝ :=
  sInf { r | ∃ T : ThetaPSDFeasible G, r = Real.log T.bound }

-- Asymptotic
noncomputable def graphThetaPSDLogSeq (G : SimpleGraph α) (N : Nat) : ℝ :=
  graphThetaPSDLogUpper (strongPow G N)

noncomputable def graphThetaPSDAsymptoticUpper (G : SimpleGraph α) : ℝ :=
  (graphThetaPSDLogSeq_subadditive G).lim  -- Fekete limit
```

---

## Build Commands

```bash
cd docs/papers/paper2_ssot/proofs
lake build Ssot.MultiFact
```

---

## Paper Context

This is for Paper 2 (Information Theory submission) on zero-error thresholds in multi-location encodings. The Lovász theta bounds establish SDP-computable upper bounds on Shannon capacity, connecting to the main contribution on exact consistency rates.

Key references:
- Lovász 1979: θ(G) bounds Shannon capacity
- Strong product ⊠ and strong powers for asymptotic capacity
- PSD formulation makes the bound computationally accessible (SDP)
