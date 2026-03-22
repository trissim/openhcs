# Native Basin Proof: Tractability Assessment

## What "proving the native basin" means

We want to prove: *starting from any sampled initial pose, the optimizer converges to the native binding mode.*

This splits into three sub-problems:

1. **Coverage**: At least one initial pose falls within the native basin.
2. **Descent**: Once inside the basin, the optimizer doesn't escape it.
3. **Uniqueness**: The native pose is the unique local minimum (no 180°-flip trap).

The current Lean infrastructure is strong on (2) — it proves step-level correctness exhaustively. It has nothing for (1) or (3).

---

## What the existing proofs cover

### Step-level correctness — `FormalLocalOptimizer.lean`
Proves that given scores `uExact`, `uCoarse` with `|uExact(a) - uCoarse(a)| ≤ δ`, the optimizer picks an action from the certified survivor set (top-k under exact scores). This is proved for every single step.

**Gap**: Says nothing about which starting poses lead to the native — only that each step picks correctly given its current position.

### Lipschitz step bounds — `LipschitzStepBounds.lean`
Has `descent_bounded_change`: with step size s and Lipschitz constant L, `|f_after - f_before| ≤ L*s`. Also has `n_step_error_bound` (trivially).

Notably the comments acknowledge the implemented steps (0.5Å, π/12 rad) are **20–100× larger than the LJ theoretical minimum**, explicitly relying on stochastic sampling to compensate. The proof doesn't certify convergence from arbitrary starting points — just per-step energy change.

**Gap**: No Lyapunov function. No basin-of-attraction theorem. No statement that the optimizer monotonically descends.

### Score approximation — `DirectionalHBondApproximation.lean`, `PiStackingApproximation.lean`
Proves the coarse scoring approximates the exact scoring within δ (Lipschitz bound). This enables the pruning certificate.

**Gap**: Says nothing about the landscape topology — whether the native basin is larger or smaller than the 180°-flip basin.

---

## What's missing for a native basin proof

### Missing piece 1: Anti-degeneracy under 180° flip

The root cause of the current failure is that the pi-stacking score uses `|cos θ|`, which is symmetric under 180° rotation. The current Lean infrastructure actually contains the relevant equation in `PiStackingApproximation.lean`:

```lean
-- piStackingScore (strengths * radial) face offset
-- where face_alignment = |cos θ|  ← THIS is symmetric under θ → θ+180°
```

To prove the 180°-flipped pose has strictly higher energy, we would need:

```lean
theorem flip_raises_energy
    (native_hbond_score flipped_hbond_score : ℝ)
    (h_directional : native_hbond_score > flipped_hbond_score)
    (h_pistack_equal : native_pistack_score = flipped_pistack_score) :
    native_total_score < flipped_total_score
```

This requires proving H-bond scoring is strictly antisymmetric under 180° rotation, which it provably is:

```lean
-- In DirectionalHBondApproximation.lean:
-- directionalHBondScore r d a = r * d * a
-- where d = face_alignment = cos(θ), NOT |cos θ|
-- So d(θ+180°) = cos(θ+180°) = -cos(θ) = -d(θ)
-- When d < 0 (wrong orientation), score is negative → penalized
```

The H-bond face term is `cos(θ)` (signed), so it IS antisymmetric under flip. **If the H-bond term is active** (non-zero strengths), the 180°-flipped pose gets penalized. The proof is structurally available.

**Why 1hk4 and 2ceq still fail**: These compounds have weak H-bond interactions (thyroxine is primarily hydrophobic; 2ceq is small with limited H-bond donors). The H-bond term has near-zero strengths → the flip penalty is near zero → the pi-stacking symmetry dominates.

So the needed Lean theorem is compound-conditional:

```lean
theorem flip_distinguishable_if_hbond_active
    (hbond_strength : ℝ) (h_active : hbond_strength > ε_threshold) :
    native_total_score < flipped_total_score
```

**Tractability**: Moderately feasible. The `DirectionalHBondApproximation.lean` already has the face alignment structure. Need ~50 lines of new Lean connecting the signed face term to a flip penalty lower bound.

---

### Missing piece 2: Basin coverage by the sampler

We need: *n sampled poses include at least one within the native basin radius R.*

This is a covering problem in SE(3) (rigid body space, 6D). The current Lean has no coverage analysis.

**What's needed**:
- A `CoverageTheorem` that the global sampler generates a δ-cover of the docking box with probability ≥ 1 - p_fail
- OR a deterministic version: for n poses, the expected minimum distance to the native is bounded

**Tractability**: Hard. SE(3) coverage analysis requires integrating over rotational and translational degrees of freedom. The `Tractability/Dimensional.lean` and `BoundedStateSpace.lean` might have relevant infrastructure, but this would likely be 200-500 lines of new Lean on rotation groups.

---

### Missing piece 3: Descent guarantee (Lyapunov)

We need: *if a pose is within basin radius R of the native, the optimizer drives it toward the native.*

The existing `LipschitzStepBounds.descent_bounded_change` bounds per-step energy change but doesn't guarantee monotone descent.

**What's needed**:
- A Lyapunov function V(pose) that decreases under every optimizer step
- OR a weaker "eventual convergence" theorem: within T rounds, the pose reaches within ε of the native

**Tractability**: Very hard for the general case. For the specific case of LJ scoring (radially symmetric, no orientation-dependent local minima), a Lyapunov argument is feasible. For EXTENDED_RICH scoring with pi-stacking, it requires the anti-degeneracy theorem from piece 1.

---

## Summary: Distance from a tractable proof

| Sub-problem | Status | Lines of new Lean needed | Timeline |
|-------------|--------|--------------------------|----------|
| Flip anti-degeneracy (H-bond active) | Structurally feasible | ~50–100 | Near-term |
| Flip degeneracy when H-bond inactive | **Not provable** — the flip IS degenerate by construction for purely hydrophobic ligands | n/a | Never (requires H-bond) |
| Basin coverage by sampler | Hard | ~200–500 | Medium-term |
| Lyapunov descent (LJ) | Feasible for LJ | ~100–200 | Medium-term |
| Lyapunov descent (EXTENDED_RICH) | Requires anti-degeneracy first | ~300+ | Long-term |
| Full native basin proof | Requires all above | ~600–1000+ | Long-term |

---

## Near-term recommendation

The native basin proof is not tractable in the near term for the general case. The 180°-flip failure specifically happens for **hydrophobic ligands with weak H-bond interactions**, where the pi-stacking symmetry creates a degenerate landscape.

The formally cleanest near-term fix is:

**Restore LJ-only optimization, use EXTENDED_RICH for ranking only.**

This is what the committed code at `656a108a` intended — the `if False:` gate was deliberate ("Temporarily disabled for speed test"). LJ has no orientation-dependent local minima for symmetric ligands, so the optimizer reliably finds the native basin. The Lean proof for LJ convergence is much more tractable than for EXTENDED_RICH.

Then separately, we can add a Lean theorem:

```lean
theorem hbond_flip_penalty
    (receptor_direction ligand_donor : ℝ³)
    (h_active : hbond_strength > 0) :
    hbondScore (flip receptor_direction) ligand_donor
      < hbondScore receptor_direction ligand_donor
```

This would be the first step toward the full anti-degeneracy proof, and it's tight, targeted, and feasible to write now.
