# MolecularSrank.lean Completion Plan

## Overview

The `MolecularSrank.lean` file currently compiles successfully but contains 10 `sorry` placeholder proofs that need to be completed. This plan outlines the exact changes needed, organized by priority and difficulty.

**File Location:** `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/MolecularSrank.lean`

**Current Status:** ✅ Compiles with warnings (6 sorry statements + 2 unused variable warnings)

---

## Critical Structural Change: No Axioms

### ❌ What NOT to Do

Do NOT use the `axiom` keyword for the cutoff-respecting property. This would introduce non-standard axioms into Lean's core logic, which reviewers may reject.

### ✅ What TO Do Instead

**Add `utility_local` as a required field to `MDBindingProblem`:**

```lean
structure MDBindingProblem where
  protein : Molecule
  ligand : Molecule
  bindingSite : BindingSite
  cutoff : ℝ  -- Interaction cutoff radius (e.g., 10Å)
  utility : MDAction → MDState → ℝ  -- Binding affinity function
  /-- The "Physical Law" of this specific problem:
      Utility only changes if atoms within the cutoff move. -/
  utility_local : ∀ (s1 s2 : MDState),
    (∀ (atom : Atom), atom ∈ protein.atoms →
      atomWithinCutoff atom bindingSite cutoff →
      (atomStateProj s1 atom = atomStateProj s2 atom)) →
    (∀ (p : Fin 6), ligandPoseProj s1 p = ligandPoseProj s2 p) →
    utility s1 = utility s2
```

**Why This Approach is Superior:**

1. **No Non-Standard Axioms**: Uses only standard Lean logic (Choice, Propext, Quotient)
2. **Scientifically Stronger**: Defines the exact conditions under which the speedup is guaranteed
3. **Type-Safe**: The compiler enforces that only valid problems can use the theorems
4. **Reviewer-Friendly**: Makes explicit what physical assumptions are being made

**Paper Wording:**
> "Our engine requires a locality constraint: utility functions must be local with respect to the cutoff radius. This is enforced at the type level—only cutoff-respecting potentials can use our accelerated algorithms."

---

## Tier 1: Critical Proofs (Blocking Everything)

### 1.1 Add Helper Projections

**Location:** After the `MDBindingProblem` structure (around line 89)

```lean
/-- Project MD state to a specific atom's position -/
def atomStateProj (s : MDState) (atom : Atom) : ℝ × ℝ × ℝ :=
  -- Find atom in state's protein list and return its position
  match s.protein.find? (fun a => a.index = atom.index) with
  | some a => a.position
  | none => atom.position  -- Fallback to atom's stored position

/-- Project MD state to a ligand pose parameter -/
def ligandPoseProj (s : MDState) (p : Fin 6) : ℝ :=
  match p.val with
  | 0 => s.ligand.position.1
  | 1 => s.ligand.position.2.1
  | 2 => s.ligand.position.2.2
  | 3 => s.ligand.rotation.1
  | 4 => s.ligand.rotation.2.1
  | 5 => s.ligand.rotation.2.2
  | _ => 0
```

### 1.2 Complete `md_relevance_criterion` (Lines 166-212)

**Current State:** Has `sorry` at lines 208, 212

**Strategy:**

#### → Direction (Relevant → Within Cutoff)

```lean
· intro h_relevant
  by_contra h_outside
  -- h_relevant: ∃ s s', differ only in coord i, Opt(s) ≠ Opt(s')
  -- h_outside: atom is outside cutoff
  -- Contradiction: changing outside-cutoff atom can't change utility
  rcases h_relevant with ⟨s, s', h_diff, h_opt_ne⟩
  -- Show utility is constant
  have h_utility_eq : prob.utility (someAction) s = prob.utility (someAction) s' := by
    apply prob.utility_local
    · intro atom hin hwithin
      -- All within-cutoff atoms are the same
      sorry
    · intro p
      -- Pose parameters are the same (we only changed atom position)
      sorry
  -- But different utility implies different Opt
  have h_opt_eq : prob.Opt s = prob.Opt s' := by
    -- This follows from utility equality
    sorry
  -- Contradiction with h_opt_ne
  contradiction
```

#### ← Direction (Within Cutoff → Relevant)

```lean
· intro h_inside
  by_contra h_irrelevant
  -- h_inside: atom is within cutoff
  -- h_irrelevant: for all s,s', if they differ only in coord i, Opt(s) = Opt(s')
  -- Contradiction: we can construct s,s' where utility changes
  -- Construct two states that differ only in this atom's position
  let s : MDState := {
    protein := prob.protein.atoms,
    ligand := basePose,
    solvent := []
  }
  let s' : MDState := {
    protein := perturbAtomPosition s.protein atom 0.1,
    ligand := basePose,
    solvent := []
  }
  have h_utility_ne : prob.utility (someAction) s ≠ prob.utility (someAction) s' := by
    -- Within cutoff → can perturb to change energy
    sorry
  -- Different utility implies different Opt
  have h_opt_ne : prob.Opt s ≠ prob.Opt s' := by
    sorry
  -- But irrelevance says they must be equal
  have h_opt_eq := (not_relevant_iff _ _).1 h_irrelevant s s' (by sorry)
  contradiction
```

**Dependencies:**
- Need `perturbAtomPosition` helper function
- Need to connect utility differences to Opt differences

### 1.3 Complete `md_srank_bound` (Line 239)

**Strategy:**

```lean
theorem md_srank_bound (prob : MDBindingProblem) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) (prob.toDecisionProblem) ≤ 3 * numRelevantAtoms prob + 6 := by
  unfold DecisionProblem.srank
  -- Partition coordinates into three sets:
  -- 1. Relevant protein atoms (3 coords each)
  -- 2. Irrelevant protein atoms (outside cutoff)
  -- 3. Ligand pose parameters (6, always relevant)

  let relevantProteinCoords : Finset (Fin (numMDCoordinates prob)) :=
    {i : Fin (numMDCoordinates prob) | ∃ atom idx,
      atom ∈ prob.protein.atoms ∧
      atom.index = idx ∧
      atomWithinCutoff atom prob.bindingSite prob.cutoff ∧
      i.val < 3 * prob.protein.numAtoms ∧
      i.val / 3 = idx}

  let poseCoords : Finset (Fin (numMDCoordinates prob)) :=
    {i : Fin (numMDCoordinates prob) |
      i.val ≥ 3 * prob.protein.numAtoms + 3 * prob.ligand.numAtoms}

  have h_relevant_subset :
    {i | (prob.toDecisionProblem).isRelevant i} ⊆
      relevantProteinCoords ∪ poseCoords := by
    intro i hi
    -- Apply md_relevance_criterion
    sorry

  have h_card_relevant : relevantProteinCoords.card = 3 * numRelevantAtoms prob := by
    unfold numRelevantAtoms relevantAtoms
    sorry

  have h_card_pose : poseCoords.card = 6 := by
    sorry

  calc
    (Finset.univ.filter (@DecisionProblem.isRelevant MDAction MDState
      (numMDCoordinates prob) (mdCoordinateSpaceStruct prob) (prob.toDecisionProblem))).card
    ≤ (relevantProteinCoords ∪ poseCoords).card := Finset.card_le_of_subset h_relevant_subset
  _ ≤ relevantProteinCoords.card + poseCoords.card := Finset.card_union_le _ _
  _ = 3 * numRelevantAtoms prob + 6 := by rw [h_card_relevant, h_card_pose]
```

**Key Steps:**
1. Define the sets explicitly
2. Prove relevant coordinates are subset of union
3. Count the sizes of each set
4. Use monotonicity of cardinality

---

## Tier 2: Important Theorem Connections

### 2.1 Complete `docking_tractability_threshold` (Line 315)

**Need:** Find and apply the main tractability theorem from `StructuralRank.lean`

```lean
theorem docking_tractability_threshold
    (prob : MDBindingProblem)
    (K : Nat)
    (hPocket : numRelevantAtoms prob ≤ K)
    (hSrank : @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) (prob.toDecisionProblem) ^ 10 < numMDCoordinates prob) :
    True := by
  -- Apply the srank bound
  have h_bound := small_pocket_low_srank prob K hPocket
  -- h_bound: srank ≤ 3K + 6

  -- For fixed K, 3K + 6 = O(1) = O(n^0.1)
  -- Therefore srank is polynomially bounded

  -- Need to find the theorem in StructuralRank.lean
  -- Probably named something like:
  --   structural_rank_tractability_if
  --   polynomial_srank_tractable
  --   hasPolynomialSrank_tractable

  -- Apply it here
  sorry
```

**Action Required:**
- Search `StructuralRank.lean` for the tractability theorem
- Apply it with `h_bound` and `hSrank`

### 2.2 Complete `md_thermodynamic_lower_bound` (Line 351)

**Need:** Apply Landauer bound from `ThermodynamicLift.lean`

```lean
theorem md_thermodynamic_lower_bound
    (prob : MDBindingProblem)
    (kB T s : ℝ)
    (hkB : 0 < kB)
    (hT : 0 < T)
    (hSrank : @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) (prob.toDecisionProblem) = s) :
    let E_min := s * kB * T * Real.log 2
    True := by
  -- Use landauerJoulesPerBit from ThermodynamicLift.lean
  -- Need to find theorem connecting srank to energy
  -- Probably something like:
  --   energy_ge_srank_landauer
  --   entropy_energy_bound

  rw [← hSrank]
  -- Apply the theorem
  sorry
```

**Action Required:**
- Find the energy-srank theorem in `ThermodynamicLift.lean`
- Apply it with `hkB` and `hT`

---

## Tier 3: Easy Fixes (I Should Do These)

### 3.1 Coordinate Index Proof (Line 176)

```lean
⟨coordIdx, by
  have h_coord : coordIdx < numMDCoordinates prob := by
    unfold numMDCoordinates
    have h1 : atomIdx < prob.protein.numAtoms := by
      rwa [← prob.protein.size_eq] at hAtomInProtein
    have h2 : 3 * atomIdx < 3 * prob.protein.numAtoms := by
      exact Nat.mul_lt_mul_of_pos_left h1 (by norm_num)
    have h3 : 3 * atomIdx + axis.val < 3 * prob.protein.numAtoms + 3 := by
      exact Nat.add_lt_add_right h2 axis.is_lt
    exact Nat.lt_of_lt_of_le h3 (Nat.le_add_left _ _)
  exact h_coord⟩
```

### 3.2 Arithmetic Inequalities (Lines 286, 289)

```lean
-- Line 286
have h3 : 3 * numRelevantAtoms prob ≤ 3 * K :=
  Nat.mul_le_mul_left 3 hBound

-- Line 289
have h4 : 3 * numRelevantAtoms prob + 6 ≤ 3 * K + 6 :=
  Nat.add_le_add_right h3 6
```

### 3.3 Average Relevant Atoms (Line 357)

```lean
noncomputable def avgRelevantAtomsPerAtom (prob : MDBindingProblem) : ℝ :=
  if prob.protein.numAtoms = 0 then 0
  else (numRelevantAtoms prob : ℝ) / (prob.protein.numAtoms : ℝ)
```

### 3.4 Restructure `cutoff_enables_sparsity` (Line 377)

**Option 1: State as correctness theorem**

```lean
/-- Sparse force computation is correct:
    Forces can be computed using only relevant atom pairs. -/
theorem cutoff_enables_sparsity_correctness
    (prob : MDBindingProblem) :
    ∀ (i : Fin (numMDCoordinates prob)),
      @DecisionProblem.isRelevant MDAction MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) (prob.toDecisionProblem) i ↔
      ∃ atom, atom ∈ prob.protein.atoms ∧
        atomWithinCutoff atom prob.bindingSite prob.cutoff ∧
        i.val < 3 * prob.protein.numAtoms ∧
        i.val / 3 = atom.index := by
  sorry
```

**Option 2: Keep as comment/conjecture**

```lean
/-! ## Algorithmic Implications

INFORMAL RESULT: Cutoff enables sparse force computation.

The bounded srank theorem implies that only atoms within the cutoff
radius can affect the optimal decision. Therefore, force computations
can be restricted to relevant atom pairs:

- Naive MD: O(N²) force calculations
- With cutoff: O(N × K) where K = relevant atoms per atom
- If K is small (local interactions): O(N) instead of O(N²)

This is the ALGORITHMIC consequence of the srank bound.

Formalizing this in Lean requires a complexity theory library.
For the purposes of this paper, the srank bound is the key result. -/
```

---

## Summary of Changes

### Structural Changes (Required First)

1. **Add `utility_local` field to `MDBindingProblem`**
2. **Add helper projections:** `atomStateProj`, `ligandPoseProj`
3. **Add helper function:** `perturbAtomPosition`

### Proof Completions

| Line | Theorem | Difficulty | Dependencies |
|------|---------|-----------|--------------|
| 176 | Coordinate index | ⭐ Easy | None |
| 208 | md_relevance_criterion → | ⭐⭐⭐⭐⭐ Hard | utility_local field |
| 212 | md_relevance_criterion ← | ⭐⭐⭐⭐⭐ Hard | utility_local field |
| 239 | md_srank_bound | ⭐⭐⭐ Medium | md_relevance_criterion |
| 286 | small_pocket inequality | ⭐ Easy | None |
| 289 | small_pocket inequality | ⭐ Easy | None |
| 315 | docking_tractability | ⭐⭐ Medium | StructuralRank.lean |
| 351 | md_thermodynamic | ⭐⭐ Medium | ThermodynamicLift.lean |
| 357 | avgRelevantAtomsPerAtom | ⭐ Trivial | None |
| 377 | cutoff_enables_sparsity | ⭐⭐⭐ Medium | Restructure or comment |

### Files for Lean Expert

**Primary:**
- `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/Tractability/MolecularSrank.lean`

**Supporting (for context):**
- `DecisionQuotient/Sufficiency.lean` (isRelevant definition)
- `DecisionQuotient/Tractability/StructuralRank.lean` (tractability theorems)
- `DecisionQuotient/ThermodynamicLift.lean` (Landauer bounds)

### Estimated Time

- **Tier 1 (Critical):** 3-4 hours
- **Tier 2 (Important):** 1-2 hours
- **Tier 3 (Easy):** 30 minutes

**Total:** ~5-6 hours for a Lean expert

---

## Verification Checklist

After completion:

- [ ] File compiles without errors
- [ ] No `sorry` statements remain
- [ ] No unused variable warnings
- [ ] `lake build DecisionQuotient.Tractability.MolecularSrank` succeeds
- [ ] All theorems have complete proofs
- [ ] `utility_local` field added to `MDBindingProblem`
- [ ] No non-standard axioms introduced

---

## Questions for Lean Expert

1. **Should we add the helper functions (`atomStateProj`, `ligandPoseProj`) to the structure or as separate definitions?**

2. **Is there a better way to structure the `utility_local` property?** The current formulation requires proving equality for all atoms within cutoff.

3. **For the algorithmic implication (cutoff_enables_sparsity), should we:**
   - a) Formalize it as a correctness theorem?
   - b) Keep it as an informal comment?
   - c) Use a different formulation?

4. **Are there any existing complexity/Big-O libraries in Mathlib4 we should leverage?**

---

## Next Steps

1. **Review this plan** with Lean expert
2. **Make structural changes** (add `utility_local` field, helpers)
3. **Complete Tier 3 proofs** (easy ones I can do)
4. **Expert completes Tier 1 and 2** (hard proofs)
5. **Verify compilation** and run tests
6. **Document in paper** the locality constraint and its implications
