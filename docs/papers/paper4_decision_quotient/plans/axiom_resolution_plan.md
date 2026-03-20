# Axiom Resolution Plan for Pocket Detection

## Expert Feedback Applied (2026-03-20)

Based on expert review, several axioms need to be fixed or reformulated before proving.

---

## Axiom Resolution Status

### ✅ RESOLVED: Axiom 4 (`concavity_equivalence`) - Already Definitional

**Status:** NOT AN AXIOM - this is just unfolding the definition.

```lean
-- Current (wrongly marked as axiom):
axiom concavity_equivalence
    {mol : Molecule}
    (p : SurfacePoint mol)
    (r_probe : ℝ) :
    isLocallyConcave p r_probe ↔
    ∃ ε > 0,
      let probeCenter := add p.position (smul ε (outwardNormal p))
      intersectingAtomCount mol probeCenter r_probe ≥ 2

-- Fix: Just use `rfl` or `simp`
theorem concavity_equivalence ... := by rfl
```

---

### ✅ RESOLVED: Axiom 1 (`concave_region_is_bounded`) - Proven via Triangle Inequality

**Status:** READY TO PROVE

**Proof (provided by expert):**
```lean
theorem concave_region_is_bounded
    (input : PocketDetectionInput)
    (region : ConcaveRegion input.mol input.probeRadius) :
    BoundingSphere.IsBounded (ImplicitSurface.SurfacePoint.position '' region.points) := by
  rcases region.bounded with ⟨R, c, hR⟩
  refine ⟨R + BoundingSphere.distance c (0, 0, 0), ?_⟩
  intro x hx
  rcases hx with ⟨p, hp, rfl⟩
  have hpR : ImplicitSurface.distance p.position c ≤ R := hR p hp
  have htri :
      BoundingSphere.distance p.position (0, 0, 0)
        ≤ BoundingSphere.distance p.position c + BoundingSphere.distance c (0, 0, 0) := by
    -- use distance_triangle_bound after rewriting
    sorry
  have hcast :
      BoundingSphere.distance p.position c = ImplicitSurface.distance p.position c := by
    rfl
  calc
    BoundingSphere.distance p.position (0, 0, 0)
        ≤ BoundingSphere.distance p.position c + BoundingSphere.distance c (0, 0, 0) := htri
    _ = ImplicitSurface.distance p.position c + BoundingSphere.distance c (0, 0, 0) := by rw [hcast]
    _ ≤ R + BoundingSphere.distance c (0, 0, 0) := by linarith
```

**Key insight:** Uses `distance_triangle_bound` from `ArrayDSL.lean:1022-1024`

---

### ⚠️ NEEDS REFORMULATION: Axiom 3 (`concave_region_has_volume`)

**Problem:** `region.points` is a set of **surface points** (2D surface in ℝ³). A surface subset has **zero 3D volume**.

**Current (WRONG):**
```lean
∃ (vol : ℝ), vol > 0  -- Surface has no 3D volume!
```

**Correct options:**
1. **Positive surface area:**
```lean
∃ (area : ℝ), area > 0
```

2. **Positive volume of enclosed pocket (new structure needed):**
```lean
def pocketInterior (region : ConcaveRegion mol r_probe) : Set Point3 := ...
axiom pocket_has_volume (region : ...) : ∃ vol > 0, volume (pocketInterior region) = vol
```

**Recommendation:** Rewrite before attempting proof.

---

### ⚠️ NEEDS REFORMULATION: Axiom 5 (`concave_region_is_accessible`)

**Problem:** The statement says `q` being near surface point `p` implies `q` is **inside** the molecule (`vdwDistance ≤ 0`). This is backwards for docking!

**Current (WRONG):**
```lean
distance q p.position ≤ r_probe - ligandRadius → vdwDistance mol q ≤ 0
-- q inside molecule? No!
```

**Correct options:**
1. **Ligand sphere fits in pocket (non-intersection):**
```lean
∀ atoms a ∈ mol, distance (ligandCenter + radius) a.center > a.radius
```

2. **Clearance statement:**
```lean
∃ clearance > 0, ligand fits without steric clash
```

**Recommendation:** Pause and rewrite the theorem statement.

---

### 📋 TODO: Axiom 6 (`sphere_from_two_points`) - Needs Helper Lemmas

**Status:** Provable but algebra-heavy due to custom distance on tuples.

**Helper lemma needed:**
```lean
noncomputable def midpoint (p q : Point3) : Point3 :=
  ((p.1 + q.1)/2, (p.2.1 + q.2.1)/2, (p.2.2 + q.2.2)/2)

lemma distance_left_midpoint (p q : Point3) :
    distance p (midpoint p q) = distance p q / 2 := by
  sorry

lemma distance_right_midpoint (p q : Point3) :
    distance q (midpoint p q) = distance p q / 2 := by
  sorry
```

**Then:**
```lean
theorem sphere_from_two_points (p q : Point3) (hpq : p ≠ q) :
    ∃ (center : Point3) (radius : ℝ) (hr : radius > 0),
    EnclosedIn {p, q} { center := center, radius := radius, radius_pos := hr } ∧
    OnBoundary p ... ∧ OnBoundary q ... := by
  use midpoint p q, distance p q / 2, by positivity
  -- follows from midpoint lemmas
```

---

### 📋 KEEP AS AXIOM: Axiom 2 (`existence_minimal_bounding_sphere`)

**Status:** Standard theorem (Ritter/Welzl), keep as axiom.

**Consider renaming:**
```lean
-- Current: existence_minimal_bounding_sphere
-- Better: existence_bounding_sphere  -- "existence" implies minimality via IsMinimal
```

---

### 📋 KEEP AS AXIOM: Axiom 7 (`blind_docking_tractable`)

**Status:** Fine as placeholder.

**Long-term improvement:**
```lean
def BlindDockingTimeBound (n : ℕ) : ℕ := ...
def PolynomiallyBounded (f : ℕ → ℕ) : Prop := ...
axiom blind_docking_polynomial : PolynomiallyBounded BlindDockingTimeBound
```

---

## Revised Priority Order

| Priority | Axiom | Action | Status |
|----------|-------|--------|--------|
| 1 | **Axiom 4** | Replace with `rfl` | ✅ Ready |
| 2 | **Axiom 1** | Prove via triangle inequality | ✅ Ready |
| 3 | **Axiom 6** | Prove with midpoint lemmas | 📋 TODO |
| 4 | **Axiom 2** | Keep as axiom | ✅ Done |
| 5 | **Axiom 7** | Keep as axiom | ✅ Done |
| 6 | **Axiom 3** | Rewrite before proving | ⚠️ Blocked |
| 7 | **Axiom 5** | Rewrite before proving | ⚠️ Blocked |

---

## Implementation Checklist

- [ ] **Axiom 4** → Replace with theorem using `rfl`
- [ ] **Axiom 1** → Prove using triangle inequality proof above
- [ ] **Axiom 6** → Add midpoint lemmas, then prove
- [ ] **Axiom 3** → Rewrite to surface area or pocket interior
- [ ] **Axiom 5** → Rewrite to clearance/non-intersection

---

## Key Corrections from Expert Review

1. **Axiom 4 is already definitional** - just use `rfl`
2. **Axiom 3 is mathematically wrong** - surface ≠ volume
3. **Axiom 5 likely states wrong property** - ligand should be in free space, not inside molecule

---

*Updated: 2026-03-20 (Expert feedback applied)*
