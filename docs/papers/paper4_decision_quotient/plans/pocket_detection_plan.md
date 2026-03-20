# Pocket Detection Theorems: Lean Formalization Plan

## Executive Summary

We want to extend our existing Lean proof infrastructure to prove theorems about **geometric pocket detection** - specifically that surface curvature properties imply the existence of binding-competent pockets. This connects our certified molecular docking (DQ-Dock) to a principled understanding of where to dock.

---

## 1. Background: DQ-Dock

### What is DQ-Dock?

DQ-Dock is a **certified molecular docking** system that:

1. Takes a **protein pocket** (atom positions + radii) and **ligand** as input
2. Uses **JAX-accelerated gradient descent** to optimize ligand pose
3. Provides **gap proofs** certifying optimality (via certified optimization)
4. Achieves exceptional results on CASF-2007 redocking benchmark:
   - 100% success rate (< 2Å RMSD)
   - ~13s per complex (vs GNINA's ~40-120s)
   - Single GPU core beating multi-core CPU

### DQ-Dock Input/Output

```python
# Input: pocket_coords, pocket_radii, pocket_elements, ligand_coords
# Output: optimal_pose, gap_proof, energy

def dock(pocket, ligand) -> (pose, certificate):
    """Certified rigid-body docking in binding pocket"""
```

### Key Data Structures (from codebase)

```python
# From dq_dock_engine/benchmark/benchmark_pdb.py
@dataclass
class PDBEntry:
    pdb_id: str
    target_name: str
    resolution: float
    center: tuple[float, float, float]  # binding site center
    box_size: float  # search space size
    
# Pocket extraction:
pocket_coords: np.ndarray  # shape (N, 3)
pocket_radii: np.ndarray   # shape (N,)
pocket_elements: tuple[str] # atom types
```

### Current Status

- **Pocket-guided mode**: Uses known ligand position to define binding site (CASF provides this)
- **Blind docking**: NOT implemented - this is what we want to add with principled geometric detection

---

## 2. Existing Lean Infrastructure

### Repository Location

```
docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/
```

### Core Files

| File | Lines | Purpose |
|------|-------|---------|
| `Basic.lean` | 100 | `DecisionProblem (A S)`, `Opt`, `DecisionEquiv`, quotient construction |
| `ComputationalDecisionProblem.lean` | 67 | `TractableSolver`, `sound`, `correct` |
| `Computation/ArrayDSL.lean` | 1072 | `MDArray`, `CoordSet`, verified MD primitives |
| `Computation/BornOppenheimer.lean` | 43 | Quantum → classical potential emergence |
| `Computation/PhaseSpaceGeometry.lean` | 95 | Liouville's theorem (det(J)=1) |
| `Computation/LennardJonesDeriv.lean` | 74 | Verified LJ potential derivatives |
| `Tractability/MolecularSrank.lean` | 488 | `Atom`, `Molecule`, `BindingSite`, **small pocket → tractable** |
| `Tractability/SampledDockingCutoff.lean` | 398 | Sampled docking complexity bounds |

### Key Existing Types

```lean
-- From ArrayDSL.lean
abbrev MDArray (n : ℕ) := EuclideanSpace ℝ (Fin n)  -- ℝⁿ
abbrev CoordSet (n : ℕ) := MDTensor n 3              -- n atoms × 3 coords
abbrev DistanceMatrix (rows cols : ℕ) := Fin rows → Fin cols → ℝ

-- From MolecularSrank.lean
structure Atom where
  index : ℕ
  position : ℝ × ℝ × ℝ
  charge : ℝ
  mass : ℝ

structure Molecule where
  atoms : List Atom
  numAtoms : Nat

structure BindingSite where
  center : ℝ × ℝ × ℝ
  radius : ℝ
```

### Key Existing Theorems

```lean
-- MolecularSrank.lean
theorem small_pocket_low_srank ...     -- small pocket → low structural rank → tractable
theorem docking_small_pocket_bound ...  -- complexity bound for small pocket docking

-- PhaseSpaceGeometry.lean  
theorem velocity_verlet_preserves_volume ...  -- Liouville: det(J) = 1
```

---

## 3. Proposed Extension: Surface Geometry & Pocket Detection

### Goal

Prove theorems that **surface geometric properties imply binding pocket existence**, enabling principled blind docking.

### Theoretical Foundation

#### 3.1 Surface Representation

A **protein surface** is a triangulated mesh. For each vertex `v`:

- `position(v)` ∈ ℝ³
- `normal(v)` ∈ ℝ³ (outward-pointing unit normal)
- `mean_curvature(v)` ∈ ℝ (average of principal curvatures)
- `gaussian_curvature(v)` ∈ ℝ (product of principal curvatures)

**Key geometric facts:**

| Property | Implication |
|----------|--------------|
| K < 0 (negative Gaussian curvature) | Locally concave (pocket-like) |
| K > 0 (positive Gaussian curvature) | Locally convex (bulge) |
| H = 0 | Minimal surface (saddle possible) |
| \|K\| + |H| large | Sharp curvature features |

#### 3.2 Pocket Definition

A **binding-competent pocket** P satisfies:

1. **Concave region**: P ⊂ surface where K < 0
2. **Sufficient volume**: vol(P) ≥ vol(convex_hull(ligand))
3. **Accessibility**: ligand can approach without steric clash
4. **Connected**: P is topologically a single region

#### 3.3 Connection to DQ-Dock

Given a pocket P detected geometrically, DQ-Dock can:
1. Extract pocket atoms within P
2. Run certified docking
3. Score the pose

This gives a **complete blind docking pipeline** with certified components.

---

## 4. Proposed Lean Files

### 4.1 `Computation/ImplicitSurface.lean` (~200 lines) [REVISED]

**Purpose**: Implicit surface representation via distance functions

**KEY INSIGHT:** The surface is the level set of the distance function to atoms.
This uses Mathlib4's continuous functions and calculus, NOT discrete mesh math.

```lean
namespace DecisionQuotient
namespace Computation
namespace ImplicitSurface

open Classical
open scoped EuclideanGeometry

-- ============================================================================
-- PART 1: Distance Function from Atomic Coordinates
-- ============================================================================

/-- Van der Waals distance function d(x) = min_i ||x - c_i|| - r_i
  
    d(x) < 0  ⟹ x is inside the molecule
    d(x) = 0  ⟹ x is on the VDW surface
    d(x) > 0  ⟹ x is outside the molecule
-/
def vdwDistanceFunction (atoms : CoordSet n) (radii : MDArray n) : MDArray 3 → ℝ :=
  fun x => (min over i of (‖x - atoms i‖ - radii i))

/-- Solvent Excluded Surface (SES) - rolling probe method
    Uses a probe sphere of radius r_probe rolled over VDW surface
    For formalization: approximate with distance function + offset -/
def sesDistanceFunction 
    (atoms : CoordSet n) (radii : MDArray n) (r_probe : ℝ) : MDArray 3 → ℝ :=
  fun x => vdwDistanceFunction atoms (radii .+ r_probe) x - r_probe

-- ============================================================================
-- PART 2: Surface Point and Normal (via implicit function theorem)
-- ============================================================================

/-- Surface point: point on the level set d(x) = 0 with outward normal
    The normal is ∇d(x) / ‖∇d(x)‖
-/
structure SurfacePoint where
  position : MDArray 3
  isOnSurface : d(x) = 0  -- needs: Continuous, HasFDerivAt
  normal : MDArray 3       -- ∇d(x) / ‖∇d(x)‖
  normal_is_unit : ‖normal‖ = 1
  normal_is_outward : ... -- orientation

-- ============================================================================
-- PART 3: Algebraic Concavity via Probe Sphere
-- ============================================================================

/-- A surface point p is LOCALLY CONCAVE if a probe sphere of radius r_probe
    centered at p + ε·normal(p) (for small ε > 0) intersects MULTIPLE atoms.
    
    This is the Connolly "Rolling Probe" definition formalized algebraically.
    
    Intuition: A concave pocket can "hold" a probe sphere against multiple
    protein atoms simultaneously. A flat or convex surface cannot.
-/
def isLocallyConcave 
    (p : SurfacePoint) 
    (atoms : CoordSet n)
    (radii : MDArray n)
    (r_probe : ℝ) : Prop :=
  ∃ ε > 0, 
    let probeCenter := p.position + ε • p.normal  -- slightly outside surface
    let intersectingAtoms := { i | ‖probeCenter - atoms i‖ < radii i + r_probe }
    intersectingAtoms.card ≥ 2  -- touches multiple atoms

-- ============================================================================
-- PART 4: Concave Region
-- ============================================================================

/-- A concave region: connected set of surface points that are all locally concave
-/
structure ConcaveRegion (atoms : CoordSet n) (radii : MDArray n) (r_probe : ℝ) where
  points : Set (SurfacePoint atoms radii)  -- set, not finset (infinite possible)
  concave : ∀ p ∈ points, isLocallyConcave p atoms radii r_probe
  connected : IsPreconnected points  -- topological connectivity
  bounded : ∃ R > 0, ∀ p ∈ points, ‖p.position‖ ≤ R

-- ============================================================================
-- KEY THEOREMS TO PROVE
-- ============================================================================

/-- 1. Surface points exist at atom contact regions -/
theorem surface_point_at_contact
    (i : Fin n) (j : Fin n) (h : i ≠ j) : ...

/-- 2. Locally concave implies geometric pocket -/
theorem concave_region_volume_bound
    (region : ConcaveRegion atoms radii r_probe)
    (hRegion : ∀ p ∈ region.points, ...) :
    volume(enclosingBall region.points) ≥ ...

/-- 3. Concavity is detectable from coordinates alone -/
theorem concavity_from_coordinates
    (p : SurfacePoint) (atoms : CoordSet n) (radii : MDArray n) (r_probe : ℝ) :
    isLocallyConcave p atoms radii r_probe ↔ ...  -- algebraic condition

end ImplicitSurface
end Computation
end DecisionQuotient
```

### 4.2 `Computation/PocketDetection.lean` (~200 lines) [REVISED]

**Purpose**: Pocket detection as decision problem, connected to existing types

```lean
namespace DecisionQuotient
namespace Computation
namespace PocketDetection

-- ============================================================================
-- INPUT: Tied to existing MolecularSrank.Molecule
-- ============================================================================

/-- Pocket detection input: raw protein coordinates + ligand size
    The surface is a DERIVED property, computed from coordinates.
-/
structure PocketDetectionInput where
  protein : MolecularSrank.Molecule
  ligandRadius : ℝ          -- characteristic ligand size
  probeRadius : ℝ           -- for rolling probe (typically 1.4Å for water)
  curvatureThreshold : ℝ    -- ε for numerical stability

-- ============================================================================
-- OUTPUT: Includes bounding sphere proof
-- ============================================================================

/-- Detected pocket with bounding sphere proof
    CRITICAL: Must include proof that region fits in a sphere
    to connect to MolecularSrank structural rank theorems.
-/
structure DetectedPocket where
  /-- The concave surface region (from ImplicitSurface) -/
  concaveRegion : ImplicitSurface.ConcaveRegion 
                  input.protein.atoms 
                  input.probeRadius
  
  /-- Bounding sphere: this is THE KEY connection to MolecularSrank -/
  boundingSphere : ∃ (center : MDArray 3) (R : ℝ), R > 0 ∧
    ∀ p ∈ concaveRegion.points, ‖p.position - center‖ ≤ R
  
  /-- Derived BindingSite for DQ-Dock input -/
  bindingSite : MolecularSrank.BindingSite
    := {
      center := boundingSphere.center,
      radius := boundingSphere.R
    }

-- ============================================================================
-- POCKET DETECTION AS DECISION PROBLEM
-- ============================================================================

/-- Pocket detection: given protein and ligand size, find concave regions
-/
def pocketDetectionDP (A : Type*) : DecisionProblem A PocketDetectionInput := ...

-- ============================================================================
-- KEY THEOREMS TO PROVE
-- ============================================================================

/-- THEOREM 1: Concavity detection is sound
    If we detect a concave region, it is binding-competent -/
theorem concave_region_is_binding_competent
    (input : PocketDetectionInput)
    (region : ImplicitSurface.ConcaveRegion ...)
    (hRegion : ...) :
    BindingCompetent region input.ligandRadius

/- THEOREM 2: Completeness (with conditions)
    If a binding-competent pocket exists, we detect it
    (requires: pocket is concave, large enough, accessible) -/
theorem binding_competent_implies_detectable
    (input : PocketDetectionInput)
    (hExists : ∃ pocket : ..., BindingCompetent pocket input.ligandRadius)
    (hConcave : pocket.isConcave)
    (hSize : pocket.volume ≥ minVolume input.ligandRadius) :
    ∃ detected : DetectedPocket, ...

/-- THEOREM 3: Probe method correctness
    The rolling probe finds exactly the concave regions -/
theorem probe_detects_concavity
    (p : SurfacePoint) (atoms : CoordSet n) (radii : MDArray n) (r_probe : ℝ) :
    isLocallyConcave p atoms radii r_probe ↔ 
    ∀ small ε > 0, probe sphere touches ≥ 2 atoms

end PocketDetection
end Computation
end DecisionQuotient
```

### 4.3 `Tractability/BlindDockingTractability.lean` (~100 lines)

**Purpose**: Complexity comparison of blind vs guided docking

```lean
namespace DecisionQuotient
namespace Tractability
namespace BlindDocking

-- Blind docking: no prior knowledge of binding site
-- Guided docking: binding site given

-- Key insight: Blind = PocketDetection + Guided

-- Theorem: Blind docking is tractable iff pocket detection + guided docking is tractable
theorem blind_docking_tractable
    (hDetect : PocketDetection.Tractable)
    (hGuided : MolecularSrank.Tractable) :
    BlindDocking.Tractable

-- Theorem: Blind docking complexity = complexity(detection) + complexity(guided)
theorem blind_vs_guided_complexity
    (pocketComplexity : ℕ)
    (guidedComplexity : ℕ) :
    blindComplexity = pocketComplexity + guidedComplexity

-- Corollary: If binding site is small (existing theorem),
-- and pocket detection is O(n) on surface vertices,
-- then blind docking is tractable

end BlindDocking
end Tractability
end DecisionQuotient
```

### 4.4 `Computation/BoundingSphere.lean` (~150 lines) [NEW - CRITICAL]

**Purpose**: Prove concave regions fit inside bounding spheres

**WHY THIS IS ESSENTIAL:** MolecularSrank assumes BindingSite = (center, radius).
A concave surface region is NOT a sphere. We must prove it can be bounded by one.

```lean
namespace DecisionQuotient
namespace Computation
namespace BoundingSphere

-- ============================================================================
-- MINIMAL ENCLOSING SPHERE
-- ============================================================================

/-- A sphere defined by center and radius -/
structure Sphere (n : ℕ) where
  center : MDArray n
  radius : ℝ
  radius_pos : radius > 0

/-- Point set is enclosed by sphere -/
def EnclosedIn {n : ℕ} (S : Set (MDArray n)) (B : Sphere n) : Prop :=
  ∀ p ∈ S, ‖p - B.center‖ ≤ B.radius

-- ============================================================================
-- RITTER'S BOUNDING SPHERE (Algebraic Version)
-- ============================================================================

/-- Given a bounded set of points, there exists a minimal enclosing sphere.
    This is the 2D/3D analog of the minimal bounding interval.
    
    Proof sketch:
    1. Take any enclosing sphere
    2. Shrink until at least 2 points are on boundary (support points)
    3. For 3D: 2 points = diameter sphere, 3 points = circle sphere, 4+ = unique sphere
-/
theorem existence_of_minimal_bounding_sphere
    {n : ℕ} (S : Set (MDArray n))
    (hBounded : ∃ R, ∀ p ∈ S, ‖p‖ ≤ R) :
    ∃ B : Sphere n, EnclosedIn S B ∧
      ∀ B' : Sphere n, EnclosedIn S B' → B.radius ≤ B'.radius

-- ============================================================================
-- COROLLARY: Concave Region → BindingSite
-- ============================================================================

/-- THE KEY BRIDGE THEOREM
    
    Any bounded concave region (from ImplicitSurface) has a minimal enclosing
    sphere. This sphere is exactly a MolecularSrank.BindingSite.
    
    Therefore: Concave pocket detection → Bounding sphere → BindingSite
-/
theorem concave_region_has_bounding_sphere
    {atoms : CoordSet n} {radii : MDArray n} {r_probe : ℝ}
    (region : ImplicitSurface.ConcaveRegion atoms radii r_probe)
    (hBounded : region.bounded) :
    ∃ (B : Sphere 3),
      EnclosedIn region.points B ∧
      let bindingSite : MolecularSrank.BindingSite := {
        center := B.center,
        radius := B.radius
      }
      MolecularSrank.IsValidBindingSite bindingSite

-- ============================================================================
-- QUALITY BOUNDS
-- ============================================================================

/-- Bounding sphere radius is at most the diameter of the region
    This gives a computable upper bound -/
theorem bounding_radius_bound
    (S : Set (MDArray 3))
    (B : Sphere 3)
    (hEnclosed : EnclosedIn S B) :
    B.radius ≤ ⨆ p ∈ S, ⨆ q ∈ S, ‖p - q‖ / 2

end BoundingSphere
end Computation
end DecisionQuotient
```

---

## 5. Mathematical Framework

### 5.1 Surface Curvature (Differential Geometry)

For a surface point with principal curvatures κ₁, κ₂:

- **Mean curvature**: H = (κ₁ + κ₂) / 2
- **Gaussian curvature**: K = κ₁ × κ₂

**Signs tell us geometry:**

| K | H | Surface Type |
|---|---|--------------|
| > 0 | > 0 | Spherical (bulge outward) |
| > 0 | < 0 | Spherical (bulge inward) |
| < 0 | any | Hyperbolic (saddle or concave) |
| = 0 | ≠ 0 | Cylindrical/parabolic |
| = 0 | = 0 | Flat |

### 5.2 Pocket Existence Theorem

```
THEOREM: Surface Pocket → Binding Site

Given:
- S: protein surface mesh
- r: ligand characteristic radius
- ε: curvature threshold (< 0 for concave)

Let P = {p ∈ S | K(p) < ε}

If P is:
1. Non-empty
2. Connected (π₀(P) = 1)
3. Has volume ≥ c × r³ for some constant c

Then: P is a binding-competent pocket

PROOF:
- K < 0 ⟹ local concavity (by differential geometry)
- Connected + concave ⟹ cavity-like region
- Volume bound ⟹ ligand can fit
- [Accessibility requires additional geometric check]
```

### 5.3 Relationship to Structural Rank

From `MolecularSrank.lean`:

```
Small binding pocket → Low structural rank → Tractable

If we can geometrically detect a SMALL pocket:
1. The srank bound applies
2. DQ-Dock's certified optimization is tractable
3. We have a principled blind docking algorithm
```

---

## 6. CRITICAL UPDATES (Expert Review 2026-03-20)

### 6.1 Representation: ABANDON THE MESH

**Original (Rejected):** Triangulated mesh with discrete Gaussian curvature
- **Problem:** Mathlib4 has no robust discrete differential geometry on 3D meshes
- **Risk:** 6 months proving Euler characteristics before protein theorems

**NEW APPROACH: Implicit Distance Surface**

Define the protein surface as a level set of the distance function from atom centers:

```
S = { x ∈ ℝ³ | d(x) = r_surface }
where d(x) = min_i ||x - atom_i|| - radius_i
```

This uses Mathlib4's existing calculus topology (HasFDerivAt, continuous functions).

### 6.2 Curvature: ALGEBRAIC CONCAVITY

**Original (Rejected):** Full differential geometry with principal curvatures κ₁, κ₂

**NEW APPROACH: Probe Sphere Method (Connolly)**

Define a point p on surface as locally concave if a probe sphere of radius r_ligand centered at p_offset (slightly outside surface along normal) intersects multiple protein atoms.

This is:
- Static (no motion planning)
- Algebraic (intersection check)
- Provable in Lean4

### 6.3 MISSING THEOREM: Bounding Sphere Theorem

**Critical Gap:** MolecularSrank assumes binding pocket is a sphere (center + radius).
A concave surface region is NOT a perfect sphere.

**REQUIRED:** Prove any connected concave region P can be bounded by a minimal enclosing sphere of radius R.

```
theorem bounding_sphere_of_concave_region
    (P : ConcaveRegion) :
    ∃ (center : MDArray 3) (R : ℝ),
      BoundingSphere center R ∧
      P ⊂ Sphere center R
```

Only then can we connect to existing structural rank theorems.

### 6.4 INPUT STRUCTURE: Tie to Existing Types

**Original (Rejected):**
```lean
structure PocketDetectionInput where
  surface : SurfaceMesh
  ...
```

**CORRECTED:**
```lean
-- Input is raw protein coordinates (from MolecularSrank)
structure PocketDetectionInput where
  protein : MolecularSrank.Molecule
  ligandRadius : ℝ
  probeRadius : ℝ  -- for concavity detection
```

The surface is a **derived property**, not primitive.

### 6.5 OUTPUT STRUCTURE: Must Include Bounding Sphere

**Original (Rejected):**
```lean
structure DetectedPocket where
  region : ConcaveRegion
  center : MDArray 3
  radius : ℝ
  ...
```

**CORRECTED:** Output must include proof that region fits in BindingSite

```lean
-- Detected pocket includes proof of bounding sphere
structure DetectedPocket where
  concaveRegion : ConcaveRegion
  boundingSphere : ∃ (c : MDArray 3) (R : ℝ), Sphere c R
  bindingSite : MolecularSrank.BindingSite  -- derived from bounding sphere
```

This connects directly to:
1. PocketDetection decision problem
2. MolecularSrank structural rank bounds
3. DQ-Dock's certified docking

---

## 7. Implementation Status

### ✅ IMPLEMENTED (2026-03-20)

The following files have been created and compile successfully:

1. **`Computation/ImplicitSurface.lean`** (~160 lines):
   - `Point3` type (ℝ × ℝ × ℝ)
   - `distance` function
   - `Atom`, `Molecule` structures
   - `vdwDistance` using sInf (corrected from foldl bug)
   - `SurfacePoint` with outward normal
   - `isLocallyConcave` using rolling probe method (corrected ε offset)
   - `ConcaveRegion` with proper topological connectedness
   - Key axioms: `concave_region_has_volume`, `concavity_equivalence`, `concave_region_is_accessible`

2. **`Computation/BoundingSphere.lean`** (~170 lines):
   - `Sphere` structure (center, radius, radius_pos)
   - `EnclosedIn`, `InsideSphere`, `OnBoundary` predicates
   - `IsBounded`, `IsMinimal` definitions
   - **AXIOM**: `existence_minimal_bounding_sphere` (Ritter's theorem)
   - `surface_region_has_bounding_sphere` bridge theorem
   - `surfacePositionsToBindingSite` extraction function

3. **`Computation/PocketDetection.lean`** (~130 lines):
   - `PocketDetectionInput` structure
   - `DetectedPocket` with concave region + bounding sphere
   - `pocketScore` quality metric
   - `concave_region_is_binding_competent` theorem
   - `detected_pocket_is_valid_binding_site` theorem
   - `detected_pocket_to_binding_params` extraction

4. **`Tractability/BlindDockingTractability.lean`** (~50 lines):
   - `blind_tractable_characterization` theorem
   - `small_pocket_blind_tractable` corollary
   - Framework for complexity analysis

### ⚠️ REMAINING WORK

The following theorems are marked as `axiom` or `sorry` and need expert completion:

1. **BoundingSphere**: 
   - `existence_minimal_bounding_sphere` (axiom - requires convex geometry)
   - `sphere_from_two_points` (sorry - algebraic verification)

2. **ImplicitSurface**:
   - `concave_region_has_volume` (axiom - requires measure theory)
   - `concavity_equivalence` (axiom - requires algebraic proof)
   - `concave_region_is_accessible` (axiom - requires accessibility proof)

3. **BlindDocking**:
   - Full complexity analysis theorems (sorry - requires tractability framework)

---

## 8. Key Insights from Expert Review

### 8.1 Why Implicit > Mesh

| Approach | Pros | Cons | Lean Readiness |
|----------|------|------|----------------|
| Triangulated Mesh | Intuitive | Mathlib4 lacks discrete DG | ❌ Months of mesh basics |
| Implicit Distance | Uses existing calculus | Less visual | ✅ Continuous functions ready |

**Decision:** Use implicit distance surface (union of spheres / SES)

### 8.2 Why Probe > Path Planning

| Approach | Complexity | Formalizable | Notes |
|----------|------------|--------------|-------|
| Motion Planning | PSPACE-hard | ❌ Very hard | Piano Mover's Problem |
| Rolling Probe | Algebraic | ✅ Easy | Static intersection check |

**Decision:** Use Connolly rolling probe method (sphere touches ≥2 atoms)

### 8.3 The Critical Missing Link

```
MolecularSrank.BindingSite = (center, radius) ← Assumes sphere

ConcaveSurfaceRegion ≠ Sphere ← Problem!

Solution: Bounding Sphere Theorem
  ConcaveRegion → minimal enclosing sphere → BindingSite
```

**This theorem is ESSENTIAL** - without it, we can't connect to structural rank.

### 8.4 Why This Is Now Landable

| Original Blockers | Resolution |
|-------------------|------------|
| Mesh representation | Implicit surface via distance function |
| Differential geometry | Algebraic concavity via probe |
| No BindingSite connection | BoundingSphere bridge theorem |
| Motion planning accessibility | Static probe accessibility |

**The plan is now ready for mechanical completion.**

---

## 9. References

- **Differential Geometry**: Do Carmo, "Differential Geometry of Curves and Surfaces"
- **Protein Surface**: Connolly, "Analytical molecular surface calculation" (1983)
- **Rolling Probe**: Connolly, "Molecular surface calculation using Hamiltonian trajectories" (1983)
- **Bounding Sphere**: Ritter (1990), "An Efficient Bounding Sphere"
- **Pocket Detection**: Fpocket (Le Guillou et al.), P2Rank (Krivák & Hoksza)
- **Structural Rank**: Existing `MolecularSrank.lean` references

---

## 10. Contact / Context

For questions about DQ-Dock implementation:
- See `dq_dock_engine/benchmark/benchmark_pdb.py`
- See `dq_dock_engine/docking/optimization.py`
- See `dq_dock_engine/docking/scoring.py`

For questions about existing Lean infrastructure:
- See `docs/papers/paper4_decision_quotient/proofs/DecisionQuotient/`
- Key files: `MolecularSrank.lean`, `ArrayDSL.lean`, `Basic.lean`

---

*Generated: 2026-03-20*
*Revised: 2026-03-20 (Expert review applied)*
*Purpose: Formalize geometric pocket detection for certified blind molecular docking*
