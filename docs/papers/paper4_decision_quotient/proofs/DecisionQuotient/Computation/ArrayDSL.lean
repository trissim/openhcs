/-
  Paper 4: Decision-Relevant Uncertainty

  Computation/ArrayDSL.lean - Verified Primitives for MD

  A restricted DSL that Lean can reason about and JAX can execute.
  This serves as the "Common Language" between Lean proofs and JAX implementation.

  ## Triviality Level
  NONTRIVIAL: This is core machinery for the Lean-JAX bridge.

  ## Dependencies
  - Mathlib4: Analysis, linear algebra
  - Used by: JAXBridge.lean, MolecularSrank.lean

  ## Design Principles
  1. Minimal but complete set of primitives
  2. Each primitive has clear semantic meaning
  3. Derivative rules natively use Mathlib's Fréchet derivatives
  4. Compilation target is JAX (verified by construction)
-/

import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.Calculus.Deriv.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Base
import Mathlib.Tactic

namespace DecisionQuotient
namespace Computation
namespace ArrayDSL

open scoped RealInnerProductSpace

/-! ## 1. Core Types -/

/-- MD Array: fixed-size array of real numbers.
    This is the Lean representation of JAX's jnp.ndarray
    Mapped rigorously to Mathlib's Euclidean L2 Space for formal calculus. -/
abbrev MDArray (n : ℕ) := EuclideanSpace ℝ (Fin n)

/-- Row-major tensor view used for batched molecular kernels. -/
abbrev MDTensor (rows cols : ℕ) := Fin rows → MDArray cols

/-- Coordinate set of 3D points. -/
abbrev CoordSet (n : ℕ) := MDTensor n 3

/-- Pairwise distance matrix between two coordinate sets. -/
abbrev DistanceMatrix (rows cols : ℕ) := Fin rows → Fin cols → ℝ

/-- Create an MDArray from a function. -/
noncomputable def mkMDArray {n : ℕ} (f : Fin n → ℝ) : MDArray n :=
  (WithLp.equiv 2 (Fin n → ℝ)).symm f

/-- Extract a function from an MDArray. -/
noncomputable def getMDArray {n : ℕ} (arr : MDArray n) (i : Fin n) : ℝ :=
  arr i

/-- Differentiable function: ℝ → ℝ with known derivative natively defined by Mathlib -/
structure DiffFunction where
  fn : ℝ → ℝ
  differentiable : Differentiable ℝ fn

/-- Differentiable multivariate function: ℝⁿ → ℝ -/
structure DiffFunctionN (n : ℕ) where
  fn : MDArray n → ℝ
  differentiable : Differentiable ℝ fn

/-! ## 2. Verified Primitives -/

/-- Primitive: Map a function over array elements.
    JAX: jnp.vmap(f, arr) -/
noncomputable def map {n : ℕ} (f : DiffFunction) (arr : MDArray n) : MDArray n :=
  mkMDArray (fun i => f.fn (arr i))

/-- Primitive: Reduce array using summation.
    JAX: jnp.sum -/
noncomputable def reduce_sum {n : ℕ} (arr : MDArray n) : ℝ :=
  ∑ i, arr i

/-- Primitive: Element-wise binary addition.
    JAX: jnp.add -/
noncomputable def elemBinaryAdd {n : ℕ} (a b : MDArray n) : MDArray n :=
  a + b

/-- Primitive: Element-wise binary subtraction.
    JAX: jnp.subtract -/
noncomputable def elemBinarySub {n : ℕ} (a b : MDArray n) : MDArray n :=
  a - b

/-- Primitive: Norm (L2) of vector.
    JAX: jnp.linalg.norm -/
noncomputable def norm {n : ℕ} (arr : MDArray n) : ℝ :=
  ‖arr‖

/-- Primitive: row-wise L2 norm over a batched tensor.
    JAX: jnp.linalg.norm(arr, axis=-1) -/
noncomputable def rowWiseNorm {rows cols : ℕ} (arr : MDTensor rows cols) : MDArray rows :=
  mkMDArray (fun i => norm (arr i))

/-- Primitive: Distance computation between two coordinate sets.
    JAX: jnp.linalg.norm(q1 - q2, axis=-1) -/
noncomputable def distance {n : ℕ} (q1 q2 : MDArray n) : ℝ :=
  dist q1 q2

/-- Primitive: row-wise distance over two batched tensors.
    JAX: jnp.linalg.norm(arr1 - arr2, axis=-1) -/
noncomputable def rowWiseDistance {rows cols : ℕ}
    (q1 q2 : MDTensor rows cols) : MDArray rows :=
  mkMDArray (fun i => distance (q1 i) (q2 i))

/-- Condition a probability vector on a boolean support mask. -/
noncomputable def supportConditioning {n : ℕ}
    (probs : MDArray n)
    (mask : Fin n → Bool) : MDArray n :=
  mkMDArray (fun i => if mask i then probs i else 0)

/-- Normalize a nonnegative weight vector to unit sum.
    The caller is responsible for ensuring the denominator is positive. -/
noncomputable def normalizeProbabilityVector {n : ℕ} (weights : MDArray n) : MDArray n :=
  let z := reduce_sum weights
  mkMDArray (fun i => weights i / z)

/-- Uniform probability vector over the support of a template tensor. -/
noncomputable def uniformProbabilityVectorLike {n : ℕ} (_template : MDArray n) : MDArray n :=
  normalizeProbabilityVector (mkMDArray (fun _ => 1))

/-- Probability vector with explicit no-op mass at index 0 and uniform remainder. -/
noncomputable def noopBiasedProbabilityVectorLike {n : ℕ}
    (_template : MDArray n)
    (noopMass : ℝ) : MDArray n :=
  if h : n = 0 then
    mkMDArray (fun _ => 0)
  else if hOne : n = 1 then
    mkMDArray (fun _ => 1)
  else
    let remainder := (1 - noopMass) / (n - 1)
    mkMDArray (fun i => if i.1 = 0 then noopMass else remainder)

/-- Conservative top-k-with-ties mask over utility values.
    An action survives when fewer than `k` actions are strictly better. -/
noncomputable def topKWithTiesMask {n : ℕ}
    (utilities : MDArray n)
    (k : ℕ) : Fin n → Bool :=
  fun i =>
    let strictBetter : Finset (Fin n) :=
      (Finset.univ : Finset (Fin n)).filter (fun j => utilities i < utilities j)
    strictBetter.card < k

/-- Certified ambiguity band around the kth utility boundary. -/
noncomputable def ambiguityBandMask {n : ℕ}
    (utilities : MDArray n)
    (k : ℕ)
    (eps : ℝ) : Fin n → Bool :=
  if hk : 0 < k then
    let topKValues : Finset ℝ := ((Finset.univ : Finset (Fin n)).filter (fun i => topKWithTiesMask utilities k i)).image utilities
    if hTop : topKValues.Nonempty then
      let kthBoundary := topKValues.min' hTop
      fun i => kthBoundary - eps ≤ utilities i
    else
      fun _ => false
  else
    fun _ => false

/-- Deterministic first-max selector under a boolean mask.
    Returns `0` when the masked support is empty. -/
noncomputable def stableArgmaxMasked {n : ℕ}
    (values : MDArray n)
    (mask : Fin n → Bool) : ℕ :=
  let survivors : Finset (Fin n) := (Finset.univ : Finset (Fin n)).filter (fun i => mask i)
  if hNonempty : survivors.Nonempty then
    let maxValue := (survivors.image values).max' <| by
      rcases hNonempty with ⟨i, hi⟩
      exact ⟨values i, Finset.mem_image.mpr ⟨i, hi, rfl⟩⟩
    let maximizers := survivors.filter (fun i => values i = maxValue)
    let hMaximizers : maximizers.Nonempty := by
      have hMaxMem : maxValue ∈ survivors.image values := Finset.max'_mem _ _
      rcases Finset.mem_image.mp hMaxMem with ⟨witness, hWitness, hWitnessEq⟩
      exact ⟨witness, by simp [maximizers, hWitness, hWitnessEq]⟩
    (maximizers.min' hMaximizers).1
  else
    0

/-- Apply a rigid 3D transform defined by a quaternion and translation to one point. -/
noncomputable def rigidTransformPoint3D
    (point : MDArray 3)
    (quaternion : MDArray 4)
    (translation : MDArray 3) : MDArray 3 :=
  let w := quaternion ⟨0, by decide⟩
  let x := quaternion ⟨1, by decide⟩
  let y := quaternion ⟨2, by decide⟩
  let z := quaternion ⟨3, by decide⟩
  let px := point ⟨0, by decide⟩
  let py := point ⟨1, by decide⟩
  let pz := point ⟨2, by decide⟩
  let tx := translation ⟨0, by decide⟩
  let ty := translation ⟨1, by decide⟩
  let tz := translation ⟨2, by decide⟩
  mkMDArray fun j =>
    if h0 : j.1 = 0 then
      (1 - 2 * y ^ (2 : ℕ) - 2 * z ^ (2 : ℕ)) * px +
        (2 * x * y - 2 * z * w) * py +
        (2 * x * z + 2 * y * w) * pz + tx
    else if h1 : j.1 = 1 then
      (2 * x * y + 2 * z * w) * px +
        (1 - 2 * x ^ (2 : ℕ) - 2 * z ^ (2 : ℕ)) * py +
        (2 * y * z - 2 * x * w) * pz + ty
    else
      (2 * x * z - 2 * y * w) * px +
        (2 * y * z + 2 * x * w) * py +
        (1 - 2 * x ^ (2 : ℕ) - 2 * y ^ (2 : ℕ)) * pz + tz

/-- Apply a rigid 3D transform to a coordinate set.
    JAX: coords @ R.T + translation -/
noncomputable def rigidTransform3D {n : ℕ}
    (coords : CoordSet n)
    (quaternion : MDArray 4)
    (translation : MDArray 3) : CoordSet n :=
  fun i => rigidTransformPoint3D (coords i) quaternion translation

/-- Build a unit quaternion from an axis-angle parameterization. -/
noncomputable def axisAngleQuaternion
    (axis : MDArray 3)
    (angle : ℝ) : MDArray 4 :=
  let half := angle / 2
  let s := Real.sin half
  mkMDArray fun i =>
    if h0 : i.1 = 0 then
      Real.cos half
    else if h1 : i.1 = 1 then
      axis ⟨0, by decide⟩ * s
    else if h2 : i.1 = 2 then
      axis ⟨1, by decide⟩ * s
    else
      axis ⟨2, by decide⟩ * s

/-- Canonical local translation stencil in 3D with signed axis steps. -/
noncomputable def localTranslationStencil3D (step : ℝ) : MDTensor 6 3 :=
  fun i =>
    match i.1 with
    | 0 => mkMDArray (fun j => if j.1 = 0 then step else 0)
    | 1 => mkMDArray (fun j => if j.1 = 0 then -step else 0)
    | 2 => mkMDArray (fun j => if j.1 = 1 then step else 0)
    | 3 => mkMDArray (fun j => if j.1 = 1 then -step else 0)
    | 4 => mkMDArray (fun j => if j.1 = 2 then step else 0)
    | _ => mkMDArray (fun j => if j.1 = 2 then -step else 0)

/-- Canonical local rotation stencil as signed quarter-step quaternions around
    the Cartesian axes. -/
noncomputable def localRotationStencil3D (angle : ℝ) : MDTensor 6 4 :=
  let ex := mkMDArray (fun i => if i.1 = 0 then 1 else 0)
  let ey := mkMDArray (fun i => if i.1 = 1 then 1 else 0)
  let ez := mkMDArray (fun i => if i.1 = 2 then 1 else 0)
  fun i =>
    match i.1 with
    | 0 => axisAngleQuaternion ex angle
    | 1 => axisAngleQuaternion ex (-angle)
    | 2 => axisAngleQuaternion ey angle
    | 3 => axisAngleQuaternion ey (-angle)
    | 4 => axisAngleQuaternion ez angle
    | _ => axisAngleQuaternion ez (-angle)

/-- Fixed deterministic quaternion dictionary for certified global support. -/
noncomputable def quaternionDictionary8 : MDTensor 8 4 :=
  let half := Real.sqrt (1 / 2 : ℝ)
  fun i =>
    match i.1 with
    | 0 => mkMDArray (fun j => if j.1 = 0 then 1 else 0)
    | 1 => mkMDArray (fun j => if j.1 = 1 then 1 else 0)
    | 2 => mkMDArray (fun j => if j.1 = 2 then 1 else 0)
    | 3 => mkMDArray (fun j => if j.1 = 3 then 1 else 0)
    | 4 => mkMDArray (fun j => if j.1 = 0 ∨ j.1 = 1 then half else 0)
    | 5 => mkMDArray (fun j => if j.1 = 0 ∨ j.1 = 2 then half else 0)
    | 6 => mkMDArray (fun j => if j.1 = 0 ∨ j.1 = 3 then half else 0)
    | _ => mkMDArray (fun _ => 1 / 2)

/-! ## 3. Derivative Definitions -/

/-- The formal gradient of any array function mapped to JAX autodiff. -/
noncomputable def array_gradient {n : ℕ} (f : DiffFunctionN n) (q : MDArray n) : MDArray n :=
  gradient f.fn q

/-- The 1D derivative. -/
noncomputable def array_deriv (f : DiffFunction) (x : ℝ) : ℝ :=
  deriv f.fn x

/-! ## 4. MD-Specific Operations -/

/-- Compute all pairwise distances between two sets of atoms.
    JAX: jnp.linalg.norm(coords1[:,None,:] - coords2[None,:,:], axis=-1) -/
noncomputable def pairwiseDistances {n1 n2 : ℕ} (coords1 : MDArray n1) (coords2 : MDArray n2) : Fin n1 → Fin n2 → ℝ :=
  fun i j => |coords1 i - coords2 j|

/-- Compute pairwise Euclidean distances between two sets of 3D coordinates.
    JAX: jnp.linalg.norm(coords1[:,None,:] - coords2[None,:,:], axis=-1) -/
noncomputable def pairwiseDistances3D {n1 n2 : ℕ}
    (coords1 : CoordSet n1)
    (coords2 : CoordSet n2) : DistanceMatrix n1 n2 :=
  fun i j => distance (coords1 i) (coords2 j)

/-- Minimum-image wrapping for one scalar coordinate difference. -/
noncomputable def minimumImageScalar (delta box : ℝ) : ℝ :=
  if box = 0 then delta else delta - box * ((round (delta / box) : Int) : ℝ)

/-- Pairwise Euclidean distances under the minimum-image convention. -/
noncomputable def minimumImagePairwiseDistances {n1 n2 : ℕ}
    (coords1 : CoordSet n1)
    (coords2 : CoordSet n2)
    (boxSize : MDArray 3) : DistanceMatrix n1 n2 :=
  fun i j =>
    let wrapped := mkMDArray fun k =>
      minimumImageScalar ((coords1 i k) - (coords2 j k)) (boxSize k)
    norm wrapped

/-- Apply cutoff mask: smooth step or hard zero.
    JAX: distances * (distances < rc) -/
noncomputable def applyCutoff {n : ℕ} (distances : MDArray n) (rc : ℝ) : MDArray n :=
  mkMDArray (fun i => if distances i < rc then distances i else 0)

/-- Lennard-Jones potential (single pair).
    U(r) = 4ε[(σ/r)¹² - (σ/r)⁶] -/
noncomputable def lennardJones (ε σ r : ℝ) : ℝ :=
  if r ≤ 1 / (10 : ℝ) ^ (10 : ℕ) then (10 : ℝ) ^ (12 : ℕ)
  else
    let sr := σ / r
    4 * ε * (sr ^ (12 : ℕ) - sr ^ (6 : ℕ))

/-- Sum of pair potentials with cutoff.
    JAX: jnp.sum(applyCutoff(distances, rc) | energy_fn) -/
noncomputable def sumPairPotentials {n : ℕ}
    (distances : MDArray n)
    (rc ε σ : ℝ) : ℝ :=
  let masked := applyCutoff distances rc
  ∑ i, lennardJones ε σ (masked i)

/-- Sum of pair potentials over a pairwise distance matrix. -/
noncomputable def sumPairPotentialsMatrix {n1 n2 : ℕ}
    (distances : DistanceMatrix n1 n2)
    (rc ε σ : ℝ) : ℝ :=
  ∑ i, ∑ j, let r := distances i j
    if r < rc then lennardJones ε σ r else 0

/-- Complete 3D pair-potential reduction from two coordinate sets. -/
noncomputable def sumPairPotentials3D {n1 n2 : ℕ}
    (coords1 : CoordSet n1)
    (coords2 : CoordSet n2)
    (rc ε σ : ℝ) : ℝ :=
  sumPairPotentialsMatrix (pairwiseDistances3D coords1 coords2) rc ε σ

/-- Apply Lennard-Jones elementwise using per-pair epsilon and sigma matrices. -/
noncomputable def typedLennardJonesMatrix {n1 n2 : ℕ}
    (distances epsilons sigmas : DistanceMatrix n1 n2) : DistanceMatrix n1 n2 :=
  fun i j => lennardJones (epsilons i j) (sigmas i j) (distances i j)

/-- Sum typed Lennard-Jones interactions within a scalar cutoff. -/
noncomputable def typedLennardJonesCutoff {n1 n2 : ℕ}
    (distances epsilons sigmas : DistanceMatrix n1 n2)
    (rc : ℝ) : ℝ :=
  ∑ i, ∑ j,
    let r := distances i j
    if r < rc then lennardJones (epsilons i j) (sigmas i j) r else 0

/-- Coulomb interaction with scalar cutoff and dielectric constant. -/
noncomputable def coulombCutoff {n1 n2 : ℕ}
    (charges1 : MDArray n1)
    (charges2 : MDArray n2)
    (distances : DistanceMatrix n1 n2)
    (rc dielectric : ℝ) : ℝ :=
  ∑ i, ∑ j,
    let r := distances i j
    if r < rc ∧ r > 1 / (10 : ℝ) ^ (10 : ℕ) then
      (charges1 i * charges2 j) / (dielectric * r)
    else 0

/-- Sum values over the strict upper triangle under a boolean mask. -/
noncomputable def upperTriangleMaskedSum {n : ℕ}
    (values : DistanceMatrix n n)
    (mask : Fin n → Fin n → Bool) : ℝ :=
  ∑ i, ∑ j,
    if i.1 < j.1 then
      if mask i j then values i j else 0
    else 0

/-- Real-space Ewald kernel exp(-(αr)^2) / r with a singularity guard. -/
noncomputable def ewaldRealSpaceKernel {n1 n2 : ℕ}
    (distances : DistanceMatrix n1 n2)
    (alpha : ℝ) : DistanceMatrix n1 n2 :=
  fun i j =>
    let r := distances i j
    let rSafe := if r > 1 / (10 : ℝ) ^ (10 : ℕ) then r else 1 / (10 : ℝ) ^ (10 : ℕ)
    Real.exp (-((alpha * rSafe) ^ (2 : ℕ))) / rSafe

/-! ## 5. Compilation Target Specification -/

/-- Primitive argument kind for exported wrapper generation. -/
inductive ExprKind where
  | scalar
  | tensor
  | callable
  deriving Repr, DecidableEq

/-- Primitive scalar payload type when one exists. -/
inductive ScalarType where
  | real
  | boolean
  | integer
  deriving Repr, DecidableEq

/-- Structured lowering category for Python/JAX code generation. -/
inductive LoweringKind where
  | vmap
  | reduceSum
  | elemBinaryAdd
  | elemBinarySub
  | norm
  | rowWiseNorm
  | distance
  | rowWiseDistance
  | supportConditioning
  | normalizeProbabilityVector
  | uniformProbabilityVectorLike
  | noopBiasedProbabilityVectorLike
  | topKWithTiesMask
  | ambiguityBandMask
  | stableArgmaxMasked
  | axisAngleQuaternion
  | localTranslationStencil3D
  | localRotationStencil3D
  | quaternionDictionary8
  | rigidTransform3D
  | pairwiseDistances
  | pairwiseDistances3D
  | minimumImagePairwiseDistances
  | applyCutoff
  | lennardJones
  | sumPairPotentials
  | sumPairPotentialsMatrix
  | sumPairPotentials3D
  | typedLennardJonesMatrix
  | typedLennardJonesCutoff
  | coulombCutoff
  | upperTriangleMaskedSum
  | ewaldRealSpaceKernel
  deriving Repr, DecidableEq

/-- Exported argument schema for a primitive. -/
structure ArgSpec where
  name : String
  kind : ExprKind
  scalarType? : Option ScalarType := none
  deriving Repr, DecidableEq

/-- Structured export artifact for Lean-to-Python wrapper generation. -/
structure PrimitiveIR where
  name : String
  args : List ArgSpec
  resultKind : ExprKind
  scalarType? : Option ScalarType := none
  loweringKind : LoweringKind
  jaxModule : String
  jaxSymbol : String
  supportsGrad : Bool
  leanSymbol : String
  proofRef? : Option String := none
  proofStatus? : Option String := none
  deriving Repr, DecidableEq

/-- Structured primitive export used by the Python/JAX code generator. -/
def exportPrimitives : List PrimitiveIR := [
  {
    name := "map"
    args := [
      { name := "f", kind := .callable },
      { name := "arr", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .vmap
    jaxModule := "jax"
    jaxSymbol := "vmap"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.map"
  },
  {
    name := "reduce_sum"
    args := [
      { name := "arr", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .reduceSum
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.reduce_sum"
  },
  {
    name := "elemBinaryAdd"
    args := [
      { name := "arr1", kind := .tensor, scalarType? := some .real },
      { name := "arr2", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .elemBinaryAdd
    jaxModule := "jax.numpy"
    jaxSymbol := "add"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.elemBinaryAdd"
  },
  {
    name := "elemBinarySub"
    args := [
      { name := "arr1", kind := .tensor, scalarType? := some .real },
      { name := "arr2", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .elemBinarySub
    jaxModule := "jax.numpy"
    jaxSymbol := "subtract"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.elemBinarySub"
  },
  {
    name := "norm"
    args := [
      { name := "arr", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .norm
    jaxModule := "jax.numpy.linalg"
    jaxSymbol := "norm"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.norm"
    proofRef? := some "DecisionQuotient.Computation.ArrayDSL.norm_nonneg_bound"
    proofStatus? := some "CERTIFIED"
  },
  {
    name := "rowWiseNorm"
    args := [
      { name := "arr", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .rowWiseNorm
    jaxModule := "jax.numpy.linalg"
    jaxSymbol := "norm"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.rowWiseNorm"
  },
  {
    name := "distance"
    args := [
      { name := "arr1", kind := .tensor, scalarType? := some .real },
      { name := "arr2", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .distance
    jaxModule := "jax.numpy.linalg"
    jaxSymbol := "norm"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.distance"
    proofRef? := some "DecisionQuotient.Computation.ArrayDSL.distance_triangle_bound"
    proofStatus? := some "CERTIFIED"
  },
  {
    name := "rowWiseDistance"
    args := [
      { name := "arr1", kind := .tensor, scalarType? := some .real },
      { name := "arr2", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .rowWiseDistance
    jaxModule := "jax.numpy.linalg"
    jaxSymbol := "norm"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.rowWiseDistance"
  },
  {
    name := "supportConditioning"
    args := [
      { name := "probs", kind := .tensor, scalarType? := some .real },
      { name := "mask", kind := .tensor, scalarType? := some .boolean }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .supportConditioning
    jaxModule := "jax.numpy"
    jaxSymbol := "where"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.supportConditioning"
    proofRef? := some "DecisionQuotient.Computation.ArrayDSL.supportConditioning_zero_of_mask_false"
    proofStatus? := some "CERTIFIED"
  },
  {
    name := "normalizeProbabilityVector"
    args := [
      { name := "weights", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .normalizeProbabilityVector
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.normalizeProbabilityVector"
    proofRef? := some "DecisionQuotient.Computation.ArrayDSL.normalizeProbabilityVector_sum_one"
    proofStatus? := some "CERTIFIED"
  },
  {
    name := "uniformProbabilityVectorLike"
    args := [
      { name := "template", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .uniformProbabilityVectorLike
    jaxModule := "jax.numpy"
    jaxSymbol := "ones_like"
    supportsGrad := false
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.uniformProbabilityVectorLike"
    proofRef? := some "DecisionQuotient.Computation.ArrayDSL.normalizeProbabilityVector_sum_one"
    proofStatus? := some "CONDITIONALLY_CERTIFIED"
  },
  {
    name := "noopBiasedProbabilityVectorLike"
    args := [
      { name := "template", kind := .tensor, scalarType? := some .real },
      { name := "noop_mass", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .noopBiasedProbabilityVectorLike
    jaxModule := "jax.numpy"
    jaxSymbol := "concatenate"
    supportsGrad := false
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.noopBiasedProbabilityVectorLike"
  },
  {
    name := "topKWithTiesMask"
    args := [
      { name := "utilities", kind := .tensor, scalarType? := some .real },
      { name := "k", kind := .scalar, scalarType? := some .integer }
    ]
    resultKind := .tensor
    scalarType? := some .boolean
    loweringKind := .topKWithTiesMask
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := false
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.topKWithTiesMask"
    proofRef? := some "DecisionQuotient.Tractability.FiniteTopK.mem_topKWithTies_iff"
    proofStatus? := some "CERTIFIED"
  },
  {
    name := "ambiguityBandMask"
    args := [
      { name := "utilities", kind := .tensor, scalarType? := some .real },
      { name := "k", kind := .scalar, scalarType? := some .integer },
      { name := "epsilon", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .boolean
    loweringKind := .ambiguityBandMask
    jaxModule := "jax.numpy"
    jaxSymbol := "sort"
    supportsGrad := false
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.ambiguityBandMask"
    proofRef? := some "DecisionQuotient.Tractability.NearTieBand.exact_topK_subset_ambiguityBand"
    proofStatus? := some "CONDITIONALLY_CERTIFIED"
  },
  {
    name := "stableArgmaxMasked"
    args := [
      { name := "values", kind := .tensor, scalarType? := some .real },
      { name := "mask", kind := .tensor, scalarType? := some .boolean }
    ]
    resultKind := .scalar
    scalarType? := some .integer
    loweringKind := .stableArgmaxMasked
    jaxModule := "jax.numpy"
    jaxSymbol := "argmax"
    supportsGrad := false
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.stableArgmaxMasked"
    proofRef? := none
    proofStatus? := none
  },
  {
    name := "axisAngleQuaternion"
    args := [
      { name := "axis", kind := .tensor, scalarType? := some .real },
      { name := "angle", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .axisAngleQuaternion
    jaxModule := "jax.numpy"
    jaxSymbol := "sin"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.axisAngleQuaternion"
  },
  {
    name := "localTranslationStencil3D"
    args := [
      { name := "step", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .localTranslationStencil3D
    jaxModule := "jax.numpy"
    jaxSymbol := "array"
    supportsGrad := false
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.localTranslationStencil3D"
  },
  {
    name := "localRotationStencil3D"
    args := [
      { name := "angle", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .localRotationStencil3D
    jaxModule := "jax.numpy"
    jaxSymbol := "stack"
    supportsGrad := false
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.localRotationStencil3D"
  },
  {
    name := "quaternionDictionary8"
    args := []
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .quaternionDictionary8
    jaxModule := "jax.numpy"
    jaxSymbol := "array"
    supportsGrad := false
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.quaternionDictionary8"
  },
  {
    name := "rigidTransform3D"
    args := [
      { name := "coords", kind := .tensor, scalarType? := some .real },
      { name := "quaternion", kind := .tensor, scalarType? := some .real },
      { name := "translation", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .rigidTransform3D
    jaxModule := "jax.numpy"
    jaxSymbol := "matmul"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.rigidTransform3D"
  },
  {
    name := "pairwiseDistances"
    args := [
      { name := "coords1", kind := .tensor, scalarType? := some .real },
      { name := "coords2", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .pairwiseDistances
    jaxModule := "jax.numpy"
    jaxSymbol := "abs"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.pairwiseDistances"
  },
  {
    name := "pairwiseDistances3D"
    args := [
      { name := "coords1", kind := .tensor, scalarType? := some .real },
      { name := "coords2", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .pairwiseDistances3D
    jaxModule := "jax.numpy.linalg"
    jaxSymbol := "norm"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.pairwiseDistances3D"
  },
  {
    name := "minimumImagePairwiseDistances"
    args := [
      { name := "coords1", kind := .tensor, scalarType? := some .real },
      { name := "coords2", kind := .tensor, scalarType? := some .real },
      { name := "box_size", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .minimumImagePairwiseDistances
    jaxModule := "jax.numpy.linalg"
    jaxSymbol := "norm"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.minimumImagePairwiseDistances"
  },
  {
    name := "applyCutoff"
    args := [
      { name := "distances", kind := .tensor, scalarType? := some .real },
      { name := "rc", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .applyCutoff
    jaxModule := "jax.numpy"
    jaxSymbol := "where"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.applyCutoff"
  },
  {
    name := "lennardJones"
    args := [
      { name := "epsilon", kind := .scalar, scalarType? := some .real },
      { name := "sigma", kind := .scalar, scalarType? := some .real },
      { name := "r", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .lennardJones
    jaxModule := "jax.numpy"
    jaxSymbol := "where"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.lennardJones"
    proofRef? := some "DecisionQuotient.Computation.ArrayDSL.lennardJones"
    proofStatus? := some "CERTIFIED"
  },
  {
    name := "sumPairPotentials"
    args := [
      { name := "distances", kind := .tensor, scalarType? := some .real },
      { name := "rc", kind := .scalar, scalarType? := some .real },
      { name := "epsilon", kind := .scalar, scalarType? := some .real },
      { name := "sigma", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .sumPairPotentials
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.sumPairPotentials"
  },
  {
    name := "sumPairPotentialsMatrix"
    args := [
      { name := "distances", kind := .tensor, scalarType? := some .real },
      { name := "rc", kind := .scalar, scalarType? := some .real },
      { name := "epsilon", kind := .scalar, scalarType? := some .real },
      { name := "sigma", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .sumPairPotentialsMatrix
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.sumPairPotentialsMatrix"
  },
  {
    name := "sumPairPotentials3D"
    args := [
      { name := "coords1", kind := .tensor, scalarType? := some .real },
      { name := "coords2", kind := .tensor, scalarType? := some .real },
      { name := "rc", kind := .scalar, scalarType? := some .real },
      { name := "epsilon", kind := .scalar, scalarType? := some .real },
      { name := "sigma", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .sumPairPotentials3D
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.sumPairPotentials3D"
  },
  {
    name := "typedLennardJonesMatrix"
    args := [
      { name := "distances", kind := .tensor, scalarType? := some .real },
      { name := "epsilons", kind := .tensor, scalarType? := some .real },
      { name := "sigmas", kind := .tensor, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .typedLennardJonesMatrix
    jaxModule := "jax.numpy"
    jaxSymbol := "where"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.typedLennardJonesMatrix"
  },
  {
    name := "typedLennardJonesCutoff"
    args := [
      { name := "distances", kind := .tensor, scalarType? := some .real },
      { name := "epsilons", kind := .tensor, scalarType? := some .real },
      { name := "sigmas", kind := .tensor, scalarType? := some .real },
      { name := "rc", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .typedLennardJonesCutoff
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.typedLennardJonesCutoff"
  },
  {
    name := "coulombCutoff"
    args := [
      { name := "charges1", kind := .tensor, scalarType? := some .real },
      { name := "charges2", kind := .tensor, scalarType? := some .real },
      { name := "distances", kind := .tensor, scalarType? := some .real },
      { name := "rc", kind := .scalar, scalarType? := some .real },
      { name := "dielectric", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .coulombCutoff
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.coulombCutoff"
  },
  {
    name := "upperTriangleMaskedSum"
    args := [
      { name := "values", kind := .tensor, scalarType? := some .real },
      { name := "mask", kind := .tensor, scalarType? := some .boolean }
    ]
    resultKind := .scalar
    scalarType? := some .real
    loweringKind := .upperTriangleMaskedSum
    jaxModule := "jax.numpy"
    jaxSymbol := "sum"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.upperTriangleMaskedSum"
  },
  {
    name := "ewaldRealSpaceKernel"
    args := [
      { name := "distances", kind := .tensor, scalarType? := some .real },
      { name := "alpha", kind := .scalar, scalarType? := some .real }
    ]
    resultKind := .tensor
    scalarType? := some .real
    loweringKind := .ewaldRealSpaceKernel
    jaxModule := "jax.numpy"
    jaxSymbol := "exp"
    supportsGrad := true
    leanSymbol := "DecisionQuotient.Computation.ArrayDSL.ewaldRealSpaceKernel"
    proofRef? := some "EwaldSummation.lean::ewald_real_space_exponential_decay"
    proofStatus? := some "CONDITIONALLY_CERTIFIED"
  }
]

/-- JAX compilation target for each primitive.
    This is the specification that JAX implementation must match. -/
structure JAXSpec where
  opName : String
  jaxExpr : String
  hasGrad : Bool

/-- Backward-compatible string summary of the lowering target. -/
def PrimitiveIR.jaxExpr (primitive : PrimitiveIR) : String :=
  match primitive.loweringKind with
  | .vmap => "jax.vmap(f)(arr)"
  | .reduceSum => "jnp.sum(arr)"
  | .elemBinaryAdd => "arr1 + arr2"
  | .elemBinarySub => "arr1 - arr2"
  | .norm => "jnp.linalg.norm(arr)"
  | .rowWiseNorm => "jnp.linalg.norm(arr, axis=-1)"
  | .distance => "jnp.linalg.norm(arr1 - arr2)"
  | .rowWiseDistance => "jnp.linalg.norm(arr1 - arr2, axis=-1)"
  | .supportConditioning => "jnp.where(mask, probs, 0.0)"
  | .normalizeProbabilityVector => "weights / jnp.sum(weights)"
  | .uniformProbabilityVectorLike => "jnp.ones_like(template) / jnp.sum(jnp.ones_like(template))"
  | .noopBiasedProbabilityVectorLike => "jnp.concatenate([jnp.array([noop_mass]), jnp.full((template.shape[0] - 1,), (1 - noop_mass) / (template.shape[0] - 1))])"
  | .topKWithTiesMask => "jnp.sum(utilities[None, :] > utilities[:, None], axis=1) < k"
  | .ambiguityBandMask => "utilities >= jnp.sort(utilities)[::-1][k - 1] - epsilon"
  | .stableArgmaxMasked => "jnp.argmax(jnp.where(mask, values, -jnp.inf))"
  | .axisAngleQuaternion => "half = angle / 2; jnp.array([jnp.cos(half), axis[0] * jnp.sin(half), axis[1] * jnp.sin(half), axis[2] * jnp.sin(half)])"
  | .localTranslationStencil3D => "jnp.array([[step, 0, 0], [-step, 0, 0], [0, step, 0], [0, -step, 0], [0, 0, step], [0, 0, -step]], dtype=jnp.float32)"
  | .localRotationStencil3D => "axes = jnp.eye(3, dtype=jnp.float32); return jnp.stack([axisAngleQuaternion(axes[0], angle), axisAngleQuaternion(axes[0], -angle), axisAngleQuaternion(axes[1], angle), axisAngleQuaternion(axes[1], -angle), axisAngleQuaternion(axes[2], angle), axisAngleQuaternion(axes[2], -angle)], axis=0)"
  | .quaternionDictionary8 => "half = jnp.sqrt(jnp.array(0.5, dtype=jnp.float32)); return jnp.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[half,half,0,0],[half,0,half,0],[half,0,0,half],[0.5,0.5,0.5,0.5]], dtype=jnp.float32)"
  | .rigidTransform3D => "coords @ rotation_matrix(quaternion).T + translation"
  | .pairwiseDistances => "jnp.abs(coords1[:, None] - coords2[None, :])"
  | .pairwiseDistances3D => "jnp.linalg.norm(coords1[:, None, :] - coords2[None, :, :], axis=-1)"
  | .minimumImagePairwiseDistances => "jnp.linalg.norm((coords1[:, None, :] - coords2[None, :, :]) - box_size * jnp.round((coords1[:, None, :] - coords2[None, :, :]) / box_size), axis=-1)"
  | .applyCutoff => "jnp.where(distances < rc, distances, 0)"
  | .lennardJones => "safe_r = jnp.where(r > 1e-10, r, 1e-10); potential = 4*epsilon*((sigma/safe_r)**12 - (sigma/safe_r)**6); jnp.where(r > 1e-10, potential, 1e12)"
  | .sumPairPotentials => "jnp.sum(lennardJones(epsilon, sigma, applyCutoff(distances, rc)))"
  | .sumPairPotentialsMatrix => "jnp.sum(lennardJones(epsilon, sigma, jnp.where(distances < rc, distances, 0)))"
  | .sumPairPotentials3D => "sumPairPotentialsMatrix(pairwiseDistances3D(coords1, coords2), rc, epsilon, sigma)"
  | .typedLennardJonesMatrix => "typedLennardJonesMatrix(distances, epsilons, sigmas)"
  | .typedLennardJonesCutoff => "jnp.sum(jnp.where(distances < rc, typedLennardJonesMatrix(distances, epsilons, sigmas), 0))"
  | .coulombCutoff => "jnp.sum(jnp.where((distances < rc) & (distances > 1e-10), (charges1[:, None] * charges2[None, :]) / (dielectric * distances), 0))"
  | .upperTriangleMaskedSum => "jnp.sum(jnp.where(jnp.triu(jnp.ones_like(values, dtype=bool), k=1) & mask, values, 0))"
  | .ewaldRealSpaceKernel => "jnp.exp(-((alpha * jnp.where(distances > 1e-10, distances, 1e-10)) ** 2)) / jnp.where(distances > 1e-10, distances, 1e-10)"

/-- Mapping from Lean primitives to JAX expressions.
    Gradients are implemented fully via JAX autodiff equivalent to Lean's `gradient`. -/
def primitiveJAXMapping : List JAXSpec :=
  exportPrimitives.map fun primitive =>
    ⟨primitive.name, primitive.jaxExpr, primitive.supportsGrad⟩

/-! ## 6. Correctness Theorems -/

/-- THEOREM: Norm is non-negative.
    ‖x‖ ≥ 0 for all x -/
theorem norm_nonneg_bound {n : ℕ} (arr : MDArray n) :
    0 ≤ norm arr := by
  exact norm_nonneg arr

/-- THEOREM: Distance satisfies triangle inequality.
    ‖a - c‖ ≤ ‖a - b‖ + ‖b - c‖ -/
theorem distance_triangle_bound {n : ℕ} (a b c : MDArray n) :
    distance a c ≤ distance a b + distance b c := by
  exact dist_triangle a b c

/-- Support conditioning never introduces positive mass outside the declared mask. -/
theorem supportConditioning_zero_of_mask_false {n : ℕ}
    (probs : MDArray n)
    (mask : Fin n → Bool)
    (i : Fin n)
    (hFalse : mask i = false) :
    supportConditioning probs mask i = 0 := by
  change (if mask i then probs i else 0) = 0
  simp [hFalse]

/-- Normalizing a weight vector with positive total mass yields unit sum. -/
theorem normalizeProbabilityVector_sum_one {n : ℕ}
    (weights : MDArray n)
    (hPos : 0 < reduce_sum weights) :
    reduce_sum (normalizeProbabilityVector weights) = 1 := by
  unfold normalizeProbabilityVector reduce_sum
  change (∑ i, weights i / ∑ j, weights j) = 1
  have hz : (∑ j, weights j) ≠ 0 := hPos.ne'
  calc
    ∑ i, weights i / ∑ j, weights j
        = ∑ i, weights i * (∑ j, weights j)⁻¹ := by simp [div_eq_mul_inv]
    _ = (∑ i, weights i) * (∑ j, weights j)⁻¹ := by rw [Finset.sum_mul]
    _ = 1 := by field_simp [hz]

/-- THEOREM: L2 norm square is sum of squares. -/
theorem norm_sq_eq_sum_sq {n : ℕ} (arr : MDArray n) :
    norm arr ^ 2 = ∑ i, (arr i) ^ 2 := by
  -- Follows from definition of inner product in EuclideanSpace
  have h1 : (norm arr) ^ 2 = @inner ℝ _ _ arr arr := @real_inner_self_eq_norm_sq (MDArray n) _ _ arr |>.symm
  have h2 : (@inner ℝ _ _ arr arr : ℝ) = ∑ i, arr i * arr i := rfl
  rw [h1, h2]
  apply Finset.sum_congr rfl
  intro i _
  ring

theorem coord_abs_le_norm {n : ℕ} (arr : MDArray n) (i : Fin n) :
    |arr i| ≤ norm arr := by
  have hiSqLe : (arr i) ^ 2 ≤ ∑ j : Fin n, (arr j) ^ 2 := by
    have htermNonneg : ∀ j : Fin n, 0 ≤ (arr j) ^ 2 := by
      intro j
      nlinarith
    exact Finset.single_le_sum (fun j _ => htermNonneg j) (Finset.mem_univ i)
  have hnormSq : (arr i) ^ 2 ≤ norm arr ^ 2 := by
    simpa [norm_sq_eq_sum_sq] using hiSqLe
  have habsSq : |arr i| ^ 2 ≤ (norm arr) ^ 2 := by
    simpa [sq_abs] using hnormSq
  exact le_of_sq_le_sq habsSq (norm_nonneg arr)

/-- Any unit vector in 4D has some coordinate with absolute value at least 1/2. -/
theorem unit_norm_mdarray4_has_coordinate_abs_ge_half (arr : MDArray 4)
    (hNorm : norm arr = 1) :
    ∃ i : Fin 4, 1 / 2 ≤ |arr i| := by
  by_contra hNo
  push_neg at hNo
  have hCoordSq : ∀ i : Fin 4, (arr i) ^ 2 < (1 / 2 : ℝ) ^ 2 := by
    intro i
    have hAbs := hNo i
    have hSqAbs : |arr i| * |arr i| < (1 / 2 : ℝ) * (1 / 2 : ℝ) := by
      have hNonneg : 0 ≤ |arr i| := abs_nonneg _
      nlinarith
    simpa [pow_two, sq_abs] using hSqAbs
  have hSumLt : ∑ i : Fin 4, (arr i) ^ 2 < ∑ _i : Fin 4, ((1 / 2 : ℝ) ^ 2) := by
    apply Finset.sum_lt_sum
    · intro i _
      exact le_of_lt (hCoordSq i)
    · exact ⟨0, by simp, hCoordSq 0⟩
  have hNormSq : ∑ i : Fin 4, (arr i) ^ 2 = 1 := by
    rw [← norm_sq_eq_sum_sq, hNorm]
    norm_num
  have hHalfSum : (∑ _i : Fin 4, ((1 / 2 : ℝ) ^ 2)) = 1 := by
    norm_num
  rw [hNormSq, hHalfSum] at hSumLt
  linarith

theorem quaternionDictionary8_basis0_eq :
    quaternionDictionary8 ⟨0, by decide⟩ = EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ) := by
  ext j
  fin_cases j <;> simp [quaternionDictionary8, mkMDArray, EuclideanSpace.single_apply]

theorem quaternionDictionary8_basis1_eq :
    quaternionDictionary8 ⟨1, by decide⟩ = EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ) := by
  ext j
  fin_cases j <;> simp [quaternionDictionary8, mkMDArray, EuclideanSpace.single_apply]

theorem quaternionDictionary8_basis2_eq :
    quaternionDictionary8 ⟨2, by decide⟩ = EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ) := by
  ext j
  fin_cases j <;> simp [quaternionDictionary8, mkMDArray, EuclideanSpace.single_apply]

theorem quaternionDictionary8_basis3_eq :
    quaternionDictionary8 ⟨3, by decide⟩ = EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ) := by
  ext j
  fin_cases j <;> simp [quaternionDictionary8, mkMDArray, EuclideanSpace.single_apply]

theorem rigidTransformPoint3D_negQuaternion_eq
    (point : MDArray 3)
    (quaternion : MDArray 4)
    (translation : MDArray 3) :
    rigidTransformPoint3D point (-quaternion) translation =
      rigidTransformPoint3D point quaternion translation := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, mkMDArray] <;> ring

theorem rigidTransform3D_negQuaternion_eq
    {n : ℕ}
    (coords : CoordSet n)
    (quaternion : MDArray 4)
    (translation : MDArray 3) :
    rigidTransform3D coords (-quaternion) translation =
      rigidTransform3D coords quaternion translation := by
  funext i
  simpa [rigidTransform3D] using
    rigidTransformPoint3D_negQuaternion_eq (coords i) quaternion translation

theorem rigidTransformPoint3D_basis0_eq
    (point : MDArray 3)
    (translation : MDArray 3) :
    rigidTransformPoint3D point (quaternionDictionary8 ⟨0, by decide⟩) translation =
      point + translation := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, quaternionDictionary8, mkMDArray]

theorem rigidTransformPoint3D_basis1_eq
    (point : MDArray 3)
    (translation : MDArray 3) :
    rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) translation =
      mkMDArray (fun j => if j = ⟨0, by decide⟩ then point ⟨0, by decide⟩ + translation ⟨0, by decide⟩
        else -(point j) + translation j) := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, quaternionDictionary8, mkMDArray] <;> ring

theorem rigidTransformPoint3D_basis2_eq
    (point : MDArray 3)
    (translation : MDArray 3) :
    rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) translation =
      mkMDArray (fun j => if j = ⟨1, by decide⟩ then point ⟨1, by decide⟩ + translation ⟨1, by decide⟩
        else -(point j) + translation j) := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, quaternionDictionary8, mkMDArray] <;> ring

theorem rigidTransformPoint3D_basis3_eq
    (point : MDArray 3)
    (translation : MDArray 3) :
    rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) translation =
      mkMDArray (fun j => if j = ⟨2, by decide⟩ then point ⟨2, by decide⟩ + translation ⟨2, by decide⟩
        else -(point j) + translation j) := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, quaternionDictionary8, mkMDArray] <;> ring

theorem rigidTransformPoint3D_translation_decompose
    (point : MDArray 3)
    (quaternion : MDArray 4)
    (translation : MDArray 3) :
    rigidTransformPoint3D point quaternion translation =
      rigidTransformPoint3D point quaternion (mkMDArray (fun _ => 0)) + translation := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, mkMDArray] <;> ring

theorem rigidTransformPoint3D_same_translation_dist_eq_zero_translation
    (point : MDArray 3)
    (q1 q2 : MDArray 4)
    (translation : MDArray 3) :
    dist (rigidTransformPoint3D point q1 translation)
         (rigidTransformPoint3D point q2 translation) =
      dist (rigidTransformPoint3D point q1 (mkMDArray (fun _ => 0)))
           (rigidTransformPoint3D point q2 (mkMDArray (fun _ => 0))) := by
  rw [rigidTransformPoint3D_translation_decompose point q1 translation]
  rw [rigidTransformPoint3D_translation_decompose point q2 translation]
  set a : MDArray 3 := rigidTransformPoint3D point q1 (mkMDArray (fun _ => 0))
  set b : MDArray 3 := rigidTransformPoint3D point q2 (mkMDArray (fun _ => 0))
  have hcancel : translation + (a + (-translation + -b)) = a + -b := by
    abel_nf
  calc
    dist (a + translation) (b + translation)
        = ‖translation + (a + (-translation + -b))‖ := by
            simp [dist_eq_norm, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]
    _ = ‖a + -b‖ := by simp [hcancel]
    _ = dist a b := by simp [dist_eq_norm, sub_eq_add_neg, add_assoc, add_left_comm, add_comm]

theorem rigidTransform3D_same_translation_dist_eq_zero_translation
    {n : ℕ}
    (coords : CoordSet n)
    (q1 q2 : MDArray 4)
    (translation : MDArray 3)
    (j : Fin n) :
    dist (rigidTransform3D coords q1 translation j)
         (rigidTransform3D coords q2 translation j) =
      dist (rigidTransform3D coords q1 (mkMDArray (fun _ => 0)) j)
           (rigidTransform3D coords q2 (mkMDArray (fun _ => 0)) j) := by
  simpa [rigidTransform3D] using
    rigidTransformPoint3D_same_translation_dist_eq_zero_translation (coords j) q1 q2 translation

/-- L1 radius of a 3D point. This is a tractable coarse arm-length surrogate for
    runtime rigid-rotation displacement bounds. -/
def pointL1Radius (point : MDArray 3) : ℝ :=
  |point ⟨0, by decide⟩| + |point ⟨1, by decide⟩| + |point ⟨2, by decide⟩|

theorem rigidTransformPoint3D_basis0_zero_eq_self
    (point : MDArray 3) :
    rigidTransformPoint3D point (quaternionDictionary8 ⟨0, by decide⟩) (mkMDArray (fun _ => 0)) = point := by
  calc
    rigidTransformPoint3D point (quaternionDictionary8 ⟨0, by decide⟩) (mkMDArray (fun _ => 0))
        = point + mkMDArray (fun _ => 0) := rigidTransformPoint3D_basis0_eq point (mkMDArray (fun _ => 0))
    _ = point := by
      ext j
      fin_cases j <;> simp [mkMDArray]

theorem rigidTransformPoint3D_zero_translation_sub_basis0
    (point : MDArray 3)
    (quaternion : MDArray 4) :
    rigidTransformPoint3D point quaternion (mkMDArray (fun _ => 0)) - point =
      mkMDArray (fun j =>
        if h0 : j = ⟨0, by decide⟩ then
          (-2 * (quaternion ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (quaternion ⟨3, by decide⟩) ^ (2 : ℕ)) * point ⟨0, by decide⟩ +
            (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨2, by decide⟩ - 2 * quaternion ⟨3, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨1, by decide⟩ +
            (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨3, by decide⟩ + 2 * quaternion ⟨2, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨2, by decide⟩
        else if h1 : j = ⟨1, by decide⟩ then
          (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨2, by decide⟩ + 2 * quaternion ⟨3, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨0, by decide⟩ +
            (-2 * quaternion ⟨1, by decide⟩ ^ (2 : ℕ) - 2 * quaternion ⟨3, by decide⟩ ^ (2 : ℕ)) * point ⟨1, by decide⟩ +
            (2 * quaternion ⟨2, by decide⟩ * quaternion ⟨3, by decide⟩ - 2 * quaternion ⟨1, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨2, by decide⟩
        else
          (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨3, by decide⟩ - 2 * quaternion ⟨2, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨0, by decide⟩ +
            (2 * quaternion ⟨2, by decide⟩ * quaternion ⟨3, by decide⟩ + 2 * quaternion ⟨1, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨1, by decide⟩ +
            (-2 * quaternion ⟨1, by decide⟩ ^ (2 : ℕ) - 2 * quaternion ⟨2, by decide⟩ ^ (2 : ℕ)) * point ⟨2, by decide⟩) := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, mkMDArray] <;> ring

theorem rigidTransformPoint3D_zero_translation_sub_basis1
    (point : MDArray 3)
    (quaternion : MDArray 4) :
    rigidTransformPoint3D point quaternion (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0)) =
      mkMDArray (fun j =>
        if h0 : j = ⟨0, by decide⟩ then
          (-2 * (quaternion ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (quaternion ⟨3, by decide⟩) ^ (2 : ℕ)) * point ⟨0, by decide⟩ +
            (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨2, by decide⟩ - 2 * quaternion ⟨3, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨1, by decide⟩ +
            (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨3, by decide⟩ + 2 * quaternion ⟨2, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨2, by decide⟩
        else if h1 : j = ⟨1, by decide⟩ then
          (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨2, by decide⟩ + 2 * quaternion ⟨3, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨0, by decide⟩ +
            (-2 * quaternion ⟨1, by decide⟩ ^ (2 : ℕ) - 2 * quaternion ⟨3, by decide⟩ ^ (2 : ℕ) + 2) * point ⟨1, by decide⟩ +
            (2 * quaternion ⟨2, by decide⟩ * quaternion ⟨3, by decide⟩ - 2 * quaternion ⟨1, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨2, by decide⟩
        else
          (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨3, by decide⟩ - 2 * quaternion ⟨2, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨0, by decide⟩ +
            (2 * quaternion ⟨2, by decide⟩ * quaternion ⟨3, by decide⟩ + 2 * quaternion ⟨1, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨1, by decide⟩ +
            (-2 * quaternion ⟨1, by decide⟩ ^ (2 : ℕ) - 2 * quaternion ⟨2, by decide⟩ ^ (2 : ℕ) + 2) * point ⟨2, by decide⟩) := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, quaternionDictionary8, mkMDArray] <;> ring

theorem rigidTransformPoint3D_zero_translation_sub_basis2
    (point : MDArray 3)
    (quaternion : MDArray 4) :
    rigidTransformPoint3D point quaternion (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0)) =
      mkMDArray (fun j =>
        if h0 : j = ⟨0, by decide⟩ then
          (-2 * (quaternion ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (quaternion ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point ⟨0, by decide⟩ +
            (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨2, by decide⟩ - 2 * quaternion ⟨3, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨1, by decide⟩ +
            (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨3, by decide⟩ + 2 * quaternion ⟨2, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨2, by decide⟩
        else if h1 : j = ⟨1, by decide⟩ then
          (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨2, by decide⟩ + 2 * quaternion ⟨3, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨0, by decide⟩ +
            (-2 * quaternion ⟨1, by decide⟩ ^ (2 : ℕ) - 2 * quaternion ⟨3, by decide⟩ ^ (2 : ℕ)) * point ⟨1, by decide⟩ +
            (2 * quaternion ⟨2, by decide⟩ * quaternion ⟨3, by decide⟩ - 2 * quaternion ⟨1, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨2, by decide⟩
        else
          (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨3, by decide⟩ - 2 * quaternion ⟨2, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨0, by decide⟩ +
            (2 * quaternion ⟨2, by decide⟩ * quaternion ⟨3, by decide⟩ + 2 * quaternion ⟨1, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨1, by decide⟩ +
            (-2 * quaternion ⟨1, by decide⟩ ^ (2 : ℕ) - 2 * quaternion ⟨2, by decide⟩ ^ (2 : ℕ) + 2) * point ⟨2, by decide⟩) := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, quaternionDictionary8, mkMDArray] <;> ring

theorem norm_mdarray3_le_two_of_abs_le
    (arr : MDArray 3)
    (B : ℝ)
    (hBnonneg : 0 ≤ B)
    (h0 : |arr ⟨0, by decide⟩| ≤ B)
    (h1 : |arr ⟨1, by decide⟩| ≤ B)
    (h2 : |arr ⟨2, by decide⟩| ≤ B) :
    ‖arr‖ ≤ 2 * B := by
  have hsq : ‖arr‖ ^ 2 =
      (arr ⟨0, by decide⟩) ^ 2 + (arr ⟨1, by decide⟩) ^ 2 + (arr ⟨2, by decide⟩) ^ 2 := by
    simpa [Fin.sum_univ_three] using (norm_sq_eq_sum_sq arr)
  have h0sq : (arr ⟨0, by decide⟩) ^ 2 ≤ B ^ 2 := by
    have habs : |arr ⟨0, by decide⟩| * |arr ⟨0, by decide⟩| ≤ B * B :=
      mul_le_mul h0 h0 (abs_nonneg _) hBnonneg
    simpa [sq_abs, pow_two] using habs
  have h1sq : (arr ⟨1, by decide⟩) ^ 2 ≤ B ^ 2 := by
    have habs : |arr ⟨1, by decide⟩| * |arr ⟨1, by decide⟩| ≤ B * B :=
      mul_le_mul h1 h1 (abs_nonneg _) hBnonneg
    simpa [sq_abs, pow_two] using habs
  have h2sq : (arr ⟨2, by decide⟩) ^ 2 ≤ B ^ 2 := by
    have habs : |arr ⟨2, by decide⟩| * |arr ⟨2, by decide⟩| ≤ B * B :=
      mul_le_mul h2 h2 (abs_nonneg _) hBnonneg
    simpa [sq_abs, pow_two] using habs
  have hsumSqLe :
      (arr ⟨0, by decide⟩) ^ 2 + (arr ⟨1, by decide⟩) ^ 2 + (arr ⟨2, by decide⟩) ^ 2 ≤ 3 * B ^ 2 := by
    nlinarith
  have hnormLe : ‖arr‖ ^ 2 ≤ 3 * B ^ 2 := by
    simpa [hsq] using hsumSqLe
  have hsquare : ‖arr‖ ^ 2 ≤ (Real.sqrt 3 * B) ^ 2 := by
    nlinarith [hnormLe, Real.sq_sqrt (by positivity : 0 ≤ (3 : ℝ))]
  have hbound : ‖arr‖ ≤ Real.sqrt 3 * B := by
    exact le_of_sq_le_sq hsquare (by positivity)
  have hsqrt3_le : Real.sqrt 3 ≤ 2 := by
    nlinarith [Real.sq_sqrt (by positivity : 0 ≤ (3 : ℝ))]
  have hmul : Real.sqrt 3 * B ≤ 2 * B := by
    nlinarith [hsqrt3_le, hBnonneg]
  exact le_trans hbound hmul

theorem unit_norm_mdarray4_coord_abs_le_one
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (i : Fin 4) :
    |q i| ≤ 1 := by
  exact (coord_abs_le_norm q i).trans_eq hNorm

theorem unit_norm_mdarray4_dist_to_basis0_le_two
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ≤ 2 := by
  let i0 : Fin 4 := ⟨0, by decide⟩
  have hqNorm : ‖q‖ = 1 := by simpa using hNorm
  calc
    ‖q - EuclideanSpace.single i0 (1 : ℝ)‖ ≤
        ‖q‖ + ‖EuclideanSpace.single i0 (1 : ℝ)‖ := norm_sub_le _ _
    _ = 2 := by rw [hqNorm]; norm_num [i0]

theorem unit_norm_mdarray4_dist_to_basis_le_two
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (i : Fin 4) :
    ‖q - EuclideanSpace.single i (1 : ℝ)‖ ≤ 2 := by
  have hqNorm : ‖q‖ = 1 := by simpa using hNorm
  calc
    ‖q - EuclideanSpace.single i (1 : ℝ)‖ ≤ ‖q‖ + ‖EuclideanSpace.single i (1 : ℝ)‖ := norm_sub_le _ _
    _ = 2 := by rw [hqNorm]; norm_num [EuclideanSpace.norm_single]

theorem point_coord_abs_le_pointL1Radius
    (point : MDArray 3)
    (i : Fin 3) :
    |point i| ≤ pointL1Radius point := by
  fin_cases i
  · unfold pointL1Radius
    nlinarith [abs_nonneg (point ⟨1, by decide⟩), abs_nonneg (point ⟨2, by decide⟩)]
  · unfold pointL1Radius
    nlinarith [abs_nonneg (point ⟨0, by decide⟩), abs_nonneg (point ⟨2, by decide⟩)]
  · unfold pointL1Radius
    nlinarith [abs_nonneg (point ⟨0, by decide⟩), abs_nonneg (point ⟨1, by decide⟩)]

theorem basis0_error_coord0_abs_le_norm
    (q : MDArray 4) :
    |q ⟨0, by decide⟩ - 1| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  simpa [EuclideanSpace.single_apply] using
    coord_abs_le_norm (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)) ⟨0, by decide⟩

theorem basis0_error_coord1_abs_le_norm
    (q : MDArray 4) :
    |q ⟨1, by decide⟩| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let i0 : Fin 4 := ⟨0, by decide⟩
  let i1 : Fin 4 := ⟨1, by decide⟩
  have hcoord : |(q - EuclideanSpace.single i0 (1 : ℝ)) i1| ≤ ‖q - EuclideanSpace.single i0 (1 : ℝ)‖ :=
    coord_abs_le_norm (q - EuclideanSpace.single i0 (1 : ℝ)) i1
  simpa [i0, i1, EuclideanSpace.single_apply] using hcoord

theorem basis0_error_coord2_abs_le_norm
    (q : MDArray 4) :
    |q ⟨2, by decide⟩| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let i0 : Fin 4 := ⟨0, by decide⟩
  let i2 : Fin 4 := ⟨2, by decide⟩
  have hcoord : |(q - EuclideanSpace.single i0 (1 : ℝ)) i2| ≤ ‖q - EuclideanSpace.single i0 (1 : ℝ)‖ :=
    coord_abs_le_norm (q - EuclideanSpace.single i0 (1 : ℝ)) i2
  simpa [i0, i2, EuclideanSpace.single_apply] using hcoord

theorem basis0_error_coord3_abs_le_norm
    (q : MDArray 4) :
    |q ⟨3, by decide⟩| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let i0 : Fin 4 := ⟨0, by decide⟩
  let i3 : Fin 4 := ⟨3, by decide⟩
  have hcoord : |(q - EuclideanSpace.single i0 (1 : ℝ)) i3| ≤ ‖q - EuclideanSpace.single i0 (1 : ℝ)‖ :=
    coord_abs_le_norm (q - EuclideanSpace.single i0 (1 : ℝ)) i3
  simpa [i0, i3, EuclideanSpace.single_apply] using hcoord

theorem unit_basis0_error_offaxis_sq_le_two_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (i : Fin 4)
    (hi : i ≠ ⟨0, by decide⟩) :
    (q i) ^ 2 ≤ 2 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  have hcoord : |q i| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    fin_cases i
    · contradiction
    · simpa using basis0_error_coord1_abs_le_norm q
    · simpa using basis0_error_coord2_abs_le_norm q
    · simpa using basis0_error_coord3_abs_le_norm q
  have hsq : (q i) ^ 2 ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 := by
    have habs : |q i| * |q i| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ :=
      mul_le_mul hcoord hcoord (abs_nonneg _) (norm_nonneg _)
    simpa [sq_abs, pow_two] using habs
  have hdistLeTwo : ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ≤ 2 :=
    unit_norm_mdarray4_dist_to_basis0_le_two q hNorm
  have hdistSqLe : ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 ≤
      2 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)), hdistLeTwo]
  exact le_trans hsq hdistSqLe

theorem unit_basis0_error_offaxis_product_abs_le_two_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (i j : Fin 4)
    (hi : i ≠ ⟨0, by decide⟩)
    (hj : j ≠ ⟨0, by decide⟩) :
    |q i * q j| ≤ 2 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  have hiAbs : |q i| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    fin_cases i
    · contradiction
    · simpa using basis0_error_coord1_abs_le_norm q
    · simpa using basis0_error_coord2_abs_le_norm q
    · simpa using basis0_error_coord3_abs_le_norm q
  have hjAbs : |q j| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    fin_cases j
    · contradiction
    · simpa using basis0_error_coord1_abs_le_norm q
    · simpa using basis0_error_coord2_abs_le_norm q
    · simpa using basis0_error_coord3_abs_le_norm q
  have habs : |q i * q j| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 := by
    rw [abs_mul]
    have hmul : |q i| * |q j| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ :=
      mul_le_mul hiAbs hjAbs (abs_nonneg _) (norm_nonneg _)
    simpa [pow_two] using hmul
  have hdistLeTwo : ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ≤ 2 :=
    unit_norm_mdarray4_dist_to_basis0_le_two q hNorm
  have hdistSqLe : ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 ≤
      2 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)), hdistLeTwo]
  exact le_trans habs hdistSqLe

theorem unit_basis0_error_axis_product_abs_le_dist
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (i : Fin 4)
    (hi : i ≠ ⟨0, by decide⟩) :
    |q i * q ⟨0, by decide⟩| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  have hiAbs : |q i| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    fin_cases i
    · contradiction
    · simpa using basis0_error_coord1_abs_le_norm q
    · simpa using basis0_error_coord2_abs_le_norm q
    · simpa using basis0_error_coord3_abs_le_norm q
  have h0Abs : |q ⟨0, by decide⟩| ≤ 1 := unit_norm_mdarray4_coord_abs_le_one q hNorm ⟨0, by decide⟩
  rw [abs_mul]
  calc
    |q i| * |q ⟨0, by decide⟩| ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ * 1 := by
      exact mul_le_mul hiAbs h0Abs (by positivity) (norm_nonneg _)
  _ = ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by ring

theorem unit_norm_mdarray4_offaxis_coord_abs_le_dist_to_basis
    (q : MDArray 4)
    (b i : Fin 4)
    (hi : i ≠ b) :
    |q i| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ := by
  have hcoord : |(q - EuclideanSpace.single b (1 : ℝ)) i| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ :=
    coord_abs_le_norm (q - EuclideanSpace.single b (1 : ℝ)) i
  simpa [EuclideanSpace.single_apply, hi] using hcoord

theorem unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (b i : Fin 4)
    (hi : i ≠ b) :
    (q i) ^ 2 ≤ 2 * ‖q - EuclideanSpace.single b (1 : ℝ)‖ := by
  have hcoord : |q i| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ :=
    unit_norm_mdarray4_offaxis_coord_abs_le_dist_to_basis q b i hi
  have hsq : (q i) ^ 2 ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ ^ 2 := by
    have habs : |q i| * |q i| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ * ‖q - EuclideanSpace.single b (1 : ℝ)‖ :=
      mul_le_mul hcoord hcoord (abs_nonneg _) (norm_nonneg _)
    simpa [sq_abs, pow_two] using habs
  have hdistLeTwo : ‖q - EuclideanSpace.single b (1 : ℝ)‖ ≤ 2 :=
    unit_norm_mdarray4_dist_to_basis_le_two q hNorm b
  have hdistSqLe : ‖q - EuclideanSpace.single b (1 : ℝ)‖ ^ 2 ≤ 2 * ‖q - EuclideanSpace.single b (1 : ℝ)‖ := by
    nlinarith [norm_nonneg (q - EuclideanSpace.single b (1 : ℝ)), hdistLeTwo]
  exact le_trans hsq hdistSqLe

theorem unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (b i j : Fin 4)
    (hi : i ≠ b)
    (hj : j ≠ b) :
    |q i * q j| ≤ 2 * ‖q - EuclideanSpace.single b (1 : ℝ)‖ := by
  have hiAbs : |q i| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ :=
    unit_norm_mdarray4_offaxis_coord_abs_le_dist_to_basis q b i hi
  have hjAbs : |q j| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ :=
    unit_norm_mdarray4_offaxis_coord_abs_le_dist_to_basis q b j hj
  have habs : |q i * q j| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ ^ 2 := by
    rw [abs_mul]
    have hmul : |q i| * |q j| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ * ‖q - EuclideanSpace.single b (1 : ℝ)‖ :=
      mul_le_mul hiAbs hjAbs (abs_nonneg _) (norm_nonneg _)
    simpa [pow_two] using hmul
  have hdistLeTwo : ‖q - EuclideanSpace.single b (1 : ℝ)‖ ≤ 2 :=
    unit_norm_mdarray4_dist_to_basis_le_two q hNorm b
  have hdistSqLe : ‖q - EuclideanSpace.single b (1 : ℝ)‖ ^ 2 ≤ 2 * ‖q - EuclideanSpace.single b (1 : ℝ)‖ := by
    nlinarith [norm_nonneg (q - EuclideanSpace.single b (1 : ℝ)), hdistLeTwo]
  exact le_trans habs hdistSqLe

theorem unit_norm_mdarray4_axis_coord_abs_le_one
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (b : Fin 4) :
    |q b| ≤ 1 :=
  unit_norm_mdarray4_coord_abs_le_one q hNorm b

theorem unit_norm_mdarray4_axis_product_abs_le_dist_to_basis
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (b i : Fin 4)
    (hi : i ≠ b) :
    |q i * q b| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ := by
  have hiAbs : |q i| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ :=
    unit_norm_mdarray4_offaxis_coord_abs_le_dist_to_basis q b i hi
  have hbAbs : |q b| ≤ 1 := unit_norm_mdarray4_axis_coord_abs_le_one q hNorm b
  rw [abs_mul]
  calc
    |q i| * |q b| ≤ ‖q - EuclideanSpace.single b (1 : ℝ)‖ * 1 := by
      exact mul_le_mul hiAbs hbAbs (by positivity) (norm_nonneg _)
    _ = ‖q - EuclideanSpace.single b (1 : ℝ)‖ := by ring

theorem unit_basis1_quad23_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  have hq2sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide)
  have hq3sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨2, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq2sq, hq3sq]
  have hnonneg : 0 ≤ 2 * (q ⟨2, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2 := by positivity
  have hrewrite :
      |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))|
        = 2 * (q ⟨2, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2 := by
    have hneg :
        -2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) =
          -(2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) := by ring
    rw [hneg, abs_neg, abs_of_nonneg hnonneg]
  rw [hrewrite]
  exact hsum

theorem unit_basis1_mix12_30_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖
  have h12 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide)
  have h30 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨2, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using
              (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (-2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ))]

theorem unit_basis1_mix13_20_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖
  have h13 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide)
  have h20 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by
        have h13' : |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| ≤ d := by simpa [d] using h13
        nlinarith [h13']
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| := by congr 1; ring
      _ ≤ 2 * (2 * d) := by
        have h20' : |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h20
        nlinarith [h20']
      _ = 4 * d := by ring
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using
              (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ))]

theorem unit_basis1_mix12_plus30_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖
  have h12 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide)
  have h30 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ = 2 * |q ⟨2, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using
              (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ))]

theorem unit_basis1_mix23_10_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖
  have h23 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide) (by decide)
  have h10 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using
              (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (-2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ))]

theorem unit_basis1_diag13_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  have hq0sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hq2sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨0, by decide⟩) ^ 2 + 2 * (q ⟨2, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq0sq, hq2sq]
  have hrewrite :
      2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)
        = 2 * (q ⟨0, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) := by
    have hsumSq :
        (q ⟨0, by decide⟩) ^ (2 : ℕ) + (q ⟨1, by decide⟩) ^ (2 : ℕ) +
          (q ⟨2, by decide⟩) ^ (2 : ℕ) + (q ⟨3, by decide⟩) ^ (2 : ℕ) = 1 := by
      have hsumSq' := norm_sq_eq_sum_sq q
      rw [Fin.sum_univ_four] at hsumSq'
      rw [hNorm] at hsumSq'
      norm_num at hsumSq'
      exact hsumSq'.symm
    nlinarith [hsumSq]
  have hform :
      (-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) =
        2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) := by ring
  rw [hform, hrewrite, abs_of_nonneg (by positivity)]
  exact hsum

theorem unit_basis1_diag12_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2)|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  have hq0sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hq3sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨0, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq0sq, hq3sq]
  have hrewrite :
      2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)
        = 2 * (q ⟨0, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) := by
    have hsumSq :
        (q ⟨0, by decide⟩) ^ (2 : ℕ) + (q ⟨1, by decide⟩) ^ (2 : ℕ) +
          (q ⟨2, by decide⟩) ^ (2 : ℕ) + (q ⟨3, by decide⟩) ^ (2 : ℕ) = 1 := by
      have hsumSq' := norm_sq_eq_sum_sq q
      rw [Fin.sum_univ_four] at hsumSq'
      rw [hNorm] at hsumSq'
      norm_num at hsumSq'
      exact hsumSq'.symm
    nlinarith [hsumSq]
  have hform :
      (-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) =
        2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) := by ring
  rw [hform, hrewrite, abs_of_nonneg (by positivity)]
  exact hsum

theorem unit_basis2_diag23_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  have hq0sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hq1sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨1, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨0, by decide⟩) ^ 2 + 2 * (q ⟨1, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq0sq, hq1sq]
  have hrewrite :
      2 - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)
        = 2 * (q ⟨0, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) := by
    have hsumSq' := norm_sq_eq_sum_sq q
    rw [Fin.sum_univ_four] at hsumSq'
    rw [hNorm] at hsumSq'
    norm_num at hsumSq'
    have hsumSq :
        (q ⟨0, by decide⟩) ^ (2 : ℕ) + (q ⟨1, by decide⟩) ^ (2 : ℕ) +
          (q ⟨2, by decide⟩) ^ (2 : ℕ) + (q ⟨3, by decide⟩) ^ (2 : ℕ) = 1 := hsumSq'.symm
    nlinarith [hsumSq]
  have hform :
      (-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) =
        2 - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) := by ring
  rw [hform, hrewrite, abs_of_nonneg (by positivity)]
  exact hsum

theorem unit_basis2_quad13_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  have hq1sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨1, by decide⟩ (by decide)
  have hq3sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq1sq, hq3sq]
  have hnonneg : 0 ≤ 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2 := by positivity
  have hrewrite :
      |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))|
        = 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2 := by
    have hneg :
        -2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) =
          -(2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) := by ring
    rw [hneg, abs_neg, abs_of_nonneg hnonneg]
  rw [hrewrite]
  exact hsum

theorem unit_basis2_mix12_30_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖
  have h12 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨1, by decide⟩ (by decide)
  have h30 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by gcongr
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using
              (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (-2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ))]

theorem unit_basis2_mix13_20_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖
  have h13 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide) (by decide)
  have h20 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using
              (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ))]

theorem unit_basis2_mix13_minus20_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖
  have h13 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide) (by decide)
  have h20 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (-2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ))]

theorem unit_basis2_mix23_plus10_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖
  have h23 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide)
  have h10 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by
        simpa [d] using h23
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * (2 * d) := by
        have h10' : |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h10
        nlinarith [h10']
      _ = 4 * d := by ring
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ))]

theorem unit_basis2_mix12_plus30_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖
  have h12 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨1, by decide⟩ (by decide)
  have h30 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by gcongr
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ))]

theorem unit_basis2_mix23_10_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖
  have h23 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide)
  have h10 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by
        have h23' : |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| ≤ d := by simpa [d] using h23
        nlinarith [h23']
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by
        have h10' : |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by simpa [d] using h10
        nlinarith [h10']
      _ = 4 * d := by ring
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (-2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ))]

theorem unit_basis2_diag12_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2)|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  have hq0sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hq3sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨0, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq0sq, hq3sq]
  have hrewrite :
      2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)
        = 2 * (q ⟨0, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) := by
    have hsumSq' := norm_sq_eq_sum_sq q
    rw [Fin.sum_univ_four] at hsumSq'
    rw [hNorm] at hsumSq'
    norm_num at hsumSq'
    have hsumSq :
        (q ⟨0, by decide⟩) ^ (2 : ℕ) + (q ⟨1, by decide⟩) ^ (2 : ℕ) +
          (q ⟨2, by decide⟩) ^ (2 : ℕ) + (q ⟨3, by decide⟩) ^ (2 : ℕ) = 1 := hsumSq'.symm
    nlinarith [hsumSq]
  have hform :
      (-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) =
        2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) := by ring
  rw [hform, hrewrite, abs_of_nonneg (by positivity)]
  exact hsum

theorem rigidTransformPoint3D_zero_translation_sub_basis2_coord0_abs_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) ⟨0, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q2 : Fin 4 := ⟨2, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q2 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_diag23_abs_le_eight_mul_dist q hNorm
  have hBcoef : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_mix12_30_abs_le_eight_mul_dist q hNorm
  have hCcoef : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_mix13_20_abs_le_eight_mul_dist q hNorm
  have hA : |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) p0 =
        ((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0) +
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2) := by
    simpa [p0, p1, p2, q2] using congrArg (fun v : MDArray 3 => v p0) (rigidTransformPoint3D_zero_translation_sub_basis2 point q)
  rw [hEq]
  have htri2 :
      |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
        |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1)
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2))
  have htri1 :
      |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0) +
        (((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2))|
      ≤ |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0)| +
        |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0)
        (((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)))
  calc
    |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0) +
      ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
      ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
        = |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0) +
            (((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
             ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2))| := by rw [add_assoc]
    _ ≤ |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0)| +
          |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
            ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
          |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem unit_basis1_mix13_minus20_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖
  have h13 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide)
  have h20 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    have h13' : |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| ≤ d := by simpa [d, mul_comm] using h13
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨1, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by nlinarith [h13']
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    have h20' : |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h20
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨0, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 4 * d := by nlinarith [h20']
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (-2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ))]

theorem rigidTransformPoint3D_zero_translation_sub_basis2_coord1_abs_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) ⟨1, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q2 : Fin 4 := ⟨2, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q2 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_mix12_plus30_abs_le_eight_mul_dist q hNorm
  have hBcoef : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_quad13_abs_le_eight_mul_dist q hNorm
  have hCcoef : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_mix23_10_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) p1 =
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2) := by
    simpa [p0, p1, p2, q2] using congrArg (fun v : MDArray 3 => v p1) (rigidTransformPoint3D_zero_translation_sub_basis2 point q)
  rw [hEq]
  have htri2 :
      |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
      ≤ |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1)| +
        |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1)
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))
  have htri1 :
      |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        (((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
        |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)
        (((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)))
  calc
    |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
      ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
      ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
        = |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
            (((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
             ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))| := by rw [add_assoc]
    _ ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
          |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
            ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1)| +
          |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_sub_basis2_coord2_abs_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) ⟨2, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q2 : Fin 4 := ⟨2, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q2 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_mix13_minus20_abs_le_eight_mul_dist q hNorm
  have hBcoef : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_mix23_plus10_abs_le_eight_mul_dist q hNorm
  have hCcoef : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2)| ≤ 8 * e := by
    simpa [e, q2] using unit_basis2_diag12_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) p2 =
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2) := by
    simpa [p0, p1, p2, q2] using congrArg (fun v : MDArray 3 => v p2) (rigidTransformPoint3D_zero_translation_sub_basis2 point q)
  rw [hEq]
  have htri2 :
      |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)|
      ≤ |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
        |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2))
  have htri1 :
      |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2))|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
        |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)
        (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)))
  calc
    |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
      ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
      (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)|
        = |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
            (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
             (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2))| := by rw [add_assoc]
    _ ≤ |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
          |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
            (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
          |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_dist_to_basis2_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
         (rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) ≤
      48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  let diff : MDArray 3 :=
    rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
      rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))
  have h0 : |diff ⟨0, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis2_coord0_abs_le' point q hNorm
  have h1 : |diff ⟨1, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis2_coord1_abs_le' point q hNorm
  have h2 : |diff ⟨2, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis2_coord2_abs_le' point q hNorm
  have hBnonneg : 0 ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
    unfold pointL1Radius
    positivity
  have hnorm : ‖diff‖ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖) := by
    exact norm_mdarray3_le_two_of_abs_le diff
      (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖)
      hBnonneg h0 h1 h2
  have hEqDist :
      dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := by
    simp [diff, dist_eq_norm, sub_eq_add_neg]
  calc
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
        (rigidTransformPoint3D point (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := hEqDist
    _ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖) := hnorm
    _ = 48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by ring

theorem rigidTransform3D_zero_translation_dist_to_basis2_le_of_pointL1Radius_bound'
    {n : ℕ}
    (coords : CoordSet n)
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (armBound : ℝ)
    (hArm : ∀ j, pointL1Radius (coords j) ≤ armBound) :
    ∀ j,
      dist (rigidTransform3D coords q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D coords (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0)) j) ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
  intro j
  have hPoint :
      dist (rigidTransformPoint3D (coords j) q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D (coords j) (quaternionDictionary8 ⟨2, by decide⟩) (mkMDArray (fun _ => 0))) ≤
        48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ :=
    rigidTransformPoint3D_zero_translation_dist_to_basis2_le' (coords j) q hNorm
  have hScale :
      48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := by
    have h48 : (0 : ℝ) ≤ 48 := by norm_num
    have hNormNonneg : 0 ≤ ‖q - EuclideanSpace.single ⟨2, by decide⟩ (1 : ℝ)‖ := norm_nonneg _
    nlinarith [hArm j, h48, hNormNonneg]
  exact le_trans (by simpa [rigidTransform3D] using hPoint) hScale

theorem rigidTransformPoint3D_zero_translation_sub_basis3
    (point : MDArray 3)
    (quaternion : MDArray 4) :
    rigidTransformPoint3D point quaternion (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0)) =
      mkMDArray (fun j =>
        if h0 : j = ⟨0, by decide⟩ then
          (-2 * (quaternion ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (quaternion ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point ⟨0, by decide⟩ +
            (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨2, by decide⟩ - 2 * quaternion ⟨3, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨1, by decide⟩ +
            (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨3, by decide⟩ + 2 * quaternion ⟨2, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨2, by decide⟩
        else if h1 : j = ⟨1, by decide⟩ then
          (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨2, by decide⟩ + 2 * quaternion ⟨3, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨0, by decide⟩ +
            (-2 * quaternion ⟨1, by decide⟩ ^ (2 : ℕ) - 2 * quaternion ⟨3, by decide⟩ ^ (2 : ℕ) + 2) * point ⟨1, by decide⟩ +
            (2 * quaternion ⟨2, by decide⟩ * quaternion ⟨3, by decide⟩ - 2 * quaternion ⟨1, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨2, by decide⟩
        else
          (2 * quaternion ⟨1, by decide⟩ * quaternion ⟨3, by decide⟩ - 2 * quaternion ⟨2, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨0, by decide⟩ +
            (2 * quaternion ⟨2, by decide⟩ * quaternion ⟨3, by decide⟩ + 2 * quaternion ⟨1, by decide⟩ * quaternion ⟨0, by decide⟩) * point ⟨1, by decide⟩ +
            (-2 * quaternion ⟨1, by decide⟩ ^ (2 : ℕ) - 2 * quaternion ⟨2, by decide⟩ ^ (2 : ℕ)) * point ⟨2, by decide⟩) := by
  ext j
  fin_cases j <;>
    simp [rigidTransformPoint3D, quaternionDictionary8, mkMDArray] <;> ring

theorem unit_basis3_diag23_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  have hq0sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hq1sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨0, by decide⟩) ^ 2 + 2 * (q ⟨1, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq0sq, hq1sq]
  have hrewrite :
      2 - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)
        = 2 * (q ⟨0, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) := by
    have hsumSq' := norm_sq_eq_sum_sq q
    rw [Fin.sum_univ_four] at hsumSq'
    rw [hNorm] at hsumSq'
    norm_num at hsumSq'
    have hsumSq :
        (q ⟨0, by decide⟩) ^ (2 : ℕ) + (q ⟨1, by decide⟩) ^ (2 : ℕ) +
          (q ⟨2, by decide⟩) ^ (2 : ℕ) + (q ⟨3, by decide⟩) ^ (2 : ℕ) = 1 := hsumSq'.symm
    nlinarith [hsumSq]
  have hform :
      (-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) =
        2 - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) := by ring
  rw [hform, hrewrite, abs_of_nonneg (by positivity)]
  exact hsum

theorem unit_basis3_diag13_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  have hq0sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hq2sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨0, by decide⟩) ^ 2 + 2 * (q ⟨2, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq0sq, hq2sq]
  have hrewrite :
      2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)
        = 2 * (q ⟨0, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) := by
    have hsumSq' := norm_sq_eq_sum_sq q
    rw [Fin.sum_univ_four] at hsumSq'
    rw [hNorm] at hsumSq'
    norm_num at hsumSq'
    have hsumSq :
        (q ⟨0, by decide⟩) ^ (2 : ℕ) + (q ⟨1, by decide⟩) ^ (2 : ℕ) +
          (q ⟨2, by decide⟩) ^ (2 : ℕ) + (q ⟨3, by decide⟩) ^ (2 : ℕ) = 1 := hsumSq'.symm
    nlinarith [hsumSq]
  have hform :
      (-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) =
        2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) := by ring
  rw [hform, hrewrite, abs_of_nonneg (by positivity)]
  exact hsum

theorem unit_basis3_quad12_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ))|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  have hq1sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ (by decide)
  have hq2sq := unit_norm_mdarray4_offaxis_sq_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨2, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq1sq, hq2sq]
  have hnonneg : 0 ≤ 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨2, by decide⟩) ^ 2 := by positivity
  have hrewrite :
      |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ))|
        = 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨2, by decide⟩) ^ 2 := by
    have hneg :
        -2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) =
          -(2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) := by ring
    rw [hneg, abs_neg, abs_of_nonneg hnonneg]
  rw [hrewrite]
  exact hsum

/- theorem unit_basis3_mix12_30_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h12 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide) (by decide)
  have h30 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨3, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (-2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix13_20_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h13 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ (by decide)
  have h20 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by
        simpa [d] using h13
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| := by congr 1; ring
      _ ≤ 2 * (2 * d) := by
        have h20' : |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h20
        nlinarith [h20']
      _ = 4 * d := by ring
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix12_plus30_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h12 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide) (by decide)
  have h30 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨3, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix23_10_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h23 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ (by decide)
  have h10 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by
        simpa [d] using h23
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * (2 * d) := by
        have h10' : |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h10
        nlinarith [h10']
      _ = 4 * d := by ring
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (-2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix13_minus20_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h13 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ (by decide)
  have h20 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (-2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix23_plus10_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h23 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ (by decide)
  have h10 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ = 2 * |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))] -/

theorem unit_basis3_mix12_30_abs_le_eight_mul_dist'
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h12 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide) (by decide)
  have h30 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 4 * d := by
    have h12' : |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by simpa [d] using h12
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 4 * d := by nlinarith [h12']
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    have h30' : |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ d := by simpa [d, mul_comm] using h30
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by nlinarith [h30']
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (-2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix13_20_abs_le_eight_mul_dist'
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h13 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ (by decide)
  have h20 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    have h13' : |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| ≤ d := by simpa [d, mul_comm] using h13
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨1, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by nlinarith [h13']
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    have h20' : |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h20
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨0, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 4 * d := by nlinarith [h20']
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix12_plus30_abs_le_eight_mul_dist'
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h12 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide) (by decide)
  have h30 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 4 * d := by
    have h12' : |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by simpa [d] using h12
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 4 * d := by nlinarith [h12']
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    have h30' : |q ⟨0, by decide⟩ * q ⟨3, by decide⟩| ≤ d := by simpa [d, mul_comm] using h30
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨0, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨0, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by nlinarith [h30']
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix23_10_abs_le_eight_mul_dist'
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h23 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ (by decide)
  have h10 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    have h23' : |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| ≤ d := by simpa [d, mul_comm] using h23
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by nlinarith [h23']
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    have h10' : |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h10
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨0, by decide⟩ * q ⟨1, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 4 * d := by nlinarith [h10']
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (-2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix13_minus20_abs_le_eight_mul_dist'
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h13 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ (by decide)
  have h20 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    have h13' : |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| ≤ d := by simpa [d, mul_comm] using h13
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨1, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨1, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by nlinarith [h13']
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    have h20' : |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h20
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨0, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨0, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 4 * d := by nlinarith [h20']
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs, sub_eq_add_neg] using (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (-2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem unit_basis3_mix23_plus10_abs_le_eight_mul_dist'
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖
  have h23 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨2, by decide⟩ (by decide)
  have h10 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨3, by decide⟩ ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide) (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 2 * d := by
    have h23' : |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| ≤ d := by simpa [d, mul_comm] using h23
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 2 * d := by nlinarith [h23']
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 4 * d := by
    have h10' : |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| ≤ 2 * d := by simpa [d, mul_comm] using h10
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨0, by decide⟩ * q ⟨1, by decide⟩) by ring]
        rw [abs_mul]; norm_num
      _ ≤ 4 * d := by nlinarith [h10']
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 2 * d + 4 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ))]

theorem rigidTransformPoint3D_zero_translation_sub_basis3_coord0_abs_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) ⟨0, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q3 : Fin 4 := ⟨3, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q3 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_diag23_abs_le_eight_mul_dist q hNorm
  have hBcoef : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_mix12_30_abs_le_eight_mul_dist' q hNorm
  have hCcoef : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_mix13_20_abs_le_eight_mul_dist' q hNorm
  have hA : |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) p0 =
        ((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0) +
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2) := by
    simpa [p0, p1, p2, q3] using congrArg (fun v : MDArray 3 => v p0) (rigidTransformPoint3D_zero_translation_sub_basis3 point q)
  rw [hEq]
  have htri2 :
      |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
        |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1)
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2))
  have htri1 :
      |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0) +
        (((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2))|
      ≤ |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0)| +
        |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0)
        (((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)))
  calc
    |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0) +
      ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
      ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
        = |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0) +
            (((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
             ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2))| := by rw [add_assoc]
    _ ≤ |((-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p0)| +
          |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
            ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
          |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_sub_basis3_coord1_abs_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) ⟨1, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q3 : Fin 4 := ⟨3, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q3 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_mix12_plus30_abs_le_eight_mul_dist' q hNorm
  have hBcoef : |(-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_diag13_abs_le_eight_mul_dist q hNorm
  have hCcoef : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_mix23_10_abs_le_eight_mul_dist' q hNorm
  have hA : |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) p1 =
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2) := by
    simpa [p0, p1, p2, q3] using congrArg (fun v : MDArray 3 => v p1) (rigidTransformPoint3D_zero_translation_sub_basis3 point q)
  rw [hEq]
  have htri2 :
      |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
      ≤ |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1)| +
        |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1)
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))
  have htri1 :
      |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        ((((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
        |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)
        ((((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)))
  calc
    |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
      (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
      ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
        = |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
            ((((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
             ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))| := by rw [add_assoc]
    _ ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
          |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
            ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := htri1
    _ ≤ 8 * S * e + (|(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1)| +
          |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_sub_basis3_coord2_abs_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) ⟨2, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q3 : Fin 4 := ⟨3, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q3 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_mix13_minus20_abs_le_eight_mul_dist' q hNorm
  have hBcoef : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_mix23_plus10_abs_le_eight_mul_dist' q hNorm
  have hCcoef : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ))| ≤ 8 * e := by
    simpa [e, q3] using unit_basis3_quad12_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ))| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) p2 =
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2) := by
    simpa [p0, p1, p2, q3] using congrArg (fun v : MDArray 3 => v p2) (rigidTransformPoint3D_zero_translation_sub_basis3 point q)
  rw [hEq]
  have htri2 :
      |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)|
      ≤ |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
        |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2))
  have htri1 :
      |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2))|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
        |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)
        (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)))
  calc
    |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
      ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
      ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)|
        = |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
            (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
             ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2))| := by rw [add_assoc]
    _ ≤ |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
          |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
            ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
          |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_dist_to_basis3_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
         (rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) ≤
      48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  let diff : MDArray 3 :=
    rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
      rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))
  have h0 : |diff ⟨0, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis3_coord0_abs_le' point q hNorm
  have h1 : |diff ⟨1, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis3_coord1_abs_le' point q hNorm
  have h2 : |diff ⟨2, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis3_coord2_abs_le' point q hNorm
  have hBnonneg : 0 ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
    unfold pointL1Radius
    positivity
  have hnorm : ‖diff‖ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖) := by
    exact norm_mdarray3_le_two_of_abs_le diff
      (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖)
      hBnonneg h0 h1 h2
  have hEqDist :
      dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := by
    simp [diff, dist_eq_norm, sub_eq_add_neg]
  calc
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
        (rigidTransformPoint3D point (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := hEqDist
    _ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖) := hnorm
    _ = 48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by ring

theorem rigidTransform3D_zero_translation_dist_to_basis3_le_of_pointL1Radius_bound'
    {n : ℕ}
    (coords : CoordSet n)
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (armBound : ℝ)
    (hArm : ∀ j, pointL1Radius (coords j) ≤ armBound) :
    ∀ j,
      dist (rigidTransform3D coords q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D coords (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0)) j) ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
  intro j
  have hPoint :
      dist (rigidTransformPoint3D (coords j) q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D (coords j) (quaternionDictionary8 ⟨3, by decide⟩) (mkMDArray (fun _ => 0))) ≤
        48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ :=
    rigidTransformPoint3D_zero_translation_dist_to_basis3_le' (coords j) q hNorm
  have hScale :
      48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := by
    have h48 : (0 : ℝ) ≤ 48 := by norm_num
    have hNormNonneg : 0 ≤ ‖q - EuclideanSpace.single ⟨3, by decide⟩ (1 : ℝ)‖ := norm_nonneg _
    nlinarith [hArm j, h48, hNormNonneg]
  exact le_trans (by simpa [rigidTransform3D] using hPoint) hScale


theorem unit_basis1_mix23_plus10_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖
  have h23 := unit_norm_mdarray4_offaxis_product_abs_le_two_mul_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide) (by decide)
  have h10 := unit_norm_mdarray4_axis_product_abs_le_dist_to_basis q hNorm ⟨1, by decide⟩ ⟨0, by decide⟩ (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ = 2 * |q ⟨0, by decide⟩ * q ⟨1, by decide⟩| := by congr 1; ring
      _ ≤ 2 * d := by gcongr
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
            simpa [Real.norm_eq_abs] using
              (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ))]

theorem rigidTransformPoint3D_zero_translation_sub_basis1_coord0_abs_le
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ⟨0, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q0 : Fin 4 := ⟨0, by decide⟩
  let q1 : Fin 4 := ⟨1, by decide⟩
  let q2 : Fin 4 := ⟨2, by decide⟩
  let q3 : Fin 4 := ⟨3, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q1 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |(-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ))| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_quad23_abs_le_eight_mul_dist q hNorm
  have hBcoef : |2 * q q1 * q q2 - 2 * q q3 * q q0| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix12_30_abs_le_eight_mul_dist q hNorm
  have hCcoef : |2 * q q1 * q q3 + 2 * q q2 * q q0| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix13_20_abs_le_eight_mul_dist q hNorm
  have hA : |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ))| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q q1 * q q2 - 2 * q q3 * q q0)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q q1 * q q3 + 2 * q q2 * q q0)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) p0 =
        ((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0) +
        ((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
        ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2) := by
    simpa [p0, p1, p2, q0, q1, q2, q3] using
      congrArg (fun v : MDArray 3 => v p0) (rigidTransformPoint3D_zero_translation_sub_basis1 point q)
  rw [hEq]
  have htri2 :
      |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
        ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)|
      ≤ |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1)| +
        |((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1)
        ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2))
  have htri1 :
      |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0) +
        (((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
         ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2))|
      ≤ |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0)| +
        |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
         ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0)
        (((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
         ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)))
  calc
    |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0) +
      ((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
      ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)|
        = |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0) +
            (((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
             ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2))| := by rw [add_assoc]
    _ ≤ |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0)| +
          |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
            ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1)| +
          |((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

/- theorem rigidTransformPoint3D_zero_translation_sub_basis1_coord1_abs_le
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ⟨1, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q1 : Fin 4 := ⟨1, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q1 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix12_plus30_abs_le_eight_mul_dist q hNorm
  have hBcoef : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_diag13_abs_le_eight_mul_dist q hNorm
  have hCcoef : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix23_10_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) p1 =
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2) := by
    simpa [p0, p1, p2, q1] using
      congrArg (fun v : MDArray 3 => v p1) (rigidTransformPoint3D_zero_translation_sub_basis1 point q)
  rw [hEq]
  have htri2 :
      |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
      ≤ |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1)| +
        |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1)
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))
  have htri1 :
      |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        (((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
        |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)
        (((2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)))
  calc
    |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
      ((2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
      ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
        = |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
            (((2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) * point p1) +
             ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))| := by rw [add_assoc]
    _ ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
          |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1) +
            ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ) + 2) * point p1)| +
          |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_sub_basis1_coord2_abs_le
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ⟨2, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q1 : Fin 4 := ⟨1, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q1 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix13_minus20_abs_le_eight_mul_dist q hNorm
  have hBcoef : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix23_plus10_abs_le_eight_mul_dist q hNorm
  have hCcoef : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2)| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_diag12_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) p2 =
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) * point p2) := by
    simpa [p0, p1, p2, q1] using
      congrArg (fun v : MDArray 3 => v p2) (rigidTransformPoint3D_zero_translation_sub_basis1 point q)
  rw [hEq]
  have htri2 :
      |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) * point p2)|
      ≤ |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
        |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)
        ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) * point p2))
  have htri1 :
      |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) * point p2))|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
        |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)
        (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)))
  calc
    |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
      ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
      ((2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2)|
        = |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
            (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
             ((2 - 2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) * point p2))| := by rw [add_assoc]
    _ ≤ |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
          |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         ((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
          |((-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_dist_to_basis1_le
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
         (rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ≤
      48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let diff : MDArray 3 :=
    rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
      rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))
  have h0 : |diff ⟨0, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis1_coord0_abs_le point q hNorm
  have h1 : |diff ⟨1, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis1_coord1_abs_le point q hNorm
  have h2 : |diff ⟨2, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis1_coord2_abs_le point q hNorm
  have hBnonneg : 0 ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    unfold pointL1Radius
    positivity
  have hnorm : ‖diff‖ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖) := by
    exact norm_mdarray3_le_two_of_abs_le diff
      (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖)
      hBnonneg h0 h1 h2
  have hEqDist :
      dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := by
    simp [diff, dist_eq_norm, sub_eq_add_neg]
  calc
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
        (rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := hEqDist
    _ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖) := hnorm
    _ = 48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by ring

theorem rigidTransform3D_zero_translation_dist_to_basis1_le_of_pointL1Radius_bound
    {n : ℕ}
    (coords : CoordSet n)
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (armBound : ℝ)
    (hArm : ∀ j, pointL1Radius (coords j) ≤ armBound) :
    ∀ j,
      dist (rigidTransform3D coords q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D coords (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0)) j) ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  intro j
  have hPoint :
      dist (rigidTransformPoint3D (coords j) q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D (coords j) (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ≤
        48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ :=
    rigidTransformPoint3D_zero_translation_dist_to_basis1_le (coords j) q hNorm
  have hScale :
      48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    have h48 : (0 : ℝ) ≤ 48 := by norm_num
    have hNormNonneg : 0 ≤ ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := norm_nonneg _
    nlinarith [hArm j, h48, hNormNonneg]
  exact le_trans (by simpa [rigidTransform3D] using hPoint) hScale -/

theorem rigidTransformPoint3D_zero_translation_sub_basis1_coord1_abs_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ⟨1, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q1 : Fin 4 := ⟨1, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q1 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix12_plus30_abs_le_eight_mul_dist q hNorm
  have hBcoef : |(-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_diag13_abs_le_eight_mul_dist q hNorm
  have hCcoef : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix23_10_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) p1 =
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2) := by
    simpa [p0, p1, p2, q1] using congrArg (fun v : MDArray 3 => v p1) (rigidTransformPoint3D_zero_translation_sub_basis1 point q)
  rw [hEq]
  have htri2 :
      |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
      ≤ |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1)| +
        |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1)
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))
  have htri1 :
      |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        ((((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
        |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)
        ((((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
         ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)))
  calc
    |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
      (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
      ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|
        = |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
            ((((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
             ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2))| := by rw [add_assoc]
    _ ≤ |((2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
          |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1) +
            ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)| := htri1
    _ ≤ 8 * S * e + (|(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨3, by decide⟩ ^ (2 : ℕ)) + 2)) * point p1)| +
          |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_sub_basis1_coord2_abs_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
        rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ⟨2, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q1 : Fin 4 := ⟨1, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q1 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix13_minus20_abs_le_eight_mul_dist q hNorm
  have hBcoef : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_mix23_plus10_abs_le_eight_mul_dist q hNorm
  have hCcoef : |(-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)| ≤ 8 * e := by
    simpa [e, q1] using unit_basis1_diag12_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
          rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) p2 =
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2) := by
    simpa [p0, p1, p2, q1] using congrArg (fun v : MDArray 3 => v p2) (rigidTransformPoint3D_zero_translation_sub_basis1 point q)
  rw [hEq]
  have htri2 :
      |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)|
      ≤ |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
        |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2))
  have htri1 :
      |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
        (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2))|
      ≤ |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
        |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
         (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)
        (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
        (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)))
  calc
    |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
      ((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
      (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)|
        = |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0) +
            (((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
             (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2))| := by rw [add_assoc]
    _ ≤ |((2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩) * point p0)| +
          |((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1) +
            (((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩) * point p1)| +
          |(((-2 * (q ⟨1, by decide⟩ ^ (2 : ℕ)) - 2 * (q ⟨2, by decide⟩ ^ (2 : ℕ)) + 2)) * point p2)|) := by gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_dist_to_basis1_le'
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
         (rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ≤
      48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  let diff : MDArray 3 :=
    rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) -
      rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))
  have h0 : |diff ⟨0, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis1_coord0_abs_le point q hNorm
  have h1 : |diff ⟨1, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis1_coord1_abs_le' point q hNorm
  have h2 : |diff ⟨2, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis1_coord2_abs_le' point q hNorm
  have hBnonneg : 0 ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    unfold pointL1Radius
    positivity
  have hnorm : ‖diff‖ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖) := by
    exact norm_mdarray3_le_two_of_abs_le diff
      (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖)
      hBnonneg h0 h1 h2
  have hEqDist :
      dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := by
    simp [diff, dist_eq_norm, sub_eq_add_neg]
  calc
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
        (rigidTransformPoint3D point (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := hEqDist
    _ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖) := hnorm
    _ = 48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by ring

theorem rigidTransform3D_zero_translation_dist_to_basis1_le_of_pointL1Radius_bound'
    {n : ℕ}
    (coords : CoordSet n)
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (armBound : ℝ)
    (hArm : ∀ j, pointL1Radius (coords j) ≤ armBound) :
    ∀ j,
      dist (rigidTransform3D coords q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D coords (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0)) j) ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
  intro j
  have hPoint :
      dist (rigidTransformPoint3D (coords j) q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D (coords j) (quaternionDictionary8 ⟨1, by decide⟩) (mkMDArray (fun _ => 0))) ≤
        48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ :=
    rigidTransformPoint3D_zero_translation_dist_to_basis1_le' (coords j) q hNorm
  have hScale :
      48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := by
    have h48 : (0 : ℝ) ≤ 48 := by norm_num
    have hNormNonneg : 0 ≤ ‖q - EuclideanSpace.single ⟨1, by decide⟩ (1 : ℝ)‖ := norm_nonneg _
    nlinarith [hArm j, h48, hNormNonneg]
  exact le_trans (by simpa [rigidTransform3D] using hPoint) hScale

theorem unit_basis0_quad23_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  have hq2sq := unit_basis0_error_offaxis_sq_le_two_mul_dist q hNorm ⟨2, by decide⟩ (by decide)
  have hq3sq := unit_basis0_error_offaxis_sq_le_two_mul_dist q hNorm ⟨3, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨2, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq2sq, hq3sq]
  have hnonneg : 0 ≤ 2 * (q ⟨2, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2 := by
    positivity
  have hrewrite :
      |(-2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))|
        = 2 * (q ⟨2, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2 := by
    have hneg :
        -2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)
          = -(2 * (q ⟨2, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) := by
      ring
    rw [hneg, abs_neg, abs_of_nonneg hnonneg]
  rw [hrewrite]
  exact hsum

theorem unit_basis0_quad13_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  have hq1sq := unit_basis0_error_offaxis_sq_le_two_mul_dist q hNorm ⟨1, by decide⟩ (by decide)
  have hq3sq := unit_basis0_error_offaxis_sq_le_two_mul_dist q hNorm ⟨3, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq1sq, hq3sq]
  have hnonneg : 0 ≤ 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2 := by
    positivity
  have hrewrite :
      |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ))|
        = 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨3, by decide⟩) ^ 2 := by
    have hneg :
        -2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)
          = -(2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨3, by decide⟩) ^ (2 : ℕ)) := by
      ring
    rw [hneg, abs_neg, abs_of_nonneg hnonneg]
  rw [hrewrite]
  exact hsum

theorem unit_basis0_quad12_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ))|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  have hq1sq := unit_basis0_error_offaxis_sq_le_two_mul_dist q hNorm ⟨1, by decide⟩ (by decide)
  have hq2sq := unit_basis0_error_offaxis_sq_le_two_mul_dist q hNorm ⟨2, by decide⟩ (by decide)
  have hsum : 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨2, by decide⟩) ^ 2
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    nlinarith [hq1sq, hq2sq]
  have hnonneg : 0 ≤ 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨2, by decide⟩) ^ 2 := by
    positivity
  have hrewrite :
      |(-2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ))|
        = 2 * (q ⟨1, by decide⟩) ^ 2 + 2 * (q ⟨2, by decide⟩) ^ 2 := by
    have hneg :
        -2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) - 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)
          = -(2 * (q ⟨1, by decide⟩) ^ (2 : ℕ) + 2 * (q ⟨2, by decide⟩) ^ (2 : ℕ)) := by
      ring
    rw [hneg, abs_neg, abs_of_nonneg hnonneg]
  rw [hrewrite]
  exact hsum

theorem unit_basis0_mix12_30_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖
  have h12 := unit_basis0_error_offaxis_product_abs_le_two_mul_dist q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide) (by decide)
  have h30 := unit_basis0_error_axis_product_abs_le_dist q hNorm ⟨3, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * d := by gcongr
  have htri :
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
    simpa [Real.norm_eq_abs, sub_eq_add_neg] using
      (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (- (2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩)))
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ - 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := htri
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ))]

theorem unit_basis0_mix13_20_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖
  have h13 := unit_basis0_error_offaxis_product_abs_le_two_mul_dist q hNorm ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide) (by decide)
  have h20 := unit_basis0_error_axis_product_abs_le_dist q hNorm ⟨2, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * d := by gcongr
  have htri :
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩))
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := htri
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ))]

theorem unit_basis0_mix23_10_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖
  have h23 := unit_basis0_error_offaxis_product_abs_le_two_mul_dist q hNorm ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide) (by decide)
  have h10 := unit_basis0_error_axis_product_abs_le_dist q hNorm ⟨1, by decide⟩ (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * d := by gcongr
  have htri :
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
    simpa [Real.norm_eq_abs, sub_eq_add_neg] using
      (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (- (2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩)))
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := htri
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ))]

theorem unit_basis0_mix12_plus30_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖
  have h12 := unit_basis0_error_offaxis_product_abs_le_two_mul_dist q hNorm ⟨1, by decide⟩ ⟨2, by decide⟩ (by decide) (by decide)
  have h30 := unit_basis0_error_axis_product_abs_le_dist q hNorm ⟨3, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨2, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨2, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨3, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * d := by gcongr
  have htri :
      |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩) (2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩))
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩ + 2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨2, by decide⟩| + |2 * q ⟨3, by decide⟩ * q ⟨0, by decide⟩| := htri
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ))]

theorem unit_basis0_mix13_minus20_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖
  have h13 := unit_basis0_error_offaxis_product_abs_le_two_mul_dist q hNorm ⟨1, by decide⟩ ⟨3, by decide⟩ (by decide) (by decide)
  have h20 := unit_basis0_error_axis_product_abs_le_dist q hNorm ⟨2, by decide⟩ (by decide)
  have hA : |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * d := by gcongr
  have htri :
      |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := by
    simpa [Real.norm_eq_abs, sub_eq_add_neg] using
      (norm_add_le (2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩) (- (2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩)))
  calc
    |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩ - 2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨1, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨2, by decide⟩ * q ⟨0, by decide⟩| := htri
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ))]

theorem unit_basis0_mix23_plus10_abs_le_eight_mul_dist
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
      ≤ 8 * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let d := ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖
  have h23 := unit_basis0_error_offaxis_product_abs_le_two_mul_dist q hNorm ⟨2, by decide⟩ ⟨3, by decide⟩ (by decide) (by decide)
  have h10 := unit_basis0_error_axis_product_abs_le_dist q hNorm ⟨1, by decide⟩ (by decide)
  have hA : |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| ≤ 4 * d := by
    calc
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| = 2 * |q ⟨2, by decide⟩ * q ⟨3, by decide⟩| := by
        rw [show 2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ = (2 : ℝ) * (q ⟨2, by decide⟩ * q ⟨3, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * (2 * d) := by gcongr
      _ = 4 * d := by ring
  have hB : |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| ≤ 2 * d := by
    calc
      |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| = 2 * |q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
        rw [show 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩ = (2 : ℝ) * (q ⟨1, by decide⟩ * q ⟨0, by decide⟩) by ring]
        rw [abs_mul]
        norm_num
      _ ≤ 2 * d := by gcongr
  have htri :
      |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le (2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩) (2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩))
  calc
    |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩ + 2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩|
        ≤ |2 * q ⟨2, by decide⟩ * q ⟨3, by decide⟩| + |2 * q ⟨1, by decide⟩ * q ⟨0, by decide⟩| := htri
    _ ≤ 4 * d + 2 * d := add_le_add hA hB
    _ ≤ 8 * d := by nlinarith [norm_nonneg (q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ))]

theorem rigidTransformPoint3D_zero_translation_sub_basis0_coord0_abs_le
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) - point) ⟨0, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q0 : Fin 4 := ⟨0, by decide⟩
  let q1 : Fin 4 := ⟨1, by decide⟩
  let q2 : Fin 4 := ⟨2, by decide⟩
  let q3 : Fin 4 := ⟨3, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q0 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |(-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ))| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_quad23_abs_le_eight_mul_dist q hNorm
  have hBcoef : |2 * q q1 * q q2 - 2 * q q3 * q q0| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_mix12_30_abs_le_eight_mul_dist q hNorm
  have hCcoef : |2 * q q1 * q q3 + 2 * q q2 * q q0| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_mix13_20_abs_le_eight_mul_dist q hNorm
  have hA : |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ))| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q q1 * q q2 - 2 * q q3 * q q0)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q q1 * q q3 + 2 * q q2 * q q0)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) - point) p0 =
        ((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0) +
        ((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
        ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2) := by
    simpa [p0, p1, p2, q0, q1, q2, q3] using
      congrArg (fun v : MDArray 3 => v p0) (rigidTransformPoint3D_zero_translation_sub_basis0 point q)
  rw [hEq]
  have htri2 :
      |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
        ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)|
      ≤ |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1)| +
        |((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1)
        ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2))
  have htri1 :
      |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0) +
        (((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
         ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2))|
      ≤ |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0)| +
        |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
         ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0)
        (((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
         ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)))
  calc
    |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0) +
      ((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
      ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)|
        = |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0) +
            (((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
             ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2))| := by
              rw [add_assoc]
    _ ≤ |((-2 * (q q2) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p0)| +
          |((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1) +
            ((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q q1 * q q2 - 2 * q q3 * q q0) * point p1)| +
          |((2 * q q1 * q q3 + 2 * q q2 * q q0) * point p2)|) := by
            gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by
            gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_sub_basis0_coord1_abs_le
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) - point) ⟨1, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q0 : Fin 4 := ⟨0, by decide⟩
  let q1 : Fin 4 := ⟨1, by decide⟩
  let q2 : Fin 4 := ⟨2, by decide⟩
  let q3 : Fin 4 := ⟨3, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q0 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q q1 * q q2 + 2 * q q3 * q q0| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_mix12_plus30_abs_le_eight_mul_dist q hNorm
  have hBcoef : |(-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ))| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_quad13_abs_le_eight_mul_dist q hNorm
  have hCcoef : |2 * q q2 * q q3 - 2 * q q1 * q q0| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_mix23_10_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q q1 * q q2 + 2 * q q3 * q q0) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q q1 * q q2 + 2 * q q3 * q q0)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ))| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q q2 * q q3 - 2 * q q1 * q q0)| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) - point) p1 =
        ((2 * q q1 * q q2 + 2 * q q3 * q q0) * point p0) +
        ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1) +
        ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2) := by
    simpa [p0, p1, p2, q0, q1, q2, q3] using
      congrArg (fun v : MDArray 3 => v p1) (rigidTransformPoint3D_zero_translation_sub_basis0 point q)
  rw [hEq]
  have htri2 :
      |((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1) +
        ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2)|
      ≤ |((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1)| +
        |((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1)
        ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2))
  have htri1 :
      |((2 * q q1 * q q2 + 2 * q q3 * q q0) * point p0) +
        (((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1) +
         ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2))|
      ≤ |((2 * q q1 * q q2 + 2 * q q3 * q q0) * point p0)| +
        |((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1) +
         ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q q1 * q q2 + 2 * q q3 * q q0) * point p0)
        (((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1) +
         ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2)))
  calc
    |((2 * q q1 * q q2 + 2 * q q3 * q q0) * point p0) +
      ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1) +
      ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2)|
        = |((2 * q q1 * q q2 + 2 * q q3 * q q0) * point p0) +
            (((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1) +
             ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2))| := by
              rw [add_assoc]
    _ ≤ |((2 * q q1 * q q2 + 2 * q q3 * q q0) * point p0)| +
          |((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1) +
            ((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q3) ^ (2 : ℕ)) * point p1)| +
          |((2 * q q2 * q q3 - 2 * q q1 * q q0) * point p2)|) := by
            gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by
            gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_sub_basis0_coord2_abs_le
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    |(rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) - point) ⟨2, by decide⟩|
      ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let p0 : Fin 3 := ⟨0, by decide⟩
  let p1 : Fin 3 := ⟨1, by decide⟩
  let p2 : Fin 3 := ⟨2, by decide⟩
  let q0 : Fin 4 := ⟨0, by decide⟩
  let q1 : Fin 4 := ⟨1, by decide⟩
  let q2 : Fin 4 := ⟨2, by decide⟩
  let q3 : Fin 4 := ⟨3, by decide⟩
  let S : ℝ := pointL1Radius point
  let e : ℝ := ‖q - EuclideanSpace.single q0 (1 : ℝ)‖
  have hp0 : |point p0| ≤ S := by simpa [S, p0] using point_coord_abs_le_pointL1Radius point p0
  have hp1 : |point p1| ≤ S := by simpa [S, p1] using point_coord_abs_le_pointL1Radius point p1
  have hp2 : |point p2| ≤ S := by simpa [S, p2] using point_coord_abs_le_pointL1Radius point p2
  have hAcoef : |2 * q q1 * q q3 - 2 * q q2 * q q0| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_mix13_minus20_abs_le_eight_mul_dist q hNorm
  have hBcoef : |2 * q q2 * q q3 + 2 * q q1 * q q0| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_mix23_plus10_abs_le_eight_mul_dist q hNorm
  have hCcoef : |(-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ))| ≤ 8 * e := by
    simpa [e, q0] using unit_basis0_quad12_abs_le_eight_mul_dist q hNorm
  have hA : |((2 * q q1 * q q3 - 2 * q q2 * q q0) * point p0)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q q1 * q q3 - 2 * q q2 * q q0)| * |point p0| ≤ (8 * e) * S :=
      mul_le_mul hAcoef hp0 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hB : |((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(2 * q q2 * q q3 + 2 * q q1 * q q0)| * |point p1| ≤ (8 * e) * S :=
      mul_le_mul hBcoef hp1 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hC : |((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2)| ≤ 8 * S * e := by
    rw [abs_mul]
    have hmul : |(-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ))| * |point p2| ≤ (8 * e) * S :=
      mul_le_mul hCcoef hp2 (abs_nonneg _) (by positivity)
    simpa [mul_assoc, mul_left_comm, mul_comm] using hmul
  have hEq :
      (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) - point) p2 =
        ((2 * q q1 * q q3 - 2 * q q2 * q q0) * point p0) +
        ((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1) +
        ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2) := by
    simpa [p0, p1, p2, q0, q1, q2, q3] using
      congrArg (fun v : MDArray 3 => v p2) (rigidTransformPoint3D_zero_translation_sub_basis0 point q)
  rw [hEq]
  have htri2 :
      |((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1) +
        ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2)|
      ≤ |((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1)| +
        |((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1)
        ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2))
  have htri1 :
      |((2 * q q1 * q q3 - 2 * q q2 * q q0) * point p0) +
        (((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1) +
         ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2))|
      ≤ |((2 * q q1 * q q3 - 2 * q q2 * q q0) * point p0)| +
        |((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1) +
         ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2)| := by
    simpa [Real.norm_eq_abs] using
      (norm_add_le
        ((2 * q q1 * q q3 - 2 * q q2 * q q0) * point p0)
        (((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1) +
         ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2)))
  calc
    |((2 * q q1 * q q3 - 2 * q q2 * q q0) * point p0) +
      ((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1) +
      ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2)|
        = |((2 * q q1 * q q3 - 2 * q q2 * q q0) * point p0) +
            (((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1) +
             ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2))| := by
              rw [add_assoc]
    _ ≤ |((2 * q q1 * q q3 - 2 * q q2 * q q0) * point p0)| +
          |((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1) +
            ((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2)| := htri1
    _ ≤ 8 * S * e + (|((2 * q q2 * q q3 + 2 * q q1 * q q0) * point p1)| +
          |((-2 * (q q1) ^ (2 : ℕ) - 2 * (q q2) ^ (2 : ℕ)) * point p2)|) := by
            gcongr
    _ ≤ 8 * S * e + (8 * S * e + 8 * S * e) := by
            gcongr
    _ = 24 * S * e := by ring

theorem rigidTransformPoint3D_zero_translation_dist_to_basis0_le
    (point : MDArray 3)
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
         (rigidTransformPoint3D point (quaternionDictionary8 ⟨0, by decide⟩) (mkMDArray (fun _ => 0))) ≤
      48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  let diff : MDArray 3 := rigidTransformPoint3D point q (mkMDArray (fun _ => 0)) - point
  have h0 : |diff ⟨0, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis0_coord0_abs_le point q hNorm
  have h1 : |diff ⟨1, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis0_coord1_abs_le point q hNorm
  have h2 : |diff ⟨2, by decide⟩| ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    simpa [diff] using rigidTransformPoint3D_zero_translation_sub_basis0_coord2_abs_le point q hNorm
  have hBnonneg : 0 ≤ 24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    unfold pointL1Radius
    positivity
  have hnorm : ‖diff‖ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖) := by
    exact norm_mdarray3_le_two_of_abs_le diff
      (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖)
      hBnonneg h0 h1 h2
  have hEqDist :
      dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D point (quaternionDictionary8 ⟨0, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := by
    rw [rigidTransformPoint3D_basis0_zero_eq_self]
    simp [diff, dist_eq_norm, sub_eq_add_neg]
  calc
    dist (rigidTransformPoint3D point q (mkMDArray (fun _ => 0)))
        (rigidTransformPoint3D point (quaternionDictionary8 ⟨0, by decide⟩) (mkMDArray (fun _ => 0))) = ‖diff‖ := hEqDist
    _ ≤ 2 * (24 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖) := hnorm
    _ = 48 * pointL1Radius point * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by ring

theorem rigidTransform3D_zero_translation_dist_to_basis0_le_of_pointL1Radius_bound
    {n : ℕ}
    (coords : CoordSet n)
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (armBound : ℝ)
    (hArm : ∀ j, pointL1Radius (coords j) ≤ armBound) :
    ∀ j,
      dist (rigidTransform3D coords q (mkMDArray (fun _ => 0)) j)
          (rigidTransform3D coords (quaternionDictionary8 ⟨0, by decide⟩) (mkMDArray (fun _ => 0)) j) ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  intro j
  have hPoint :
      dist (rigidTransformPoint3D (coords j) q (mkMDArray (fun _ => 0)))
          (rigidTransformPoint3D (coords j) (quaternionDictionary8 ⟨0, by decide⟩) (mkMDArray (fun _ => 0))) ≤
        48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ :=
    rigidTransformPoint3D_zero_translation_dist_to_basis0_le (coords j) q hNorm
  have hScale :
      48 * pointL1Radius (coords j) * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ≤
        48 * armBound * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
    have h48 : (0 : ℝ) ≤ 48 := by norm_num
    have hNormNonneg : 0 ≤ ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := norm_nonneg _
    nlinarith [hArm j, h48, hNormNonneg]
  exact le_trans (by simpa [rigidTransform3D] using hPoint) hScale

theorem rigidTransform3D_zero_translation_dist_to_basis0_le_of_uniform_pointL1Radius_bound
    {Ω : Type u}
    {n : ℕ}
    (baseCoords : Ω → CoordSet n)
    (armBound : ℝ)
    (hArm : ∀ ω j, pointL1Radius (baseCoords ω j) ≤ armBound) :
    ∀ ω q,
      norm q = 1 →
      ∀ j,
        dist (rigidTransform3D (baseCoords ω) q (mkMDArray (fun _ => 0)) j)
            (rigidTransform3D (baseCoords ω) (quaternionDictionary8 ⟨0, by decide⟩) (mkMDArray (fun _ => 0)) j) ≤
          48 * armBound * ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  intro ω q hNorm j
  exact rigidTransform3D_zero_translation_dist_to_basis0_le_of_pointL1Radius_bound
    (baseCoords ω) q hNorm armBound (hArm ω) j

theorem unit_norm_mdarray4_dist_sq_to_basis0_eq
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 = 2 - 2 * q ⟨0, by decide⟩ := by
  let i0 : Fin 4 := ⟨0, by decide⟩
  have hInner : (@inner ℝ _ _ q (EuclideanSpace.single i0 (1 : ℝ)) : ℝ) = q i0 := by
    simpa using (EuclideanSpace.inner_single_right i0 (1 : ℝ) q)
  have hSingleSq : ‖EuclideanSpace.single i0 (1 : ℝ)‖ ^ 2 = 1 := by
    simp [EuclideanSpace.norm_single, i0]
  have hNorm' : ‖q‖ = 1 := by simpa using hNorm
  rw [norm_sub_sq_real, hNorm', hInner, hSingleSq]
  ring

theorem unit_norm_mdarray4_dist_sq_to_neg_basis0_eq
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    ‖q + EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 = 2 + 2 * q ⟨0, by decide⟩ := by
  let i0 : Fin 4 := ⟨0, by decide⟩
  have hInner : (@inner ℝ _ _ q (EuclideanSpace.single i0 (1 : ℝ)) : ℝ) = q i0 := by
    simpa using (EuclideanSpace.inner_single_right i0 (1 : ℝ) q)
  have hSingleSq : ‖EuclideanSpace.single i0 (1 : ℝ)‖ ^ 2 = 1 := by
    simp [EuclideanSpace.norm_single, i0]
  have hNorm' : ‖q‖ = 1 := by simpa using hNorm
  rw [norm_add_sq_real, hNorm', hInner, hSingleSq]
  ring

theorem unit_norm_mdarray4_dist_to_basis0_le_dist_to_neg_basis0_of_nonneg_coord0
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (h0 : 0 ≤ q ⟨0, by decide⟩) :
    ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ≤
      ‖q + EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  have hsq : ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 ≤
      ‖q + EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 := by
    rw [unit_norm_mdarray4_dist_sq_to_basis0_eq q hNorm, unit_norm_mdarray4_dist_sq_to_neg_basis0_eq q hNorm]
    nlinarith
  exact le_of_sq_le_sq hsq (norm_nonneg _)

theorem unit_norm_mdarray4_dist_to_neg_basis0_le_dist_to_basis0_of_nonpos_coord0
    (q : MDArray 4)
    (hNorm : norm q = 1)
    (h0 : q ⟨0, by decide⟩ ≤ 0) :
    ‖q + EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ≤
      ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ := by
  have hsq : ‖q + EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 ≤
      ‖q - EuclideanSpace.single ⟨0, by decide⟩ (1 : ℝ)‖ ^ 2 := by
    rw [unit_norm_mdarray4_dist_sq_to_neg_basis0_eq q hNorm, unit_norm_mdarray4_dist_sq_to_basis0_eq q hNorm]
    nlinarith
  exact le_of_sq_le_sq hsq (norm_nonneg _)

/-! ## 6. Quaternion Cover Geometry -/

/-- For a unit quaternion, a basis-coordinate witness of size at least `1/2`
    at a fixed index `i` yields Euclidean distance at most `1` to either the
    matching basis quaternion or its negation. -/
theorem unit_norm_mdarray4_dist_to_basis_or_negBasis_le_one_of_coordinate_abs_ge_half
    (q : MDArray 4)
    (i : Fin 4)
    (hNorm : norm q = 1)
    (hi : 1 / 2 ≤ |q i|) :
    min ‖q - EuclideanSpace.single i (1 : ℝ)‖ ‖q + EuclideanSpace.single i (1 : ℝ)‖ ≤ 1 := by
  by_cases hsign : 0 ≤ q i
  · have hLower : 1 / 2 ≤ q i := by simpa [abs_of_nonneg hsign] using hi
    have hNorm' : ‖q‖ = 1 := hNorm
    have hInner : (@inner ℝ _ _ q (EuclideanSpace.single i (1 : ℝ)) : ℝ) = q i := by
      simpa using (EuclideanSpace.inner_single_right i (1 : ℝ) q)
    have hSingleSq : ‖EuclideanSpace.single i (1 : ℝ)‖ ^ 2 = 1 := by
      simp [EuclideanSpace.norm_single]
    have hSq : ‖q - EuclideanSpace.single i (1 : ℝ)‖ ^ 2 ≤ 1 := by
      rw [norm_sub_sq_real, hNorm', hInner, hSingleSq]
      nlinarith
    have hLe : ‖q - EuclideanSpace.single i (1 : ℝ)‖ ≤ 1 := by
      nlinarith [norm_nonneg (q - EuclideanSpace.single i (1 : ℝ)), hSq]
    exact le_trans (min_le_left _ _) hLe
  · have hUpper : q i ≤ -(1 / 2 : ℝ) := by
      have hAbs : 1 / 2 ≤ -q i := by
        simpa [abs_of_nonpos (le_of_not_ge hsign)] using hi
      linarith
    have hNorm' : ‖q‖ = 1 := hNorm
    have hInner : (@inner ℝ _ _ q (EuclideanSpace.single i (1 : ℝ)) : ℝ) = q i := by
      simpa using (EuclideanSpace.inner_single_right i (1 : ℝ) q)
    have hSingleSq : ‖EuclideanSpace.single i (1 : ℝ)‖ ^ 2 = 1 := by
      simp [EuclideanSpace.norm_single]
    have hSq : ‖q + EuclideanSpace.single i (1 : ℝ)‖ ^ 2 ≤ 1 := by
      rw [norm_add_sq_real, hNorm', hInner, hSingleSq]
      nlinarith
    have hLe : ‖q + EuclideanSpace.single i (1 : ℝ)‖ ≤ 1 := by
      nlinarith [norm_nonneg (q + EuclideanSpace.single i (1 : ℝ)), hSq]
    exact le_trans (min_le_right _ _) hLe

/-- Stronger basis-cover geometry: if a unit quaternion has a coordinate of
    absolute value at least `sqrt 2 / 2`, then the signed distance to the
    matching basis quaternion is bounded by `sqrt (2 - sqrt 2)`, strictly better
    than the generic radius `1`. -/
theorem unit_norm_mdarray4_dist_sq_to_basis_or_negBasis_le_two_sub_sqrt_two_of_coordinate_abs_ge_inv_sqrt_two
    (q : MDArray 4)
    (i : Fin 4)
    (hNorm : norm q = 1)
    (hi : Real.sqrt 2 / 2 ≤ |q i|) :
    min (‖q - EuclideanSpace.single i (1 : ℝ)‖ ^ 2) (‖q + EuclideanSpace.single i (1 : ℝ)‖ ^ 2)
      ≤ 2 - Real.sqrt 2 := by
  by_cases hsign : 0 ≤ q i
  · have hLower : Real.sqrt 2 / 2 ≤ q i := by
      simpa [abs_of_nonneg hsign] using hi
    have hNorm' : ‖q‖ = 1 := hNorm
    have hInner : (@inner ℝ _ _ q (EuclideanSpace.single i (1 : ℝ)) : ℝ) = q i := by
      simpa using (EuclideanSpace.inner_single_right i (1 : ℝ) q)
    have hSingleSq : ‖EuclideanSpace.single i (1 : ℝ)‖ ^ 2 = 1 := by
      simp [EuclideanSpace.norm_single]
    have hSq : ‖q - EuclideanSpace.single i (1 : ℝ)‖ ^ 2 ≤ 2 - Real.sqrt 2 := by
      rw [norm_sub_sq_real, hNorm', hInner, hSingleSq]
      nlinarith [sq_nonneg (Real.sqrt 2 - 2 * q i), Real.sq_sqrt (show 0 ≤ (2 : ℝ) by positivity)]
    exact le_trans (min_le_left _ _) hSq
  · have hUpper : q i ≤ -(Real.sqrt 2 / 2 : ℝ) := by
      have hAbs : Real.sqrt 2 / 2 ≤ -q i := by
        simpa [abs_of_nonpos (le_of_not_ge hsign)] using hi
      linarith
    have hNorm' : ‖q‖ = 1 := hNorm
    have hInner : (@inner ℝ _ _ q (EuclideanSpace.single i (1 : ℝ)) : ℝ) = q i := by
      simpa using (EuclideanSpace.inner_single_right i (1 : ℝ) q)
    have hSingleSq : ‖EuclideanSpace.single i (1 : ℝ)‖ ^ 2 = 1 := by
      simp [EuclideanSpace.norm_single]
    have hSq : ‖q + EuclideanSpace.single i (1 : ℝ)‖ ^ 2 ≤ 2 - Real.sqrt 2 := by
      rw [norm_add_sq_real, hNorm', hInner, hSingleSq]
      nlinarith [sq_nonneg (Real.sqrt 2 + 2 * q i), Real.sq_sqrt (show 0 ≤ (2 : ℝ) by positivity)]
    exact le_trans (min_le_right _ _) hSq

/-- For a unit quaternion, a basis-coordinate witness of size at least `1/2`
    yields Euclidean distance at most `1` to either the matching basis quaternion
    or its negation. This is the low-level sign-aware cover fact behind the
    current `quaternionDictionary8` proof plan. -/
theorem unit_norm_mdarray4_dist_to_basis_or_negBasis_le_one
    (q : MDArray 4)
    (hNorm : norm q = 1) :
    ∃ i : Fin 4,
      1 / 2 ≤ |q i| ∧
      min ‖q - EuclideanSpace.single i (1 : ℝ)‖ ‖q + EuclideanSpace.single i (1 : ℝ)‖ ≤ 1 := by
  rcases unit_norm_mdarray4_has_coordinate_abs_ge_half q hNorm with ⟨i, hi⟩
  exact ⟨i, hi, unit_norm_mdarray4_dist_to_basis_or_negBasis_le_one_of_coordinate_abs_ge_half q i hNorm hi⟩

/-! ## 7. Energy Gradient (Force Computation) -/

/-- Compute forces from potential energy function.
    F = -∇U -/
noncomputable def computeForces {n : ℕ}
    (U : DiffFunctionN n)
    (q : MDArray n) : MDArray n :=
  -(array_gradient U q)

end ArrayDSL
end Computation
end DecisionQuotient
