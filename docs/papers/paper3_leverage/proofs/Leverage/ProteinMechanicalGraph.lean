import Leverage.BridgeToDQ
import Mathlib.Tactic
import DecisionQuotient.Tractability.MolecularSrank

namespace Leverage

open Classical DecisionQuotient
open DecisionQuotient.Tractability
open DecisionQuotient.Tractability.MolecularSrank
open scoped BigOperators

/-- Generic docking rank bound from any explicit finite cover of the decision-relevant protein atoms. -/
theorem molecularDocking_srank_bound_of_atomCover
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (proteinCover : Finset (Fin prob.protein.numAtoms))
    (hCover : relevantAtomPositions prob ⊆ proteinCover) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * proteinCover.card + 3 * prob.ligand.numAtoms := by
  have hBase := md_srank_bound prob hStrictAll hBounded
  have hAtoms : numRelevantAtoms prob ≤ proteinCover.card := by
    simpa [numRelevantAtoms] using Finset.card_le_card hCover
  omega

/-- Euclidean distance is symmetric. -/
theorem pointDistance_comm (p1 p2 : ℝ × ℝ × ℝ) :
    pointDistance p1 p2 = pointDistance p2 p1 := by
  rcases p1 with ⟨x1, yz1⟩
  rcases yz1 with ⟨y1, z1⟩
  rcases p2 with ⟨x2, yz2⟩
  rcases yz2 with ⟨y2, z2⟩
  have hArg :
      (x1 - x2) ^ (2 : ℕ) + (y1 - y2) ^ (2 : ℕ) + (z1 - z2) ^ (2 : ℕ) =
        (x2 - x1) ^ (2 : ℕ) + (y2 - y1) ^ (2 : ℕ) + (z2 - z1) ^ (2 : ℕ) := by
    ring_nf
  unfold pointDistance
  exact congrArg Real.sqrt hArg

/-- Reference-structure distance between two protein atoms. -/
noncomputable def proteinAtomDistance (prob : MDBindingProblem)
    (i j : Fin prob.protein.numAtoms) : ℝ :=
  pointDistance (proteinAtom prob i.1 i.2).position (proteinAtom prob j.1 j.2).position

theorem proteinAtomDistance_comm (prob : MDBindingProblem)
    (i j : Fin prob.protein.numAtoms) :
    proteinAtomDistance prob i j = proteinAtomDistance prob j i := by
  simpa [proteinAtomDistance] using
    pointDistance_comm
      (p1 := (proteinAtom prob i.1 i.2).position)
      (p2 := (proteinAtom prob j.1 j.2).position)

/-- Explicit undirected rigid-mechanical adjacency graph on protein atoms. -/
structure ProteinMechanicalGraph (prob : MDBindingProblem) where
  rigidEdge : Fin prob.protein.numAtoms → Fin prob.protein.numAtoms → Prop
  symmetric : Symmetric rigidEdge

/-- Geometry-derived contact graph on the reference protein structure: two atoms are mechanically
adjacent when their reference positions lie within the declared contact radius. -/
noncomputable def proteinContactGraph (prob : MDBindingProblem) (contactRadius : ℝ) :
    ProteinMechanicalGraph prob where
  rigidEdge i j := proteinAtomDistance prob i j ≤ contactRadius
  symmetric := by
    intro i j hij
    simpa [proteinAtomDistance_comm prob j i] using hij

/-- Geometry-derived active pocket consisting of all protein atoms whose reference positions lie within
radius `pocketRadius` of the binding-site center. -/
noncomputable def activePocketByRadius (prob : MDBindingProblem) (pocketRadius : ℝ) :
    Finset (Fin prob.protein.numAtoms) :=
  Finset.univ.filter (fun i =>
    atomWithinCutoff (proteinAtom prob i.1 i.2) prob.bindingSite pocketRadius)

theorem mem_activePocketByRadius
    (prob : MDBindingProblem) (pocketRadius : ℝ)
    (i : Fin prob.protein.numAtoms) :
    i ∈ activePocketByRadius prob pocketRadius ↔
      atomWithinCutoff (proteinAtom prob i.1 i.2) prob.bindingSite pocketRadius := by
  simp [activePocketByRadius]

/-- Canonical active pocket induced by the declared binding-site radius. -/
noncomputable def bindingSiteActivePocket (prob : MDBindingProblem) :
    Finset (Fin prob.protein.numAtoms) :=
  activePocketByRadius prob prob.bindingSite.radius

theorem relevantAtomPositions_eq_activePocketByRadius_cutoff (prob : MDBindingProblem) :
    relevantAtomPositions prob = activePocketByRadius prob prob.cutoff := by
  ext i
  simp [relevantAtomPositions, activePocketByRadius]

/-- Explicit rigid path between two protein atoms in a mechanical graph. -/
structure RigidMechanicalPath {prob : MDBindingProblem}
    (G : ProteinMechanicalGraph prob)
    (start finish : Fin prob.protein.numAtoms) where
  length : ℕ
  vertex : Fin (length + 1) → Fin prob.protein.numAtoms
  start_eq : vertex 0 = start
  finish_eq : vertex (Fin.last length) = finish
  rigidStep : ∀ t : Fin length, G.rigidEdge (vertex t.castSucc) (vertex t.succ)

/-- A single rigid edge gives a length-one rigid path. -/
def RigidMechanicalPath.ofEdge
    {prob : MDBindingProblem} {G : ProteinMechanicalGraph prob}
    {start finish : Fin prob.protein.numAtoms}
    (hEdge : G.rigidEdge start finish) :
    RigidMechanicalPath G start finish where
  length := 1
  vertex := fun t => if t.1 = 0 then start else finish
  start_eq := by simp
  finish_eq := by simp
  rigidStep := by
    intro t
    fin_cases t
    simp [hEdge]

namespace RigidMechanicalPath

/-- Set-valued relation view of a mechanical graph edge predicate. -/
private def rigidSetRel
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob) :
    SetRel (Fin prob.protein.numAtoms) (Fin prob.protein.numAtoms) :=
  { ij | G.rigidEdge ij.1 ij.2 }

/-- Concatenation of two rigid mechanical paths meeting at a common intermediate atom. -/
@[simps length]
def append
    {prob : MDBindingProblem} {G : ProteinMechanicalGraph prob}
    {start middle finish : Fin prob.protein.numAtoms}
    (p : RigidMechanicalPath G start middle)
    (q : RigidMechanicalPath G middle finish) :
    RigidMechanicalPath G start finish := by
  let pSeries : RelSeries (rigidSetRel G) :=
    { length := p.length
      toFun := p.vertex
      step := by
        intro i
        simpa [rigidSetRel] using p.rigidStep i }
  let qSeries : RelSeries (rigidSetRel G) :=
    { length := q.length
      toFun := q.vertex
      step := by
        intro i
        simpa [rigidSetRel] using q.rigidStep i }
  have hConnect : pSeries.last = qSeries.head := by
    have hpLast : pSeries.last = middle := by
      simpa [pSeries, RelSeries.last] using p.finish_eq
    have hqHead : qSeries.head = middle := by
      simpa [qSeries, RelSeries.head] using q.start_eq
    exact hpLast.trans hqHead.symm
  refine
    { length := (pSeries.smash qSeries hConnect).length
      vertex := (pSeries.smash qSeries hConnect).toFun
      start_eq := ?_
      finish_eq := ?_
      rigidStep := by
        intro i
        simpa [rigidSetRel] using (pSeries.smash qSeries hConnect).step i }
  · have hpHead : pSeries.head = start := by
      simpa [pSeries, RelSeries.head] using p.start_eq
    exact (RelSeries.head_smash (p := pSeries) (q := qSeries) hConnect).trans hpHead
  · have hqLast : qSeries.last = finish := by
      simpa [qSeries, RelSeries.last] using q.finish_eq
    exact (RelSeries.last_smash (p := pSeries) (q := qSeries) hConnect).trans hqLast

end RigidMechanicalPath

namespace ProteinMechanicalGraph

/-- Reachability in the rigid mechanical graph, witnessed by an explicit rigid path. -/
def Reaches {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (start finish : Fin prob.protein.numAtoms) : Prop :=
  Nonempty (RigidMechanicalPath G start finish)

/-- Every atom is mechanically reachable from itself by the length-zero rigid path. -/
theorem reaches_refl {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (i : Fin prob.protein.numAtoms) :
    G.Reaches i i := by
  refine ⟨{
    length := 0
    vertex := fun _ => i
    start_eq := rfl
    finish_eq := by simp
    rigidStep := ?_
  }⟩
  intro t
  exact t.elim0

/-- Mechanical reachability composes along concatenated rigid paths. -/
theorem reaches_trans {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    {start middle finish : Fin prob.protein.numAtoms}
    (hStart : G.Reaches start middle) (hFinish : G.Reaches middle finish) :
    G.Reaches start finish := by
  rcases hStart with ⟨p⟩
  rcases hFinish with ⟨q⟩
  exact ⟨p.append q⟩

end ProteinMechanicalGraph

/-- Every explicit rigid path certifies graph reachability between its endpoints. -/
theorem RigidMechanicalPath.toReaches
    {prob : MDBindingProblem} {G : ProteinMechanicalGraph prob}
    {start finish : Fin prob.protein.numAtoms}
    (p : RigidMechanicalPath G start finish) :
    G.Reaches start finish :=
  ⟨p⟩

namespace ProteinMechanicalGraph

/-- Mechanical neighborhood of an active pocket: all atoms connected to the pocket through some rigid
path in the protein mechanical graph. -/
noncomputable def mechanicalNeighborhood
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    Finset (Fin prob.protein.numAtoms) :=
  Finset.univ.filter (fun i => ∃ j ∈ activePocket, G.Reaches i j)

theorem mem_mechanicalNeighborhood
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (i : Fin prob.protein.numAtoms) :
    i ∈ G.mechanicalNeighborhood activePocket ↔
      ∃ j ∈ activePocket, G.Reaches i j := by
  simp [mechanicalNeighborhood]

/-- The active pocket sits inside its own rigid mechanical neighborhood. -/
theorem subset_mechanicalNeighborhood
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    activePocket ⊆ G.mechanicalNeighborhood activePocket := by
  intro i hi
  exact (G.mem_mechanicalNeighborhood activePocket i).2 ⟨i, hi, G.reaches_refl i⟩

/-- Any neighborhood generated from a set already lying in the full mechanical neighborhood remains inside
that same full neighborhood. -/
theorem mechanicalNeighborhood_subset_of_subset
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    {activePocket inner : Finset (Fin prob.protein.numAtoms)}
    (hInner : inner ⊆ G.mechanicalNeighborhood activePocket) :
    G.mechanicalNeighborhood inner ⊆ G.mechanicalNeighborhood activePocket := by
  intro i hi
  rcases (G.mem_mechanicalNeighborhood inner i).1 hi with ⟨j, hj, hij⟩
  rcases (G.mem_mechanicalNeighborhood activePocket j).1 (hInner hj) with ⟨k, hk, hjk⟩
  exact (G.mem_mechanicalNeighborhood activePocket i).2 ⟨k, hk, G.reaches_trans hij hjk⟩

/-- Direct one-hop contact shell around an active pocket. -/
noncomputable def contactShell
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    Finset (Fin prob.protein.numAtoms) :=
  Finset.univ.filter (fun i => ∃ j ∈ activePocket, G.rigidEdge i j)

theorem mem_contactShell
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (i : Fin prob.protein.numAtoms) :
    i ∈ G.contactShell activePocket ↔
      ∃ j ∈ activePocket, G.rigidEdge i j := by
  simp [contactShell]

/-- Every direct one-hop contact-shell atom lies in the full mechanical neighborhood. -/
theorem contactShell_subset_mechanicalNeighborhood
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    G.contactShell activePocket ⊆ G.mechanicalNeighborhood activePocket := by
  intro i hi
  rcases (G.mem_contactShell activePocket i).1 hi with ⟨j, hj, hEdge⟩
  exact (G.mem_mechanicalNeighborhood activePocket i).2
    ⟨j, hj, RigidMechanicalPath.toReaches (RigidMechanicalPath.ofEdge hEdge)⟩

/-- Outer direct-contact shell around an active pocket, excluding the pocket atoms themselves. -/
noncomputable def outerContactShell
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    Finset (Fin prob.protein.numAtoms) :=
  G.contactShell activePocket \ activePocket

theorem mem_outerContactShell
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (i : Fin prob.protein.numAtoms) :
    i ∈ G.outerContactShell activePocket ↔
      i ∉ activePocket ∧ ∃ j ∈ activePocket, G.rigidEdge i j := by
  constructor
  · intro hi
    rcases Finset.mem_sdiff.mp hi with ⟨hShell, hNot⟩
    exact ⟨hNot, (G.mem_contactShell activePocket i).1 hShell⟩
  · rintro ⟨hNot, hEdge⟩
    exact Finset.mem_sdiff.mpr ⟨(G.mem_contactShell activePocket i).2 hEdge, hNot⟩

theorem disjoint_activePocket_outerContactShell
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    Disjoint activePocket (G.outerContactShell activePocket) := by
  refine Finset.disjoint_left.mpr ?_
  intro i hiPocket hiOuter
  exact (Finset.mem_sdiff.mp hiOuter).2 hiPocket

theorem outerContactShell_subset_mechanicalNeighborhood
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    G.outerContactShell activePocket ⊆ G.mechanicalNeighborhood activePocket := by
  intro i hi
  exact G.contactShell_subset_mechanicalNeighborhood activePocket
    ((Finset.mem_sdiff.mp hi).1)

/-- If a working set is already inside the full mechanical neighborhood of an active pocket, then its direct
contact shell also stays inside that same full neighborhood. -/
theorem contactShell_subset_mechanicalNeighborhood_of_subset
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    {activePocket current : Finset (Fin prob.protein.numAtoms)}
    (hCurrent : current ⊆ G.mechanicalNeighborhood activePocket) :
    G.contactShell current ⊆ G.mechanicalNeighborhood activePocket := by
  intro i hi
  rcases (G.mem_contactShell current i).1 hi with ⟨j, hj, hEdge⟩
  rcases (G.mem_mechanicalNeighborhood activePocket j).1 (hCurrent hj) with ⟨k, hk, hjk⟩
  exact (G.mem_mechanicalNeighborhood activePocket i).2
    ⟨k, hk, G.reaches_trans (RigidMechanicalPath.ofEdge hEdge).toReaches hjk⟩

/-- The outer contact shell of any subset of the full mechanical neighborhood remains inside the same full
mechanical neighborhood. -/
theorem outerContactShell_subset_mechanicalNeighborhood_of_subset
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    {activePocket current : Finset (Fin prob.protein.numAtoms)}
    (hCurrent : current ⊆ G.mechanicalNeighborhood activePocket) :
    G.outerContactShell current ⊆ G.mechanicalNeighborhood activePocket := by
  intro i hi
  exact G.contactShell_subset_mechanicalNeighborhood_of_subset hCurrent
    ((Finset.mem_sdiff.mp hi).1)

/-- Iterated outer-contact expansion of an active pocket. Step `0` is the pocket itself; step `k+1`
adjoins the outer contact shell of the current `k`-hop neighborhood. -/
noncomputable def kHopNeighborhood
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    ℕ → Finset (Fin prob.protein.numAtoms)
  | 0 => activePocket
  | k + 1 => kHopNeighborhood G activePocket k ∪
      G.outerContactShell (kHopNeighborhood G activePocket k)

@[simp] theorem kHopNeighborhood_zero
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) :
    G.kHopNeighborhood activePocket 0 = activePocket := rfl

@[simp] theorem kHopNeighborhood_succ
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) (k : ℕ) :
    G.kHopNeighborhood activePocket (k + 1) =
      G.kHopNeighborhood activePocket k ∪
        G.outerContactShell (G.kHopNeighborhood activePocket k) := rfl

theorem subset_kHopNeighborhood_succ
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) (k : ℕ) :
    G.kHopNeighborhood activePocket k ⊆ G.kHopNeighborhood activePocket (k + 1) := by
  intro i hi
  simp [hi]

/-- Monotonicity in hop depth: deeper expansions contain shallower expansions. -/
theorem kHopNeighborhood_mono
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    {k m : ℕ} (hkm : k ≤ m) :
    G.kHopNeighborhood activePocket k ⊆ G.kHopNeighborhood activePocket m := by
  induction hkm with
  | refl =>
      intro i hi
      exact hi
  | @step m hkm ih =>
      intro i hi
      exact G.subset_kHopNeighborhood_succ activePocket m (ih hi)

theorem kHopNeighborhood_card_mono
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    {k m : ℕ} (hkm : k ≤ m) :
    (G.kHopNeighborhood activePocket k).card ≤ (G.kHopNeighborhood activePocket m).card := by
  exact Finset.card_le_card (G.kHopNeighborhood_mono activePocket hkm)

/-- Every bounded-depth contact expansion of an active pocket stays inside the full mechanical neighborhood
generated by unrestricted graph reachability. -/
theorem kHopNeighborhood_subset_mechanicalNeighborhood
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) (k : ℕ) :
    G.kHopNeighborhood activePocket k ⊆ G.mechanicalNeighborhood activePocket := by
  induction k with
  | zero =>
      simpa using G.subset_mechanicalNeighborhood activePocket
  | succ k ih =>
      intro i hi
      rw [G.kHopNeighborhood_succ] at hi
      rcases Finset.mem_union.mp hi with hi | hi
      · exact ih hi
      · exact G.outerContactShell_subset_mechanicalNeighborhood_of_subset ih hi

theorem kHopNeighborhood_card_le_mechanicalNeighborhood_card
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) (k : ℕ) :
    (G.kHopNeighborhood activePocket k).card ≤ (G.mechanicalNeighborhood activePocket).card := by
  exact Finset.card_le_card (G.kHopNeighborhood_subset_mechanicalNeighborhood activePocket k)

theorem kHopNeighborhood_card_succ
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms)) (k : ℕ) :
    (G.kHopNeighborhood activePocket (k + 1)).card =
      (G.kHopNeighborhood activePocket k).card +
        (G.outerContactShell (G.kHopNeighborhood activePocket k)).card := by
  rw [G.kHopNeighborhood_succ, Finset.card_union_of_disjoint]
  exact G.disjoint_activePocket_outerContactShell (G.kHopNeighborhood activePocket k)

theorem kHopNeighborhood_card_le_of_shellBounds
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (shellBound : ℕ → ℕ) (k : ℕ)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤ shellBound m) :
    (G.kHopNeighborhood activePocket k).card ≤
      activePocket.card + Finset.sum (Finset.range k) shellBound := by
  induction k with
  | zero =>
      simp [G.kHopNeighborhood_zero]
  | succ k ih =>
      have hPrev :
          (G.kHopNeighborhood activePocket k).card ≤
            activePocket.card + Finset.sum (Finset.range k) shellBound := by
        apply ih
        intro m hm
        exact hShell m (Nat.lt_trans hm (Nat.lt_succ_self k))
      have hLayer :
          (G.outerContactShell (G.kHopNeighborhood activePocket k)).card ≤ shellBound k :=
        hShell k (Nat.lt_succ_self k)
      have hSum :
          Finset.sum (Finset.range (k + 1)) shellBound =
            Finset.sum (Finset.range k) shellBound + shellBound k := by
        simpa using (Finset.sum_range_succ (f := shellBound) k)
      rw [G.kHopNeighborhood_card_succ, hSum]
      omega

theorem kHopNeighborhood_card_le_of_uniformShell
    {prob : MDBindingProblem} (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k K : ℕ)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤ K) :
    (G.kHopNeighborhood activePocket k).card ≤ activePocket.card + k * K := by
  have hBase := G.kHopNeighborhood_card_le_of_shellBounds activePocket (fun _ => K) k hShell
  simpa using hBase

/-- Distance-indexed upper profile for per-shell allosteric rank contribution
(`3` coordinates per shell atom in the current coordinate model). -/
def allostericDistanceContributionUpper (distanceProfile : ℕ → ℕ) (d : ℕ) : ℕ :=
  3 * distanceProfile d

/-- If the shell envelope is non-increasing with graph distance, the induced
allosteric contribution upper profile is also non-increasing. -/
theorem allostericDistanceContributionUpper_antitone
    (distanceProfile : ℕ → ℕ)
    (hDecay : ∀ d : ℕ, distanceProfile (d + 1) ≤ distanceProfile d) :
    ∀ d : ℕ,
      allostericDistanceContributionUpper distanceProfile (d + 1) ≤
        allostericDistanceContributionUpper distanceProfile d := by
  intro d
  unfold allostericDistanceContributionUpper
  exact Nat.mul_le_mul_left 3 (hDecay d)

/-- Qualitative allosteric decay law: under a non-increasing shell envelope,
the declared per-distance allosteric contribution upper bound is monotone
non-increasing with graph distance. -/
theorem allosteric_contribution_decay_law
    (distanceProfile : ℕ → ℕ)
    (hDecay : ∀ d : ℕ, distanceProfile (d + 1) ≤ distanceProfile d) :
    ∀ d : ℕ,
      allostericDistanceContributionUpper distanceProfile (d + 1) ≤
        allostericDistanceContributionUpper distanceProfile d :=
  allostericDistanceContributionUpper_antitone distanceProfile hDecay

/-- Real-valued exponential shell envelope for distance-decay allostery. -/
noncomputable def allostericExponentialShellEnvelope (C α : ℝ) (d : ℕ) : ℝ :=
  C * Real.exp (-α * (d : ℝ))

/-- For nonnegative envelope scale and decay rate, the exponential shell envelope
is non-increasing with graph distance. -/
theorem allostericExponentialShellEnvelope_antitone
    {C α : ℝ} (hC : 0 ≤ C) (hα : 0 ≤ α) :
    ∀ d : ℕ,
      allostericExponentialShellEnvelope C α (d + 1) ≤
        allostericExponentialShellEnvelope C α d := by
  intro d
  unfold allostericExponentialShellEnvelope
  have hStep : (d : ℝ) ≤ (((d + 1 : ℕ) : ℝ)) := by
    exact_mod_cast Nat.le_succ d
  have hArg : -α * (((d + 1 : ℕ) : ℝ)) ≤ -α * (d : ℝ) := by
    nlinarith [hStep, hα]
  have hExp :
      Real.exp (-α * (((d + 1 : ℕ) : ℝ))) ≤ Real.exp (-α * (d : ℝ)) :=
    Real.exp_le_exp.mpr hArg
  exact mul_le_mul_of_nonneg_left hExp hC

/-- Integer shell budget extracted from the exponential real envelope by
ceiling each shell level. -/
noncomputable def allostericExponentialDistanceProfile (C α : ℝ) (d : ℕ) : ℕ :=
  Nat.ceil (allostericExponentialShellEnvelope C α d)

/-- The ceiling-based exponential distance profile is non-increasing whenever
the underlying real exponential envelope is non-increasing. -/
theorem allostericExponentialDistanceProfile_antitone
    {C α : ℝ} (hC : 0 ≤ C) (hα : 0 ≤ α) :
    ∀ d : ℕ,
      allostericExponentialDistanceProfile C α (d + 1) ≤
        allostericExponentialDistanceProfile C α d := by
  intro d
  unfold allostericExponentialDistanceProfile
  exact Nat.ceil_mono (allostericExponentialShellEnvelope_antitone hC hα d)

/-- Exponential distance-decay corollary of the qualitative allosteric decay
law for the declared integer shell-contribution profile. -/
theorem allosteric_exponential_contribution_decay_law
    {C α : ℝ} (hC : 0 ≤ C) (hα : 0 ≤ α) :
    ∀ d : ℕ,
      allostericDistanceContributionUpper (allostericExponentialDistanceProfile C α) (d + 1) ≤
        allostericDistanceContributionUpper (allostericExponentialDistanceProfile C α) d :=
  allostericDistanceContributionUpper_antitone
    (allostericExponentialDistanceProfile C α)
    (allostericExponentialDistanceProfile_antitone hC hα)

/-- Real-valued polynomial shell envelope for distance-decay allostery. -/
noncomputable def allostericPolynomialShellEnvelope (C : ℝ) (p : ℕ) (d : ℕ) : ℝ :=
  C / ((d : ℝ) + 1) ^ p

/-- For nonnegative envelope scale, the polynomial shell envelope is
non-increasing with graph distance. -/
theorem allostericPolynomialShellEnvelope_antitone
    {C : ℝ} {p : ℕ} (hC : 0 ≤ C) :
    ∀ d : ℕ,
      allostericPolynomialShellEnvelope C p (d + 1) ≤
        allostericPolynomialShellEnvelope C p d := by
  intro d
  unfold allostericPolynomialShellEnvelope
  have hStep : (d : ℝ) + 1 ≤ (((d + 1 : ℕ) : ℝ) + 1) := by
    have hNat : (d : ℝ) ≤ ((d + 1 : ℕ) : ℝ) := by
      exact_mod_cast Nat.le_succ d
    linarith
  have hBaseNonneg : 0 ≤ (d : ℝ) + 1 := by positivity
  have hDenLe :
      ((d : ℝ) + 1) ^ p ≤ ((((d + 1 : ℕ) : ℝ) + 1) ^ p) :=
    pow_le_pow_left₀ hBaseNonneg hStep p
  have hDenPos : 0 < ((d : ℝ) + 1) ^ p := by positivity
  exact div_le_div_of_nonneg_left hC hDenPos hDenLe

/-- Integer shell budget extracted from the polynomial real envelope by
ceiling each shell level. -/
noncomputable def allostericPolynomialDistanceProfile (C : ℝ) (p : ℕ) (d : ℕ) : ℕ :=
  Nat.ceil (allostericPolynomialShellEnvelope C p d)

/-- The ceiling-based polynomial distance profile is non-increasing whenever
the underlying real polynomial envelope is non-increasing. -/
theorem allostericPolynomialDistanceProfile_antitone
    {C : ℝ} {p : ℕ} (hC : 0 ≤ C) :
    ∀ d : ℕ,
      allostericPolynomialDistanceProfile C p (d + 1) ≤
        allostericPolynomialDistanceProfile C p d := by
  intro d
  unfold allostericPolynomialDistanceProfile
  exact Nat.ceil_mono (allostericPolynomialShellEnvelope_antitone hC d)

/-- Polynomial distance-decay corollary of the qualitative allosteric decay
law for the declared integer shell-contribution profile. -/
theorem allosteric_polynomial_contribution_decay_law
    {C : ℝ} {p : ℕ} (hC : 0 ≤ C) :
    ∀ d : ℕ,
      allostericDistanceContributionUpper (allostericPolynomialDistanceProfile C p) (d + 1) ≤
        allostericDistanceContributionUpper (allostericPolynomialDistanceProfile C p) d :=
  allostericDistanceContributionUpper_antitone
    (allostericPolynomialDistanceProfile C p)
    (allostericPolynomialDistanceProfile_antitone hC)

end ProteinMechanicalGraph

/-- The starting atom of a rigid path belongs to the mechanical neighborhood of any pocket that contains
the terminal site atom. -/
theorem RigidMechanicalPath.start_mem_mechanicalNeighborhood
    {prob : MDBindingProblem} {G : ProteinMechanicalGraph prob}
    {start finish : Fin prob.protein.numAtoms}
    (p : RigidMechanicalPath G start finish)
    {activePocket : Finset (Fin prob.protein.numAtoms)}
    (hFinish : finish ∈ activePocket) :
    start ∈ G.mechanicalNeighborhood activePocket := by
  exact (G.mem_mechanicalNeighborhood activePocket start).2
    ⟨finish, hFinish, RigidMechanicalPath.toReaches p⟩

/-- Reusable allosteric locality theorem: if every decision-relevant protein atom is mechanically coupled to
the active pocket through the protein mechanical graph, docking rank is controlled by the graph-neighborhood
cardinality together with the ligand degrees of freedom. -/
theorem allosteric_srank_graph_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (hCovered : relevantAtomPositions prob ⊆ G.mechanicalNeighborhood activePocket) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (G.mechanicalNeighborhood activePocket).card + 3 * prob.ligand.numAtoms := by
  exact molecularDocking_srank_bound_of_atomCover prob hStrictAll hBounded
    (G.mechanicalNeighborhood activePocket) hCovered

/-- Bridge from bounded-depth coverage to the full mechanical-neighborhood theorem: any k-hop cover
immediately yields the unbounded graph-local rank bound. -/
theorem allosteric_srank_graph_bound_of_kHopCover_mechanicalNeighborhood
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (G.mechanicalNeighborhood activePocket).card + 3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound prob hStrictAll hBounded G activePocket
    (fun i hi => G.kHopNeighborhood_subset_mechanicalNeighborhood activePocket k (hCovered hi))

/-- Concrete one-hop version of the allosteric graph theorem: if every relevant atom lies either in the
active pocket itself or in its outer direct contact shell, the rank bound splits into pocket, shell, and
ligand contributions without double-counting pocket atoms. -/
theorem allosteric_srank_graph_bound_of_contactShell_cover
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (hCovered : relevantAtomPositions prob ⊆ activePocket ∪ G.outerContactShell activePocket) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * activePocket.card + 3 * (G.outerContactShell activePocket).card +
        3 * prob.ligand.numAtoms := by
  have hBase := molecularDocking_srank_bound_of_atomCover prob hStrictAll hBounded
    (activePocket ∪ G.outerContactShell activePocket) hCovered
  have hCard : (activePocket ∪ G.outerContactShell activePocket).card ≤
      activePocket.card + (G.outerContactShell activePocket).card :=
    Finset.card_union_le _ _
  omega

/-- k-hop version of graph-local allostery: if every relevant atom lies in the `k`-hop outer-contact
expansion of the active pocket, docking rank is controlled by the size of that finite expansion. -/
theorem allosteric_srank_graph_bound_of_kHopNeighborhood
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (G.kHopNeighborhood activePocket k).card + 3 * prob.ligand.numAtoms := by
  exact molecularDocking_srank_bound_of_atomCover prob hStrictAll hBounded
    (G.kHopNeighborhood activePocket k) hCovered

/-- Coverage transfer across hop depth: if relevant atoms are covered at depth `k`, then any deeper
depth `m ≥ k` also covers them. -/
theorem relevantAtomPositions_subset_kHopNeighborhood_of_le
    (prob : MDBindingProblem)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    {k m : ℕ} (hkm : k ≤ m)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k) :
    relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket m := by
  intro i hi
  exact G.kHopNeighborhood_mono activePocket hkm (hCovered hi)

/-- Monotone-depth corollary of the k-hop rank bound: any deeper finite cover also certifies a valid
rank bound, with the deeper k-hop cardinality in the right-hand side. -/
theorem allosteric_srank_graph_bound_of_kHopNeighborhood_of_le
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k m : ℕ) (hkm : k ≤ m)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (G.kHopNeighborhood activePocket m).card + 3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound_of_kHopNeighborhood prob hStrictAll hBounded G activePocket m
    (relevantAtomPositions_subset_kHopNeighborhood_of_le prob G activePocket hkm hCovered)

/-- Shell-budget form of the k-hop graph-local theorem. Each expansion layer can be assigned its own finite
budget, and the total docking rank is bounded by the initial pocket plus the sum of those layer budgets. -/
theorem allosteric_srank_graph_bound_of_kHopShellBounds
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (shellBound : ℕ → ℕ)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤ shellBound m) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * activePocket.card + 3 * Finset.sum (Finset.range k) shellBound +
        3 * prob.ligand.numAtoms := by
  have hBase := allosteric_srank_graph_bound_of_kHopNeighborhood prob hStrictAll hBounded
    G activePocket k hCovered
  have hCard := G.kHopNeighborhood_card_le_of_shellBounds activePocket shellBound k hShell
  omega

/-- Distance-decay profile form of the k-hop allosteric theorem: any explicit
distance-indexed shell envelope yields a finite structural-rank bound. -/
theorem allosteric_srank_bound_of_distanceDecayProfile
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (distanceProfile : ℕ → ℕ)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤ distanceProfile m) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * activePocket.card + 3 * Finset.sum (Finset.range k) distanceProfile +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound_of_kHopShellBounds prob hStrictAll hBounded
    G activePocket k distanceProfile hCovered hShell

/-- Explicit exponential-envelope specialization of the distance-profile theorem:
if each shell is bounded by the ceiling of an exponential decay envelope, then
the docking structural rank is bounded by the corresponding finite shell sum. -/
theorem allosteric_srank_bound_of_exponentialDecayProfile
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (C α : ℝ)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤
        ProteinMechanicalGraph.allostericExponentialDistanceProfile C α m) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * activePocket.card +
        3 * Finset.sum (Finset.range k)
          (ProteinMechanicalGraph.allostericExponentialDistanceProfile C α) +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_bound_of_distanceDecayProfile prob hStrictAll hBounded
    G activePocket k
    (ProteinMechanicalGraph.allostericExponentialDistanceProfile C α)
    hCovered hShell

/-- Explicit polynomial-envelope specialization of the distance-profile theorem:
if each shell is bounded by the ceiling of a polynomial decay envelope, then
the docking structural rank is bounded by the corresponding finite shell sum. -/
theorem allosteric_srank_bound_of_polynomialDecayProfile
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (C : ℝ) (p : ℕ)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤
        ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p m) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * activePocket.card +
        3 * Finset.sum (Finset.range k)
          (ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p) +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_bound_of_distanceDecayProfile prob hStrictAll hBounded
    G activePocket k
    (ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p)
    hCovered hShell

/-- Real-valued budget corollary for polynomial shell envelopes. If the reciprocal-power
sum on the first `k` shells is bounded by `Z`, then the polynomial distance-profile
rank theorem yields an explicit closed-form real bound with affine `k` term and
`C * Z` envelope term. For `p > 1`, one can instantiate `Z` using any finite
p-series upper bound. -/
theorem allosteric_srank_bound_of_polynomialDecayProfile_of_seriesBound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (C : ℝ) (p : ℕ)
    (hC : 0 ≤ C)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤
        ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p m)
    (Z : ℝ)
    (hSeries : Finset.sum (Finset.range k)
      (fun d : ℕ => (1 : ℝ) / (((d : ℝ) + 1) ^ p)) ≤ Z) :
    ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
      3 * (activePocket.card : ℝ) + 3 * (k : ℝ) +
        3 * C * Z + 3 * (prob.ligand.numAtoms : ℝ) := by
  let profile := ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p
  have hNat := allosteric_srank_bound_of_polynomialDecayProfile prob hStrictAll hBounded
    G activePocket k C p hCovered hShell
  have hNatR :
      ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
      (3 * activePocket.card + 3 * Finset.sum (Finset.range k) profile +
        3 * prob.ligand.numAtoms : ℕ) := by
    exact_mod_cast hNat
  have hBase :
      ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
      3 * (activePocket.card : ℝ) +
        3 * ((Finset.sum (Finset.range k) profile : ℕ) : ℝ) +
        3 * (prob.ligand.numAtoms : ℝ) := by
    simpa [Nat.cast_add, Nat.cast_mul] using hNatR
  have hTerm : ∀ d : ℕ,
      ((profile d : ℕ) : ℝ) ≤ C / (((d : ℝ) + 1) ^ p) + 1 := by
    intro d
    have hDenPos : 0 < (((d : ℝ) + 1) ^ p) := by positivity
    have hArgNonneg : 0 ≤ C / (((d : ℝ) + 1) ^ p) :=
      div_nonneg hC hDenPos.le
    have hCeil : ((profile d : ℕ) : ℝ) < C / (((d : ℝ) + 1) ^ p) + 1 := by
      simpa [profile, ProteinMechanicalGraph.allostericPolynomialDistanceProfile,
        ProteinMechanicalGraph.allostericPolynomialShellEnvelope] using
        (Nat.ceil_lt_add_one hArgNonneg)
    exact hCeil.le
  have hProfileSum :
      ((Finset.sum (Finset.range k) profile : ℕ) : ℝ) ≤
        Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p) + 1) := by
    calc
      ((Finset.sum (Finset.range k) profile : ℕ) : ℝ)
          = Finset.sum (Finset.range k) (fun d : ℕ => ((profile d : ℕ) : ℝ)) := by
              simp
      _ ≤ Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p) + 1) := by
            exact Finset.sum_le_sum (fun d _ => hTerm d)
  have hScaledSeries :
      Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p)) ≤ C * Z := by
    have hMulSeries :
        Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p)) =
          C * Finset.sum (Finset.range k) (fun d : ℕ => (1 : ℝ) / (((d : ℝ) + 1) ^ p)) := by
      calc
        Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p))
            = Finset.sum (Finset.range k)
                (fun d : ℕ => C * ((1 : ℝ) / (((d : ℝ) + 1) ^ p))) := by
                  simp [div_eq_mul_inv]
        _ = C * Finset.sum (Finset.range k) (fun d : ℕ => (1 : ℝ) / (((d : ℝ) + 1) ^ p)) := by
              simpa using
                (Finset.mul_sum (s := Finset.range k)
                  (f := fun d : ℕ => (1 : ℝ) / (((d : ℝ) + 1) ^ p))
                  (a := C)).symm
    rw [hMulSeries]
    exact mul_le_mul_of_nonneg_left hSeries hC
  have hBudget :
      ((Finset.sum (Finset.range k) profile : ℕ) : ℝ) ≤ C * Z + k := by
    have hOneSum :
        Finset.sum (Finset.range k) (fun _ : ℕ => (1 : ℝ)) = k := by
      simp
    have hCombined :
        Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p) + 1)
          = Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p) ) + k := by
      calc
        Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p) + 1)
            = Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p)) +
                Finset.sum (Finset.range k) (fun _ : ℕ => (1 : ℝ)) := by
                    simp [Finset.sum_add_distrib]
        _ = Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p)) + k := by
              rw [hOneSum]
    have hTmp :
        ((Finset.sum (Finset.range k) profile : ℕ) : ℝ) ≤
          Finset.sum (Finset.range k) (fun d : ℕ => C / (((d : ℝ) + 1) ^ p)) + k := by
      exact le_trans hProfileSum (by simpa [hCombined])
    exact le_trans hTmp (by gcongr)
  have hScaledProfile :
      3 * ((Finset.sum (Finset.range k) profile : ℕ) : ℝ) ≤ 3 * (C * Z + k) := by
    exact mul_le_mul_of_nonneg_left hBudget (by norm_num)
  have hConclude :
      ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
      3 * (activePocket.card : ℝ) +
        3 * (C * Z + k) +
        3 * (prob.ligand.numAtoms : ℝ) := by
    linarith [hBase, hScaledProfile]
  have hRearrange :
      3 * (activePocket.card : ℝ) + 3 * (C * Z + k) + 3 * (prob.ligand.numAtoms : ℝ) =
        3 * (activePocket.card : ℝ) + 3 * (k : ℝ) +
          3 * C * Z + 3 * (prob.ligand.numAtoms : ℝ) := by ring
  simpa [hRearrange] using hConclude

/-- Elementary finite reciprocal-power upper bound for integer exponents `p ≥ 2`:
the first `k` terms satisfy
`∑_{d<k} 1/(d+1)^p ≤ 1 + k/4`.
This gives an explicit no-parameter series budget for polynomial allosteric profiles. -/
theorem reciprocalPowerSeriesBound_of_two_le
    (k p : ℕ) (hp : 2 ≤ p) :
    Finset.sum (Finset.range k) (fun d : ℕ => (1 : ℝ) / (((d : ℝ) + 1) ^ p)) ≤
      1 + (k : ℝ) / 4 := by
  cases k with
  | zero =>
      simp
  | succ k =>
      have hMain :
          Finset.sum (Finset.range (k + 1)) (fun d : ℕ => (1 : ℝ) / (((d : ℝ) + 1) ^ p)) ≤
            1 + ((k + 1 : ℕ) : ℝ) / 4 := by
        induction k with
        | zero =>
            simp
        | succ k ih =>
            let f : ℕ → ℝ := fun d => (1 : ℝ) / (((d : ℝ) + 1) ^ p)
            have hTwoLeBase : (2 : ℝ) ≤ (((k + 1 : ℕ) : ℝ) + 1) := by
              have hOneLe : (1 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) := by
                exact_mod_cast Nat.succ_le_succ (Nat.zero_le k)
              linarith
            have hPowTwoLe : (2 : ℝ) ^ p ≤ ((((k + 1 : ℕ) : ℝ) + 1) ^ p) :=
              pow_le_pow_left₀ (by positivity) hTwoLeBase p
            have hFourLePowTwo : (4 : ℝ) ≤ (2 : ℝ) ^ p := by
              have hPowMono : (2 : ℝ) ^ 2 ≤ (2 : ℝ) ^ p :=
                pow_right_mono₀ (by norm_num : (1 : ℝ) ≤ 2) hp
              have hPowTwoEq : (2 : ℝ) ^ 2 = 4 := by norm_num
              linarith [hPowMono, hPowTwoEq]
            have hFourLeDen : (4 : ℝ) ≤ ((((k + 1 : ℕ) : ℝ) + 1) ^ p) :=
              le_trans hFourLePowTwo hPowTwoLe
            have hTermLe : f (k + 1) ≤ (1 : ℝ) / 4 := by
              exact one_div_le_one_div_of_le (by norm_num : (0 : ℝ) < 4) hFourLeDen
            calc
              Finset.sum (Finset.range ((k + 1) + 1)) f
                  = Finset.sum (Finset.range (k + 1)) f + f (k + 1) := by
                      simpa using (Finset.sum_range_succ (f := f) (k + 1))
              _ ≤ (1 + ((k + 1 : ℕ) : ℝ) / 4) + (1 / 4 : ℝ) := by
                    gcongr
              _ = 1 + ((k + 2 : ℕ) : ℝ) / 4 := by
                    have hCast : (((k + 2 : ℕ) : ℝ)) = ((k + 1 : ℕ) : ℝ) + 1 := by
                      exact_mod_cast (by omega : k + 2 = (k + 1) + 1)
                    nlinarith [hCast]
      simpa using hMain

/-- Explicit no-parameter polynomial `p`-series instantiation (`p ≥ 2`) of the
polynomial distance-profile rank theorem. -/
theorem allosteric_srank_bound_of_polynomialDecayProfile_of_two_le
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k : ℕ)
    (C : ℝ) (p : ℕ)
    (hp : 2 ≤ p)
    (hC : 0 ≤ C)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤
        ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p m) :
    ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
      3 * (activePocket.card : ℝ) + 3 * (k : ℝ) +
        3 * C * (1 + (k : ℝ) / 4) + 3 * (prob.ligand.numAtoms : ℝ) := by
  exact allosteric_srank_bound_of_polynomialDecayProfile_of_seriesBound
    prob hStrictAll hBounded G activePocket k C p hC hCovered hShell
    (1 + (k : ℝ) / 4)
    (reciprocalPowerSeriesBound_of_two_le k p hp)

/-- Uniform-shell corollary of the k-hop theorem. If every hop adds at most `K` new outer-shell atoms, then
the rank grows at most linearly in the hop depth. -/
theorem allosteric_srank_graph_bound_of_uniform_kHopShell
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (k K : ℕ)
    (hCovered : relevantAtomPositions prob ⊆ G.kHopNeighborhood activePocket k)
    (hShell : ∀ m < k,
      (G.outerContactShell (G.kHopNeighborhood activePocket m)).card ≤ K) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * activePocket.card + 3 * (k * K) + 3 * prob.ligand.numAtoms := by
  have hBase := allosteric_srank_graph_bound_of_kHopShellBounds prob hStrictAll hBounded
    G activePocket k (fun _ => K) hCovered hShell
  simpa using hBase

/-- Geometry-derived allosteric locality theorem: the protein mechanical graph and active pocket are
constructed directly from the reference protein geometry using a contact radius and a site radius. -/
theorem geometric_contact_allosteric_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).mechanicalNeighborhood
        (activePocketByRadius prob pocketRadius)) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 *
        ((proteinContactGraph prob contactRadius).mechanicalNeighborhood
          (activePocketByRadius prob pocketRadius)).card +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) hCovered

/-- Geometry-derived one-hop contact-shell theorem. The remaining allosteric assumption is reduced to a
concrete direct-contact cover around the radius-defined active pocket. -/
theorem geometric_contact_shell_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (hCovered : relevantAtomPositions prob ⊆
      activePocketByRadius prob pocketRadius ∪
        (proteinContactGraph prob contactRadius).outerContactShell
          (activePocketByRadius prob pocketRadius)) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (activePocketByRadius prob pocketRadius).card +
        3 *
          ((proteinContactGraph prob contactRadius).outerContactShell
            (activePocketByRadius prob pocketRadius)).card +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound_of_contactShell_cover prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) hCovered

/-- Geometry-derived k-hop contact theorem: the relevant protein atoms are allowed to propagate through a
bounded number of outer contact-shell expansions around the radius-defined site pocket. -/
theorem geometric_contact_kHop_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k : ℕ)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 *
        ((proteinContactGraph prob contactRadius).kHopNeighborhood
          (activePocketByRadius prob pocketRadius) k).card +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound_of_kHopNeighborhood prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) k hCovered

/-- Geometry-derived coverage transfer in hop depth. If a radius/contact model covers relevant atoms at depth
`k`, any deeper depth `m ≥ k` also covers them and yields the corresponding k-hop rank bound. -/
theorem geometric_contact_kHop_srank_bound_of_le
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k m : ℕ) (hkm : k ≤ m)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 *
        ((proteinContactGraph prob contactRadius).kHopNeighborhood
          (activePocketByRadius prob pocketRadius) m).card +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound_of_kHopNeighborhood_of_le prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) k m hkm hCovered

/-- Geometry-derived bridge from bounded-depth contact coverage to the full mechanical-neighborhood theorem. -/
theorem geometric_contact_kHop_cover_mechanicalNeighborhood_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k : ℕ)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 *
        ((proteinContactGraph prob contactRadius).mechanicalNeighborhood
          (activePocketByRadius prob pocketRadius)).card +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound_of_kHopCover_mechanicalNeighborhood prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) k hCovered

/-- Geometry-derived shell-budget theorem for bounded-depth contact propagation. -/
theorem geometric_contact_kHop_srank_bound_of_shellBounds
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k : ℕ)
    (shellBound : ℕ → ℕ)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k)
    (hShell : ∀ m < k,
      ((proteinContactGraph prob contactRadius).outerContactShell
        ((proteinContactGraph prob contactRadius).kHopNeighborhood
          (activePocketByRadius prob pocketRadius) m)).card ≤ shellBound m) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (activePocketByRadius prob pocketRadius).card +
        3 * Finset.sum (Finset.range k) shellBound + 3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound_of_kHopShellBounds prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) k shellBound hCovered hShell

/-- Geometry-derived exponential-distance specialization: if each of the first
`k` contact shells around the radius-defined active pocket is bounded by an
explicit exponential envelope, the docking structural rank obeys the
corresponding finite exponential shell-sum bound. -/
theorem geometric_contact_kHop_srank_bound_of_exponentialDecayProfile
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k : ℕ)
    (C α : ℝ)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k)
    (hShell : ∀ m < k,
      ((proteinContactGraph prob contactRadius).outerContactShell
        ((proteinContactGraph prob contactRadius).kHopNeighborhood
          (activePocketByRadius prob pocketRadius) m)).card ≤
            ProteinMechanicalGraph.allostericExponentialDistanceProfile C α m) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (activePocketByRadius prob pocketRadius).card +
        3 * Finset.sum (Finset.range k)
          (ProteinMechanicalGraph.allostericExponentialDistanceProfile C α) +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_bound_of_exponentialDecayProfile prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) k C α hCovered hShell

/-- Geometry-derived polynomial-distance specialization: if each of the first
`k` contact shells around the radius-defined active pocket is bounded by an
explicit polynomial envelope, the docking structural rank obeys the
corresponding finite polynomial shell-sum bound. -/
theorem geometric_contact_kHop_srank_bound_of_polynomialDecayProfile
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k : ℕ)
    (C : ℝ) (p : ℕ)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k)
    (hShell : ∀ m < k,
      ((proteinContactGraph prob contactRadius).outerContactShell
        ((proteinContactGraph prob contactRadius).kHopNeighborhood
          (activePocketByRadius prob pocketRadius) m)).card ≤
            ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p m) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (activePocketByRadius prob pocketRadius).card +
        3 * Finset.sum (Finset.range k)
          (ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p) +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_bound_of_polynomialDecayProfile prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) k C p hCovered hShell

/-- Geometry-contact specialization of the polynomial series-budget corollary. -/
theorem geometric_contact_kHop_srank_bound_of_polynomialDecayProfile_of_seriesBound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k : ℕ)
    (C : ℝ) (p : ℕ)
    (hC : 0 ≤ C)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k)
    (hShell : ∀ m < k,
      ((proteinContactGraph prob contactRadius).outerContactShell
        ((proteinContactGraph prob contactRadius).kHopNeighborhood
          (activePocketByRadius prob pocketRadius) m)).card ≤
            ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p m)
    (Z : ℝ)
    (hSeries : Finset.sum (Finset.range k)
      (fun d : ℕ => (1 : ℝ) / (((d : ℝ) + 1) ^ p)) ≤ Z) :
    ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
      3 * ((activePocketByRadius prob pocketRadius).card : ℝ) +
        3 * (k : ℝ) +
        3 * C * Z +
        3 * (prob.ligand.numAtoms : ℝ) := by
  exact allosteric_srank_bound_of_polynomialDecayProfile_of_seriesBound
    prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius)
    k C p hC hCovered hShell Z hSeries

/-- Explicit no-parameter (`p ≥ 2`) geometry-contact specialization of the
polynomial distance-profile series budget theorem. -/
theorem geometric_contact_kHop_srank_bound_of_polynomialDecayProfile_of_two_le
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k : ℕ)
    (C : ℝ) (p : ℕ)
    (hp : 2 ≤ p)
    (hC : 0 ≤ C)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k)
    (hShell : ∀ m < k,
      ((proteinContactGraph prob contactRadius).outerContactShell
        ((proteinContactGraph prob contactRadius).kHopNeighborhood
          (activePocketByRadius prob pocketRadius) m)).card ≤
            ProteinMechanicalGraph.allostericPolynomialDistanceProfile C p m) :
    ((@DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℕ) : ℝ) ≤
      3 * ((activePocketByRadius prob pocketRadius).card : ℝ) +
        3 * (k : ℝ) +
        3 * C * (1 + (k : ℝ) / 4) +
        3 * (prob.ligand.numAtoms : ℝ) := by
  exact allosteric_srank_bound_of_polynomialDecayProfile_of_two_le
    prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius)
    k C p hp hC hCovered hShell

/-- Uniform-shell geometry corollary: if each of the first `k` outer contact shells around the site pocket
contains at most `K` new atoms, the docking rank grows at most linearly in `k`. -/
theorem geometric_contact_uniform_kHop_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ)
    (k K : ℕ)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).kHopNeighborhood
        (activePocketByRadius prob pocketRadius) k)
    (hShell : ∀ m < k,
      ((proteinContactGraph prob contactRadius).outerContactShell
        ((proteinContactGraph prob contactRadius).kHopNeighborhood
          (activePocketByRadius prob pocketRadius) m)).card ≤ K) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (activePocketByRadius prob pocketRadius).card + 3 * (k * K) +
        3 * prob.ligand.numAtoms := by
  exact allosteric_srank_graph_bound_of_uniform_kHopShell prob hStrictAll hBounded
    (proteinContactGraph prob contactRadius)
    (activePocketByRadius prob pocketRadius) k K hCovered hShell

/-- Cardinality-parametrized form of the allosteric graph-locality theorem. Any future rigidity or geometry
engine only needs to upper-bound the mechanical neighborhood size to obtain the same structural-rank bound. -/
theorem allosteric_srank_graph_bound_of_neighborhood_card
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (G : ProteinMechanicalGraph prob)
    (activePocket : Finset (Fin prob.protein.numAtoms))
    (K : ℕ)
    (hCovered : relevantAtomPositions prob ⊆ G.mechanicalNeighborhood activePocket)
    (hCard : (G.mechanicalNeighborhood activePocket).card ≤ K) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * K + 3 * prob.ligand.numAtoms := by
  have hBase := allosteric_srank_graph_bound prob hStrictAll hBounded G activePocket hCovered
  omega

/-- Cardinality-parametrized form of the geometry-derived contact-graph theorem. This isolates the only
remaining external burden for a future geometry or rigidity engine: prove that the contact-neighborhood
of the active pocket has size at most `K`. -/
theorem geometric_contact_allosteric_srank_bound_of_neighborhood_card
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ) (K : ℕ)
    (hCovered : relevantAtomPositions prob ⊆
      (proteinContactGraph prob contactRadius).mechanicalNeighborhood
        (activePocketByRadius prob pocketRadius))
    (hCard :
      ((proteinContactGraph prob contactRadius).mechanicalNeighborhood
        (activePocketByRadius prob pocketRadius)).card ≤ K) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * K + 3 * prob.ligand.numAtoms := by
  have hBase := geometric_contact_allosteric_srank_bound prob hStrictAll hBounded
    contactRadius pocketRadius hCovered
  omega

/-- Cardinality-parametrized direct-contact-shell theorem. This packages the remaining geometric allostery
burden as a finite shell-size parameter around the radius-defined active pocket. -/
theorem geometric_contact_shell_srank_bound_of_shell_card
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ) (K : ℕ)
    (hCovered : relevantAtomPositions prob ⊆
      activePocketByRadius prob pocketRadius ∪
        (proteinContactGraph prob contactRadius).outerContactShell
          (activePocketByRadius prob pocketRadius))
    (hShell :
      ((proteinContactGraph prob contactRadius).outerContactShell
        (activePocketByRadius prob pocketRadius)).card ≤ K) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * (activePocketByRadius prob pocketRadius).card + 3 * K +
        3 * prob.ligand.numAtoms := by
  have hBase := geometric_contact_shell_srank_bound prob hStrictAll hBounded
    contactRadius pocketRadius hCovered
  omega

/-- Bounded direct-contact-shell regime: if both the radius-defined active pocket and its direct contact
neighborhood admit explicit cardinality budgets, then exact docking lies in the corresponding low-rank
regime. -/
theorem geometric_contact_shell_bounded_regime
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (contactRadius pocketRadius : ℝ) (P K : ℕ)
    (hCovered : relevantAtomPositions prob ⊆
      activePocketByRadius prob pocketRadius ∪
        (proteinContactGraph prob contactRadius).outerContactShell
          (activePocketByRadius prob pocketRadius))
    (hPocket : (activePocketByRadius prob pocketRadius).card ≤ P)
    (hShell :
      ((proteinContactGraph prob contactRadius).outerContactShell
        (activePocketByRadius prob pocketRadius)).card ≤ K) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * P + 3 * K + 3 * prob.ligand.numAtoms := by
  have hBase := geometric_contact_shell_srank_bound prob hStrictAll hBounded
    contactRadius pocketRadius hCovered
  omega

end Leverage
