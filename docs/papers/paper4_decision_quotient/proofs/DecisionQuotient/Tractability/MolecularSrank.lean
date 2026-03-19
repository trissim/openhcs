/-
  Paper 4: Decision-Relevant Uncertainty

  Tractability/MolecularSrank.lean - Structural Rank for Molecular Docking

  KEY RESULT: For molecular docking with finite cutoff radius,
  the structural rank is bounded by the number of atoms within the cutoff.

  This is the "Decision-Theoretic Backdoor" — the mathematical necessity
  of efficient docking simulations.

  ## Triviality Level
  NONTRIVIAL: This is the core applied theorem for MD docking.

  ## Dependencies
  - DecisionQuotient.Basic (DecisionProblem, srank definition)
  - DecisionQuotient.Sufficiency (isRelevant, isSufficient)
  - DecisionQuotient.Computation.ArrayDSL (MD operations)
  - Mathlib4: Analysis, measure theory

  ## Key Insight
  In molecular docking with cutoff r_c:
    - Atom i affects binding iff distance(atom_i, binding_site) < r_c
    - Therefore: srank(docking_problem) ≤ |{atoms within r_c}|
    - If binding pocket is small → srank is small → TRACTABLE
-/

import DecisionQuotient.Basic
import DecisionQuotient.Sufficiency
import DecisionQuotient.Tractability.StructuralRank
import DecisionQuotient.ThermodynamicLift
import Mathlib.Data.Real.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Tactic
import DecisionQuotient.Tractability.EpsilonUtilityGap

namespace DecisionQuotient
namespace Tractability
namespace MolecularSrank

open Classical

/-! ## 1. Molecular Types -/

/-- Atomic position in 3D space -/
structure Atom where
  index : ℕ
  position : ℝ × ℝ × ℝ  -- (x, y, z)
  charge : ℝ
  mass : ℝ

/-- Molecular structure: collection of atoms -/
structure Molecule where
  atoms : List Atom
  numAtoms : Nat
  size_eq : atoms.length = numAtoms

/-- Binding site location (e.g., active site of protein) -/
structure BindingSite where
  center : ℝ × ℝ × ℝ  -- (x, y, z)
  radius : ℝ            -- Binding pocket radius

/-! ## 2. MD as Decision Problem -/

/-- MD State: molecular configuration. Both protein and ligand are represented purely by atomic coordinates, natively supporting fully flexible ligands. -/
structure MDState where
  protein : List Atom
  ligand : List Atom

/-- MD Action: target ligand conformation and pose to evaluate -/
structure MDAction where
  ligand : List Atom

/-! ## 2.1 Helper Functions -/

/-- Project MD state to a specific atom's position -/
def atomStateProj (s : MDState) (atomIdx : Nat) : ℝ × ℝ × ℝ :=
  (s.protein.getD atomIdx {index := 0, position := (0,0,0), charge := 0, mass := 0}).position

/-- Perturb an atom's position by a small delta in one axis -/
def perturbAtomPosition (atoms : List Atom) (atomIdx axis : Nat) (delta : ℝ) : List Atom :=
  atoms.modify atomIdx (fun atom =>
    match axis with
    | 0 => {atom with position := (atom.position.1 + delta, atom.position.2.1, atom.position.2.2)}
    | 1 => {atom with position := (atom.position.1, atom.position.2.1 + delta, atom.position.2.2)}
    | 2 => {atom with position := (atom.position.1, atom.position.2.1, atom.position.2.2 + delta)}
    | _ => atom)

/-! ## 2.2 MD Problem Structure -/

/-- Molecular Binding Decision Problem -/
structure MDBindingProblem where
  protein : Molecule
  ligand : Molecule
  bindingSite : BindingSite
  cutoff : ℝ  -- Interaction cutoff radius (e.g., 10Å)
  utility : MDAction → MDState → ℝ  -- Binding affinity function

/-- Convert MD problem to DecisionProblem type -/
def MDBindingProblem.toDecisionProblem (prob : MDBindingProblem) :
    DecisionProblem MDAction MDState where
  utility := prob.utility

/-- Coordinate space for MD: 3 coords per protein atom + 3 coords per ligand atom -/
def numMDCoordinates (prob : MDBindingProblem) : Nat :=
  3 * prob.protein.numAtoms + 3 * prob.ligand.numAtoms

/-- Project MD state to a specific coordinate value -/
def mdProj (prob : MDBindingProblem) (s : MDState) (i : Fin (numMDCoordinates prob)) : ℝ :=
  let nP := 3 * prob.protein.numAtoms
  if h : i.val < nP then
    -- Protein atom coordinate
    let atomIdx := i.val / 3
    let axis := i.val % 3
    let atom := s.protein.getD atomIdx {index := 0, position := (0,0,0), charge := 0, mass := 0}
    match axis with
    | 0 => atom.position.1
    | 1 => atom.position.2.1
    | 2 => atom.position.2.2
    | _ => 0
  else
    -- Ligand atom coordinate
    let lIdx := (i.val - nP) / 3
    let axis := (i.val - nP) % 3
    let atom := s.ligand.getD lIdx {index := 0, position := (0,0,0), charge := 0, mass := 0}
    match axis with
    | 0 => atom.position.1
    | 1 => atom.position.2.1
    | 2 => atom.position.2.2
    | _ => 0

def mdCoordinateSpaceStruct (prob : MDBindingProblem) : CoordinateSpace MDState (numMDCoordinates prob) where
  Coord := fun _ => ℝ
  proj := mdProj prob

/-! ## 3. Distance and Relevance -/

/-- Euclidean distance between two 3D points -/
noncomputable def pointDistance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := p1
  let (x2, y2, z2) := p2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2 + (z1 - z2)^2)

/-- Distance from atom to binding site center -/
noncomputable def atomBindingDistance (atom : Atom) (site : BindingSite) : ℝ :=
  pointDistance atom.position site.center

/-- Atom is "within cutoff" of binding site -/
noncomputable def atomWithinCutoff (atom : Atom) (site : BindingSite) (cutoff : ℝ) : Prop :=
  atomBindingDistance atom site < cutoff

/-! ## Added Helpers for Proofs -/

def proteinAtom (prob : MDBindingProblem) (atomIdx : Nat)
    (hAtomInProtein : atomIdx < prob.protein.numAtoms) : Atom :=
  prob.protein.atoms.get ⟨atomIdx, by
    simpa [prob.protein.size_eq] using hAtomInProtein
  ⟩

def proteinCoordFin (prob : MDBindingProblem) (atomIdx : Nat)
    (hAtomInProtein : atomIdx < prob.protein.numAtoms) (axis : Fin 3) :
    Fin (numMDCoordinates prob) :=
  ⟨atomIdx * 3 + axis.1, by
    have h1 : atomIdx * 3 + axis.1 < 3 * prob.protein.numAtoms := by
      have hMul : atomIdx * 3 < prob.protein.numAtoms * 3 :=
        Nat.mul_lt_mul_of_pos_right hAtomInProtein (by decide)
      have hAxis : axis.1 < 3 := axis.2
      omega
    unfold numMDCoordinates
    omega
  ⟩

/--
  Using our rigorous `epsilon_margin_invariance` theorem, we can define the required physical bounds.
  Instead of assuming forces are exactly zero outside the cutoff, we guarantee the structural rank
  is invariant as long as the far-field forces (δ) sum to less than half the StrictUtilityGap.
-/
def OutsideCutoffApproximationBounded
    (prob : MDBindingProblem)
    [Fintype MDAction] : Prop :=
  ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
    (axis : Fin 3) (s s' : MDState) (a_star : MDAction),
    ∃ (δ : ℝ), 0 ≤ δ ∧
    ((∀ j : Fin (numMDCoordinates prob),
      j ≠ proteinCoordFin prob atomIdx hAtomInProtein axis →
      mdProj prob s j = mdProj prob s' j) →
    ¬ atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein)
        prob.bindingSite prob.cutoff →
    StrictOpt prob.toDecisionProblem a_star s →
    ((∀ a : MDAction, |prob.utility a s - prob.utility a s'| ≤ δ) ∧
    δ < (StrictUtilityGap prob.toDecisionProblem a_star s) / 2))

/--
  THEOREM: Decision-Theoretic Backdoor (Stage 9 Gate)
  Any atom outside the cutoff radius is formally irrelevant to the decision problem.
  Its movement mathematically cannot change the optimal binding pose.
 -/
theorem outside_cutoff_is_irrelevant
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (atomIdx : Nat)
    (hAtomInProtein : atomIdx < prob.protein.numAtoms)
    (axis : Fin 3)
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hOutside : ¬ atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite prob.cutoff) :
    @DecisionProblem.isIrrelevant MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem
      (proteinCoordFin prob atomIdx hAtomInProtein axis) := by
  intro s s' hSame
  rcases hStrictAll s with ⟨a_opt, hStrict⟩
  have h_bound_hyp := hBounded atomIdx hAtomInProtein axis s s' a_opt
  rcases h_bound_hyp with ⟨δ, hδ, h_impl⟩
  have h_conseq := h_impl hSame hOutside hStrict
  rcases h_conseq with ⟨hPerturb, hBoundLt⟩
  exact epsilon_margin_invariance prob.toDecisionProblem s s' a_opt δ hδ hStrict hPerturb hBoundLt

/-! ## 4. Relevance for MD -/

theorem md_relevant_only_if_within_cutoff
    (prob : MDBindingProblem)
    (atomIdx : Nat)
    (hAtomInProtein : atomIdx < prob.protein.numAtoms)
    (axis : Fin 3)
    [Fintype MDAction]
    (a_star : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem a_star s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob) :
    let coord := proteinCoordFin prob atomIdx hAtomInProtein axis
    @DecisionProblem.isRelevant MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem coord →
    atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite prob.cutoff := by
  intro coord hRel
  by_contra hOutside
  rcases hRel with ⟨s, s', hSame, hOptNe⟩
  -- Extract strict optimal action and delta explicitly for s
  have ⟨a_opt, hStrict⟩ : ∃ a, StrictOpt prob.toDecisionProblem a s := by
    rcases hStrictAll s with h1 | h2
    · exact ⟨a_star, h1⟩
    · exact h2
  
  have h_bound_hyp := hBounded atomIdx hAtomInProtein axis s s' a_opt
  rcases h_bound_hyp with ⟨δ, hδ, h_impl⟩
  -- The implication evaluates the structural bounds
  have h_conseq := h_impl hSame hOutside hStrict
  rcases h_conseq with ⟨hPerturb, hBoundLt⟩
  
  have hOptEq : prob.toDecisionProblem.Opt s = prob.toDecisionProblem.Opt s' := by
    exact epsilon_margin_invariance prob.toDecisionProblem s s' a_opt δ hδ hStrict hPerturb hBoundLt
  exact hOptNe hOptEq

/-! ## 5. Main Theorem: Bounded srank from Cutoff -/

/-- Set of relevant atoms for a binding problem, keyed by position -/
noncomputable def relevantAtomPositions (prob : MDBindingProblem) :
    Finset (Fin prob.protein.numAtoms) :=
  Finset.univ.filter (fun i =>
    atomWithinCutoff
      (proteinAtom prob i.1 i.2)
      prob.bindingSite prob.cutoff)

/-- Number of relevant atoms -/
noncomputable def numRelevantAtoms (prob : MDBindingProblem) : Nat :=
  (relevantAtomPositions prob).card

noncomputable def avgRelevantAtomsPerAtom (prob : MDBindingProblem) : ℝ :=
  if prob.protein.numAtoms = 0 then 0
  else (numRelevantAtoms prob : ℝ) / (prob.protein.numAtoms : ℝ)

def proteinRelevantCoordEmbedding (prob : MDBindingProblem) :
    (Fin prob.protein.numAtoms × Fin 3) ↪ Fin (numMDCoordinates prob) :=
{ toFun := fun x =>
    proteinCoordFin prob x.1.1 x.1.2 x.2
  inj' := by
    intro x y h
    rcases x with ⟨i,a⟩
    rcases y with ⟨j,b⟩
    simp [proteinCoordFin] at h
    have hm1 : (i.1 * 3 + a.1) / 3 = i.1 := by
      have : a.1 < 3 := a.2
      omega
    have hm2 : (j.1 * 3 + b.1) / 3 = j.1 := by
      have : b.1 < 3 := b.2
      omega
    have heqi : i.1 = j.1 := by
      rw [←hm1, ←hm2, h]
    have heqa : a.1 = b.1 := by
      omega
    ext
    · exact heqi
    · exact heqa }

noncomputable def potentialProteinRelevantCoords (prob : MDBindingProblem) :
    Finset (Fin (numMDCoordinates prob)) :=
  ((relevantAtomPositions prob) ×ˢ (Finset.univ : Finset (Fin 3))).map
    (proteinRelevantCoordEmbedding prob)

def ligandCoordFin (prob : MDBindingProblem) (atomIdx : Fin prob.ligand.numAtoms) (axis : Fin 3) :
    Fin (numMDCoordinates prob) :=
  ⟨3 * prob.protein.numAtoms + atomIdx.1 * 3 + axis.1, by
    have h1 : atomIdx.1 * 3 + axis.1 < 3 * prob.ligand.numAtoms := by
      have hMul : atomIdx.1 * 3 < prob.ligand.numAtoms * 3 :=
        Nat.mul_lt_mul_of_pos_right atomIdx.2 (by decide)
      have hAxis : axis.1 < 3 := axis.2
      omega
    unfold numMDCoordinates
    omega
  ⟩

def ligandCoordEmbedding (prob : MDBindingProblem) :
    (Fin prob.ligand.numAtoms × Fin 3) ↪ Fin (numMDCoordinates prob) :=
{ toFun := fun x => ligandCoordFin prob x.1 x.2
  inj' := by
    intro x y h
    rcases x with ⟨i,a⟩
    rcases y with ⟨j,b⟩
    simp [ligandCoordFin] at h
    have h2 : i.1 * 3 + a.1 = j.1 * 3 + b.1 := by omega
    have hm1 : (i.1 * 3 + a.1) / 3 = i.1 := by
      have : a.1 < 3 := a.2
      omega
    have hm2 : (j.1 * 3 + b.1) / 3 = j.1 := by
      have : b.1 < 3 := b.2
      omega
    have heqi : i.1 = j.1 := by
      rw [←hm1, ←hm2, h2]
    have heqa : a.1 = b.1 := by omega
    ext
    · exact heqi
    · exact heqa }

noncomputable def potentialLigandCoords (prob : MDBindingProblem) :
    Finset (Fin (numMDCoordinates prob)) :=
  ((Finset.univ : Finset (Fin prob.ligand.numAtoms)) ×ˢ (Finset.univ : Finset (Fin 3))).map
    (ligandCoordEmbedding prob)

noncomputable def potentialRelevantCoords (prob : MDBindingProblem) :
    Finset (Fin (numMDCoordinates prob)) :=
  potentialProteinRelevantCoords prob ∪ potentialLigandCoords prob

/-- THEOREM: Structural Rank Bound for Molecular Docking -/
theorem md_srank_bound
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  classical
  rw [DecisionProblem.srank_eq_relevant_card]
  let R : Finset (Fin (numMDCoordinates prob)) :=
    Finset.univ.filter (fun i =>
      @DecisionProblem.isRelevant MDAction MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) prob.toDecisionProblem i)

  have hsub : R ⊆ potentialRelevantCoords prob := by
    intro i hi
    have hrel : @DecisionProblem.isRelevant MDAction MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob) prob.toDecisionProblem i := by
      simpa [R] using hi
    by_cases hProtein : i.1 < 3 * prob.protein.numAtoms
    · have hIdx : i.1 / 3 < prob.protein.numAtoms := by omega
      have hAxis : i.1 % 3 < 3 := Nat.mod_lt _ (by decide)
      have heq : i = proteinCoordFin prob (i.1 / 3) hIdx ⟨i.1 % 3, hAxis⟩ := by
        ext; simp [proteinCoordFin]; omega
      rw [heq] at hrel
      have hWithin : atomWithinCutoff
          (proteinAtom prob (i.1 / 3) hIdx) prob.bindingSite prob.cutoff := by
        -- We instantiate md_relevant_only_if_within_cutoff for any strict optimal function
        -- First, construct a trivial dummy state to extract a valid MDAction signature
        let dummy_state : MDState := { protein := [], ligand := [] }
        let a_dummy := (hStrictAll dummy_state).choose
        have hStrictMod (state : MDState) : StrictOpt prob.toDecisionProblem a_dummy state ∨ ∃ a, StrictOpt prob.toDecisionProblem a state := by
          right
          exact hStrictAll state
        exact md_relevant_only_if_within_cutoff prob (i.1 / 3) hIdx ⟨i.1 % 3, hAxis⟩ a_dummy hStrictMod hBounded hrel
      apply Finset.subset_union_left
      rw [potentialProteinRelevantCoords, Finset.mem_map]
      use ⟨⟨i.1 / 3, hIdx⟩, ⟨i.1 % 3, hAxis⟩⟩
      constructor
      · rw [Finset.mem_product]
        constructor
        · rw [relevantAtomPositions, Finset.mem_filter]
          exact ⟨Finset.mem_univ _, hWithin⟩
        · exact Finset.mem_univ _
      · change proteinCoordFin prob (i.1 / 3) hIdx ⟨i.1 % 3, hAxis⟩ = i
        exact heq.symm
    · apply Finset.subset_union_right
      rw [potentialLigandCoords, Finset.mem_map]
      let lIdx := (i.1 - 3 * prob.protein.numAtoms) / 3
      let axis := (i.1 - 3 * prob.protein.numAtoms) % 3
      have hl : lIdx < prob.ligand.numAtoms := by
        have : i.1 < numMDCoordinates prob := i.2
        unfold numMDCoordinates at this
        omega
      have haxis : axis < 3 := Nat.mod_lt _ (by decide)
      use ⟨⟨lIdx, hl⟩, ⟨axis, haxis⟩⟩
      constructor
      · rw [Finset.mem_product]
        exact ⟨Finset.mem_univ _, Finset.mem_univ _⟩
      · change ligandCoordFin prob ⟨lIdx, hl⟩ ⟨axis, haxis⟩ = i
        ext; simp [ligandCoordFin]; omega

  have hcard : R.card ≤ (potentialRelevantCoords prob).card :=
    Finset.card_le_card hsub

  have hProteinCard :
      (potentialProteinRelevantCoords prob).card = 3 * numRelevantAtoms prob := by
    rw [potentialProteinRelevantCoords, Finset.card_map, Finset.card_product, Finset.card_univ]
    simp [numRelevantAtoms]
    ring

  have hLigandCard :
      (potentialLigandCoords prob).card = 3 * prob.ligand.numAtoms := by
    rw [potentialLigandCoords, Finset.card_map, Finset.card_product, Finset.card_univ, Finset.card_univ]
    simp
    ring

  have hUnion :
      (potentialRelevantCoords prob).card ≤
        (potentialProteinRelevantCoords prob).card + (potentialLigandCoords prob).card :=
    Finset.card_union_le _ _

  have hSum : (potentialProteinRelevantCoords prob).card + (potentialLigandCoords prob).card = 3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
    rw [hProteinCard, hLigandCard]

  calc R.card ≤ (potentialRelevantCoords prob).card := hcard
    _ ≤ (potentialProteinRelevantCoords prob).card + (potentialLigandCoords prob).card := hUnion
    _ = 3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := hSum

/-- COROLLARY: Small binding pocket implies low srank. -/
theorem small_pocket_low_srank
    (prob : MDBindingProblem)
    [Fintype MDAction]
    (K L : Nat)
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hBound : numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤ 3 * K + 3 * L := by
  have h1 := md_srank_bound prob hStrictAll hBounded
  omega

/-- COROLLARY: Tractability threshold. -/
theorem docking_small_pocket_bound
    (prob : MDBindingProblem) (K L: Nat)
    [Fintype MDAction]
    (hStrictAll : ∀ s, ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hPocket : numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem ≤ 3 * K + 3 * L := by
  exact small_pocket_low_srank prob K L hStrictAll hBounded hPocket hLigand

/-! ## 6. Connection to Decision Quotient Framework -/

/-- THEOREM: MD structural rank matches Decision Quotient definition. -/
theorem md_srank_matches_dq_definition (prob : MDBindingProblem) :
    @DecisionProblem.srank MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem =
      Finset.card (Finset.univ.filter (@DecisionProblem.isRelevant MDAction MDState
        (numMDCoordinates prob) (mdCoordinateSpaceStruct prob) (prob.toDecisionProblem))) := by
  rfl

/-! ## 7. Thermodynamic Connection -/

/-- THEOREM: Thermodynamic cost of MD decision is bounded by srank. -/
theorem md_thermodynamic_lower_bound
    (prob : MDBindingProblem)
    [Fintype MDState] [Nonempty MDState] [DecidableEq (Set MDAction)]
    (kB T E : ℝ)
    (hkB : 0 < kB)
    (hT : 0 < T)
    (hE : E ≥ (@DecisionProblem.srank MDAction MDState (numMDCoordinates prob) (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℝ) * (kB * T * Real.log 2))
    (hEntropy : prob.toDecisionProblem.quotientEntropy ≤ (@DecisionProblem.srank MDAction MDState (numMDCoordinates prob) (mdCoordinateSpaceStruct prob) prob.toDecisionProblem : ℝ)) :
    E ≥ kB * T * Real.log (prob.toDecisionProblem.numOptClasses : ℝ) := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  exact ThermodynamicLift.energy_ge_kbt_nat_entropy prob.toDecisionProblem kB T hkB hT E hE hEntropy

/-! ## 8. Algorithmic Implications -/

end MolecularSrank
end Tractability
end DecisionQuotient
