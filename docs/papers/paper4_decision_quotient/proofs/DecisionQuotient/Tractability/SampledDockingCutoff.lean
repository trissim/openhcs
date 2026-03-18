/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/SampledDockingCutoff.lean

  Transport cutoff locality from the ambient molecular docking problem to a
  sampled restricted action family, under the explicit hypothesis that the
  sampled family still captures at least one ambient optimum at every state.
-/
import DecisionQuotient.Tractability.MolecularSrank
import DecisionQuotient.Tractability.SampledDocking
import DecisionQuotient.Information

namespace DecisionQuotient
namespace Tractability
namespace SampledDockingCutoff

open MolecularSrank
open SampledDocking

universe u v

noncomputable instance : DecidableEq Atom := Classical.decEq _
noncomputable instance : DecidableEq MDAction := Classical.decEq _

/-- If the sampled support captures at least one ambient optimal action at every
    state, relevance in the restricted sampled problem implies relevance in the
    ambient problem. -/
theorem restricted_relevance_implies_ambient_relevance
    {A : Type u} {S : Type v} {n : ℕ}
    [DecidableEq A] [CoordinateSpace S n]
    (dp : DecisionProblem A S)
    (F : SampledActionFamily A)
    (coord : Fin n)
    (hCapture : ∀ s, ∃ a : SupportedAction F, a.1 ∈ dp.Opt s)
    (hRel : @DecisionProblem.isRelevant (SupportedAction F) S n inferInstance
      (restrictedDecisionProblem dp F) coord) :
    @DecisionProblem.isRelevant A S n inferInstance dp coord := by
  rcases hRel with ⟨s, s', hSame, hOptNe⟩
  refine ⟨s, s', hSame, ?_⟩
  intro hAmbientEq
  have hRestrS :=
    restricted_opt_eq_ambient_slice_of_exists_global_sampled_opt dp F s (hCapture s)
  have hRestrS' :=
    restricted_opt_eq_ambient_slice_of_exists_global_sampled_opt dp F s' (hCapture s')
  have hRestrEq : (restrictedDecisionProblem dp F).Opt s =
      (restrictedDecisionProblem dp F).Opt s' := by
    rw [hRestrS, hRestrS']
    ext a
    change a.1 ∈ dp.Opt s ↔ a.1 ∈ dp.Opt s'
    simpa [hAmbientEq]
  exact hOptNe hRestrEq

/-- Cutoff locality transport for sampled docking: if the sampled action family
    captures at least one ambient optimum at every state, then any coordinate
    relevant in the sampled restricted problem must still lie within the cutoff.
-/
theorem sampled_md_relevant_only_if_within_cutoff
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    (atomIdx : Nat)
    (hAtomInProtein : atomIdx < prob.protein.numAtoms)
    (axis : Fin 3)
    [Fintype MDAction]
    (aStar : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s) :
    let coord := proteinCoordFin prob atomIdx hAtomInProtein axis
    @DecisionProblem.isRelevant (SupportedAction samples) MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (restrictedDecisionProblem prob.toDecisionProblem samples) coord →
    atomWithinCutoff (proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite prob.cutoff := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  intro coord hRel
  have hAmbientRel : @DecisionProblem.isRelevant MDAction MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob) prob.toDecisionProblem coord :=
    restricted_relevance_implies_ambient_relevance
      prob.toDecisionProblem samples coord hCapture hRel
  exact md_relevant_only_if_within_cutoff
    prob atomIdx hAtomInProtein axis aStar hStrictAll hBounded hAmbientRel

/-- The sampled retained coordinate set: protein coordinates whose atoms lie
    within cutoff, together with all ligand coordinates. -/
noncomputable def sampledRetainedCoords (prob : MDBindingProblem) :
    Finset (Fin (numMDCoordinates prob)) :=
  potentialRelevantCoords prob

/-- Coordinates outside the sampled retained set. -/
noncomputable def sampledOutsideCutoffCoords (prob : MDBindingProblem) :
    Finset (Fin (numMDCoordinates prob)) :=
  Finset.univ \ sampledRetainedCoords prob

/-- Any sampled restricted coordinate outside the retained set is irrelevant. -/
theorem sampled_outsideCutoffCoord_irrelevant
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    (i : Fin (numMDCoordinates prob))
    [Fintype MDAction]
    (aStar : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hOut : i ∈ sampledOutsideCutoffCoords prob) :
    @DecisionProblem.isIrrelevant (SupportedAction samples) MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (restrictedDecisionProblem prob.toDecisionProblem samples) i := by
  classical
  rw [@DecisionProblem.irrelevant_iff_not_relevant (SupportedAction samples) MDState
    (numMDCoordinates prob) (mdCoordinateSpaceStruct prob)
    (restrictedDecisionProblem prob.toDecisionProblem samples) i]
  intro hRel
  have hNotRetained : i ∉ sampledRetainedCoords prob := by
    simpa [sampledOutsideCutoffCoords, sampledRetainedCoords, Finset.mem_sdiff] using hOut
  by_cases hProtein : i.1 < 3 * prob.protein.numAtoms
  · have hIdx : i.1 / 3 < prob.protein.numAtoms := by omega
    have hAxis : i.1 % 3 < 3 := Nat.mod_lt _ (by decide)
    have heq : i = proteinCoordFin prob (i.1 / 3) hIdx ⟨i.1 % 3, hAxis⟩ := by
      ext
      simp [proteinCoordFin]
      omega
    rw [heq] at hRel
    have hWithin : atomWithinCutoff
        (proteinAtom prob (i.1 / 3) hIdx) prob.bindingSite prob.cutoff := by
      exact sampled_md_relevant_only_if_within_cutoff
        prob samples (i.1 / 3) hIdx ⟨i.1 % 3, hAxis⟩ aStar hStrictAll hBounded hCapture hRel
    have hInRetained : i ∈ sampledRetainedCoords prob := by
      rw [sampledRetainedCoords]
      apply Finset.mem_union.mpr
      left
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
    exact hNotRetained hInRetained
  · have hLigand : i ∈ potentialLigandCoords prob := by
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
        ext
        simp [ligandCoordFin]
        omega
    have hInRetained : i ∈ sampledRetainedCoords prob := by
      rw [sampledRetainedCoords]
      exact Finset.mem_union.mpr (Or.inr hLigand)
    exact hNotRetained hInRetained

/-- Structural-rank bound for the sampled restricted docking problem. If the
    sampled support captures an ambient optimum at every state, then the sampled
    restricted problem inherits the same cutoff-based srank upper bound as the
    ambient molecular docking problem. -/
theorem sampled_md_srank_bound
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [Fintype MDAction]
    (aStar : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s) :
    @DecisionProblem.srank (SupportedAction samples) MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (restrictedDecisionProblem prob.toDecisionProblem samples) ≤
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
  letI : CoordinateSpace MDState (numMDCoordinates prob) := mdCoordinateSpaceStruct prob
  classical
  rw [DecisionProblem.srank_eq_relevant_card]
  let R : Finset (Fin (numMDCoordinates prob)) :=
    Finset.univ.filter (fun i =>
      @DecisionProblem.isRelevant (SupportedAction samples) MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (restrictedDecisionProblem prob.toDecisionProblem samples) i)

  have hsub : R ⊆ potentialRelevantCoords prob := by
    intro i hi
    have hrel : @DecisionProblem.isRelevant (SupportedAction samples) MDState (numMDCoordinates prob)
        (mdCoordinateSpaceStruct prob)
        (restrictedDecisionProblem prob.toDecisionProblem samples) i := by
      simpa [R] using hi
    by_cases hProtein : i.1 < 3 * prob.protein.numAtoms
    · have hIdx : i.1 / 3 < prob.protein.numAtoms := by omega
      have hAxis : i.1 % 3 < 3 := Nat.mod_lt _ (by decide)
      have heq : i = proteinCoordFin prob (i.1 / 3) hIdx ⟨i.1 % 3, hAxis⟩ := by
        ext
        simp [proteinCoordFin]
        omega
      rw [heq] at hrel
      have hWithin : atomWithinCutoff
          (proteinAtom prob (i.1 / 3) hIdx) prob.bindingSite prob.cutoff := by
        exact sampled_md_relevant_only_if_within_cutoff
          prob samples (i.1 / 3) hIdx ⟨i.1 % 3, hAxis⟩ aStar hStrictAll hBounded hCapture hrel
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
        ext
        simp [ligandCoordFin]
        omega

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

  have hSum : (potentialProteinRelevantCoords prob).card + (potentialLigandCoords prob).card =
      3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := by
    rw [hProteinCard, hLigandCard]

  calc R.card ≤ (potentialRelevantCoords prob).card := hcard
    _ ≤ (potentialProteinRelevantCoords prob).card + (potentialLigandCoords prob).card := hUnion
    _ = 3 * numRelevantAtoms prob + 3 * prob.ligand.numAtoms := hSum

/-- Small-pocket corollary for the sampled restricted docking problem. -/
theorem sampled_small_pocket_low_srank
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    (K L : Nat)
    [Fintype MDAction]
    (aStar : MDAction)
    (hStrictAll : ∀ s, StrictOpt prob.toDecisionProblem aStar s ∨ ∃ a, StrictOpt prob.toDecisionProblem a s)
    (hBounded : OutsideCutoffApproximationBounded prob)
    (hCapture : ∀ s, ∃ a : SupportedAction samples, a.1 ∈ prob.toDecisionProblem.Opt s)
    (hPocket : numRelevantAtoms prob ≤ K)
    (hLigand : prob.ligand.numAtoms ≤ L) :
    @DecisionProblem.srank (SupportedAction samples) MDState (numMDCoordinates prob)
      (mdCoordinateSpaceStruct prob)
      (restrictedDecisionProblem prob.toDecisionProblem samples) ≤ 3 * K + 3 * L := by
  have hBase := sampled_md_srank_bound prob samples aStar hStrictAll hBounded hCapture
  omega

/-- Generic sufficiency bridge: once one proves that every relevant coordinate
    of the sampled restricted problem lies in `potentialRelevantCoords`, the
    retained coordinate set is sufficient provided the sampled state
    representation supports the usual `ProductSpace` machinery and the full
    coordinate projection is injective. This isolates the remaining work to a
    compatibility theorem between the chosen `ProductSpace` instance and the
    cutoff-locality coordinate space. -/
theorem sampled_potentialRelevantCoords_sufficient_of_relevance_subset
    (prob : MDBindingProblem)
    (samples : SampledActionFamily MDAction)
    [ProductSpace MDState (numMDCoordinates prob)]
    [DecidableEq (Fin (numMDCoordinates prob))]
    (hRelevantSubset : ∀ i,
      (restrictedDecisionProblem prob.toDecisionProblem samples).isRelevant i →
      i ∈ potentialRelevantCoords prob)
    (hinj : ∀ s s' : MDState,
      (∀ i : Fin (numMDCoordinates prob), CoordinateSpace.proj s i = CoordinateSpace.proj s' i) → s = s') :
    (restrictedDecisionProblem prob.toDecisionProblem samples).isSufficient
      (potentialRelevantCoords prob) := by
  classical
  let dp := restrictedDecisionProblem prob.toDecisionProblem samples
  let R : Finset (Fin (numMDCoordinates prob)) := Finset.univ.filter dp.isRelevant
  have hRelevantSuff : dp.isSufficient R := by
    -- The `relevantSet_isSufficient` lemma expects a `ProductSpace` instance
    -- for the ambient type and an injectivity hypothesis. We provide those via
    -- the `hinj` argument and the `[ProductSpace MDState _]` instance above.
    simpa [R] using relevantSet_isSufficient dp hinj
  apply dp.sufficient_superset R (potentialRelevantCoords prob) hRelevantSuff
  intro i hi
  have hiR : i ∈ R := hi
  have hiR' : i ∈ Finset.univ ∧ dp.isRelevant i := by
    simpa [R, Finset.mem_filter] using hiR
  exact hRelevantSubset i hiR'.2
end SampledDockingCutoff
end Tractability
end DecisionQuotient
