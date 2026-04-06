import Mathlib.Combinatorics.Matroid.IndepAxioms
import Mathlib.Data.Finset.Sum
import Mathlib.LinearAlgebra.Dual.Lemmas
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.Dimension.OrzechProperty
import Mathlib.LinearAlgebra.LinearIndependent.Lemmas
import Mathlib.LinearAlgebra.Pi
import Mathlib.LinearAlgebra.Prod

open Set Submodule LinearMap

namespace Ssot
namespace FactMatroid

variable (K : Type*) [Field K]
variable (ι : Type*) [Fintype ι] [DecidableEq ι]

abbrev Ambient := ι → K
abbrev DirectionSpace := Submodule K (Ambient K ι)

structure AffineFamily where
  origin : Ambient K ι
  directions : DirectionSpace K ι

namespace AffineFamily

variable {K ι} [Field K] [Fintype ι] [DecidableEq ι]

abbrev stateSet (A : AffineFamily K ι) : Set (Ambient K ι) :=
  {x | x - A.origin ∈ A.directions}

@[simp] theorem origin_mem_stateSet (A : AffineFamily K ι) : A.origin ∈ A.stateSet := by
  simp [stateSet]

theorem sub_mem_directions_of_mem_stateSet (A : AffineFamily K ι) {x y : Ambient K ι}
    (hx : x ∈ A.stateSet) (hy : y ∈ A.stateSet) : x - y ∈ A.directions := by
  have hx' : x - A.origin ∈ A.directions := by simpa [stateSet] using hx
  have hy' : y - A.origin ∈ A.directions := by simpa [stateSet] using hy
  have hsub : (x - A.origin) - (y - A.origin) ∈ A.directions := A.directions.sub_mem hx' hy'
  simpa [Pi.sub_apply, sub_eq_add_neg, add_comm, add_left_comm, add_assoc] using hsub

end AffineFamily

noncomputable def coordFun {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (i : ι) : Module.Dual K V where
  toFun := fun x => x.1 i
  map_add' _ _ := rfl
  map_smul' _ _ := rfl

@[simp] theorem coordFun_apply {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (i : ι) (x : V) : coordFun V i x = x.1 i := rfl

abbrev factSpan {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Set ι) : Submodule K (Module.Dual K V) :=
  Submodule.span K ((coordFun V) '' S)

abbrev factSpanFinset {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) : Submodule K (Module.Dual K V) :=
  factSpan V (S : Set ι)

noncomputable abbrev factRankFinset {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) : Nat :=
  Module.finrank K (factSpanFinset V S)

noncomputable def coordProjection {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) : V →ₗ[K] (↑S → K) where
  toFun := fun x i => x.1 i.1
  map_add' _ _ := by ext i; rfl
  map_smul' _ _ := by ext i; rfl

@[simp] theorem coordProjection_apply {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) (x : V) (i : ↑S) :
    coordProjection V S x i = x.1 i.1 := rfl

theorem factRankFinset_le_card {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) :
    factRankFinset V S ≤ S.card := by
  classical
  let simg : Finset (Module.Dual K V) := S.image (coordFun V)
  have hsimg_eq :
      (simg : Set (Module.Dual K V)) = (coordFun V) '' (S : Set ι) := by
    ext φ
    simp [simg]
  calc
    factRankFinset V S = Module.finrank K (Submodule.span K (simg : Set (Module.Dual K V))) := by
      rw [show factRankFinset V S = Module.finrank K (Submodule.span K ((coordFun V) '' (S : Set ι))) by rfl]
      rw [hsimg_eq]
    _ ≤ simg.card := by
      simpa using (finrank_span_finset_le_card (R := K) simg)
    _ ≤ S.card := Finset.card_image_le

theorem coordProjection_range_le_card {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) :
    Module.finrank K (LinearMap.range (coordProjection V S)) ≤ S.card := by
  classical
  calc
    Module.finrank K (LinearMap.range (coordProjection V S)) ≤ Module.finrank K (↑S → K) :=
      Submodule.finrank_le _
    _ = S.card := by simp

theorem factSpanFinset_eq_range_coordProjection_dualMap
    {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) :
    factSpanFinset V S = LinearMap.range (coordProjection V S).dualMap := by
  classical
  let b : Module.Basis ↑S K (Module.Dual K (↑S → K)) := (Pi.basisFun K (↑S)).dualBasis
  have himage : ((coordProjection V S).dualMap '' (Set.range b)) = (coordFun V) '' (S : Set ι) := by
    ext φ
    constructor
    · rintro ⟨ψ, ⟨i, rfl⟩, rfl⟩
      refine ⟨i.1, i.2, ?_⟩
      ext x
      simp [b, coordProjection, Module.Basis.dualBasis_apply, Pi.basisFun_repr]
    · rintro ⟨i, hi, hφ⟩
      subst hφ
      refine ⟨b ⟨i, hi⟩, ⟨⟨i, hi⟩, rfl⟩, ?_⟩
      ext x
      simp [b, coordProjection, Module.Basis.dualBasis_apply, Pi.basisFun_repr]
  calc
    factSpanFinset V S = Submodule.span K ((coordFun V) '' (S : Set ι)) := rfl
    _ = Submodule.span K ((coordProjection V S).dualMap '' (Set.range b)) := by rw [himage]
    _ = Submodule.map (coordProjection V S).dualMap (Submodule.span K (Set.range b)) := by
          rw [Submodule.map_span]
    _ = Submodule.map (coordProjection V S).dualMap ⊤ := by rw [b.span_eq]
    _ = LinearMap.range (coordProjection V S).dualMap := by rw [LinearMap.range_eq_map]

theorem factRankFinset_eq_finrank_range_coordProjection
    {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) :
    factRankFinset V S = Module.finrank K (LinearMap.range (coordProjection V S)) := by
  rw [show factRankFinset V S = Module.finrank K (factSpanFinset V S) by rfl]
  rw [factSpanFinset_eq_range_coordProjection_dualMap (V := V) (S := S)]
  exact LinearMap.finrank_range_dualMap_eq_finrank_range (coordProjection V S)

theorem factRankFinset_le_finrank_directionSpace
    {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Finset ι) :
    factRankFinset V S ≤ Module.finrank K V := by
  rw [factRankFinset_eq_finrank_range_coordProjection]
  exact LinearMap.finrank_range_le (coordProjection V S)

def DeterminesDiff {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Set ι) (i : ι) : Prop :=
  ∀ x : V, (∀ j ∈ S, coordFun V j x = 0) → coordFun V i x = 0

def DeterminesFact {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Set ι) (i : ι) : Prop :=
  ∀ ⦃x y : Ambient K ι⦄, x ∈ A.stateSet → y ∈ A.stateSet →
    (∀ j ∈ S, x j = y j) → x i = y i

def DeterminesAll {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Set ι) : Prop :=
  ∀ i, DeterminesFact A S i

lemma determinesDiff_iff_mem_factSpan {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (V : DirectionSpace K ι) (S : Set ι) [Finite S] (i : ι) :
    DeterminesDiff V S i ↔ coordFun V i ∈ factSpan V S := by
  classical
  let L : S → Module.Dual K V := fun j => coordFun V j.1
  have hrange : Set.range L = (coordFun V) '' S := by
    ext φ
    constructor
    · rintro ⟨j, rfl⟩
      exact ⟨j.1, j.2, rfl⟩
    · rintro ⟨j, hj, rfl⟩
      exact ⟨⟨j, hj⟩, rfl⟩
  constructor
  · intro h
    haveI : Finite S := ‹Finite S›
    have hker : (⨅ j : S, LinearMap.ker (L j)) ≤ LinearMap.ker (coordFun V i) := by
      intro x hx
      rw [LinearMap.mem_ker]
      apply h x
      intro j hj
      have hxj : x ∈ LinearMap.ker (L ⟨j, hj⟩) := (Submodule.mem_iInf _).1 hx ⟨j, hj⟩
      simpa [L, LinearMap.mem_ker] using hxj
    have hmem : coordFun V i ∈ Submodule.span K (Set.range L) :=
      mem_span_of_iInf_ker_le_ker (L := L) (K := coordFun V i) hker
    simpa [factSpan, hrange] using hmem
  · intro hi
    have hmem : coordFun V i ∈ Submodule.span K (Set.range L) := by
      simpa [factSpan, hrange] using hi
    obtain ⟨c, hc⟩ := (mem_span_range_iff_exists_fun K).1 hmem
    intro x hx
    have hx' : ∀ j : S, L j x = 0 := fun j => hx j.1 j.2
    have happly := congrArg (fun φ : Module.Dual K V => φ x) hc
    simpa [L, LinearMap.sum_apply, hx'] using happly.symm

lemma determinesFact_iff_determinesDiff {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Set ι) (i : ι) :
    DeterminesFact A S i ↔ DeterminesDiff A.directions S i := by
  constructor
  · intro h x hx
    let u : Ambient K ι := A.origin + x.1
    have hu : u ∈ A.stateSet := by
      change u - A.origin ∈ A.directions
      simp [u, x.2]
    have hEq : ∀ j ∈ S, u j = A.origin j := by
      intro j hj
      have hxj : coordFun A.directions j x = 0 := hx j hj
      simpa [u] using hxj
    have hiEq := h hu A.origin_mem_stateSet hEq
    simpa [u] using hiEq
  · intro h x y hx hy hxy
    let d : A.directions := ⟨x - y, A.sub_mem_directions_of_mem_stateSet hx hy⟩
    have hd : ∀ j ∈ S, coordFun A.directions j d = 0 := by
      intro j hj
      change x j - y j = 0
      simp [hxy j hj]
    have hi0 := h d hd
    change x i - y i = 0 at hi0
    exact sub_eq_zero.mp hi0

lemma determinesFact_iff_mem_factSpan {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Set ι) [Finite S] (i : ι) :
    DeterminesFact A S i ↔ coordFun A.directions i ∈ factSpan A.directions S := by
  rw [determinesFact_iff_determinesDiff, determinesDiff_iff_mem_factSpan]

lemma determinesAll_iff_span_eq {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Set ι) [Finite S] :
    DeterminesAll A S ↔ factSpan A.directions S = factSpan A.directions (Set.univ : Set ι) := by
  constructor
  · intro h
    apply le_antisymm
    · exact Submodule.span_mono <| by
        intro φ hφ
        rcases hφ with ⟨i, hi, rfl⟩
        exact ⟨i, by simp, rfl⟩
    · rw [factSpan]
      refine Submodule.span_le.2 ?_
      rintro _ ⟨i, -, rfl⟩
      exact (determinesFact_iff_mem_factSpan A S i).1 (h i)
  · intro h i
    rw [determinesFact_iff_mem_factSpan A S i]
    have hi : coordFun A.directions i ∈ factSpan A.directions (Set.univ : Set ι) :=
      Submodule.subset_span ⟨i, by simp, rfl⟩
    simpa [h] using hi

abbrev IndepFacts {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Finset ι) : Prop :=
  LinearIndepOn K (coordFun A.directions) (S : Set ι)

lemma indepFacts_empty {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) : IndepFacts A ∅ := by
  simpa [IndepFacts] using (linearIndepOn_empty (R := K) (v := coordFun A.directions))

lemma indepFacts_mono {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {I J : Finset ι} (hJ : IndepFacts A J) (hIJ : I ⊆ J) :
    IndepFacts A I :=
  hJ.mono (by simpa using hIJ)

lemma factRankFinset_eq_card_of_indepFacts {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {S : Finset ι} (hS : IndepFacts A S) :
    factRankFinset A.directions S = S.card := by
  classical
  let simg : Finset (Module.Dual K A.directions) := S.image (coordFun A.directions)
  have hsimg_ind : LinearIndepOn K _root_.id (simg : Set (Module.Dual K A.directions)) := by
    simpa [simg] using hS.id_image
  have hsimg_eq :
      (simg : Set (Module.Dual K A.directions)) = (coordFun A.directions) '' (S : Set ι) := by
    ext φ
    simp [simg]
  have hfinrank :
      Module.finrank K (Submodule.span K (simg : Set (Module.Dual K A.directions))) = simg.card :=
    finrank_span_finset_eq_card hsimg_ind
  have hSinj : Set.InjOn (coordFun A.directions) (S : Set ι) := hS.injOn
  have hcardimg : simg.card = S.card := Finset.card_image_of_injOn (by simpa using hSinj)
  calc
    factRankFinset A.directions S
        = Module.finrank K (Submodule.span K (simg : Set (Module.Dual K A.directions))) := by
            rw [show factRankFinset A.directions S =
              Module.finrank K (Submodule.span K ((coordFun A.directions) '' (S : Set ι))) by rfl]
            rw [hsimg_eq]
    _ = simg.card := hfinrank
    _ = S.card := hcardimg

lemma indepFacts_aug {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {I J : Finset ι}
    (hI : IndepFacts A I) (hJ : IndepFacts A J) (hcard : I.card < J.card) :
    ∃ e ∈ J, e ∉ I ∧ IndepFacts A (insert e I) := by
  classical
  by_contra hno
  push_neg at hno
  let t : Set (Module.Dual K A.directions) := (coordFun A.directions) '' (I : Set ι)
  have hspan : (coordFun A.directions) '' (J : Set ι) ⊆ Submodule.span K t := by
    intro φ hφ
    rcases hφ with ⟨j, hjJ, rfl⟩
    by_cases hjI : j ∈ (I : Set ι)
    · exact Submodule.subset_span ⟨j, hjI, rfl⟩
    · have hnot : ¬ IndepFacts A (insert j I) := hno j hjJ hjI
      have hinsert :
          LinearIndepOn K (coordFun A.directions) (insert j (I : Set ι)) ↔
            LinearIndepOn K (coordFun A.directions) (I : Set ι) ∧
              coordFun A.directions j ∉ Submodule.span K t := by
        simpa [t] using
          (linearIndepOn_insert (K := K) (f := coordFun A.directions) (s := (I : Set ι))
            (a := j) hjI)
      have : coordFun A.directions j ∈ Submodule.span K t := by
        by_contra hmem
        exact hnot (by simpa [IndepFacts] using (hinsert.2 ⟨hI, hmem⟩))
      exact this
  let s : Set (Module.Dual K A.directions) := (coordFun A.directions) '' (J : Set ι)
  have htfin : t.Finite := I.finite_toSet.image (coordFun A.directions)
  have hsind : LinearIndepOn K _root_.id s := by
    simpa [s] using hJ.id_image
  obtain ⟨hsfin, hle⟩ :=
    exists_finite_card_le_of_finite_of_linearIndependent_of_span
      (K := K) htfin hsind hspan
  have hJinj : Set.InjOn (coordFun A.directions) (J : Set ι) := hJ.injOn
  have hIinj : Set.InjOn (coordFun A.directions) (I : Set ι) := hI.injOn
  have hs_toFinset : hsfin.toFinset = J.image (coordFun A.directions) := by
    ext φ
    simp [s]
  have ht_toFinset : htfin.toFinset = I.image (coordFun A.directions) := by
    ext φ
    simp [t]
  have hJcard : (J.image (coordFun A.directions)).card = J.card :=
    Finset.card_image_of_injOn (by simpa using hJinj)
  have hIcard : (I.image (coordFun A.directions)).card = I.card :=
    Finset.card_image_of_injOn (by simpa using hIinj)
  rw [hs_toFinset, ht_toFinset, hJcard, hIcard] at hle
  exact (Nat.not_le_of_lt hcard) hle

noncomputable def factMatroid {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) : Matroid ι :=
  (IndepMatroid.ofFinset
    (E := (Set.univ : Set ι))
    (Indep := IndepFacts A)
    (indep_empty := indepFacts_empty A)
    (indep_subset := fun {_ _} hJ hIJ => indepFacts_mono A hJ hIJ)
    (indep_aug := fun {_ _} hI hJ hcard => indepFacts_aug A hI hJ hcard)
    (subset_ground := by intro I hI; simp)).matroid

@[simp] lemma factMatroid_indep_iff {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {I : Finset ι} : (factMatroid A).Indep I ↔ IndepFacts A I := by
  simpa [factMatroid] using
    (IndepMatroid.ofFinset_indep
      (E := (Set.univ : Set ι))
      (Indep := IndepFacts A)
      (indep_empty := indepFacts_empty A)
      (indep_subset := fun {_ _} hJ hIJ => indepFacts_mono A hJ hIJ)
      (indep_aug := fun {_ _} hI hJ hcard => indepFacts_aug A hI hJ hcard)
      (subset_ground := by intro I hI; simp)
      (I := I))

abbrev DeterminesAllFinset {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Finset ι) : Prop :=
  DeterminesAll A (S : Set ι)

lemma determinesAllFinset_iff_span_eq {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Finset ι) :
    DeterminesAllFinset A S ↔
      factSpanFinset A.directions S = factSpan A.directions (Set.univ : Set ι) := by
  simpa [DeterminesAllFinset, factSpanFinset] using determinesAll_iff_span_eq A (S : Set ι)

lemma factRankFinset_eq_total_of_determinesAllFinset {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {S : Finset ι} (hS : DeterminesAllFinset A S) :
    factRankFinset A.directions S = Module.finrank K (factSpan A.directions (Set.univ : Set ι)) := by
  rw [show factRankFinset A.directions S = Module.finrank K (factSpanFinset A.directions S) by rfl]
  rw [(determinesAllFinset_iff_span_eq A S).1 hS]

lemma coordProjection_range_eq_card_of_indepFacts {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {S : Finset ι} (hS : IndepFacts A S) :
    Module.finrank K (LinearMap.range (coordProjection A.directions S)) = S.card := by
  rw [← factRankFinset_eq_finrank_range_coordProjection, factRankFinset_eq_card_of_indepFacts A hS]

lemma coordProjection_range_eq_total_of_determinesAllFinset {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {S : Finset ι} (hS : DeterminesAllFinset A S) :
    Module.finrank K (LinearMap.range (coordProjection A.directions S)) =
      Module.finrank K (factSpan A.directions (Set.univ : Set ι)) := by
  rw [← factRankFinset_eq_finrank_range_coordProjection,
    factRankFinset_eq_total_of_determinesAllFinset A hS]

abbrev BasisFacts {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Finset ι) : Prop :=
  IndepFacts A S ∧ DeterminesAllFinset A S

abbrev MinimalDetermining {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) (S : Finset ι) : Prop :=
  DeterminesAllFinset A S ∧ ∀ ⦃T : Finset ι⦄, T ⊂ S → ¬ DeterminesAllFinset A T

lemma finrank_le_card_of_determinesAllFinset {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {S : Finset ι} (hS : DeterminesAllFinset A S) :
    Module.finrank K (factSpan A.directions (Set.univ : Set ι)) ≤ S.card := by
  classical
  let simg : Finset (Module.Dual K A.directions) := S.image (coordFun A.directions)
  have hsimg_eq :
      (simg : Set (Module.Dual K A.directions)) = (coordFun A.directions) '' (S : Set ι) := by
    ext φ
    simp [simg]
  calc
    Module.finrank K (factSpan A.directions (Set.univ : Set ι))
        = Module.finrank K (factSpanFinset A.directions S) := by
            rw [(determinesAllFinset_iff_span_eq A S).1 hS]
    _ = Module.finrank K (Submodule.span K (simg : Set (Module.Dual K A.directions))) := by
          rw [hsimg_eq]
    _ ≤ simg.card := by
          simpa using (finrank_span_finset_le_card (R := K) simg)
    _ ≤ S.card := Finset.card_image_le

lemma basisFacts_card_eq_finrank {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {S : Finset ι} (hS : BasisFacts A S) :
    S.card = Module.finrank K (factSpan A.directions (Set.univ : Set ι)) := by
  classical
  let simg : Finset (Module.Dual K A.directions) := S.image (coordFun A.directions)
  have hsimg_ind : LinearIndepOn K _root_.id (simg : Set (Module.Dual K A.directions)) := by
    simpa [simg] using hS.1.id_image
  have hsimg_eq :
      (simg : Set (Module.Dual K A.directions)) = (coordFun A.directions) '' (S : Set ι) := by
    ext φ
    simp [simg]
  have hspan_eq :
      Submodule.span K (simg : Set (Module.Dual K A.directions)) =
        factSpan A.directions (Set.univ : Set ι) := by
    rw [hsimg_eq]
    exact (determinesAllFinset_iff_span_eq A S).1 hS.2
  have hfinrank :
      Module.finrank K (Submodule.span K (simg : Set (Module.Dual K A.directions))) = simg.card :=
    finrank_span_finset_eq_card hsimg_ind
  have hSinj : Set.InjOn (coordFun A.directions) (S : Set ι) := hS.1.injOn
  have hcardimg : simg.card = S.card := Finset.card_image_of_injOn (by simpa using hSinj)
  rw [← hcardimg, ← hfinrank, hspan_eq]

lemma basisFacts_minimal {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {S : Finset ι} (hS : BasisFacts A S) :
    MinimalDetermining A S := by
  constructor
  · exact hS.2
  · intro T hT hdetT
    have hTcard : T.card < S.card := Finset.card_lt_card hT
    have hrank_le : Module.finrank K (factSpan A.directions (Set.univ : Set ι)) ≤ T.card :=
      finrank_le_card_of_determinesAllFinset A hdetT
    have hrank_eq : S.card = Module.finrank K (factSpan A.directions (Set.univ : Set ι)) :=
      basisFacts_card_eq_finrank A hS
    exact (Nat.not_le_of_lt hTcard) (by simpa [hrank_eq] using hrank_le)

lemma minimalDetermining_basis {K ι} [Field K] [Fintype ι] [DecidableEq ι]
    (A : AffineFamily K ι) {S : Finset ι} (hS : MinimalDetermining A S) :
    BasisFacts A S := by
  classical
  refine ⟨?_, hS.1⟩
  by_contra hdep
  let simg : Finset (Module.Dual K A.directions) := S.image (coordFun A.directions)
  have hsimg_eq :
      (simg : Set (Module.Dual K A.directions)) = (coordFun A.directions) '' (S : Set ι) := by
    ext φ
    simp [simg]
  let b : ↥(S : Set ι) → Module.Dual K A.directions := fun i => coordFun A.directions i.1
  have hbdep : ¬ LinearIndependent K b := by
    simpa [IndepFacts, LinearIndepOn, b] using hdep
  have hrange_eq : Set.range b = (simg : Set (Module.Dual K A.directions)) := by
    ext φ
    simp [b, simg]
  have hrank_lt :
      Module.finrank K (Submodule.span K (simg : Set (Module.Dual K A.directions))) < S.card := by
    have hnotle : ¬ Fintype.card (S : Set ι) ≤ (Set.range b).finrank K := by
      intro hle
      exact hbdep ((linearIndependent_iff_card_le_finrank_span (R := K) (b := b)).2 hle)
    have hlt : Module.finrank K (Submodule.span K (Set.range b)) < Fintype.card (S : Set ι) :=
      Nat.lt_of_not_ge hnotle
    rw [hrange_eq] at hlt
    simpa using hlt
  obtain ⟨t, hts, htcard, htspan, htind⟩ :=
    exists_finset_span_eq_linearIndepOn K (simg : Set (Module.Dual K A.directions))
  have htcard_lt : t.card < S.card := by
    rw [htcard]
    exact hrank_lt
  have hpre :
      ∀ φ, φ ∈ t → ∃ i ∈ S, coordFun A.directions i = φ := by
    intro φ hφ
    have hφsimg : φ ∈ (simg : Set (Module.Dual K A.directions)) := hts hφ
    simpa [simg] using hφsimg
  choose pick hpickS hpickEq using hpre
  let T : Finset ι := t.attach.image fun φ => pick φ.1 φ.2
  have hTsub : T ⊆ S := by
    intro i hiT
    rcases Finset.mem_image.mp hiT with ⟨φ, hφ, rfl⟩
    exact hpickS φ.1 φ.2
  have hTcard_le : T.card ≤ t.card := by
    have : T.card ≤ t.attach.card := by
      simpa [T] using (Finset.card_image_le (s := t.attach) (f := fun φ => pick φ.1 φ.2))
    simpa using this
  have hTcard_lt : T.card < S.card := lt_of_le_of_lt hTcard_le htcard_lt
  have hTss : T ⊂ S := by
    refine (Finset.ssubset_iff_of_subset hTsub).2 ?_
    have hnotST : ¬ S ⊆ T := by
      intro hST
      exact (Nat.not_lt_of_ge (Finset.card_le_card hST)) hTcard_lt
    exact not_subset.mp hnotST
  have hspan_t_le :
      Submodule.span K (t : Set (Module.Dual K A.directions)) ≤ factSpanFinset A.directions T := by
    refine Submodule.span_le.2 ?_
    intro φ hφ
    have hmemT : pick φ hφ ∈ (T : Set ι) := by
      exact Finset.mem_image.mpr ⟨⟨φ, hφ⟩, by simp [T], rfl⟩
    exact Submodule.subset_span ⟨pick φ hφ, hmemT, hpickEq φ hφ⟩
  have hfull_S :
      factSpanFinset A.directions S = factSpan A.directions (Set.univ : Set ι) :=
    (determinesAllFinset_iff_span_eq A S).1 hS.1
  have hfull_t :
      Submodule.span K (t : Set (Module.Dual K A.directions)) =
        factSpan A.directions (Set.univ : Set ι) := by
    calc
      Submodule.span K (t : Set (Module.Dual K A.directions))
          = Submodule.span K (simg : Set (Module.Dual K A.directions)) := htspan
      _ = factSpanFinset A.directions S := by rw [hsimg_eq]
      _ = factSpan A.directions (Set.univ : Set ι) := hfull_S
  have hfull_T : DeterminesAllFinset A T := by
    rw [determinesAllFinset_iff_span_eq A T]
    apply le_antisymm
    · exact Submodule.span_mono <| by
        intro φ hφ
        rcases hφ with ⟨i, hi, rfl⟩
        exact ⟨i, by simp, rfl⟩
    · calc
        factSpan A.directions (Set.univ : Set ι)
            = Submodule.span K (t : Set (Module.Dual K A.directions)) := hfull_t.symm
        _ ≤ factSpanFinset A.directions T := hspan_t_le
  exact (hS.2 hTss) hfull_T

section Product

variable {ι₁ ι₂ : Type*} [Fintype ι₁] [DecidableEq ι₁] [Fintype ι₂] [DecidableEq ι₂]

/-- The ambient sum-coordinate space is linearly equivalent to the product of the two ambient spaces. -/
noncomputable abbrev ambientSumEquivProd : Ambient K (ι₁ ⊕ ι₂) ≃ₗ[K] Ambient K ι₁ × Ambient K ι₂ :=
  LinearEquiv.sumArrowLequivProdArrow ι₁ ι₂ K K

/-- The product direction space, represented on the disjoint sum of coordinate indices. -/
noncomputable def productDirections (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂) :
    DirectionSpace K (ι₁ ⊕ ι₂) :=
  (Submodule.prod V₁ V₂).comap (ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂) :
    Ambient K (ι₁ ⊕ ι₂) →ₗ[K] Ambient K ι₁ × Ambient K ι₂)

@[simp] theorem mem_productDirections {V₁ : DirectionSpace K ι₁} {V₂ : DirectionSpace K ι₂}
    {x : Ambient K (ι₁ ⊕ ι₂)} :
    x ∈ productDirections (K := K) V₁ V₂ ↔
      (fun i => x (Sum.inl i)) ∈ V₁ ∧ (fun j => x (Sum.inr j)) ∈ V₂ := by
  change
    ((ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂) x).1 ∈ V₁ ∧
      (ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂) x).2 ∈ V₂) ↔
      (fun i => x (Sum.inl i)) ∈ V₁ ∧ (fun j => x (Sum.inr j)) ∈ V₂
  constructor
  · rintro ⟨h₁, h₂⟩
    exact ⟨by simpa [ambientSumEquivProd] using h₁, by simpa [ambientSumEquivProd] using h₂⟩
  · rintro ⟨h₁, h₂⟩
    exact ⟨by simpa [ambientSumEquivProd] using h₁, by simpa [ambientSumEquivProd] using h₂⟩

/-- The affine product family on the disjoint sum of coordinate indices. -/
noncomputable def productFamily (A₁ : AffineFamily K ι₁) (A₂ : AffineFamily K ι₂) :
    AffineFamily K (ι₁ ⊕ ι₂) where
  origin := (ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂)).symm (A₁.origin, A₂.origin)
  directions := productDirections (K := K) A₁.directions A₂.directions

theorem map_productDirections_ambientSumEquivProd (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂) :
    (productDirections (K := K) V₁ V₂).map
        (ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂) :
          Ambient K (ι₁ ⊕ ι₂) →ₗ[K] Ambient K ι₁ × Ambient K ι₂) =
      Submodule.prod V₁ V₂ := by
  ext x
  constructor
  · rintro ⟨y, hy, rfl⟩
    exact hy
  · intro hx
    refine ⟨(ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂)).symm x, ?_, by simp⟩
    simpa [productDirections] using hx

/-- The product direction space is linearly equivalent to the direct product of the two component
direction spaces. -/
noncomputable def productDirectionsEquiv (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂) :
    productDirections (K := K) V₁ V₂ ≃ₗ[K] V₁ × V₂ where
  toFun := fun x =>
    let hx : (ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂) x.1) ∈ Submodule.prod V₁ V₂ := x.2
    (⟨fun i => x.1 (Sum.inl i), by simpa [ambientSumEquivProd] using hx.1⟩,
      ⟨fun j => x.1 (Sum.inr j), by simpa [ambientSumEquivProd] using hx.2⟩)
  invFun := fun x =>
    ⟨(ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂)).symm (x.1.1, x.2.1), by
      change
        (ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂))
            ((ambientSumEquivProd (K := K) (ι₁ := ι₁) (ι₂ := ι₂)).symm (x.1.1, x.2.1)) ∈
          Submodule.prod V₁ V₂
      simpa using ⟨x.1.2, x.2.2⟩⟩
  map_add' := by
    intro x y
    ext <;> rfl
  map_smul' := by
    intro a x
    ext <;> rfl
  left_inv := by
    intro x
    ext i <;> cases i <;> simp [ambientSumEquivProd]
  right_inv := by
    rintro ⟨x, y⟩
    ext i <;> rfl

@[simp] theorem productDirectionsEquiv_apply_inl
    (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂)
    (x : productDirections (K := K) V₁ V₂) (i : ι₁) :
    ((productDirectionsEquiv (K := K) V₁ V₂ x).1 : Ambient K ι₁) i = x.1 (Sum.inl i) := rfl

@[simp] theorem productDirectionsEquiv_apply_inr
    (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂)
    (x : productDirections (K := K) V₁ V₂) (j : ι₂) :
    ((productDirectionsEquiv (K := K) V₁ V₂ x).2 : Ambient K ι₂) j = x.1 (Sum.inr j) := rfl

@[simp] theorem productDirectionsEquiv_symm_apply_inl
    (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂)
    (x₁ : V₁) (x₂ : V₂) (i : ι₁) :
    ((productDirectionsEquiv (K := K) V₁ V₂).symm (x₁, x₂)).1 (Sum.inl i) = x₁.1 i := by
  simp [productDirectionsEquiv, ambientSumEquivProd]

@[simp] theorem productDirectionsEquiv_symm_apply_inr
    (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂)
    (x₁ : V₁) (x₂ : V₂) (j : ι₂) :
    ((productDirectionsEquiv (K := K) V₁ V₂).symm (x₁, x₂)).1 (Sum.inr j) = x₂.1 j := by
  simp [productDirectionsEquiv, ambientSumEquivProd]

/-- The product direction space has additive finite dimension. -/
theorem finrank_productDirections (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂) :
    Module.finrank K (productDirections (K := K) V₁ V₂) =
      Module.finrank K V₁ + Module.finrank K V₂ := by
  rw [(productDirectionsEquiv (K := K) V₁ V₂).finrank_eq, Module.finrank_prod]

/-- The direction space of the affine product family has the expected additive dimension. -/
theorem finrank_productFamily_directions (A₁ : AffineFamily K ι₁) (A₂ : AffineFamily K ι₂) :
    Module.finrank K (productFamily (K := K) A₁ A₂).directions =
      Module.finrank K A₁.directions + Module.finrank K A₂.directions := by
  simpa [productFamily] using finrank_productDirections (K := K) A₁.directions A₂.directions

/-- The subtype of a `disjSum` finset is equivalent to the sum of the subtypes. -/
noncomputable def disjSumSubtypeEquiv (S₁ : Finset ι₁) (S₂ : Finset ι₂) :
    ↥(S₁.disjSum S₂) ≃ (↥S₁ ⊕ ↥S₂) where
  toFun x := by
    rcases x with ⟨x, hx⟩
    cases x with
    | inl i => exact Sum.inl ⟨i, by simpa using hx⟩
    | inr j => exact Sum.inr ⟨j, by simpa using hx⟩
  invFun x := by
    cases x with
    | inl i => exact ⟨Sum.inl i.1, by simpa using i.2⟩
    | inr j => exact ⟨Sum.inr j.1, by simpa using j.2⟩
  left_inv := by
    rintro ⟨x, hx⟩
    cases x <;> rfl
  right_inv := by
    intro x
    cases x <;> rfl

/-- Reindexing the projection codomain on a disjoint block split yields the product codomain. -/
noncomputable def disjSumCoordEquiv (S₁ : Finset ι₁) (S₂ : Finset ι₂) :
    (↑(S₁.disjSum S₂) → K) ≃ₗ[K] (↑S₁ → K) × (↑S₂ → K) where
  toFun f :=
    (fun i => f ⟨Sum.inl i.1, by simpa using i.2⟩,
      fun j => f ⟨Sum.inr j.1, by simpa using j.2⟩)
  invFun g := fun x =>
    match x with
    | ⟨Sum.inl i, hx⟩ => g.1 ⟨i, by simpa using hx⟩
    | ⟨Sum.inr j, hx⟩ => g.2 ⟨j, by simpa using hx⟩
  map_add' := by
    intro f g
    ext <;> rfl
  map_smul' := by
    intro a f
    ext <;> rfl
  left_inv := by
    intro f
    funext x
    rcases x with ⟨x, hx⟩
    cases x <;> rfl
  right_inv := by
    intro g
    ext <;> rfl

@[simp] theorem disjSumCoordEquiv_apply_fst (S₁ : Finset ι₁) (S₂ : Finset ι₂)
    (f : ↑(S₁.disjSum S₂) → K) (i : ↑S₁) :
    (disjSumCoordEquiv (K := K) S₁ S₂ f).1 i = f ⟨Sum.inl i.1, by simpa using i.2⟩ := by
  simp [disjSumCoordEquiv, disjSumSubtypeEquiv]

@[simp] theorem disjSumCoordEquiv_apply_snd (S₁ : Finset ι₁) (S₂ : Finset ι₂)
    (f : ↑(S₁.disjSum S₂) → K) (j : ↑S₂) :
    (disjSumCoordEquiv (K := K) S₁ S₂ f).2 j = f ⟨Sum.inr j.1, by simpa using j.2⟩ := by
  simp [disjSumCoordEquiv, disjSumSubtypeEquiv]

lemma range_comp_linearEquiv {M N P : Type*} [AddCommGroup M] [AddCommGroup N] [AddCommGroup P]
    [Module K M] [Module K N] [Module K P] (e : N ≃ₗ[K] P) (f : M →ₗ[K] N) :
    LinearMap.range ((e : N →ₗ[K] P).comp f) = (LinearMap.range f).map (e : N →ₗ[K] P) := by
  ext y
  constructor
  · rintro ⟨x, rfl⟩
    exact ⟨f x, ⟨x, rfl⟩, rfl⟩
  · rintro ⟨z, hz, rfl⟩
    rcases hz with ⟨x, rfl⟩
    exact ⟨x, rfl⟩

lemma range_comp_of_surjective {M N P : Type*} [AddCommGroup M] [AddCommGroup N] [AddCommGroup P]
    [Module K M] [Module K N] [Module K P] (f : M →ₗ[K] N) (g : N →ₗ[K] P)
    (hf : Function.Surjective f) :
    LinearMap.range (g.comp f) = LinearMap.range g := by
  ext y
  constructor
  · rintro ⟨x, rfl⟩
    exact ⟨f x, rfl⟩
  · rintro ⟨z, rfl⟩
    rcases hf z with ⟨x, rfl⟩
    exact ⟨x, rfl⟩

lemma range_prodMap {M₁ M₂ N₁ N₂ : Type*} [AddCommGroup M₁] [AddCommGroup M₂]
    [AddCommGroup N₁] [AddCommGroup N₂] [Module K M₁] [Module K M₂] [Module K N₁] [Module K N₂]
    (f : M₁ →ₗ[K] N₁) (g : M₂ →ₗ[K] N₂) :
    LinearMap.range (f.prodMap g) = (LinearMap.range f).prod (LinearMap.range g) := by
  ext y
  constructor
  · rintro ⟨x, rfl⟩
    exact ⟨⟨x.1, rfl⟩, ⟨x.2, rfl⟩⟩
  · rintro ⟨hy₁, hy₂⟩
    rcases hy₁ with ⟨x₁, hx₁⟩
    rcases hy₂ with ⟨x₂, hx₂⟩
    refine ⟨(x₁, x₂), ?_⟩
    ext <;> simp [LinearMap.prodMap_apply, hx₁, hx₂]

lemma finrank_range_comp_linearEquiv {M N P : Type*} [AddCommGroup M] [AddCommGroup N] [AddCommGroup P]
    [Module K M] [Module K N] [Module K P] (e : N ≃ₗ[K] P) (f : M →ₗ[K] N) :
    Module.finrank K (LinearMap.range ((e : N →ₗ[K] P).comp f)) =
      Module.finrank K (LinearMap.range f) := by
  calc
    Module.finrank K (LinearMap.range ((e : N →ₗ[K] P).comp f))
      = Module.finrank K ((LinearMap.range f).map (e : N →ₗ[K] P)) := by
          rw [range_comp_linearEquiv (K := K) e]
    _ = Module.finrank K (LinearMap.range f) := LinearEquiv.finrank_map_eq e _

noncomputable def submoduleProdEquiv {M N : Type*} [AddCommGroup M] [AddCommGroup N]
    [Module K M] [Module K N] (p : Submodule K M) (q : Submodule K N) :
    p.prod q ≃ₗ[K] p × q where
  toFun x := (⟨x.1.1, x.2.1⟩, ⟨x.1.2, x.2.2⟩)
  invFun x := ⟨(x.1.1, x.2.1), ⟨x.1.2, x.2.2⟩⟩
  map_add' := by
    intro x y
    ext <;> rfl
  map_smul' := by
    intro a x
    ext <;> rfl
  left_inv := by
    intro x
    rfl
  right_inv := by
    intro x
    rfl

lemma finrank_submoduleProd {M N : Type*} [AddCommGroup M] [AddCommGroup N]
    [Module K M] [Module K N] [Module.Finite K M] [Module.Finite K N]
    (p : Submodule K M) (q : Submodule K N) :
    Module.finrank K (p.prod q) = Module.finrank K p + Module.finrank K q := by
  rw [(submoduleProdEquiv (K := K) p q).finrank_eq, Module.finrank_prod]

theorem coordProjection_productDirections_disjSum
    (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂)
    (S₁ : Finset ι₁) (S₂ : Finset ι₂) :
    ((disjSumCoordEquiv (K := K) S₁ S₂ : (↑(S₁.disjSum S₂) → K) →ₗ[K] (↑S₁ → K) × (↑S₂ → K)).comp
        (coordProjection (productDirections (K := K) V₁ V₂) (S₁.disjSum S₂)))
      =
    (((coordProjection V₁ S₁).prodMap (coordProjection V₂ S₂)).comp
      (productDirectionsEquiv (K := K) V₁ V₂ :
        productDirections (K := K) V₁ V₂ →ₗ[K] V₁ × V₂)) := by
  ext x i <;> cases i <;>
    simp [disjSumCoordEquiv, disjSumSubtypeEquiv, ambientSumEquivProd, productDirectionsEquiv]

/-- Matroid rank is additive on disjoint coordinate blocks of the product family. -/
theorem range_coordProjection_productDirections_disjSum
    (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂)
    (S₁ : Finset ι₁) (S₂ : Finset ι₂) :
    LinearMap.range
      (((disjSumCoordEquiv (K := K) S₁ S₂ : (↑(S₁.disjSum S₂) → K) →ₗ[K] (↑S₁ → K) × (↑S₂ → K)).comp
        (coordProjection (productDirections (K := K) V₁ V₂) (S₁.disjSum S₂))))
      = (LinearMap.range (coordProjection V₁ S₁)).prod (LinearMap.range (coordProjection V₂ S₂)) := by
  let eDom := productDirectionsEquiv (K := K) V₁ V₂
  let eCod := disjSumCoordEquiv (K := K) S₁ S₂
  let f : productDirections (K := K) V₁ V₂ →ₗ[K] (↑(S₁.disjSum S₂) → K) :=
    coordProjection (productDirections (K := K) V₁ V₂) (S₁.disjSum S₂)
  let g : V₁ × V₂ →ₗ[K] (↑S₁ → K) × (↑S₂ → K) :=
    (coordProjection V₁ S₁).prodMap (coordProjection V₂ S₂)
  have hfg : (eCod : (↑(S₁.disjSum S₂) → K) →ₗ[K] (↑S₁ → K) × (↑S₂ → K)).comp f =
      g.comp (eDom : productDirections (K := K) V₁ V₂ →ₗ[K] V₁ × V₂) := by
    simpa [f, g] using coordProjection_productDirections_disjSum (K := K) V₁ V₂ S₁ S₂
  calc
    LinearMap.range ((eCod : (↑(S₁.disjSum S₂) → K) →ₗ[K] (↑S₁ → K) × (↑S₂ → K)).comp f)
      = LinearMap.range (g.comp (eDom : productDirections (K := K) V₁ V₂ →ₗ[K] V₁ × V₂)) := by rw [hfg]
    _ = LinearMap.range g := by
      rw [range_comp_of_surjective (K := K)
        (f := (eDom : productDirections (K := K) V₁ V₂ →ₗ[K] V₁ × V₂))
        (g := g) eDom.surjective]
    _ = (LinearMap.range (coordProjection V₁ S₁)).prod (LinearMap.range (coordProjection V₂ S₂)) := by
      rw [show g = (coordProjection V₁ S₁).prodMap (coordProjection V₂ S₂) by rfl]
      rw [range_prodMap (K := K) (coordProjection V₁ S₁) (coordProjection V₂ S₂)]

theorem finrank_range_coordProjection_productDirections_disjSum
    (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂)
    (S₁ : Finset ι₁) (S₂ : Finset ι₂) :
    Module.finrank K
        (LinearMap.range
          (((disjSumCoordEquiv (K := K) S₁ S₂ :
                (↑(S₁.disjSum S₂) → K) →ₗ[K] (↑S₁ → K) × (↑S₂ → K)).comp
            (coordProjection (productDirections (K := K) V₁ V₂) (S₁.disjSum S₂)))))
      = Module.finrank K (LinearMap.range (coordProjection V₁ S₁))
        + Module.finrank K (LinearMap.range (coordProjection V₂ S₂)) := by
  rw [range_coordProjection_productDirections_disjSum (K := K) V₁ V₂ S₁ S₂]
  rw [finrank_submoduleProd (K := K)]

-- Matroid rank is additive on disjoint coordinate blocks of the product family.
set_option maxHeartbeats 20000000 in
theorem factRankFinset_product_disjSum (V₁ : DirectionSpace K ι₁) (V₂ : DirectionSpace K ι₂)
    (S₁ : Finset ι₁) (S₂ : Finset ι₂) :
    factRankFinset (productDirections (K := K) V₁ V₂) (S₁.disjSum S₂) =
      factRankFinset V₁ S₁ + factRankFinset V₂ S₂ := by
  let eCod := disjSumCoordEquiv (K := K) S₁ S₂
  let f : productDirections (K := K) V₁ V₂ →ₗ[K] (↑(S₁.disjSum S₂) → K) :=
    coordProjection (productDirections (K := K) V₁ V₂) (S₁.disjSum S₂)
  rw [factRankFinset_eq_finrank_range_coordProjection]
  calc
    Module.finrank K (LinearMap.range f)
      = Module.finrank K (LinearMap.range ((eCod : (↑(S₁.disjSum S₂) → K) →ₗ[K] _).comp f)) := by
          simpa [f] using (finrank_range_comp_linearEquiv (K := K) eCod f).symm
    _ = Module.finrank K (LinearMap.range (coordProjection V₁ S₁)) +
          Module.finrank K (LinearMap.range (coordProjection V₂ S₂)) := by
          simpa using finrank_range_coordProjection_productDirections_disjSum (K := K) V₁ V₂ S₁ S₂
    _ = factRankFinset V₁ S₁ + factRankFinset V₂ S₂ := by
          rw [← factRankFinset_eq_finrank_range_coordProjection,
            ← factRankFinset_eq_finrank_range_coordProjection]

lemma indepFacts_iff_factRankFinset_eq_card {A : AffineFamily K ι} {S : Finset ι} :
    IndepFacts A S ↔ factRankFinset A.directions S = S.card := by
  constructor
  · exact factRankFinset_eq_card_of_indepFacts A
  · intro hRank
    classical
    let simg : Finset (Module.Dual K A.directions) := S.image (coordFun A.directions)
    have hsimg_eq :
        (simg : Set (Module.Dual K A.directions)) = (coordFun A.directions) '' (S : Set ι) := by
      ext φ
      simp [simg]
    have hfinrank_eq' :
        Module.finrank K (Submodule.span K ((coordFun A.directions) '' (S : Set ι))) = S.card := by
      simpa [factRankFinset, factSpanFinset, factSpan] using hRank
    have hfinrank_eq : Module.finrank K (Submodule.span K (simg : Set (Module.Dual K A.directions))) = S.card := by
      rw [hsimg_eq]
      exact hfinrank_eq'
    have hsimg_card : simg.card = S.card := by
      refine le_antisymm Finset.card_image_le ?_
      calc
        S.card = Module.finrank K (Submodule.span K (simg : Set (Module.Dual K A.directions))) := hfinrank_eq.symm
        _ ≤ simg.card := by simpa using (finrank_span_finset_le_card (R := K) simg)
    have hInj : Set.InjOn (coordFun A.directions) (S : Set ι) := by
      simpa [simg] using (Finset.injOn_of_card_image_eq (s := S) (f := coordFun A.directions) hsimg_card)
    have hsimg_lin : LinearIndependent K (fun φ : (simg : Set (Module.Dual K A.directions)) =>
        (φ : Module.Dual K A.directions)) := by
      apply (linearIndependent_iff_card_eq_finrank_span (R := K)
        (b := fun φ : (simg : Set (Module.Dual K A.directions)) => (φ : Module.Dual K A.directions))).2
      have hrange :
          Set.range (fun φ : (simg : Set (Module.Dual K A.directions)) =>
            (φ : Module.Dual K A.directions)) = (simg : Set (Module.Dual K A.directions)) := by
        ext φ
        simp
      calc
        Fintype.card (simg : Set (Module.Dual K A.directions)) = simg.card := by simp
        _ = Module.finrank K (Submodule.span K (simg : Set (Module.Dual K A.directions))) := by
              rw [hsimg_card]
              exact hfinrank_eq.symm
        _ = (Set.range (fun φ : (simg : Set (Module.Dual K A.directions)) =>
              (φ : Module.Dual K A.directions))).finrank K := by
              rw [hrange]
              rfl
    change LinearIndepOn K (coordFun A.directions) (S : Set ι)
    rw [LinearIndepOn_iff_linearIndepOn_image_injOn]
    rw [← hsimg_eq]
    refine ⟨?_, hInj⟩
    simpa [LinearIndepOn] using hsimg_lin

private lemma add_eq_add_iff_of_le_le {a b c d : Nat} (ha : a ≤ c) (hb : b ≤ d) :
    a + b = c + d ↔ a = c ∧ b = d := by
  constructor
  · intro h
    have hca : c ≤ a := by
      apply Nat.le_of_add_le_add_right
      calc
        c + b ≤ c + d := Nat.add_le_add_left hb c
        _ = a + b := by simpa [h]
    have hac : a ≤ c := ha
    have hEqA : a = c := Nat.le_antisymm hac hca
    have hEqB : b = d := by
      have h' : c + b = c + d := by simpa [hEqA] using h
      exact Nat.add_left_cancel h'
    exact ⟨hEqA, hEqB⟩
  · rintro ⟨rfl, rfl⟩
    rfl

/-- Independence in the product affine family splits across the two coordinate blocks. -/
theorem indepFacts_productFamily_iff (A₁ : AffineFamily K ι₁) (A₂ : AffineFamily K ι₂)
    (I : Finset (ι₁ ⊕ ι₂)) :
    IndepFacts (productFamily (K := K) A₁ A₂) I ↔
      IndepFacts A₁ I.toLeft ∧ IndepFacts A₂ I.toRight := by
  rw [indepFacts_iff_factRankFinset_eq_card,
    indepFacts_iff_factRankFinset_eq_card,
    indepFacts_iff_factRankFinset_eq_card]
  rw [show (productFamily (K := K) A₁ A₂).directions =
      productDirections (K := K) A₁.directions A₂.directions by rfl]
  rw [← Finset.toLeft_disjSum_toRight (u := I),
    factRankFinset_product_disjSum (K := K) A₁.directions A₂.directions I.toLeft I.toRight,
    Finset.toLeft_disjSum, Finset.toRight_disjSum,
    Finset.card_disjSum]
  exact add_eq_add_iff_of_le_le
    (factRankFinset_le_card A₁.directions I.toLeft)
    (factRankFinset_le_card A₂.directions I.toRight)

/-- The representable matroid of the product family is the direct sum of the component matroids. -/
theorem factMatroid_product_indep_iff (A₁ : AffineFamily K ι₁) (A₂ : AffineFamily K ι₂)
    {I : Finset (ι₁ ⊕ ι₂)} :
    (factMatroid (productFamily (K := K) A₁ A₂)).Indep I ↔
      (factMatroid A₁).Indep I.toLeft ∧ (factMatroid A₂).Indep I.toRight := by
  simp [factMatroid_indep_iff, indepFacts_productFamily_iff]

end Product

end FactMatroid
end Ssot
