import Mathlib.Combinatorics.Matroid.IndepAxioms
import Mathlib.LinearAlgebra.Dual.Lemmas
import Mathlib.LinearAlgebra.Dimension.Constructions
import Mathlib.LinearAlgebra.Dimension.OrzechProperty
import Mathlib.LinearAlgebra.LinearIndependent.Lemmas

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

end FactMatroid
end Ssot
