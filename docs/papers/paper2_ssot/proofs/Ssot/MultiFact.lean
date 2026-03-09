import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Fin.Tuple.Basic
import Mathlib.Data.Fin.Tuple.Take
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Finset.Powerset
import Mathlib.Analysis.Subadditive
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.InnerProductSpace.Orthonormal
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.ENNReal.Real
import Mathlib.Combinatorics.SimpleGraph.Coloring
import Mathlib.Combinatorics.SimpleGraph.Clique
import Mathlib.Tactic

namespace MultiFact

open Set Filter Topology

/-- A latent multi-fact state with `F` coordinates, each taking values in `Fin A`. -/
abbrev State (F A : Nat) := Fin F → Fin A

/-- A partial observation masks unrevealed coordinates with `none`. -/
abbrev PartialObservation (F A : Nat) := Fin F → Option (Fin A)

instance instFintypeState (F A : Nat) : Fintype (State F A) := by
  dsimp [State]
  infer_instance

instance instDecidableEqState (F A : Nat) : DecidableEq (State F A) := by
  dsimp [State]
  infer_instance

instance instFintypePartialObservation (F A : Nat) : Fintype (PartialObservation F A) := by
  dsimp [PartialObservation]
  infer_instance

instance instDecidableEqPartialObservation (F A : Nat) : DecidableEq (PartialObservation F A) := by
  dsimp [PartialObservation]
  infer_instance

/-- A location reveals exactly the coordinates in its finite view. -/
def observe {F A : Nat} (view : Finset (Fin F)) (x : State F A) : PartialObservation F A :=
  fun i => if i ∈ view then some (x i) else none

/-- A family of observation locations, each with its own revealed coordinate set. -/
abbrev ViewFamily (F L : Nat) := Fin L → Finset (Fin F)

/--
Two latent states are confusable when some allowed location yields the same partial observation.
This is the natural zero-error adjacency relation for decoders that must succeed uniformly over
all allowed locations.
-/
def Confusable {F A L : Nat} (views : ViewFamily F L) (x y : State F A) : Prop :=
  x ≠ y ∧ ∃ ℓ : Fin L, observe (views ℓ) x = observe (views ℓ) y

instance instDecidableConfusable {F A L : Nat}
    (views : ViewFamily F L) (x y : State F A) : Decidable (Confusable views x y) := by
  unfold Confusable
  infer_instance

/-- A finite auxiliary tag is a proper coloring when it separates every confusable pair. -/
def ProperColoring {F A L T : Nat} (views : ViewFamily F L) (tag : State F A → Fin T) : Prop :=
  ∀ ⦃x y : State F A⦄, Confusable views x y → tag x ≠ tag y

instance instDecidableProperColoring {F A L T : Nat}
    (views : ViewFamily F L) (tag : State F A → Fin T) : Decidable (ProperColoring views tag) := by
  unfold ProperColoring
  infer_instance

/-- The confusability graph induced by `views` is `T`-colorable. -/
def Colorable {F A L T : Nat} (views : ViewFamily F L) : Prop :=
  ∃ tag : State F A → Fin T, ProperColoring views tag

instance instDecidableColorable {F A L T : Nat}
    (views : ViewFamily F L) : Decidable (Colorable (F := F) (A := A) (L := L) (T := T) views) := by
  unfold Colorable
  infer_instance

/--
Exact zero-error recovery from location index, partial observation, and auxiliary tag.
The decoder must succeed uniformly for every state and every allowed location.
-/
def ExactRecovery {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A)) : Prop :=
  ∀ ℓ x, decode ℓ (observe (views ℓ) x) (tag x) = some x

theorem exactRecovery_implies_properColoring
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A))
    (hexact : ExactRecovery views tag decode) :
    ProperColoring views tag := by
  intro x y hconf htag
  rcases hconf with ⟨hxy, ℓ, hobs⟩
  have hx : decode ℓ (observe (views ℓ) x) (tag x) = some x := hexact ℓ x
  have hy : decode ℓ (observe (views ℓ) y) (tag y) = some y := hexact ℓ y
  have hy' : decode ℓ (observe (views ℓ) y) (tag x) = some y := by
    simpa [htag] using hy
  have hy'' : decode ℓ (observe (views ℓ) x) (tag x) = some y := by
    simpa [hobs] using hy'
  have : some x = some y := hx.symm.trans hy''
  exact hxy (Option.some.inj this)

theorem properColoring_implies_exactRecovery
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (hcolor : ProperColoring views tag) :
    ∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
      ExactRecovery views tag decode := by
  classical
  refine ⟨fun ℓ o t =>
    if h : ∃ x : State F A, observe (views ℓ) x = o ∧ tag x = t
    then some (Classical.choose h)
    else none, ?_⟩
  intro ℓ x
  have hex : ∃ y : State F A, observe (views ℓ) y = observe (views ℓ) x ∧ tag y = tag x := by
    exact ⟨x, rfl, rfl⟩
  change (if h : ∃ y : State F A, observe (views ℓ) y = observe (views ℓ) x ∧ tag y = tag x
    then some (Classical.choose h)
    else none) = some x
  rw [dif_pos hex]
  have hchoose :
      Classical.choose hex = x := by
    rcases Classical.choose_spec hex with ⟨hobs, htag⟩
    by_cases hEq : Classical.choose hex = x
    · exact hEq
    · exfalso
      have hconf : Confusable views (Classical.choose hex) x := by
        refine ⟨hEq, ?_⟩
        exact ⟨ℓ, hobs⟩
      exact (hcolor hconf) htag
  simp [hchoose]

theorem exactRecovery_iff_properColoring
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T) :
    (∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
        ExactRecovery views tag decode) ↔
      ProperColoring views tag := by
  constructor
  · rintro ⟨decode, hexact⟩
    exact exactRecovery_implies_properColoring views tag decode hexact
  · exact properColoring_implies_exactRecovery views tag

/-- Existence of an exact zero-error decoder is equivalent to `T`-colorability. -/
theorem exists_exactRecovery_iff_exists_properColoring
    {F A L T : Nat}
    (views : ViewFamily F L) :
    (∃ tag : State F A → Fin T,
        ∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
          ExactRecovery views tag decode) ↔
      ∃ tag : State F A → Fin T, ProperColoring views tag := by
  constructor
  · rintro ⟨tag, decode, hexact⟩
    exact ⟨tag, exactRecovery_implies_properColoring views tag decode hexact⟩
  · rintro ⟨tag, hcolor⟩
    rcases properColoring_implies_exactRecovery views tag hcolor with ⟨decode, hexact⟩
    exact ⟨tag, decode, hexact⟩

/-- Zero-error recovery with `T` auxiliary symbols is equivalent to `T`-colorability. -/
theorem exists_exactRecovery_iff_colorable
    {F A L T : Nat}
    (views : ViewFamily F L) :
    (∃ tag : State F A → Fin T,
        ∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
          ExactRecovery views tag decode) ↔
      Colorable (F := F) (A := A) (L := L) (T := T) views := by
  exact exists_exactRecovery_iff_exists_properColoring views

theorem confusable_symm
    {F A L : Nat}
    (views : ViewFamily F L)
    {x y : State F A} :
    Confusable views x y → Confusable views y x := by
  rintro ⟨hxy, ℓ, hobs⟩
  exact ⟨hxy.symm, ℓ, hobs.symm⟩

/-- The simple confusability graph induced by the observation family. -/
def confusabilityGraph {F A L : Nat} (views : ViewFamily F L) : SimpleGraph (State F A) where
  Adj := Confusable views
  symm := by
    intro x y h
    exact confusable_symm views h
  loopless := by
    intro x h
    exact h.1 rfl

def graphColoringOfProperColoring
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (hcolor : ProperColoring views tag) :
    (confusabilityGraph (A := A) views).Coloring (Fin T) :=
  SimpleGraph.Coloring.mk tag (by
    intro x y hadj
    exact hcolor hadj)

theorem properColoring_of_graphColoring
    {F A L T : Nat}
    (views : ViewFamily F L)
    (c : (confusabilityGraph (A := A) views).Coloring (Fin T)) :
    ProperColoring views c := by
  intro x y hconf
  exact c.valid hconf

theorem colorable_iff_graphColorable
    {F A L T : Nat}
    (views : ViewFamily F L) :
    Colorable (F := F) (A := A) (L := L) (T := T) views ↔
      (confusabilityGraph (A := A) views).Colorable T := by
  constructor
  · rintro ⟨tag, htag⟩
    exact ⟨graphColoringOfProperColoring (A := A) views tag htag⟩
  · rintro ⟨c⟩
    exact ⟨c, properColoring_of_graphColoring (A := A) views c⟩

/-- Finite independence predicate for simple graphs. -/
def GraphIndependent {V : Type} (G : SimpleGraph V) (s : Finset V) : Prop :=
  ∀ ⦃x y : V⦄, x ∈ s → y ∈ s → x ≠ y → ¬ G.Adj x y

/-- Largest finite independent-set size of a finite simple graph. -/
noncomputable def graphMaxIndependentCard {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) : Nat :=
  by
    classical
    exact (((Finset.univ : Finset V).powerset.filter fun s : Finset V => GraphIndependent G s).image Finset.card).max'
      (by
        refine ⟨0, ?_⟩
        refine Finset.mem_image.mpr ?_
        refine ⟨∅, ?_, by simp⟩
        simp [GraphIndependent])

theorem graphIndependent_card_le_graphMaxIndependentCard
    {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V)
    (s : Finset V)
    (hindep : GraphIndependent G s) :
    s.card ≤ graphMaxIndependentCard G := by
  classical
  have hmem : s.card ∈ (((Finset.univ : Finset V).powerset.filter
      fun t : Finset V => GraphIndependent G t).image Finset.card) := by
    refine Finset.mem_image.mpr ?_
    refine ⟨s, ?_, rfl⟩
    simp [hindep]
  exact Finset.le_max' _ _ hmem

theorem exists_graphIndependent_card_eq_graphMaxIndependentCard
    {V : Type} [Fintype V] [DecidableEq V]
    (G : SimpleGraph V) :
    ∃ s : Finset V, GraphIndependent G s ∧ s.card = graphMaxIndependentCard G := by
  classical
  have hmem : graphMaxIndependentCard G ∈
      (((Finset.univ : Finset V).powerset.filter
        fun t : Finset V => GraphIndependent G t).image Finset.card) := by
    exact Finset.max'_mem _ _
  rcases Finset.mem_image.mp hmem with ⟨s, hs, hcard⟩
  exact ⟨s, by simpa using hs, hcard⟩

theorem graphIndependent_image
    {V W : Type} [Fintype V] [Fintype W] [DecidableEq V] [DecidableEq W]
    {G : SimpleGraph V} {H : SimpleGraph W}
    (e : G ≃g H)
    {s : Finset V}
    (hs : GraphIndependent G s) :
    GraphIndependent H (s.image e) := by
  intro x y hx hy hxy hAdj
  rcases Finset.mem_image.mp hx with ⟨x', hx', rfl⟩
  rcases Finset.mem_image.mp hy with ⟨y', hy', hey⟩
  have hneq : x' ≠ y' := by
    intro hEq
    apply hxy
    simpa [hEq] using hey
  have : ¬ G.Adj x' y' := hs hx' hy' hneq
  apply this
  exact (e.map_rel_iff').mp (by simpa [hey] using hAdj)

theorem graphMaxIndependentCard_iso_eq
    {V W : Type} [Fintype V] [Fintype W] [DecidableEq V] [DecidableEq W]
    {G : SimpleGraph V} {H : SimpleGraph W}
    (e : G ≃g H) :
    graphMaxIndependentCard G = graphMaxIndependentCard H := by
  classical
  rcases exists_graphIndependent_card_eq_graphMaxIndependentCard G with ⟨s, hs, hcard⟩
  rcases exists_graphIndependent_card_eq_graphMaxIndependentCard H with ⟨t, ht, hcardt⟩
  have hs' : GraphIndependent H (s.image e) := graphIndependent_image e hs
  have hle₁ : graphMaxIndependentCard G ≤ graphMaxIndependentCard H := by
    calc
      graphMaxIndependentCard G = (s.image e).card := by
        rw [← hcard, Finset.card_image_of_injective (s := s) e.injective]
      _ ≤ graphMaxIndependentCard H := graphIndependent_card_le_graphMaxIndependentCard H _ hs'
  have ht' : GraphIndependent G (t.image e.symm) := graphIndependent_image e.symm ht
  have hle₂ : graphMaxIndependentCard H ≤ graphMaxIndependentCard G := by
    calc
      graphMaxIndependentCard H = (t.image e.symm).card := by
        rw [← hcardt, Finset.card_image_of_injective (s := t) e.symm.injective]
      _ ≤ graphMaxIndependentCard G := graphIndependent_card_le_graphMaxIndependentCard G _ ht'
  exact le_antisymm hle₁ hle₂

/-- A Lovász-theta-style upper-bound witness built from an orthonormal representation
of the complement relation together with a unit handle vector. -/
structure ThetaWitness {V β : Type} [Fintype β] [DecidableEq β] (G : SimpleGraph V) where
  repr : V → EuclideanSpace ℝ β
  handle : EuclideanSpace ℝ β
  repr_unit : ∀ v, ‖repr v‖ = 1
  handle_unit : ‖handle‖ = 1
  ortho_of_nonadj : ∀ ⦃v w : V⦄, v ≠ w → ¬ G.Adj v w → inner ℝ (repr v) (repr w) = 0
  bound : ℝ
  one_le_bound : 1 ≤ bound
  one_le_bound_inner_sq : ∀ v, 1 ≤ bound * (inner ℝ (repr v) handle)^2

noncomputable def thetaWitnessOfColoring
    {V β : Type} [Fintype β] [DecidableEq β] [Nonempty β]
    {G : SimpleGraph V}
    (C : Gᶜ.Coloring β) :
    ThetaWitness (β := β) G where
  repr := fun v => EuclideanSpace.single (C v) (1 : ℝ)
  handle := (EuclideanSpace.equiv (𝕜 := ℝ) (ι := β)).symm
    (fun _ => (Real.sqrt (Fintype.card β : ℝ))⁻¹)
  repr_unit := by
    intro v
    simpa using (EuclideanSpace.norm_single (C v) (1 : ℝ))
  handle_unit := by
    have hβpos_nat : 0 < Fintype.card β := Fintype.card_pos_iff.mpr ‹Nonempty β›
    have hβpos : 0 < (Fintype.card β : ℝ) := by exact_mod_cast hβpos_nat
    have hβsqrt_ne : Real.sqrt (Fintype.card β : ℝ) ≠ 0 := by
      exact ne_of_gt (Real.sqrt_pos.2 hβpos)
    let c : ℝ := (Real.sqrt (Fintype.card β : ℝ))⁻¹
    have hsq :
        ‖((EuclideanSpace.equiv (𝕜 := ℝ) (ι := β)).symm (fun _ : β => c) : EuclideanSpace ℝ β)‖ ^ 2 = 1 := by
      rw [EuclideanSpace.norm_sq_eq]
      simp only [EuclideanSpace.equiv, Real.norm_eq_abs, inv_pow, sq_abs, Finset.sum_const,
        Finset.card_univ]
      have hcalc : ((Fintype.card β : ℝ) : ℝ) * c ^ 2 = 1 := by
        dsimp [c]
        field_simp [hβsqrt_ne]
        rw [Real.sq_sqrt (show 0 ≤ (Fintype.card β : ℝ) by positivity)]
      simpa [c, pow_two] using hcalc
    have hnonneg :
        0 ≤ ‖((EuclideanSpace.equiv (𝕜 := ℝ) (ι := β)).symm (fun _ : β => c) : EuclideanSpace ℝ β)‖ := norm_nonneg _
    nlinarith
  ortho_of_nonadj := by
    intro v w hvw hnotAdj
    have hcompl : (Gᶜ).Adj v w := (SimpleGraph.compl_adj G v w).2 ⟨hvw, hnotAdj⟩
    have hneq : C v ≠ C w := C.valid hcompl
    rw [EuclideanSpace.inner_single_left]
    simp [EuclideanSpace.single_apply, hneq]
  bound := Fintype.card β
  one_le_bound := by exact_mod_cast (Nat.succ_le_of_lt (Fintype.card_pos_iff.mpr ‹Nonempty β›))
  one_le_bound_inner_sq := by
    intro v
    have hβpos : 0 < (Fintype.card β : ℝ) := by
      exact_mod_cast (Fintype.card_pos_iff.mpr ‹Nonempty β›)
    have hβsqrt_ne : Real.sqrt (Fintype.card β : ℝ) ≠ 0 := by
      exact ne_of_gt (Real.sqrt_pos.2 hβpos)
    let c : ℝ := (Real.sqrt (Fintype.card β : ℝ))⁻¹
    have hinner :
        inner ℝ (EuclideanSpace.single (C v) (1 : ℝ))
            (((EuclideanSpace.equiv (𝕜 := ℝ) (ι := β)).symm (fun _ : β => c)) : EuclideanSpace ℝ β)
          = c := by
      rw [EuclideanSpace.inner_single_left]
      simp [EuclideanSpace.equiv, c]
    have hcalc :
        (Fintype.card β : ℝ) * c^2 = 1 := by
      dsimp [c]
      field_simp [hβsqrt_ne]
      rw [Real.sq_sqrt (show 0 ≤ (Fintype.card β : ℝ) by positivity)]
    have hone : 1 ≤ (Fintype.card β : ℝ) * c ^ 2 := by
      linarith [hcalc]
    rw [hinner]
    simpa [c] using hone

theorem ThetaWitness.orthonormal_on_independent
    {V β : Type} [Fintype β] [DecidableEq β] {G : SimpleGraph V}
    (W : ThetaWitness (β := β) G)
    {s : Finset V}
    (hs : GraphIndependent G s) :
    Orthonormal ℝ (fun i : s => W.repr i.1) := by
  classical
  rw [orthonormal_iff_ite]
  intro i j
  by_cases hij : i = j
  · subst hij
    have hnorm := W.repr_unit i.1
    simpa [sq, real_inner_self_eq_norm_sq, hnorm]
  · have hneq : i.1 ≠ j.1 := by
      intro h
      apply hij
      ext
      simpa using h
    have hnotAdj : ¬ G.Adj i.1 j.1 := hs i.2 j.2 hneq
    have hortho := W.ortho_of_nonadj hneq hnotAdj
    simp [hij, hortho]

theorem graphIndependent_card_le_thetaWitness
    {V β : Type} [Fintype V] [DecidableEq V] [Fintype β] [DecidableEq β]
    {G : SimpleGraph V}
    (W : ThetaWitness (β := β) G)
    (s : Finset V)
    (hs : GraphIndependent G s) :
    (s.card : ℝ) ≤ W.bound := by
  classical
  let hv := W.orthonormal_on_independent hs
  have hbessel :
      ∑ i : s, (inner ℝ (W.repr i.1) W.handle)^2 ≤ 1 := by
    have h := hv.sum_inner_products_le (x := W.handle) (s := (Finset.univ : Finset s))
    simpa [Real.norm_eq_abs, sq_abs, W.handle_unit] using h
  have hsum_le :
      (s.card : ℝ) ≤ ∑ i : s, W.bound * (inner ℝ (W.repr i.1) W.handle)^2 := by
    have hsum :
        ∑ i : s, (1 : ℝ) ≤ ∑ i : s, W.bound * (inner ℝ (W.repr i.1) W.handle)^2 := by
      exact Finset.sum_le_sum (fun i _ => W.one_le_bound_inner_sq i.1)
    simpa using hsum
  have hbound_nonneg : 0 ≤ W.bound := le_trans zero_le_one W.one_le_bound
  calc
    (s.card : ℝ) ≤ ∑ i : s, W.bound * (inner ℝ (W.repr i.1) W.handle)^2 := hsum_le
    _ = W.bound * ∑ i : s, (inner ℝ (W.repr i.1) W.handle)^2 := by
      rw [Finset.mul_sum]
    _ ≤ W.bound * 1 := by
      exact mul_le_mul_of_nonneg_left hbessel hbound_nonneg
    _ = W.bound := by ring

/-- Coordinatewise tensor product of two finite Euclidean vectors. -/
noncomputable def tensorVec {β γ : Type} [Fintype β] [DecidableEq β] [Fintype γ] [DecidableEq γ]
    (u : EuclideanSpace ℝ β) (v : EuclideanSpace ℝ γ) : EuclideanSpace ℝ (β × γ) :=
  (EuclideanSpace.equiv (𝕜 := ℝ) (ι := β × γ)).symm (fun p : β × γ => u p.1 * v p.2)

@[simp] theorem tensorVec_apply
    {β γ : Type} [Fintype β] [DecidableEq β] [Fintype γ] [DecidableEq γ]
    (u : EuclideanSpace ℝ β) (v : EuclideanSpace ℝ γ) (p : β × γ) :
    tensorVec u v p = u p.1 * v p.2 := by
  simp [tensorVec, EuclideanSpace.equiv]

theorem inner_tensorVec
    {β γ : Type} [Fintype β] [DecidableEq β] [Fintype γ] [DecidableEq γ]
    (u u' : EuclideanSpace ℝ β) (v v' : EuclideanSpace ℝ γ) :
    inner ℝ (tensorVec u v) (tensorVec u' v') =
      inner ℝ u u' * inner ℝ v v' := by
  rw [PiLp.inner_apply, Fintype.sum_prod_type]
  rw [PiLp.inner_apply, PiLp.inner_apply]
  have hlhs :
      (∑ x, ∑ y, inner ℝ ((tensorVec u v).ofLp (x, y)) ((tensorVec u' v').ofLp (x, y))) =
        ∑ x, ∑ y, (u x * u' x) * (v y * v' y) := by
    simp [tensorVec_apply, mul_assoc, mul_left_comm, mul_comm]
  rw [hlhs, ← Finset.sum_mul_sum]
  simpa [RCLike.inner_apply, mul_comm]

theorem norm_tensorVec
    {β γ : Type} [Fintype β] [DecidableEq β] [Fintype γ] [DecidableEq γ]
    (u : EuclideanSpace ℝ β) (v : EuclideanSpace ℝ γ) :
    ‖tensorVec u v‖ = ‖u‖ * ‖v‖ := by
  have hsq : ‖tensorVec u v‖ ^ 2 = (‖u‖ * ‖v‖) ^ 2 := by
    rw [← real_inner_self_eq_norm_sq, inner_tensorVec, real_inner_self_eq_norm_sq,
      real_inner_self_eq_norm_sq]
    ring
  have hten : 0 ≤ ‖tensorVec u v‖ := norm_nonneg _
  have hu : 0 ≤ ‖u‖ := norm_nonneg u
  have hv : 0 ≤ ‖v‖ := norm_nonneg v
  exact (sq_eq_sq₀ hten (mul_nonneg hu hv)).1 hsq

def ThetaWitness.reindex
    {V W β : Type} [Fintype β] [DecidableEq β]
    {G : SimpleGraph V} {H : SimpleGraph W}
    (e : H ≃g G)
    (Wit : ThetaWitness (β := β) G) :
    ThetaWitness (β := β) H where
  repr := fun v => Wit.repr (e v)
  handle := Wit.handle
  repr_unit := by intro v; simpa using Wit.repr_unit (e v)
  handle_unit := Wit.handle_unit
  ortho_of_nonadj := by
    intro v w hneq hnonadj
    apply Wit.ortho_of_nonadj
    · intro hEq
      apply hneq
      exact e.injective hEq
    · intro hAdj
      apply hnonadj
      exact (e.map_rel_iff').mp hAdj
  bound := Wit.bound
  one_le_bound := Wit.one_le_bound
  one_le_bound_inner_sq := by
    intro v
    simpa using Wit.one_le_bound_inner_sq (e v)

/-- Strong product of simple graphs. Vertices are adjacent when each coordinate is either equal
or adjacent, and at least one coordinate is adjacent. -/
def strongProd {α β : Type} (G : SimpleGraph α) (H : SimpleGraph β) : SimpleGraph (α × β) where
  Adj x y :=
    ((x.1 = y.1) ∨ G.Adj x.1 y.1) ∧
    ((x.2 = y.2) ∨ H.Adj x.2 y.2) ∧
    (G.Adj x.1 y.1 ∨ H.Adj x.2 y.2)
  symm := by
    intro x y h
    rcases h with ⟨h1, h2, h3⟩
    refine ⟨?_, ?_, ?_⟩
    · rcases h1 with hEq | hAdj
      · exact Or.inl hEq.symm
      · exact Or.inr (G.symm hAdj)
    · rcases h2 with hEq | hAdj
      · exact Or.inl hEq.symm
      · exact Or.inr (H.symm hAdj)
    · rcases h3 with hAdj | hAdj
      · exact Or.inl (G.symm hAdj)
      · exact Or.inr (H.symm hAdj)
  loopless := by
    intro x h
    rcases h with ⟨_, _, h3⟩
    rcases h3 with hAdj | hAdj
    · exact G.loopless _ hAdj
    · exact H.loopless _ hAdj

@[simp] theorem strongProd_adj_iff {α β : Type} {G : SimpleGraph α} {H : SimpleGraph β}
    {x y : α × β} :
    (strongProd G H).Adj x y ↔
      (((x.1 = y.1) ∨ G.Adj x.1 y.1) ∧
       ((x.2 = y.2) ∨ H.Adj x.2 y.2) ∧
       (G.Adj x.1 y.1 ∨ H.Adj x.2 y.2)) := by
  rfl

theorem strongProd_nonadj_cases
    {α β : Type} {G : SimpleGraph α} {H : SimpleGraph β}
    {x y : α × β}
    (hneq : x ≠ y)
    (hnonadj : ¬ (strongProd G H).Adj x y) :
    (x.1 ≠ y.1 ∧ ¬ G.Adj x.1 y.1) ∨ (x.2 ≠ y.2 ∧ ¬ H.Adj x.2 y.2) := by
  by_cases hbad1 : x.1 ≠ y.1 ∧ ¬ G.Adj x.1 y.1
  · exact Or.inl hbad1
  · right
    have hright : x.2 ≠ y.2 := by
      intro hEq₂
      apply hnonadj
      have hleft_ok : x.1 = y.1 ∨ G.Adj x.1 y.1 := by
        by_cases hEq₁ : x.1 = y.1
        · exact Or.inl hEq₁
        · have : G.Adj x.1 y.1 := by
            by_contra hNotAdj
            exact hbad1 ⟨hEq₁, hNotAdj⟩
          exact Or.inr this
      exact ⟨hleft_ok, Or.inl hEq₂, by
        rcases hleft_ok with hEq₁ | hAdj₁
        · exfalso
          apply hneq
          exact Prod.ext hEq₁ hEq₂
        · exact Or.inl hAdj₁⟩
    have hnot₂ : ¬ H.Adj x.2 y.2 := by
      intro hAdj₂
      apply hnonadj
      have hleft_ok : x.1 = y.1 ∨ G.Adj x.1 y.1 := by
        by_cases hEq₁ : x.1 = y.1
        · exact Or.inl hEq₁
        · have : G.Adj x.1 y.1 := by
            by_contra hNotAdj
            exact hbad1 ⟨hEq₁, hNotAdj⟩
          exact Or.inr this
      exact ⟨hleft_ok, Or.inr hAdj₂, Or.inr hAdj₂⟩
    exact ⟨hright, hnot₂⟩

noncomputable def ThetaWitness.prod
    {V W β γ : Type}
    [Fintype β] [DecidableEq β] [Fintype γ] [DecidableEq γ]
    {G : SimpleGraph V} {H : SimpleGraph W}
    (WG : ThetaWitness (β := β) G)
    (WH : ThetaWitness (β := γ) H) :
    ThetaWitness (β := β × γ) (strongProd G H) where
  repr := fun x : V × W => tensorVec (WG.repr x.1) (WH.repr x.2)
  handle := tensorVec WG.handle WH.handle
  repr_unit := by
    intro x
    rw [norm_tensorVec, WG.repr_unit, WH.repr_unit]
    ring
  handle_unit := by
    rw [norm_tensorVec, WG.handle_unit, WH.handle_unit]
    ring
  ortho_of_nonadj := by
    intro x y hneq hnonadj
    rcases strongProd_nonadj_cases (G := G) (H := H) hneq hnonadj with hcase | hcase
    · rcases hcase with ⟨hxy, hnadj⟩
      rw [inner_tensorVec]
      simp [WG.ortho_of_nonadj hxy hnadj]
    · rcases hcase with ⟨hxy, hnadj⟩
      rw [inner_tensorVec]
      simp [WH.ortho_of_nonadj hxy hnadj]
  bound := WG.bound * WH.bound
  one_le_bound := by
    have hWG : 0 ≤ WG.bound := le_trans zero_le_one WG.one_le_bound
    calc
      1 ≤ WG.bound := WG.one_le_bound
      _ ≤ WG.bound * WH.bound := by
        simpa [one_mul] using (mul_le_mul_of_nonneg_left WH.one_le_bound hWG)
  one_le_bound_inner_sq := by
    intro x
    have h1 := WG.one_le_bound_inner_sq x.1
    have h2 := WH.one_le_bound_inner_sq x.2
    rw [inner_tensorVec]
    nlinarith [h1, h2, WG.one_le_bound, WH.one_le_bound]

/-- The `N`-fold strong power of a graph on `Fin N`-indexed tuples. -/
def strongPow {α : Type} (G : SimpleGraph α) (N : Nat) : SimpleGraph (Fin N → α) where
  Adj x y :=
    (∀ i : Fin N, x i = y i ∨ G.Adj (x i) (y i)) ∧
      ∃ i : Fin N, G.Adj (x i) (y i)
  symm := by
    intro x y h
    rcases h with ⟨hcoord, i, hi⟩
    refine ⟨?_, i, G.symm hi⟩
    intro j
    rcases hcoord j with hEq | hAdj
    · exact Or.inl hEq.symm
    · exact Or.inr (G.symm hAdj)
  loopless := by
    intro x h
    rcases h with ⟨_, i, hi⟩
    exact G.loopless _ hi

@[simp] theorem strongPow_adj_iff {α : Type} {G : SimpleGraph α} {N : Nat}
    {x y : Fin N → α} :
    (strongPow G N).Adj x y ↔
      ((∀ i : Fin N, x i = y i ∨ G.Adj (x i) (y i)) ∧
        ∃ i : Fin N, G.Adj (x i) (y i)) := by
  rfl

/-- Concatenation of vertices for strong powers. -/
def concatPowVertex {α : Type} {M N : Nat}
    (x : Fin M → α) (y : Fin N → α) : Fin (M + N) → α :=
  Fin.append x y

theorem concatPowVertex_injective {α : Type} {M N : Nat} :
    Function.Injective (fun p : (Fin M → α) × (Fin N → α) => concatPowVertex p.1 p.2) := by
  intro a b h
  rcases a with ⟨x₁, y₁⟩
  rcases b with ⟨x₂, y₂⟩
  have hleft : x₁ = x₂ := by
    funext i
    simpa [concatPowVertex] using congrFun h (Fin.castAdd N i)
  have hright : y₁ = y₂ := by
    funext j
    simpa [concatPowVertex] using congrFun h (Fin.natAdd M j)
  simp [hleft, hright]

theorem strongPowAdj_add_iff_strongProd
    {α : Type} {G : SimpleGraph α}
    {M N : Nat} {x₁ y₁ : Fin M → α} {x₂ y₂ : Fin N → α} :
    (strongPow G (M + N)).Adj (concatPowVertex x₁ x₂) (concatPowVertex y₁ y₂) ↔
      (strongProd (strongPow G M) (strongPow G N)).Adj (x₁, x₂) (y₁, y₂) := by
  constructor
  · intro h
    rcases h with ⟨hcoord, i0, hi0⟩
    have hleft : x₁ = y₁ ∨ (strongPow G M).Adj x₁ y₁ := by
      by_cases hEq : x₁ = y₁
      · exact Or.inl hEq
      · refine Or.inr ?_
        have : ∃ i : Fin M, G.Adj (x₁ i) (y₁ i) := by
          by_contra hnone
          apply hEq
          funext i
          rcases hcoord (Fin.castAdd N i) with hEqi | hAdji
          · simpa [concatPowVertex] using hEqi
          · exfalso
            exact hnone ⟨i, by simpa [concatPowVertex] using hAdji⟩
        rcases this with ⟨i, hi⟩
        refine ⟨?_, i, hi⟩
        intro j
        simpa [concatPowVertex] using hcoord (Fin.castAdd N j)
    have hright : x₂ = y₂ ∨ (strongPow G N).Adj x₂ y₂ := by
      by_cases hEq : x₂ = y₂
      · exact Or.inl hEq
      · refine Or.inr ?_
        have : ∃ j : Fin N, G.Adj (x₂ j) (y₂ j) := by
          by_contra hnone
          apply hEq
          funext j
          rcases hcoord (Fin.natAdd M j) with hEqj | hAdjj
          · simpa [concatPowVertex] using hEqj
          · exfalso
            exact hnone ⟨j, by simpa [concatPowVertex] using hAdjj⟩
        rcases this with ⟨j, hj⟩
        refine ⟨?_, j, hj⟩
        intro k
        simpa [concatPowVertex] using hcoord (Fin.natAdd M k)
    have hsome : (strongPow G M).Adj x₁ y₁ ∨ (strongPow G N).Adj x₂ y₂ := by
      rcases hleft with hEq₁ | hAdj₁
      · rcases hright with hEq₂ | hAdj₂
        · exfalso
          have hxy : concatPowVertex x₁ x₂ = concatPowVertex y₁ y₂ := by
            funext k
            refine Fin.addCases ?_ ?_ k
            · intro i
              simpa [concatPowVertex] using congrFun hEq₁ i
            · intro j
              simpa [concatPowVertex] using congrFun hEq₂ j
          apply G.loopless (concatPowVertex x₁ x₂ i0)
          simpa [hxy] using hi0
        · exact Or.inr hAdj₂
      · exact Or.inl hAdj₁
    exact ⟨hleft, hright, hsome⟩
  · intro h
    rcases h with ⟨hleft, hright, hsome⟩
    refine ⟨?_, ?_⟩
    · intro k
      refine Fin.addCases ?_ ?_ k
      · intro i
        rcases hleft with hEq | hAdj
        · exact Or.inl (by simpa [concatPowVertex] using congrFun hEq i)
        · rcases hAdj with ⟨hcoord, _, _⟩
          simpa [concatPowVertex] using hcoord i
      · intro j
        rcases hright with hEq | hAdj
        · exact Or.inl (by simpa [concatPowVertex] using congrFun hEq j)
        · rcases hAdj with ⟨hcoord, _, _⟩
          simpa [concatPowVertex] using hcoord j
    · rcases hsome with hAdj | hAdj
      · rcases hAdj with ⟨_, i, hi⟩
        exact ⟨Fin.castAdd N i, by simpa [concatPowVertex] using hi⟩
      · rcases hAdj with ⟨_, j, hj⟩
        exact ⟨Fin.natAdd M j, by simpa [concatPowVertex] using hj⟩

def strongPow_addIsoStrongProd
    {α : Type} (G : SimpleGraph α) (M N : Nat) :
    strongPow G (M + N) ≃g strongProd (strongPow G M) (strongPow G N) :=
  let e : (Fin (M + N) → α) ≃ ((Fin M → α) × (Fin N → α)) :=
    { toFun := fun x => ((fun i => x (Fin.castAdd N i)), (fun j => x (Fin.natAdd M j)))
      invFun := fun p => concatPowVertex p.1 p.2
      left_inv := by
        intro x
        funext k
        refine Fin.addCases ?_ ?_ k
        · intro i
          simp [concatPowVertex]
        · intro j
          simp [concatPowVertex]
      right_inv := by
        intro p
        rcases p with ⟨x, y⟩
        simp [concatPowVertex] }
  ⟨e, by
    intro x y
    let x₁ : Fin M → α := fun i => x (Fin.castAdd N i)
    let x₂ : Fin N → α := fun j => x (Fin.natAdd M j)
    let y₁ : Fin M → α := fun i => y (Fin.castAdd N i)
    let y₂ : Fin N → α := fun j => y (Fin.natAdd M j)
    have hx : concatPowVertex x₁ x₂ = x := by
      funext k
      refine Fin.addCases ?_ ?_ k
      · intro i
        simp [x₁, concatPowVertex]
      · intro j
        simp [x₂, concatPowVertex]
    have hy : concatPowVertex y₁ y₂ = y := by
      funext k
      refine Fin.addCases ?_ ?_ k
      · intro i
        simp [y₁, concatPowVertex]
      · intro j
        simp [y₂, concatPowVertex]
    change (strongProd (strongPow G M) (strongPow G N)).Adj (x₁, x₂) (y₁, y₂) ↔
      (strongPow G (M + N)).Adj x y
    simpa [e, x₁, x₂, y₁, y₂, hx, hy] using
      (strongPowAdj_add_iff_strongProd (G := G) (x₁ := x₁) (x₂ := x₂) (y₁ := y₁) (y₂ := y₂)).symm ⟩

theorem graphMaxIndependentCard_strongPow_add_eq_strongProd
    {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (M N : Nat) :
    graphMaxIndependentCard (strongPow G (M + N)) =
      graphMaxIndependentCard (strongProd (strongPow G M) (strongPow G N)) := by
  exact graphMaxIndependentCard_iso_eq (strongPow_addIsoStrongProd G M N)

structure ThetaWitnessPackage {V : Type} (G : SimpleGraph V) where
  β : Type
  instFintypeβ : Fintype β
  instDecidableEqβ : DecidableEq β
  witness : @ThetaWitness V β instFintypeβ instDecidableEqβ G

attribute [instance] ThetaWitnessPackage.instFintypeβ ThetaWitnessPackage.instDecidableEqβ

noncomputable def ThetaWitnessPackage.reindex
    {V W : Type} {G : SimpleGraph V} {H : SimpleGraph W}
    (e : H ≃g G)
    (P : ThetaWitnessPackage G) :
    ThetaWitnessPackage H where
  β := P.β
  instFintypeβ := P.instFintypeβ
  instDecidableEqβ := P.instDecidableEqβ
  witness := ThetaWitness.reindex e P.witness

noncomputable def ThetaWitnessPackage.prod
    {V W : Type} {G : SimpleGraph V} {H : SimpleGraph W}
    (PG : ThetaWitnessPackage G)
    (PH : ThetaWitnessPackage H) :
    ThetaWitnessPackage (strongProd G H) where
  β := PG.β × PH.β
  instFintypeβ := inferInstance
  instDecidableEqβ := inferInstance
  witness := @ThetaWitness.prod V W PG.β PH.β
    PG.instFintypeβ PG.instDecidableEqβ PH.instFintypeβ PH.instDecidableEqβ G H
    PG.witness PH.witness

def strongPow_oneIso
    {α : Type} (G : SimpleGraph α) :
    strongPow G 1 ≃g G :=
  ⟨{ toFun := fun x => x 0
     invFun := fun a _ => a
     left_inv := by
       intro x
       funext i
       fin_cases i
       rfl
     right_inv := by
       intro a
       rfl }, by
    intro x y
    constructor
    · intro h
      exact (strongPow_adj_iff).2 ⟨by
        intro i
        fin_cases i
        exact Or.inr h, ⟨0, by simpa using h⟩⟩
    · intro h
      rcases (strongPow_adj_iff.mp h) with ⟨_, i, hi⟩
      fin_cases i
      simpa using hi ⟩

noncomputable def thetaWitnessZeroPackage
    {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) :
    ThetaWitnessPackage (strongPow G 0) where
  β := Fin 1
  instFintypeβ := inferInstance
  instDecidableEqβ := inferInstance
  witness :=
    { repr := fun _ => EuclideanSpace.single 0 (1 : ℝ)
      handle := EuclideanSpace.single 0 (1 : ℝ)
      repr_unit := by
        intro x
        simp
      handle_unit := by simp
      ortho_of_nonadj := by
        intro x y hneq _
        exfalso
        apply hneq
        funext i
        exact Fin.elim0 i
      bound := 1
      one_le_bound := le_rfl
      one_le_bound_inner_sq := by
        intro x
        simp }

noncomputable def thetaWitnessOnePackage
    {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]
    {G : SimpleGraph α}
    (W : ThetaWitness (β := β) G) :
    ThetaWitnessPackage (strongPow G 1) where
  β := β
  instFintypeβ := inferInstance
  instDecidableEqβ := inferInstance
  witness := ThetaWitness.reindex (strongPow_oneIso G) W

noncomputable def thetaWitnessPowPackage
    {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]
    {G : SimpleGraph α}
    (W : ThetaWitness (β := β) G) :
    (N : Nat) → ThetaWitnessPackage (strongPow G N)
  | 0 => thetaWitnessZeroPackage G
  | Nat.succ n =>
      let Pn := thetaWitnessPowPackage W n
      let P1 := thetaWitnessOnePackage W
      (ThetaWitnessPackage.prod Pn P1).reindex (strongPow_addIsoStrongProd G n 1)

theorem thetaWitnessPowPackage_bound
    {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]
    {G : SimpleGraph α}
    (W : ThetaWitness (β := β) G) :
    ∀ N : Nat, (thetaWitnessPowPackage W N).witness.bound = W.bound ^ N
  | 0 => by simp [thetaWitnessPowPackage, thetaWitnessZeroPackage]
  | Nat.succ n => by
      simp [thetaWitnessPowPackage, thetaWitnessPowPackage_bound W n, thetaWitnessOnePackage,
        ThetaWitnessPackage.reindex, ThetaWitnessPackage.prod, ThetaWitness.reindex,
        ThetaWitness.prod, thetaWitnessZeroPackage, pow_succ, mul_assoc, mul_left_comm, mul_comm]

theorem graphMaxIndependentCard_le_thetaWitness_real
    {V β : Type} [Fintype V] [DecidableEq V] [Fintype β] [DecidableEq β]
    {G : SimpleGraph V}
    (W : ThetaWitness (β := β) G) :
    (graphMaxIndependentCard G : ℝ) ≤ W.bound := by
  rcases exists_graphIndependent_card_eq_graphMaxIndependentCard G with ⟨s, hs, hcard⟩
  calc
    (graphMaxIndependentCard G : ℝ) = (s.card : ℝ) := by exact_mod_cast hcard.symm
    _ ≤ W.bound := graphIndependent_card_le_thetaWitness W s hs

theorem graphMaxIndependentCard_strongPow_le_thetaWitnessPow
    {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]
    {G : SimpleGraph α}
    (W : ThetaWitness (β := β) G) (N : Nat) :
    (graphMaxIndependentCard (strongPow G N) : ℝ) ≤ W.bound ^ N := by
  have hbase := graphMaxIndependentCard_le_thetaWitness_real ((thetaWitnessPowPackage W N).witness)
  simpa [thetaWitnessPowPackage_bound W N] using hbase

theorem graphMaxIndependentCard_strongPow_pos
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) (N : Nat) :
    0 < graphMaxIndependentCard (strongPow G N) := by
  let x : Fin N → α := fun _ => Classical.choice ‹Nonempty α›
  have hs : GraphIndependent (strongPow G N) ({x} : Finset (Fin N → α)) := by
    intro u v hu hv hneq hAdj
    simp at hu hv
    subst hu
    subst hv
    exact hneq rfl
  have hbound : ({x} : Finset (Fin N → α)).card ≤ graphMaxIndependentCard (strongPow G N) := by
    exact graphIndependent_card_le_graphMaxIndependentCard (strongPow G N) ({x} : Finset (Fin N → α)) hs
  simpa [x] using hbound

theorem graphIndependent_product_of_independent
    {α β : Type} {G : SimpleGraph α} {H : SimpleGraph β}
    {s : Finset α} {t : Finset β}
    (hs : GraphIndependent G s)
    (ht : GraphIndependent H t) :
    GraphIndependent (strongProd G H) (s.product t) := by
  intro x y hx hy hneq hAdj
  rcases Finset.mem_product.mp hx with ⟨hx₁, hx₂⟩
  rcases Finset.mem_product.mp hy with ⟨hy₁, hy₂⟩
  rcases hAdj with ⟨h1, h2, h3⟩
  by_cases hEq₁ : x.1 = y.1
  · have hneq₂ : x.2 ≠ y.2 := by
      intro hEq₂
      exact hneq (Prod.ext hEq₁ hEq₂)
    rcases h2 with hEq₂ | hAdj₂
    · exact hneq₂ hEq₂
    · exact ht hx₂ hy₂ hneq₂ hAdj₂
  · rcases h1 with hEq₁' | hAdj₁
    · exact hEq₁ hEq₁'
    · exact hs hx₁ hy₁ hEq₁ hAdj₁

theorem graphMaxIndependentCard_strongProd_lower_bound
    {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β]
    (G : SimpleGraph α) (H : SimpleGraph β) :
    graphMaxIndependentCard G * graphMaxIndependentCard H ≤
      graphMaxIndependentCard (strongProd G H) := by
  rcases exists_graphIndependent_card_eq_graphMaxIndependentCard G with ⟨s, hs, hcardS⟩
  rcases exists_graphIndependent_card_eq_graphMaxIndependentCard H with ⟨t, ht, hcardT⟩
  have hprod : GraphIndependent (strongProd G H) (s.product t) :=
    graphIndependent_product_of_independent hs ht
  have hbound :
      (s.product t).card ≤ graphMaxIndependentCard (strongProd G H) := by
    exact graphIndependent_card_le_graphMaxIndependentCard (strongProd G H) _ hprod
  simpa [hcardS, hcardT, Finset.card_product] using hbound

theorem graphMaxIndependentCard_strongPow_add_lower_bound
    {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (M N : Nat) :
    graphMaxIndependentCard (strongPow G M) *
        graphMaxIndependentCard (strongPow G N) ≤
      graphMaxIndependentCard (strongPow G (M + N)) := by
  calc
    graphMaxIndependentCard (strongPow G M) * graphMaxIndependentCard (strongPow G N)
      ≤ graphMaxIndependentCard (strongProd (strongPow G M) (strongPow G N)) :=
        graphMaxIndependentCard_strongProd_lower_bound (strongPow G M) (strongPow G N)
    _ = graphMaxIndependentCard (strongPow G (M + N)) := by
      symm
      exact graphMaxIndependentCard_strongPow_add_eq_strongProd G M N

/-! ## Block-composed view families and multiplicative budgets -/

/-- Product latent state for two fact blocks over a common alphabet. -/
abbrev PairState (F₁ F₂ A : Nat) := State F₁ A × State F₂ A

/-- Product partial observation records the observed data on each block separately. -/
abbrev PairObservation (F₁ F₂ A : Nat) := PartialObservation F₁ A × PartialObservation F₂ A

/-- Simultaneous observation of one location from each block. -/
def pairObserve {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type}
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (ℓ : Loc₁ × Loc₂)
    (x : PairState F₁ F₂ A) : PairObservation F₁ F₂ A :=
  (observe (views₁ ℓ.1) x.1, observe (views₂ ℓ.2) x.2)

/-- Confusability in the block-composed observation family. -/
def PairConfusable {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type}
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (x y : PairState F₁ F₂ A) : Prop :=
  x ≠ y ∧ ∃ ℓ : Loc₁ × Loc₂, pairObserve views₁ views₂ ℓ x = pairObserve views₁ views₂ ℓ y

instance instDecidablePairConfusable {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type}
    [Fintype Loc₁] [Fintype Loc₂] [DecidableEq Loc₁] [DecidableEq Loc₂]
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (x y : PairState F₁ F₂ A) : Decidable (PairConfusable views₁ views₂ x y) := by
  unfold PairConfusable
  infer_instance

/-- Confusability graph of the block-composed observation family. -/
def pairConfusabilityGraph {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type}
    [Fintype Loc₁] [Fintype Loc₂] [DecidableEq Loc₁] [DecidableEq Loc₂]
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂)) : SimpleGraph (PairState F₁ F₂ A) where
  Adj := PairConfusable views₁ views₂
  symm := by
    intro x y h
    rcases h with ⟨hxy, ℓ, hobs⟩
    exact ⟨hxy.symm, ℓ, hobs.symm⟩
  loopless := by
    intro x h
    exact h.1 rfl

theorem pairConfusable_iff_strongProdAdj
    {F₁ F₂ A L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (hL₁ : 0 < L₁)
    (hL₂ : 0 < L₂)
    {x y : PairState F₁ F₂ A} :
    PairConfusable views₁ views₂ x y ↔
      (strongProd (confusabilityGraph (A := A) views₁)
        (confusabilityGraph (A := A) views₂)).Adj x y := by
  let d₁ : Fin L₁ := ⟨0, hL₁⟩
  let d₂ : Fin L₂ := ⟨0, hL₂⟩
  constructor
  · rintro ⟨hxy, ⟨ℓ₁, ℓ₂⟩, hobs⟩
    rcases x with ⟨x₁, x₂⟩
    rcases y with ⟨y₁, y₂⟩
    dsimp [pairObserve] at hobs
    rcases Prod.mk.inj hobs with ⟨hobs₁, hobs₂⟩
    have hleft : x₁ = y₁ ∨ Confusable views₁ x₁ y₁ := by
      by_cases hEq : x₁ = y₁
      · exact Or.inl hEq
      · exact Or.inr ⟨hEq, ⟨ℓ₁, hobs₁⟩⟩
    have hright : x₂ = y₂ ∨ Confusable views₂ x₂ y₂ := by
      by_cases hEq : x₂ = y₂
      · exact Or.inl hEq
      · exact Or.inr ⟨hEq, ⟨ℓ₂, hobs₂⟩⟩
    have hsome : Confusable views₁ x₁ y₁ ∨ Confusable views₂ x₂ y₂ := by
      by_cases hEq1 : x₁ = y₁
      · right
        have hneq₂ : x₂ ≠ y₂ := by
          intro hEq2
          apply hxy
          cases hEq1
          cases hEq2
          rfl
        exact ⟨hneq₂, ⟨ℓ₂, hobs₂⟩⟩
      · left
        exact ⟨hEq1, ⟨ℓ₁, hobs₁⟩⟩
    exact ⟨hleft, hright, hsome⟩
  · intro h
    rcases x with ⟨x₁, x₂⟩
    rcases y with ⟨y₁, y₂⟩
    rcases h with ⟨hleft, hright, hsome⟩
    have hneqpair : (x₁, x₂) ≠ (y₁, y₂) := by
      intro hEq
      apply hsome.elim
      · intro hAdj
        exact hAdj.1 (congrArg Prod.fst hEq)
      · intro hAdj
        exact hAdj.1 (congrArg Prod.snd hEq)
    rcases hleft with hEq₁ | hAdj₁
    · rcases hright with hEq₂ | hAdj₂
      · exfalso
        exact hsome.elim (fun hAdj => hAdj.1 hEq₁) (fun hAdj => hAdj.1 hEq₂)
      · rcases hAdj₂ with ⟨_, ⟨ℓ₂, hobs₂⟩⟩
        refine ⟨hneqpair, ⟨(d₁, ℓ₂), ?_⟩⟩
        dsimp [pairObserve]
        exact Prod.ext (congrArg (observe (views₁ d₁)) hEq₁) hobs₂
    · rcases hright with hEq₂ | hAdj₂
      · rcases hAdj₁ with ⟨_, ⟨ℓ₁, hobs₁⟩⟩
        refine ⟨hneqpair, ⟨(ℓ₁, d₂), ?_⟩⟩
        dsimp [pairObserve]
        exact Prod.ext hobs₁ (congrArg (observe (views₂ d₂)) hEq₂)
      · rcases hAdj₁ with ⟨_, ⟨ℓ₁, hobs₁⟩⟩
        rcases hAdj₂ with ⟨_, ⟨ℓ₂, hobs₂⟩⟩
        refine ⟨hneqpair, ⟨(ℓ₁, ℓ₂), ?_⟩⟩
        dsimp [pairObserve]
        exact Prod.ext hobs₁ hobs₂

theorem pairConfusabilityGraph_eq_strongProd
    {F₁ F₂ A L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (hL₁ : 0 < L₁)
    (hL₂ : 0 < L₂) :
    pairConfusabilityGraph views₁ views₂ =
      strongProd (confusabilityGraph (A := A) views₁) (confusabilityGraph (A := A) views₂) := by
  ext x y
  exact pairConfusable_iff_strongProdAdj views₁ views₂ hL₁ hL₂

/-- Proper coloring for the block-composed confusability relation. -/
def PairProperColoring {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type} {T : Nat}
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (tag : PairState F₁ F₂ A → Fin T) : Prop :=
  ∀ ⦃x y : PairState F₁ F₂ A⦄, PairConfusable views₁ views₂ x y → tag x ≠ tag y

/-- Encode a pair of finite colors as one color in the product alphabet. -/
noncomputable def pairColorCode {T₁ T₂ : Nat} : Fin T₁ × Fin T₂ → Fin (T₁ * T₂) :=
  fun p => finCongr (by simp) (Fintype.equivFin (Fin T₁ × Fin T₂) p)

theorem pairColorCode_injective {T₁ T₂ : Nat} :
    Function.Injective (@pairColorCode T₁ T₂) := by
  intro a b h
  have h' :
      Fintype.equivFin (Fin T₁ × Fin T₂) a = Fintype.equivFin (Fin T₁ × Fin T₂) b := by
    exact (finCongr (by simp)).injective h
  exact (Fintype.equivFin (Fin T₁ × Fin T₂)).injective h'

/-- Product coloring from component colorings. -/
noncomputable def pairTag {F₁ F₂ A T₁ T₂ : Nat}
    (tag₁ : State F₁ A → Fin T₁)
    (tag₂ : State F₂ A → Fin T₂) :
    PairState F₁ F₂ A → Fin (T₁ * T₂) :=
  fun x => pairColorCode (tag₁ x.1, tag₂ x.2)

/--
If each block admits a proper coloring, then the simultaneously observed product system admits
a proper coloring in the product alphabet. This is the multiplicative zero-error budget law for
block-composed views.
-/
theorem pairProperColoring_of_componentColorings
    {F₁ F₂ A L₁ L₂ T₁ T₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (tag₁ : State F₁ A → Fin T₁)
    (tag₂ : State F₂ A → Fin T₂)
    (hcolor₁ : ProperColoring views₁ tag₁)
    (hcolor₂ : ProperColoring views₂ tag₂) :
    PairProperColoring views₁ views₂ (pairTag tag₁ tag₂) := by
  intro x y hconf htag
  rcases hconf with ⟨hxy, ⟨ℓ₁, ℓ₂⟩, hobs⟩
  have hpair : (tag₁ x.1, tag₂ x.2) = (tag₁ y.1, tag₂ y.2) := by
    exact pairColorCode_injective htag
  have htag₁ : tag₁ x.1 = tag₁ y.1 := congrArg Prod.fst hpair
  have htag₂ : tag₂ x.2 = tag₂ y.2 := congrArg Prod.snd hpair
  rcases x with ⟨x₁, x₂⟩
  rcases y with ⟨y₁, y₂⟩
  dsimp [pairObserve] at hobs
  rcases Prod.mk.inj hobs with ⟨hobs₁, hobs₂⟩
  by_cases hleft : x₁ = y₁
  · have hright : x₂ ≠ y₂ := by
      intro hEq
      apply hxy
      cases hleft
      cases hEq
      rfl
    have hconf₂ : Confusable views₂ x₂ y₂ := ⟨hright, ⟨ℓ₂, hobs₂⟩⟩
    exact (hcolor₂ hconf₂) htag₂
  · have hconf₁ : Confusable views₁ x₁ y₁ := ⟨hleft, ⟨ℓ₁, hobs₁⟩⟩
    exact (hcolor₁ hconf₁) htag₁

theorem pairColorable_of_colorable
    {F₁ F₂ A L₁ L₂ T₁ T₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (hcolor₁ : Colorable (F := F₁) (A := A) (L := L₁) (T := T₁) views₁)
    (hcolor₂ : Colorable (F := F₂) (A := A) (L := L₂) (T := T₂) views₂) :
    ∃ tag : PairState F₁ F₂ A → Fin (T₁ * T₂),
      PairProperColoring views₁ views₂ tag := by
  rcases hcolor₁ with ⟨tag₁, htag₁⟩
  rcases hcolor₂ with ⟨tag₂, htag₂⟩
  exact ⟨pairTag tag₁ tag₂, pairProperColoring_of_componentColorings views₁ views₂ tag₁ tag₂ htag₁ htag₂⟩

/-- A finite support set is a clique when every distinct pair in it is confusable. -/
def IsClique {F A L : Nat} (views : ViewFamily F L) (s : Finset (State F A)) : Prop :=
  ∀ ⦃x y : State F A⦄, x ∈ s → y ∈ s → x ≠ y → Confusable views x y

/-- A clique in the block-composed confusability graph. -/
def PairIsClique {F₁ F₂ A : Nat} {L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (s : Finset (PairState F₁ F₂ A)) : Prop :=
  ∀ ⦃x y : PairState F₁ F₂ A⦄, x ∈ s → y ∈ s → x ≠ y → PairConfusable views₁ views₂ x y

theorem pairClique_card_le_tag_alphabet
    {F₁ F₂ A L₁ L₂ T : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (tag : PairState F₁ F₂ A → Fin T)
    (hcolor : PairProperColoring views₁ views₂ tag)
    {s : Finset (PairState F₁ F₂ A)}
    (hs : PairIsClique views₁ views₂ s) :
    s.card ≤ T := by
  classical
  have hinj : Set.InjOn tag {x | x ∈ s} := by
    intro x hx y hy htag
    by_cases hEq : x = y
    · exact hEq
    · exfalso
      exact (hcolor (hs hx hy hEq)) htag
  have hcard : s.card = (s.image tag).card := by
    symm
    exact Finset.card_image_of_injOn (by
      intro a ha b hb hab
      exact hinj ha hb hab)
  calc
    s.card = (s.image tag).card := hcard
    _ ≤ (Finset.univ : Finset (Fin T)).card := Finset.card_le_card (by
      intro t ht
      simp)
    _ = T := by simp

theorem pair_product_clique_of_cliques
    {F₁ F₂ A L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (hL₁ : 0 < L₁)
    (hL₂ : 0 < L₂)
    {s₁ : Finset (State F₁ A)}
    {s₂ : Finset (State F₂ A)}
    (h₁ : IsClique views₁ s₁)
    (h₂ : IsClique views₂ s₂) :
    PairIsClique views₁ views₂ (s₁.product s₂) := by
  intro x y hx hy hxy
  rcases Finset.mem_product.mp hx with ⟨hx₁, hx₂⟩
  rcases Finset.mem_product.mp hy with ⟨hy₁, hy₂⟩
  rcases x with ⟨x₁, x₂⟩
  rcases y with ⟨y₁, y₂⟩
  by_cases hleft : x₁ = y₁
  · have hright : x₂ ≠ y₂ := by
      intro hEq
      apply hxy
      cases hleft
      cases hEq
      rfl
    refine ⟨hxy, ?_⟩
    have hconf₂ : Confusable views₂ x₂ y₂ := h₂ hx₂ hy₂ hright
    rcases hconf₂ with ⟨_, ℓ₂, hobs₂⟩
    refine ⟨(⟨0, hL₁⟩, ℓ₂), ?_⟩
    dsimp [pairObserve]
    cases hleft
    simp [hobs₂]
  · refine ⟨hxy, ?_⟩
    by_cases hright : x₂ = y₂
    · have hconf₁ : Confusable views₁ x₁ y₁ := h₁ hx₁ hy₁ hleft
      rcases hconf₁ with ⟨_, ℓ₁, hobs₁⟩
      refine ⟨(ℓ₁, ⟨0, hL₂⟩), ?_⟩
      dsimp [pairObserve]
      cases hright
      simp [hobs₁]
    · have hconf₁ : Confusable views₁ x₁ y₁ := h₁ hx₁ hy₁ hleft
      have hconf₂ : Confusable views₂ x₂ y₂ := h₂ hx₂ hy₂ hright
      rcases hconf₁ with ⟨_, ℓ₁, hobs₁⟩
      rcases hconf₂ with ⟨_, ℓ₂, hobs₂⟩
      refine ⟨(ℓ₁, ℓ₂), ?_⟩
      dsimp [pairObserve]
      simp [hobs₁, hobs₂]

theorem pair_product_clique_budget_lower_bound
    {F₁ F₂ A L₁ L₂ T : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (hL₁ : 0 < L₁)
    (hL₂ : 0 < L₂)
    (tag : PairState F₁ F₂ A → Fin T)
    (hcolor : PairProperColoring views₁ views₂ tag)
    {s₁ : Finset (State F₁ A)}
    {s₂ : Finset (State F₂ A)}
    (h₁ : IsClique views₁ s₁)
    (h₂ : IsClique views₂ s₂) :
    s₁.card * s₂.card ≤ T := by
  have hprod : PairIsClique views₁ views₂ (s₁.product s₂) :=
    pair_product_clique_of_cliques views₁ views₂ hL₁ hL₂ h₁ h₂
  have hbound : (s₁.product s₂).card ≤ T :=
    pairClique_card_le_tag_alphabet views₁ views₂ tag hcolor hprod
  simpa using hbound

/-- Exact recovery restricted to a designated finite success set. -/
def ExactOn {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A))
    (s : Finset (State F A)) : Prop :=
  ∀ ⦃x : State F A⦄, x ∈ s → ∀ ℓ, decode ℓ (observe (views ℓ) x) (tag x) = some x

/-- Exactness on a success set forces tag separation on confusable pairs inside that set. -/
theorem exactOn_implies_properColoringOn
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A))
    {s : Finset (State F A)}
    (hexact : ExactOn views tag decode s) :
    ∀ ⦃x y : State F A⦄, x ∈ s → y ∈ s → Confusable views x y → tag x ≠ tag y := by
  intro x y hx hy hconf htag
  rcases hconf with ⟨hxy, ℓ, hobs⟩
  have hx' : decode ℓ (observe (views ℓ) x) (tag x) = some x := hexact hx ℓ
  have hy' : decode ℓ (observe (views ℓ) y) (tag y) = some y := hexact hy ℓ
  have hy'' : decode ℓ (observe (views ℓ) x) (tag x) = some y := by
    simpa [hobs, htag] using hy'
  have : some x = some y := hx'.symm.trans hy''
  exact hxy (Option.some.inj this)

/-- Proper coloring restricted to a designated success set. -/
def ProperColoringOn {F A L T : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A))
    (tag : State F A → Fin T) : Prop :=
  ∀ ⦃x y : State F A⦄, x ∈ s → y ∈ s → Confusable views x y → tag x ≠ tag y

/-- Decidability for restricted proper colorings. -/
instance instDecidableProperColoringOn {F A L T : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A))
    (tag : State F A → Fin T) : Decidable (ProperColoringOn views s tag) := by
  unfold ProperColoringOn
  infer_instance

/-- `T`-colorability of the induced confusability subgraph on a success set. -/
def ColorableOn {F A L T : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A)) : Prop :=
  ∃ tag : State F A → Fin T, ProperColoringOn views s tag

instance instDecidableColorableOn {F A L T : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A)) : Decidable (ColorableOn (F := F) (A := A) (L := L) (T := T) views s) := by
  unfold ColorableOn
  infer_instance

theorem exactOn_implies_colorableOn
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A))
    (s : Finset (State F A))
    (hexact : ExactOn views tag decode s) :
    ColorableOn (F := F) (A := A) (L := L) (T := T) views s := by
  refine ⟨tag, ?_⟩
  intro x y hx hy hconf
  exact exactOn_implies_properColoringOn views tag decode hexact hx hy hconf

theorem colorableOn_implies_exactOn
    {F A L T : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A))
    (tag : State F A → Fin T)
    (hcolor : ProperColoringOn views s tag) :
    ∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
      ExactOn views tag decode s := by
  classical
  refine ⟨fun ℓ o t =>
    if h : ∃ x : State F A, x ∈ s ∧ observe (views ℓ) x = o ∧ tag x = t
    then some (Classical.choose h)
    else none, ?_⟩
  intro x hx ℓ
  have hex : ∃ y : State F A, y ∈ s ∧ observe (views ℓ) y = observe (views ℓ) x ∧ tag y = tag x := by
    exact ⟨x, hx, rfl, rfl⟩
  change (if h : ∃ y : State F A, y ∈ s ∧ observe (views ℓ) y = observe (views ℓ) x ∧ tag y = tag x
    then some (Classical.choose h)
    else none) = some x
  rw [dif_pos hex]
  have hchoose : Classical.choose hex = x := by
    rcases Classical.choose_spec hex with ⟨hy, hobs, htag⟩
    by_cases hEq : Classical.choose hex = x
    · exact hEq
    · exfalso
      have hconf : Confusable views (Classical.choose hex) x := ⟨hEq, ⟨ℓ, hobs⟩⟩
      exact (hcolor hy hx hconf) htag
  simp [hchoose]

theorem exists_exactOn_iff_colorableOn
    {F A L T : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A)) :
    (∃ tag : State F A → Fin T,
        ∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
          ExactOn views tag decode s) ↔
      ColorableOn (F := F) (A := A) (L := L) (T := T) views s := by
  constructor
  · rintro ⟨tag, decode, hexact⟩
    exact exactOn_implies_colorableOn views tag decode s hexact
  · rintro ⟨tag, hcolor⟩
    rcases colorableOn_implies_exactOn views s tag hcolor with ⟨decode, hexact⟩
    exact ⟨tag, decode, hexact⟩

theorem empty_colorableOn
    {F A L T : Nat}
    (views : ViewFamily F L)
    (hT : 0 < T) :
    ColorableOn (F := F) (A := A) (L := L) (T := T) views ∅ := by
  haveI : NeZero T := ⟨Nat.ne_of_gt hT⟩
  refine ⟨fun _ => 0, ?_⟩
  intro x y hx
  simpa using hx

/-- All success sets whose induced confusability subgraph is `T`-colorable. -/
def colorableSubsets {F A L T : Nat}
    (views : ViewFamily F L) : Finset (Finset (State F A)) :=
  (Finset.univ.powerset.filter fun s =>
    ColorableOn (F := F) (A := A) (L := L) (T := T) views s)

/-- Maximum size of a success set that can be decoded exactly with `T` tags. -/
noncomputable def maxColorableCard {F A L T : Nat}
    (views : ViewFamily F L)
    (hT : 0 < T) : Nat :=
  ((colorableSubsets (F := F) (A := A) (L := L) (T := T) views).image Finset.card).max'
    (by
      refine ⟨0, ?_⟩
      refine Finset.mem_image.mpr ?_
      refine ⟨∅, ?_, by simp⟩
      simp [colorableSubsets, empty_colorableOn views hT])

theorem colorableOn_card_le_maxColorableCard
    {F A L T : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A))
    (hT : 0 < T)
    (hcolor : ColorableOn (F := F) (A := A) (L := L) (T := T) views s) :
    s.card ≤ maxColorableCard (F := F) (A := A) (L := L) (T := T) views hT := by
  have hmem : s.card ∈
      (colorableSubsets (F := F) (A := A) (L := L) (T := T) views).image Finset.card := by
    refine Finset.mem_image.mpr ?_
    refine ⟨s, ?_, rfl⟩
    simp [colorableSubsets, hcolor]
  exact Finset.le_max' _ _ hmem

theorem exactOn_card_le_maxColorableCard
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A))
    (s : Finset (State F A))
    (hT : 0 < T)
    (hexact : ExactOn views tag decode s) :
    s.card ≤ maxColorableCard (F := F) (A := A) (L := L) (T := T) views hT := by
  exact colorableOn_card_le_maxColorableCard views s hT
    (exactOn_implies_colorableOn views tag decode s hexact)

theorem exists_colorableOn_card_eq_maxColorableCard
    {F A L T : Nat}
    (views : ViewFamily F L)
    (hT : 0 < T) :
    ∃ s : Finset (State F A),
      ColorableOn (F := F) (A := A) (L := L) (T := T) views s ∧
      s.card = maxColorableCard (F := F) (A := A) (L := L) (T := T) views hT := by
  have hmem :
      maxColorableCard (F := F) (A := A) (L := L) (T := T) views hT ∈
        (colorableSubsets (F := F) (A := A) (L := L) (T := T) views).image Finset.card := by
    exact Finset.max'_mem _ _
  rcases Finset.mem_image.mp hmem with ⟨s, hs, hcard⟩
  exact ⟨s, by simpa [colorableSubsets] using hs, hcard⟩

theorem exists_exactOn_card_eq_maxColorableCard
    {F A L T : Nat}
    (views : ViewFamily F L)
    (hT : 0 < T) :
    ∃ s : Finset (State F A),
      ∃ tag : State F A → Fin T,
        ∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
          ExactOn views tag decode s ∧
          s.card = maxColorableCard (F := F) (A := A) (L := L) (T := T) views hT := by
  rcases exists_colorableOn_card_eq_maxColorableCard (F := F) (A := A) (L := L) (T := T) views hT
    with ⟨s, hs, hcard⟩
  rcases hs with ⟨tag, htag⟩
  rcases colorableOn_implies_exactOn views s tag htag with ⟨decode, hexact⟩
  exact ⟨s, tag, decode, hexact, hcard⟩

/-- Weighted mass of a success set under an arbitrary finite source weight. -/
def subsetMass {F A : Nat}
    (w : State F A → ℝ)
    (s : Finset (State F A)) : ℝ :=
  s.sum w

/-- Maximum weighted success mass among exactly decodable `T`-colorable success sets. -/
noncomputable def maxColorableMass {F A L T : Nat}
    (views : ViewFamily F L)
    (hT : 0 < T)
    (w : State F A → ℝ) : ℝ :=
  ((colorableSubsets (F := F) (A := A) (L := L) (T := T) views).image (subsetMass w)).max'
    (by
      refine ⟨0, ?_⟩
      refine Finset.mem_image.mpr ?_
      refine ⟨∅, ?_, by simp [subsetMass]⟩
      simp [colorableSubsets, empty_colorableOn views hT])

theorem colorableOn_mass_le_maxColorableMass
    {F A L T : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A))
    (hT : 0 < T)
    (w : State F A → ℝ)
    (hcolor : ColorableOn (F := F) (A := A) (L := L) (T := T) views s) :
    subsetMass w s ≤ maxColorableMass (F := F) (A := A) (L := L) (T := T) views hT w := by
  have hmem : subsetMass w s ∈
      (colorableSubsets (F := F) (A := A) (L := L) (T := T) views).image (subsetMass w) := by
    refine Finset.mem_image.mpr ?_
    refine ⟨s, ?_, rfl⟩
    simp [colorableSubsets, hcolor]
  exact Finset.le_max' _ _ hmem

theorem exactOn_mass_le_maxColorableMass
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A))
    (s : Finset (State F A))
    (hT : 0 < T)
    (w : State F A → ℝ)
    (hexact : ExactOn views tag decode s) :
    subsetMass w s ≤ maxColorableMass (F := F) (A := A) (L := L) (T := T) views hT w := by
  exact colorableOn_mass_le_maxColorableMass views s hT w
    (exactOn_implies_colorableOn views tag decode s hexact)

theorem exists_exactOn_mass_eq_maxColorableMass
    {F A L T : Nat}
    (views : ViewFamily F L)
    (hT : 0 < T)
    (w : State F A → ℝ) :
    ∃ s : Finset (State F A),
      ∃ tag : State F A → Fin T,
        ∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
          ExactOn views tag decode s ∧
          subsetMass w s = maxColorableMass (F := F) (A := A) (L := L) (T := T) views hT w := by
  have hmem :
      maxColorableMass (F := F) (A := A) (L := L) (T := T) views hT w ∈
        (colorableSubsets (F := F) (A := A) (L := L) (T := T) views).image (subsetMass w) := by
    exact Finset.max'_mem _ _
  rcases Finset.mem_image.mp hmem with ⟨s, hs, hmass⟩
  have hs' : ColorableOn (F := F) (A := A) (L := L) (T := T) views s := by
    simpa [colorableSubsets] using hs
  rcases hs' with ⟨tag, htag⟩
  rcases colorableOn_implies_exactOn views s tag htag with ⟨decode, hexact⟩
  exact ⟨s, tag, decode, hexact, hmass⟩

/-- Exact finite weighted-success value for the multi-fact model at tag budget `T`. -/
noncomputable def exactRateDistortionValue {F A L T : Nat}
    (views : ViewFamily F L)
    (hT : 0 < T)
    (w : State F A → ℝ) : ℝ :=
  maxColorableMass (F := F) (A := A) (L := L) (T := T) views hT w

theorem exactRateDistortionValue_upper
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A))
    (s : Finset (State F A))
    (hT : 0 < T)
    (w : State F A → ℝ)
    (hexact : ExactOn views tag decode s) :
    subsetMass w s ≤ exactRateDistortionValue (F := F) (A := A) (L := L) (T := T) views hT w := by
  exact exactOn_mass_le_maxColorableMass views tag decode s hT w hexact

theorem exists_exactRateDistortionValue
    {F A L T : Nat}
    (views : ViewFamily F L)
    (hT : 0 < T)
    (w : State F A → ℝ) :
    ∃ s : Finset (State F A),
      ∃ tag : State F A → Fin T,
        ∃ decode : Fin L → PartialObservation F A → Fin T → Option (State F A),
          ExactOn views tag decode s ∧
          subsetMass w s = exactRateDistortionValue (F := F) (A := A) (L := L) (T := T) views hT w := by
  exact exists_exactOn_mass_eq_maxColorableMass views hT w

/-- Restricted proper coloring on a finite product support. -/
def PairProperColoringOn {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type} {T : Nat}
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (s : Finset (PairState F₁ F₂ A))
    (tag : PairState F₁ F₂ A → Fin T) : Prop :=
  ∀ ⦃x y : PairState F₁ F₂ A⦄, x ∈ s → y ∈ s → PairConfusable views₁ views₂ x y → tag x ≠ tag y

/-- Restricted colorability of a product confusability graph. -/
def PairColorableOn {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type} {T : Nat}
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (s : Finset (PairState F₁ F₂ A)) : Prop :=
  ∃ tag : PairState F₁ F₂ A → Fin T, PairProperColoringOn views₁ views₂ s tag

instance instDecidablePairProperColoringOn {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type} {T : Nat}
    [Fintype Loc₁] [Fintype Loc₂] [DecidableEq Loc₁] [DecidableEq Loc₂]
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (s : Finset (PairState F₁ F₂ A))
    (tag : PairState F₁ F₂ A → Fin T) : Decidable (PairProperColoringOn views₁ views₂ s tag) := by
  unfold PairProperColoringOn
  infer_instance

instance instDecidablePairColorableOn {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type} {T : Nat}
    [Fintype Loc₁] [Fintype Loc₂] [DecidableEq Loc₁] [DecidableEq Loc₂]
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (s : Finset (PairState F₁ F₂ A)) : Decidable (PairColorableOn (T := T) views₁ views₂ s) := by
  unfold PairColorableOn
  infer_instance

theorem pairColorableOn_of_colorableOn
    {F₁ F₂ A L₁ L₂ T₁ T₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    {s₁ : Finset (State F₁ A)}
    {s₂ : Finset (State F₂ A)}
    (h₁ : ColorableOn (F := F₁) (A := A) (L := L₁) (T := T₁) views₁ s₁)
    (h₂ : ColorableOn (F := F₂) (A := A) (L := L₂) (T := T₂) views₂ s₂) :
    PairColorableOn (T := T₁ * T₂) views₁ views₂ (s₁.product s₂) := by
  rcases h₁ with ⟨tag₁, htag₁⟩
  rcases h₂ with ⟨tag₂, htag₂⟩
  refine ⟨pairTag tag₁ tag₂, ?_⟩
  intro x y hx hy hconf htag
  rcases Finset.mem_product.mp hx with ⟨hx₁, hx₂⟩
  rcases Finset.mem_product.mp hy with ⟨hy₁, hy₂⟩
  rcases hconf with ⟨hxy, ⟨ℓ₁, ℓ₂⟩, hobs⟩
  have hpair : (tag₁ x.1, tag₂ x.2) = (tag₁ y.1, tag₂ y.2) := pairColorCode_injective htag
  have htag₁eq : tag₁ x.1 = tag₁ y.1 := congrArg Prod.fst hpair
  have htag₂eq : tag₂ x.2 = tag₂ y.2 := congrArg Prod.snd hpair
  rcases x with ⟨x₁, x₂⟩
  rcases y with ⟨y₁, y₂⟩
  dsimp [pairObserve] at hobs
  rcases Prod.mk.inj hobs with ⟨hobs₁, hobs₂⟩
  by_cases hleft : x₁ = y₁
  · have hright : x₂ ≠ y₂ := by
      intro hEq
      apply hxy
      cases hleft
      cases hEq
      rfl
    exact (htag₂ hx₂ hy₂ ⟨hright, ⟨ℓ₂, hobs₂⟩⟩) htag₂eq
  · exact (htag₁ hx₁ hy₁ ⟨hleft, ⟨ℓ₁, hobs₁⟩⟩) htag₁eq

/-- The empty product support is colorable whenever the color alphabet is nonempty. -/
theorem empty_pairColorableOn {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type} {T : Nat}
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (hT : 0 < T) :
    PairColorableOn (F₁ := F₁) (F₂ := F₂) (A := A) (T := T) views₁ views₂ ∅ := by
  haveI : NeZero T := ⟨Nat.ne_of_gt hT⟩
  refine ⟨fun _ => 0, ?_⟩
  intro x y hx
  simpa using hx

/-- Maximum exactly decodable success-set size in the product model. -/
noncomputable def pairMaxColorableCard {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type} {T : Nat}
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (hT : 0 < T) [Fintype Loc₁] [Fintype Loc₂] [DecidableEq Loc₁] [DecidableEq Loc₂] : Nat :=
  ((Finset.univ.powerset.filter fun s => PairColorableOn (F₁ := F₁) (F₂ := F₂) (A := A) (T := T) views₁ views₂ s).image Finset.card).max'
    (by
      refine ⟨0, ?_⟩
      refine Finset.mem_image.mpr ?_
      refine ⟨∅, ?_, by simp⟩
      simp [empty_pairColorableOn (F₁ := F₁) (F₂ := F₂) (A := A) views₁ views₂ hT])

theorem pairColorableOn_card_le_pairMaxColorableCard
    {F₁ F₂ A : Nat} {Loc₁ Loc₂ : Type} {T : Nat}
    [Fintype Loc₁] [Fintype Loc₂] [DecidableEq Loc₁] [DecidableEq Loc₂]
    (views₁ : Loc₁ → Finset (Fin F₁))
    (views₂ : Loc₂ → Finset (Fin F₂))
    (s : Finset (PairState F₁ F₂ A))
    (hT : 0 < T)
    (hcolor : PairColorableOn (T := T) views₁ views₂ s) :
    s.card ≤ pairMaxColorableCard (F₁ := F₁) (F₂ := F₂) (A := A) (T := T) views₁ views₂ hT := by
  have hmem : s.card ∈
      (Finset.univ.powerset.filter fun s => PairColorableOn (F₁ := F₁) (F₂ := F₂) (A := A) (T := T) views₁ views₂ s).image Finset.card := by
    refine Finset.mem_image.mpr ?_
    refine ⟨s, ?_, rfl⟩
    simp [hcolor]
  exact Finset.le_max' _ _ hmem

theorem pair_product_success_card_lower_bound
    {F₁ F₂ A L₁ L₂ T₁ T₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (hT₁ : 0 < T₁)
    (hT₂ : 0 < T₂) :
    maxColorableCard (F := F₁) (A := A) (L := L₁) (T := T₁) views₁ hT₁ *
      maxColorableCard (F := F₂) (A := A) (L := L₂) (T := T₂) views₂ hT₂
      ≤
      pairMaxColorableCard (F₁ := F₁) (F₂ := F₂) (A := A) (T := T₁ * T₂)
        views₁ views₂ (Nat.mul_pos hT₁ hT₂) := by
  rcases exists_colorableOn_card_eq_maxColorableCard (F := F₁) (A := A) (L := L₁) (T := T₁) views₁ hT₁
    with ⟨s₁, hs₁, hcard₁⟩
  rcases exists_colorableOn_card_eq_maxColorableCard (F := F₂) (A := A) (L := L₂) (T := T₂) views₂ hT₂
    with ⟨s₂, hs₂, hcard₂⟩
  have hpair : PairColorableOn (T := T₁ * T₂) views₁ views₂ (s₁.product s₂) :=
    pairColorableOn_of_colorableOn views₁ views₂ hs₁ hs₂
  have hbound :=
    pairColorableOn_card_le_pairMaxColorableCard
      (F₁ := F₁) (F₂ := F₂) (A := A) (T := T₁ * T₂)
      views₁ views₂ (s₁.product s₂) (Nat.mul_pos hT₁ hT₂) hpair
  simpa [hcard₁, hcard₂, Finset.card_product] using hbound

/-- A finite set is independent when it contains no confusable distinct pair. -/
def Independent {F A L : Nat} (views : ViewFamily F L) (s : Finset (State F A)) : Prop :=
  ∀ ⦃x y : State F A⦄, x ∈ s → y ∈ s → x ≠ y → ¬ Confusable views x y

instance instDecidableIndependent {F A L : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A)) : Decidable (Independent views s) := by
  unfold Independent
  infer_instance

/-- Largest one-shot zero-error code size for the confusability graph. -/
noncomputable def maxIndependentCard {F A L : Nat}
    (views : ViewFamily F L) : Nat :=
  (((Finset.univ : Finset (State F A)).powerset.filter
      fun s : Finset (State F A) => Independent (A := A) views s).image Finset.card).max'
    (by
      refine ⟨0, ?_⟩
      refine Finset.mem_image.mpr ?_
      refine ⟨∅, ?_, by simp⟩
      simp [Independent])

theorem independent_card_le_maxIndependentCard
    {F A L : Nat}
    (views : ViewFamily F L)
    (s : Finset (State F A))
    (hindep : Independent views s) :
    s.card ≤ maxIndependentCard (F := F) (A := A) (L := L) views := by
  have hmem : s.card ∈
      (((Finset.univ : Finset (State F A)).powerset.filter
        fun s : Finset (State F A) => Independent (A := A) views s).image Finset.card) := by
    refine Finset.mem_image.mpr ?_
    refine ⟨s, ?_, rfl⟩
    simp [hindep]
  exact Finset.le_max' _ _ hmem

theorem maxIndependentCard_pos
    {F A L : Nat}
    (views : ViewFamily F L)
    [Nonempty (State F A)] :
    0 < maxIndependentCard (F := F) (A := A) (L := L) views := by
  classical
  let x : State F A := Classical.choice ‹Nonempty (State F A)›
  have hsingle : Independent views ({x} : Finset (State F A)) := by
    intro a b ha hb hneq hconf
    simp at ha hb
    subst ha
    subst hb
    exact hneq rfl
  have hbound :
      ({x} : Finset (State F A)).card ≤ maxIndependentCard (F := F) (A := A) (L := L) views :=
    independent_card_le_maxIndependentCard views {x} hsingle
  simpa [x] using hbound

/-- Independence in the product confusability graph. -/
def PairIndependent {F₁ F₂ A : Nat} {L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (s : Finset (PairState F₁ F₂ A)) : Prop :=
  ∀ ⦃x y : PairState F₁ F₂ A⦄, x ∈ s → y ∈ s → x ≠ y → ¬ PairConfusable views₁ views₂ x y

instance instDecidablePairIndependent {F₁ F₂ A : Nat} {L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    (s : Finset (PairState F₁ F₂ A)) : Decidable (PairIndependent views₁ views₂ s) := by
  unfold PairIndependent
  infer_instance

theorem pairIndependent_of_independent
    {F₁ F₂ A L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂)
    {s₁ : Finset (State F₁ A)}
    {s₂ : Finset (State F₂ A)}
    (h₁ : Independent views₁ s₁)
    (h₂ : Independent views₂ s₂) :
    PairIndependent views₁ views₂ (s₁.product s₂) := by
  intro x y hx hy hxy hconf
  rcases Finset.mem_product.mp hx with ⟨hx₁, hx₂⟩
  rcases Finset.mem_product.mp hy with ⟨hy₁, hy₂⟩
  rcases hconf with ⟨_, ⟨ℓ₁, ℓ₂⟩, hobs⟩
  rcases x with ⟨x₁, x₂⟩
  rcases y with ⟨y₁, y₂⟩
  dsimp [pairObserve] at hobs
  rcases Prod.mk.inj hobs with ⟨hobs₁, hobs₂⟩
  by_cases hleft : x₁ = y₁
  · have hright : x₂ ≠ y₂ := by
      intro hEq
      apply hxy
      cases hleft
      cases hEq
      rfl
    exact h₂ hx₂ hy₂ hright ⟨hright, ⟨ℓ₂, hobs₂⟩⟩
  · exact h₁ hx₁ hy₁ hleft ⟨hleft, ⟨ℓ₁, hobs₁⟩⟩

/-- Largest one-shot zero-error code size in the block-composed product system. -/
noncomputable def pairMaxIndependentCard {F₁ F₂ A L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂) : Nat :=
  (((Finset.univ : Finset (PairState F₁ F₂ A)).powerset.filter
      fun s : Finset (PairState F₁ F₂ A) => PairIndependent (A := A) views₁ views₂ s).image Finset.card).max'
    (by
      refine ⟨0, ?_⟩
      refine Finset.mem_image.mpr ?_
      refine ⟨∅, ?_, by simp⟩
      simp [PairIndependent])

theorem pair_product_independent_lower_bound
    {F₁ F₂ A L₁ L₂ : Nat}
    (views₁ : ViewFamily F₁ L₁)
    (views₂ : ViewFamily F₂ L₂) :
    maxIndependentCard (F := F₁) (A := A) (L := L₁) views₁ *
      maxIndependentCard (F := F₂) (A := A) (L := L₂) views₂
      ≤
      pairMaxIndependentCard (F₁ := F₁) (F₂ := F₂) (A := A) views₁ views₂ := by
  have hmem₁ : maxIndependentCard (F := F₁) (A := A) (L := L₁) views₁ ∈
      (((Finset.univ : Finset (State F₁ A)).powerset.filter
        fun s : Finset (State F₁ A) => Independent (A := A) views₁ s).image Finset.card) := by
    exact Finset.max'_mem _ _
  have hmem₂ : maxIndependentCard (F := F₂) (A := A) (L := L₂) views₂ ∈
      (((Finset.univ : Finset (State F₂ A)).powerset.filter
        fun s : Finset (State F₂ A) => Independent (A := A) views₂ s).image Finset.card) := by
    exact Finset.max'_mem _ _
  rcases Finset.mem_image.mp hmem₁ with ⟨s₁, hs₁, hcard₁⟩
  rcases Finset.mem_image.mp hmem₂ with ⟨s₂, hs₂, hcard₂⟩
  have hpair : PairIndependent views₁ views₂ (s₁.product s₂) :=
    pairIndependent_of_independent views₁ views₂ (by simpa using hs₁) (by simpa using hs₂)
  have hmem :
      (s₁.product s₂).card ∈
        (((Finset.univ : Finset (PairState F₁ F₂ A)).powerset.filter
          fun s : Finset (PairState F₁ F₂ A) => PairIndependent (A := A) views₁ views₂ s).image Finset.card) := by
    refine Finset.mem_image.mpr ?_
    refine ⟨s₁.product s₂, ?_, rfl⟩
    simp
    exact hpair
  have hbound :
      (s₁.product s₂).card ≤ pairMaxIndependentCard (F₁ := F₁) (F₂ := F₂) (A := A) views₁ views₂ := by
    exact Finset.le_max' _ _ hmem
  simpa [hcard₁, hcard₂, Finset.card_product] using hbound

/-! ## Repeated block powers and asymptotic lower-bound scaffolding -/

/-- An `N`-block latent tuple of states over a common one-fact observation family. -/
abbrev BlockState (N F A : Nat) := Fin N → State F A

/-- A block observation records one partial observation per block. -/
abbrev BlockObservation (N F A : Nat) := Fin N → PartialObservation F A

/-- A block location chooses one allowed location in each block. -/
abbrev BlockLoc (N L : Nat) := Fin N → Fin L

/-- Simultaneous observation of all blocks under a tuple of locations. -/
def blockObserve {N F A L : Nat}
    (views : ViewFamily F L)
    (ℓ : BlockLoc N L)
    (x : BlockState N F A) : BlockObservation N F A :=
  fun i => observe (views (ℓ i)) (x i)

/-- Two block states are confusable when some tuple of locations yields identical block observations. -/
def BlockConfusable {N F A L : Nat}
    (views : ViewFamily F L)
    (x y : BlockState N F A) : Prop :=
  x ≠ y ∧ ∃ ℓ : BlockLoc N L, blockObserve views ℓ x = blockObserve views ℓ y

instance instDecidableBlockConfusable {N F A L : Nat}
    (views : ViewFamily F L)
    (x y : BlockState N F A) : Decidable (BlockConfusable views x y) := by
  unfold BlockConfusable
  infer_instance

/-- Confusability graph of the `N`-block product observation family. -/
def blockConfusabilityGraph {N F A L : Nat}
    (views : ViewFamily F L) : SimpleGraph (BlockState N F A) where
  Adj := BlockConfusable views
  symm := by
    intro x y h
    rcases h with ⟨hxy, ℓ, hobs⟩
    exact ⟨hxy.symm, ℓ, hobs.symm⟩
  loopless := by
    intro x h
    exact h.1 rfl

/-- A finite set of block states is independent when no two distinct elements are confusable. -/
def BlockIndependent {N F A L : Nat}
    (views : ViewFamily F L)
    (s : Finset (BlockState N F A)) : Prop :=
  ∀ ⦃x y : BlockState N F A⦄, x ∈ s → y ∈ s → x ≠ y → ¬ BlockConfusable views x y

instance instDecidableBlockIndependent {N F A L : Nat}
    (views : ViewFamily F L)
    (s : Finset (BlockState N F A)) : Decidable (BlockIndependent views s) := by
  unfold BlockIndependent
  infer_instance

theorem blockIndependent_iff_graphIndependent
    {N F A L : Nat}
    (views : ViewFamily F L)
    (s : Finset (BlockState N F A)) :
    BlockIndependent views s ↔ GraphIndependent (blockConfusabilityGraph (N := N) (F := F) (A := A) views) s := by
  rfl

theorem blockConfusable_iff_strongPowAdj
    {N F A L : Nat}
    (views : ViewFamily F L)
    {x y : BlockState N F A} :
    BlockConfusable views x y ↔
      (strongPow (confusabilityGraph (A := A) views) N).Adj x y := by
  constructor
  · rintro ⟨hneq, ℓ, hobs⟩
    refine ⟨?_, ?_⟩
    · intro i
      by_cases hEq : x i = y i
      · exact Or.inl hEq
      · have hobsi : observe (views (ℓ i)) (x i) = observe (views (ℓ i)) (y i) := by
          exact congrFun hobs i
        exact Or.inr ⟨hEq, ⟨ℓ i, hobsi⟩⟩
    · by_contra hnone
      push_neg at hnone
      apply hneq
      funext i
      rcases (by
        have hi := (show x i = y i ∨ (confusabilityGraph (A := A) views).Adj (x i) (y i) from by
          by_cases hEq : x i = y i
          · exact Or.inl hEq
          · exact Or.inr ⟨hEq, ⟨ℓ i, congrFun hobs i⟩⟩)
        exact hi) with hEq | hAdj
      · exact hEq
      · exfalso
        exact hnone i hAdj
  · intro h
    rcases h with ⟨hcoord, i0, hi0⟩
    rcases hi0 with ⟨hneq0, ⟨ℓ0, hobs0⟩⟩
    refine ⟨?_, ?_⟩
    · intro hEq
      apply hneq0
      simpa [hEq]
    · classical
      let ℓ : BlockLoc N L := fun i =>
        if hEq : x i = y i then
          ℓ0
        else
          Classical.choose <|
            by
              rcases hcoord i with hEq' | hAdj
              · exact False.elim (hEq hEq')
              · exact hAdj.2
      refine ⟨ℓ, ?_⟩
      funext i
      dsimp [ℓ]
      by_cases hEq : x i = y i
      · simpa [blockObserve, hEq] using congrArg (observe (views ℓ0)) hEq
      · simp [blockObserve, hEq]
        exact Classical.choose_spec <|
          by
            rcases hcoord i with hEq' | hAdj
            · exact False.elim (hEq hEq')
            · exact hAdj.2

theorem blockConfusabilityGraph_eq_strongPow
    {N F A L : Nat}
    (views : ViewFamily F L) :
    blockConfusabilityGraph (N := N) (F := F) (A := A) views =
      strongPow (confusabilityGraph (A := A) views) N := by
  ext x y
  exact blockConfusable_iff_strongPowAdj views

/-- Product codebooks from a one-shot independent set remain independent in the `N`-block product model. -/
theorem blockIndependent_piFinset_of_independent
    {N F A L : Nat}
    (views : ViewFamily F L)
    {s : Finset (State F A)}
    (hindep : Independent views s) :
    BlockIndependent views (Fintype.piFinset fun _ : Fin N => s) := by
  intro x y hx hy hxy hconf
  have hxmem : ∀ i : Fin N, x i ∈ s := by
    simpa using (Fintype.mem_piFinset.mp hx)
  have hymem : ∀ i : Fin N, y i ∈ s := by
    simpa using (Fintype.mem_piFinset.mp hy)
  have hnotall : ¬ ∀ i : Fin N, x i = y i := by
    intro hall
    apply hxy
    funext i
    exact hall i
  push_neg at hnotall
  rcases hnotall with ⟨i, hneqi⟩
  rcases hconf with ⟨_, ℓ, hobs⟩
  have hobsi : observe (views (ℓ i)) (x i) = observe (views (ℓ i)) (y i) := by
    exact congrFun hobs i
  exact hindep (hxmem i) (hymem i) hneqi ⟨hneqi, ⟨ℓ i, hobsi⟩⟩

/-- Largest one-shot zero-error code size in the `N`-block product model. -/
noncomputable def blockMaxIndependentCard {N F A L : Nat}
    (views : ViewFamily F L) : Nat :=
  (((Finset.univ : Finset (BlockState N F A)).powerset.filter
      fun s : Finset (BlockState N F A) => BlockIndependent (N := N) (F := F) (A := A) views s).image
      Finset.card).max'
    (by
      refine ⟨0, ?_⟩
      refine Finset.mem_image.mpr ?_
      refine ⟨∅, ?_, by simp⟩
      simp [BlockIndependent])

theorem blockMaxIndependentCard_eq_graphMaxIndependentCard
    {N F A L : Nat}
    (views : ViewFamily F L) :
    blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views =
      graphMaxIndependentCard (blockConfusabilityGraph (N := N) (F := F) (A := A) views) := by
  classical
  unfold blockMaxIndependentCard graphMaxIndependentCard
  simp [blockIndependent_iff_graphIndependent]

theorem blockIndependent_card_le_blockMaxIndependentCard
    {N F A L : Nat}
    (views : ViewFamily F L)
    (s : Finset (BlockState N F A))
    (hindep : BlockIndependent views s) :
    s.card ≤ blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views := by
  have hmem : s.card ∈
      (((Finset.univ : Finset (BlockState N F A)).powerset.filter
        fun s : Finset (BlockState N F A) => BlockIndependent (N := N) (F := F) (A := A) views s).image
        Finset.card) := by
    refine Finset.mem_image.mpr ?_
    refine ⟨s, ?_, rfl⟩
    simp [hindep]
  exact Finset.le_max' _ _ hmem

theorem blockMaxIndependentCard_pos
    {N F A L : Nat}
    (views : ViewFamily F L)
    [Nonempty (State F A)] :
    0 < blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views := by
  classical
  let x : BlockState N F A := fun _ => Classical.choice ‹Nonempty (State F A)›
  have hsingle : BlockIndependent views ({x} : Finset (BlockState N F A)) := by
    intro a b ha hb hneq hconf
    simp at ha hb
    subst ha
    subst hb
    exact hneq rfl
  have hbound :
      ({x} : Finset (BlockState N F A)).card ≤
        blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views :=
    blockIndependent_card_le_blockMaxIndependentCard views {x} hsingle
  simpa [x] using hbound

theorem exists_independent_card_eq_maxIndependentCard
    {F A L : Nat}
    (views : ViewFamily F L) :
    ∃ s : Finset (State F A),
      Independent views s ∧
      s.card = maxIndependentCard (F := F) (A := A) (L := L) views := by
  have hmem : maxIndependentCard (F := F) (A := A) (L := L) views ∈
      (((Finset.univ : Finset (State F A)).powerset.filter
        fun s : Finset (State F A) => Independent (A := A) views s).image Finset.card) := by
    exact Finset.max'_mem _ _
  rcases Finset.mem_image.mp hmem with ⟨s, hs, hcard⟩
  exact ⟨s, by simpa using hs, hcard⟩

/-- Repeated block composition yields an exponential one-shot zero-error lower bound. -/
theorem block_power_independent_lower_bound
    {N F A L : Nat}
    (views : ViewFamily F L) :
    maxIndependentCard (F := F) (A := A) (L := L) views ^ N ≤
      blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views := by
  rcases exists_independent_card_eq_maxIndependentCard (F := F) (A := A) (L := L) views with
    ⟨s, hs, hcard⟩
  have hblock :
      BlockIndependent views (Fintype.piFinset fun _ : Fin N => s) :=
    blockIndependent_piFinset_of_independent views hs
  have hbound :=
    blockIndependent_card_le_blockMaxIndependentCard
      (N := N) (F := F) (A := A) (L := L) views (Fintype.piFinset fun _ : Fin N => s) hblock
  simpa [hcard, Fintype.card_piFinset_const] using hbound

/-- Concatenate two block states into one longer block state. -/
def concatBlockState {M N F A : Nat}
    (x : BlockState M F A)
    (y : BlockState N F A) : BlockState (M + N) F A :=
  Fin.append x y

/-- Concatenate two block-location tuples. -/
def concatBlockLoc {M N L : Nat}
    (ℓ₁ : BlockLoc M L)
    (ℓ₂ : BlockLoc N L) : BlockLoc (M + N) L :=
  Fin.append ℓ₁ ℓ₂

theorem concatBlockState_pair_injective {M N F A : Nat} :
    Function.Injective (fun p : BlockState M F A × BlockState N F A => concatBlockState p.1 p.2) := by
  intro a b h
  rcases a with ⟨x₁, y₁⟩
  rcases b with ⟨x₂, y₂⟩
  have hleft : x₁ = x₂ := by
    funext i
    simpa [concatBlockState] using congrFun h (Fin.castAdd N i)
  have hright : y₁ = y₂ := by
    funext j
    simpa [concatBlockState] using congrFun h (Fin.natAdd M j)
  simp [hleft, hright]

theorem concatBlockConfusable_iff_strongProdAdj
    {M N F A L : Nat}
    (views : ViewFamily F L)
    (hL : 0 < L)
    {x₁ y₁ : BlockState M F A}
    {x₂ y₂ : BlockState N F A} :
    BlockConfusable views (concatBlockState x₁ x₂) (concatBlockState y₁ y₂) ↔
      (strongProd (blockConfusabilityGraph (N := M) (F := F) (A := A) views)
        (blockConfusabilityGraph (N := N) (F := F) (A := A) views)).Adj (x₁, x₂) (y₁, y₂) := by
  let d : Fin L := ⟨0, hL⟩
  constructor
  · rintro ⟨hxy, ℓ, hobs⟩
    have hleft : x₁ = y₁ ∨ BlockConfusable views x₁ y₁ := by
      by_cases hEq : x₁ = y₁
      · exact Or.inl hEq
      · have hobsLeft :
            ∀ i : Fin M,
              observe (views (ℓ (Fin.castAdd N i))) (x₁ i) =
                observe (views (ℓ (Fin.castAdd N i))) (y₁ i) := by
            intro i
            simpa [blockObserve, concatBlockState] using congrFun hobs (Fin.castAdd N i)
        let ℓ₁ : BlockLoc M L := fun i => ℓ (Fin.castAdd N i)
        exact Or.inr ⟨hEq, ⟨ℓ₁, by funext i; exact hobsLeft i⟩⟩
    have hright : x₂ = y₂ ∨ BlockConfusable views x₂ y₂ := by
      by_cases hEq : x₂ = y₂
      · exact Or.inl hEq
      · have hobsRight :
            ∀ j : Fin N,
              observe (views (ℓ (Fin.natAdd M j))) (x₂ j) =
                observe (views (ℓ (Fin.natAdd M j))) (y₂ j) := by
            intro j
            simpa [blockObserve, concatBlockState] using congrFun hobs (Fin.natAdd M j)
        let ℓ₂ : BlockLoc N L := fun j => ℓ (Fin.natAdd M j)
        exact Or.inr ⟨hEq, ⟨ℓ₂, by funext j; exact hobsRight j⟩⟩
    have hsome : BlockConfusable views x₁ y₁ ∨ BlockConfusable views x₂ y₂ := by
      by_cases hEq1 : x₁ = y₁
      · right
        have hneq₂ : x₂ ≠ y₂ := by
          intro hEq2
          apply hxy
          cases hEq1
          cases hEq2
          rfl
        have hobsRight :
            ∀ j : Fin N,
              observe (views (ℓ (Fin.natAdd M j))) (x₂ j) =
                observe (views (ℓ (Fin.natAdd M j))) (y₂ j) := by
            intro j
            simpa [blockObserve, concatBlockState] using congrFun hobs (Fin.natAdd M j)
        let ℓ₂ : BlockLoc N L := fun j => ℓ (Fin.natAdd M j)
        exact ⟨hneq₂, ⟨ℓ₂, by funext j; exact hobsRight j⟩⟩
      · left
        have hobsLeft :
            ∀ i : Fin M,
              observe (views (ℓ (Fin.castAdd N i))) (x₁ i) =
                observe (views (ℓ (Fin.castAdd N i))) (y₁ i) := by
            intro i
            simpa [blockObserve, concatBlockState] using congrFun hobs (Fin.castAdd N i)
        let ℓ₁ : BlockLoc M L := fun i => ℓ (Fin.castAdd N i)
        exact ⟨hEq1, ⟨ℓ₁, by funext i; exact hobsLeft i⟩⟩
    exact ⟨hleft, hright, hsome⟩
  · intro h
    rcases h with ⟨hleft, hright, hsome⟩
    have hneqpair : concatBlockState x₁ x₂ ≠ concatBlockState y₁ y₂ := by
      intro hEq
      apply hsome.elim
      · intro hAdj
        exact hAdj.1 (by
          funext i
          simpa [concatBlockState] using congrFun hEq (Fin.castAdd N i))
      · intro hAdj
        exact hAdj.1 (by
          funext j
          simpa [concatBlockState] using congrFun hEq (Fin.natAdd M j))
    rcases hleft with hEq₁ | hAdj₁
    · rcases hright with hEq₂ | hAdj₂
      · exfalso
        exact hsome.elim (fun hAdj => hAdj.1 hEq₁) (fun hAdj => hAdj.1 hEq₂)
      · rcases hAdj₂ with ⟨_, ⟨ℓ₂, hobs₂⟩⟩
        let ℓ : BlockLoc (M + N) L := concatBlockLoc (fun _ => d) ℓ₂
        refine ⟨hneqpair, ⟨ℓ, ?_⟩⟩
        funext k
        refine Fin.addCases ?_ ?_ k
        · intro i
          simpa [ℓ, blockObserve, concatBlockLoc, concatBlockState] using
            congrArg (observe (views d)) (congrFun hEq₁ i)
        · intro j
          simpa [ℓ, blockObserve, concatBlockLoc, concatBlockState] using congrFun hobs₂ j
    · rcases hright with hEq₂ | hAdj₂
      · rcases hAdj₁ with ⟨_, ⟨ℓ₁, hobs₁⟩⟩
        let ℓ : BlockLoc (M + N) L := concatBlockLoc ℓ₁ (fun _ => d)
        refine ⟨hneqpair, ⟨ℓ, ?_⟩⟩
        funext k
        refine Fin.addCases ?_ ?_ k
        · intro i
          simpa [ℓ, blockObserve, concatBlockLoc, concatBlockState] using congrFun hobs₁ i
        · intro j
          simpa [ℓ, blockObserve, concatBlockLoc, concatBlockState] using
            congrArg (observe (views d)) (congrFun hEq₂ j)
      · rcases hAdj₁ with ⟨_, ⟨ℓ₁, hobs₁⟩⟩
        rcases hAdj₂ with ⟨_, ⟨ℓ₂, hobs₂⟩⟩
        let ℓ : BlockLoc (M + N) L := concatBlockLoc ℓ₁ ℓ₂
        refine ⟨hneqpair, ⟨ℓ, ?_⟩⟩
        funext k
        refine Fin.addCases ?_ ?_ k
        · intro i
          simpa [ℓ, blockObserve, concatBlockLoc, concatBlockState] using congrFun hobs₁ i
        · intro j
          simpa [ℓ, blockObserve, concatBlockLoc, concatBlockState] using congrFun hobs₂ j

def blockConfusabilityGraph_addIsoStrongProd
    {M N F A L : Nat}
    (views : ViewFamily F L)
    (hL : 0 < L) :
    blockConfusabilityGraph (N := M + N) (F := F) (A := A) views ≃g
      strongProd (blockConfusabilityGraph (N := M) (F := F) (A := A) views)
        (blockConfusabilityGraph (N := N) (F := F) (A := A) views) :=
  let e : BlockState (M + N) F A ≃ (BlockState M F A × BlockState N F A) :=
    { toFun := fun x =>
        ((fun i => x (Fin.castAdd N i)), (fun j => x (Fin.natAdd M j)))
      invFun := fun p => concatBlockState p.1 p.2
      left_inv := by
        intro x
        funext k
        refine Fin.addCases ?_ ?_ k
        · intro i
          simp [concatBlockState]
        · intro j
          simp [concatBlockState]
      right_inv := by
        intro p
        rcases p with ⟨x₁, x₂⟩
        simp [concatBlockState] }
  ⟨e, by
    intro x y
    let x₁ : BlockState M F A := fun i => x (Fin.castAdd N i)
    let x₂ : BlockState N F A := fun j => x (Fin.natAdd M j)
    let y₁ : BlockState M F A := fun i => y (Fin.castAdd N i)
    let y₂ : BlockState N F A := fun j => y (Fin.natAdd M j)
    have hx : concatBlockState x₁ x₂ = x := by
      funext k
      refine Fin.addCases ?_ ?_ k
      · intro i
        simp [x₁, concatBlockState]
      · intro j
        simp [x₂, concatBlockState]
    have hy : concatBlockState y₁ y₂ = y := by
      funext k
      refine Fin.addCases ?_ ?_ k
      · intro i
        simp [y₁, concatBlockState]
      · intro j
        simp [y₂, concatBlockState]
    change
      (strongProd (blockConfusabilityGraph (N := M) (F := F) (A := A) views)
        (blockConfusabilityGraph (N := N) (F := F) (A := A) views)).Adj (x₁, x₂) (y₁, y₂) ↔
      BlockConfusable views x y
    simpa [e, x₁, x₂, y₁, y₂, hx, hy] using
      (concatBlockConfusable_iff_strongProdAdj (M := M) (N := N) (F := F) (A := A) views hL
        (x₁ := x₁) (x₂ := x₂) (y₁ := y₁) (y₂ := y₂)).symm ⟩

theorem blockGraphMaxIndependentCard_add_eq_strongProd
    {M N F A L : Nat}
    (views : ViewFamily F L)
    (hL : 0 < L) :
    graphMaxIndependentCard (blockConfusabilityGraph (N := M + N) (F := F) (A := A) views) =
      graphMaxIndependentCard
        (strongProd (blockConfusabilityGraph (N := M) (F := F) (A := A) views)
          (blockConfusabilityGraph (N := N) (F := F) (A := A) views)) := by
  exact graphMaxIndependentCard_iso_eq (blockConfusabilityGraph_addIsoStrongProd (M := M) (N := N) (F := F) (A := A) views hL)

/-- Concatenating independent block codebooks preserves independence in the longer block system. -/
theorem concatBlockIndependent_of_blockIndependent
    {M N F A L : Nat}
    (views : ViewFamily F L)
    {s₁ : Finset (BlockState M F A)}
    {s₂ : Finset (BlockState N F A)}
    (h₁ : BlockIndependent views s₁)
    (h₂ : BlockIndependent views s₂) :
    BlockIndependent views ((s₁ ×ˢ s₂).image (fun p => concatBlockState p.1 p.2)) := by
  intro x y hx hy hxy hconf
  rcases Finset.mem_image.mp hx with ⟨⟨x₁, y₁⟩, hp₁, rfl⟩
  rcases Finset.mem_image.mp hy with ⟨⟨x₂, y₂⟩, hp₂, hxy'⟩
  cases hxy'
  rcases Finset.mem_product.mp hp₁ with ⟨hx₁, hy₁⟩
  rcases Finset.mem_product.mp hp₂ with ⟨hx₂, hy₂⟩
  have hneq : concatBlockState x₁ y₁ ≠ concatBlockState x₂ y₂ := by
    simpa using hxy
  rcases hconf with ⟨_, ℓ, hobs⟩
  by_cases hleft : x₁ = x₂
  · have hright : y₁ ≠ y₂ := by
      intro hyEq
      apply hneq
      cases hleft
      cases hyEq
      rfl
    have hobsRight :
        ∀ j : Fin N,
          observe (views (ℓ (Fin.natAdd M j))) (y₁ j) =
            observe (views (ℓ (Fin.natAdd M j))) (y₂ j) := by
      intro j
      simpa [blockObserve, concatBlockState] using congrFun hobs (Fin.natAdd M j)
    let ℓ₂ : BlockLoc N L := fun j => ℓ (Fin.natAdd M j)
    have hconf₂ : BlockConfusable views y₁ y₂ := by
      refine ⟨hright, ?_⟩
      refine ⟨ℓ₂, ?_⟩
      funext j
      exact hobsRight j
    exact h₂ hy₁ hy₂ hright hconf₂
  · have hobsLeft :
        ∀ i : Fin M,
          observe (views (ℓ (Fin.castAdd N i))) (x₁ i) =
            observe (views (ℓ (Fin.castAdd N i))) (x₂ i) := by
      intro i
      simpa [blockObserve, concatBlockState] using congrFun hobs (Fin.castAdd N i)
    let ℓ₁ : BlockLoc M L := fun i => ℓ (Fin.castAdd N i)
    have hconf₁ : BlockConfusable views x₁ x₂ := by
      refine ⟨hleft, ?_⟩
      refine ⟨ℓ₁, ?_⟩
      funext i
      exact hobsLeft i
    exact h₁ hx₁ hx₂ hleft hconf₁

theorem exists_blockIndependent_card_eq_blockMaxIndependentCard
    {N F A L : Nat}
    (views : ViewFamily F L) :
    ∃ s : Finset (BlockState N F A),
      BlockIndependent views s ∧
      s.card = blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views := by
  have hmem : blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views ∈
      (((Finset.univ : Finset (BlockState N F A)).powerset.filter
        fun s : Finset (BlockState N F A) => BlockIndependent (N := N) (F := F) (A := A) views s).image
        Finset.card) := by
    exact Finset.max'_mem _ _
  rcases Finset.mem_image.mp hmem with ⟨s, hs, hcard⟩
  exact ⟨s, by simpa using hs, hcard⟩

/-- The block-independent code size is supermultiplicative under block concatenation. -/
theorem block_add_independent_lower_bound
    {M N F A L : Nat}
    (views : ViewFamily F L) :
    blockMaxIndependentCard (N := M) (F := F) (A := A) (L := L) views *
      blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views
      ≤
      blockMaxIndependentCard (N := M + N) (F := F) (A := A) (L := L) views := by
  rcases exists_blockIndependent_card_eq_blockMaxIndependentCard (N := M) (F := F) (A := A) (L := L) views with
    ⟨s₁, hs₁, hcard₁⟩
  rcases exists_blockIndependent_card_eq_blockMaxIndependentCard (N := N) (F := F) (A := A) (L := L) views with
    ⟨s₂, hs₂, hcard₂⟩
  let s : Finset (BlockState (M + N) F A) := (s₁ ×ˢ s₂).image (fun p => concatBlockState p.1 p.2)
  have hs : BlockIndependent views s :=
    concatBlockIndependent_of_blockIndependent views hs₁ hs₂
  have hbound :
      s.card ≤ blockMaxIndependentCard (N := M + N) (F := F) (A := A) (L := L) views := by
    exact blockIndependent_card_le_blockMaxIndependentCard views s hs
  have hcard :
      s.card = s₁.card * s₂.card := by
    dsimp [s]
    calc
      (Finset.image (fun p => concatBlockState p.1 p.2) (s₁ ×ˢ s₂)).card = (s₁ ×ˢ s₂).card := by
        exact Finset.card_image_of_injective (s := s₁ ×ˢ s₂) concatBlockState_pair_injective
      _ = s₁.card * s₂.card := Finset.card_product _ _
  simpa [hcard, hcard₁, hcard₂] using hbound

/-- Repeating a fixed `M`-block system `K` times gives at least the `K`th power of its
one-block code size. -/
theorem block_mul_independent_lower_bound
    {K M F A L : Nat}
    (views : ViewFamily F L)
    [Nonempty (State F A)] :
    blockMaxIndependentCard (N := M) (F := F) (A := A) (L := L) views ^ K ≤
      blockMaxIndependentCard (N := K * M) (F := F) (A := A) (L := L) views := by
  induction K with
  | zero =>
      have hpos := blockMaxIndependentCard_pos (N := 0) (F := F) (A := A) (L := L) views
      simpa using Nat.succ_le_of_lt hpos
  | succ k ih =>
      have hstep :=
        block_add_independent_lower_bound (M := k * M) (N := M) (F := F) (A := A) (L := L) views
      have hmul :
          blockMaxIndependentCard (N := M) (F := F) (A := A) (L := L) views ^ k *
            blockMaxIndependentCard (N := M) (F := F) (A := A) (L := L) views
            ≤
            blockMaxIndependentCard (N := k * M) (F := F) (A := A) (L := L) views *
              blockMaxIndependentCard (N := M) (F := F) (A := A) (L := L) views :=
        Nat.mul_le_mul_right _ ih
      exact le_trans (by simpa [pow_succ] using hmul) (by
        simpa [Nat.succ_mul, Nat.add_comm, Nat.add_left_comm, Nat.add_assoc] using hstep)

abbrev PosNat := {N : Nat // 0 < N}

/-- Normalized zero-error block rate for the `N`-block product system. -/
noncomputable def blockRate
    {F A L : Nat}
    (views : ViewFamily F L)
    (N : PosNat) : ℝ :=
  Real.log (blockMaxIndependentCard (N := N.1) (F := F) (A := A) (L := L) views) / (N.1 : ℝ)

/-- Lower asymptotic zero-error capacity envelope generated by block rates. -/
noncomputable def shannonLowerCapacity
    {F A L : Nat}
    (views : ViewFamily F L) : ENNReal :=
  ⨆ N : PosNat, ENNReal.ofReal (blockRate (F := F) (A := A) (L := L) views N)

/-- Lower Shannon-capacity envelope of a graph via normalized independence growth on strong powers. -/
noncomputable def graphShannonLowerCapacity {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) : ENNReal :=
  ⨆ N : PosNat,
    ENNReal.ofReal (Real.log (graphMaxIndependentCard (strongPow G N.1)) / (N.1 : ℝ))

theorem graphMaxIndependentCard_strongPow_mul_lower_bound
    {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (K M : Nat) :
    graphMaxIndependentCard (strongPow G M) ^ K ≤
      graphMaxIndependentCard (strongPow G (K * M)) := by
  induction K with
  | zero =>
      let x : Fin 0 → α := fun i => nomatch i
      have hs : GraphIndependent (strongPow G 0) ({x} : Finset (Fin 0 → α)) := by
        intro u v hu hv hneq hAdj
        simp at hu hv
        subst hu
        subst hv
        exact hneq rfl
      have hbound : ({x} : Finset (Fin 0 → α)).card ≤ graphMaxIndependentCard (strongPow G 0) := by
        exact graphIndependent_card_le_graphMaxIndependentCard (strongPow G 0) ({x} : Finset (Fin 0 → α)) hs
      have hbound0 : (1 : Nat) ≤ graphMaxIndependentCard (strongPow G 0) := by
        simpa [x] using hbound
      have hzero : graphMaxIndependentCard (strongPow G (0 * M)) = graphMaxIndependentCard (strongPow G 0) := by
        rw [Nat.zero_mul]
      rw [pow_zero, hzero]
      exact hbound0
  | succ k ih =>
      have hprod :
          graphMaxIndependentCard (strongPow G (k * M)) *
            graphMaxIndependentCard (strongPow G M) ≤
            graphMaxIndependentCard (strongProd (strongPow G (k * M)) (strongPow G M)) :=
        graphMaxIndependentCard_strongProd_lower_bound (strongPow G (k * M)) (strongPow G M)
      have hpow :
          graphMaxIndependentCard (strongPow G M) ^ k *
            graphMaxIndependentCard (strongPow G M) ≤
            graphMaxIndependentCard (strongPow G (k * M)) *
              graphMaxIndependentCard (strongPow G M) :=
        Nat.mul_le_mul_right _ ih
      calc
        graphMaxIndependentCard (strongPow G M) ^ (k + 1)
            = graphMaxIndependentCard (strongPow G M) ^ k *
                graphMaxIndependentCard (strongPow G M) := by
                  simp [pow_succ]
        _ ≤ graphMaxIndependentCard (strongPow G (k * M)) *
              graphMaxIndependentCard (strongPow G M) := hpow
        _ ≤ graphMaxIndependentCard (strongProd (strongPow G (k * M)) (strongPow G M)) := hprod
        _ = graphMaxIndependentCard (strongPow G ((k * M) + M)) := by
              symm
              exact graphMaxIndependentCard_strongPow_add_eq_strongProd G (k * M) M
        _ = graphMaxIndependentCard (strongPow G ((k + 1) * M)) := by
              rw [Nat.succ_mul]


noncomputable def graphPowerRate {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (N : PosNat) : ℝ :=
  Real.log (graphMaxIndependentCard (strongPow G N.1)) / (N.1 : ℝ)

theorem graphPowerRate_le_graphShannonLowerCapacity
    {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (N : PosNat) :
    ENNReal.ofReal (graphPowerRate G N) ≤ graphShannonLowerCapacity G := by
  exact le_iSup (fun N : PosNat =>
    ENNReal.ofReal (Real.log (graphMaxIndependentCard (strongPow G N.1)) / (N.1 : ℝ))) N

theorem graphPowerRate_le_graphPowerRate_mul
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) (K M : PosNat) :
    graphPowerRate G M ≤ graphPowerRate G ⟨K.1 * M.1, Nat.mul_pos K.2 M.2⟩ := by
  let αM := graphMaxIndependentCard (strongPow G M.1)
  let αKM := graphMaxIndependentCard (strongPow G (K.1 * M.1))
  have hpow : αM ^ K.1 ≤ αKM := graphMaxIndependentCard_strongPow_mul_lower_bound G K.1 M.1
  have hαM : 0 < αM := by
    let x : Fin M.1 → α := fun _ => Classical.choice ‹Nonempty α›
    have hs : GraphIndependent (strongPow G M.1) ({x} : Finset (Fin M.1 → α)) := by
      intro u v hu hv hneq hAdj
      simp at hu hv
      subst hu
      subst hv
      exact hneq rfl
    have hbound : ({x} : Finset (Fin M.1 → α)).card ≤ αM := by
      simpa [αM] using graphIndependent_card_le_graphMaxIndependentCard (strongPow G M.1) ({x} : Finset (Fin M.1 → α)) hs
    simpa [x] using hbound
  have hαKM : 0 < αKM := lt_of_lt_of_le (pow_pos hαM _) hpow
  have hlog :
      Real.log ((αM : ℝ) ^ K.1) ≤ Real.log (αKM : ℝ) := by
    have hle : ((αM : ℝ) ^ K.1) ≤ (αKM : ℝ) := by
      exact_mod_cast hpow
    exact Real.log_le_log (by exact_mod_cast (pow_pos hαM _)) hle
  have hscaled :
      (K.1 : ℝ) * Real.log (αM : ℝ) ≤ Real.log (αKM : ℝ) := by
    simpa [Real.log_pow] using hlog
  have hK : 0 < (K.1 : ℝ) := by exact_mod_cast K.2
  have hM : 0 < (M.1 : ℝ) := by exact_mod_cast M.2
  have hdivK : Real.log (αM : ℝ) ≤ Real.log (αKM : ℝ) / (K.1 : ℝ) := by
    exact (le_div_iff₀ hK).2 (by simpa [mul_comm, mul_left_comm, mul_assoc] using hscaled)
  have hmul_pos : 0 < (M.1 : ℝ) := hM
  have hmul :
      Real.log (αM : ℝ) / (M.1 : ℝ) ≤ Real.log (αKM : ℝ) / ((K.1 * M.1 : Nat) : ℝ) := by
    have hdivM : Real.log (αM : ℝ) / (M.1 : ℝ) ≤ (Real.log (αKM : ℝ) / (K.1 : ℝ)) / (M.1 : ℝ) :=
      div_le_div_of_nonneg_right hdivK (le_of_lt hM)
    have hrewrite :
        (Real.log (αKM : ℝ) / (K.1 : ℝ)) / (M.1 : ℝ) =
      Real.log (αKM : ℝ) / ((K.1 * M.1 : Nat) : ℝ) := by
      rw [Nat.cast_mul]
      field_simp [show (K.1 : ℝ) ≠ 0 by exact_mod_cast (Nat.ne_of_gt K.2),
        show (M.1 : ℝ) ≠ 0 by exact_mod_cast (Nat.ne_of_gt M.2)]
    exact hdivM.trans (by simpa [hrewrite])
  simpa [graphPowerRate, αM, αKM, Nat.cast_mul, mul_assoc, mul_comm, mul_left_comm] using hmul

/-- Normalized graph-power log-independence rate on all blocklengths, with the `0`th term equal to `0`. -/
noncomputable def graphPowerRateNat {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (N : Nat) : ℝ :=
  Real.log (graphMaxIndependentCard (strongPow G N)) / (N : ℝ)

/-- Negative log-independence sequence used to invoke Fekete's lemma. -/
noncomputable def graphNegLogSeq {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (N : Nat) : ℝ :=
  -Real.log (graphMaxIndependentCard (strongPow G N))

theorem graphNegLogSeq_subadditive
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    Subadditive (graphNegLogSeq G) := by
  intro M N
  let αM := graphMaxIndependentCard (strongPow G M)
  let αN := graphMaxIndependentCard (strongPow G N)
  let αMN := graphMaxIndependentCard (strongPow G (M + N))
  have hMpos : 0 < αM := by
    let x : Fin M → α := fun _ => Classical.choice ‹Nonempty α›
    have hs : GraphIndependent (strongPow G M) ({x} : Finset (Fin M → α)) := by
      intro u v hu hv hneq hAdj
      simp at hu hv
      subst hu
      subst hv
      exact hneq rfl
    have hbound : ({x} : Finset (Fin M → α)).card ≤ αM := by
      simpa [αM] using
        graphIndependent_card_le_graphMaxIndependentCard (strongPow G M) ({x} : Finset (Fin M → α)) hs
    simpa [x] using hbound
  have hNpos : 0 < αN := by
    let x : Fin N → α := fun _ => Classical.choice ‹Nonempty α›
    have hs : GraphIndependent (strongPow G N) ({x} : Finset (Fin N → α)) := by
      intro u v hu hv hneq hAdj
      simp at hu hv
      subst hu
      subst hv
      exact hneq rfl
    have hbound : ({x} : Finset (Fin N → α)).card ≤ αN := by
      simpa [αN] using
        graphIndependent_card_le_graphMaxIndependentCard (strongPow G N) ({x} : Finset (Fin N → α)) hs
    simpa [x] using hbound
  have hMNle :
      (αM : ℝ) * (αN : ℝ) ≤ (αMN : ℝ) := by
    exact_mod_cast graphMaxIndependentCard_strongPow_add_lower_bound G M N
  have hlog :
      Real.log (αM : ℝ) + Real.log (αN : ℝ) ≤ Real.log (αMN : ℝ) := by
    have hMne : (αM : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hMpos)
    have hNne : (αN : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hNpos)
    have hmulpos : 0 < (αM : ℝ) * (αN : ℝ) := by positivity
    calc
      Real.log (αM : ℝ) + Real.log (αN : ℝ)
          = Real.log ((αM : ℝ) * (αN : ℝ)) := by
              rw [Real.log_mul hMne hNne]
      _ ≤ Real.log (αMN : ℝ) := Real.log_le_log hmulpos hMNle
  dsimp [graphNegLogSeq]
  linarith

theorem graphMaxIndependentCard_strongPow_le_card
    {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (N : Nat) :
    graphMaxIndependentCard (strongPow G N) ≤ Fintype.card (Fin N → α) := by
  rcases exists_graphIndependent_card_eq_graphMaxIndependentCard (strongPow G N) with ⟨s, hs, hcard⟩
  calc
    graphMaxIndependentCard (strongPow G N) = s.card := hcard.symm
    _ ≤ Fintype.card (Fin N → α) := Finset.card_le_univ s

theorem graphNegLogSeq_div_bddBelow
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    BddBelow (Set.range (fun N : Nat => graphNegLogSeq G N / (N : ℝ))) := by
  refine ⟨-Real.log (Fintype.card α), ?_⟩
  have hcardα : 0 < Fintype.card α := Fintype.card_pos_iff.mpr ‹Nonempty α›
  intro y hy
  rcases hy with ⟨N, rfl⟩
  by_cases hN : N = 0
  · subst hN
    have hzero : graphNegLogSeq G 0 / (0 : ℝ) = 0 := by
      simp [graphNegLogSeq]
    have hcardα_ge1 : (1 : ℝ) ≤ Fintype.card α := by
      exact_mod_cast (Nat.succ_le_of_lt hcardα)
    have hlog_nonneg : 0 ≤ Real.log (Fintype.card α : ℝ) := by
      simpa using Real.log_nonneg hcardα_ge1
    simpa [hzero] using hlog_nonneg
  have hNpos : 0 < N := Nat.pos_iff_ne_zero.mpr hN
  have hle_card :
      (graphMaxIndependentCard (strongPow G N) : ℝ) ≤ (Fintype.card α : ℝ) ^ N := by
    calc
      (graphMaxIndependentCard (strongPow G N) : ℝ)
          ≤ (Fintype.card (Fin N → α) : ℝ) := by
              exact_mod_cast graphMaxIndependentCard_strongPow_le_card G N
      _ = ((Fintype.card α ^ N : Nat) : ℝ) := by
            simpa using
              (Fintype.card_fun : Fintype.card (Fin N → α) = Fintype.card α ^ Fintype.card (Fin N))
      _ = (Fintype.card α : ℝ) ^ N := by
            exact_mod_cast rfl
  have hpos :
      0 < (graphMaxIndependentCard (strongPow G N) : ℝ) := by
    let x : Fin N → α := fun _ => Classical.choice ‹Nonempty α›
    have hs : GraphIndependent (strongPow G N) ({x} : Finset (Fin N → α)) := by
      intro u v hu hv hneq hAdj
      simp at hu hv
      subst hu
      subst hv
      exact hneq rfl
    have hbound : ({x} : Finset (Fin N → α)).card ≤ graphMaxIndependentCard (strongPow G N) := by
      exact graphIndependent_card_le_graphMaxIndependentCard (strongPow G N) ({x} : Finset (Fin N → α)) hs
    have : (1 : Nat) ≤ graphMaxIndependentCard (strongPow G N) := by simpa [x] using hbound
    exact_mod_cast this
  have hlog :
      Real.log (graphMaxIndependentCard (strongPow G N) : ℝ) ≤
        Real.log ((Fintype.card α : ℝ) ^ N) := by
    exact Real.log_le_log hpos hle_card
  have hpowpos : 0 < ((Fintype.card α : ℝ) ^ N) := by positivity
  have hcalc :
      Real.log ((Fintype.card α : ℝ) ^ N) = (N : ℝ) * Real.log (Fintype.card α : ℝ) := by
    rw [← Real.rpow_natCast]
    rw [Real.log_rpow (by positivity)]
  have hdiv :
      Real.log (graphMaxIndependentCard (strongPow G N) : ℝ) / (N : ℝ) ≤
        Real.log (Fintype.card α : ℝ) := by
    have hNreal : 0 < (N : ℝ) := by exact_mod_cast hNpos
    exact (_root_.div_le_iff₀ hNreal).2
      (by simpa [hcalc, mul_comm, mul_left_comm, mul_assoc] using hlog)
  have hfinal :
      -Real.log (Fintype.card α : ℝ) ≤
        graphNegLogSeq G N / (N : ℝ) := by
    have hnegdiv :
        -Real.log (Fintype.card α : ℝ) ≤
          -(Real.log (graphMaxIndependentCard (strongPow G N) : ℝ) / (N : ℝ)) := by
      linarith
    simpa [graphNegLogSeq, neg_div] using hnegdiv
  simpa using hfinal

/-- The real asymptotic Shannon capacity of a graph, obtained as the Fekete limit of the block rates. -/
noncomputable def graphShannonCapacityReal
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) : ℝ :=
  -(graphNegLogSeq_subadditive G).lim

theorem tendsto_graphPowerRateNat_graphShannonCapacityReal
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    Tendsto (graphPowerRateNat G) atTop (𝓝 (graphShannonCapacityReal G)) := by
  have hsub := graphNegLogSeq_subadditive G
  have htend :=
    Subadditive.tendsto_lim (h := hsub) (graphNegLogSeq_div_bddBelow G)
  have hneg :
      Tendsto (fun N : Nat => -(graphNegLogSeq G N / (N : ℝ))) atTop
        (𝓝 (-(hsub.lim))) := by
    simpa using htend.neg
  have hEq :
      (fun N : Nat => -(graphNegLogSeq G N / (N : ℝ))) = graphPowerRateNat G := by
    funext N
    dsimp [graphPowerRateNat, graphNegLogSeq]
    ring
  simpa [graphShannonCapacityReal, hEq] using hneg

theorem graphPowerRate_le_graphShannonCapacityReal
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) (N : PosNat) :
    graphPowerRate G N ≤ graphShannonCapacityReal G := by
  have hsub := graphNegLogSeq_subadditive G
  have hlim :=
    hsub.lim_le_div (graphNegLogSeq_div_bddBelow G) N.2.ne'
  have hneg := neg_le_neg hlim
  simpa [graphShannonCapacityReal, graphPowerRate, graphNegLogSeq, neg_div] using hneg

theorem graphShannonCapacityReal_nonneg
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    0 ≤ graphShannonCapacityReal G := by
  have hpos :
      0 < graphMaxIndependentCard (strongPow G 1) := by
    let x : Fin 1 → α := fun _ => Classical.choice ‹Nonempty α›
    have hs : GraphIndependent (strongPow G 1) ({x} : Finset (Fin 1 → α)) := by
      intro u v hu hv hneq hAdj
      simp at hu hv
      subst hu
      subst hv
      exact hneq rfl
    have hbound :
        ({x} : Finset (Fin 1 → α)).card ≤ graphMaxIndependentCard (strongPow G 1) := by
      exact graphIndependent_card_le_graphMaxIndependentCard (strongPow G 1) ({x} : Finset (Fin 1 → α)) hs
    simpa [x] using hbound
  have hlog : 0 ≤ Real.log (graphMaxIndependentCard (strongPow G 1) : ℝ) := by
    have hone : (1 : ℝ) ≤ graphMaxIndependentCard (strongPow G 1) := by
      exact_mod_cast (Nat.succ_le_of_lt hpos)
    simpa using Real.log_nonneg hone
  have hrate : 0 ≤ graphPowerRate G 1 := by
    simpa [graphPowerRate] using hlog
  exact le_trans hrate (graphPowerRate_le_graphShannonCapacityReal G 1)

theorem graphPowerRate_le_log_thetaWitness
    {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β] [Nonempty α]
    {G : SimpleGraph α}
    (W : ThetaWitness (β := β) G) (N : PosNat) :
    graphPowerRate G N ≤ Real.log W.bound := by
  have hle :
      (graphMaxIndependentCard (strongPow G N.1) : ℝ) ≤ W.bound ^ N.1 :=
    graphMaxIndependentCard_strongPow_le_thetaWitnessPow W N.1
  have hpos :
      0 < (graphMaxIndependentCard (strongPow G N.1) : ℝ) := by
    exact_mod_cast graphMaxIndependentCard_strongPow_pos G N.1
  have hWpos : 0 < W.bound := lt_of_lt_of_le zero_lt_one W.one_le_bound
  have hpowpos : 0 < W.bound ^ N.1 := by positivity
  have hlog :
      Real.log (graphMaxIndependentCard (strongPow G N.1) : ℝ) ≤ Real.log (W.bound ^ N.1) := by
    exact Real.log_le_log hpos hle
  have hcalc :
      Real.log (W.bound ^ N.1) = (N.1 : ℝ) * Real.log W.bound := by
    rw [← Real.rpow_natCast, Real.log_rpow]
    positivity
  have hNpos : 0 < (N.1 : ℝ) := by exact_mod_cast N.2
  have hdiv :
      Real.log (graphMaxIndependentCard (strongPow G N.1) : ℝ) / (N.1 : ℝ) ≤ Real.log W.bound := by
    exact (_root_.div_le_iff₀ hNpos).2
      (by simpa [hcalc, mul_assoc, mul_comm, mul_left_comm] using hlog)
  simpa [graphPowerRate] using hdiv

@[simp] theorem compl_strongPow_adj_iff_exists
    {α : Type} [DecidableEq α] {G : SimpleGraph α} {N : Nat}
    {x y : Fin N → α} :
    ((strongPow G N)ᶜ).Adj x y ↔ ∃ i, (Gᶜ).Adj (x i) (y i) := by
  rw [SimpleGraph.compl_adj]
  constructor
  · intro h
    rcases h with ⟨hxy, hnotAdj⟩
    by_contra hnone
    have hall : ∀ i, x i = y i ∨ G.Adj (x i) (y i) := by
      intro i
      by_cases hiEq : x i = y i
      · exact Or.inl hiEq
      · have hnotCompl : ¬ (Gᶜ).Adj (x i) (y i) := by
          exact fun hcompl => hnone ⟨i, hcompl⟩
        rw [SimpleGraph.compl_adj] at hnotCompl
        by_cases hiAdj : G.Adj (x i) (y i)
        · exact Or.inr hiAdj
        · exfalso
          exact hnotCompl ⟨hiEq, hiAdj⟩
    have hsomeAdj : ∃ i, G.Adj (x i) (y i) := by
      by_contra hnoneAdj
      apply hxy
      funext i
      rcases hall i with hiEq | hiAdj
      · exact hiEq
      · exfalso
        exact hnoneAdj ⟨i, hiAdj⟩
    exact hnotAdj ((strongPow_adj_iff (G := G) (x := x) (y := y)).2 ⟨hall, hsomeAdj⟩)
  · intro h
    rcases h with ⟨i, hi⟩
    rcases (SimpleGraph.compl_adj G (x i) (y i)).1 hi with ⟨hineq, hnotAdj⟩
    refine (SimpleGraph.compl_adj (strongPow G N) x y).2 ?_
    refine ⟨?_, ?_⟩
    · intro hxy
      exact hineq (congrFun hxy i)
    · intro hAdj
      rcases (strongPow_adj_iff (G := G) (x := x) (y := y)).1 hAdj with ⟨hall, _⟩
      rcases hall i with hiEq | hiAdj
      · exact hineq hiEq
      · exact hnotAdj hiAdj

def complStrongPowColoring
    {α : Type} [DecidableEq α] {G : SimpleGraph α} {n N : Nat}
    (C : Gᶜ.Coloring (Fin n)) :
    ((strongPow G N)ᶜ).Coloring (Fin N → Fin n) :=
  SimpleGraph.Coloring.mk
    (fun x i => C (x i))
    (by
      intro x y hAdj hEq
      rcases (compl_strongPow_adj_iff_exists (G := G) (x := x) (y := y)).1 hAdj with ⟨i, hi⟩
      exact (C.valid hi) (congrFun hEq i))

theorem complStrongPow_colorable
    {α : Type} [Fintype α] [DecidableEq α] {G : SimpleGraph α} {n N : Nat}
    (hc : Gᶜ.Colorable n) :
    ((strongPow G N)ᶜ).Colorable (n ^ N) := by
  rcases hc with ⟨C⟩
  let D : ((strongPow G N)ᶜ).Coloring (Fin N → Fin n) := complStrongPowColoring C
  have hD : ((strongPow G N)ᶜ).Colorable (Fintype.card (Fin N → Fin n)) := D.colorable
  simpa using hD

theorem graphIndependent_isClique_compl
    {α : Type} [DecidableEq α] {G : SimpleGraph α} {s : Finset α}
    (hs : GraphIndependent G s) :
    (Gᶜ).IsClique s := by
  intro v hv w hw hne
  exact (SimpleGraph.compl_adj G v w).2 ⟨hne, hs hv hw hne⟩

theorem graphMaxIndependentCard_strongPow_le_complChromaticPow
    {α : Type} [Fintype α] [DecidableEq α]
    (G : SimpleGraph α) (N n : Nat)
    (hc : Gᶜ.Colorable n) :
    graphMaxIndependentCard (strongPow G N) ≤ n ^ N := by
  rcases exists_graphIndependent_card_eq_graphMaxIndependentCard (strongPow G N) with ⟨s, hs, hcard⟩
  have hclique : ((strongPow G N)ᶜ).IsClique s :=
    graphIndependent_isClique_compl hs
  have hcolor : ((strongPow G N)ᶜ).Colorable (n ^ N) :=
    complStrongPow_colorable (G := G) (N := N) hc
  calc
    graphMaxIndependentCard (strongPow G N) = s.card := hcard.symm
    _ ≤ n ^ N := hclique.card_le_of_colorable hcolor

theorem graphPowerRate_le_log_complChromatic
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) (N : PosNat) {n : Nat}
    (hc : Gᶜ.Colorable n) :
    graphPowerRate G N ≤ Real.log n := by
  have hnpos : 0 < n := by
    by_contra hn
    have hzero : n = 0 := Nat.eq_zero_of_not_pos hn
    have hempty : IsEmpty α := by
      rw [← (Gᶜ).colorable_zero_iff]
      simpa [hzero] using hc
    exact (not_isEmpty_of_nonempty α) hempty
  have hle :
      (graphMaxIndependentCard (strongPow G N.1) : ℝ) ≤ (n : ℝ) ^ N.1 := by
    exact_mod_cast graphMaxIndependentCard_strongPow_le_complChromaticPow G N.1 n hc
  have hpos :
      0 < (graphMaxIndependentCard (strongPow G N.1) : ℝ) := by
    let x : Fin N.1 → α := fun _ => Classical.choice ‹Nonempty α›
    have hs : GraphIndependent (strongPow G N.1) ({x} : Finset (Fin N.1 → α)) := by
      intro u v hu hv hneq hAdj
      simp at hu hv
      subst hu
      subst hv
      exact hneq rfl
    have hbound : ({x} : Finset (Fin N.1 → α)).card ≤ graphMaxIndependentCard (strongPow G N.1) := by
      exact graphIndependent_card_le_graphMaxIndependentCard (strongPow G N.1) ({x} : Finset (Fin N.1 → α)) hs
    have hone : (1 : Nat) ≤ graphMaxIndependentCard (strongPow G N.1) := by
      simpa [x] using hbound
    exact_mod_cast hone
  have hlog :
      Real.log (graphMaxIndependentCard (strongPow G N.1) : ℝ) ≤ Real.log ((n : ℝ) ^ N.1) := by
    exact Real.log_le_log hpos hle
  have hcalc :
      Real.log ((n : ℝ) ^ N.1) = (N.1 : ℝ) * Real.log n := by
    rw [← Real.rpow_natCast, Real.log_rpow]
    positivity
  have hNpos : 0 < (N.1 : ℝ) := by exact_mod_cast N.2
  have hdiv :
      Real.log (graphMaxIndependentCard (strongPow G N.1) : ℝ) / (N.1 : ℝ) ≤ Real.log n := by
    exact (_root_.div_le_iff₀ hNpos).2 (by simpa [hcalc, mul_assoc, mul_comm, mul_left_comm] using hlog)
  simpa [graphPowerRate] using hdiv

theorem exists_graphPowerRate_gt_of_lt_graphShannonCapacityReal
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) {w : ℝ}
    (hw : w < graphShannonCapacityReal G) :
    ∃ N : PosNat, w < graphPowerRate G N := by
  let c := graphShannonCapacityReal G
  let ε : ℝ := (c - w) / 2
  have hε : 0 < ε := by
    dsimp [ε, c]
    linarith
  have hnhds : {x : ℝ | |x - c| < ε} ∈ 𝓝 c := by
    simpa [Real.dist_eq] using Metric.ball_mem_nhds c hε
  have hEv : ∀ᶠ n : Nat in atTop, |graphPowerRateNat G n - c| < ε :=
    tendsto_graphPowerRateNat_graphShannonCapacityReal G hnhds
  rcases Filter.eventually_atTop.1 hEv with ⟨N0, hN0⟩
  let Nnat : Nat := max N0 1
  have hNnat_pos : 0 < Nnat := by
    dsimp [Nnat]
    exact lt_of_lt_of_le Nat.zero_lt_one (Nat.le_max_right _ _)
  let N : PosNat := ⟨Nnat, hNnat_pos⟩
  have hclose : |graphPowerRateNat G Nnat - c| < ε := hN0 Nnat (Nat.le_max_left _ _)
  have hlow : c - ε < graphPowerRateNat G Nnat := by
    have habs := abs_lt.mp hclose
    linarith
  have hw' : w < c - ε := by
    dsimp [ε, c]
    linarith
  refine ⟨N, ?_⟩
  have : w < graphPowerRateNat G Nnat := lt_trans hw' hlow
  simpa [graphPowerRateNat, graphPowerRate, N, Nnat] using this

theorem iSup_graphPowerRate_eq_graphShannonCapacityReal
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    (⨆ N : PosNat, graphPowerRate G N) = graphShannonCapacityReal G := by
  refine ciSup_eq_of_forall_le_of_forall_lt_exists_gt
    (fun N => graphPowerRate_le_graphShannonCapacityReal G N) ?_
  intro w hw
  exact exists_graphPowerRate_gt_of_lt_graphShannonCapacityReal G hw

theorem graphShannonCapacityReal_le_log_thetaWitness
    {α β : Type} [Fintype α] [DecidableEq α] [Fintype β] [DecidableEq β] [Nonempty α]
    {G : SimpleGraph α}
    (W : ThetaWitness (β := β) G) :
    graphShannonCapacityReal G ≤ Real.log W.bound := by
  rw [← iSup_graphPowerRate_eq_graphShannonCapacityReal]
  exact ciSup_le (fun N => graphPowerRate_le_log_thetaWitness W N)

def graphThetaLogBoundSet
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) : Set ℝ :=
  { r | ∃ (β : Type) (_ : Fintype β) (_ : DecidableEq β) (_ : Nonempty β),
      ∃ W : ThetaWitness (β := β) G, r = Real.log W.bound }

noncomputable def graphThetaLogUpper
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) : ℝ :=
  sInf (graphThetaLogBoundSet G)

theorem graphThetaLogBoundSet_nonempty
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    (graphThetaLogBoundSet G).Nonempty := by
  classical
  refine ⟨Real.log (Fintype.card α), ?_⟩
  refine ⟨α, inferInstance, inferInstance, inferInstance, ?_, ?_⟩
  · exact thetaWitnessOfColoring ((Gᶜ).selfColoring)
  · rfl

theorem graphThetaLogBoundSet_bddBelow
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    BddBelow (graphThetaLogBoundSet G) := by
  refine ⟨0, ?_⟩
  intro r hr
  rcases hr with ⟨β, _, _, _, W, rfl⟩
  exact Real.log_nonneg W.one_le_bound

theorem graphShannonCapacityReal_le_graphThetaLogUpper
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    graphShannonCapacityReal G ≤ graphThetaLogUpper G := by
  dsimp [graphThetaLogUpper]
  exact le_csInf
    (s := graphThetaLogBoundSet G)
    (a := graphShannonCapacityReal G)
    (graphThetaLogBoundSet_nonempty G)
    (by
      intro r hr
      rcases hr with ⟨β, _, _, _, W, rfl⟩
      exact graphShannonCapacityReal_le_log_thetaWitness W)

theorem graphThetaLogUpper_le_log_complChromatic
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) {n : Nat}
    (hc : Gᶜ.Colorable n) :
    graphThetaLogUpper G ≤ Real.log n := by
  have hnpos : 0 < n := by
    by_contra hn
    have hzero : n = 0 := Nat.eq_zero_of_not_pos hn
    have hempty : IsEmpty α := by
      rw [← (Gᶜ).colorable_zero_iff]
      simpa [hzero] using hc
    exact (not_isEmpty_of_nonempty α) hempty
  rcases hc with ⟨C⟩
  haveI : Nonempty (Fin n) := ⟨⟨0, hnpos⟩⟩
  have hmem : Real.log n ∈ graphThetaLogBoundSet G := by
    refine ⟨Fin n, inferInstance, inferInstance, inferInstance, ?_, ?_⟩
    · exact thetaWitnessOfColoring C
    · change Real.log n = Real.log (Fintype.card (Fin n))
      simp
  dsimp [graphThetaLogUpper]
  exact csInf_le (graphThetaLogBoundSet_bddBelow G) hmem

theorem graphThetaLogUpper_le_log_complChromaticNumber
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    graphThetaLogUpper G ≤ Real.log (ENat.toNat (Gᶜ).chromaticNumber) := by
  have hc : Gᶜ.Colorable (ENat.toNat (Gᶜ).chromaticNumber) :=
    SimpleGraph.colorable_chromaticNumber_of_fintype (G := Gᶜ)
  exact graphThetaLogUpper_le_log_complChromatic G hc

theorem graphThetaLogUpper_strongProd_le
    {α γ : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    [Fintype γ] [DecidableEq γ] [Nonempty γ]
    (G : SimpleGraph α) (H : SimpleGraph γ) :
    graphThetaLogUpper (strongProd G H) ≤ graphThetaLogUpper G + graphThetaLogUpper H := by
  refine le_of_forall_pos_lt_add ?_
  intro ε hε
  have hltG :
      sInf (graphThetaLogBoundSet G) < graphThetaLogUpper G + ε / 2 := by
    dsimp [graphThetaLogUpper]
    linarith
  have hltH :
      sInf (graphThetaLogBoundSet H) < graphThetaLogUpper H + ε / 2 := by
    dsimp [graphThetaLogUpper]
    linarith
  rcases exists_lt_of_csInf_lt (graphThetaLogBoundSet_nonempty G) hltG with
    ⟨rG, hrGmem, hrGlt⟩
  rcases exists_lt_of_csInf_lt (graphThetaLogBoundSet_nonempty H) hltH with
    ⟨rH, hrHmem, hrHlt⟩
  rcases hrGmem with ⟨β, _, _, _, WG, hrGdef⟩
  rcases hrHmem with ⟨δ, _, _, _, WH, hrHdef⟩
  have hmemProd :
      Real.log (WG.bound * WH.bound) ∈ graphThetaLogBoundSet (strongProd G H) := by
    refine ⟨β × δ, inferInstance, inferInstance, inferInstance, ThetaWitness.prod WG WH, ?_⟩
    simp [ThetaWitness.prod]
  have hupper :
      graphThetaLogUpper (strongProd G H) ≤ Real.log (WG.bound * WH.bound) := by
    dsimp [graphThetaLogUpper]
    exact csInf_le (graphThetaLogBoundSet_bddBelow (strongProd G H)) hmemProd
  have hWGpos : 0 < WG.bound := lt_of_lt_of_le zero_lt_one WG.one_le_bound
  have hWHpos : 0 < WH.bound := lt_of_lt_of_le zero_lt_one WH.one_le_bound
  have hlogmul :
      Real.log (WG.bound * WH.bound) = Real.log WG.bound + Real.log WH.bound := by
    rw [Real.log_mul (ne_of_gt hWGpos) (ne_of_gt hWHpos)]
  calc
    graphThetaLogUpper (strongProd G H)
      ≤ Real.log (WG.bound * WH.bound) := hupper
    _ = rG + rH := by rw [hlogmul, ← hrGdef, ← hrHdef]
    _ < (graphThetaLogUpper G + ε / 2) + (graphThetaLogUpper H + ε / 2) := by
          linarith
    _ = graphThetaLogUpper G + graphThetaLogUpper H + ε := by ring

theorem graphShannonCapacityReal_le_log_complChromatic
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) {n : Nat}
    (hc : Gᶜ.Colorable n) :
    graphShannonCapacityReal G ≤ Real.log n := by
  rw [← iSup_graphPowerRate_eq_graphShannonCapacityReal]
  exact ciSup_le (fun N => graphPowerRate_le_log_complChromatic G N hc)

theorem graphShannonCapacityReal_le_log_complChromaticNumber
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    graphShannonCapacityReal G ≤ Real.log (ENat.toNat (Gᶜ).chromaticNumber) := by
  have hc : Gᶜ.Colorable (ENat.toNat (Gᶜ).chromaticNumber) :=
    SimpleGraph.colorable_chromaticNumber_of_fintype (G := Gᶜ)
  exact graphShannonCapacityReal_le_log_complChromatic G hc

theorem graphShannonLowerCapacity_eq_ofReal_graphShannonCapacityReal
    {α : Type} [Fintype α] [DecidableEq α] [Nonempty α]
    (G : SimpleGraph α) :
    graphShannonLowerCapacity G = ENNReal.ofReal (graphShannonCapacityReal G) := by
  have hbdd : BddAbove (Set.range (graphPowerRate G)) := by
    refine ⟨graphShannonCapacityReal G, ?_⟩
    intro y hy
    rcases hy with ⟨N, rfl⟩
    exact graphPowerRate_le_graphShannonCapacityReal G N
  have hmap :
      ENNReal.ofReal (⨆ N : PosNat, graphPowerRate G N) =
        ⨆ N : PosNat, ENNReal.ofReal (graphPowerRate G N) :=
    Monotone.map_ciSup_of_continuousAt
      (ι := PosNat)
      (f := ENNReal.ofReal)
      (g := graphPowerRate G)
      ENNReal.continuous_ofReal.continuousAt
      (fun a b hab => ENNReal.ofReal_le_ofReal hab)
      hbdd
  calc
    graphShannonLowerCapacity G = ⨆ N : PosNat, ENNReal.ofReal (graphPowerRate G N) := rfl
    _ = ENNReal.ofReal (⨆ N : PosNat, graphPowerRate G N) := by
          simpa using hmap.symm
    _ = ENNReal.ofReal (graphShannonCapacityReal G) := by
          rw [iSup_graphPowerRate_eq_graphShannonCapacityReal]

/-- The real asymptotic zero-error capacity of a view family, via the confusability graph. -/
noncomputable def shannonCapacityReal
    {F A L : Nat} [Nonempty (State F A)]
    (views : ViewFamily F L) : ℝ :=
  graphShannonCapacityReal (confusabilityGraph (A := A) views)

/-- A canonical witness-based asymptotic upper invariant for a view family, obtained by
taking the infimum of all graph-level theta-style witness bounds on the confusability graph. -/
noncomputable def shannonThetaLogUpper
    {F A L : Nat} [Nonempty (State F A)]
    (views : ViewFamily F L) : ℝ :=
  graphThetaLogUpper (confusabilityGraph (A := A) views)

theorem shannonCapacityReal_eq_iSup_blockRate
    {F A L : Nat} [Nonempty (State F A)]
    (views : ViewFamily F L) :
    (⨆ N : PosNat, blockRate (F := F) (A := A) (L := L) views N) =
      shannonCapacityReal (F := F) (A := A) (L := L) views := by
  classical
  simpa [shannonCapacityReal, blockRate, blockMaxIndependentCard_eq_graphMaxIndependentCard,
    blockConfusabilityGraph_eq_strongPow]
    using iSup_graphPowerRate_eq_graphShannonCapacityReal
      (G := confusabilityGraph (A := A) views)

theorem blockRate_le_shannonCapacityReal
    {F A L : Nat} [Nonempty (State F A)]
    (views : ViewFamily F L)
    (N : PosNat) :
    blockRate (F := F) (A := A) (L := L) views N ≤
      shannonCapacityReal (F := F) (A := A) (L := L) views := by
  have h := graphPowerRate_le_graphShannonCapacityReal
      (G := confusabilityGraph (A := A) views) N
  simpa [shannonCapacityReal, blockRate, blockMaxIndependentCard_eq_graphMaxIndependentCard,
    blockConfusabilityGraph_eq_strongPow] using h

theorem shannonCapacityReal_le_shannonThetaLogUpper
    {F A L : Nat} [Nonempty (State F A)]
    (views : ViewFamily F L) :
    shannonCapacityReal (F := F) (A := A) (L := L) views ≤
      shannonThetaLogUpper (F := F) (A := A) (L := L) views := by
  simpa [shannonCapacityReal, shannonThetaLogUpper] using
    graphShannonCapacityReal_le_graphThetaLogUpper
      (G := confusabilityGraph (A := A) views)

theorem shannonLowerCapacity_eq_iSup_graphRates
    {F A L : Nat}
    (views : ViewFamily F L) :
    shannonLowerCapacity (F := F) (A := A) (L := L) views =
      ⨆ N : PosNat,
        ENNReal.ofReal
          (Real.log (graphMaxIndependentCard
            (blockConfusabilityGraph (N := N.1) (F := F) (A := A) views)) / (N.1 : ℝ)) := by
  simp [shannonLowerCapacity, blockRate, blockMaxIndependentCard_eq_graphMaxIndependentCard]

theorem shannonLowerCapacity_eq_graphShannonLowerCapacity
    {F A L : Nat}
    (views : ViewFamily F L) :
    shannonLowerCapacity (F := F) (A := A) (L := L) views =
      graphShannonLowerCapacity (confusabilityGraph (A := A) views) := by
  classical
  simp [shannonLowerCapacity, graphShannonLowerCapacity, blockRate,
    blockMaxIndependentCard_eq_graphMaxIndependentCard, blockConfusabilityGraph_eq_strongPow]

theorem blockRate_le_shannonLowerCapacity
    {F A L : Nat}
    (views : ViewFamily F L)
    (N : PosNat) :
    ENNReal.ofReal (blockRate (F := F) (A := A) (L := L) views N) ≤
      shannonLowerCapacity (F := F) (A := A) (L := L) views := by
  exact le_iSup (fun N : PosNat => ENNReal.ofReal (blockRate (F := F) (A := A) (L := L) views N)) N

theorem oneShotLogRate_le_blockRate
    {F A L : Nat}
    (views : ViewFamily F L)
    [Nonempty (State F A)]
    (N : PosNat) :
    Real.log (maxIndependentCard (F := F) (A := A) (L := L) views) ≤
      blockRate (F := F) (A := A) (L := L) views N := by
  let α₁ := maxIndependentCard (F := F) (A := A) (L := L) views
  let αN := blockMaxIndependentCard (N := N.1) (F := F) (A := A) (L := L) views
  have hα₁ : 0 < α₁ := maxIndependentCard_pos views
  have hpow :
      α₁ ^ N.1 ≤ αN := by
    simpa [α₁, αN] using block_power_independent_lower_bound (N := N.1) (F := F) (A := A) (L := L) views
  have hαN : 0 < αN := lt_of_lt_of_le (pow_pos hα₁ _) hpow
  have hlog :
      Real.log ((α₁ : ℝ) ^ N.1) ≤ Real.log (αN : ℝ) := by
    have hle : ((α₁ : ℝ) ^ N.1) ≤ (αN : ℝ) := by
      exact_mod_cast hpow
    exact Real.log_le_log (by exact_mod_cast (pow_pos hα₁ _)) hle
  have hscaled :
      (N.1 : ℝ) * Real.log (α₁ : ℝ) ≤ Real.log (αN : ℝ) := by
    simpa [Real.log_pow] using hlog
  have hN : 0 < (N.1 : ℝ) := by
    exact_mod_cast N.2
  have hdiv :
      Real.log (α₁ : ℝ) ≤ Real.log (αN : ℝ) / (N.1 : ℝ) := by
    exact (le_div_iff₀ hN).2 (by simpa [mul_comm, mul_left_comm, mul_assoc] using hscaled)
  simpa [blockRate, α₁, αN] using hdiv

theorem oneShotLogRate_le_shannonLowerCapacity
    {F A L : Nat}
    (views : ViewFamily F L)
    [Nonempty (State F A)] :
    ENNReal.ofReal (Real.log (maxIndependentCard (F := F) (A := A) (L := L) views)) ≤
      shannonLowerCapacity (F := F) (A := A) (L := L) views := by
  let N₁ : PosNat := ⟨1, by decide⟩
  exact (ENNReal.ofReal_le_ofReal (oneShotLogRate_le_blockRate views N₁)).trans
    (blockRate_le_shannonLowerCapacity views N₁)

theorem weighted_average_blockRate_le_blockRate_add
    {F A L : Nat}
    (views : ViewFamily F L)
    [Nonempty (State F A)]
    (M N : PosNat) :
    ((M.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views M +
        (N.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views N) /
        ((M.1 + N.1 : Nat) : ℝ)
      ≤
      blockRate (F := F) (A := A) (L := L) views
        ⟨M.1 + N.1, lt_of_lt_of_le M.2 (Nat.le_add_right M.1 N.1)⟩ := by
  let αM := blockMaxIndependentCard (N := M.1) (F := F) (A := A) (L := L) views
  let αN := blockMaxIndependentCard (N := N.1) (F := F) (A := A) (L := L) views
  let αMN := blockMaxIndependentCard (N := M.1 + N.1) (F := F) (A := A) (L := L) views
  have hprod : αM * αN ≤ αMN := by
    simpa [αM, αN, αMN] using
      block_add_independent_lower_bound (M := M.1) (N := N.1) (F := F) (A := A) (L := L) views
  have hαM : 0 < αM := by simpa [αM] using blockMaxIndependentCard_pos (N := M.1) (F := F) (A := A) (L := L) views
  have hαN : 0 < αN := by simpa [αN] using blockMaxIndependentCard_pos (N := N.1) (F := F) (A := A) (L := L) views
  have hαMN : 0 < αMN := lt_of_lt_of_le (Nat.mul_pos hαM hαN) hprod
  have hαM0 : (αM : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hαM)
  have hαN0 : (αN : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hαN)
  have hlog :
      Real.log (αM : ℝ) + Real.log (αN : ℝ) ≤ Real.log (αMN : ℝ) := by
    have hle : ((αM : ℝ) * (αN : ℝ)) ≤ (αMN : ℝ) := by
      exact_mod_cast hprod
    have hlog' : Real.log ((αM : ℝ) * (αN : ℝ)) ≤ Real.log (αMN : ℝ) :=
      Real.log_le_log (by exact_mod_cast (Nat.mul_pos hαM hαN)) hle
    rw [Real.log_mul hαM0 hαN0] at hlog'
    exact hlog'
  have hM0 : (M.1 : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt M.2)
  have hN0 : (N.1 : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt N.2)
  have hnum :
      (M.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views M +
        (N.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views N
      ≤ Real.log (αMN : ℝ) := by
    have hM :
        (M.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views M = Real.log (αM : ℝ) := by
      change (M.1 : ℝ) * (Real.log (αM : ℝ) / (M.1 : ℝ)) = Real.log (αM : ℝ)
      field_simp [hM0]
    have hN :
        (N.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views N = Real.log (αN : ℝ) := by
      change (N.1 : ℝ) * (Real.log (αN : ℝ) / (N.1 : ℝ)) = Real.log (αN : ℝ)
      field_simp [hN0]
    calc
      (M.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views M +
          (N.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views N
        = Real.log (αM : ℝ) + Real.log (αN : ℝ) := by rw [hM, hN]
      _ ≤ Real.log (αMN : ℝ) := hlog
  have hden : 0 < ((M.1 + N.1 : Nat) : ℝ) := by
    exact_mod_cast (lt_of_lt_of_le M.2 (Nat.le_add_right M.1 N.1))
  have hdiv :
      ((M.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views M +
          (N.1 : ℝ) * blockRate (F := F) (A := A) (L := L) views N) /
          ((M.1 + N.1 : Nat) : ℝ)
        ≤
        Real.log (αMN : ℝ) / ((M.1 + N.1 : Nat) : ℝ) := by
    exact div_le_div_of_nonneg_right hnum hden.le
  simpa [blockRate, αMN] using hdiv

theorem blockRate_le_blockRate_mul
    {F A L : Nat}
    (views : ViewFamily F L)
    [Nonempty (State F A)]
    (M K : PosNat) :
    blockRate (F := F) (A := A) (L := L) views M ≤
      blockRate (F := F) (A := A) (L := L) views ⟨K.1 * M.1, Nat.mul_pos K.2 M.2⟩ := by
  let αM := blockMaxIndependentCard (N := M.1) (F := F) (A := A) (L := L) views
  let αKM := blockMaxIndependentCard (N := K.1 * M.1) (F := F) (A := A) (L := L) views
  have hpow : αM ^ K.1 ≤ αKM := by
    simpa [αM, αKM] using block_mul_independent_lower_bound (K := K.1) (M := M.1) (F := F) (A := A) (L := L) views
  have hαM : 0 < αM := by simpa [αM] using blockMaxIndependentCard_pos (N := M.1) (F := F) (A := A) (L := L) views
  have hαKM : 0 < αKM := lt_of_lt_of_le (pow_pos hαM _) hpow
  have hlog :
      (K.1 : ℝ) * Real.log (αM : ℝ) ≤ Real.log (αKM : ℝ) := by
    have hle : ((αM : ℝ) ^ K.1) ≤ (αKM : ℝ) := by exact_mod_cast hpow
    have hlog' : Real.log ((αM : ℝ) ^ K.1) ≤ Real.log (αKM : ℝ) := by
      exact Real.log_le_log (by exact_mod_cast (pow_pos hαM _)) hle
    simpa [Real.log_pow] using hlog'
  have hK : 0 < (K.1 : ℝ) := by exact_mod_cast K.2
  have hM : 0 < (M.1 : ℝ) := by exact_mod_cast M.2
  have hKM : 0 < ((K.1 * M.1 : Nat) : ℝ) := by exact_mod_cast (Nat.mul_pos K.2 M.2)
  have hdivK :
      Real.log (αM : ℝ) ≤ Real.log (αKM : ℝ) / (K.1 : ℝ) := by
    exact (le_div_iff₀ hK).2 (by simpa [mul_comm, mul_left_comm, mul_assoc] using hlog)
  have hdivM :
      Real.log (αM : ℝ) / (M.1 : ℝ) ≤ (Real.log (αKM : ℝ) / (K.1 : ℝ)) / (M.1 : ℝ) := by
    exact div_le_div_of_nonneg_right hdivK hM.le
  have hrewrite :
      (Real.log (αKM : ℝ) / (K.1 : ℝ)) / (M.1 : ℝ) =
        Real.log (αKM : ℝ) / ((K.1 * M.1 : Nat) : ℝ) := by
    have hK0 : (K.1 : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt K.2)
    have hM0 : (M.1 : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt M.2)
    rw [div_eq_mul_inv, div_eq_mul_inv, div_eq_mul_inv]
    simp [Nat.cast_mul, mul_assoc, mul_comm, hK0, hM0]
  calc
    blockRate (F := F) (A := A) (L := L) views M
      = Real.log (αM : ℝ) / (M.1 : ℝ) := by simp [blockRate, αM]
    _ ≤ (Real.log (αKM : ℝ) / (K.1 : ℝ)) / (M.1 : ℝ) := hdivM
    _ = Real.log (αKM : ℝ) / ((K.1 * M.1 : Nat) : ℝ) := hrewrite
    _ = blockRate (F := F) (A := A) (L := L) views ⟨K.1 * M.1, Nat.mul_pos K.2 M.2⟩ := by
      simp [blockRate, αKM]

/-- Any exact recovery scheme yields a proper coloring, so every clique is bounded by the tag alphabet. -/
theorem clique_card_le_tag_alphabet
    {F A L T : Nat}
    (views : ViewFamily F L)
    (tag : State F A → Fin T)
    (decode : Fin L → PartialObservation F A → Fin T → Option (State F A))
    (hexact : ExactRecovery views tag decode)
    {s : Finset (State F A)}
    (hs : IsClique views s) :
    s.card ≤ T := by
  classical
  have hcolor : ProperColoring views tag :=
    exactRecovery_implies_properColoring views tag decode hexact
  have hinj : Set.InjOn tag {x | x ∈ s} := by
    intro x hx y hy htag
    by_cases hEq : x = y
    · exact hEq
    · exfalso
      exact (hcolor (hs hx hy hEq)) htag
  have hcard : s.card = (s.image tag).card := by
    symm
    exact Finset.card_image_of_injOn (by
      intro a ha b hb hab
      exact hinj ha hb hab)
  calc
    s.card = (s.image tag).card := hcard
    _ ≤ (Finset.univ : Finset (Fin T)).card := Finset.card_le_card (by
      intro t ht
      simp)
    _ = T := by simp

/-! ## A first non-clique confusability example -/

/-- Binary two-fact states written explicitly for a concrete four-cycle example. -/
def bitState (b₀ b₁ : Fin 2) : State 2 2
  | ⟨0, _⟩ => b₀
  | ⟨1, _⟩ => b₁

def s00 : State 2 2 := bitState 0 0
def s01 : State 2 2 := bitState 0 1
def s10 : State 2 2 := bitState 1 0
def s11 : State 2 2 := bitState 1 1

/-- Two locations, each revealing exactly one coordinate. -/
def binaryViews : ViewFamily 2 2
  | ⟨0, _⟩ => {⟨0, by decide⟩}
  | ⟨1, _⟩ => {⟨1, by decide⟩}

@[simp] theorem s00_zero : s00 ⟨0, by decide⟩ = 0 := rfl
@[simp] theorem s00_one : s00 ⟨1, by decide⟩ = 0 := rfl
@[simp] theorem s01_zero : s01 ⟨0, by decide⟩ = 0 := rfl
@[simp] theorem s01_one : s01 ⟨1, by decide⟩ = 1 := rfl
@[simp] theorem s10_zero : s10 ⟨0, by decide⟩ = 1 := rfl
@[simp] theorem s10_one : s10 ⟨1, by decide⟩ = 0 := rfl
@[simp] theorem s11_zero : s11 ⟨0, by decide⟩ = 1 := rfl
@[simp] theorem s11_one : s11 ⟨1, by decide⟩ = 1 := rfl

theorem s00_ne_s11 : s00 ≠ s11 := by
  decide

theorem s00_confusable_s01 : Confusable binaryViews s00 s01 := by
  decide

theorem s01_confusable_s11 : Confusable binaryViews s01 s11 := by
  decide

theorem s00_not_confusable_s11 : ¬ Confusable binaryViews s00 s11 := by
  decide

/-- The confusability graph from the binary single-coordinate views is not a clique. -/
theorem binaryViews_nonclique_witness :
    Confusable binaryViews s00 s01 ∧
    Confusable binaryViews s01 s11 ∧
    ¬ Confusable binaryViews s00 s11 := by
  exact ⟨s00_confusable_s01, s01_confusable_s11, s00_not_confusable_s11⟩

theorem same_firstCoord_confusable
    {x y : State 2 2}
    (hxy : x ≠ y)
    (hfirst : x ⟨0, by decide⟩ = y ⟨0, by decide⟩) :
    Confusable binaryViews x y := by
  refine ⟨hxy, ⟨⟨0, by decide⟩, ?_⟩⟩
  funext i
  fin_cases i
  · simp [observe, binaryViews]
    simpa using hfirst
  · simp [observe, binaryViews]

theorem same_secondCoord_confusable
    {x y : State 2 2}
    (hxy : x ≠ y)
    (hsecond : x ⟨1, by decide⟩ = y ⟨1, by decide⟩) :
    Confusable binaryViews x y := by
  refine ⟨hxy, ⟨⟨1, by decide⟩, ?_⟩⟩
  funext i
  fin_cases i
  · simp [observe, binaryViews]
  · simp [observe, binaryViews]
    simpa using hsecond

/-- A two-coloring for the binary four-cycle: equal bits get color `0`, unequal bits get color `1`. -/
def parityTag : State 2 2 → Fin 2 :=
  fun x => if x ⟨0, by decide⟩ = x ⟨1, by decide⟩ then 0 else 1

theorem parityTag_properColoring : ProperColoring binaryViews parityTag := by
  decide

/-- The binary non-clique example is exactly recoverable with two tags. -/
theorem binaryViews_exactRecovery_two_tags :
    ∃ decode : Fin 2 → PartialObservation 2 2 → Fin 2 → Option (State 2 2),
      ExactRecovery binaryViews parityTag decode := by
  exact properColoring_implies_exactRecovery binaryViews parityTag parityTag_properColoring

theorem binaryViews_colorable_two : Colorable (F := 2) (A := 2) (L := 2) (T := 2) binaryViews := by
  exact ⟨parityTag, parityTag_properColoring⟩

/-- No one-tag architecture can exactly recover a nontrivial confusable pair. -/
theorem binaryViews_no_exactRecovery_one_tag :
    ¬ ∃ decode : Fin 2 → PartialObservation 2 2 → Fin 1 → Option (State 2 2),
        ExactRecovery binaryViews (fun _ => (0 : Fin 1)) decode := by
  intro h
  rcases h with ⟨decode, hexact⟩
  have hcolor : ProperColoring binaryViews (fun _ => (0 : Fin 1)) :=
    exactRecovery_implies_properColoring binaryViews (fun _ => (0 : Fin 1)) decode hexact
  have hconf : Confusable binaryViews s00 s01 := s00_confusable_s01
  exact (hcolor hconf) rfl

theorem binaryViews_not_colorable_one :
    ¬ Colorable (F := 2) (A := 2) (L := 2) (T := 1) binaryViews := by
  intro h
  rcases h with ⟨tag, hcolor⟩
  have hconst : tag s00 = tag s01 := Subsingleton.elim _ _
  exact (hcolor s00_confusable_s01) hconst

theorem binaryViews_independent_card_le_two
    (s : Finset (State 2 2))
    (hindep : Independent binaryViews s) :
    s.card ≤ 2 := by
  classical
  let firstCoord : State 2 2 → Fin 2 := fun x => x ⟨0, by decide⟩
  have hinj : Set.InjOn firstCoord {x | x ∈ s} := by
    intro x hx y hy hfirst
    by_cases hEq : x = y
    · exact hEq
    · exfalso
      exact hindep hx hy hEq (same_firstCoord_confusable hEq hfirst)
  have hcard : s.card = (s.image firstCoord).card := by
    symm
    exact Finset.card_image_of_injOn (by
      intro a ha b hb hab
      exact hinj ha hb hab)
  calc
    s.card = (s.image firstCoord).card := hcard
    _ ≤ (Finset.univ : Finset (Fin 2)).card := Finset.card_le_card (by
      intro t ht
      simp)
    _ = 2 := by simp

theorem binaryViews_oneTag_exactOn_card_le_two
    (decode : Fin 2 → PartialObservation 2 2 → Fin 1 → Option (State 2 2))
    (s : Finset (State 2 2))
    (hexact : ExactOn binaryViews (fun _ => (0 : Fin 1)) decode s) :
    s.card ≤ 2 := by
  apply binaryViews_independent_card_le_two
  intro x y hx hy hxy hconf
  have hneq :=
    exactOn_implies_properColoringOn binaryViews (fun _ => (0 : Fin 1)) decode hexact hx hy hconf
  exact hneq (Subsingleton.elim _ _)

theorem binaryViews_oneTag_uniform_successProb_le_half
    (decode : Fin 2 → PartialObservation 2 2 → Fin 1 → Option (State 2 2))
    (s : Finset (State 2 2))
    (hexact : ExactOn binaryViews (fun _ => (0 : Fin 1)) decode s) :
    ((s.card : ℝ) / 4) ≤ (1 / 2 : ℝ) := by
  have hcard : s.card ≤ 2 := binaryViews_oneTag_exactOn_card_le_two decode s hexact
  norm_num
  have hs : (s.card : ℝ) ≤ 2 := by exact_mod_cast hcard
  linarith

theorem binaryViews_maxColorableCard_one_eq_two :
    maxColorableCard (F := 2) (A := 2) (L := 2) (T := 1) binaryViews (by decide) = 2 := by
  apply le_antisymm
  · rcases exists_exactOn_card_eq_maxColorableCard
      (F := 2) (A := 2) (L := 2) (T := 1) binaryViews (by decide) with
        ⟨s, tag, decode, hexact, hcard⟩
    have hconst : tag = fun _ => (0 : Fin 1) := by
      funext x
      exact Subsingleton.elim _ _
    have hbound : s.card ≤ 2 := by
      simpa [hconst] using binaryViews_oneTag_exactOn_card_le_two decode s (by simpa [hconst] using hexact)
    simpa [hcard] using hbound
  · let s : Finset (State 2 2) := {s00, s11}
    have hscol : ColorableOn (F := 2) (A := 2) (L := 2) (T := 1) binaryViews s := by
      decide
    have hcard : s.card = 2 := by decide
    have hbound := colorableOn_card_le_maxColorableCard
      (F := 2) (A := 2) (L := 2) (T := 1) binaryViews s (by decide) hscol
    simpa [hcard] using hbound

end MultiFact
