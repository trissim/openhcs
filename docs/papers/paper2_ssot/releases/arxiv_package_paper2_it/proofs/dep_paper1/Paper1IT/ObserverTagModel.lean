import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Prod
import Mathlib.Data.Finset.Card

namespace ObserverModel


/-- The finite ambiguity fiber induced by an observation value. -/
def observationFiber {K O : Nat} (obs : Fin K → Fin O) (o : Fin O) :=
  { v : Fin K // obs v = o }

/-- Exact zero-error recovery from `(observation, tag)` forces pair injectivity. -/
theorem exact_recovery_implies_pair_injective
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hzero : ∀ v, decode (obs v) (tag v) = some v) :
    Function.Injective (fun v => (obs v, tag v)) := by
  intro v₁ v₂ hpair
  have hdecode := congrArg (fun p : Fin O × Fin T => decode p.1 p.2) hpair
  simpa [hzero v₁, hzero v₂] using hdecode

/-- Pair injectivity suffices to build an exact zero-error decoder. -/
theorem pair_injective_implies_exact_recovery
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hinj : Function.Injective (fun v => (obs v, tag v))) :
    ∃ decode : Fin O → Fin T → Option (Fin K),
      ∀ v, decode (obs v) (tag v) = some v := by
  classical
  refine ⟨fun o t =>
    if h : ∃ v : Fin K, (obs v, tag v) = (o, t)
    then some (Classical.choose h)
    else none, ?_⟩
  intro v
  have hex : ∃ w : Fin K, (obs w, tag w) = (obs v, tag v) := ⟨v, rfl⟩
  have hchoose : Classical.choose hex = v := by
    apply hinj
    exact Classical.choose_spec hex
  change (if h : ∃ w : Fin K, (obs w, tag w) = (obs v, tag v)
    then some (Classical.choose h)
    else none) = some v
  rw [dif_pos hex]
  simpa using hchoose

/-- Exact zero-error recovery is equivalent to injectivity of the `(obs, tag)` pair map. -/
theorem exact_recovery_iff_pair_injective
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T) :
    (∃ decode : Fin O → Fin T → Option (Fin K),
        ∀ v, decode (obs v) (tag v) = some v) ↔
      Function.Injective (fun v => (obs v, tag v)) := by
  constructor
  · rintro ⟨decode, hzero⟩
    exact exact_recovery_implies_pair_injective obs tag decode hzero
  · exact pair_injective_implies_exact_recovery obs tag

/-- Exact recovery forces tags to separate each observation fiber. -/
theorem exact_recovery_implies_tag_injective_on_fiber
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hzero : ∀ v, decode (obs v) (tag v) = some v)
    (o : Fin O) :
    Function.Injective (fun x : observationFiber obs o => tag x.1) := by
  have hpairinj : Function.Injective (fun v => (obs v, tag v)) :=
    exact_recovery_implies_pair_injective obs tag decode hzero
  intro x y htag
  apply Subtype.ext
  apply hpairinj
  exact Prod.ext (x.2.trans y.2.symm) htag

/-- Each observation fiber requires at most as many states as available tags. -/
theorem fiber_card_le_tag_alphabet
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hzero : ∀ v, decode (obs v) (tag v) = some v)
    (o : Fin O) :
    Fintype.card { v : Fin K // obs v = o } ≤ T := by
  classical
  let f : { v : Fin K // obs v = o } → Fin T := fun x => tag x.1
  have hinj : Function.Injective f :=
    exact_recovery_implies_tag_injective_on_fiber obs tag decode hzero o
  simpa [f] using Fintype.card_le_of_injective f hinj

/-- Global counting converse for exact recovery from observation plus finite tags. -/
theorem exact_recovery_global_count
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hzero : ∀ v, decode (obs v) (tag v) = some v) :
    K ≤ O * T := by
  classical
  let pairMap : Fin K → Fin O × Fin T := fun v => (obs v, tag v)
  have hinj : Function.Injective pairMap :=
    exact_recovery_implies_pair_injective obs tag decode hzero
  have hcard : Fintype.card (Fin K) ≤ Fintype.card (Fin O × Fin T) :=
    Fintype.card_le_of_injective pairMap hinj
  simpa [pairMap] using hcard

/-- Two states are confusable if the observation channel cannot distinguish them. -/
def Confusable {K O : Nat} (obs : Fin K → Fin O) (v w : Fin K) : Prop :=
  obs v = obs w

/-- A finite confusability clique is a set whose members all share one observation. -/
def IsClique {K O : Nat} (obs : Fin K → Fin O) (s : Finset (Fin K)) : Prop :=
  ∀ ⦃v w : Fin K⦄, v ∈ s → w ∈ s → v ≠ w → Confusable obs v w

/-- Exact recovery forces tag injectivity on every confusability clique. -/
theorem exact_recovery_implies_tag_injOn_clique
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hzero : ∀ v, decode (obs v) (tag v) = some v)
    {s : Finset (Fin K)}
    (hs : IsClique obs s) :
    Set.InjOn tag { v | v ∈ s } := by
  have hpairinj : Function.Injective (fun v => (obs v, tag v)) :=
    exact_recovery_implies_pair_injective obs tag decode hzero
  intro v hv w hw htag
  by_cases hEq : v = w
  · exact hEq
  · have hobs : obs v = obs w := hs hv hw hEq
    exact hpairinj (Prod.ext hobs htag)

/-- A confusability clique needs at least as many tags as vertices. -/
theorem clique_card_le_tag_alphabet
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hzero : ∀ v, decode (obs v) (tag v) = some v)
    {s : Finset (Fin K)}
    (hs : IsClique obs s) :
    s.card ≤ T := by
  classical
  have hinjOn : Set.InjOn tag { v | v ∈ s } :=
    exact_recovery_implies_tag_injOn_clique obs tag decode hzero hs
  have hcardImg : s.card = (s.image tag).card := by
    symm
    exact Finset.card_image_of_injOn (by
      intro a ha b hb hab
      exact hinjOn ha hb hab)
  calc
    s.card = (s.image tag).card := hcardImg
    _ ≤ (Finset.univ : Finset (Fin T)).card := Finset.card_le_card (Finset.image_subset_iff.2 (by intro x hx; simp))
    _ = T := by simp

/-- Nontrivial confusability cliques force nontrivial auxiliary tag alphabets. -/
theorem nontrivial_clique_requires_nontrivial_tags
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (hzero : ∀ v, decode (obs v) (tag v) = some v)
    {s : Finset (Fin K)}
    (hs : IsClique obs s)
    (hsize : 1 < s.card) :
    1 < T := by
  have hbound : s.card ≤ T := clique_card_le_tag_alphabet obs tag decode hzero hs
  omega

/-- Observation alone cannot exactly resolve a nontrivial confusability clique. -/
theorem no_observation_only_exact_recovery_of_nontrivial_clique
    {K O : Nat}
    (obs : Fin K → Fin O)
    {s : Finset (Fin K)}
    (hs : IsClique obs s)
    (hsize : 1 < s.card) :
    ¬ ∃ decode : Fin O → Fin 1 → Option (Fin K),
        ∀ v, decode (obs v) 0 = some v := by
  intro h
  rcases h with ⟨decode, hzero⟩
  let tag : Fin K → Fin 1 := fun _ => 0
  have htag : 1 < 1 :=
    nontrivial_clique_requires_nontrivial_tags obs tag decode (by
      intro v
      simpa [tag] using hzero v) hs hsize
  omega

/-- If the state space exceeds the observation/tag budget, exact recovery is impossible. -/
theorem no_exact_recovery_of_large_state_space
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (hlarge : K > O * T) :
    ¬ ∃ decode : Fin O → Fin T → Option (Fin K),
        ∀ v, decode (obs v) (tag v) = some v := by
  intro h
  rcases h with ⟨decode, hzero⟩
  have hbound : K ≤ O * T := exact_recovery_global_count obs tag decode hzero
  omega

/-!
## Recovery architectures and formal support-count lower bounds

We package the observer/tag model into an explicit architecture object whose
auxiliary support count plays the role of a formal independent-support budget.
-/

structure RecoveryArchitecture (K O : Nat) where
  tagAlphabet : Nat
  obs : Fin K → Fin O
  tag : Fin K → Fin tagAlphabet
  decode : Fin O → Fin tagAlphabet → Option (Fin K)
  exact : ∀ v, decode (obs v) (tag v) = some v

/-- Auxiliary support count of a recovery architecture. -/
def supportCount {K O : Nat} (A : RecoveryArchitecture K O) : Nat :=
  A.tagAlphabet

/-- Exact-recovery architectures obey the global observation/support budget. -/
theorem architecture_support_global_bound
    {K O : Nat} (A : RecoveryArchitecture K O) :
    K ≤ O * supportCount A := by
  exact exact_recovery_global_count A.obs A.tag A.decode A.exact

/-- Any confusability clique lower-bounds the required auxiliary support count. -/
theorem architecture_support_bound_from_clique
    {K O : Nat} (A : RecoveryArchitecture K O)
    {s : Finset (Fin K)}
    (hs : IsClique A.obs s) :
    s.card ≤ supportCount A := by
  exact clique_card_le_tag_alphabet A.obs A.tag A.decode A.exact hs

/-- Nontrivial cliques force architectures to use more than one support value. -/
theorem architecture_support_gt_one_of_nontrivial_clique
    {K O : Nat} (A : RecoveryArchitecture K O)
    {s : Finset (Fin K)}
    (hs : IsClique A.obs s)
    (hsize : 1 < s.card) :
    1 < supportCount A := by
  exact nontrivial_clique_requires_nontrivial_tags A.obs A.tag A.decode A.exact hs hsize

/-- Observation-only architectures have support count 1. -/
def ObservationOnlyArchitecture (K O : Nat) := RecoveryArchitecture K O

/-- Observation-only exact recovery is impossible on nontrivial cliques. -/
theorem observation_only_architecture_impossible_on_nontrivial_clique
    {K O : Nat}
    (obs : Fin K → Fin O)
    {s : Finset (Fin K)}
    (hs : IsClique obs s)
    (hsize : 1 < s.card) :
    ¬ ∃ A : ObservationOnlyArchitecture K O,
        A.obs = obs ∧ supportCount A = 1 := by
  intro h
  rcases h with ⟨A, hobs, hsupport⟩
  subst hobs
  have hgt : 1 < supportCount A :=
    architecture_support_gt_one_of_nontrivial_clique A hs hsize
  rw [hsupport] at hgt
  omega

/-- Exact recovery on a subset of states. -/
def ExactOn {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    (S : Finset (Fin K)) : Prop :=
  ∀ v, v ∈ S → decode (obs v) (tag v) = some v

/-- Any clique that is recovered exactly on a subset lower-bounds the support budget there. -/
theorem exactOn_clique_card_le_tag_alphabet
    {K O T : Nat}
    (obs : Fin K → Fin O)
    (tag : Fin K → Fin T)
    (decode : Fin O → Fin T → Option (Fin K))
    {S : Finset (Fin K)}
    (hexact : ExactOn obs tag decode S)
    (hs : IsClique obs S) :
    S.card ≤ T := by
  classical
  have hinjOn : Set.InjOn tag { v | v ∈ S } := by
    intro v hv w hw htag
    by_cases hEq : v = w
    · exact hEq
    · have hv' := hexact v hv
      have hw' := hexact w hw
      have hobs : obs v = obs w := hs hv hw hEq
      have hdecode : some v = some w := by
        calc
          some v = decode (obs v) (tag v) := by simpa using hv'.symm
          _ = decode (obs w) (tag w) := by simp [hobs, htag]
          _ = some w := hw'
      simpa using hdecode
  have hcardImg : S.card = (S.image tag).card := by
    symm
    exact Finset.card_image_of_injOn (by
      intro a ha b hb hab
      exact hinjOn ha hb hab)
  calc
    S.card = (S.image tag).card := hcardImg
    _ ≤ (Finset.univ : Finset (Fin T)).card := Finset.card_le_card (Finset.image_subset_iff.2 (by intro x hx; simp))
    _ = T := by simp


end ObserverModel
