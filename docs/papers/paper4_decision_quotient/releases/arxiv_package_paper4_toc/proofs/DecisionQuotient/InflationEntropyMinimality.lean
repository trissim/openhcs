import DecisionQuotient.InflationEntropyBridge

open Classical

namespace DecisionQuotient
namespace InflationEntropyMinimality

open InflationEntropyBridge

def dpSwitch : DecisionProblem Bool Bool where
  utility := fun a s =>
    if s then (if a then 1 else 0)
    else (if a then 0 else 1)

def dpSingle : DecisionProblem Bool Unit where
  utility := fun a _ => if a then 1 else 0

theorem dpSwitch_opt_true : dpSwitch.Opt true = ({true} : Set Bool) := by
  ext a
  cases a <;> simp [dpSwitch, DecisionProblem.Opt, DecisionProblem.isOptimal]

theorem dpSwitch_opt_false : dpSwitch.Opt false = ({false} : Set Bool) := by
  ext a
  cases a <;> simp [dpSwitch, DecisionProblem.Opt, DecisionProblem.isOptimal]

theorem dpSingle_opt : dpSingle.Opt () = ({true} : Set Bool) := by
  ext a
  cases a <;> simp [dpSingle, DecisionProblem.Opt, DecisionProblem.isOptimal]

theorem dpSwitch_numOptClasses : dpSwitch.numOptClasses = 2 := by
  classical
  unfold DecisionProblem.numOptClasses
  have huniv : (Finset.univ : Finset Bool) = {false, true} := by decide
  rw [huniv]
  simp [dpSwitch_opt_true, dpSwitch_opt_false]

theorem dpSingle_numOptClasses : dpSingle.numOptClasses = 1 := by
  classical
  unfold DecisionProblem.numOptClasses
  simp [dpSingle_opt]

theorem not_redundant_A2_for_mono_classes :
    ∃ (dp₁ : DecisionProblem Bool Bool)
      (dp₂ : DecisionProblem Bool Unit)
      (emb : Bool → Unit),
      dp₁.numOptClasses > dp₂.numOptClasses ∧
      ¬ (∀ s : Bool, dp₂.Opt (emb s) = dp₁.Opt s) := by
  refine ⟨dpSwitch, dpSingle, (fun _ => ()), ?_⟩
  constructor
  · rw [dpSwitch_numOptClasses, dpSingle_numOptClasses]
    decide
  · intro hCompat
    have htrue := hCompat true
    have hfalse := hCompat false
    rw [dpSingle_opt, dpSwitch_opt_true] at htrue
    rw [dpSingle_opt, dpSwitch_opt_false] at hfalse
    have hmem : (true : Bool) ∈ ({false} : Set Bool) := by
      have hInTrue : (true : Bool) ∈ ({true} : Set Bool) := by simp
      simpa [hfalse] using hInTrue
    simp at hmem

def constFamily : DynamicDecisionFamily Bool where
  State := fun _ => Unit
  instFintype := fun _ => inferInstance
  instNonempty := fun _ => inferInstance
  dp := fun _ => dpSingle
  emb := fun {_ _} _ s => s
  optCompat := by intro t₁ t₂ h s; rfl

theorem constFamily_numOptClasses (t : ℕ) :
    numOptClassesAt constFamily t = 1 := by
  simpa [numOptClassesAt, constFamily] using dpSingle_numOptClasses

theorem not_redundant_A3_for_strict_entropy :
    ¬ quotientEntropyAt constFamily 0 < quotientEntropyAt constFamily 1 := by
  have h0 : quotientEntropyAt constFamily 0 = 0 := by
    simp [quotientEntropyAt, constFamily, dpSingle_numOptClasses, DecisionProblem.quotientEntropy]
  have h1 : quotientEntropyAt constFamily 1 = 0 := by
    simp [quotientEntropyAt, constFamily, dpSingle_numOptClasses, DecisionProblem.quotientEntropy]
  rw [h0, h1]
  exact not_lt_of_ge le_rfl

def switchFamily : DynamicDecisionFamily Bool where
  State := fun _ => Bool
  instFintype := fun _ => inferInstance
  instNonempty := fun _ => inferInstance
  dp := fun _ => dpSwitch
  emb := fun {_ _} _ s => s
  optCompat := by intro t₁ t₂ h s; rfl

theorem switchFamily_numOptClasses (t : ℕ) :
    numOptClassesAt switchFamily t = 2 := by
  simpa [numOptClassesAt, switchFamily] using dpSwitch_numOptClasses

theorem switchFamily_positive_floor_requires_T_pos :
    ¬ (0 < (1 : ℝ) * 0 * Real.log (numOptClassesAt switchFamily 0 : ℝ)) := by
  rw [switchFamily_numOptClasses 0]
  norm_num

theorem switchFamily_positive_floor_requires_kB_pos :
    ¬ (0 < (0 : ℝ) * 1 * Real.log (numOptClassesAt switchFamily 0 : ℝ)) := by
  rw [switchFamily_numOptClasses 0]
  norm_num

theorem not_redundant_P2_for_positive_floor :
    ∃ (kB T : ℝ),
      0 < kB ∧ T = 0 ∧ 1 < numOptClassesAt switchFamily 0 ∧
      ¬ (0 < kB * T * Real.log (numOptClassesAt switchFamily 0 : ℝ)) := by
  refine ⟨1, 0, by norm_num, rfl, ?_, ?_⟩
  · rw [switchFamily_numOptClasses 0]
    norm_num
  · simpa using switchFamily_positive_floor_requires_T_pos

theorem not_redundant_P1_for_positive_floor :
    ∃ (kB T : ℝ),
      kB = 0 ∧ 0 < T ∧ 1 < numOptClassesAt switchFamily 0 ∧
      ¬ (0 < kB * T * Real.log (numOptClassesAt switchFamily 0 : ℝ)) := by
  refine ⟨0, 1, rfl, by norm_num, ?_, ?_⟩
  · rw [switchFamily_numOptClasses 0]
    norm_num
  · simpa using switchFamily_positive_floor_requires_kB_pos

/-! ## Additional irreducibility witnesses for assumption matrix completeness -/

structure WeakDynamicFamily (A : Type*) where
  State : ℕ → Type*
  instFintype : ∀ t : ℕ, Fintype (State t)
  instNonempty : ∀ t : ℕ, Nonempty (State t)
  dp : ∀ t : ℕ, DecisionProblem A (State t)

attribute [instance] WeakDynamicFamily.instFintype
attribute [instance] WeakDynamicFamily.instNonempty

def weakNumOptClassesAt [DecidableEq (Set A)]
    (M : WeakDynamicFamily A) (t : ℕ) : ℕ :=
  (M.dp t).numOptClasses

def weakState : ℕ → Type
  | 0 => Bool
  | Nat.succ _ => Unit

instance instFintypeWeakState (t : ℕ) : Fintype (weakState t) := by
  cases t with
  | zero =>
      simpa [weakState] using (inferInstance : Fintype Bool)
  | succ _ =>
      simpa [weakState] using (inferInstance : Fintype Unit)

instance instNonemptyWeakState (t : ℕ) : Nonempty (weakState t) := by
  cases t with
  | zero =>
      simpa [weakState] using (inferInstance : Nonempty Bool)
  | succ _ =>
      simpa [weakState] using (inferInstance : Nonempty Unit)

def weakDp : (t : ℕ) → DecisionProblem Bool (weakState t)
  | 0 => by simpa [weakState] using dpSwitch
  | Nat.succ _ => by simpa [weakState] using dpSingle

def weakCounterFamily : WeakDynamicFamily Bool where
  State := weakState
  instFintype := instFintypeWeakState
  instNonempty := instNonemptyWeakState
  dp := weakDp

theorem weakCounterFamily_numOptClasses_zero :
    weakNumOptClassesAt weakCounterFamily 0 = 2 := by
  change (weakDp 0).numOptClasses = 2
  simpa [weakDp] using dpSwitch_numOptClasses

theorem weakCounterFamily_numOptClasses_one :
    weakNumOptClassesAt weakCounterFamily 1 = 1 := by
  change (weakDp 1).numOptClasses = 1
  simpa [weakDp] using dpSingle_numOptClasses

theorem not_redundant_A1_for_mono_classes_weak :
    weakNumOptClassesAt weakCounterFamily 1 < weakNumOptClassesAt weakCounterFamily 0 := by
  rw [weakCounterFamily_numOptClasses_one, weakCounterFamily_numOptClasses_zero]
  decide

structure WeakNoNonemptyFamily (A : Type*) where
  State : ℕ → Type*
  instFintype : ∀ t : ℕ, Fintype (State t)
  dp : ∀ t : ℕ, DecisionProblem A (State t)

attribute [instance] WeakNoNonemptyFamily.instFintype

def weakNoNonemptyNumOptClassesAt [DecidableEq (Set A)]
    (M : WeakNoNonemptyFamily A) (t : ℕ) : ℕ :=
  (M.dp t).numOptClasses

def emptyStateFamily : WeakNoNonemptyFamily Unit where
  State := fun _ => PEmpty
  instFintype := fun _ => inferInstance
  dp := fun _ =>
    { utility := fun _ s => nomatch s }

theorem not_redundant_F2_for_numOptClasses_pos :
    weakNoNonemptyNumOptClassesAt emptyStateFamily 0 = 0 := by
  simp [weakNoNonemptyNumOptClassesAt, emptyStateFamily, DecisionProblem.numOptClasses]

theorem not_redundant_F1_for_finite_counting_requirement :
    ¬ Nonempty (Fintype ℕ) := by
  intro h
  letI : Fintype ℕ := Classical.choice h
  exact (Finite.false (α := ℕ) (Finite.of_fintype ℕ))

theorem not_redundant_P3_for_energy_from_entropy_bridge :
    ∃ E : ℝ,
      1 < numOptClassesAt switchFamily 0 ∧
      ¬ (E ≥ (1 : ℝ) * 1 * Real.log (numOptClassesAt switchFamily 0 : ℝ)) := by
  refine ⟨0, ?_, ?_⟩
  · rw [switchFamily_numOptClasses 0]
    norm_num
  · have hlog2 : 0 < Real.log (numOptClassesAt switchFamily 0 : ℝ) := by
      rw [switchFamily_numOptClasses 0]
      exact Real.log_pos (by norm_num)
    have hpos : 0 < (1 : ℝ) * 1 * Real.log (numOptClassesAt switchFamily 0 : ℝ) := by
      nlinarith
    exact not_le_of_gt hpos

end InflationEntropyMinimality
end DecisionQuotient
