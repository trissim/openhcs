import DecisionQuotient.Reduction_AllCoords
import DecisionQuotient.Hardness.Sigma2PHardness
import DecisionQuotient.AlgorithmComplexity
import Mathlib.Data.Finset.Card
import Mathlib.Tactic

namespace DecisionQuotient

open Classical
open ReductionAction ReductionState

abbrev ShiftedCoord (n : ℕ) (i : Fin (n + 1)) : Type :=
  Fin.cases Bool (fun _ => ReductionState n) i

abbrev ShiftedState (n : ℕ) := (i : Fin (n + 1)) → ShiftedCoord n i

instance shiftedCoordinateSpace (n : ℕ) : CoordinateSpace (ShiftedState n) (n + 1) where
  Coord := ShiftedCoord n
  proj := fun s i => s i

instance shiftedProductSpace (n : ℕ) : ProductSpace (ShiftedState n) (n + 1) where
  Coord := ShiftedCoord n
  proj := fun s i => s i
  replace := fun s i s' => Function.update s i (s' i)
  replace_proj_eq := by
    intro s s' i
    simp [Function.update]
  replace_proj_ne := by
    intro s s' i j hne
    simp [Function.update, hne]

def shiftedReferenceState (n : ℕ) : ShiftedState n :=
  fun i => Fin.cases true (fun _ => ReductionState.reference) i

def shiftedClosedState (n : ℕ) : ShiftedState n :=
  Function.update (shiftedReferenceState n) 0 false

def shiftedFalsifyingState (n : ℕ) (i : Fin n) (a : BoolAssignment n) : ShiftedState n :=
  Function.update (shiftedReferenceState n) i.succ (ReductionState.assignment a)

noncomputable def reductionUtilityShifted (φ : Formula n) :
    ReductionAction → ShiftedState n → ℝ
  | accept, s =>
      if s 0 = true ∧ ∀ j : Fin n, reductionUtility φ accept (s j.succ) = 1 then 1 else 0
  | reject, _ => 0

noncomputable def reductionProblemShifted (φ : Formula n) :
    DecisionProblem ReductionAction (ShiftedState n) where
  utility := reductionUtilityShifted φ

theorem opt_shifted_open_reference (φ : Formula n) :
    (reductionProblemShifted φ).Opt (shiftedReferenceState n) = {accept} := by
  have hcond :
      (shiftedReferenceState n) 0 = true ∧
        ∀ j : Fin n, reductionUtility φ accept ((shiftedReferenceState n) j.succ) = 1 := by
    constructor
    · simp [shiftedReferenceState]
    · intro j
      simp [shiftedReferenceState, reductionUtility]
  ext a
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq, Set.mem_singleton_iff]
  constructor
  · intro h
    cases a with
    | accept => rfl
    | reject =>
      have h1 := h accept
      simp [reductionProblemShifted, reductionUtilityShifted, hcond] at h1
      linarith
  · intro h
    subst h
    intro a'
    cases a' with
    | accept =>
        simp [reductionProblemShifted, reductionUtilityShifted, hcond]
    | reject =>
        simp [reductionProblemShifted, reductionUtilityShifted, hcond]

theorem opt_shifted_closed_reference (φ : Formula n) :
    (reductionProblemShifted φ).Opt (shiftedClosedState n) = {accept, reject} := by
  have hnot :
      ¬ ((shiftedClosedState n) 0 = true ∧
          ∀ j : Fin n, reductionUtility φ accept ((shiftedClosedState n) j.succ) = 1) := by
    intro h
    simpa [shiftedClosedState, Function.update] using h.1
  ext a
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq,
    Set.mem_insert_iff, Set.mem_singleton_iff]
  constructor
  · intro _
    cases a <;> simp
  · intro _
    intro a'
    cases a <;> cases a' <;> simp [reductionProblemShifted, reductionUtilityShifted, hnot]

theorem opt_shifted_open_falsifying (φ : Formula n) (i : Fin n) (a : BoolAssignment n)
    (hfalse : Formula.eval a φ = false) :
    (reductionProblemShifted φ).Opt (shiftedFalsifyingState n i a) = {accept, reject} := by
  have hnot :
      ¬ ((shiftedFalsifyingState n i a) 0 = true ∧
          ∀ j : Fin n, reductionUtility φ accept ((shiftedFalsifyingState n i a) j.succ) = 1) := by
    intro h
    have hi := h.2 i
    simpa [shiftedFalsifyingState, reductionUtility, hfalse, Function.update] using hi
  ext x
  simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq,
    Set.mem_insert_iff, Set.mem_singleton_iff]
  constructor
  · intro _
    cases x <;> simp
  · intro _
    intro a'
    cases x <;> cases a' <;> simp [reductionProblemShifted, reductionUtilityShifted, hnot]

theorem opt_shifted_of_tautology (φ : Formula n) (htaut : φ.isTautology) (s : ShiftedState n) :
    (reductionProblemShifted φ).Opt s = if s 0 = true then ({accept} : Set ReductionAction)
      else {accept, reject} := by
  by_cases hgate : s 0 = true
  · have hrest : ∀ j : Fin n, reductionUtility φ accept (s j.succ) = 1 := by
      intro j
      cases hs : s j.succ with
      | reference =>
          simp [reductionUtility]
      | assignment a =>
          have hsat : Formula.eval a φ = true := htaut a
          simp [reductionUtility, hs, hsat]
    have hcond :
        s 0 = true ∧ ∀ j : Fin n, reductionUtility φ accept (s j.succ) = 1 := ⟨hgate, hrest⟩
    have hopen : (reductionProblemShifted φ).Opt s = ({accept} : Set ReductionAction) := by
      ext a
      simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq, Set.mem_singleton_iff]
      constructor
      · intro hopt
        cases a with
        | accept => rfl
        | reject =>
            have h1 := hopt accept
            simp [reductionProblemShifted, reductionUtilityShifted, hcond] at h1
            linarith
      · intro ha
        subst ha
        intro a'
        cases a' <;> simp [reductionProblemShifted, reductionUtilityShifted, hcond]
    simp [hgate, hopen]
  · have hnot :
        ¬ (s 0 = true ∧ ∀ j : Fin n, reductionUtility φ accept (s j.succ) = 1) := by
      intro h
      exact hgate h.1
    have hclosed : (reductionProblemShifted φ).Opt s = ({accept, reject} : Set ReductionAction) := by
      ext a
      simp only [DecisionProblem.Opt, DecisionProblem.isOptimal, Set.mem_setOf_eq,
        Set.mem_insert_iff, Set.mem_singleton_iff]
      constructor
      · intro _
        cases a <;> simp
      · intro _
        intro a'
        cases a <;> cases a' <;> simp [reductionProblemShifted, reductionUtilityShifted, hnot]
    simp [hgate, hclosed]

theorem gate_coordinate_relevant_shifted (φ : Formula n) :
    (reductionProblemShifted φ).isRelevant 0 := by
  refine ⟨shiftedReferenceState n, shiftedClosedState n, ?_, ?_⟩
  · intro j hj
    change (shiftedReferenceState n) j = (shiftedClosedState n) j
    simp [shiftedClosedState, shiftedReferenceState, Function.update, hj]
  · intro heq
    have hmemClosed : reject ∈ ((reductionProblemShifted φ).Opt (shiftedClosedState n)) := by
      simp [opt_shifted_closed_reference]
    have hmemOpen : reject ∈ ((reductionProblemShifted φ).Opt (shiftedReferenceState n)) := by
      exact heq ▸ hmemClosed
    simpa [opt_shifted_open_reference] using hmemOpen

theorem formula_coordinate_irrelevant_of_tautology (φ : Formula n)
    (htaut : φ.isTautology) (i : Fin n) :
    ¬ (reductionProblemShifted φ).isRelevant i.succ := by
  intro hrel
  rcases hrel with ⟨s, s', hagree, hneq⟩
  have hgateIndexNe : (0 : Fin (n + 1)) ≠ i.succ := by
    intro hEq
    exact Fin.succ_ne_zero i hEq.symm
  have hgate : s 0 = s' 0 := hagree 0 hgateIndexNe
  rw [opt_shifted_of_tautology (φ := φ) htaut s,
      opt_shifted_of_tautology (φ := φ) htaut s', hgate] at hneq
  exact hneq rfl

theorem formula_coordinates_relevant_of_not_tautology (φ : Formula n) (h : ¬ φ.isTautology) :
    ∀ i : Fin n, (reductionProblemShifted φ).isRelevant i.succ := by
  have hnotall : ¬ ∀ a, Formula.eval a φ = true := by
    simpa [Formula.isTautology] using h
  rcases not_forall.mp hnotall with ⟨a0, ha0⟩
  intro i
  cases hval : Formula.eval a0 φ with
  | true =>
      exact False.elim (ha0 hval)
  | false =>
      refine ⟨shiftedReferenceState n, shiftedFalsifyingState n i a0, ?_, ?_⟩
      · intro j hj
        change (shiftedReferenceState n) j = (shiftedFalsifyingState n i a0) j
        simp [shiftedReferenceState, shiftedFalsifyingState, Function.update, hj]
      · intro heq
        have hmemFalse : reject ∈ ((reductionProblemShifted φ).Opt (shiftedFalsifyingState n i a0)) := by
          simp [opt_shifted_open_falsifying (φ := φ) (i := i) (a := a0) (hfalse := hval)]
        have hmemOpen : reject ∈ ((reductionProblemShifted φ).Opt (shiftedReferenceState n)) := by
          exact heq ▸ hmemFalse
        simpa [opt_shifted_open_reference] using hmemOpen

theorem singleton_gate_sufficient_of_tautology (φ : Formula n)
    (htaut : φ.isTautology) :
    (reductionProblemShifted φ).isSufficient ({0} : Finset (Fin (n + 1))) := by
  intro s s' hagree
  have hgate : s 0 = s' 0 := hagree 0 (by simp)
  rw [opt_shifted_of_tautology (φ := φ) htaut s,
      opt_shifted_of_tautology (φ := φ) htaut s', hgate]

theorem all_coordinates_necessary_of_not_tautology (φ : Formula n)
    (h : ¬ φ.isTautology) (I : Finset (Fin (n + 1)))
    (hI : (reductionProblemShifted φ).isSufficient I) :
    ∀ i : Fin (n + 1), i ∈ I := by
  intro i
  rcases Fin.eq_zero_or_eq_succ i with rfl | ⟨j, rfl⟩
  · exact (reductionProblemShifted φ).sufficient_contains_relevant I hI 0
      (gate_coordinate_relevant_shifted (φ := φ))
  · exact (reductionProblemShifted φ).sufficient_contains_relevant I hI j.succ
      (formula_coordinates_relevant_of_not_tautology (φ := φ) h j)

def FactorApproxMinSolver :=
  ∀ {n : ℕ}, Formula n → Finset (Fin (n + 1))

def IsFactorApproxMinSolver (ρ : ℕ) (solver : FactorApproxMinSolver) : Prop :=
  ∀ {n : ℕ} (φ : Formula n),
    let out := solver φ
    (reductionProblemShifted φ).isSufficient out ∧
      ∀ J : Finset (Fin (n + 1)),
        (reductionProblemShifted φ).isSufficient J → out.card ≤ ρ * J.card

theorem tautology_decidable_from_factor_approx
    (ρ : ℕ) (solver : FactorApproxMinSolver)
    (hsolver : IsFactorApproxMinSolver ρ solver)
    {n : ℕ} (φ : Formula n) (hgap : ρ < n + 1) :
    φ.isTautology ↔ (solver φ).card ≤ ρ := by
  constructor
  · intro htaut
    rcases hsolver φ with ⟨_hsuff, happrox⟩
    have hsuffGate := singleton_gate_sufficient_of_tautology (φ := φ) htaut
    have hbound := happrox ({0} : Finset (Fin (n + 1))) hsuffGate
    simpa using hbound
  · intro hout
    by_contra hnot
    rcases hsolver φ with ⟨hsuff, _happrox⟩
    have hall : ∀ i : Fin (n + 1), i ∈ solver φ :=
      all_coordinates_necessary_of_not_tautology (φ := φ) hnot (solver φ) hsuff
    have hcard :
        (Finset.univ : Finset (Fin (n + 1))).card ≤ (solver φ).card := by
      exact Finset.card_le_card (by intro i _; exact hall i)
    have hnotSmall : ¬ (solver φ).card ≤ ρ := by
      intro hsmall
      exact not_le_of_gt hgap (le_trans (by simpa using hcard) hsmall)
    exact hnotSmall hout

/-! ## Counted Runtime Upgrade

This section adds an explicit counted-time interface for factor-approximation
solvers on the shifted reduction family, together with a derived counted
tautology test obtained by thresholding the returned support size. -/

structure CountedFactorApproxMinSolver (ρ : ℕ) where
  run : {n : ℕ} → Formula n → Counted (Finset (Fin (n + 1)))
  sufficient_ok :
    ∀ {n : ℕ} (φ : Formula n),
      (reductionProblemShifted φ).isSufficient (run φ).result
  approx_ok :
    ∀ {n : ℕ} (φ : Formula n) (J : Finset (Fin (n + 1))),
      (reductionProblemShifted φ).isSufficient J → (run φ).result.card ≤ ρ * J.card
  polytime :
    ∃ c k : ℕ, ∀ {n : ℕ} (φ : Formula n),
      (run φ).steps ≤ c * (sizeOf φ + 1) ^ k + c

def countedResultSolver (ρ : ℕ) (solver : CountedFactorApproxMinSolver ρ) :
    FactorApproxMinSolver :=
  fun {n} φ => (solver.run φ).result

theorem countedResultSolver_spec
    (ρ : ℕ) (solver : CountedFactorApproxMinSolver ρ) :
    IsFactorApproxMinSolver ρ (countedResultSolver ρ solver) := by
  intro n φ
  exact ⟨solver.sufficient_ok φ, solver.approx_ok φ⟩

def countedGapTautologyRun
    (ρ : ℕ) (solver : CountedFactorApproxMinSolver ρ)
    {n : ℕ} (φ : Formula n) : Counted Bool :=
  Counted.bind (solver.run φ) fun out =>
    Counted.tick (decide (out.card ≤ ρ))

theorem countedGapTautologyRun_correct
    (ρ : ℕ) (solver : CountedFactorApproxMinSolver ρ)
    {n : ℕ} (φ : Formula n) (hgap : ρ < n + 1) :
    (countedGapTautologyRun ρ solver φ).result = true ↔ φ.isTautology := by
  have hspec : IsFactorApproxMinSolver ρ (countedResultSolver ρ solver) := by
    intro n φ
    exact ⟨solver.sufficient_ok φ, solver.approx_ok φ⟩
  cases hrun : solver.run φ with
  | mk steps out =>
    constructor
    · intro htrue
      have hout : out.card ≤ ρ := by
        have hdec : decide (out.card ≤ ρ) = true := by
          simpa [Counted.result, countedGapTautologyRun, Counted.bind, Counted.tick, hrun] using htrue
        simpa using hdec
      have hcard : (countedResultSolver ρ solver φ).card ≤ ρ := by
        simpa [countedResultSolver, hrun] using hout
      exact (tautology_decidable_from_factor_approx ρ (countedResultSolver ρ solver) hspec φ hgap).2 hcard
    · intro htaut
      have hcard :
          (countedResultSolver ρ solver φ).card ≤ ρ :=
        (tautology_decidable_from_factor_approx ρ (countedResultSolver ρ solver) hspec φ hgap).1 htaut
      have hout : out.card ≤ ρ := by
        simpa [countedResultSolver, hrun] using hcard
      have hdec : decide (out.card ≤ ρ) = true := by
        simpa using hout
      simpa [Counted.result, countedGapTautologyRun, Counted.bind, Counted.tick, hrun] using hdec

theorem countedGapTautologyRun_polytime
    (ρ : ℕ) (solver : CountedFactorApproxMinSolver ρ) :
    ∃ c k : ℕ, ∀ {n : ℕ} (φ : Formula n),
      (countedGapTautologyRun ρ solver φ).steps ≤ c * (sizeOf φ + 1) ^ k + c := by
  rcases solver.polytime with ⟨c, k, hpoly⟩
  refine ⟨c + 1, k, ?_⟩
  intro n φ
  have hbase := hpoly φ
  have hpow_pos : 1 ≤ (sizeOf φ + 1) ^ k := by
    exact Nat.one_le_pow k (sizeOf φ + 1) (by omega)
  have hstep_le :
      (countedGapTautologyRun ρ solver φ).steps ≤ (solver.run φ).steps + 1 := by
    unfold countedGapTautologyRun
    cases hrun : solver.run φ with
    | mk steps out =>
        simp [Counted.bind, Counted.tick, Counted.steps]
  calc
    (countedGapTautologyRun ρ solver φ).steps ≤ (solver.run φ).steps + 1 := hstep_le
    _ ≤ (c * (sizeOf φ + 1) ^ k + c) + 1 := by omega
    _ ≤ c * (sizeOf φ + 1) ^ k + c + (sizeOf φ + 1) ^ k := by omega
    _ = (c + 1) * (sizeOf φ + 1) ^ k + c := by ring
    _ ≤ (c + 1) * (sizeOf φ + 1) ^ k + (c + 1) := by omega

end DecisionQuotient
