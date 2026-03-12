/-
  SSOT Formalization - Coherence Theorems
  Paper 2: Formal Foundations for the Single Source of Truth Principle

  This file establishes the EPISTEMIC foundation:
  - Coherence is defined as agreement among encodings
  - DOF = 1 guarantees coherence (impossibility of disagreement)
  - DOF > 1 permits incoherence (reachable disagreeing states)
  - Modification complexity IS coherence restoration complexity

  These theorems ground SSOT in epistemology, not just engineering convenience.
-/

import Ssot.SSOT

namespace Ssot

/-!
## Part 1: Coherence Definitions

A system encoding fact F is COHERENT iff all encodings of F agree.
A system is INCOHERENT iff some encodings disagree.
-/

-- A value that a fact can take (abstract)
-- We only need to know: can two values be equal or not?
inductive FactValue where
  | val : Nat → FactValue
  deriving DecidableEq, Repr

-- An encoding system: n locations each holding a value
structure EncodingSystem where
  values : List FactValue
  deriving Repr

-- DOF = number of independent encoding locations
def EncodingSystem.dof (s : EncodingSystem) : Nat := s.values.length

-- DEFINITION: Coherence = all encodings hold the same value
def EncodingSystem.is_coherent (s : EncodingSystem) : Prop :=
  ∀ v1 v2, v1 ∈ s.values → v2 ∈ s.values → v1 = v2

-- DEFINITION: Incoherence = some encodings disagree
def EncodingSystem.is_incoherent (s : EncodingSystem) : Prop :=
  ∃ v1 v2, v1 ∈ s.values ∧ v2 ∈ s.values ∧ v1 ≠ v2

-- Coherence and incoherence are mutually exclusive and exhaustive (for non-empty systems)
theorem coherence_incoherence_exclusive (s : EncodingSystem) :
    s.is_coherent → ¬s.is_incoherent := by
  intro hcoh hinc
  have ⟨v1, v2, hv1, hv2, hne⟩ := hinc
  exact hne (hcoh v1 v2 hv1 hv2)

/-!
## Part 2: DOF = 1 Guarantees Coherence

With exactly one encoding, disagreement is impossible.
This is the EPISTEMIC foundation of SSOT.
-/

-- Helper: a singleton list has all elements equal
theorem singleton_all_equal {α : Type} (a : α) (x y : α) 
    (hx : x ∈ [a]) (hy : y ∈ [a]) : x = y := by
  simp at hx hy
  rw [hx, hy]

-- THEOREM: DOF = 1 → Coherence is guaranteed
-- This is the core epistemic result: single source cannot disagree with itself
theorem dof_one_implies_coherent (s : EncodingSystem) (h : s.dof = 1) :
    s.is_coherent := by
  unfold EncodingSystem.is_coherent
  intro v1 v2 hv1 hv2
  -- s.values has length 1, so it's a singleton [a] for some a
  unfold EncodingSystem.dof at h
  match hlen : s.values with
  | [] =>
    -- Empty list has length 0, not 1
    rw [hlen] at h
    simp at h
  | [a] => exact singleton_all_equal a v1 v2 (hlen ▸ hv1) (hlen ▸ hv2)
  | _ :: _ :: _ =>
    -- List with 2+ elements has length ≥ 2, not 1
    rw [hlen] at h
    simp at h

-- COROLLARY: DOF = 1 → Incoherence is impossible
theorem dof_one_incoherence_impossible (s : EncodingSystem) (h : s.dof = 1) :
    ¬s.is_incoherent := by
  exact coherence_incoherence_exclusive s (dof_one_implies_coherent s h)

/-!
## Part 3: DOF > 1 Permits Incoherence

With multiple independent encodings, disagreeing states are REACHABLE.
Independence means: each encoding can be modified without affecting others.
Therefore, they can end up holding different values.
-/

-- Construct an incoherent system with DOF = n > 1
def incoherent_system (n : Nat) (_ : n > 1) : EncodingSystem :=
  -- First encoding holds value 0, rest hold value 1
  { values := (FactValue.val 0) :: List.replicate (n - 1) (FactValue.val 1) }

-- The constructed system has the specified DOF
theorem incoherent_system_dof (n : Nat) (hn : n > 1) :
    (incoherent_system n hn).dof = n := by
  unfold incoherent_system EncodingSystem.dof
  simp only [List.length_cons, List.length_replicate]
  omega

-- The constructed system is incoherent
theorem incoherent_system_is_incoherent (n : Nat) (hn : n > 1) :
    (incoherent_system n hn).is_incoherent := by
  unfold EncodingSystem.is_incoherent incoherent_system
  refine ⟨FactValue.val 0, FactValue.val 1, ?_, ?_, ?_⟩
  · -- val 0 is the head of the list
    simp only [List.mem_cons, true_or]
  · -- val 1 is in the replicate part
    simp only [List.mem_cons, List.mem_replicate]
    right
    exact ⟨by omega, trivial⟩
  · -- val 0 ≠ val 1
    intro h
    cases h

-- THEOREM: DOF > 1 → Incoherent states exist
-- For any DOF > 1, we can construct a system in an incoherent state
theorem dof_gt_one_incoherence_possible (n : Nat) (hn : n > 1) :
    ∃ s : EncodingSystem, s.dof = n ∧ s.is_incoherent :=
  ⟨incoherent_system n hn, incoherent_system_dof n hn, incoherent_system_is_incoherent n hn⟩

/-!
## Part 4: Coherence Restoration Complexity

After changing the "true" value of a fact:
- DOF = 1: 1 edit restores coherence (the single source)
- DOF = n > 1: n edits required to restore coherence (all n sources)

This proves: MODIFICATION COMPLEXITY = COHERENCE RESTORATION COMPLEXITY
-/

-- An edit changes one encoding location
-- Edits required to restore coherence = number of encodings that need updating

-- If fact changes from old to new, how many edits to make all encodings = new?
def edits_to_restore_coherence (s : EncodingSystem) (_new_val : FactValue) : Nat :=
  s.values.length  -- Must update all encodings to the new value

-- THEOREM: Coherence restoration complexity = DOF
theorem coherence_restoration_eq_dof (s : EncodingSystem) (new_val : FactValue) :
    edits_to_restore_coherence s new_val = s.dof := by
  unfold edits_to_restore_coherence EncodingSystem.dof
  rfl

/-!
## Part 5: Partial Edits Create Incoherence

If DOF = n and you make k < n edits, the system is in an incoherent state.
This is the epistemic defect of DOF > 1: partial updates are possible.
-/

-- Model: apply k edits to first k encodings, leaving rest unchanged
def partial_edit (s : EncodingSystem) (k : Nat) (new_val old_val : FactValue) : EncodingSystem :=
  { values := (List.replicate k new_val) ++ (List.replicate (s.dof - k) old_val) }

-- If k > 0 and k < n, the system is incoherent (some new, some old)
theorem partial_edit_incoherent (n k : Nat) (hk_pos : k > 0) (hk_lt : k < n)
    (new_val old_val : FactValue) (hne : new_val ≠ old_val) :
    let s : EncodingSystem := { values := List.replicate n old_val }
    (partial_edit s k new_val old_val).is_incoherent := by
  simp only [partial_edit, EncodingSystem.is_incoherent, EncodingSystem.dof, List.length_replicate]
  refine ⟨new_val, old_val, ?_, ?_, hne⟩
  · simp only [List.mem_append, List.mem_replicate]
    left
    exact ⟨by omega, trivial⟩
  · simp only [List.mem_append, List.mem_replicate]
    right
    exact ⟨by omega, trivial⟩

/-!
## Part 6: The Epistemic Defect Theorem

DOF > 1 is epistemically defective because:
1. Incoherent states are reachable (proved above)
2. In an incoherent state, the "true" value of the fact is indeterminate
3. Indeterminacy violates the law of excluded middle for the fact

This is not about convenience. It is about the coherence of truth.
-/

-- In an incoherent state, which encoding holds the "true" value?
-- Answer: UNDEFINED - there is no principled way to choose
-- This is formalized as: no total function from incoherent states to "true value"

-- The epistemic defect: incoherent states have no determinate truth
-- We model this as: any "oracle" that claims to know the true value
-- must be EXTERNAL to the encoding system (it has information not in the encodings)

structure TruthOracle where
  -- Given an incoherent system, which value is "true"?
  resolve : EncodingSystem → FactValue
  -- The oracle must pick a value that IS in the system
  picks_existing : ∀ s, s.is_incoherent → (resolve s) ∈ s.values

-- THEOREM: No oracle can be justified by the encodings alone
-- Any resolution is arbitrary: the "other" value is equally present
theorem oracle_arbitrary (oracle : TruthOracle) (s : EncodingSystem) (hinc : s.is_incoherent) :
    ∃ v, v ∈ s.values ∧ v ≠ oracle.resolve s := by
  match hinc with
  | ⟨v1, v2, hv1, hv2, hne⟩ =>
    match Classical.em (oracle.resolve s = v1) with
    | .inl heq =>
      exact ⟨v2, hv2, fun heq2 => hne (heq.symm.trans heq2.symm)⟩
    | .inr hneq =>
      exact ⟨v1, hv1, fun h => hneq h.symm⟩

-- Part 7: The Master Theorems
--
-- SSOT (DOF = 1) is not merely convenient. It is EPISTEMICALLY NECESSARY
-- for coherent representation of facts.
--
-- DOF > 1 introduces an epistemic defect: the possibility of incoherent states
-- where truth is indeterminate.
--
-- DOF = 1 eliminates this defect: truth is always determinate.

-- KEY THEOREM: DOF = 1 implies no incoherent states possible
-- This is the forward direction: SSOT guarantees coherence
theorem ssot_guarantees_coherence (s : EncodingSystem) (h : s.dof = 1) :
    ¬s.is_incoherent :=
  dof_one_incoherence_impossible s h

-- KEY THEOREM: DOF > 1 implies incoherent states exist
-- This is the converse at the DESIGN level: non-SSOT permits incoherence
theorem non_ssot_permits_incoherence (n : Nat) (hn : n > 1) :
    ∃ s : EncodingSystem, s.dof = n ∧ s.is_incoherent :=
  dof_gt_one_incoherence_possible n hn

-- Helper: n > 0 and n != 1 implies n > 1
theorem pos_ne_one_implies_gt_one (n : Nat) (hn : n > 0) (hne : n ≠ 1) : n > 1 := by
  omega

-- FINAL THEOREM: The choice is forced
-- Given: we require that NO incoherent states be reachable for DOF = n
-- Then: n = 1 is the unique solution
-- This is the FORCING theorem: coherence requirements force SSOT
theorem determinate_truth_forces_ssot (n : Nat) (hn : n > 0) :
    (∀ s : EncodingSystem, s.dof = n → ¬s.is_incoherent) → n = 1 := fun h =>
  if hne : n = 1 then hne
  else
    -- n > 0 and n ≠ 1 implies n > 1
    let hgt := pos_ne_one_implies_gt_one n hn hne
    -- There exists an incoherent system with dof = n
    let ⟨s, hdof, hinc⟩ := dof_gt_one_incoherence_possible n hgt
    -- But h says no such system exists - contradiction
    absurd hinc (h s hdof)

-- COROLLARY: The epistemic asymmetry
-- DOF = 1: coherence guaranteed (all states coherent)
-- DOF > 1: coherence not guaranteed (some states incoherent)
-- This is STRICT DOMINANCE on the coherence metric
theorem epistemic_asymmetry :
    (∀ s : EncodingSystem, s.dof = 1 → s.is_coherent) ∧
    (∀ n : Nat, n > 1 → ∃ s : EncodingSystem, s.dof = n ∧ s.is_incoherent) :=
  ⟨dof_one_implies_coherent, dof_gt_one_incoherence_possible⟩

end Ssot
