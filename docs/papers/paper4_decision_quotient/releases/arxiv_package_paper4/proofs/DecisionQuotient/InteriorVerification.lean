import DecisionQuotient.Sufficiency
import Mathlib.Data.Set.Basic
import Mathlib.Data.Finset.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Tactic

/-!
  Paper 4 expansion: interior verification via tautological-coordinate dominance.

  This module formalizes the "check from the inside" layer:
  - Goal-class uncertainty over evaluators (utility functions),
  - class-tautological coordinates,
  - interior Pareto dominance on tautological coordinates,
  - tractable (boolean-decider) certificate interface,
  - one-sidedness: interior dominance is not full sufficiency.

  ## Triviality Level
  TRIVIAL: This explores a different verification approach (interior vs exterior).
  The nontrivial work is in the main sufficiency proofs.

  ## Dependencies
  - Chain: Sufficiency.lean (core definitions) → here
  - The nontrivial work is already in Sufficiency.lean and Reduction.lean
-/

namespace DecisionQuotient
namespace InteriorVerification

open Set

universe u v

variable {A : Type u} {S : Type v} {n : ℕ}
variable [CoordinateSpace S n]

/-! ## Goal-class model -/

/-- Goal class (audience contract): a non-empty class of admissible utility models
    over the same action/state structure. -/
structure GoalClass (A : Type u) (S : Type v) where
  utilityClass : Set (DecisionProblem A S)
  nonempty : utilityClass.Nonempty

/-- Coordinate `i` is class-tautological when it is relevant for every
    admissible utility model in the goal class. -/
def GoalClass.isTautologicalCoord (G : GoalClass A S) (i : Fin n) : Prop :=
  ∀ dp, dp ∈ G.utilityClass → dp.isRelevant i

/-- Tautological coordinate set `J_G`. -/
def GoalClass.tautologicalSet (G : GoalClass A S) : Set (Fin n) :=
  {i | G.isTautologicalCoord (n := n) i}

/-- Score model used to compare coordinate-wise improvement in a heterogeneous
    state space. (This is a formal stand-in for "improving coordinate j".) -/
abbrev CoordScore (S : Type v) (n : ℕ) := S → Fin n → ℚ

/-- Non-negative monotonicity at coordinate `i`:
    if only coordinate `i` improves (by the score model), utility does not drop. -/
def DecisionProblem.nonnegativelyMonotoneCoord
    (dp : DecisionProblem A S) (score : CoordScore S n) (i : Fin n) : Prop :=
  ∀ a s s',
    (∀ j : Fin n, j ≠ i → score s j = score s' j) →
    score s' i ≤ score s i →
    dp.utility a s' ≤ dp.utility a s

/-- Weaker class-tautological variant: coordinate `i` is non-negatively
    tautological when every class member is non-negatively monotone at `i`. -/
def GoalClass.isNonNegativelyTautologicalCoord
    (G : GoalClass A S) (score : CoordScore S n) (i : Fin n) : Prop :=
  ∀ dp, dp ∈ G.utilityClass →
    DecisionProblem.nonnegativelyMonotoneCoord (n := n) dp score i

/-! ## Interior dominance -/

/-- Agreement relation on a coordinate set (set form, not finset form). -/
def agreeOnSet (J : Set (Fin n)) (s s' : S) : Prop :=
  ∀ j : Fin n, j ∈ J → CoordinateSpace.proj s j = CoordinateSpace.proj s' j

/-- Interior Pareto dominance on a coordinate set `J` under a score model:
    weakly dominates on all `J` coordinates and strictly dominates on one. -/
def interiorParetoDominates
    (score : CoordScore S n) (J : Set (Fin n)) (s s' : S) : Prop :=
  (∀ j : Fin n, j ∈ J → score s' j ≤ score s j) ∧
  (∃ j : Fin n, j ∈ J ∧ score s' j < score s j)

/-! ## Utility-side monotonicity and universal non-rejection -/

/-- A utility model is class-monotone on `J` if weak coordinate dominance on `J`
    implies weak utility dominance for every action. -/
def DecisionProblem.classMonotoneOn
    (dp : DecisionProblem A S) (score : CoordScore S n) (J : Set (Fin n)) : Prop :=
  ∀ a s s',
    (∀ j : Fin n, j ∈ J → score s' j ≤ score s j) →
    dp.utility a s' ≤ dp.utility a s

/-- Goal-class monotonicity on `J`: every admissible utility model is monotone on `J`. -/
def GoalClass.classMonotoneOn
    (G : GoalClass A S) (score : CoordScore S n) (J : Set (Fin n)) : Prop :=
  ∀ dp, dp ∈ G.utilityClass →
    DecisionProblem.classMonotoneOn (n := n) dp score J

/-- Interior dominance implies universal non-rejection for any admissible utility
    model, under class-monotonicity on the checked coordinate set. -/
theorem interior_dominance_implies_universal_non_rejection
    (G : GoalClass A S) (score : CoordScore S n) (J : Set (Fin n))
    (hMono : GoalClass.classMonotoneOn (n := n) G score J)
    {dp : DecisionProblem A S} (hdp : dp ∈ G.utilityClass)
    {s s' : S} (hDom : interiorParetoDominates (n := n) score J s s') :
    ∀ a : A, dp.utility a s' ≤ dp.utility a s := by
  intro a
  exact hMono dp hdp a s s' hDom.1

/-! ## Tractable interior verification interface -/

/-- Detectability proxy for the tautological coordinate set:
    membership is decided by a boolean detector. -/
structure TautologicalSetIdentifiable (J : Set (Fin n)) where
  detect : Fin n → Bool
  detect_correct : ∀ i : Fin n, detect i = true ↔ i ∈ J

/-- Verifiability proxy for interior dominance:
    dominance is decided by a boolean checker. -/
structure InteriorDominanceVerifiable
    (score : CoordScore S n) (J : Set (Fin n)) where
  verify : S → S → Bool
  verify_correct :
    ∀ s s' : S, verify s s' = true ↔ interiorParetoDominates (n := n) score J s s'

/-- Tractable certificate form:
    if the goal-class slice is identifiable and interior dominance is verifiable,
    there exists a checker whose `true` output is exactly interior dominance. -/
theorem interior_verification_tractable_certificate
    (score : CoordScore S n) (J : Set (Fin n))
    (_hId : TautologicalSetIdentifiable (n := n) J)
    (hVer : InteriorDominanceVerifiable (n := n) score J) :
    ∃ verify : S → S → Bool,
      ∀ s s' : S, verify s s' = true ↔ interiorParetoDominates (n := n) score J s s' := by
  exact ⟨hVer.verify, hVer.verify_correct⟩

/-- Verified interior certificate implies universal non-rejection (for class members). -/
theorem interior_certificate_implies_non_rejection
    (G : GoalClass A S) (score : CoordScore S n) (J : Set (Fin n))
    (_hId : TautologicalSetIdentifiable (n := n) J)
    (hVer : InteriorDominanceVerifiable (n := n) score J)
    (hMono : GoalClass.classMonotoneOn (n := n) G score J)
    {dp : DecisionProblem A S} (hdp : dp ∈ G.utilityClass)
    {s s' : S} (hCert : hVer.verify s s' = true) :
    ∀ a : A, dp.utility a s' ≤ dp.utility a s := by
  have hDom : interiorParetoDominates (n := n) score J s s' :=
    (hVer.verify_correct s s').1 hCert
  exact interior_dominance_implies_universal_non_rejection
    (n := n) G score J hMono hdp hDom

/-! ## One-sidedness: interior dominance is not full sufficiency -/

/-- Set-indexed sufficiency notion (set version of coordinate sufficiency). -/
def DecisionProblem.isSufficientOnSet
    (dp : DecisionProblem A S) (J : Set (Fin n)) : Prop :=
  ∀ s s' : S, agreeOnSet (n := n) J s s' → dp.Opt s = dp.Opt s'

/-- Any explicit counterexample on coordinates outside the checked set
    blocks full sufficiency on that set. -/
theorem not_sufficientOnSet_of_counterexample
    (dp : DecisionProblem A S) (J : Set (Fin n))
    (hCounter : ∃ t t' : S,
      agreeOnSet (n := n) J t t' ∧ dp.Opt t ≠ dp.Opt t') :
    ¬ DecisionProblem.isSufficientOnSet (n := n) dp J := by
  intro hSuff
  rcases hCounter with ⟨t, t', hAgree, hNe⟩
  exact hNe (hSuff t t' hAgree)

/-- Interior dominance certificate is one-sided:
    it does not imply full coordinate sufficiency in the presence of a
    non-tautological counterexample. -/
theorem interior_dominance_not_full_sufficiency
    (dp : DecisionProblem A S) (score : CoordScore S n) (J : Set (Fin n))
    {s s' : S}
    (_hDom : interiorParetoDominates (n := n) score J s s')
    (hCounter : ∃ t t' : S,
      agreeOnSet (n := n) J t t' ∧ dp.Opt t ≠ dp.Opt t') :
    ¬ DecisionProblem.isSufficientOnSet (n := n) dp J := by
  exact not_sufficientOnSet_of_counterexample (n := n) dp J hCounter

/-! ## Singleton-coordinate certificate (artifact-style specialization) -/

/-- Singleton-coordinate interior certificate:
    if one coordinate is checked and strictly improved, then under class
    monotonicity on that singleton coordinate we obtain universal non-rejection. -/
theorem singleton_coordinate_interior_certificate
    (G : GoalClass A S) (score : CoordScore S n) (i : Fin n)
    (hMono : GoalClass.classMonotoneOn (n := n) G score {j : Fin n | j = i})
    {dp : DecisionProblem A S} (hdp : dp ∈ G.utilityClass)
    {s s' : S}
    (hWeak : score s' i ≤ score s i)
    (hStrict : score s' i < score s i) :
    ∀ a : A, dp.utility a s' ≤ dp.utility a s := by
  have hDom : interiorParetoDominates (n := n) score {j : Fin n | j = i} s s' := by
    refine ⟨?_, ?_⟩
    · intro j hj
      have hjEq : j = i := by simpa using hj
      simpa [hjEq] using hWeak
    · exact ⟨i, by simp, hStrict⟩
  exact interior_dominance_implies_universal_non_rejection
    (n := n) G score {j : Fin n | j = i} hMono hdp hDom

end InteriorVerification
end DecisionQuotient
