/-
  Paper 1: Typing Discipline and Axis Framework

  CrossPaperDependencies.lean

  Paper 1 is FOUNDATIONAL. It does NOT depend on other papers.
  Papers 2 and 3 depend on Paper 1.

  This file documents what Paper 1 provides for other papers to use.
  NO IMPORTS FROM OTHER PAPERS.
-/

import axis_framework
import AbstractClassSystem.AxisClosure

namespace Paper1.CrossPaperDeps

/-! ## What Paper 1 Exports (No Dependencies) -/

/-- Paper 1 provides Axis type and AxisSet (= List Axis).
    Other papers can use these to model typed coordinates. -/
theorem axis_set_is_list : (AxisSet : Type _) = List Axis := rfl

/-- Paper 1 provides derivable relation on axes.
    Paper 2 uses this for redundancy analysis.
    Paper 3 uses this for axis independence. -/
theorem derivable_is_reflexive (a : Axis) : derivable a a := derivable_refl a

/-- Paper 1 provides derivable transitivity.
    Used by Paper 3 for derivability chains. -/
theorem derivable_is_transitive {a b c : Axis}
    (hab : derivable a b) (hbc : derivable b c) :
    derivable a c := derivable_trans hab hbc

/-- Paper 1 provides minimal sets.
    Paper 2 uses this for SSOT characterization.
    Paper 3 uses this for architecture minimality. -/
theorem minimal_def (A : AxisSet) (D : Domain α) :
    minimal A D ↔ complete A D ∧ ∀ a ∈ A, ¬ complete (A.erase a) D := Iff.rfl

/-- Paper 1 provides independent axes.
    Paper 2 uses this for coherence.
    Paper 3 uses this for DOF counting. -/
theorem independent_def (a : Axis) (A : AxisSet) (D : Domain α) :
    independent a A D ↔ ∃ q ∈ D, a ∈ q.requires ∧ ¬ canAnswer (A.erase a) q := Iff.rfl

/-- Paper 1 provides orthogonal axes.
    Paper 3 uses this for leverage analysis. -/
theorem orthogonal_def (A : AxisSet) :
    OrthogonalAxes A ↔ ∀ a ∈ A, ∀ b ∈ A, a ≠ b → ¬derivable a b := Iff.rfl

/-- Paper 1 provides exchange property.
    Paper 3 uses this for capability-preserving refactoring. -/
theorem exchange_def (A : AxisSet) :
    exchangeProperty A ↔ ∀ I J : AxisSet, J.Nodup → I ⊆ A → J ⊆ A →
        axisIndependent I → axisIndependent J → I.length < J.length →
        ∃ x ∈ J, x ∉ I ∧ axisIndependent (x :: I) := Iff.rfl

/-- Paper 1 proves minimal sets are independent.
    Used by Paper 2 and Paper 3. -/
theorem minimal_implies_independent {A : AxisSet} {D : Domain α}
    {a : Axis} (ha : a ∈ A) (hmin : minimal A D) :
    independent a A D := independence_of_minimal ha hmin

/-- Paper 1 proves semantically minimal implies orthogonal.
    Used by Paper 3 for leverage bounds. -/
theorem sem_minimal_orthogonal {A : AxisSet} {D : Domain α}
    (hmin : semanticallyMinimal A D) :
    OrthogonalAxes A := semantically_minimal_implies_orthogonal hmin

/-- Paper 1 proves orthogonal implies exchange.
    Used by Paper 3 for refactoring theorems. -/
theorem orth_exchange {A : AxisSet}
    (horth : OrthogonalAxes A) :
    exchangeProperty A := orthogonal_implies_exchange horth

/-- Paper 1 provides Domain and Query types.
    Paper 2 uses these for SSOT grounding.
    Paper 3 uses these for architecture queries.

    Note: Domain α is definitionally equal to List (Query α). -/
example (α : Type) : Domain α = List (Query α) := rfl

end Paper1.CrossPaperDeps
