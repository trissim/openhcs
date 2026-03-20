/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/CoulombApproximation.lean

  Finite-domain exact/coarse approximation for Coulomb-style scoring.
-/
import DecisionQuotient.Tractability.EwaldSummation
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.CutoffEpsilon
import DecisionQuotient.Tractability.LatticeSum
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace CoulombApproximation

open Ewald
open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open LatticeSum
open Classical

universe u v

/-- Exact single-pair Coulomb score. -/
noncomputable def exactCoulombScore (q_i q_j r : ℝ) : ℝ :=
  coulombPotential q_i q_j r

/-- Hard-cutoff Coulomb score. -/
noncomputable def cutoffCoulombScore (q_i q_j rc r : ℝ) : ℝ :=
  if r < rc then coulombPotential q_i q_j r else 0

noncomputable def exactCoulombDecisionProblem {A : Type u} {S : Type v}
    (q_i q_j : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => exactCoulombScore q_i q_j (distance a s)

noncomputable def cutoffCoulombDecisionProblem {A : Type u} {S : Type v}
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffCoulombScore q_i q_j rc (distance a s)

noncomputable def coulombCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactCoulombScore q_i q_j (distance p.1 p.2) - cutoffCoulombScore q_i q_j rc (distance p.1 p.2)|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem coulombCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ)
    (a : A) (s : S) :
    |exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)| ≤
      coulombCutoffErrorRadius q_i q_j rc distance := by
  classical
  let diffs : Finset ℝ :=
    (Finset.univ : Finset (A × S)).image
      (fun p => |exactCoulombScore q_i q_j (distance p.1 p.2) - cutoffCoulombScore q_i q_j rc (distance p.1 p.2)|)
  have hMem : |exactCoulombScore q_i q_j (distance a s) - cutoffCoulombScore q_i q_j rc (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  unfold coulombCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

theorem exact_vs_cutoff_coulomb_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (q_i q_j rc : ℝ) (distance : A → S → ℝ) :
    UniformUtilityApprox
      (exactCoulombDecisionProblem q_i q_j distance)
      (cutoffCoulombDecisionProblem q_i q_j rc distance)
      (coulombCutoffErrorRadius q_i q_j rc distance) := by
  intro a s
  simpa [exactCoulombDecisionProblem, cutoffCoulombDecisionProblem] using
    coulombCutoffErrorRadius_spec q_i q_j rc distance a s

/-- Coulomb-family packaging theorem: once a Coulomb tail perturbation bound is
    proved for the chosen utility, the finite-gap theorem yields an explicit
    `SatisfiesBoundedPotential` witness. -/
theorem exactCoulomb_satisfiesBoundedPotential_of_tailBound
    (prob : MolecularSrank.MDBindingProblem)
    [Fintype MolecularSrank.MDAction] [Fintype MolecularSrank.MDState] [Nonempty MolecularSrank.MDState]
    (distance : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    (q_i q_j tail_coefficient : ℝ)
    (w : ∀ s : MolecularSrank.MDState,
      { a : MolecularSrank.MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => exactCoulombScore q_i q_j (distance a s))
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MolecularSrank.MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (MolecularSrank.numMDCoordinates prob),
        j ≠ MolecularSrank.proteinCoordFin prob atomIdx hAtomInProtein axis →
        MolecularSrank.mdProj prob s j = MolecularSrank.mdProj prob s' j) →
      ¬ MolecularSrank.atomWithinCutoff
        (MolecularSrank.proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MolecularSrank.MDAction,
        |exactCoulombScore q_i q_j (distance a s) - exactCoulombScore q_i q_j (distance a s')|
          ≤ tail_coefficient * latticeTailSum 6 R)) :
    SatisfiesBoundedPotential prob tail_coefficient (finiteMinimumGap prob w) := by
  apply satisfiesBoundedPotential_of_tailBound_and_finiteGap prob tail_coefficient w hGapPos
  intro atomIdx hAtomInProtein axis s s' R hR hSame hOutside a
  simpa [hUtility] using hTail atomIdx hAtomInProtein axis s s' R hR hSame hOutside a

/-- Exact Real-Space Ewald Coulomb score. The error function forces exponential decay. -/
noncomputable def exactRealEwaldScore (q_i q_j alpha r : ℝ) : ℝ :=
  coulombPotential q_i q_j r * DecisionQuotient.Tractability.Ewald.erfc (alpha * r)

/-- Absolute-value envelope for the exact real-space Ewald score. -/
theorem abs_exactRealEwaldScore_le_charge_envelope
    (q_i q_j alpha r : ℝ) (hr : 0 < r) (ha : 0 < alpha) :
    |exactRealEwaldScore q_i q_j alpha r| ≤
      |q_i * q_j| * ewaldRealSpaceCore r alpha := by
  have hx : 0 < alpha * r := by positivity
  unfold exactRealEwaldScore coulombPotential ewaldRealSpaceCore
  calc
    |(q_i * q_j) / r * DecisionQuotient.Tractability.Ewald.erfc (alpha * r)|
      = |(q_i * q_j) / r| * |DecisionQuotient.Tractability.Ewald.erfc (alpha * r)| := by rw [abs_mul]
    _ ≤ |(q_i * q_j) / r| * Real.exp (-((alpha * r) ^ 2)) := by
      gcongr
      exact DecisionQuotient.Tractability.Ewald.erfc_abs_le_exp_neg_sq hx
    _ = (|q_i * q_j| / r) * Real.exp (-((alpha * r) ^ 2)) := by
      rw [abs_div, abs_of_pos hr]
    _ = |q_i * q_j| * (Real.exp (-((alpha * r) ^ 2)) / r) := by
      field_simp [hr.ne']

/--
  Because Real-Space Ewald decays exponentially, it is easily dominated by
  the 6-power polynomial tail at large R, allowing us to re-use our lattice sum bounds.
 -/
theorem exactRealEwald_satisfiesBoundedPotential_of_tailBound
    (prob : MolecularSrank.MDBindingProblem)
    [Fintype MolecularSrank.MDAction] [Fintype MolecularSrank.MDState] [Nonempty MolecularSrank.MDState]
    (distance : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    (q_i q_j alpha tail_coefficient : ℝ)
    (w : ∀ s : MolecularSrank.MDState,
      { a : MolecularSrank.MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => exactRealEwaldScore q_i q_j alpha (distance a s))
    (hTail : ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
      (axis : Fin 3) (s s' : MolecularSrank.MDState) (R : ℝ),
      0 < R →
      (∀ j : Fin (MolecularSrank.numMDCoordinates prob),
        j ≠ MolecularSrank.proteinCoordFin prob atomIdx hAtomInProtein axis →
        MolecularSrank.mdProj prob s j = MolecularSrank.mdProj prob s' j) →
      ¬ MolecularSrank.atomWithinCutoff
        (MolecularSrank.proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
      (∀ a : MolecularSrank.MDAction,
        |exactRealEwaldScore q_i q_j alpha (distance a s) - exactRealEwaldScore q_i q_j alpha (distance a s')|
          ≤ tail_coefficient * latticeTailSum 6 R)) :
    SatisfiesBoundedPotential prob tail_coefficient (finiteMinimumGap prob w) := by
  apply satisfiesBoundedPotential_of_tailBound_and_finiteGap prob tail_coefficient w hGapPos
  intro atomIdx hAtomInProtein axis s s' R hR hSame hOutside a
  simpa [hUtility] using hTail atomIdx hAtomInProtein axis s s' R hR hSame hOutside a

end CoulombApproximation
end Tractability
end DecisionQuotient
