/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/LJApproximation.lean

  First concrete exact/coarse scorer pair: exact Lennard-Jones versus cutoff
  Lennard-Jones on a finite sampled domain.
-/
import DecisionQuotient.Computation.ArrayDSL
import DecisionQuotient.Tractability.CoarseApproximation
import DecisionQuotient.Tractability.CutoffEpsilon
import DecisionQuotient.Tractability.FormalLocalOptimizer
import DecisionQuotient.Tractability.LatticeSum
import Mathlib.Data.Finset.Max

namespace DecisionQuotient
namespace Tractability
namespace LJApproximation

open Computation.ArrayDSL
open CoarseApproximation
open CertifiedPruning
open FiniteTopK
open NearTieBand
open FormalLocalOptimizer
open Tractability
open LatticeSum
open Classical

universe u v

/-- Exact single-pair Lennard-Jones score. -/
noncomputable def exactLJScore (ε σ : ℝ) (r : ℝ) : ℝ :=
  lennardJones ε σ r

/-- Hard-cutoff coarse single-pair Lennard-Jones score. -/
noncomputable def cutoffLJScore (ε σ rc : ℝ) (r : ℝ) : ℝ :=
  if r < rc then lennardJones ε σ r else 0

/-- Decision problem induced by an exact LJ score over a sampled distance map. -/
noncomputable def exactLJDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ : ℝ) : DecisionProblem A S where
  utility := fun a s => exactLJScore ε σ (distance a s)

/-- Decision problem induced by a cutoff LJ score over a sampled distance map. -/
noncomputable def cutoffLJDecisionProblem {A : Type u} {S : Type v}
    (distance : A → S → ℝ) (ε σ rc : ℝ) : DecisionProblem A S where
  utility := fun a s => cutoffLJScore ε σ rc (distance a s)

/-- The finite set of sampled distances realized by the action/state domain. -/
noncomputable def sampledDistances {A : Type u} {S : Type v}
    [Fintype A] [Fintype S]
    (distance : A → S → ℝ) : Finset ℝ :=
  (Finset.univ : Finset (A × S)).image (fun p => distance p.1 p.2)

/-- Max exact-vs-cutoff discrepancy over the sampled distance set. -/
noncomputable def ljCutoffErrorRadius {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) : ℝ :=
  let diffs : Finset ℝ :=
    (sampledDistances distance).image (fun r => |exactLJScore ε σ r - cutoffLJScore ε σ rc r|)
  diffs.max' <| by
    rcases ‹Nonempty A› with ⟨a⟩
    rcases ‹Nonempty S› with ⟨s⟩
    refine ⟨|exactLJScore ε σ (distance a s) - cutoffLJScore ε σ rc (distance a s)|, ?_⟩
    refine Finset.mem_image.mpr ?_
    refine ⟨distance a s, ?_, rfl⟩
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩

theorem ljCutoffErrorRadius_spec {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ)
    (a : A) (s : S) :
    |exactLJScore ε σ (distance a s) - cutoffLJScore ε σ rc (distance a s)| ≤
      ljCutoffErrorRadius distance ε σ rc := by
  classical
  let diffs : Finset ℝ :=
    (sampledDistances distance).image (fun r => |exactLJScore ε σ r - cutoffLJScore ε σ rc r|)
  have hDistMem : distance a s ∈ sampledDistances distance := by
    refine Finset.mem_image.mpr ?_
    exact ⟨(a, s), by simp, rfl⟩
  have hMem : |exactLJScore ε σ (distance a s) - cutoffLJScore ε σ rc (distance a s)| ∈ diffs := by
    refine Finset.mem_image.mpr ?_
    exact ⟨distance a s, hDistMem, rfl⟩
  unfold ljCutoffErrorRadius
  exact Finset.le_max' diffs _ hMem

/-- Explicit cutoff-side coefficient obtained by normalizing the finite-domain
    LJ cutoff error radius by the lattice tail term at the chosen cutoff. -/
noncomputable def ljCutoffTailCoefficient {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) : ℝ :=
  ljCutoffErrorRadius distance ε σ rc / latticeTailSum 6 rc

theorem ljCutoffErrorRadius_eq_tailCoefficient_mul
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ)
    (hTail : latticeTailSum 6 rc ≠ 0) :
    ljCutoffErrorRadius distance ε σ rc =
      ljCutoffTailCoefficient distance ε σ rc * latticeTailSum 6 rc := by
  unfold ljCutoffTailCoefficient
  field_simp [hTail]

/-- Concrete uniform-approximation theorem for exact LJ versus cutoff LJ on a
    finite sampled domain. -/
theorem exact_vs_cutoff_lj_uniformApprox {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) :
    UniformUtilityApprox
      (exactLJDecisionProblem distance ε σ)
      (cutoffLJDecisionProblem distance ε σ rc)
      (ljCutoffErrorRadius distance ε σ rc) := by
  intro a s
  simpa [exactLJDecisionProblem, cutoffLJDecisionProblem] using
    ljCutoffErrorRadius_spec distance ε σ rc a s

theorem exact_vs_cutoff_lj_uniformApprox_with_tailCoefficient {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ)
    (hTail : latticeTailSum 6 rc ≠ 0) :
    UniformUtilityApprox
      (exactLJDecisionProblem distance ε σ)
      (cutoffLJDecisionProblem distance ε σ rc)
      (ljCutoffTailCoefficient distance ε σ rc * latticeTailSum 6 rc) := by
  rw [← ljCutoffErrorRadius_eq_tailCoefficient_mul distance ε σ rc hTail]
  exact exact_vs_cutoff_lj_uniformApprox distance ε σ rc

theorem exact_vs_cutoff_lj_opt_invariance {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ)
    (s : S) (aStar : A)
    (hStrict : StrictOpt (exactLJDecisionProblem distance ε σ) aStar s)
    (hBound :
      ljCutoffErrorRadius distance ε σ rc <
        StrictUtilityGap (exactLJDecisionProblem distance ε σ) aStar s / 2) :
    (exactLJDecisionProblem distance ε σ).Opt s =
      (cutoffLJDecisionProblem distance ε σ rc).Opt s :=
  by
    have hDelta : 0 ≤ ljCutoffErrorRadius distance ε σ rc := by
      rcases ‹Nonempty A› with ⟨a⟩
      rcases ‹Nonempty S› with ⟨s0⟩
      exact le_trans (abs_nonneg _) (ljCutoffErrorRadius_spec distance ε σ rc a s0)
    exact uniform_approx_implies_opt_invariance
      (exactLJDecisionProblem distance ε σ)
      (cutoffLJDecisionProblem distance ε σ rc)
      (ljCutoffErrorRadius distance ε σ rc)
      (exact_vs_cutoff_lj_uniformApprox distance ε σ rc)
      s aStar hDelta hStrict hBound

theorem ljCutoffErrorRadius_nonneg {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) :
    0 ≤ ljCutoffErrorRadius distance ε σ rc := by
  rcases ‹Nonempty A› with ⟨a⟩
  rcases ‹Nonempty S› with ⟨s⟩
  exact le_trans (abs_nonneg _) (ljCutoffErrorRadius_spec distance ε σ rc a s)

/--
  Exact-vs-cutoff Lennard-Jones induces a theorem-backed certified top-1 survivor
  set at every sampled state.
-/
noncomputable def exact_vs_cutoff_lj_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) (s : S) :
    CertifiedSurvivorSet A :=
  certified_top1_survivor_set_of_uniformApprox
    (fun a => exactLJDecisionProblem distance ε σ |>.utility a s)
    (fun a => cutoffLJDecisionProblem distance ε σ rc |>.utility a s)
    (ljCutoffErrorRadius distance ε σ rc)
    (fun a => exact_vs_cutoff_lj_uniformApprox distance ε σ rc a s)
    (ljCutoffErrorRadius_nonneg distance ε σ rc)

/-- Soundness of the exact-vs-cutoff LJ certified top-1 survivor set. -/
theorem exact_vs_cutoff_lj_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (distance : A → S → ℝ) (ε σ rc : ℝ) (s : S) :
    (certificate_of_top1_coarse_ambiguityBand
      (fun a => exactLJDecisionProblem distance ε σ |>.utility a s)
      (fun a => cutoffLJDecisionProblem distance ε σ rc |>.utility a s)
      (ljCutoffErrorRadius distance ε σ rc)
      (fun a => exact_vs_cutoff_lj_uniformApprox distance ε σ rc a s)
      (ljCutoffErrorRadius_nonneg distance ε σ rc)).exactTopK
      ⊆ (exact_vs_cutoff_lj_certified_top1 distance ε σ rc s).survivors := by
  simpa [exact_vs_cutoff_lj_certified_top1]
    using certified_top1_survivor_set_of_uniformApprox_sound
      (fun a => exactLJDecisionProblem distance ε σ |>.utility a s)
      (fun a => cutoffLJDecisionProblem distance ε σ rc |>.utility a s)
      (ljCutoffErrorRadius distance ε σ rc)
      (fun a => exact_vs_cutoff_lj_uniformApprox distance ε σ rc a s)
      (ljCutoffErrorRadius_nonneg distance ε σ rc)

/--
  Exact-vs-cutoff Lennard-Jones also yields a runtime-facing optimizer witness
  for the ambiguity-band selection branch.
-/
noncomputable def exact_vs_cutoff_lj_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rc : ℝ) (s : S) :
    CoherentOptimizerWitness A :=
  coherent_optimizer_witness_of_uniformApprox_top1
    (fun a => exactLJDecisionProblem distance ε σ |>.utility a s)
    (fun a => cutoffLJDecisionProblem distance ε σ rc |>.utility a s)
    (ljCutoffErrorRadius distance ε σ rc)
    (fun a => exact_vs_cutoff_lj_uniformApprox distance ε σ rc a s)
    (ljCutoffErrorRadius_nonneg distance ε σ rc)

noncomputable def exact_vs_cutoff_lj_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (distance : A → S → ℝ) (ε σ rc : ℝ) (s : S) :
    OptimizerWitness A :=
  (exact_vs_cutoff_lj_coherent_optimizer_witness distance ε σ rc s).toOptimizerWitness

noncomputable def exact_vs_cutoff_lj_pruning_certificate {A : Type u}
    [Fintype A] [DecidableEq A] [Nonempty A]
    (uDistance : A → ℝ) (ε σ rc : ℝ)
    (k : Nat)
    (tau : ℝ)
    (hMargin : ∀ a,
      a ∈ topKWithTies (fun x => exactLJScore ε σ (uDistance x)) k →
      tau + ljCutoffErrorRadius (fun (a : A) (_ : Unit) => uDistance a) ε σ rc ≤ exactLJScore ε σ (uDistance a)) :
    PruningCertificate A :=
  uniform_approx_pruning_certificate
    (fun a => exactLJScore ε σ (uDistance a))
    (fun a => cutoffLJScore ε σ rc (uDistance a))
    k tau (ljCutoffErrorRadius (fun (a : A) (_ : Unit) => uDistance a) ε σ rc)
    (by
      intro a
      simpa using ljCutoffErrorRadius_spec (fun (a : A) (_ : Unit) => uDistance a) ε σ rc a ())
    hMargin

/--
  A physical distance map guarantees that moving an atom outside the cutoff
  perturbs the exact Lennard-Jones energy by at most the 6-power lattice tail.
 -/
def PhysicalDistanceDecayLJ
    (prob : MolecularSrank.MDBindingProblem)
    (distance : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    (ε σ tail_coeff : ℝ) : Prop :=
  ∀ (atomIdx : Nat) (hAtomInProtein : atomIdx < prob.protein.numAtoms)
    (axis : Fin 3) (s s' : MolecularSrank.MDState) (R : ℝ),
    0 < R →
    (∀ j : Fin (MolecularSrank.numMDCoordinates prob),
      j ≠ MolecularSrank.proteinCoordFin prob atomIdx hAtomInProtein axis →
      MolecularSrank.mdProj prob s j = MolecularSrank.mdProj prob s' j) →
    ¬ MolecularSrank.atomWithinCutoff
      (MolecularSrank.proteinAtom prob atomIdx hAtomInProtein) prob.bindingSite R →
    (∀ a : MolecularSrank.MDAction,
      |exactLJScore ε σ (distance a s) - exactLJScore ε σ (distance a s')|
        ≤ tail_coeff * latticeTailSum 6 R)

/-- Final concrete instantiation: Exact LJ satisfies bounded potential. -/
theorem exactLJ_is_BoundedPotential
    (prob : MolecularSrank.MDBindingProblem)
    [Fintype MolecularSrank.MDAction] [Fintype MolecularSrank.MDState] [Nonempty MolecularSrank.MDState]
    (distance : MolecularSrank.MDAction → MolecularSrank.MDState → ℝ)
    (ε σ tail_coeff : ℝ)
    (w : ∀ s : MolecularSrank.MDState,
      { a : MolecularSrank.MDAction // StrictOpt prob.toDecisionProblem a s })
    (hGapPos : 0 < finiteMinimumGap prob w)
    (hUtility : prob.utility = fun a s => exactLJScore ε σ (distance a s))
    (hPhys : PhysicalDistanceDecayLJ prob distance ε σ tail_coeff) :
    SatisfiesBoundedPotential prob tail_coeff (finiteMinimumGap prob w) := by
  apply satisfiesBoundedPotential_of_tailBound_and_finiteGap prob tail_coeff w hGapPos
  intro atomIdx hAtomInProtein axis s s' R hR hSame hOutside a
  simpa [hUtility] using hPhys atomIdx hAtomInProtein axis s s' R hR hSame hOutside a

end LJApproximation
end Tractability
end DecisionQuotient
