/-
  Paper 4: Decision-Relevant Uncertainty
  Tractability/HalogenBondApproximation.lean

  Halogen-bonding is modeled as a bounded three-factor multiplicative surrogate
  and reuses the generic finite exact-vs-coarse machinery for three-factor
  families.
-/
import DecisionQuotient.Tractability.DirectionalHBondApproximation

namespace DecisionQuotient
namespace Tractability
namespace HalogenBondApproximation

open DirectionalHBondApproximation

abbrev halogenBondScore := directionalHBondScore

noncomputable abbrev halogenBondDecisionProblem {A : Type u} {S : Type v}
    (radial donor acceptor : A → S → ℝ) : DecisionProblem A S :=
  directionalHBondDecisionProblem radial donor acceptor

noncomputable abbrev attractiveHalogenBondDecisionProblem {A : Type u} {S : Type v}
    (radial donor acceptor : A → S → ℝ) : DecisionProblem A S :=
  attractiveDirectionalHBondDecisionProblem radial donor acceptor

noncomputable abbrev finiteHalogenBondErrorRadius
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) : ℝ :=
  finiteDirectionalHBondErrorRadius
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

abbrev finiteHalogenBondErrorRadius_witnesses_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

abbrev finiteHalogenBondErrorRadius_nonneg
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_nonneg
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

abbrev exact_vs_coarse_halogenBond_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :=
  finiteDirectionalHBondErrorRadius_witnesses_uniformApprox
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

noncomputable abbrev exact_vs_coarse_halogenBond_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

abbrev exact_vs_coarse_halogenBond_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_certified_top1_sound
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

noncomputable abbrev exact_vs_coarse_halogenBond_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_coherent_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

noncomputable abbrev exact_vs_coarse_halogenBond_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_directionalHBond_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

abbrev exact_vs_coarse_attractiveHalogenBond_uniformApprox
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ) :=
  exact_vs_coarse_attractiveDirectionalHBond_uniformApprox
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse

noncomputable abbrev exact_vs_coarse_attractiveHalogenBond_certified_top1
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

abbrev exact_vs_coarse_attractiveHalogenBond_certified_top1_sound
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_certified_top1_sound
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

noncomputable abbrev exact_vs_coarse_attractiveHalogenBond_coherent_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_coherent_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

noncomputable abbrev exact_vs_coarse_attractiveHalogenBond_optimizer_witness
    {A : Type u} {S : Type v}
    [Fintype A] [Fintype S] [DecidableEq A] [Nonempty A] [Nonempty S] [LinearOrder A]
    (radialExact donorExact acceptorExact : A → S → ℝ)
    (radialCoarse donorCoarse acceptorCoarse : A → S → ℝ)
    (s : S) :=
  exact_vs_coarse_attractiveDirectionalHBond_optimizer_witness
    radialExact donorExact acceptorExact radialCoarse donorCoarse acceptorCoarse s

end HalogenBondApproximation
end Tractability
end DecisionQuotient
