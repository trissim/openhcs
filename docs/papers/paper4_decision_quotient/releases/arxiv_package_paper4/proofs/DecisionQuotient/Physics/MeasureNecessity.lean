import Mathlib.MeasureTheory.Measure.Count
import Mathlib.MeasureTheory.Measure.Typeclasses.Probability
import Mathlib.MeasureTheory.Integral.Lebesgue.Basic
import Mathlib.Tactic

/-!
  Paper 4: Decision-Relevant Uncertainty

  MeasureNecessity.lean

  This module formalizes a typing-level separation:
  - quantitative claims require an underlying measure;
  - stochastic claims additionally require probability normalization.

  No new axioms are introduced.
-/

namespace DecisionQuotient
namespace Physics
namespace MeasureNecessity

open MeasureTheory

universe u

/-- Quantitative claims are represented by measurable observables under a measure. -/
structure QuantitativeClaim (α : Type u) [MeasurableSpace α] where
  μ : Measure α
  observable : α → ENNReal
  measurable_observable : Measurable observable

/-- Stochastic claims add the probability-measure normalization layer. -/
structure StochasticClaim (α : Type u) [MeasurableSpace α] extends QuantitativeClaim α where
  isProbability : IsProbabilityMeasure μ

/-- Quantitative claims are exactly measure-indexed measurable observables. -/
def quantitativeClaimEquivMeasureObservable
    (α : Type u) [MeasurableSpace α] :
    QuantitativeClaim α ≃ {p : Measure α × (α → ENNReal) // Measurable p.2} where
  toFun q := ⟨(q.μ, q.observable), q.measurable_observable⟩
  invFun p := { μ := p.1.1, observable := p.1.2, measurable_observable := p.2 }
  left_inv q := by
    cases q
    rfl
  right_inv p := by
    rcases p with ⟨⟨μ, f⟩, hf⟩
    rfl

/-- Stochastic claims are exactly probability-measure-indexed measurable observables. -/
def stochasticClaimEquivProbabilityObservable
    (α : Type u) [MeasurableSpace α] :
    StochasticClaim α ≃
      {p : Measure α × (α → ENNReal) // Measurable p.2 ∧ IsProbabilityMeasure p.1} where
  toFun q := ⟨(q.μ, q.observable), ⟨q.measurable_observable, q.isProbability⟩⟩
  invFun p := {
    μ := p.1.1
    observable := p.1.2
    measurable_observable := p.2.1
    isProbability := p.2.2
  }
  left_inv q := by
    cases q
    rfl
  right_inv p := by
    rcases p with ⟨⟨μ, f⟩, ⟨hf, hprob⟩⟩
    rfl

/-- Any quantitative claim carries an explicit underlying measure. -/
theorem quantitative_claim_has_measure
    {α : Type u} [MeasurableSpace α]
    (q : QuantitativeClaim α) :
    ∃ _μ : Measure α, True := by
  exact ⟨q.μ, trivial⟩

/-- Any stochastic claim carries an explicit probability measure. -/
theorem stochastic_claim_has_probability_measure
    {α : Type u} [MeasurableSpace α]
    (q : StochasticClaim α) :
    ∃ μ : Measure α, IsProbabilityMeasure μ := by
  exact ⟨q.μ, q.isProbability⟩

/-- Any stochastic claim also carries an underlying measure. -/
theorem stochastic_claim_has_measure
    {α : Type u} [MeasurableSpace α]
    (q : StochasticClaim α) :
    ∃ _μ : Measure α, True := by
  exact ⟨q.μ, trivial⟩

/-- On `Bool`, counting measure assigns total mass `2`. -/
theorem count_univ_bool :
    (Measure.count : Measure Bool) Set.univ = (2 : ENNReal) := by
  simpa using (Measure.count_univ (α := Bool))

/-- Counting measure on `Bool` is not a probability measure. -/
theorem counting_measure_not_probability_on_bool :
    ¬ IsProbabilityMeasure (Measure.count : Measure Bool) := by
  intro hprob
  have h1 : (Measure.count : Measure Bool) Set.univ = (1 : ENNReal) := by
    simpa using hprob.measure_univ
  have h2 : (Measure.count : Measure Bool) Set.univ = (2 : ENNReal) := by
    simpa using (Measure.count_univ (α := Bool))
  have : (1 : ENNReal) = (2 : ENNReal) := by
    calc
      (1 : ENNReal) = (Measure.count : Measure Bool) Set.univ := h1.symm
      _ = (2 : ENNReal) := h2
  norm_num at this

/-- Deterministic Dirac models remain measure-based and probability-normalized. -/
theorem deterministic_dirac_is_probability
    {α : Type u} [MeasurableSpace α] [MeasurableSingletonClass α]
    (a : α) :
    IsProbabilityMeasure (Measure.dirac a) := by
  infer_instance

/-- Quantitative integral values depend on the chosen measure. -/
theorem quantitative_value_depends_on_measure :
    (∫⁻ _ : Bool, (1 : ENNReal) ∂(Measure.count : Measure Bool)) ≠
    (∫⁻ _ : Bool, (1 : ENNReal) ∂(Measure.dirac true : Measure Bool)) := by
  have hLeft :
      (∫⁻ _ : Bool, (1 : ENNReal) ∂(Measure.count : Measure Bool)) = (2 : ENNReal) := by
    calc
      (∫⁻ _ : Bool, (1 : ENNReal) ∂(Measure.count : Measure Bool))
          = (Measure.count : Measure Bool) Set.univ := by simp
      _ = (2 : ENNReal) := by simpa using (Measure.count_univ (α := Bool))
  have hRight :
      (∫⁻ _ : Bool, (1 : ENNReal) ∂(Measure.dirac true : Measure Bool)) = (1 : ENNReal) := by
    simp
  intro hEq
  have : (2 : ENNReal) = (1 : ENNReal) := by
    calc
      (2 : ENNReal) = (∫⁻ _ : Bool, (1 : ENNReal) ∂(Measure.count : Measure Bool)) := hLeft.symm
      _ = (∫⁻ _ : Bool, (1 : ENNReal) ∂(Measure.dirac true : Measure Bool)) := hEq
      _ = (1 : ENNReal) := hRight
  norm_num at this

/-- Deterministic models still instantiate an explicit underlying measure. -/
theorem deterministic_models_still_measure_based
    {α : Type u} [MeasurableSpace α] [MeasurableSingletonClass α]
    (a : α) :
    ∃ μ : Measure α, IsProbabilityMeasure μ := by
  exact ⟨Measure.dirac a, deterministic_dirac_is_probability a⟩

/-- Measure structure alone does not imply probability normalization. -/
theorem measure_does_not_imply_probability :
    ∃ μ : Measure Bool, ¬ IsProbabilityMeasure μ := by
  exact ⟨Measure.count, counting_measure_not_probability_on_bool⟩

/-- Type-level prerequisite form: no quantitative claim can be represented
without a measure component in its typed payload. -/
def quantitative_measure_is_logical_prerequisite
    {α : Type u} [MeasurableSpace α] :
    QuantitativeClaim α ≃ {p : Measure α × (α → ENNReal) // Measurable p.2} :=
  quantitativeClaimEquivMeasureObservable α

/-- Type-level prerequisite form: stochastic claims require probability
normalization in addition to measure structure. -/
def stochastic_probability_is_logical_prerequisite
    {α : Type u} [MeasurableSpace α] :
    StochasticClaim α ≃
      {p : Measure α × (α → ENNReal) // Measurable p.2 ∧ IsProbabilityMeasure p.1} :=
  stochasticClaimEquivProbabilityObservable α

end MeasureNecessity
end Physics
end DecisionQuotient
