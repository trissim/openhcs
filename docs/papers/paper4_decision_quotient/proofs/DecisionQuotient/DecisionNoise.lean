/-
  Paper 4: Decision-Relevant Uncertainty

  DecisionNoise.lean - Decision noise is exactly decision-irrelevant variation.

  The point of this file is to package three equivalent readings of "noise":

  1. State-level noise: two states differ only by decision-noise when they induce
     the same optimal-action set, hence lie in the same optimizer-quotient class.
  2. Coordinate-level noise: a coordinate is decision-noise exactly when varying it
     alone never changes the optimal-action set.
  3. Probabilistic noise: in the finite setting, a noisy coordinate is exactly one
     that is conditionally independent of the decision given the remaining
     coordinates.
-/

import DecisionQuotient.Quotient
import DecisionQuotient.Statistics.ProbabilisticBridge

namespace DecisionQuotient

open Classical

variable {A S : Type*}

/-- State-level decision noise is exactly the kernel relation of the optimizer
map: two states differ only by decision-noise when they are decision-equivalent. -/
def DecisionProblem.stateDecisionNoise (dp : DecisionProblem A S) (s s' : S) : Prop :=
  dp.DecisionEquiv s s'

/-- The optimizer quotient removes exactly the state-level decision noise: two
states are identified precisely when they differ only by decision-noise. -/
theorem DecisionProblem.stateDecisionNoise_iff_same_quotient
    (dp : DecisionProblem A S) (s s' : S) :
    dp.stateDecisionNoise s s' ↔ dp.quotientMap s = dp.quotientMap s' := by
  simpa [DecisionProblem.stateDecisionNoise] using
    (dp.quotient_represents_opt_equiv s s').symm

section CoordinateNoise

variable {n : ℕ} [CoordinateSpace S n]

/-- A coordinate is decision-noise exactly when it is irrelevant to the optimal
action set. This names the paper's geometric irrelevance condition in the more
interpretive "noise" language. -/
def DecisionProblem.isDecisionNoise (dp : DecisionProblem A S) (i : Fin n) : Prop :=
  dp.isIrrelevant i

/-- Decision-noise is exactly geometric irrelevance. -/
theorem DecisionProblem.decisionNoise_iff_irrelevant
    (dp : DecisionProblem A S) (i : Fin n) :
    dp.isDecisionNoise i ↔ dp.isIrrelevant i := by
  rfl

/-- Signal is exactly what survives after removing decision-noise. -/
theorem DecisionProblem.decisionNoise_iff_not_relevant
    (dp : DecisionProblem A S) (i : Fin n) :
    dp.isDecisionNoise i ↔ ¬ dp.isRelevant i := by
  simpa [DecisionProblem.isDecisionNoise] using dp.irrelevant_iff_not_relevant i

/-- A coordinate is decision-noise exactly when varying only that coordinate never
changes the optimizer-quotient class. -/
theorem DecisionProblem.decisionNoise_iff_quotient_invariant
    (dp : DecisionProblem A S) (i : Fin n) :
    dp.isDecisionNoise i ↔
      ∀ s s' : S,
        (∀ j : Fin n, j ≠ i → CoordinateSpace.proj s j = CoordinateSpace.proj s' j) →
        dp.quotientMap s = dp.quotientMap s' := by
  constructor
  · intro hnoise s s' hagree
    apply Quotient.sound
    exact hnoise s s' hagree
  · intro hquot s s' hagree
    exact (dp.quotient_represents_opt_equiv s s').mp (hquot s s' hagree)

section Probabilistic

variable [Fintype S]

/-- Decision-noise implies conditional independence of the decision from that
coordinate under any distribution. -/
theorem DecisionProblem.decisionNoise_implies_condIndep
    (dp : DecisionProblem A S) (i : Fin n)
    (hnoise : dp.isDecisionNoise i)
    (d : StochasticSequential.Distribution S) :
    Statistics.coordCondIndep dp i d := by
  exact Statistics.irrelevant_implies_condIndep dp i hnoise d

/-- In the finite deterministic setting, coordinate-level decision-noise is
equivalent to conditional independence of the decision given the remaining
coordinates. -/
theorem DecisionProblem.decisionNoise_iff_condIndep
    (dp : DecisionProblem A S) (i : Fin n)
    (d : StochasticSequential.Distribution S) :
    dp.isDecisionNoise i ↔ Statistics.coordCondIndep dp i d := by
  simpa [DecisionProblem.isDecisionNoise] using
    (Statistics.condIndep_iff_irrelevant dp i d).symm

end Probabilistic

end CoordinateNoise

end DecisionQuotient
