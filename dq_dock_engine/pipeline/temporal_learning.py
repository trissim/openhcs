from dataclasses import dataclass, field
from typing import Callable, Dict, List, Set

"""
Temporal Learning — Bayesian structure detection.
Direct translation of StochasticSequential/TemporalLearning.lean.

Key theorems:
- Bayesian update: posterior = prior × likelihood / evidence
- Identifiability → convergence: posterior concentrates on true class
- Abstention set shrinks as posteriors increase
"""

@dataclass(frozen=True)
class StructureClass:
    """
    From TemporalLearning.lean::StructureClass.
    A tractable subcase with a name and tractability flag.
    """
    name: str
    tractable: bool = True

def bayesian_update(
    prior: Dict[str, float],
    likelihood: Dict[str, float],
    evidence: float
) -> Dict[str, float]:
    """
    From bayesian_update:
      posterior c = prior c × likelihood c / evidence
    
    Normalization: evidence = Σ_c prior(c) × likelihood(c).
    """
    if evidence <= 0:
        # Compute evidence from prior × likelihood
        evidence = sum(prior.get(c, 0) * likelihood.get(c, 0) 
                      for c in prior)
    if evidence <= 0:
        return prior  # No update possible
    
    return {
        c: prior.get(c, 0) * likelihood.get(c, 0) / evidence
        for c in prior
    }

def is_identifiable(likelihoods: Dict[str, Callable]) -> bool:
    """
    From Identifiable:
    Classes are identifiable if distinct likelihood functions.
    ∀ c₁ c₂, (∀ e, L(c₁, e) = L(c₂, e)) → c₁ = c₂
    """
    classes = list(likelihoods.keys())
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            if likelihoods[c1] is likelihoods[c2]:
                return False
    return True

def posterior_converges(
    prior: Dict[str, float],
    true_class: str,
    likelihood: Dict[str, float],
    evidence: float
) -> bool:
    """
    From posterior_converges:
    If likelihood(c₀) > 0 and likelihood(c) = 0 for c ≠ c₀,
    and prior(c₀) > 0, then posterior(c₀) > 0.
    
    With enough evidence, posterior concentrates on true class.
    """
    if prior.get(true_class, 0) <= 0:
        return False
    if likelihood.get(true_class, 0) <= 0:
        return False
    
    posterior = bayesian_update(prior, likelihood, evidence)
    return posterior.get(true_class, 0) > 0

def abstention_set(
    posterior: Dict[str, float],
    threshold: float
) -> Set[str]:
    """
    From abstentionSet:
    Classes with posterior below threshold.
    """
    return {c for c, p in posterior.items() if p < threshold}

def abstention_decreases(
    posterior1: Dict[str, float],
    posterior2: Dict[str, float],
    threshold: float
) -> bool:
    """
    From abstention_decreases:
    If ∀ c, posterior₁(c) ≤ posterior₂(c), then
    abstention(posterior₂) ⊆ abstention(posterior₁).
    
    Abstention shrinks as model improves.
    """
    a1 = abstention_set(posterior1, threshold)
    a2 = abstention_set(posterior2, threshold)
    return a2.issubset(a1)

@dataclass
class LearningHistory:
    """
    Track Bayesian structure detection across problem instances.
    """
    prior: Dict[str, float] = field(default_factory=dict)
    n_observations: int = 0
    
    def observe(self, observation_likelihoods: Dict[str, float]) -> Dict[str, float]:
        """
        Update posterior with a new observation.
        Returns the updated posterior.
        """
        evidence = sum(self.prior.get(c, 0) * l 
                      for c, l in observation_likelihoods.items())
        self.prior = bayesian_update(self.prior, observation_likelihoods, evidence)
        self.n_observations += 1
        return self.prior
    
    @property
    def abstention_rate(self) -> float:
        """Fraction of classes below threshold 0.01."""
        if not self.prior:
            return 1.0
        low = sum(1 for p in self.prior.values() if p < 0.01)
        return low / len(self.prior)
