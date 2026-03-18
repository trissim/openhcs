from dataclasses import dataclass

"""
Fixed-Parameter Tractability analysis.
Direct translation of Tractability/FPT.lean.

Key results:
- SUFFICIENCY-CHECK is FPT in (|A|, |I|): O(|S|² · |A|²)
- FPT kernel: effective |S| ≤ 2^|I|
- MIN-SUFFICIENT-SET is W[2]-hard in k = |I*|
"""

def fpt_running_time(n_actions: int, n_states: int) -> int:
    """
    From FPT.lean::fptRunningTime:
      |S|² × |A|²
    """
    return n_states * n_states * n_actions * n_actions

def fpt_kernel_bound(sufficient_set_size: int) -> int:
    """
    From FPT.lean::fpt_kernel_bound:
      Effective |S| ≤ 2^|I| (states agreeing on I can be merged).
    """
    return 2 ** sufficient_set_size

@dataclass(frozen=True)
class FPTResult:
    """Parameterized complexity classification."""
    n_actions: int
    n_states: int
    sufficient_set_size: int
    
    @property
    def fpt_time(self) -> int:
        """FPT running time: O(|S|² · |A|²)."""
        return fpt_running_time(self.n_actions, self.n_states)
    
    @property
    def kernel_size(self) -> int:
        """FPT kernel: 2^|I|."""
        return fpt_kernel_bound(self.sufficient_set_size)
    
    @property
    def is_fpt_tractable(self) -> bool:
        """FPT tractable when |A| is bounded (constant)."""
        return self.n_actions <= 1000  # |A| bounded
    
    @property
    def complexity_class(self) -> str:
        """
        From parameterized_complexity_summary:
        |A| bounded → FPT
        |A| unbounded → W[1]-hard
        MIN-SUFFICIENT-SET → W[2]-hard
        """
        if self.n_actions <= 100:
            return "FPT"
        elif self.n_actions <= 1000:
            return "FPT (large kernel)"
        else:
            return "W[1]-hard (unbounded actions)"
