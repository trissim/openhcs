import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

"""
Results aggregation for benchmarking.
Computes speedup, RMSE, Kendall τ, success rate.
"""

@dataclass(frozen=True)
class BenchmarkEntry:
    """Single benchmark comparison entry."""
    pdb_code: str
    dq_time_s: float
    vina_time_s: float
    dq_affinity: float
    vina_affinity: float
    experimental_affinity: float
    dq_srank: int
    dq_tractability: str

@dataclass
class BenchmarkResults:
    """Aggregated benchmark results."""
    entries: List[BenchmarkEntry] = field(default_factory=list)
    
    @property
    def n_complexes(self) -> int:
        return len(self.entries)
    
    @property
    def mean_speedup(self) -> float:
        """DQ speedup over Vina."""
        speedups = [e.vina_time_s / max(e.dq_time_s, 1e-6) for e in self.entries]
        return sum(speedups) / max(len(speedups), 1)
    
    @property
    def median_speedup(self) -> float:
        speedups = sorted(e.vina_time_s / max(e.dq_time_s, 1e-6) for e in self.entries)
        n = len(speedups)
        if n == 0:
            return 0
        return speedups[n // 2]
    
    def rmse_dq(self) -> float:
        """RMSE of DQ predictions vs experimental."""
        if not self.entries:
            return 0
        errors = [(e.dq_affinity - e.experimental_affinity) ** 2 for e in self.entries]
        return (sum(errors) / len(errors)) ** 0.5
    
    def rmse_vina(self) -> float:
        """RMSE of Vina predictions vs experimental."""
        if not self.entries:
            return 0
        errors = [(e.vina_affinity - e.experimental_affinity) ** 2 for e in self.entries]
        return (sum(errors) / len(errors)) ** 0.5
    
    def kendall_tau(self) -> float:
        """Kendall τ rank correlation of DQ vs experimental."""
        if len(self.entries) < 2:
            return 0
        n = len(self.entries)
        concordant = 0
        discordant = 0
        for i in range(n):
            for j in range(i + 1, n):
                dq_diff = self.entries[i].dq_affinity - self.entries[j].dq_affinity
                exp_diff = self.entries[i].experimental_affinity - self.entries[j].experimental_affinity
                if dq_diff * exp_diff > 0:
                    concordant += 1
                elif dq_diff * exp_diff < 0:
                    discordant += 1
        total = concordant + discordant
        return (concordant - discordant) / total if total > 0 else 0
    
    def success_rate(self, rmsd_threshold: float = 2.0) -> float:
        """Fraction of poses within RMSD threshold of native."""
        # Placeholder: requires RMSD computation
        return 0.0
    
    def by_tractability(self) -> Dict[str, 'BenchmarkResults']:
        """Group results by tractability class."""
        groups: Dict[str, List[BenchmarkEntry]] = {}
        for entry in self.entries:
            key = entry.dq_tractability
            groups.setdefault(key, []).append(entry)
        return {k: BenchmarkResults(entries=v) for k, v in groups.items()}
    
    def summary(self) -> dict:
        return {
            "n_complexes": self.n_complexes,
            "mean_speedup": self.mean_speedup,
            "median_speedup": self.median_speedup,
            "rmse_dq": self.rmse_dq(),
            "rmse_vina": self.rmse_vina(),
            "kendall_tau": self.kendall_tau(),
        }
    
    def save(self, path: str) -> None:
        """Save results to JSON."""
        data = {
            "summary": self.summary(),
            "entries": [
                {
                    "pdb_code": e.pdb_code,
                    "dq_time_s": e.dq_time_s,
                    "vina_time_s": e.vina_time_s,
                    "dq_affinity": e.dq_affinity,
                    "vina_affinity": e.vina_affinity,
                    "experimental_affinity": e.experimental_affinity,
                    "dq_srank": e.dq_srank,
                    "dq_tractability": e.dq_tractability,
                }
                for e in self.entries
            ]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load(path: str) -> 'BenchmarkResults':
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)
        entries = [BenchmarkEntry(**e) for e in data["entries"]]
        return BenchmarkResults(entries=entries)
