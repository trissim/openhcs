"""
Gap-Based Certification Engine

Orthogonal to scoring - handles certification decisions only.
"""

from dq_dock_engine.docking.core import GapCertification


class CertificationEngine:
    def __init__(self, error_bound: float):
        self.error_bound = error_bound

    def certify(self, energy_a: float, energy_b: float) -> GapCertification:
        return GapCertification.from_energies(energy_a, energy_b, self.error_bound)

    def certify_top_poses(
        self, scores: list[float], top_k: int = 1
    ) -> list[GapCertification]:
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i])
        best_idx = sorted_indices[0]
        certifications = []
        for i in range(1, min(top_k, len(sorted_indices))):
            cert = self.certify(scores[best_idx], scores[sorted_indices[i]])
            certifications.append(cert)
        return certifications
