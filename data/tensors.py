import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class MolecularTensor:
    """
    Core tensor representation for molecular systems.
    Shape: (N_atoms, 3) for positions.
    Minimal fields — no duplication of charge/type data held by potentials.
    """
    positions: jnp.ndarray   # (N, 3)
    masses: jnp.ndarray      # (N,)

    @property
    def n_atoms(self) -> int:
        return self.positions.shape[0]

@dataclass(frozen=True)
class InteractionTensor:
    """
    Tensor representing pairwise interactions.
    Shape: (n_types, n_a, n_b)
    """
    tensor: jnp.ndarray

    @property
    def n_interaction_types(self) -> int:
        return self.tensor.shape[0]
