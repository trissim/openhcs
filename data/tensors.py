import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class MolecularTensor:
    """
    5D tensor representation for OpenHCS-compatible docking.
    Physical shape: (batch, ligand, protein, conformation, feature)
    OpenHCS 3D mapping: (C, Y, X)
    - C = Feature (Interaction type)
    - Y = Conformation/Pose
    - X = Atom/Residue coordinate
    """
    data: jnp.ndarray
    atom_types: jnp.ndarray
    charges: jnp.ndarray
    
    @property
    def shape(self):
        return self.data.shape

    def to_openhcs_format(self) -> jnp.ndarray:
        """
        Flatten/Reshape to OpenHCS (C, Y, X) format.
        Assuming incoming data is (N_atoms, 3) for now.
        """
        # C = 1 (feature is just position), Y = 1 (single conformation), X = N_atoms * 3
        return self.data.reshape(1, 1, -1)

@dataclass(frozen=True)
class InteractionTensor:
    """
    Tensor representing interaction between ligand and protein.
    Shape: (n_interactions, n_ligand_atoms, n_protein_residues)
    """
    tensor: jnp.ndarray
    interaction_types: Optional[jnp.ndarray] = None
    
    @property
    def n_interactions(self):
        return self.tensor.shape[0]
