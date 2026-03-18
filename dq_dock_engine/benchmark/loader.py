import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path

"""
PDBbind Refined Set Loader.
Parses PDBbind data for benchmarking DQ-Dock against AutoDock Vina.

Expected PDBbind directory structure:
  pdbbind_dir/
    index/INDEX_refined_data.2020
    refined-set/
      XXXX/  (PDB code)
        XXXX_protein.pdb
        XXXX_ligand.mol2
        XXXX_pocket.pdb
"""

@dataclass(frozen=True)
class Complex:
    """A protein-ligand complex from PDBbind."""
    pdb_code: str
    protein_path: str
    ligand_path: str
    pocket_path: str
    binding_affinity: float  # -log(Kd/Ki) in pK units
    resolution: float        # Å
    year: int

@dataclass
class PDBbindDataset:
    """PDBbind refined set."""
    complexes: List[Complex] = field(default_factory=list)
    root_dir: str = ""
    
    @property
    def size(self) -> int:
        return len(self.complexes)
    
    def filter_by_resolution(self, max_resolution: float) -> 'PDBbindDataset':
        """Filter complexes by resolution cutoff."""
        filtered = [c for c in self.complexes if c.resolution <= max_resolution]
        return PDBbindDataset(complexes=filtered, root_dir=self.root_dir)
    
    def filter_by_affinity(self, min_pk: float, max_pk: float) -> 'PDBbindDataset':
        """Filter by binding affinity range."""
        filtered = [c for c in self.complexes 
                    if min_pk <= c.binding_affinity <= max_pk]
        return PDBbindDataset(complexes=filtered, root_dir=self.root_dir)
    
    def split(self, train_frac: float = 0.8) -> Tuple['PDBbindDataset', 'PDBbindDataset']:
        """Deterministic train/test split."""
        n_train = int(len(self.complexes) * train_frac)
        return (
            PDBbindDataset(self.complexes[:n_train], self.root_dir),
            PDBbindDataset(self.complexes[n_train:], self.root_dir)
        )

def load_pdbbind_index(index_path: str) -> List[dict]:
    """Parse PDBbind INDEX file."""
    entries = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                entries.append({
                    'pdb_code': parts[0],
                    'resolution': float(parts[1]) if parts[1] != 'NMR' else 99.0,
                    'year': int(parts[2]) if parts[2].isdigit() else 0,
                    'binding_affinity': _parse_affinity(parts[3]),
                })
    return entries

def _parse_affinity(s: str) -> float:
    """Parse binding affinity string like 'Kd=1.5uM' → pK value."""
    try:
        # Format: Kd=X.XXuM or Ki=X.XXnM etc.
        for prefix in ['Kd=', 'Ki=', 'IC50=']:
            if s.startswith(prefix):
                val_str = s[len(prefix):]
                # Extract numeric part and unit
                for unit, factor in [('fM', 1e-15), ('pM', 1e-12), ('nM', 1e-9), 
                                     ('uM', 1e-6), ('mM', 1e-3), ('M', 1.0)]:
                    if val_str.endswith(unit):
                        val = float(val_str[:-len(unit)]) * factor
                        import math
                        return -math.log10(val) if val > 0 else 0.0
        return 0.0
    except (ValueError, TypeError):
        return 0.0

def load_pdbbind(root_dir: str) -> PDBbindDataset:
    """
    Load PDBbind refined set from disk.
    Falls back to empty dataset if path doesn't exist.
    """
    root = Path(root_dir)
    index_path = root / "index" / "INDEX_refined_data.2020"
    
    if not index_path.exists():
        # Try alternate names
        for alt in ["INDEX_refined_data.2019", "INDEX_refined_data.2018"]:
            alt_path = root / "index" / alt
            if alt_path.exists():
                index_path = alt_path
                break
        else:
            return PDBbindDataset(root_dir=root_dir)
    
    entries = load_pdbbind_index(str(index_path))
    refined_dir = root / "refined-set"
    
    complexes = []
    for entry in entries:
        code = entry['pdb_code']
        complex_dir = refined_dir / code
        protein = complex_dir / f"{code}_protein.pdb"
        ligand = complex_dir / f"{code}_ligand.mol2"
        pocket = complex_dir / f"{code}_pocket.pdb"
        
        if complex_dir.exists():
            complexes.append(Complex(
                pdb_code=code,
                protein_path=str(protein),
                ligand_path=str(ligand),
                pocket_path=str(pocket),
                binding_affinity=entry['binding_affinity'],
                resolution=entry['resolution'],
                year=entry['year'],
            ))
    
    return PDBbindDataset(complexes=complexes, root_dir=root_dir)
