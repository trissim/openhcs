"""
Charge assignment methods.

Reference:
    - Jakalian & Bayly (2002) J. Comput. Chem. 23:1623-1641 (AM1-BCC)
    - Gasteiger & Marsili (1980) Tetrahedron 36(22):3219-3228 (Gasteiger)

OpenHCS Compliance:
- Enum-driven method selection
- Factory function with explicit dependencies
- Frozen dataclass for results
- Fail-loud validation
"""

from __future__ import annotations

import math
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

from dq_dock_engine.docking.rdkit_io import load_rdkit_molecule


class ChargeMethod(Enum):
    """Supported charge assignment methods."""

    AM1BCC = auto()
    GASTEIGER = auto()
    SIMPLE = auto()


@dataclass(frozen=True)
class ChargeResult:
    """Immutable result of charge assignment."""

    charges: jnp.ndarray
    method: ChargeMethod


_SIMPLE_CHARGE_RULES: dict[str, float] = {
    "C": 0.0,
    "N": -0.3,
    "O": -0.4,
    "S": 0.0,
    "H": 0.1,
    "P": 0.5,
    "NA": 1.0,
    "K": 1.0,
    "MG": 2.0,
    "CA": 2.0,
    "MN": 2.0,
    "FE": 2.0,
    "CO": 2.0,
    "NI": 2.0,
    "CU": 2.0,
    "ZN": 2.0,
    "CD": 2.0,
    "CL": -1.0,
    "BR": -1.0,
    "I": -1.0,
}


class ChargeAssigner(ABC):
    """ABC contract for charge assigners.

    assign() must accept a source which is one of:
      - tuple[str, ...]: per-atom element symbols (only supported by SimpleChargeAssigner)
      - pathlib.Path or str: path to a molecule file (PDB, MOL2, ...) understood by the assigner
      - RDKit Mol instance (rdkit.Chem.Mol) when supported

    Implementations MUST fail loudly (raise) if the provided source is unsupported.
    """

    @abstractmethod
    def assign(self, source: Any) -> ChargeResult:
        """Assign charges to atoms from provided source."""

    @property
    @abstractmethod
    def method(self) -> ChargeMethod:
        """Direct access to method (no defensive getattr)."""


class SimpleChargeAssigner(ChargeAssigner):
    """Simple element-based charge assigner."""

    @property
    def method(self) -> ChargeMethod:
        return ChargeMethod.SIMPLE

    def assign(self, source: Any) -> ChargeResult:
        """Assign charges using simple element rules.

        Expects `source` to be a tuple/list of element symbols.
        """
        if not isinstance(source, (tuple, list)):
            raise ValueError(
                "SimpleChargeAssigner requires a tuple/list of element symbols as source"
            )

        charges = []
        for elem in source:
            upper = elem.upper()
            if upper not in _SIMPLE_CHARGE_RULES:
                raise ValueError(f"Unknown element for simple charges: {elem}")
            charges.append(_SIMPLE_CHARGE_RULES[upper])
        return ChargeResult(charges=jnp.array(charges), method=ChargeMethod.SIMPLE)


class GasteigerChargeAssigner(ChargeAssigner):
    """Gasteiger charge assigner using RDKit."""

    @property
    def method(self) -> ChargeMethod:
        return ChargeMethod.GASTEIGER

    def assign(self, source: Any) -> ChargeResult:
        """Assign Gasteiger charges.

        Expects `elements` to be one of:
          - RDKit Mol instance
          - pathlib.Path / str pointing to a PDB (or other RDKit-readable) file

        This method requires RDKit to be installed and will raise ImportError otherwise.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdPartialCharges
        except (
            Exception
        ) as e:  # ImportError often but catch broad to give helpful message
            raise ImportError(
                "RDKit is required for Gasteiger charges. Install RDKit and retry"
            ) from e

        # Accept RDKit Mol directly
        mol = None
        if isinstance(source, Chem.Mol):
            mol = source
        else:
            # Expect a file path (PDB recommended)
            if not isinstance(source, (str, Path)):
                raise ValueError(
                    "GasteigerChargeAssigner expects an RDKit Mol or a file path (PDB) as source"
                )
            p = Path(source)
            if not p.exists():
                raise FileNotFoundError(f"Molecule file not found: {p}")
            mol = load_rdkit_molecule(p, remove_hs=False, sanitize=True)

        # Compute Gasteiger charges
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
        except Exception as e:
            raise RuntimeError("RDKit failed to compute Gasteiger charges") from e

        charges = []
        for atom_index, atom in enumerate(mol.GetAtoms()):
            # RDKit stores Gasteiger charge as property '_GasteigerCharge'
            try:
                q = atom.GetDoubleProp("_GasteigerCharge")
            except Exception:
                # Fallback to text prop conversion
                try:
                    q = float(atom.GetProp("_GasteigerCharge"))
                except Exception:
                    raise RuntimeError(
                        "RDKit did not produce per-atom Gasteiger charges"
                    )
            if not math.isfinite(q):
                raise RuntimeError(
                    f"RDKit produced non-finite Gasteiger charge at atom index {atom_index}"
                )
            charges.append(q)

        return ChargeResult(charges=jnp.array(charges), method=ChargeMethod.GASTEIGER)


class AM1BCCChargeAssigner(ChargeAssigner):
    """AM1-BCC charge assigner using RDKit + AmberTools."""

    @property
    def method(self) -> ChargeMethod:
        return ChargeMethod.AM1BCC

    def assign(self, source: Any) -> ChargeResult:
        """Assign AM1-BCC charges.

        This implementation shells out to AmberTools' `antechamber` utility.
        It expects the `antechamber` executable to be available in PATH.

        Accepts RDKit Mol or a file path (PDB recommended). Raises informative
        errors if required tools are not present.
        """
        try:
            from rdkit import Chem
        except Exception as e:
            raise ImportError("RDKit is required for AM1-BCC charge assignment") from e

        # Determine input file path
        input_path = None
        if isinstance(source, Chem.Mol):
            mol = source
            # Write PDB block to temp file
            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
                tmp.write(Chem.MolToPDBBlock(mol).encode("utf-8"))
                input_path = Path(tmp.name)
        else:
            if not isinstance(source, (str, Path)):
                raise ValueError(
                    "AM1BCCChargeAssigner expects an RDKit Mol or a file path (PDB) as source"
                )
            input_path = Path(source)
            if not input_path.exists():
                raise FileNotFoundError(f"Molecule file not found: {input_path}")

        # Locate antechamber
        antechamber_path = shutil.which("antechamber")
        if antechamber_path is None:
            raise RuntimeError(
                "AmberTools `antechamber` executable not found in PATH. Install AmberTools to use AM1-BCC"
            )

        # Run antechamber to compute AM1-BCC charges and write mol2 output
        with tempfile.TemporaryDirectory() as tmpdir:
            out_mol2 = Path(tmpdir) / "out.mol2"
            cmd = [
                antechamber_path,
                "-i",
                str(input_path),
                "-fi",
                "pdb",
                "-o",
                str(out_mol2),
                "-fo",
                "mol2",
                "-c",
                "bcc",
                "-s",
                "2",
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    raise RuntimeError(f"antechamber failed: {result.stderr[:200]}")
            except FileNotFoundError:
                raise RuntimeError(
                    "antechamber not found - ensure AmberTools is installed and 'antechamber' is in PATH"
                )

            # Parse mol2 for partial charges (ATOM section)
            if not out_mol2.exists():
                raise RuntimeError("antechamber did not produce output mol2 file")

            charges = []
            in_atom_section = False
            for line in out_mol2.read_text().splitlines():
                if line.strip().startswith("@<TRIPOS>ATOM"):
                    in_atom_section = True
                    continue
                if line.strip().startswith("@<TRIPOS>BOND"):
                    break
                if in_atom_section:
                    parts = line.split()
                    if len(parts) >= 9:
                        # charge is last column
                        try:
                            q = float(parts[8])
                        except ValueError:
                            raise RuntimeError(
                                "Failed to parse charge from antechamber mol2 output"
                            )
                        charges.append(q)

            if not charges:
                raise RuntimeError("No charges parsed from antechamber output")

            return ChargeResult(charges=jnp.array(charges), method=ChargeMethod.AM1BCC)


def create_charge_assigner(method: ChargeMethod) -> ChargeAssigner:
    """
    Factory function with explicit dependency injection.

    OpenHCS Compliance:
    - Explicit factory
    - Enum-driven dispatch
    """
    match method:
        case ChargeMethod.SIMPLE:
            return SimpleChargeAssigner()
        case ChargeMethod.GASTEIGER:
            return GasteigerChargeAssigner()
        case ChargeMethod.AM1BCC:
            return AM1BCCChargeAssigner()
        case _:
            raise ValueError(f"Unknown ChargeMethod: {method}")
