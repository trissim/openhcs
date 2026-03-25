from __future__ import annotations

from pathlib import Path


def load_rdkit_molecule(
    path: str | Path,
    *,
    remove_hs: bool = False,
    sanitize: bool = True,
):
    """Load a small-molecule file into an RDKit Mol.

    Supports the ligand chemistry source formats used by the benchmark and
    docking pipeline. Fails loudly when the format is unsupported or the file
    cannot be parsed.
    """

    try:
        from rdkit import Chem
    except ImportError as exc:
        raise ImportError("RDKit is required to load ligand chemistry sources") from exc

    source_path = Path(path)
    suffix = source_path.suffix.lower()

    mol = None
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(
            str(source_path), removeHs=remove_hs, sanitize=sanitize
        )
        for candidate in supplier:
            if candidate is not None:
                mol = candidate
                break
    elif suffix in {".mol", ".mdl"}:
        mol = Chem.MolFromMolFile(
            str(source_path), removeHs=remove_hs, sanitize=sanitize
        )
    elif suffix in {".pdb", ".ent"}:
        mol = Chem.MolFromPDBFile(
            str(source_path), removeHs=remove_hs, sanitize=sanitize
        )
        if mol is None:
            block = source_path.read_text()
            mol = Chem.MolFromPDBBlock(block, removeHs=remove_hs, sanitize=sanitize)
    else:
        raise ValueError(
            "Unsupported ligand chemistry source format: "
            f"{source_path.suffix or '<no suffix>'}"
        )

    if mol is None:
        raise ValueError(
            f"RDKit failed to parse ligand chemistry source: {source_path}"
        )
    return mol
