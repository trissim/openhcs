Integration: Composite Scoring + Multi-Stage Pipeline
====================================================

Goal
----

Materialize a concrete, no‑handwaving implementation plan to integrate the existing
composite scoring infrastructure (Vinardo + AD4 H-bonds + ΔSASA + electrostatics)
into the multi‑stage pipeline. The plan enforces OpenHCS principles (ABCs, explicit
DI, frozen dataclasses, fail‑loud) and avoids defensive programming.

I reviewed these canonical docs before writing the plan and the work below strictly
follows their rules:

- `docs/source/development/systematic_refactoring_framework.rst`
- `docs/source/development/respecting_codebase_architecture.rst`
- `docs/source/development/refactoring_principles.rst`
- `docs/source/development/architectural_refactoring_patterns.rst`

Scope & Constraints
-------------------

- Non‑breaking where possible: preserve public APIs unless a safe, documented
  migration is required. Use optional parameters and defaults to remain backward compatible.
- Fail‑loud: do not add `hasattr`/`getattr(..., default)` defensive patterns for
  guaranteed attributes. If a field is required, populate it at construction time.
- Explicit DI: inject backends & helpers via factory functions (already present).
- Memory safe: score large batches in configurable chunks (default chunk_size=256).

High-level plan (ordered)
-------------------------

1. Parser: return per‑atom element information (backward compatible).
2. Core: extend `LigandContext` to carry `elements` and `charges` (defaults to preserve callers).
3. Pipeline: assign receptor/ligand charges via `create_charge_assigner()` and populate `ReceptorData` properly.
4. Composite scorer: implement `score_composite_batch(...)` with chunking that combines steric (JAX/Vinardo) + HBond + ΔSASA + electrostatics per chunk.
5. Scoring stages: wire composite batch scorer into `FullAccuracyStageCalculator.score` and make Stage2 use a cheap soft‑LJ backend.
6. Tests & validation: unit tests for parser, composite scorer shapes, pipeline multistage run; integration benchmark tuning.
7. Documentation & PR: checklist, performance guidance, and validation metrics to include in PR.

Concrete file edits (what to change and exact snippets)
----------------------------------------------------

Step 1 — Parser: `dq_dock_engine/docking/pdb_io.py`

Change: Add per‑atom `elements` capture and an opt‑in `return_elements` flag to keep backward compatibility.

Suggested signature change and return behavior:

```python
def parse_structure(pdb_path: Path, *, strip_hydrogens: bool = True, return_elements: bool = False):
    """Parse a PDB into (coords, radii) or (coords, radii, elements).

    If return_elements is True the function returns a 3‑tuple where the third
    element is a tuple of element strings (len == len(coords)).
    """
    coords = []
    radii = []
    elements = []

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            element = line[76:78].strip() if len(line) > 77 else ""
            if not element:
                atom_name = line[12:16].strip()
                element = atom_name.lstrip("0123456789")[:1]

            if strip_hydrogens and element.upper() == "H":
                continue

            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            coords.append([x, y, z])
            radii.append(get_vdw_radius(element))
            elements.append(element)

    coords_arr = np.array(coords, dtype=np.float64)
    radii_arr = np.array(radii, dtype=np.float64)

    if return_elements:
        return coords_arr, radii_arr, tuple(elements)
    return coords_arr, radii_arr
```

Call site changes (explicit, minimal): update the benchmark and any test that
needs elements to call with `return_elements=True`.

Example (benchmark): `dq_dock_engine/benchmark/benchmark_pdb.py`

Replace:

```python
ligand_coords, ligand_radii = parse_structure(ligand_pdb)
prot_coords, prot_radii = parse_structure(receptor_pdb)
```

With:

```python
ligand_coords, ligand_radii, ligand_elements = parse_structure(ligand_pdb, return_elements=True)
prot_coords, prot_radii, prot_elements = parse_structure(receptor_pdb, return_elements=True)
```

Write a unit test: `tests/docking/test_pdb_io.py::test_parse_structure_returns_elements`:

```python
def test_parse_structure_returns_elements(tmp_path):
    p = tmp_path / "toy.pdb"
    p.write_text("""
ATOM      1  N   ALA A   1      -0.000  -0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
""")
    coords, radii, elements = parse_structure(p, return_elements=True)
    assert len(coords) == len(radii) == len(elements)
    assert elements[0] == 'N' and elements[1] == 'C'
```

Rationale: preserves existing 2‑tuple result for callers that do not need
elements, while enabling element propagation for scoring/charges.

Step 2 — Core: `dq_dock_engine/docking/core.py`

Change: extend `LigandContext` to carry `elements` and `charges` (defaults keep callers working):

```python
@dataclass(frozen=True)
class LigandContext:
    base_coords: jnp.ndarray
    base_radii: jnp.ndarray
    center_of_mass: jnp.ndarray
    elements: tuple[str, ...] = ()          # NEW, default empty tuple
    charges: jnp.ndarray | None = None      # NEW, default None (assigned later)
```

Change `build_ligand_context` in `dq_dock_engine/docking/pdb_io.py` to accept an
optional `elements` argument and populate the `LigandContext.elements` field.

```python
def build_ligand_context(ligand_coords, ligand_radii, elements: Optional[Sequence[str]] = None) -> LigandContext:
    coords_jnp = jnp.array(ligand_coords)
    com = jnp.mean(coords_jnp, axis=0)
    base_coords = coords_jnp - com
    elements_tuple = tuple(elements) if elements is not None else ()
    return LigandContext(base_coords=base_coords, base_radii=jnp.array(ligand_radii), center_of_mass=com, elements=elements_tuple, charges=None)
```

Rationale: dataclass additions use defaults, preserving all existing call sites.

Step 3 — Pipeline: `dq_dock_engine/docking/pipeline.py`

Change: assign charges for receptor & ligand and populate `ReceptorData` with elements & charges.

Concrete patch (replace placeholder zeros/('C',...)) — inside `run_docking_pipeline` when `use_multi_stage` is True:

```python
from dq_dock_engine.docking.charges import create_charge_assigner, ChargeMethod

# Assign charges (explicit DI). Default to SIMPLE method for robustness.
assigner = create_charge_assigner(ChargeMethod.SIMPLE)

# receptor_elements must be available via parse_structure(return_elements=True)
receptor_charges = assigner.assign(tuple(receptor_elements)).charges

receptor_data = ReceptorData(
    coords=protein_coords,
    radii=receptor_radii,
    charges=receptor_charges,
    elements=tuple(receptor_elements),
)

# For ligand: assign and pass charges to subsequent stages
ligand_charge_result = assigner.assign(tuple(ligand_ctx.elements))
ligand_charges = ligand_charge_result.charges
# store on ligand_ctx by rebuilding with charges if needed, or pass separately
```

Non‑breaking detail: callers that do not use `use_multi_stage` are unaffected.

Step 4 — Composite scorer: `dq_dock_engine/docking/scoring_composite.py`

Add a batcher function `score_composite_batch(...)` that scores in memory‑safe chunks.

Signature (concrete):

```python
def score_composite_batch(
    receptor_coords: jnp.ndarray,
    poses_coords: jnp.ndarray,
    receptor_radii: jnp.ndarray,
    ligand_radii: jnp.ndarray,
    receptor_charges: jnp.ndarray,
    ligand_charges: jnp.ndarray,
    receptor_elements: tuple[str, ...],
    ligand_elements: tuple[str, ...],
    config: CompositeScoringConfig | None = None,
    steric_backend=None,      # injected via create_scoring_backend(ScoringFamily.VINARDO)
    chunk_size: int | None = None,  # default from config
) -> jnp.ndarray:
    """Score poses in chunks and return (N_poses,) energies as jnp.ndarray.

    Implementation notes:
    - For each chunk: run steric JAX backend (Vinardo) in batch (JAX native).
    - Compute HBond / ΔSASA / electrostatics per pose in Python (module functions exist).
    - Combine weighted components with config weights.
    - Convert results to a single jnp.ndarray before returning.
    """
```

Concrete chunking pseudocode:

```python
cfg = config if config is not None else CompositeScoringConfig()
cfg.validate()
chunk = cfg.chunk_size if chunk_size is None else chunk_size
if steric_backend is None:
    steric_backend = create_scoring_backend(ScoringFamily.VINARDO, cfg.steric)

N = int(poses_coords.shape[0])
out_scores = []
for start in range(0, N, chunk):
    end = min(N, start + chunk)
    poses_chunk = poses_coords[start:end]

    # 1) Steric: JAX backend (returns jnp) — convert to numpy for the Python components
    steric_jnp = steric_backend.score_batch(receptor_coords, poses_chunk, receptor_radii, ligand_radii)
    steric_np = np.array(steric_jnp)

    # 2) HBond: use create_hbond_calculator(...) and total_hbond_energy per pose
    #    (these functions are Python/JAX hybrids; compute on host per pose)
    hbond_scores = np.zeros(len(poses_chunk))
    for i_pose in range(len(poses_chunk)):
        pose = poses_chunk[i_pose]
        # build donors/acceptors for pose and receptor (use existing helper functions)
        hbond_scores[i_pose] = compute_hbond_for_pose(pose, receptor_coords, receptor_elements, ligand_elements, cfg)

    # 3) ΔSASA: call create_sasa_config(...) and compute desolvation per pose (host)
    desolvation_scores = compute_desolvation_chunk(poses_chunk, ...)

    # 4) Electrostatics: create_electrostatics_calculator(...).compute(...) per pose or vectorized in numpy
    elec_scores = compute_electrostatics_chunk(poses_chunk, receptor_coords, ligand_charges, receptor_charges, cfg)

    combined = cfg.w_steric * steric_np + cfg.w_hbond * hbond_scores + cfg.w_desolv * desolvation_scores + cfg.w_electrostatic * elec_scores
    out_scores.append(combined)

return jnp.array(np.concatenate(out_scores, axis=0))
```

Concrete helper targets to implement (small functions):

- `compute_hbond_for_pose(pose_coords, receptor_coords, receptor_elements, ligand_elements, cfg)`
- `compute_desolvation_chunk(...)` (wrap existing `sasa.compute_sasa_batch` and `delta_sasa_energy`)
- `compute_electrostatics_chunk(...)` (wrap existing electrostatics calculator)

Notes:

- Keep HBond/SASA/electrostatics as host computations for now — they are lower volume than the steric, and this avoids complex jax tracing problems.
- The steric heavy work remains in JAX (Vinardo) to keep GPU acceleration where it matters.

Step 5 — Wire into stages: `dq_dock_engine/docking/scoring_stages.py`

Change: `FullAccuracyStageCalculator.score` should call `score_composite_batch(...)` and return its output.

Minimal example replacement inside `FullAccuracyStageCalculator.score`:

```python
from dq_dock_engine.docking.scoring_composite import score_composite_batch, CompositeScoringConfig

def score(self, receptor_data: ReceptorData, poses_coords: jnp.ndarray) -> jnp.ndarray:
    cfg = CompositeScoringConfig()
    return score_composite_batch(
        receptor_data.coords,
        poses_coords,
        receptor_data.radii,
        ligand_radii,  # supply from caller (pipeline)
        receptor_data.charges,
        ligand_charges,
        receptor_data.elements,
        ligand_elements,
        cfg,
        steric_backend=None,
    )
```

Make sure that `create_stage_calculator` or the `create_pipeline` factory is adjusted so that the `FullAccuracyStageCalculator` receives `ligand_radii`, `ligand_charges` and `ligand_elements` (explicit DI). The pipeline constructs `ReceptorData` and has `ligand_ctx` — the pipeline should pass these explicit dependencies to `pipeline.run()` or inject to the stage objects at creation time.

Step 6 — Stage2 uses cheap backend

Change: In `MediumFidelityStageCalculator.score` call a cheap batch scorer (soft‑LJ) instead of placeholder zeros. For example:

```python
from dq_dock_engine.docking.scoring_vinardo import create_scoring_backend, ScoringFamily
backend = create_scoring_backend(ScoringFamily.SOFT_LJ)
scores = backend.score_batch(receptor_data.coords, poses_coords, receptor_data.radii, ligand_radii)
return scores
```

This keeps Stage2 fast and reduces the number of candidates for Stage3.

Step 7 — Tests & Validation
---------------------------

Add the following tests (concrete files & assertions):

- `tests/docking/test_pdb_io.py::test_parse_structure_returns_elements` (see above)
- `tests/docking/test_core.py::test_ligand_context_has_fields`:

```python
def test_ligand_context_has_fields():
    ctx = build_ligand_context(np.zeros((1,3)), np.array([1.0]), elements=("C",))
    assert hasattr(ctx, 'elements') and ctx.elements == ("C",)
    assert ctx.charges is None
```

- `tests/docking/test_scoring_composite.py::test_score_composite_batch_shape`:

```python
def test_score_composite_batch_shape():
    # small synthetic receptor/ligand
    rec = jnp.array([[0.,0.,0.]])
    rec_r = jnp.array([1.8])
    lig_coords = jnp.array([[[1.5, 0., 0.]]])  # one pose, one atom
    lig_r = jnp.array([1.0])
    rec_ch = jnp.array([0.0])
    lig_ch = jnp.array([0.0])
    scores = score_composite_batch(rec, lig_coords, rec_r, lig_r, rec_ch, lig_ch, ("C",), ("C",), chunk_size=1)
    assert scores.shape == (1,)
```

- `tests/docking/test_pipeline_multistage.py::test_run_docking_pipeline_multi_stage_smoke`:

Run the pipeline on a tiny synthetic system with `use_multi_stage=True` and assert it returns a list of `ScoredPose` objects and that `ValidationMetrics` (if produced) is well‑formed (no exceptions).

Test runner:

```
pytest tests/docking -q
```

Step 8 — Benchmark tuning & run
-------------------------------

Local smoke tests:

```
export DQ_DOCK_CHUNK_SIZE=128  # optional environment override
python dq_dock_engine/benchmark/benchmark_pdb.py  # run full benchmark
```

For development iterate with fewer complexes and `n_poses` reduced to 2k–5k in
`run_dq_dock` while tuning `CompositeScoringConfig.chunk_size` to avoid OOM.

PR / Code Review checklist (concrete)
------------------------------------

- [ ] All tests in `tests/docking/` pass.
- [ ] No new `getattr(..., default)` / `hasattr` defensive patterns introduced.
- [ ] All new constructors use explicit dependency injection (factories), and required fields are set at construction time.
- [ ] Composite scoring chunk_size is configurable from `CompositeScoringConfig` and default is 256.
- [ ] Documentation updated: add short usage example and how to tune `chunk_size` in `docs/`.

Rollout plan
------------

1. Landing PR that adds parser + core dataclass changes + small tests.
2. PR that implements `score_composite_batch` + unit tests.
3. PR that wires `score_composite_batch` into `scoring_stages` and updates the pipeline (small, reviewed changes). Run the benchmark on a dev GPU node and tune `chunk_size`.
4. Final PR: extend benchmark to enable `use_multi_stage=True` by default for the experimental run and publish metrics.

Risk matrix and mitigations (concrete)
-------------------------------------

- OOM when scoring big batches: mitigated by chunked scoring with default 256; increase/decrease based on hardware.
- Slow host HBond/SASA code: keep steric (heavy) on GPU; HBond/SASA are per‑pose host costs but limited by chunk size. If slow, vectorize or port to JAX in a follow‑up.
- RDKit / Vina optional dependencies: keep calls behind try/except and explicit errors only when those backends are requested (no silent fallback).

Next actions I will take (if you want me to proceed)
--------------------------------------------------

- Implement Step 1 and Step 2 (parser + `LigandContext`) and add the small tests listed above. This is low risk and allows subsequent steps to be implemented safely.

If you confirm, I will start applying the code edits in the repo and run the unit tests. If you prefer a different ordering, tell me which step to prioritize.
