#!/usr/bin/env python3
from __future__ import annotations

"""
Real PDB-based Docking Benchmark

Downloads famous drug-target complexes and runs DQ-Dock vs SMINA comparison.
These are well-studied, difficult targets where docking actually matters.

Famous drug targets:
- HIV-1 Protease: Major AIDS drug target
- CDK2 Kinase: Cancer target
- SARS-CoV-2 Mpro: COVID-19 drug target
- Carbonic anhydrase: Anti-glaucoma, altitude sickness
- Thrombin: Blood thinner target
- BACE-1: Alzheimer's target

Usage:
    python benchmark_pdb.py --n_complexes 10

Requirements:
    smina: yay -S smina-bin
"""

import argparse
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from enum import Enum
import json
import math
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Union, cast
import urllib.request
import gzip
import shutil
from typing import Any, Callable, Generic, Iterator, TypeVar
from typing import TYPE_CHECKING
import zlib

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, replace

from dq_dock_engine.docking.core import (
    BenchmarkResult,
    BlindDockingExecutionPath,
    CertifiedBlindDockingResult,
    GapCertification,
    GeometricBlindDockingResult,
    NativeCertification,
    ScoringEngine,
    ScoredPose,
)
from dq_dock_engine.docking.pipeline import (
    CertifiedBlindDockingRequest,
    DockingRequestBase,
    GeometricBlindDockingRequest,
    PipelineDockingRequest,
    derive_request,
    run_certified_blind_docking_request,
    run_docking_pipeline_request,
    run_geometric_blind_docking_request,
)
from dq_dock_engine.docking.pdb_io import parse_structure
from dq_dock_engine.docking.receptor_preparation import (
    EssentialSiteComponentsPolicy,
    PreparedReceptorSystem,
    ReceptorChemistryPlan,
    prepare_protein_only_pocket,
    prepare_protein_only_receptor,
    ResidueKey,
    assess_receptor_compatibility,
    BenchmarkChemistryProtocol,
    prepare_receptor_system,
)
from dq_dock_engine.docking.metrics import compute_docking_rmsd_batched
from dq_dock_engine.docking_config import (
    CERTIFIED_DOCKING,
    CertifiedScoringFamily,
    DockingMode,
    OptimizerBackend,
)
from dq_dock_engine.benchmark.large_pdb_ids import (
    BenchmarkScope,
    PDBEntry,
    _extract_pdb_title,
    get_benchmark_entries,
    get_casf_2007_entries,
    get_casf_2007_ids,
    get_default_ligand_entries,
)
from dq_dock_engine.benchmark.redocking_report import render_redocking_report
from dq_dock_engine.docking.formal_handles import (
    ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT,
    ACTIVE_SINGLETON_HYBRID_RUNTIME_CONTRACT,
    CP3,
    FLO18,
    SD10,
    STAGED_COARSE_TOP1_RUNTIME_CONTRACT,
    STAGED_SINGLETON_TOP1_RUNTIME_CONTRACT,
    handle_bundle_from_contracts,
    runtime_contract_record,
    scoring_family_theorem_handles,
)
from dq_dock_engine.docking.charges import (
    ChargeAssigner,
    ChargeMethod,
    create_charge_assigner,
)
from dq_dock_engine.docking_config import FormalRoundStrategy

if TYPE_CHECKING:
    from dq_dock_engine.docking.core import (
        BlindDockingExecutionPath,
        CertifiedPocketFailureReason,
        DockingBox,
        GapCertification,
        LigandContext,
        NativeCertification,
        ScoringEngine,
    )
    from dq_dock_engine.docking_config import DockingConfig


DEFAULT_BENCHMARK_BOX_SIZE_ANGSTROMS = 12.0
DEFAULT_BENCHMARK_CHARGE_METHOD = ChargeMethod.GASTEIGER
DEFAULT_BENCHMARK_OPTIMIZER_BACKEND = OptimizerBackend.FORMAL
DEFAULT_BENCHMARK_FORMAL_ROUND_STRATEGY = FormalRoundStrategy.SINGLETON_HYBRID
DEFAULT_BENCHMARK_CERTIFIED_SCORING_FAMILY = CertifiedScoringFamily.LJ_REALSPACE_EWALD
DEFAULT_BENCHMARK_TOP_K_TO_OPTIMIZE = 20
DEFAULT_BENCHMARK_USE_POCKET_GUIDED = True
DEFAULT_BENCHMARK_USE_MULTI_STAGE = False
DEFAULT_BENCHMARK_PROCESS_START_METHOD = "spawn"


def derive_benchmark_pocket_radius_from_box_size(box_size: float) -> float:
    """Derive the pocket extraction sphere radius from the docking box size.

    The formal benchmark protocol uses a cubic search box of side ``box_size``.
    To guarantee the extracted pocket covers every atom the ligand can reach
    during the search, we extract atoms inside a sphere of radius
    ``box_size * sqrt(3) / 2`` — which is the half-diagonal of the cube.

    This is proven by Lean theorem SD10
    (``cube_side_eq_radius_half_diagonal_le_radius``), which shows that a
    cube of side r fits inside a sphere of radius r because its half-diagonal
    is ``sqrt(3) * r / 2 ≤ r``.

    Previously the code used ``pocket_radius = box_size / 2`` (a 6A sphere for
    a 12A box), which only covered atoms up to 6A from the center.  The ligand
    can reach the box corner at ``10.4A`` from the center — beyond the
    extraction boundary — so the truncated receptor produced wrong certified
    rankings on pockets where the true optimum lies near the box edge.

    Args:
        box_size: Cubic search box side length in Angstroms.

    Returns:
        Pocket extraction sphere radius in Angstroms.  For the default
        12A box this gives ``12 * sqrt(3) / 2 ≈ 10.39A``.
    """
    if box_size <= 0:
        raise ValueError("box_size must be positive")
    return box_size * math.sqrt(3.0) / 2.0


def derive_benchmark_box_size_from_pocket_radius(pocket_radius: float) -> float:
    """Deprecated: prefer box_size as the independent parameter.

    Reverse-derives the box side from a pocket radius.
    Kept for CLI / benchmark compatibility only.
    """
    if pocket_radius <= 0:
        raise ValueError("pocket_radius must be positive")
    return pocket_radius


def benchmark_box_protocol_metadata(box_size: float) -> tuple[str, str | None]:
    if np.isclose(box_size, BENCHMARK_PROTOCOL.box_size_a):
        return ("box_size_derived", BENCHMARK_PROTOCOL.box_geometry_theorem)
    return ("custom_override", None)


def _ligand_receptor_runtime_args(
    *,
    receptor_elements: list[str] | tuple[str, ...] | None,
    precomputed_receptor_charges: np.ndarray | None,
) -> tuple[tuple[str, ...] | None, jnp.ndarray | None]:
    return (
        tuple(receptor_elements) if receptor_elements is not None else None,
        None
        if precomputed_receptor_charges is None
        else jnp.array(precomputed_receptor_charges),
    )


def active_formal_runtime_contract(formal_round_strategy: FormalRoundStrategy):
    if formal_round_strategy == FormalRoundStrategy.EXACT:
        return ACTIVE_EXACT_CERTIFIED_RUNTIME_CONTRACT
    return ACTIVE_SINGLETON_HYBRID_RUNTIME_CONTRACT


DEFAULT_BENCHMARK_BOX_SIZE_ANGSTROMS = 12.0

DERIVED_BENCHMARK_POCKET_RADIUS_ANGSTROMS = (
    derive_benchmark_pocket_radius_from_box_size(DEFAULT_BENCHMARK_BOX_SIZE_ANGSTROMS)
)


@dataclass(frozen=True)
class BenchmarkProtocol:
    box_size_a: float
    pocket_radius_a: float
    box_geometry_theorem: str


BENCHMARK_PROTOCOL = BenchmarkProtocol(
    box_size_a=DEFAULT_BENCHMARK_BOX_SIZE_ANGSTROMS,
    pocket_radius_a=DERIVED_BENCHMARK_POCKET_RADIUS_ANGSTROMS,
    box_geometry_theorem=SD10,
)


class BenchmarkTargetSelectionMode(Enum):
    DATASET = "dataset"
    EXPLICIT_PDB_IDS = "explicit_pdb_ids"


@dataclass(frozen=True)
class BenchmarkTargetSelection:
    mode: BenchmarkTargetSelectionMode
    label: str
    cache_dir: Path
    n_complexes_requested: int
    entries: tuple[PDBEntry, ...]


@dataclass(frozen=True)
class BlindDockingExecutionOutcome:
    poses: list
    certification: "NativeCertification | GapCertification | None"
    execution_path: "BlindDockingExecutionPath"
    certified_failure_reason: "CertifiedPocketFailureReason | None"
    theorem_handles: tuple[str, ...]


@dataclass(frozen=True)
class BlindDockingAttemptContext(DockingRequestBase):
    engine: ScoringEngine
    use_multi_stage: bool


RequestT = TypeVar("RequestT", bound=DockingRequestBase)
ExecutionT = TypeVar("ExecutionT")


class BlindDockingExecutor(ABC):
    execution_path: BlindDockingExecutionPath

    @abstractmethod
    def execute(
        self, context: BlindDockingAttemptContext
    ) -> BlindDockingExecutionOutcome:
        """Execute one blind-docking attempt."""


class RequestBlindDockingExecutor(
    BlindDockingExecutor,
    Generic[RequestT, ExecutionT],
    ABC,
):
    request_type: type[RequestT]

    def execute(
        self, context: BlindDockingAttemptContext
    ) -> BlindDockingExecutionOutcome:
        request = derive_request(
            self.request_type,
            context,
            **self.derived_request_kwargs(context),
        )
        return self.build_outcome(self.run_request(request))

    def derived_request_kwargs(
        self, context: BlindDockingAttemptContext
    ) -> dict[str, object]:
        return {}

    @abstractmethod
    def run_request(self, request: RequestT) -> ExecutionT:
        """Execute the concrete pipeline request."""

    @abstractmethod
    def build_outcome(self, execution: ExecutionT) -> BlindDockingExecutionOutcome:
        """Lift the concrete execution result into the benchmark outcome."""


class StrictCertifiedBlindDockingExecutor(
    RequestBlindDockingExecutor[
        CertifiedBlindDockingRequest,
        CertifiedBlindDockingResult,
    ]
):
    execution_path = BlindDockingExecutionPath.STRICT_CERTIFIED
    request_type = CertifiedBlindDockingRequest

    def run_request(
        self, request: CertifiedBlindDockingRequest
    ) -> CertifiedBlindDockingResult:
        blind_result = run_certified_blind_docking_request(request)
        return blind_result

    def build_outcome(
        self, execution: CertifiedBlindDockingResult
    ) -> BlindDockingExecutionOutcome:
        blind_result = execution
        return BlindDockingExecutionOutcome(
            poses=list(blind_result.poses),
            certification=blind_result.certification,
            execution_path=BlindDockingExecutionPath.STRICT_CERTIFIED,
            certified_failure_reason=blind_result.plan.certified_failure_reason,
            theorem_handles=blind_result.plan.theorem_handles,
        )


class GeometricBlindDockingExecutor(
    RequestBlindDockingExecutor[
        GeometricBlindDockingRequest,
        GeometricBlindDockingResult,
    ]
):
    execution_path = BlindDockingExecutionPath.GEOMETRIC_FALLBACK
    request_type = GeometricBlindDockingRequest

    def run_request(
        self, request: GeometricBlindDockingRequest
    ) -> GeometricBlindDockingResult:
        blind_result = run_geometric_blind_docking_request(request)
        return blind_result

    def build_outcome(
        self, execution: GeometricBlindDockingResult
    ) -> BlindDockingExecutionOutcome:
        blind_result = execution
        return BlindDockingExecutionOutcome(
            poses=list(blind_result.poses),
            certification=None,
            execution_path=BlindDockingExecutionPath.GEOMETRIC_FALLBACK,
            certified_failure_reason=None,
            theorem_handles=(),
        )


class GenericPipelineBlindDockingExecutor(
    RequestBlindDockingExecutor[
        PipelineDockingRequest,
        tuple[List[ScoredPose], Union[NativeCertification, GapCertification, None]],
    ]
):
    execution_path = BlindDockingExecutionPath.GENERIC_PIPELINE
    request_type = PipelineDockingRequest

    def derived_request_kwargs(
        self, context: BlindDockingAttemptContext
    ) -> dict[str, object]:
        return {
            "use_pocket_guided": False,
            "use_multi_stage": context.use_multi_stage,
        }

    def run_request(
        self, request: PipelineDockingRequest
    ) -> tuple[List[ScoredPose], Union[NativeCertification, GapCertification, None]]:
        return run_docking_pipeline_request(request)

    def build_outcome(
        self,
        execution: tuple[
            List[ScoredPose], Union[NativeCertification, GapCertification, None]
        ],
    ) -> BlindDockingExecutionOutcome:
        poses, certification = execution
        return BlindDockingExecutionOutcome(
            poses=list(poses),
            certification=certification,
            execution_path=BlindDockingExecutionPath.GENERIC_PIPELINE,
            certified_failure_reason=None,
            theorem_handles=(),
        )


def derive_blind_docking_executor(
    docking_mode: "DockingMode",
    *,
    use_pocket_guided: bool,
) -> BlindDockingExecutor:
    if docking_mode == DockingMode.CERTIFIED and use_pocket_guided:
        return StrictCertifiedBlindDockingExecutor()
    if use_pocket_guided:
        return GeometricBlindDockingExecutor()
    return GenericPipelineBlindDockingExecutor()


@dataclass(frozen=True)
class ComplexSummaryRow:
    pdb_id: str
    target_name: str
    ligand_atoms: int
    pocket_atoms: int
    rmsd: float | None
    time: float
    dock_time: float | None = None
    dq_pose_pdb: str | None = None
    receptor_pdb: str | None = None
    native_ligand_pdb: str | None = None

    def to_csv_record(
        self,
        complex_entry: "PreparedBenchmarkComplex",
        result: BenchmarkResult,
    ) -> dict[str, object]:
        score = result.energy if result.success else None
        rmsd = self.rmsd if result.success else None
        gap_proof = format_gap_proof_label(result.certified) if result.success else None
        native_rank = result.native_rank if result.success else None
        energy_gap = result.energy_gap if result.success else None
        return {
            "method": "DQ-Dock",
            "engine_id": "dq_dock",
            "pdb_id": self.pdb_id,
            "target_name": self.target_name,
            "center_x": complex_entry.center[0],
            "center_y": complex_entry.center[1],
            "center_z": complex_entry.center[2],
            "box_x": complex_entry.box_size[0],
            "box_y": complex_entry.box_size[1],
            "box_z": complex_entry.box_size[2],
            "ligand_atoms": self.ligand_atoms,
            "pocket_atoms": self.pocket_atoms,
            "score_name": "energy",
            "score": score,
            "top_rmsd": rmsd,
            "best_mode_rmsd": rmsd,
            "time_s": self.time,
            "dock_time_s": _safe_json_value(result.dock_time_s),
            "checkpoint_time_s": _safe_json_value(result.checkpoint_time_s),
            "total_wall_time_s": _safe_json_value(result.total_wall_time_s),
            "pose_pdb": self.dq_pose_pdb,
            "gap_proof": gap_proof,
            "execution_path": None
            if result.execution_path is None
            else result.execution_path.name,
            "certified_failure_reason": None
            if result.certified_failure_reason is None
            else result.certified_failure_reason.name,
            "native_rank": native_rank,
            "energy_gap": energy_gap,
            "receptor_pdb": self.receptor_pdb,
            "native_ligand_pdb": self.native_ligand_pdb,
            "status": "success" if result.success else "failure",
            "error": result.error,
        }

    def to_json_record(
        self, complex_entry: "PreparedBenchmarkComplex", result: BenchmarkResult
    ) -> dict[str, object]:
        rmsd = self.rmsd if result.success else None
        energy = result.energy if result.success else None
        gap_proof = format_gap_proof_label(result.certified) if result.success else None
        native_rank = result.native_rank if result.success else None
        energy_gap = result.energy_gap if result.success else None
        return {
            "pdb_id": self.pdb_id,
            "target_name": self.target_name,
            "center_x": complex_entry.center[0],
            "center_y": complex_entry.center[1],
            "center_z": complex_entry.center[2],
            "box_x": complex_entry.box_size[0],
            "box_y": complex_entry.box_size[1],
            "box_z": complex_entry.box_size[2],
            "ligand_atoms": self.ligand_atoms,
            "pocket_atoms": self.pocket_atoms,
            "success": result.success,
            "status": "success" if result.success else "failure",
            "rmsd": _safe_json_value(rmsd),
            "time_s": _safe_json_value(self.time),
            "dock_time_s": _safe_json_value(result.dock_time_s),
            "checkpoint_time_s": _safe_json_value(result.checkpoint_time_s),
            "total_wall_time_s": _safe_json_value(result.total_wall_time_s),
            "energy": _safe_json_value(energy),
            "pose_pdb": self.dq_pose_pdb,
            "receptor_pdb": self.receptor_pdb,
            "native_ligand_pdb": self.native_ligand_pdb,
            "gap_proof": gap_proof,
            "execution_path": None
            if result.execution_path is None
            else result.execution_path.name,
            "certified_failure_reason": None
            if result.certified_failure_reason is None
            else result.certified_failure_reason.name,
            "native_rank": _safe_json_value(native_rank),
            "energy_gap": _safe_json_value(energy_gap),
            "error": result.error,
        }


@dataclass(frozen=True)
class BenchmarkComplex:
    pdb_id: str
    target_name: str
    path: Path
    center: tuple[float, float, float]
    preferred_resname: str | None = None
    ligand_residue: LigandResidue | None = None
    receptor_system: PreparedReceptorSystem | None = None
    chemistry_plan: ReceptorChemistryPlan | None = None


@dataclass(frozen=True)
class PreparedBenchmarkComplex:
    pdb_id: str
    target_name: str
    path: Path
    center: tuple[float, float, float]
    box_size: tuple[float, float, float]
    receptor_pdb: Path
    pocket_receptor_pdb: Path
    ligand_pdb: Path
    receptor_view_pdb: Path
    native_ligand_view_pdb: Path
    ligand_coords: np.ndarray
    ligand_radii: np.ndarray
    ligand_elements: tuple[str, ...]
    ligand_charges: np.ndarray | None
    pocket_coords: np.ndarray
    pocket_radii: np.ndarray
    pocket_elements: tuple[str, ...]
    pocket_charges: np.ndarray | None
    preferred_resname: str | None = None
    receptor_system: PreparedReceptorSystem | None = None
    charge_method: ChargeMethod = DEFAULT_BENCHMARK_CHARGE_METHOD
    chemistry_protocol_name: str = "standard"


@dataclass(frozen=True)
class PreparedBenchmarkElectrostatics:
    ligand_charges: np.ndarray
    pocket_charges: np.ndarray


@dataclass(frozen=True)
class BenchmarkParallelism:
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")

    @property
    def is_parallel(self) -> bool:
        return self.max_workers > 1


@dataclass(frozen=True)
class CompetitorStatistics:
    display_name: str
    score_name: str
    prep_protocol: str
    n_completed: int
    n_successful: int
    avg_top_rmsd: float | None
    avg_best_mode_rmsd: float | None
    total_time_s: float

    @classmethod
    def from_rows(
        cls,
        metadata: "CompetitorMetadata",
        rows: list["CompetitorResultRow"],
    ) -> "CompetitorStatistics":
        successful_rows = [row for row in rows if row.success]
        return cls(
            display_name=metadata.display_name,
            score_name=metadata.score_name,
            prep_protocol=metadata.prep_protocol,
            n_completed=len(rows),
            n_successful=len(successful_rows),
            avg_top_rmsd=None
            if not successful_rows
            else float(np.mean([cast(float, row.top_rmsd) for row in successful_rows])),
            avg_best_mode_rmsd=None
            if not successful_rows
            else float(
                np.mean([cast(float, row.best_mode_rmsd) for row in successful_rows])
            ),
            total_time_s=float(sum(row.time for row in successful_rows)),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "display_name": self.display_name,
            "score_name": self.score_name,
            "prep_protocol": self.prep_protocol,
            "n_completed": self.n_completed,
            "n_successful": self.n_successful,
            "avg_top_rmsd": self.avg_top_rmsd,
            "avg_best_mode_rmsd": self.avg_best_mode_rmsd,
            "total_time_s": self.total_time_s,
        }


@dataclass(frozen=True)
class DockingRunResult:
    success: bool
    time: float
    score: float | None = None
    top_coords: np.ndarray | None = None
    model_coords: tuple[np.ndarray, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class CompetitorMetadata:
    engine_id: str
    display_name: str
    score_name: str
    prep_protocol: str


@dataclass(frozen=True)
class CompetitorResultRow:
    engine_id: str
    display_name: str
    score_name: str
    pdb_id: str
    target_name: str
    success: bool
    time: float
    score: float | None = None
    top_rmsd: float | None = None
    best_mode_rmsd: float | None = None
    pose_pdb: str | None = None
    error: str | None = None

    def to_csv_record(
        self, complex_entry: "PreparedBenchmarkComplex"
    ) -> dict[str, object]:
        return {
            "method": self.display_name,
            "engine_id": self.engine_id,
            "pdb_id": self.pdb_id,
            "target_name": self.target_name,
            "center_x": complex_entry.center[0],
            "center_y": complex_entry.center[1],
            "center_z": complex_entry.center[2],
            "box_x": complex_entry.box_size[0],
            "box_y": complex_entry.box_size[1],
            "box_z": complex_entry.box_size[2],
            "ligand_atoms": None,
            "pocket_atoms": None,
            "score_name": self.score_name,
            "score": self.score,
            "top_rmsd": self.top_rmsd,
            "best_mode_rmsd": self.best_mode_rmsd,
            "time_s": self.time,
            "dock_time_s": None,
            "checkpoint_time_s": None,
            "total_wall_time_s": self.time,
            "pose_pdb": self.pose_pdb,
            "gap_proof": None,
            "native_rank": None,
            "energy_gap": None,
            "receptor_pdb": None,
            "native_ligand_pdb": None,
            "status": "success" if self.success else "failure",
            "error": self.error,
        }

    def to_json_record(
        self, complex_entry: "PreparedBenchmarkComplex"
    ) -> dict[str, object]:
        return {
            "engine_id": self.engine_id,
            "display_name": self.display_name,
            "score_name": self.score_name,
            "pdb_id": self.pdb_id,
            "target_name": self.target_name,
            "center_x": complex_entry.center[0],
            "center_y": complex_entry.center[1],
            "center_z": complex_entry.center[2],
            "box_x": complex_entry.box_size[0],
            "box_y": complex_entry.box_size[1],
            "box_z": complex_entry.box_size[2],
            "success": self.success,
            "top_rmsd": _safe_json_value(self.top_rmsd),
            "best_returned_mode_rmsd": _safe_json_value(self.best_mode_rmsd),
            "time_s": _safe_json_value(self.time),
            "dock_time_s": None,
            "checkpoint_time_s": None,
            "total_wall_time_s": _safe_json_value(self.time),
            "score": _safe_json_value(self.score),
            "pose_pdb": self.pose_pdb,
            "error": self.error,
        }


@dataclass(frozen=True)
class BenchmarkOutputPaths:
    json_path: Path
    csv_path: Path


@dataclass(frozen=True)
class PreparedDockingInputs:
    protocol_label: str
    receptor_path: Path
    ligand_path: Path
    cleanup_dir: Path | None = None


@dataclass(frozen=True)
class DQDockExecutionOptions:
    charge_method: ChargeMethod
    certified_scoring_family: CertifiedScoringFamily
    n_poses: int
    n_opt_steps: int
    top_k_to_optimize: int
    box_size: float
    formal_round_strategy: FormalRoundStrategy
    use_multi_stage: bool
    use_pocket_guided: bool
    use_rich_exact_rescoring: bool
    optimizer_backend: "OptimizerBackend"
    max_retries: int
    retry_break_rmsd: float
    retry_preserve_seed: bool


@dataclass(frozen=True)
class BenchmarkExecutionContext(DQDockExecutionOptions):
    parallelism: BenchmarkParallelism
    complexes: tuple[PreparedBenchmarkComplex, ...]
    n_complexes_requested: int
    output_paths: BenchmarkOutputPaths
    pose_dir: Path
    competitors: tuple[CompetitorMetadata, ...]
    attempt_timeout_seconds: float | None
    bench_start: float

    @property
    def charge_method_name(self) -> str:
        return self.charge_method.name.lower()

    def elapsed(self) -> float:
        return time.time() - self.bench_start


JobType = TypeVar("JobType")
ResultType = TypeVar("ResultType")


def execute_benchmark_jobs(
    jobs: Sequence[JobType],
    *,
    run_job: Callable[[JobType], ResultType],
    parallelism: BenchmarkParallelism,
) -> Iterator[ResultType]:
    if not parallelism.is_parallel:
        for job in jobs:
            yield run_job(job)
        return

    mp_context = mp.get_context(DEFAULT_BENCHMARK_PROCESS_START_METHOD)
    executor = ProcessPoolExecutor(
        max_workers=parallelism.max_workers,
        mp_context=mp_context,
    )
    try:
        futures = [executor.submit(run_job, job) for job in jobs]
        for future in as_completed(futures):
            yield future.result()
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    executor.shutdown(wait=True)


@dataclass
class BenchmarkState:
    dq_results: list[BenchmarkResult]
    dq_rows: list[ComplexSummaryRow]
    competitor_rows: list[CompetitorResultRow]

    def save(
        self,
        phase: str,
        context: "BenchmarkExecutionContext",
        excluded_rows: list["ExcludedComplexRow"],
        *,
        render_report: bool = False,
    ) -> tuple[Path, Path]:
        return save_benchmark_results(
            context.output_paths,
            charge_method_name=context.charge_method_name,
            certified_scoring_family=context.certified_scoring_family,
            n_complexes_requested=context.n_complexes_requested,
            n_poses=context.n_poses,
            n_opt_steps=context.n_opt_steps,
            top_k_to_optimize=context.top_k_to_optimize,
            box_size=context.box_size,
            formal_round_strategy=context.formal_round_strategy,
            attempt_timeout_seconds=context.attempt_timeout_seconds,
            use_multi_stage=context.use_multi_stage,
            use_pocket_guided=context.use_pocket_guided,
            use_rich_exact_rescoring=context.use_rich_exact_rescoring,
            competitors=context.competitors,
            bench_elapsed=context.elapsed(),
            phase=phase,
            complexes=list(context.complexes),
            dq_rows=self.dq_rows,
            dq_results=self.dq_results,
            competitor_rows=self.competitor_rows,
            excluded_rows=excluded_rows,
            render_report=render_report,
        )


@dataclass(frozen=True)
class ExcludedComplexRow:
    pdb_id: str
    reason: str


@dataclass(frozen=True)
class LigandResidue:
    resname: str
    chain_id: str
    residue_id: str
    atom_count: int
    # Additional HETATM residues covalently bonded to the primary residue
    # (e.g. the other half of a multi-residue ligand like a dipeptide).
    # Stored as (resname, chain_id, residue_id) triples.
    bonded_residues: tuple[tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class ScreeningDecision:
    accepted: bool
    reasons: tuple[str, ...]
    ligand_residue: LigandResidue | None = None
    receptor_system: PreparedReceptorSystem | None = None
    chemistry_plan: ReceptorChemistryPlan | None = None
    center: tuple[float, float, float] | None = None
    heavy_atom_count: int = 0
    resolution: float | None = None
    model_count: int = 1


MIN_HEAVY_ATOMS = 10
MAX_RESOLUTION_ANGSTROMS = 2.5
COVALENT_DISTANCE_ANGSTROMS = 1.55
METAL_RESNAMES = {
    "ZN",
    "MG",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "CA",
    "NA",
    "K",
    "CD",
}


def format_gap_proof_label(certified: bool | None) -> str | None:
    if certified is None:
        return None
    return "proved" if certified else "inconclusive"


def _parse_hetatm_residue_coords(
    pdb_path: Path,
) -> dict[tuple[str, str, str], list[tuple[float, float, float]]]:
    """Return per-residue atom coordinates for all non-water HETATM records."""
    coords: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in {"HOH", "DOD", "WAT"}:
                continue
            chain_id = line[21].strip()
            residue_id = f"{line[22:26].strip()}{line[26].strip()}"
            key = (resname, chain_id, residue_id)
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            except ValueError:
                continue
            coords.setdefault(key, []).append((x, y, z))
    return coords


_COVALENT_BOND_DISTANCE_A = 1.8


def _find_covalently_bonded_hetatm_residues(
    residue_coords: dict[tuple[str, str, str], list[tuple[float, float, float]]],
    start: tuple[str, str, str],
) -> frozenset[tuple[str, str, str]]:
    """Flood-fill to find all HETATM residues covalently bonded to `start`.

    Two residues are considered bonded if any pair of their atoms is within
    ``_COVALENT_BOND_DISTANCE_A`` of each other.  Returns the full connected
    component including ``start``.
    """
    visited: set[tuple[str, str, str]] = {start}
    queue: list[tuple[str, str, str]] = [start]
    all_keys = list(residue_coords.keys())
    while queue:
        current = queue.pop()
        current_arr = np.array(residue_coords[current])
        for other in all_keys:
            if other in visited:
                continue
            other_arr = np.array(residue_coords[other])
            # Compute minimum inter-residue distance via broadcasting
            diffs = current_arr[:, None, :] - other_arr[None, :, :]
            min_dist = float(np.sqrt((diffs**2).sum(axis=-1)).min())
            if min_dist <= _COVALENT_BOND_DISTANCE_A:
                visited.add(other)
                queue.append(other)
    return frozenset(visited)


def _non_water_hetatm_residue_counts(pdb_path: Path) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in {"HOH", "DOD", "WAT"}:
                continue
            chain_id = line[21].strip()
            residue_id = f"{line[22:26].strip()}{line[26].strip()}"
            key = (resname, chain_id, residue_id)
            counts[key] = counts.get(key, 0) + 1
    return counts


def detect_primary_ligand_residue(
    pdb_path: Path, preferred_resname: str | None = None
) -> LigandResidue:
    counts = _non_water_hetatm_residue_counts(pdb_path)
    if not counts:
        raise ValueError(f"No ligand found in {pdb_path}")
    if preferred_resname is not None:
        counts = {
            key: value for key, value in counts.items() if key[0] == preferred_resname
        }
        if not counts:
            raise ValueError(
                f"Preferred ligand residue {preferred_resname} not found in {pdb_path}"
            )
    (resname, chain_id, residue_id), _ = max(
        counts.items(), key=lambda x: x[1]
    )
    primary_key = (resname, chain_id, residue_id)

    # Detect multi-residue ligands (e.g. dipeptides) by flood-filling covalent bonds
    residue_coords = _parse_hetatm_residue_coords(pdb_path)
    bonded_group = _find_covalently_bonded_hetatm_residues(residue_coords, primary_key)
    bonded_residues = tuple(
        sorted(key for key in bonded_group if key != primary_key)
    )
    total_atom_count = sum(counts.get(key, 0) for key in bonded_group)

    return LigandResidue(
        resname=resname,
        chain_id=chain_id,
        residue_id=residue_id,
        atom_count=total_atom_count,
        bonded_residues=bonded_residues,
    )


def parse_model_count(pdb_path: Path) -> int:
    model_count = 0
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                model_count += 1
    return model_count if model_count > 0 else 1


def parse_resolution_angstroms(pdb_path: Path) -> float | None:
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("REMARK   2 RESOLUTION."):
                parts = line.split()
                for idx, token in enumerate(parts):
                    if token == "RESOLUTION." and idx + 1 < len(parts):
                        try:
                            return float(parts[idx + 1])
                        except ValueError:
                            return None
    return None


def has_covalent_contact(protein_path: Path, ligand_path: Path) -> bool:
    protein_coords, _ = cast(
        tuple[np.ndarray, np.ndarray],
        parse_structure(protein_path, strip_hydrogens=True),
    )
    ligand_coords, _ = cast(
        tuple[np.ndarray, np.ndarray],
        parse_structure(ligand_path, strip_hydrogens=True),
    )
    if len(protein_coords) == 0 or len(ligand_coords) == 0:
        return False
    diffs = protein_coords[:, None, :] - ligand_coords[None, :, :]
    distances = np.linalg.norm(diffs, axis=-1)
    return bool(np.any(distances < COVALENT_DISTANCE_ANGSTROMS))


def _ligand_residue_keys(ligand_residue: LigandResidue) -> frozenset[ResidueKey]:
    """Return all ResidueKeys belonging to this ligand (primary + bonded residues)."""
    keys: set[ResidueKey] = {
        ResidueKey(
            resname=ligand_residue.resname,
            chain_id=ligand_residue.chain_id,
            residue_id=ligand_residue.residue_id,
        )
    }
    for resname, chain_id, residue_id in ligand_residue.bonded_residues:
        keys.add(ResidueKey(resname=resname, chain_id=chain_id, residue_id=residue_id))
    return frozenset(keys)


def screen_complex(
    entry: PDBEntry,
    pdb_path: Path,
    protein_path: Path | None = None,
    *,
    charge_method: ChargeMethod = DEFAULT_BENCHMARK_CHARGE_METHOD,
    certified_scoring_family: CertifiedScoringFamily = DEFAULT_BENCHMARK_CERTIFIED_SCORING_FAMILY,
    receptor_policy: EssentialSiteComponentsPolicy | None = None,
) -> ScreeningDecision:
    reasons: list[str] = []
    center: tuple[float, float, float] | None = None
    receptor_system: PreparedReceptorSystem | None = None

    if entry.scope != BenchmarkScope.LIGAND:
        reasons.append(f"scope={entry.scope.value} is out of scope for ligand docking")

    if entry.exclusion_reason is not None:
        reasons.append(entry.exclusion_reason)

    resolution = parse_resolution_angstroms(pdb_path)
    if resolution is not None and resolution >= MAX_RESOLUTION_ANGSTROMS:
        reasons.append(
            f"resolution {resolution:.2f}A exceeds gate < {MAX_RESOLUTION_ANGSTROMS:.2f}A"
        )

    model_count = parse_model_count(pdb_path)
    if model_count != 1:
        reasons.append(f"MODEL count {model_count} != 1")

    try:
        ligand_residue = detect_primary_ligand_residue(
            pdb_path, preferred_resname=entry.preferred_resname
        )
    except ValueError as exc:
        reasons.append(str(exc))
        return ScreeningDecision(
            accepted=False,
            reasons=tuple(reasons),
            ligand_residue=None,
            receptor_system=None,
            center=None,
            heavy_atom_count=0,
            resolution=resolution,
            model_count=model_count,
        )

    if ligand_residue.atom_count < MIN_HEAVY_ATOMS:
        reasons.append(
            f"heavy atom count {ligand_residue.atom_count} is below gate >= {MIN_HEAVY_ATOMS}"
        )

    if ligand_residue.resname in METAL_RESNAMES:
        reasons.append(
            f"ligand residue {ligand_residue.resname} is a metal-only species"
        )

    center = find_docking_center(pdb_path, preferred_resname=entry.preferred_resname)
    ligand_path = prepare_ligand(pdb_path, preferred_resname=entry.preferred_resname)
    receptor_path = (
        protein_path if protein_path is not None else prepare_protein(pdb_path)
    )
    if has_covalent_contact(receptor_path, ligand_path):
        reasons.append(
            f"minimum protein-ligand distance is below {COVALENT_DISTANCE_ANGSTROMS:.2f}A"
        )

    policy = (
        EssentialSiteComponentsPolicy() if receptor_policy is None else receptor_policy
    )
    receptor_system = prepare_receptor_system(
        pdb_path,
        center=center,
        pocket_radius=BENCHMARK_PROTOCOL.pocket_radius_a,
        primary_ligand=_ligand_residue_keys(ligand_residue),
        policy=policy,
        receptor_output_path=pdb_path.parent / f"{pdb_path.stem}_protein.pdb",
        pocket_output_path=pdb_path.parent / f"{pdb_path.stem}_protein_pocket.pdb",
    )
    chemistry_plan = BenchmarkChemistryProtocol.derive_plan(
        receptor_system,
        requested_charge_method=charge_method,
        certified_scoring_family=certified_scoring_family,
    )
    reasons.extend(chemistry_plan.compatibility.reasons)

    return ScreeningDecision(
        accepted=not reasons,
        reasons=tuple(reasons),
        ligand_residue=ligand_residue,
        receptor_system=receptor_system,
        chemistry_plan=chemistry_plan,
        center=center,
        heavy_atom_count=ligand_residue.atom_count,
        resolution=resolution,
        model_count=model_count,
    )


def download_pdb(pdb_id: str, cache_dir: Path) -> Path | None:
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
    gz_path = cache_dir / f"{pdb_id}.pdb.gz"
    pdb_path = cache_dir / f"{pdb_id}.pdb"

    if pdb_path.exists():
        return pdb_path

    print(f"  Downloading {pdb_id}...", flush=True)
    try:
        urllib.request.urlretrieve(url, gz_path)
        with gzip.open(gz_path, "rt") as f_in:
            with open(pdb_path, "w") as f_out:
                f_out.write(f_in.read())
        gz_path.unlink()  # remove gz
        return pdb_path
    except Exception as e:
        print(f"  ⚠️  Failed to download {pdb_id}: {e}")
        return None


def _iter_pdb_id_tokens(raw_text: str) -> Iterator[str]:
    for raw_line in raw_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for token in line.replace(",", " ").split():
            pdb_id = token.strip().lower()
            if pdb_id:
                yield pdb_id


def load_explicit_pdb_ids(
    pdb_ids_text: str | None = None, pdb_file: Path | None = None
) -> tuple[str, ...]:
    raw_ids: list[str] = []
    if pdb_ids_text is not None:
        raw_ids.extend(_iter_pdb_id_tokens(pdb_ids_text))
    if pdb_file is not None:
        raw_ids.extend(_iter_pdb_id_tokens(pdb_file.read_text()))
    if not raw_ids:
        return ()

    invalid_ids = [
        pdb_id for pdb_id in raw_ids if len(pdb_id) != 4 or not pdb_id.isalnum()
    ]
    if invalid_ids:
        raise ValueError(
            "PDB IDs must be 4-character alphanumeric codes: "
            + ", ".join(sorted(set(invalid_ids)))
        )

    seen: set[str] = set()
    duplicates: list[str] = []
    normalized_ids: list[str] = []
    for pdb_id in raw_ids:
        if pdb_id in seen:
            duplicates.append(pdb_id)
            continue
        seen.add(pdb_id)
        normalized_ids.append(pdb_id)
    if duplicates:
        raise ValueError(
            "Duplicate explicit PDB IDs are not allowed: "
            + ", ".join(sorted(set(duplicates)))
        )

    return tuple(normalized_ids)


def build_explicit_pdb_entries(pdb_ids: Sequence[str]) -> tuple[PDBEntry, ...]:
    return tuple(
        PDBEntry(
            pdb_id=pdb_id,
            target_name=pdb_id,
            category="explicit",
            scope=BenchmarkScope.LIGAND,
            include_by_default=False,
        )
        for pdb_id in pdb_ids
    )


def derive_benchmark_target_selection(
    *,
    dataset: Literal["curated", "casf2007"],
    n_complexes: int,
    explicit_pdb_ids: Sequence[str] = (),
) -> BenchmarkTargetSelection:
    if explicit_pdb_ids:
        cache_dir = Path("./pdb_cache")
        cache_dir.mkdir(exist_ok=True)
        return BenchmarkTargetSelection(
            mode=BenchmarkTargetSelectionMode.EXPLICIT_PDB_IDS,
            label="explicit PDB IDs",
            cache_dir=cache_dir,
            n_complexes_requested=len(explicit_pdb_ids),
            entries=build_explicit_pdb_entries(explicit_pdb_ids),
        )

    if dataset == "casf2007":
        cache_dir = Path("./casf2007_pdb_cache")
        cache_dir.mkdir(exist_ok=True)
        print("Downloading CASF-2007 PDBs to cache...", flush=True)
        casf_ids = get_casf_2007_ids()
        for pdb_id in casf_ids:
            pdb_path = cache_dir / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                download_pdb(pdb_id, cache_dir)
                print(f"  Downloaded {pdb_id}", flush=True)
        print(f"CASF-2007 PDBs cached: {len(casf_ids)}", flush=True)
        available_entries = tuple(get_casf_2007_entries())
        print(
            f"CASF-2007 QC-passed: {len(available_entries)} / {len(casf_ids)}",
            flush=True,
        )
        return BenchmarkTargetSelection(
            mode=BenchmarkTargetSelectionMode.DATASET,
            label="casf2007 dataset",
            cache_dir=cache_dir,
            n_complexes_requested=n_complexes,
            entries=available_entries[:n_complexes],
        )

    cache_dir = Path("./pdb_cache")
    cache_dir.mkdir(exist_ok=True)
    dataset_exclusions = [
        entry
        for entry in get_benchmark_entries()
        if entry.scope == BenchmarkScope.LIGAND
        and not entry.include_by_default
        and entry.exclusion_reason is not None
    ]
    if dataset_exclusions:
        print("Configured dataset exclusions:", flush=True)
        for entry in dataset_exclusions:
            print(f"  {entry.pdb_id}: {entry.exclusion_reason}", flush=True)
    available_entries = tuple(get_default_ligand_entries())
    return BenchmarkTargetSelection(
        mode=BenchmarkTargetSelectionMode.DATASET,
        label="curated dataset",
        cache_dir=cache_dir,
        n_complexes_requested=n_complexes,
        entries=available_entries[:n_complexes],
    )


def hydrate_selected_benchmark_entry(
    entry: PDBEntry,
    pdb_path: Path,
    selection_mode: BenchmarkTargetSelectionMode,
) -> PDBEntry:
    if selection_mode != BenchmarkTargetSelectionMode.EXPLICIT_PDB_IDS:
        return entry
    return replace(entry, target_name=_extract_pdb_title(pdb_path))


def prepare_protein(pdb_path: Path) -> Path:
    protein_path = pdb_path.parent / f"{pdb_path.stem}_protein.pdb"
    if protein_path.exists():
        protein_path.unlink(missing_ok=True)
    return prepare_protein_only_receptor(pdb_path, output_path=protein_path)


def prepare_pocket_protein(
    protein_path: Path,
    center: tuple[float, float, float],
    pocket_radius: float = BENCHMARK_PROTOCOL.pocket_radius_a,
) -> Path:
    pocket_path = protein_path.parent / f"{protein_path.stem}_pocket.pdb"
    return prepare_protein_only_pocket(
        protein_path,
        center=center,
        pocket_radius=pocket_radius,
        output_path=pocket_path,
    )


def prepare_ligand(pdb_path: Path, preferred_resname: str | None = None) -> Path:
    """Prepare ligand for Vina - extract primary ligand HETATM records from PDB file.

    Keep only blank / `A` alternate locations so the extracted ligand matches the
    atom set used by chemistry tools such as RDKit.

    Raises error if no ligand found in the PDB.
    """
    ligand_path = pdb_path.parent / f"{pdb_path.stem}_ligand.pdb"

    # Check if already extracted
    if ligand_path.exists():
        # Remove it if we are re-running to fix old bad extractions
        ligand_path.unlink(missing_ok=True)

    target_residue = detect_primary_ligand_residue(
        pdb_path, preferred_resname=preferred_resname
    )
    # Collect all (resname, chain_id, residue_id) keys for the full ligand group
    target_keys: set[tuple[str, str, str]] = {
        (target_residue.resname, target_residue.chain_id, target_residue.residue_id)
    }
    target_keys.update(target_residue.bonded_residues)

    ligand_lines = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            if line[16] not in (" ", "A"):
                continue
            chain_id = line[21].strip()
            residue_id = f"{line[22:26].strip()}{line[26].strip()}"
            resname = line[17:20].strip()
            if (resname, chain_id, residue_id) in target_keys:
                ligand_lines.append(line)

    if not ligand_lines:
        raise ValueError(f"No ligand found in {pdb_path}")

    with open(ligand_path, "w") as f:
        f.writelines(ligand_lines)
    return ligand_path


def _charge_source(
    assigner: ChargeAssigner,
    *,
    elements: tuple[str, ...],
    pdb_path: Path,
) -> tuple[str, ...] | Path:
    return elements if assigner.method == ChargeMethod.SIMPLE else pdb_path


def prepare_benchmark_electrostatics(
    *,
    assigner: ChargeAssigner,
    pdb_id: str,
    ligand_pdb: Path,
    ligand_elements: tuple[str, ...],
    ligand_atom_count: int,
    pocket_receptor_pdb: Path,
    pocket_elements: tuple[str, ...],
    pocket_atom_count: int,
) -> PreparedBenchmarkElectrostatics:
    ligand_charges = np.asarray(
        assigner.assign(
            _charge_source(assigner, elements=ligand_elements, pdb_path=ligand_pdb)
        ).charges
    )
    pocket_charges = np.asarray(
        assigner.assign(
            _charge_source(
                assigner,
                elements=pocket_elements,
                pdb_path=pocket_receptor_pdb,
            )
        ).charges
    )
    if len(ligand_charges) != ligand_atom_count:
        raise ValueError(
            f"Ligand charge count mismatch for {pdb_id}: charges={len(ligand_charges)} atoms={ligand_atom_count}"
        )
    if len(pocket_charges) != pocket_atom_count:
        raise ValueError(
            f"Pocket charge count mismatch for {pdb_id}: charges={len(pocket_charges)} atoms={pocket_atom_count}"
        )
    return PreparedBenchmarkElectrostatics(
        ligand_charges=ligand_charges,
        pocket_charges=pocket_charges,
    )


def prepare_benchmark_complex(
    complex_entry: BenchmarkComplex,
    pose_dir: Path,
    box_size: float = BENCHMARK_PROTOCOL.box_size_a,
    certified_scoring_family: CertifiedScoringFamily = DEFAULT_BENCHMARK_CERTIFIED_SCORING_FAMILY,
) -> PreparedBenchmarkComplex:
    pocket_radius = derive_benchmark_pocket_radius_from_box_size(box_size)
    receptor_system = complex_entry.receptor_system
    if receptor_system is None:
        ligand_residue = (
            complex_entry.ligand_residue
            if complex_entry.ligand_residue is not None
            else detect_primary_ligand_residue(
                complex_entry.path,
                preferred_resname=complex_entry.preferred_resname,
            )
        )
        receptor_system = prepare_receptor_system(
            complex_entry.path,
            center=complex_entry.center,
            pocket_radius=pocket_radius,
            primary_ligand=_ligand_residue_keys(ligand_residue),
            policy=EssentialSiteComponentsPolicy(),
            receptor_output_path=complex_entry.path.parent
            / f"{complex_entry.path.stem}_protein.pdb",
            pocket_output_path=complex_entry.path.parent
            / f"{complex_entry.path.stem}_protein_pocket.pdb",
        )
    receptor_pdb = receptor_system.receptor_pdb
    pocket_receptor_pdb = receptor_system.pocket_receptor_pdb
    chemistry_plan = complex_entry.chemistry_plan
    if chemistry_plan is None:
        chemistry_plan = BenchmarkChemistryProtocol.derive_plan(
            receptor_system,
            requested_charge_method=DEFAULT_BENCHMARK_CHARGE_METHOD,
            certified_scoring_family=certified_scoring_family,
        )
    ligand_pdb = prepare_ligand(
        complex_entry.path,
        preferred_resname=complex_entry.preferred_resname,
    )

    ligand_coords, ligand_radii, ligand_elements = cast(
        tuple[np.ndarray, np.ndarray, list[str]],
        parse_structure(ligand_pdb, return_elements=True),
    )
    pocket_coords_np = np.asarray(receptor_system.pocket_coords)
    pocket_radii_np = np.asarray(receptor_system.pocket_radii)
    pocket_elements = list(receptor_system.pocket_elements)
    if len(pocket_coords_np) == 0:
        raise ValueError("No protein atoms in pocket")

    ligand_charges: np.ndarray | None = None
    pocket_charges: np.ndarray | None = None
    if certified_scoring_family == CertifiedScoringFamily.LJ_REALSPACE_EWALD:
        charge_assigner = create_charge_assigner(chemistry_plan.charge_method)
        electrostatics = prepare_benchmark_electrostatics(
            assigner=charge_assigner,
            pdb_id=complex_entry.pdb_id,
            ligand_pdb=ligand_pdb,
            ligand_elements=tuple(ligand_elements),
            ligand_atom_count=len(ligand_coords),
            pocket_receptor_pdb=pocket_receptor_pdb,
            pocket_elements=tuple(pocket_elements),
            pocket_atom_count=len(pocket_coords_np),
        )
        ligand_charges = electrostatics.ligand_charges
        pocket_charges = electrostatics.pocket_charges

    receptor_view_pdb = pose_dir / f"{complex_entry.pdb_id}_receptor.pdb"
    native_ligand_view_pdb = pose_dir / f"{complex_entry.pdb_id}_native_ligand.pdb"
    copy_structure_for_viewing(receptor_pdb, receptor_view_pdb)
    copy_structure_for_viewing(ligand_pdb, native_ligand_view_pdb)

    return PreparedBenchmarkComplex(
        pdb_id=complex_entry.pdb_id,
        target_name=complex_entry.target_name,
        path=complex_entry.path,
        center=complex_entry.center,
        box_size=(box_size, box_size, box_size),
        receptor_pdb=receptor_pdb,
        pocket_receptor_pdb=pocket_receptor_pdb,
        ligand_pdb=ligand_pdb,
        receptor_view_pdb=receptor_view_pdb,
        native_ligand_view_pdb=native_ligand_view_pdb,
        ligand_coords=ligand_coords,
        ligand_radii=ligand_radii,
        ligand_elements=tuple(ligand_elements),
        ligand_charges=ligand_charges,
        pocket_coords=pocket_coords_np,
        pocket_radii=pocket_radii_np,
        pocket_elements=tuple(pocket_elements),
        pocket_charges=pocket_charges,
        preferred_resname=complex_entry.preferred_resname,
        receptor_system=receptor_system,
        charge_method=chemistry_plan.charge_method,
        chemistry_protocol_name=chemistry_plan.protocol_name,
    )


from dq_dock_engine.docking.physics_params import get_vdw_radius


def extract_coords_and_radii(
    pdb_path: Path, is_ligand: bool = True
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract heavy atom coordinates and VdW radii from PDB file."""
    coords = []
    radii = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM") or line.startswith("ATOM"):
                element = line[76:78].strip()
                if not element:
                    element_str = line[12:16].strip()
                    if element_str.startswith("H"):
                        if is_ligand:
                            continue
                        element = "H"
                    else:
                        element = element_str[0]
                elif element == "H" and is_ligand:
                    continue

                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                    radii.append(get_vdw_radius(element))
                except ValueError:
                    continue

    if coords:
        return np.array(coords), np.array(radii)
    return None, None


def compute_pocket_center(ligand_coords: np.ndarray) -> tuple:
    """Compute center of ligand for docking box."""
    center = ligand_coords.mean(axis=0)
    return tuple(center)


def find_docking_center(pdb_path: Path, preferred_resname: str | None = None) -> tuple:
    """Find docking center from ligand in PDB."""
    ligand_pdb = prepare_ligand(pdb_path, preferred_resname=preferred_resname)
    coords, _, _ = cast(
        tuple[np.ndarray, np.ndarray, list[str]],
        parse_structure(ligand_pdb, return_elements=True),
    )
    return compute_pocket_center(coords)


def find_docking_binary(candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        path = shutil.which(name)
        if path is not None:
            return path
    for name in candidates:
        local = Path(name)
        if local.exists():
            return str(local.absolute())
    return None


def convert_ligand_pdb_to_sdf(ligand_pdb: Path, ligand_sdf: Path) -> None:
    from rdkit import Chem

    mol = Chem.MolFromPDBFile(str(ligand_pdb), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit failed to parse ligand PDB: {ligand_pdb}")
    mol = Chem.AddHs(mol, addCoords=True)
    writer = Chem.SDWriter(str(ligand_sdf))
    writer.write(mol)
    writer.close()


class DockingPreparationStrategy(ABC):
    @abstractmethod
    def protocol_label(self) -> str:
        """Stable label for reports and saved metadata."""

    @abstractmethod
    def prepare_inputs(
        self,
        receptor_pdb: Path,
        ligand_pdb: Path,
    ) -> PreparedDockingInputs:
        """Prepare docking inputs for one external engine run."""


class DirectPDBPreparationStrategy(DockingPreparationStrategy):
    def protocol_label(self) -> str:
        return "direct_pdb"

    def prepare_inputs(
        self,
        receptor_pdb: Path,
        ligand_pdb: Path,
    ) -> PreparedDockingInputs:
        return PreparedDockingInputs(
            protocol_label=self.protocol_label(),
            receptor_path=receptor_pdb,
            ligand_path=ligand_pdb,
        )


class PDBQTPreparationStrategy(DockingPreparationStrategy, ABC):
    def __init__(self, receptor_tool: str, ligand_tool: str):
        self.receptor_tool = receptor_tool
        self.ligand_tool = ligand_tool

    def prepare_inputs(
        self,
        receptor_pdb: Path,
        ligand_pdb: Path,
    ) -> PreparedDockingInputs:
        temp_dir = Path(tempfile.mkdtemp(prefix="vina_prep_"))
        receptor_pdbqt = temp_dir / f"{receptor_pdb.stem}.pdbqt"
        ligand_pdbqt = temp_dir / f"{ligand_pdb.stem}.pdbqt"
        ligand_source = self.prepare_ligand_source(ligand_pdb, temp_dir)
        subprocess.run(
            self.build_receptor_command(receptor_pdb, receptor_pdbqt, temp_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            self.build_ligand_command(ligand_source, ligand_pdbqt, temp_dir),
            check=True,
            capture_output=True,
            text=True,
        )
        return PreparedDockingInputs(
            protocol_label=self.protocol_label(),
            receptor_path=receptor_pdbqt,
            ligand_path=ligand_pdbqt,
            cleanup_dir=temp_dir,
        )

    def prepare_ligand_source(self, ligand_pdb: Path, temp_dir: Path) -> Path:
        return ligand_pdb

    @abstractmethod
    def build_receptor_command(
        self,
        receptor_pdb: Path,
        receptor_pdbqt: Path,
        temp_dir: Path,
    ) -> list[str]:
        """Build the receptor preparation command."""

    @abstractmethod
    def build_ligand_command(
        self,
        ligand_source: Path,
        ligand_pdbqt: Path,
        temp_dir: Path,
    ) -> list[str]:
        """Build the ligand preparation command."""


class MeekoPDBQTPreparationStrategy(PDBQTPreparationStrategy):
    def protocol_label(self) -> str:
        return "meeko_pdbqt"

    def prepare_ligand_source(self, ligand_pdb: Path, temp_dir: Path) -> Path:
        ligand_sdf = temp_dir / f"{ligand_pdb.stem}.sdf"
        convert_ligand_pdb_to_sdf(ligand_pdb, ligand_sdf)
        return ligand_sdf

    def build_receptor_command(
        self,
        receptor_pdb: Path,
        receptor_pdbqt: Path,
        temp_dir: Path,
    ) -> list[str]:
        return [
            self.receptor_tool,
            "--read_pdb",
            str(receptor_pdb),
            "-o",
            str(temp_dir / receptor_pdb.stem),
            "-p",
            str(receptor_pdbqt),
            "-a",
            "--default_altloc",
            "A",
        ]

    def build_ligand_command(
        self,
        ligand_source: Path,
        ligand_pdbqt: Path,
        temp_dir: Path,
    ) -> list[str]:
        return [
            self.ligand_tool,
            "-i",
            str(ligand_source),
            "-o",
            str(ligand_pdbqt),
        ]


class AutoDockToolsPDBQTPreparationStrategy(PDBQTPreparationStrategy):
    def protocol_label(self) -> str:
        return "autodocktools_pdbqt"

    def build_receptor_command(
        self,
        receptor_pdb: Path,
        receptor_pdbqt: Path,
        temp_dir: Path,
    ) -> list[str]:
        return [
            self.receptor_tool,
            "-r",
            str(receptor_pdb),
            "-o",
            str(receptor_pdbqt),
            "-A",
            "checkhydrogens",
        ]

    def build_ligand_command(
        self,
        ligand_source: Path,
        ligand_pdbqt: Path,
        temp_dir: Path,
    ) -> list[str]:
        return [
            self.ligand_tool,
            "-l",
            str(ligand_source),
            "-o",
            str(ligand_pdbqt),
            "-A",
            "checkhydrogens",
        ]


def detect_pdbqt_preparation_strategies() -> tuple[DockingPreparationStrategy, ...]:
    strategies: list[DockingPreparationStrategy] = []
    mk_receptor = shutil.which("mk_prepare_receptor.py")
    mk_ligand = shutil.which("mk_prepare_ligand.py")
    if mk_receptor is not None and mk_ligand is not None:
        strategies.append(MeekoPDBQTPreparationStrategy(mk_receptor, mk_ligand))
    receptor_tool = shutil.which("prepare_receptor4.py")
    ligand_tool = shutil.which("prepare_ligand4.py")
    if receptor_tool is not None and ligand_tool is not None:
        strategies.append(
            AutoDockToolsPDBQTPreparationStrategy(receptor_tool, ligand_tool)
        )
    return tuple(strategies)


def _extract_pdb_element(line: str) -> str:
    element = line[76:78].strip()
    if element:
        return element
    atom_name = line[12:16].strip()
    return atom_name[0] if atom_name else ""


def parse_smina_models(pdb_path: Path) -> tuple[np.ndarray, ...]:
    """Extract heavy-atom coordinates for every returned docking model."""
    models: list[np.ndarray] = []
    current_coords: list[list[float]] = []
    saw_model = False

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("MODEL"):
                saw_model = True
                current_coords = []
                continue
            if line.startswith("ENDMDL"):
                if current_coords:
                    models.append(np.array(current_coords, dtype=np.float64))
                current_coords = []
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            element = _extract_pdb_element(line)
            if element == "H":
                continue

            try:
                current_coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError:
                continue

    if current_coords:
        models.append(np.array(current_coords, dtype=np.float64))

    if not saw_model and models:
        return (models[0],)
    return tuple(models)


def parse_smina_affinities(stdout: str) -> tuple[float, ...]:
    affinities: list[float] = []
    in_results = False
    for line in stdout.split("\n"):
        if "mode |" in line and "affinity" in line:
            in_results = True
            continue
        stripped = line.strip()
        if not in_results or not stripped or not stripped[0].isdigit():
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        try:
            affinities.append(float(parts[1]))
        except ValueError:
            continue
    return tuple(affinities)


def compute_pose_rmsd(pose_coords: np.ndarray, native_coords: np.ndarray) -> float:
    min_len = min(len(pose_coords), len(native_coords))
    if min_len == 0:
        return float("nan")
    pose_jnp = jnp.expand_dims(pose_coords[:min_len], axis=0)
    native_jnp = jnp.array(native_coords[:min_len])
    return float(compute_docking_rmsd_batched(pose_jnp, native_jnp)[0])


def _safe_json_value(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def _format_benchmark_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def create_benchmark_output_paths(output_dir: Path) -> BenchmarkOutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"pdb_redocking_{timestamp}"
    return BenchmarkOutputPaths(
        json_path=output_dir / f"{stem}.json",
        csv_path=output_dir / f"{stem}.csv",
    )


def benchmark_pose_dir(output_paths: BenchmarkOutputPaths) -> Path:
    pose_dir = output_paths.json_path.with_suffix("").with_name(
        output_paths.json_path.stem + "_poses"
    )
    pose_dir.mkdir(parents=True, exist_ok=True)
    return pose_dir


def save_pose_from_template(
    coords: np.ndarray, template_pdb: Path, output_pdb: Path
) -> None:
    from dq_dock_engine.docking.scoring import _write_pdb

    _write_pdb(coords, str(template_pdb), str(output_pdb))


def copy_structure_for_viewing(source: Path, destination: Path) -> None:
    shutil.copy2(source, destination)


def save_benchmark_results(
    output_paths: BenchmarkOutputPaths,
    *,
    charge_method_name: str,
    certified_scoring_family: CertifiedScoringFamily,
    n_complexes_requested: int,
    n_poses: int,
    n_opt_steps: int,
    top_k_to_optimize: int,
    box_size: float,
    formal_round_strategy: FormalRoundStrategy,
    attempt_timeout_seconds: float | None,
    use_multi_stage: bool,
    use_pocket_guided: bool,
    use_rich_exact_rescoring: bool,
    competitors: tuple[CompetitorMetadata, ...],
    bench_elapsed: float,
    phase: str,
    complexes: Sequence[PreparedBenchmarkComplex],
    dq_rows: list[ComplexSummaryRow],
    dq_results: list[BenchmarkResult],
    competitor_rows: list[CompetitorResultRow],
    excluded_rows: list[ExcludedComplexRow],
    render_report: bool = False,
) -> tuple[Path, Path]:
    json_path = output_paths.json_path
    csv_path = output_paths.csv_path

    if len(dq_rows) != len(dq_results):
        raise ValueError(
            "DQ-Dock row/result mismatch: "
            f"{len(dq_rows)} rows vs {len(dq_results)} results"
        )

    complex_map = {cx.pdb_id: cx for cx in complexes}
    dq_map = {row.pdb_id: row for row in dq_rows}
    dq_result_map = {row.pdb_id: result for row, result in zip(dq_rows, dq_results)}
    competitor_map = {
        metadata.engine_id: {
            row.pdb_id: row
            for row in competitor_rows
            if row.engine_id == metadata.engine_id
        }
        for metadata in competitors
    }
    competitor_ids = {metadata.engine_id for metadata in competitors}
    if phase == "complete":
        for metadata in competitors:
            rows_for_engine = competitor_map[metadata.engine_id]
            missing = sorted(set(dq_map) - set(rows_for_engine))
            extra = sorted(set(rows_for_engine) - set(dq_map))
            if missing or extra:
                raise ValueError(
                    f"{metadata.display_name} results are out of sync with DQ-Dock results: "
                    f"missing={missing}, extra={extra}"
                )
    elif phase in competitor_ids:
        partial_rows = competitor_map[phase]
        extra = sorted(set(partial_rows) - set(dq_map))
        if extra:
            raise ValueError(
                f"{phase} results contain complexes absent from DQ-Dock rows: extra={extra}"
            )

    competitor_stats = {
        metadata.engine_id: CompetitorStatistics.from_rows(
            metadata,
            list(competitor_map[metadata.engine_id].values()),
        )
        for metadata in competitors
    }
    box_protocol_rule, box_protocol_theorem = benchmark_box_protocol_metadata(box_size)
    active_runtime_contract = active_formal_runtime_contract(formal_round_strategy)
    active_runtime_handles = handle_bundle_from_contracts(
        active_runtime_contract,
        extra_theorem_handles=scoring_family_theorem_handles(certified_scoring_family),
    )
    staged_runtime_handles = handle_bundle_from_contracts(
        STAGED_COARSE_TOP1_RUNTIME_CONTRACT,
        STAGED_SINGLETON_TOP1_RUNTIME_CONTRACT,
        extra_theorem_handles=(CP3,),
        extra_witness_handles=(FLO18,),
    )

    csv_rows: list[dict[str, object]] = []
    for pdb_id, row in dq_map.items():
        cx = complex_map[pdb_id]
        dq_result = dq_result_map[pdb_id]
        csv_rows.append(row.to_csv_record(cx, dq_result))
    for row in competitor_rows:
        cx = complex_map[row.pdb_id]
        csv_rows.append(row.to_csv_record(cx))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(csv_rows[0].keys())
            if csv_rows
            else [
                "method",
                "engine_id",
                "pdb_id",
                "target_name",
                "center_x",
                "center_y",
                "center_z",
                "box_x",
                "box_y",
                "box_z",
                "ligand_atoms",
                "pocket_atoms",
                "score_name",
                "score",
                "top_rmsd",
                "best_mode_rmsd",
                "time_s",
                "dock_time_s",
                "checkpoint_time_s",
                "total_wall_time_s",
                "pose_pdb",
                "gap_proof",
                "native_rank",
                "energy_gap",
                "receptor_pdb",
                "native_ligand_pdb",
                "status",
                "error",
            ],
        )
        writer.writeheader()
        for row in csv_rows:
            writer.writerow({k: _safe_json_value(v) for k, v in row.items()})

    successful_dq_rmsds = [
        row.rmsd
        for row, result in zip(dq_rows, dq_results)
        if result.success and row.rmsd is not None
    ]

    summary: dict[str, object] = {
        "timestamp": json_path.stem.removeprefix("pdb_redocking_"),
        "phase": phase,
        "charge_method": charge_method_name,
        "n_complexes_requested": n_complexes_requested,
        "n_complexes_run": len(dq_rows),
        "n_complexes_excluded": len(excluded_rows),
        "n_poses": n_poses,
        "n_opt_steps": n_opt_steps,
        "top_k_to_optimize": top_k_to_optimize,
        "formal_round_strategy": formal_round_strategy.value,
        "attempt_timeout_seconds": attempt_timeout_seconds,
        "benchmark_pocket_radius_a": BENCHMARK_PROTOCOL.pocket_radius_a,
        "benchmark_box_size_a": box_size,
        "benchmark_box_protocol_rule": box_protocol_rule,
        "benchmark_box_protocol_theorem": box_protocol_theorem,
        "active_certified_runtime_theorems": active_runtime_handles.theorem_handles,
        "active_certified_runtime_witnesses": active_runtime_handles.witness_handles,
        "active_certified_runtime_contract": runtime_contract_record(
            active_runtime_contract
        ),
        "staged_coarse_pruning_theorems": staged_runtime_handles.theorem_handles,
        "staged_coarse_pruning_witnesses": staged_runtime_handles.witness_handles,
        "staged_coarse_runtime_contracts": [
            runtime_contract_record(STAGED_COARSE_TOP1_RUNTIME_CONTRACT),
            runtime_contract_record(STAGED_SINGLETON_TOP1_RUNTIME_CONTRACT),
        ],
        "use_multi_stage": use_multi_stage,
        "use_pocket_guided": use_pocket_guided,
        "use_rich_exact_rescoring": use_rich_exact_rescoring,
        "competitors": [
            {
                "engine_id": metadata.engine_id,
                "display_name": metadata.display_name,
                "score_name": metadata.score_name,
                "prep_protocol": metadata.prep_protocol,
            }
            for metadata in competitors
        ],
        "competitor_stats": {
            engine_id: stats.to_record()
            for engine_id, stats in competitor_stats.items()
        },
        "total_benchmark_time_s": bench_elapsed,
        "dq_avg_rmsd": None
        if not successful_dq_rmsds
        else float(np.mean(successful_dq_rmsds)),
        "dq_total_time_s": float(sum(row.time for row in dq_rows)),
        "dq_total_dock_time_s": float(
            sum(result.dock_time_s or result.time for result in dq_results)
        ),
        "dq_total_checkpoint_time_s": float(
            sum(result.checkpoint_time_s or 0.0 for result in dq_results)
        ),
    }

    payload = {
        "summary": {k: _safe_json_value(v) for k, v in summary.items()},
        "dq_dock": [
            row.to_json_record(complex_map[row.pdb_id], dq_result_map[row.pdb_id])
            for row in dq_rows
        ],
        "competitors": [
            row.to_json_record(complex_map[row.pdb_id]) for row in competitor_rows
        ],
        "excluded": [
            {"pdb_id": row.pdb_id, "reason": row.reason} for row in excluded_rows
        ],
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    if render_report:
        render_redocking_report(json_path)

    return json_path, csv_path


def run_cli_docking(
    command: list[str],
    saved_output_pdb: Path | None = None,
    timeout_seconds: float | None = None,
) -> DockingRunResult:
    """Run one external docking command with Vina-style stdout/PDB output."""
    # Use a safe temp file path
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdb")
    os.close(temp_fd)
    output_path = Path(temp_path)

    command_with_output = [*command, "--out", str(output_path)]

    start = time.time()
    try:
        result = subprocess.run(
            command_with_output,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.time() - start

        if result.returncode != 0:
            return DockingRunResult(
                success=False,
                error=result.stderr[:200],
                time=elapsed,
            )

        affinities = parse_smina_affinities(result.stdout)
        models = parse_smina_models(output_path) if output_path.exists() else ()
        top_coords = models[0] if models else None
        if saved_output_pdb is not None and output_path.exists():
            shutil.copy2(output_path, saved_output_pdb)

        return DockingRunResult(
            success=True,
            score=affinities[0] if affinities else None,
            time=elapsed,
            top_coords=top_coords,
            model_coords=models,
        )
    except subprocess.TimeoutExpired:
        return DockingRunResult(
            success=False,
            error=(
                f"timeout after {timeout_seconds:.1f}s"
                if timeout_seconds is not None
                else "timeout"
            ),
            time=time.time() - start,
        )
    except Exception as e:
        return DockingRunResult(success=False, error=str(e), time=time.time() - start)
    finally:
        if output_path.exists():
            output_path.unlink()


def run_dq_dock(
    pocket_coords: np.ndarray,
    pocket_radii: np.ndarray,
    ligand_coords: np.ndarray,
    ligand_radii: np.ndarray,
    center: tuple,
    ligand_file: Path,
    receptor_file: Path,
    charge_method: ChargeMethod = DEFAULT_BENCHMARK_CHARGE_METHOD,
    precomputed_ligand_charges: np.ndarray | None = None,
    precomputed_receptor_charges: np.ndarray | None = None,
    n_poses: int = 2000,
    use_multi_stage: bool = DEFAULT_BENCHMARK_USE_MULTI_STAGE,
    use_pocket_guided: bool = DEFAULT_BENCHMARK_USE_POCKET_GUIDED,
    engine: ScoringEngine = ScoringEngine.INTERNAL_LJ,
    ligand_elements: list[str] | tuple[str, ...] | None = None,
    receptor_elements: list[str] | tuple[str, ...] | None = None,
    n_opt_steps: int = 50,
    top_k_to_optimize: int = DEFAULT_BENCHMARK_TOP_K_TO_OPTIMIZE,
    use_rich_exact_rescoring: bool = True,
    max_retries: int = 6,
    optimizer_backend: "OptimizerBackend" = DEFAULT_BENCHMARK_OPTIMIZER_BACKEND,
    formal_round_strategy: FormalRoundStrategy = DEFAULT_BENCHMARK_FORMAL_ROUND_STRATEGY,
    certified_scoring_family: CertifiedScoringFamily = DEFAULT_BENCHMARK_CERTIFIED_SCORING_FAMILY,
    retry_break_rmsd: float = 0.0,
    retry_preserve_seed: bool = False,
    box_size: float = BENCHMARK_PROTOCOL.box_size_a,
) -> tuple[BenchmarkResult, np.ndarray | None]:
    """Run true DQ-Dock pipeline on complex using core infrastructure."""
    import os

    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    from dq_dock_engine.docking.core import (
        CertifiedPocketFailureReason,
        DockingBox,
        ScoringEngine,
    )
    from dq_dock_engine.docking.pdb_io import (
        build_ligand_context,
        build_receptor_arrays,
    )
    from dq_dock_engine.docking_config import (
        FormalRoundStrategy,
        create_config,
    )
    from dq_dock_engine.docking.metrics import compute_docking_rmsd_batched

    start = time.time()

    center_jnp = jnp.array(center)
    box = DockingBox(center=center_jnp, size=jnp.array([box_size, box_size, box_size]))

    optimizer_str = (
        "formal" if optimizer_backend == OptimizerBackend.FORMAL else "gradient"
    )
    mode_str = (
        "certified" if optimizer_backend == OptimizerBackend.FORMAL else "heuristic"
    )
    docking_config = create_config(
        mode=mode_str,
        optimizer=optimizer_str,
        formal_round_strategy=formal_round_strategy,
        certified_scoring_family=certified_scoring_family,
        use_rich_exact_rescoring=use_rich_exact_rescoring,
        target_error=0.001,
        min_energy_gap=0.0,
        use_external_scorer=False,
    )

    if ligand_elements is None:
        ligand_elements = None
    else:
        # ensure tuple input for assigner
        ligand_elements = tuple(ligand_elements)

    ligand_charges = precomputed_ligand_charges
    if (
        ligand_charges is None
        and (
            use_multi_stage
            or certified_scoring_family == CertifiedScoringFamily.LJ_REALSPACE_EWALD
        )
        and ligand_elements is not None
    ):
        assigner = create_charge_assigner(charge_method)
        ligand_source = (
            ligand_elements if assigner.method.name == "SIMPLE" else ligand_file
        )
        ligand_charges = assigner.assign(ligand_source).charges

    # Build LigandContext via core infra (auto-centers, stores radii/elements/charges)
    ligand_ctx = build_ligand_context(
        ligand_coords,
        ligand_radii,
        elements=list(ligand_elements) if ligand_elements is not None else None,
        charges=np.asarray(ligand_charges) if ligand_charges is not None else None,
    )

    runtime_receptor_elements, runtime_receptor_charges = _ligand_receptor_runtime_args(
        receptor_elements=receptor_elements,
        precomputed_receptor_charges=precomputed_receptor_charges,
    )
    runtime_protein_coords = jnp.array(pocket_coords)
    runtime_receptor_radii = jnp.array(pocket_radii)
    attempt_executor = derive_blind_docking_executor(
        docking_config.mode,
        use_pocket_guided=use_pocket_guided,
    )

    def finalize_attempt_failure(
        failure_result: BenchmarkResult,
    ) -> tuple[BenchmarkResult, np.ndarray | None]:
        if best_result is not None:
            print(
                "    [Info] Preserving best successful retry result after a later failed attempt.",
                flush=True,
            )
            return best_result, best_pose_coords
        return failure_result, None

    def build_failure_result(
        *,
        error: str,
        execution_path_override: BlindDockingExecutionPath | None = None,
        certified_failure_reason_override=None,
        theorem_handles_override: tuple[str, ...] | None = None,
        elapsed: float | None = None,
        formal_status_override: str | None = None,
    ) -> BenchmarkResult:
        return BenchmarkResult(
            success=False,
            energy=0.0,
            rmsd=0.0,
            time=time.time() - start if elapsed is None else elapsed,
            n_atoms=0,
            formal_status=(
                formal_status
                if formal_status_override is None
                else formal_status_override
            ),
            execution_path=(
                execution_path
                if execution_path_override is None
                else execution_path_override
            ),
            certified_failure_reason=(
                certified_failure_reason
                if certified_failure_reason_override is None
                else certified_failure_reason_override
            ),
            theorem_handles=(
                theorem_handles
                if theorem_handles_override is None
                else theorem_handles_override
            ),
            error=error,
        )

    def execute_attempt(attempt_key: jax.Array) -> BlindDockingExecutionOutcome:
        return attempt_executor.execute(
            BlindDockingAttemptContext(
                protein_coords=runtime_protein_coords,
                receptor_radii=runtime_receptor_radii,
                ligand_ctx=ligand_ctx,
                box=box,
                n_poses=n_poses,
                key=attempt_key,
                charge_method=charge_method,
                receptor_file=receptor_file,
                precomputed_receptor_charges=runtime_receptor_charges,
                receptor_elements=runtime_receptor_elements,
                config=docking_config,
                top_k=1,
                optimize=True,
                n_opt_steps=n_opt_steps,
                top_k_to_optimize=top_k_to_optimize,
                include_native=True,
                engine=engine,
                use_multi_stage=use_multi_stage,
            )
        )

    base_seed = zlib.adler32(str(ligand_file).encode("utf-8"))
    base_key = jax.random.fold_in(jax.random.PRNGKey(42), base_seed)
    formal_status = "DQ-Dock"
    best_result: BenchmarkResult | None = None
    best_pose_coords: np.ndarray | None = None
    execution_path = BlindDockingExecutionPath.GENERIC_PIPELINE
    certified_failure_reason = None
    theorem_handles: tuple[str, ...] = ()

    for attempt in range(max_retries):
        execution_path = BlindDockingExecutionPath.GENERIC_PIPELINE
        certified_failure_reason = None
        theorem_handles = ()
        try:
            if retry_preserve_seed:
                attempt_key = base_key
            else:
                attempt_key = jax.random.fold_in(base_key, attempt)
            key_parts = jax.random.key_data(attempt_key)
            k0 = int(key_parts[0])
            k1 = int(key_parts[1])
            backend_name = (
                optimizer_backend.name if optimizer_backend is not None else "GRADIENT"
            )
            print(
                f"    [Debug] {ligand_file.stem} attempt {attempt + 1}/{max_retries} seed={k0:08x}-{k1:08x} backend={backend_name} n_poses={n_poses} n_opt_steps={n_opt_steps}",
                flush=True,
            )

            execution = execute_attempt(attempt_key)
            best_poses, cert = execution.poses, execution.certification
            execution_path = execution.execution_path
            certified_failure_reason = execution.certified_failure_reason
            theorem_handles = execution.theorem_handles

            if not best_poses:
                if attempt < max_retries - 1:
                    n_poses *= 2
                    n_opt_steps *= 2
                    print(
                        f"    [Retry {attempt + 2}/{max_retries}] n_poses={n_poses}, n_opt_steps={n_opt_steps} (no poses)"
                    )
                    continue
                return finalize_attempt_failure(
                    build_failure_result(
                        error="ValueError: No poses returned by docking pipeline"
                    )
                )

            elapsed = time.time() - start
            if not best_poses:
                raise ValueError("No poses returned by docking pipeline")
            best_pose = best_poses[0]
            pose_jnp = jnp.expand_dims(best_pose.coords, axis=0)
            native_jnp = jnp.array(ligand_coords)
            rmsd = float(compute_docking_rmsd_batched(pose_jnp, native_jnp)[0])

            attempt_result = BenchmarkResult.from_certification(
                pose_energy=best_pose.energy,
                pose_rmsd=rmsd,
                elapsed=elapsed,
                n_atoms=len(pocket_coords) + len(ligand_coords),
                formal_status=formal_status,
                cert=cert,
                execution_path=execution_path,
                certified_failure_reason=certified_failure_reason,
                theorem_handles=theorem_handles,
            )

            if best_result is None or attempt_result.rmsd < best_result.rmsd:
                best_result = attempt_result
                best_pose_coords = np.asarray(best_pose.coords)

            # If we reached an acceptably low RMSD, break early
            if retry_break_rmsd and attempt_result.rmsd <= retry_break_rmsd:
                print(
                    f"    [Info] Breaking retries: RMSD {attempt_result.rmsd:.2f} <= {retry_break_rmsd:.2f}"
                )
                return best_result, best_pose_coords

            if attempt < max_retries - 1:
                n_poses *= 2
                n_opt_steps *= 2
                print(
                    f"    [Retry {attempt + 2}/{max_retries}] RMSD={rmsd:.2f}A, n_poses={n_poses}, n_opt_steps={n_opt_steps}"
                )
                continue
        except Exception as e:
            if (
                docking_config.mode == DockingMode.CERTIFIED
                and use_pocket_guided
                and "Certified blind docking could not derive a theorem-backed pocket/binding-site plan"
                in str(e)
            ):
                reason = CertifiedPocketFailureReason.NO_CERTIFIED_POCKET
                if "NO_LOCAL_REGION" in str(e):
                    reason = CertifiedPocketFailureReason.NO_LOCAL_REGION
                return finalize_attempt_failure(
                    build_failure_result(
                        error=_format_benchmark_error(e),
                        execution_path_override=BlindDockingExecutionPath.STRICT_CERTIFIED,
                        certified_failure_reason_override=reason,
                        theorem_handles_override=(),
                    )
                )
            if attempt < max_retries - 1:
                import traceback

                traceback.print_exc()
                n_poses *= 2
                n_opt_steps *= 2
                print(
                    f"    [Retry {attempt + 2}/{max_retries}] n_poses={n_poses}, n_opt_steps={n_opt_steps} (exception: {e})"
                )
                continue
            return finalize_attempt_failure(
                build_failure_result(error=_format_benchmark_error(e))
            )

    if best_result is None:
        return (
            build_failure_result(
                error="RuntimeError: Docking failed without producing a result",
                execution_path_override=BlindDockingExecutionPath.GENERIC_PIPELINE,
            ),
            None,
        )

    return best_result, best_pose_coords


EngineRunT = TypeVar("EngineRunT")


class BenchmarkEngine(ABC, Generic[EngineRunT]):
    engine_id: str
    display_name: str
    include_in_competitors: bool = False

    def announce(self) -> None:
        return None

    def is_available(self) -> bool:
        return True

    @abstractmethod
    def section_title(self, context: BenchmarkExecutionContext) -> str:
        """Human-readable section title for this engine."""

    def _print_section_header(self, context: BenchmarkExecutionContext) -> None:
        print("\n" + "=" * 70, flush=True)
        print(self.section_title(context), flush=True)
        print("=" * 70, flush=True)

    def run_all(
        self,
        complexes: Sequence[PreparedBenchmarkComplex],
        state: BenchmarkState,
        context: BenchmarkExecutionContext,
        excluded_rows: list[ExcludedComplexRow],
    ) -> None:
        if not self.is_available():
            return
        self._print_section_header(context)
        for run_result in self.iter_run_results(complexes, context):
            self.record_run_result(run_result, state, context, excluded_rows)

    def iter_run_results(
        self,
        complexes: Sequence[PreparedBenchmarkComplex],
        context: BenchmarkExecutionContext,
    ) -> Iterator[EngineRunT]:
        for complex_entry in complexes:
            yield self.run_single(complex_entry, context)

    @abstractmethod
    def run_single(
        self,
        complex_entry: PreparedBenchmarkComplex,
        context: BenchmarkExecutionContext,
    ) -> EngineRunT:
        """Run one complex and return the engine-specific result record."""

    @abstractmethod
    def record_run_result(
        self,
        run_result: EngineRunT,
        state: BenchmarkState,
        context: BenchmarkExecutionContext,
        excluded_rows: list[ExcludedComplexRow],
    ) -> None:
        """Fold a completed result into benchmark state."""

    @abstractmethod
    def print_summary(
        self,
        state: BenchmarkState,
        complexes: Sequence[PreparedBenchmarkComplex],
    ) -> None:
        """Print summary lines for this engine."""


@dataclass(frozen=True)
class DQDockBenchmarkJob(DQDockExecutionOptions):
    complex_entry: PreparedBenchmarkComplex
    precomputed_ligand_charges: np.ndarray | None
    precomputed_receptor_charges: np.ndarray | None


@dataclass(frozen=True)
class DQDockBenchmarkJobResult:
    complex_entry: PreparedBenchmarkComplex
    result: BenchmarkResult
    dq_pose_coords: np.ndarray | None


def run_dq_dock_benchmark_job(job: DQDockBenchmarkJob) -> DQDockBenchmarkJobResult:
    result, dq_pose_coords = run_dq_dock(
        job.complex_entry.pocket_coords,
        job.complex_entry.pocket_radii,
        job.complex_entry.ligand_coords,
        job.complex_entry.ligand_radii,
        job.complex_entry.center,
        ligand_file=job.complex_entry.ligand_pdb,
        receptor_file=job.complex_entry.pocket_receptor_pdb,
        charge_method=job.charge_method,
        precomputed_ligand_charges=job.precomputed_ligand_charges,
        precomputed_receptor_charges=job.precomputed_receptor_charges,
        n_poses=job.n_poses,
        n_opt_steps=job.n_opt_steps,
        top_k_to_optimize=job.top_k_to_optimize,
        use_rich_exact_rescoring=job.use_rich_exact_rescoring,
        use_multi_stage=job.use_multi_stage,
        use_pocket_guided=job.use_pocket_guided,
        ligand_elements=job.complex_entry.ligand_elements,
        receptor_elements=job.complex_entry.pocket_elements,
        optimizer_backend=job.optimizer_backend,
        formal_round_strategy=job.formal_round_strategy,
        certified_scoring_family=job.certified_scoring_family,
        max_retries=job.max_retries,
        retry_break_rmsd=job.retry_break_rmsd,
        retry_preserve_seed=job.retry_preserve_seed,
        box_size=job.box_size,
    )
    return DQDockBenchmarkJobResult(
        complex_entry=job.complex_entry,
        result=result,
        dq_pose_coords=dq_pose_coords,
    )


def _dq_job_worker(
    job: DQDockBenchmarkJob,
    queue: mp.Queue,
) -> None:
    try:
        queue.put(run_dq_dock_benchmark_job(job))
    except BaseException as exc:  # pragma: no cover - subprocess boundary
        queue.put(exc)


def _dq_job_worker_loop(
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    while True:
        job = job_queue.get()
        if job is None:
            return
        try:
            result_queue.put(("result", run_dq_dock_benchmark_job(job)))
        except BaseException as exc:  # pragma: no cover - subprocess boundary
            result_queue.put(("error", exc))


def _timed_out_dq_job_result(
    job: DQDockBenchmarkJob,
    timeout_seconds: float,
) -> DQDockBenchmarkJobResult:
    return DQDockBenchmarkJobResult(
        complex_entry=job.complex_entry,
        result=BenchmarkResult(
            success=False,
            energy=0.0,
            rmsd=0.0,
            time=timeout_seconds,
            n_atoms=0,
            formal_status=f"timeout after {timeout_seconds:.1f}s",
            error=f"timeout after {timeout_seconds:.1f}s",
        ),
        dq_pose_coords=None,
    )


def run_dq_dock_benchmark_job_with_timeout(
    job: DQDockBenchmarkJob,
    timeout_seconds: float,
) -> DQDockBenchmarkJobResult:
    ctx = mp.get_context(DEFAULT_BENCHMARK_PROCESS_START_METHOD)
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_dq_job_worker, args=(job, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        return _timed_out_dq_job_result(job, timeout_seconds)
    payload = queue.get() if not queue.empty() else None
    if isinstance(payload, BaseException):
        raise payload
    if payload is None:
        raise RuntimeError("DQ-Dock timeout worker exited without returning a result")
    return payload


class _PersistentDQDockTimeoutWorker:
    def __init__(self, ctx: Any):
        self._ctx = ctx
        self._job_queue: mp.Queue = ctx.Queue()
        self._result_queue: mp.Queue = ctx.Queue()
        self._process: mp.Process | None = None
        self._current_job: DQDockBenchmarkJob | None = None
        self._job_started_at: float | None = None
        self.start()

    @property
    def current_job(self) -> DQDockBenchmarkJob | None:
        return self._current_job

    @property
    def job_started_at(self) -> float | None:
        return self._job_started_at

    @property
    def is_idle(self) -> bool:
        return self._current_job is None

    def start(self) -> None:
        process = self._ctx.Process(
            target=_dq_job_worker_loop,
            args=(self._job_queue, self._result_queue),
        )
        process.start()
        self._process = process

    def restart(self) -> None:
        self.stop()
        self._job_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._current_job = None
        self._job_started_at = None
        self.start()

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.is_alive():
            try:
                self._job_queue.put(None)
                self._process.join(timeout=1.0)
            except BaseException:
                pass
        if self._process.is_alive():
            self._process.terminate()
            self._process.join()
        self._process = None

    def submit(self, job: DQDockBenchmarkJob) -> None:
        if not self.is_idle:
            raise RuntimeError("cannot submit a new job to a busy timeout worker")
        self._current_job = job
        self._job_started_at = time.monotonic()
        self._job_queue.put(job)

    def poll_result(self) -> DQDockBenchmarkJobResult | BaseException | None:
        try:
            kind, payload = self._result_queue.get_nowait()
        except queue.Empty:
            return None
        self._current_job = None
        self._job_started_at = None
        if kind == "error":
            return cast(BaseException, payload)
        return cast(DQDockBenchmarkJobResult, payload)

    def has_timed_out(self, timeout_seconds: float) -> bool:
        if self._current_job is None or self._job_started_at is None:
            return False
        return (time.monotonic() - self._job_started_at) > timeout_seconds


def iter_dq_dock_benchmark_jobs_with_persistent_timeout(
    jobs: Sequence[DQDockBenchmarkJob],
    *,
    timeout_seconds: float,
    parallelism: BenchmarkParallelism,
) -> Iterator[DQDockBenchmarkJobResult]:
    ctx = mp.get_context(DEFAULT_BENCHMARK_PROCESS_START_METHOD)
    worker_count = parallelism.max_workers if parallelism.is_parallel else 1
    workers = [_PersistentDQDockTimeoutWorker(ctx) for _ in range(worker_count)]
    pending_jobs = iter(jobs)
    active_workers = 0

    def try_submit(worker: _PersistentDQDockTimeoutWorker) -> bool:
        nonlocal active_workers
        if not worker.is_idle:
            return False
        try:
            job = next(pending_jobs)
        except StopIteration:
            return False
        worker.submit(job)
        active_workers += 1
        return True

    try:
        for worker in workers:
            try_submit(worker)
        while active_workers > 0:
            progressed = False
            for worker in workers:
                payload = worker.poll_result()
                if payload is not None:
                    active_workers -= 1
                    progressed = True
                    if isinstance(payload, BaseException):
                        raise payload
                    yield payload
                    try_submit(worker)
                    continue
                current_job = worker.current_job
                if current_job is not None and worker.has_timed_out(timeout_seconds):
                    active_workers -= 1
                    progressed = True
                    timed_out_job = current_job
                    worker.restart()
                    yield _timed_out_dq_job_result(timed_out_job, timeout_seconds)
                    try_submit(worker)
            if not progressed:
                time.sleep(0.05)
    finally:
        for worker in workers:
            worker.stop()


class DQDockBenchmarkEngine(BenchmarkEngine[DQDockBenchmarkJobResult]):
    engine_id = "dq_dock"
    display_name = "DQ-Dock"

    def section_title(self, context: BenchmarkExecutionContext) -> str:
        mode_label = (
            "multi-stage composite" if context.use_multi_stage else "random picking"
        )
        if context.use_pocket_guided and not context.use_multi_stage:
            mode_label = "pocket-guided picking"
        worker_label = (
            f", {context.parallelism.max_workers} workers"
            if context.parallelism.is_parallel
            else ""
        )
        return f"RUNNING DQ-DOCK ({context.n_poses} poses, {context.n_opt_steps} opt steps, {mode_label}{worker_label})"

    def iter_run_results(
        self,
        complexes: Sequence[PreparedBenchmarkComplex],
        context: BenchmarkExecutionContext,
    ) -> Iterator[DQDockBenchmarkJobResult]:
        jobs = tuple(
            self._build_job(complex_entry, context) for complex_entry in complexes
        )
        timeout_seconds = context.attempt_timeout_seconds
        if timeout_seconds is not None:
            yield from iter_dq_dock_benchmark_jobs_with_persistent_timeout(
                jobs,
                timeout_seconds=timeout_seconds,
                parallelism=context.parallelism,
            )
            return
        for job_result in execute_benchmark_jobs(
            jobs,
            run_job=run_dq_dock_benchmark_job,
            parallelism=context.parallelism,
        ):
            yield job_result

    def run_single(
        self,
        complex_entry: PreparedBenchmarkComplex,
        context: BenchmarkExecutionContext,
    ) -> DQDockBenchmarkJobResult:
        return run_dq_dock_benchmark_job(self._build_job(complex_entry, context))

    def record_run_result(
        self,
        run_result: DQDockBenchmarkJobResult,
        state: BenchmarkState,
        context: BenchmarkExecutionContext,
        excluded_rows: list[ExcludedComplexRow],
    ) -> None:
        complex_entry = run_result.complex_entry
        result = run_result.result
        dq_pose_coords = run_result.dq_pose_coords
        print(f"\n{complex_entry.pdb_id}:", flush=True)
        print(
            f"  Ligand atoms: {len(complex_entry.ligand_coords)}, Pocket atoms: {len(complex_entry.pocket_coords)}",
            flush=True,
        )
        pose_save_start = time.perf_counter()
        dq_pose_path = self._save_pose(complex_entry, context, dq_pose_coords)
        pose_save_elapsed = time.perf_counter() - pose_save_start

        dock_time = (
            result.dock_time_s if result.dock_time_s is not None else result.time
        )
        state.dq_results.append(result)
        state.dq_rows.append(
            ComplexSummaryRow(
                pdb_id=complex_entry.pdb_id,
                target_name=complex_entry.target_name,
                ligand_atoms=len(complex_entry.ligand_coords),
                pocket_atoms=len(complex_entry.pocket_coords),
                rmsd=result.rmsd if result.success else None,
                time=result.time,
                dock_time=dock_time,
                dq_pose_pdb=None if dq_pose_path is None else str(dq_pose_path),
                receptor_pdb=str(complex_entry.receptor_view_pdb),
                native_ligand_pdb=str(complex_entry.native_ligand_view_pdb),
            )
        )

        checkpoint_start = time.perf_counter()
        state.save(
            "dq_dock",
            context,
            excluded_rows=excluded_rows,
            render_report=True,
        )
        checkpoint_elapsed = time.perf_counter() - checkpoint_start
        wall_time = dock_time + pose_save_elapsed + checkpoint_elapsed
        finalized_result = replace(
            result,
            time=wall_time,
            dock_time_s=dock_time,
            checkpoint_time_s=checkpoint_elapsed,
            total_wall_time_s=wall_time,
        )
        state.dq_results[-1] = finalized_result
        state.dq_rows[-1] = replace(
            state.dq_rows[-1],
            time=wall_time,
            dock_time=dock_time,
        )

        if finalized_result.success:
            print(
                f"  Best Energy: {finalized_result.energy:.2f} kcal/mol, Sampled Pose RMSD: {finalized_result.rmsd:.2f}A, Dock: {dock_time:.2f}s, Wall: {wall_time:.2f}s"
            )
            print("  DQ-Dock Redocking", end="")
            gap_proof = format_gap_proof_label(finalized_result.certified)
            if gap_proof is not None:
                print(f", Gap Proof: {gap_proof}", end="")
                if finalized_result.native_rank is not None:
                    print(f", Native Rank: {finalized_result.native_rank}", end="")
                if finalized_result.energy_gap is not None:
                    print(f", Energy Gap: {finalized_result.energy_gap:.4f}", end="")
            print()
        else:
            print(
                f"  DQ-Dock failed, Dock: {dock_time:.2f}s, Wall: {wall_time:.2f}s, Error: {finalized_result.error or finalized_result.formal_status}"
            )
        print(f"  Checkpoint save time: {checkpoint_elapsed:.2f}s", flush=True)
        if finalized_result.native_rank is not None:
            print(
                "  Gap proof compares native pre-optimization energy against sampled poses.",
                flush=True,
            )

    def _build_job(
        self,
        complex_entry: PreparedBenchmarkComplex,
        context: BenchmarkExecutionContext,
    ) -> DQDockBenchmarkJob:
        return DQDockBenchmarkJob(
            complex_entry=complex_entry,
            charge_method=complex_entry.charge_method,
            precomputed_ligand_charges=complex_entry.ligand_charges,
            precomputed_receptor_charges=complex_entry.pocket_charges,
            n_poses=context.n_poses,
            n_opt_steps=context.n_opt_steps,
            top_k_to_optimize=context.top_k_to_optimize,
            use_multi_stage=context.use_multi_stage,
            use_pocket_guided=context.use_pocket_guided,
            use_rich_exact_rescoring=context.use_rich_exact_rescoring,
            optimizer_backend=context.optimizer_backend,
            formal_round_strategy=context.formal_round_strategy,
            certified_scoring_family=context.certified_scoring_family,
            max_retries=context.max_retries,
            retry_break_rmsd=context.retry_break_rmsd,
            retry_preserve_seed=context.retry_preserve_seed,
            box_size=context.box_size,
        )

    def _save_pose(
        self,
        complex_entry: PreparedBenchmarkComplex,
        context: BenchmarkExecutionContext,
        dq_pose_coords: np.ndarray | None,
    ) -> Path | None:
        dq_pose_path: Path | None = None
        if dq_pose_coords is not None:
            dq_pose_path = context.pose_dir / f"{complex_entry.pdb_id}_dq_dock_pose.pdb"
            save_pose_from_template(
                dq_pose_coords, complex_entry.ligand_pdb, dq_pose_path
            )
        return dq_pose_path

    def print_summary(
        self,
        state: BenchmarkState,
        complexes: Sequence[PreparedBenchmarkComplex],
    ) -> None:
        if state.dq_rows:
            print("\nDQ-Dock Per-Complex")
            print("-" * 70)
            print(
                f"{'PDB':<8} {'Ligand':<8} {'Pocket':<8} {'Status':<10} {'RMSD':<10} {'Dock':<10} {'Wall':<10}"
            )
            for row, result in zip(state.dq_rows, state.dq_results):
                rmsd_text = "n/a" if row.rmsd is None else f"{row.rmsd:.2f}"
                print(
                    f"{row.pdb_id:<8} {row.ligand_atoms:<8} {row.pocket_atoms:<8} {('success' if result.success else 'failure'):<10} {rmsd_text:<10} {(row.dock_time or float('nan')):<10.2f} {row.time:<10.2f}"
                )

        if state.dq_results:
            successful_results = [
                result for result in state.dq_results if result.success
            ]
            avg_time = np.mean([r.time for r in state.dq_results])
            avg_dock_time = np.mean([r.dock_time_s or r.time for r in state.dq_results])
            min_time = np.min([r.time for r in state.dq_results])
            max_time = np.max([r.time for r in state.dq_results])
            dq_total = sum(r.time for r in state.dq_results)
            dq_total_dock = sum(r.dock_time_s or r.time for r in state.dq_results)
            print(
                f"{self.display_name:<20} {avg_time:<12.2f} {dq_total:<12.2f} {len(successful_results)}/{len(state.dq_results)}"
            )
            if successful_results:
                avg_rmsd = np.mean([r.rmsd for r in successful_results])
                print(f"  Avg RMSD: {avg_rmsd:.2f}A")
            else:
                print("  Avg RMSD: n/a (no successful DQ-Dock runs)")
            print(
                f"  Avg dock time: {avg_dock_time:.2f}s, Total dock time: {dq_total_dock:.2f}s"
            )
            print(f"  Time range: {min_time:.2f}s - {max_time:.2f}s per complex")


class CLIDockingBenchmarkEngine(BenchmarkEngine[CompetitorResultRow]):
    include_in_competitors = True

    def __init__(
        self,
        binary_path: str | None,
        input_preparation_strategies: tuple[DockingPreparationStrategy, ...],
    ):
        self.binary_path = binary_path
        self.input_preparation_strategies = input_preparation_strategies

    @classmethod
    @abstractmethod
    def candidate_binary_names(cls) -> tuple[str, ...]:
        """Ordered binary candidates for this engine."""

    @classmethod
    @abstractmethod
    def score_name(cls) -> str:
        """Primary score label reported by this engine."""

    @classmethod
    @abstractmethod
    def installation_message(cls) -> str:
        """Help text when the external engine is missing."""

    @classmethod
    @abstractmethod
    def preparation_strategies(cls) -> tuple[DockingPreparationStrategy, ...]:
        """Ordered input preparation strategies for this engine."""

    @classmethod
    def build(cls) -> "CLIDockingBenchmarkEngine":
        return cls(
            find_docking_binary(cls.candidate_binary_names()),
            cls.preparation_strategies(),
        )

    def metadata(self) -> CompetitorMetadata:
        return CompetitorMetadata(
            engine_id=self.engine_id,
            display_name=self.display_name,
            score_name=self.score_name(),
            prep_protocol=self.preparation_protocol_label(),
        )

    def preparation_protocol_label(self) -> str:
        labels = [
            strategy.protocol_label() for strategy in self.input_preparation_strategies
        ]
        return labels[0] if len(labels) == 1 else "_fallback_".join(labels)

    def is_available(self) -> bool:
        return self.binary_path is not None

    def announce(self) -> None:
        if self.binary_path is None:
            print(self.installation_message())
            return
        print(f"\n✅ {self.display_name} found: {self.binary_path}")
        print(
            f"✅ Input preparation order: {', '.join(strategy.protocol_label() for strategy in self.input_preparation_strategies)}"
        )

    def section_title(self, context: BenchmarkExecutionContext) -> str:
        return f"RUNNING {self.display_name.upper()}"

    @abstractmethod
    def build_command(
        self,
        prepared_inputs: PreparedDockingInputs,
        complex_entry: PreparedBenchmarkComplex,
        output_path: Path,
    ) -> list[str]:
        """Build the CLI command for one docking run."""

    def run_single(
        self,
        complex_entry: PreparedBenchmarkComplex,
        context: BenchmarkExecutionContext,
    ) -> CompetitorResultRow:
        if self.binary_path is None:
            return CompetitorResultRow(
                engine_id=self.engine_id,
                display_name=self.display_name,
                score_name=self.score_name(),
                pdb_id=complex_entry.pdb_id,
                target_name=complex_entry.target_name,
                success=False,
                time=0.0,
                error=f"{self.display_name} is unavailable",
            )
        print(f"\n{complex_entry.pdb_id}...", flush=True)

        pose_output_path = (
            context.pose_dir / f"{complex_entry.pdb_id}_{self.engine_id}_pose.pdb"
        )

        final_row: CompetitorResultRow | None = None
        for strategy in self.input_preparation_strategies:
            cleanup_dir: Path | None = None
            try:
                prepared_inputs = strategy.prepare_inputs(
                    complex_entry.receptor_pdb,
                    complex_entry.ligand_pdb,
                )
                cleanup_dir = prepared_inputs.cleanup_dir
                print(f"  Receptor: {prepared_inputs.receptor_path.name}", flush=True)
                print(f"  Ligand: {prepared_inputs.ligand_path.name}", flush=True)
                command = self.build_command(
                    prepared_inputs,
                    complex_entry,
                    pose_output_path,
                )
                result = run_cli_docking(
                    command,
                    pose_output_path,
                    timeout_seconds=context.attempt_timeout_seconds,
                )
                if not result.success or result.score is None:
                    error_msg = result.error or f"{self.display_name} run failed"
                    print(
                        f"  {prepared_inputs.protocol_label} attempt failed ({error_msg})",
                        flush=True,
                    )
                    final_row = CompetitorResultRow(
                        engine_id=self.engine_id,
                        display_name=self.display_name,
                        score_name=self.score_name(),
                        pdb_id=complex_entry.pdb_id,
                        target_name=complex_entry.target_name,
                        success=False,
                        time=result.time,
                        error=error_msg,
                    )
                    continue

                top_rmsd = float("nan")
                best_mode_rmsd = float("nan")
                if result.top_coords is not None:
                    pose_coords = result.top_coords
                    if len(pose_coords) != len(complex_entry.ligand_coords):
                        print(
                            f"  ⚠️  Atom count mismatch: Native={len(complex_entry.ligand_coords)}, {self.display_name}={len(pose_coords)}"
                        )
                    top_rmsd = compute_pose_rmsd(
                        pose_coords,
                        complex_entry.ligand_coords,
                    )
                if result.model_coords:
                    model_rmsds = [
                        compute_pose_rmsd(model_coords, complex_entry.ligand_coords)
                        for model_coords in result.model_coords
                    ]
                    finite_model_rmsds = [r for r in model_rmsds if not np.isnan(r)]
                    if finite_model_rmsds:
                        best_mode_rmsd = min(finite_model_rmsds)

                print(
                    f"  {self.score_name().title()}: {result.score:.2f}, Top Pose RMSD: {top_rmsd:.2f}A, Best Returned Mode RMSD: {best_mode_rmsd:.2f}A, Time: {result.time:.1f}s"
                )
                final_row = CompetitorResultRow(
                    engine_id=self.engine_id,
                    display_name=self.display_name,
                    score_name=self.score_name(),
                    pdb_id=complex_entry.pdb_id,
                    target_name=complex_entry.target_name,
                    success=True,
                    time=result.time,
                    score=result.score,
                    top_rmsd=top_rmsd,
                    best_mode_rmsd=best_mode_rmsd,
                    pose_pdb=str(pose_output_path),
                )
                break
            finally:
                if cleanup_dir is not None:
                    shutil.rmtree(cleanup_dir, ignore_errors=True)

        if final_row is None:
            final_row = CompetitorResultRow(
                engine_id=self.engine_id,
                display_name=self.display_name,
                score_name=self.score_name(),
                pdb_id=complex_entry.pdb_id,
                target_name=complex_entry.target_name,
                success=False,
                time=0.0,
                error=f"{self.display_name} produced no result",
            )
        return final_row

    def record_run_result(
        self,
        run_result: CompetitorResultRow,
        state: BenchmarkState,
        context: BenchmarkExecutionContext,
        excluded_rows: list[ExcludedComplexRow],
    ) -> None:
        state.competitor_rows.append(run_result)
        state.save(
            self.engine_id,
            context,
            excluded_rows,
            render_report=True,
        )

    def print_summary(
        self,
        state: BenchmarkState,
        complexes: Sequence[PreparedBenchmarkComplex],
    ) -> None:
        rows = [row for row in state.competitor_rows if row.engine_id == self.engine_id]
        if not rows:
            return
        successful_rows = [row for row in rows if row.success]
        successful_times = [row.time for row in successful_rows]
        avg_time = (
            float(np.mean(successful_times)) if successful_times else float("nan")
        )
        total_time = sum(successful_times)
        print(
            f"{self.display_name:<20} {avg_time:<12.1f} {total_time:<12.1f} {len(successful_rows)}/{len(rows)}"
        )
        if successful_rows:
            avg_top_rmsd = np.mean(
                [cast(float, row.top_rmsd) for row in successful_rows]
            )
            avg_best_mode_rmsd = np.mean(
                [cast(float, row.best_mode_rmsd) for row in successful_rows]
            )
            print(f"  Avg top-pose RMSD: {avg_top_rmsd:.2f}A")
            print(f"  Avg best-returned RMSD: {avg_best_mode_rmsd:.2f}A")


class VinaLikeBenchmarkEngine(CLIDockingBenchmarkEngine):
    @classmethod
    def preparation_strategies(cls) -> tuple[DockingPreparationStrategy, ...]:
        return (DirectPDBPreparationStrategy(),)

    @classmethod
    def score_name(cls) -> str:
        return "affinity"

    def build_command(
        self,
        prepared_inputs: PreparedDockingInputs,
        complex_entry: PreparedBenchmarkComplex,
        output_path: Path,
    ) -> list[str]:
        assert self.binary_path is not None
        return [
            self.binary_path,
            "--receptor",
            str(prepared_inputs.receptor_path),
            "--ligand",
            str(prepared_inputs.ligand_path),
            "--center_x",
            str(complex_entry.center[0]),
            "--center_y",
            str(complex_entry.center[1]),
            "--center_z",
            str(complex_entry.center[2]),
            "--size_x",
            str(complex_entry.box_size[0]),
            "--size_y",
            str(complex_entry.box_size[1]),
            "--size_z",
            str(complex_entry.box_size[2]),
            "--exhaustiveness",
            "64",
            "--num_modes",
            "20",
            *self.additional_cli_flags(),
            "--min_rmsd_filter",
            "0.5",
            "--seed",
            "42",
        ]

    def additional_cli_flags(self) -> tuple[str, ...]:
        return ()


class SminaBenchmarkEngine(VinaLikeBenchmarkEngine):
    engine_id = "smina"
    display_name = "Vina"

    @classmethod
    def candidate_binary_names(cls) -> tuple[str, ...]:
        return ("smina", "vina", "autodock_vina")

    @classmethod
    def installation_message(cls) -> str:
        return """
❌ Vina not found!

To run this benchmark with real Vina comparisons:

1. Install Vina:
   conda install -c conda-forge vina

   Or download from:
   https://github.com/ccsb-scripps/AutoDock-Vina/releases

2. Make sure 'vina' is in your PATH

3. Re-run this benchmark

For now, we'll run DQ-Dock only on PDB files.
"""

    def section_title(self, context: BenchmarkExecutionContext) -> str:
        return "RUNNING SMINA/VINA"

    def additional_cli_flags(self) -> tuple[str, ...]:
        return ("--energy_range", "12.0")


class GninaBenchmarkEngine(VinaLikeBenchmarkEngine):
    engine_id = "gnina"
    display_name = "GNINA"

    @classmethod
    def candidate_binary_names(cls) -> tuple[str, ...]:
        return ("gnina",)

    @classmethod
    def installation_message(cls) -> str:
        return """
GNINA not found or not functional.

To run this benchmark with GNINA comparisons:

1. Download the CUDA 12 build: gnina v1.3.2 from github.com/gnina/gnina/releases
2. Install CUDA 12 runtimes into a venv (see setup instructions)
3. Re-run this benchmark
"""


def run_benchmark(
    n_complexes: int = 10,
    charge_method: ChargeMethod = DEFAULT_BENCHMARK_CHARGE_METHOD,
    n_poses: int = 2000,
    n_opt_steps: int = 50,
    top_k_to_optimize: int = DEFAULT_BENCHMARK_TOP_K_TO_OPTIMIZE,
    box_size: float = BENCHMARK_PROTOCOL.box_size_a,
    formal_round_strategy: FormalRoundStrategy = DEFAULT_BENCHMARK_FORMAL_ROUND_STRATEGY,
    certified_scoring_family: CertifiedScoringFamily = DEFAULT_BENCHMARK_CERTIFIED_SCORING_FAMILY,
    use_multi_stage: bool = DEFAULT_BENCHMARK_USE_MULTI_STAGE,
    use_pocket_guided: bool = DEFAULT_BENCHMARK_USE_POCKET_GUIDED,
    use_rich_exact_rescoring: bool = True,
    results_dir: Path = Path("benchmark_results"),
    competitors: Literal["all", "none", "smina", "gnina"] = "all",
    dataset: Literal["curated", "casf2007"] = "casf2007",
    pdb_ids: Sequence[str] = (),
    optimizer_backend: OptimizerBackend = DEFAULT_BENCHMARK_OPTIMIZER_BACKEND,
    parallelism: BenchmarkParallelism = BenchmarkParallelism(),
    attempt_timeout_seconds: float | None = None,
    max_retries: int = 6,
    retry_break_rmsd: float = 0.0,
    retry_preserve_seed: bool = False,
):
    """Run full benchmark."""
    bench_start = time.time()

    print("=" * 70, flush=True)
    print("REAL PDB DOCKING BENCHMARK", flush=True)
    print("=" * 70, flush=True)

    _COMPETITOR_ENGINES: dict[str, type[CLIDockingBenchmarkEngine]] = {
        "smina": SminaBenchmarkEngine,
        "gnina": GninaBenchmarkEngine,
    }
    _selected = set(_COMPETITOR_ENGINES) if competitors == "all" else set()
    if competitors not in ("all", "none"):
        _selected = {competitors}

    engines: list[BenchmarkEngine] = [
        DQDockBenchmarkEngine(),
        *(_COMPETITOR_ENGINES[name].build() for name in _selected),
    ]
    for engine in engines:
        engine.announce()

    output_paths = create_benchmark_output_paths(results_dir)
    pose_dir = benchmark_pose_dir(output_paths)
    # Ensure pose output directory exists before any per-complex writes
    pose_dir.mkdir(parents=True, exist_ok=True)
    competitor_metadata = tuple(
        cast(CLIDockingBenchmarkEngine, engine).metadata()
        for engine in engines
        if engine.include_in_competitors and engine.is_available()
    )
    selection = derive_benchmark_target_selection(
        dataset=dataset,
        n_complexes=n_complexes,
        explicit_pdb_ids=pdb_ids,
    )
    cache_dir = selection.cache_dir

    print(
        f"\nPreparing {selection.label} ({selection.n_complexes_requested} complexes)...",
        flush=True,
    )
    print(f"Live Results JSON: {output_paths.json_path}", flush=True)
    print(f"Live Results CSV: {output_paths.csv_path}", flush=True)
    complexes: list[BenchmarkComplex] = []
    excluded_rows: list[ExcludedComplexRow] = []

    for entry in selection.entries:
        pdb_path = download_pdb(entry.pdb_id, cache_dir)
        if pdb_path is None:
            excluded_rows.append(
                ExcludedComplexRow(
                    pdb_id=entry.pdb_id,
                    reason="failed to download PDB from RCSB",
                )
            )
            continue

        hydrated_entry = hydrate_selected_benchmark_entry(
            entry,
            pdb_path,
            selection.mode,
        )
        try:
            protein_path = prepare_protein(pdb_path)
            screening = screen_complex(
                hydrated_entry,
                pdb_path,
                protein_path,
                charge_method=charge_method,
                certified_scoring_family=certified_scoring_family,
            )
            if not screening.accepted:
                reason = "; ".join(screening.reasons)
                excluded_rows.append(
                    ExcludedComplexRow(pdb_id=entry.pdb_id, reason=reason)
                )
                print(f"  {entry.pdb_id}: excluded ({reason})", flush=True)
                continue

            if screening.center is None:
                raise ValueError(
                    f"Screening did not provide a docking center for {entry.pdb_id}"
                )
            center = screening.center
        except ValueError as e:
            excluded_rows.append(ExcludedComplexRow(pdb_id=entry.pdb_id, reason=str(e)))
            print(f"  {entry.pdb_id}: excluded ({e})", flush=True)
            continue

        complexes.append(
            BenchmarkComplex(
                pdb_id=hydrated_entry.pdb_id,
                target_name=hydrated_entry.target_name,
                path=pdb_path,
                center=center,
                preferred_resname=hydrated_entry.preferred_resname,
                ligand_residue=screening.ligand_residue,
                receptor_system=screening.receptor_system,
                chemistry_plan=screening.chemistry_plan,
            )
        )
        print(
            f"  {entry.pdb_id}: center = ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})"
        )

    print(f"\nDownloaded {len(complexes)} complexes", flush=True)
    if excluded_rows:
        print(f"Excluded during QC: {len(excluded_rows)}", flush=True)

    prepared_complexes: list[PreparedBenchmarkComplex] = []
    for complex_entry in complexes:
        try:
            prepared_complexes.append(
                prepare_benchmark_complex(
                    complex_entry,
                    pose_dir,
                    box_size=box_size,
                    certified_scoring_family=certified_scoring_family,
                )
            )
        except (ValueError, RuntimeError, ImportError, FileNotFoundError) as exc:
            excluded_rows.append(
                ExcludedComplexRow(pdb_id=complex_entry.pdb_id, reason=str(exc))
            )
            print(f"  {complex_entry.pdb_id}: excluded ({exc})", flush=True)

    context = BenchmarkExecutionContext(
        charge_method=charge_method,
        certified_scoring_family=certified_scoring_family,
        parallelism=parallelism,
        complexes=tuple(prepared_complexes),
        n_complexes_requested=selection.n_complexes_requested,
        n_poses=n_poses,
        n_opt_steps=n_opt_steps,
        top_k_to_optimize=top_k_to_optimize,
        box_size=box_size,
        formal_round_strategy=formal_round_strategy,
        use_multi_stage=use_multi_stage,
        use_pocket_guided=use_pocket_guided,
        use_rich_exact_rescoring=use_rich_exact_rescoring,
        output_paths=output_paths,
        pose_dir=pose_dir,
        competitors=competitor_metadata,
        attempt_timeout_seconds=attempt_timeout_seconds,
        bench_start=bench_start,
        optimizer_backend=optimizer_backend,
        max_retries=max_retries,
        retry_break_rmsd=retry_break_rmsd,
        retry_preserve_seed=retry_preserve_seed,
    )
    state = BenchmarkState(dq_results=[], dq_rows=[], competitor_rows=[])
    state.save("curation", context, excluded_rows, render_report=True)

    if not prepared_complexes:
        print("❌ No complexes downloaded")
        return

    for engine in engines:
        engine.run_all(prepared_complexes, state, context, excluded_rows)

    bench_elapsed = context.elapsed()

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 70, flush=True)

    print(f"\n{'Method':<20} {'Avg Time':<12} {'Total Time':<12} {'Success Rate':<15}")
    print("-" * 60)

    if excluded_rows:
        print("\nExcluded By QC")
        print("-" * 70)
        for row in excluded_rows:
            print(f"{row.pdb_id:<8} {row.reason}")

    for engine in engines:
        engine.print_summary(state, prepared_complexes)

    print(f"\n  Total benchmark time: {bench_elapsed:.2f}s")
    print(
        f"  {sum(result.success for result in state.dq_results)}/{len(prepared_complexes)} complexes completed successfully"
    )
    print(f"  {len(excluded_rows)} complexes excluded by QC")

    json_path, csv_path = state.save(
        "complete",
        context,
        excluded_rows,
        render_report=True,
    )
    print(f"  Results JSON: {json_path}")
    print(f"  Results CSV: {csv_path}")
    print(f"  Pose Files: {pose_dir}")
    print(
        f"  Report Markdown: {json_path.with_suffix('').with_name(json_path.stem + '_report.md')}"
    )
    print(
        f"  RMSD Plot: {json_path.with_suffix('').with_name(json_path.stem + '_rmsd.png')}"
    )
    print(
        f"  Timing Plot: {json_path.with_suffix('').with_name(json_path.stem + '_timing.png')}"
    )
    print(
        f"  Timing Plot (Log): {json_path.with_suffix('').with_name(json_path.stem + '_timing_log.png')}"
    )
    print(
        f"  Scatter Plot: {json_path.with_suffix('').with_name(json_path.stem + '_scatter.png')}"
    )
    print(
        f"  Pose Comparison Figures: {json_path.with_suffix('').with_name(json_path.stem + '_pose_figures')}"
    )
    print(
        f"  PyMOL 3D Figures: {json_path.with_suffix('').with_name(json_path.stem + '_pymol')} (front: *_3d.png, side: *_3d_side.png)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real PDB docking benchmark")
    parser.add_argument(
        "--n_complexes", type=int, default=10, help="Number of complexes"
    )
    parser.add_argument(
        "--n_poses",
        type=int,
        default=2000,
        help="Number of sampled poses for DQ-Dock",
    )
    parser.add_argument(
        "--n_opt_steps",
        type=int,
        default=50,
        help="Number of optimization steps per pose",
    )
    parser.add_argument(
        "--top_k_to_optimize",
        type=int,
        default=DEFAULT_BENCHMARK_TOP_K_TO_OPTIMIZE,
        help="Number of certified survivors to optimize after pruning",
    )
    parser.add_argument(
        "--box-size",
        type=float,
        default=BENCHMARK_PROTOCOL.box_size_a,
        help="Certified cubic docking box size in Angstroms",
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Directory for CSV/JSON benchmark outputs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of DQ-Dock worker processes across complexes (default: 1)",
    )
    parser.add_argument(
        "--no_rich_exact_rescoring",
        action="store_false",
        dest="use_rich_exact_rescoring",
        help="Disable rich chemistry rescoring",
    )
    parser.add_argument(
        "--competitors",
        type=str,
        choices=("all", "none", "smina", "gnina"),
        default="all",
        help="Which competitor engines to run alongside the canonical certified DQ-Dock path",
    )
    parser.add_argument(
        "--attempt-timeout-seconds",
        type=float,
        default=None,
        help="Skip any docking attempt if it exceeds this timeout in seconds",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("curated", "casf2007"),
        default="casf2007",
        help="Which dataset to benchmark (default: casf2007). "
        "'casf2007' uses the 195-entry CASF-2007 core set.",
    )
    parser.add_argument(
        "--pdb_ids",
        type=str,
        default=None,
        help="Explicit PDB IDs to benchmark. Accepts comma or whitespace-separated IDs and overrides --dataset/--n_complexes.",
    )
    parser.add_argument(
        "--pdb_file",
        type=Path,
        default=None,
        help="Path to a text file listing explicit PDB IDs (comments with # supported). Overrides --dataset/--n_complexes.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum retry attempts for the canonical certified DQ-Dock run",
    )
    args = parser.parse_args()
    explicit_pdb_ids = load_explicit_pdb_ids(args.pdb_ids, args.pdb_file)

    is_valid, warnings = CERTIFIED_DOCKING.validate()
    for warning in warnings:
        print(f"Config warning: {warning}")

    print("Using canonical certified benchmark path:", flush=True)
    print(
        f"  charge_method={DEFAULT_BENCHMARK_CHARGE_METHOD.name.lower()} "
        f"optimizer={DEFAULT_BENCHMARK_OPTIMIZER_BACKEND.name.lower()} "
        f"formal_round_strategy={DEFAULT_BENCHMARK_FORMAL_ROUND_STRATEGY.value} "
        f"certified_scoring_family={DEFAULT_BENCHMARK_CERTIFIED_SCORING_FAMILY.value} "
        f"pocket_guided={DEFAULT_BENCHMARK_USE_POCKET_GUIDED} "
        f"top_k_to_optimize={args.top_k_to_optimize} "
        f"workers={args.workers}",
        flush=True,
    )

    run_benchmark(
        args.n_complexes,
        charge_method=DEFAULT_BENCHMARK_CHARGE_METHOD,
        n_poses=args.n_poses,
        n_opt_steps=args.n_opt_steps,
        top_k_to_optimize=args.top_k_to_optimize,
        box_size=args.box_size,
        formal_round_strategy=DEFAULT_BENCHMARK_FORMAL_ROUND_STRATEGY,
        certified_scoring_family=DEFAULT_BENCHMARK_CERTIFIED_SCORING_FAMILY,
        use_multi_stage=DEFAULT_BENCHMARK_USE_MULTI_STAGE,
        use_pocket_guided=DEFAULT_BENCHMARK_USE_POCKET_GUIDED,
        use_rich_exact_rescoring=args.use_rich_exact_rescoring,
        results_dir=args.results_dir,
        competitors=args.competitors,
        attempt_timeout_seconds=args.attempt_timeout_seconds,
        dataset=args.dataset,
        pdb_ids=explicit_pdb_ids,
        optimizer_backend=DEFAULT_BENCHMARK_OPTIMIZER_BACKEND,
        parallelism=BenchmarkParallelism(max_workers=args.workers),
        max_retries=args.max_retries,
    )
