"""Benchmark runner."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from tokenize import open as tokenize_open
from typing import Any, Iterable, Mapping

from benchmark.adapters.cellprofiler import CellProfilerAdapter
from benchmark.adapters.openhcs import OpenHCSAdapter
from benchmark.contracts.dataset import DatasetSpec
from benchmark.contracts.tool_adapter import BenchmarkResult, ToolAdapter
from benchmark.datasets.acquire import acquire_dataset
from benchmark.pipelines.registry import get_pipeline_spec


_BENCHMARK_CACHE_SCHEMA_VERSION = 1
_OPENHCS_CACHE_MANIFEST_NAME = ".openhcs_benchmark_cache.json"
_OPENHCS_EXECUTION_CACHE_MANIFEST_NAME = ".openhcs_runtime_execution_cache.json"
_SOURCE_CACHE_ROOTS = ("benchmark", "openhcs")
_SOURCE_CACHE_FILES = ("pyproject.toml",)
_SOURCE_CACHE_SUFFIXES = (".py", ".toml")
_SOURCE_CACHE_EXCLUDED_DIRS = (
    Path("benchmark") / "cellprofiler_source",
)
BENCHMARK_CACHE_DOMAINS = frozenset({"harness"})
_BENCHMARK_CACHE_DOMAINS_SYMBOL = "BENCHMARK_CACHE_DOMAINS"
_EXECUTION_SOURCE_CACHE_EXCLUDED_DOMAINS = frozenset(
    {"harness", "native_reference", "parity"}
)
_LEGACY_SOURCE_TREE_CACHE_KEY = "legacy_source_tree"
_EXECUTION_CACHE_IGNORED_PARAM_KEYS = frozenset(
    {
        "equivalence_reference_output_dir",
        "runtime_execution_cache_manifest",
        "runtime_execution_cache_key",
        "reuse_runtime_execution_cache",
    }
)
_TREE_METADATA_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class CellProfilerCompatibilityResult:
    """Native CellProfiler reference plus equivalent OpenHCS candidate result."""

    native_cellprofiler: BenchmarkResult
    openhcs_converted: BenchmarkResult

    @property
    def is_equivalent(self) -> bool:
        """Return whether the OpenHCS run reported zero semantic differences."""
        provenance = self.openhcs_converted.provenance or {}
        return (
            self.native_cellprofiler.success
            and self.openhcs_converted.success
            and provenance.get("equivalence_difference_count") == 0
        )


def run_benchmark(
    dataset_spec: DatasetSpec,
    tool_adapters: list[ToolAdapter],
    pipeline_name: str,
    metrics: Iterable,
) -> list[BenchmarkResult]:
    """
    Run benchmark across tools.

    1. Validate all tools
    2. Acquire dataset
    3. For each tool: run with metrics
    4. Return results
    """
    # Validate tools are installed
    for adapter in tool_adapters:
        adapter.validate_installation()

    acquired = acquire_dataset(dataset_spec)
    pipeline_spec = get_pipeline_spec(pipeline_name)

    # Merge pipeline parameters with dataset-specific context
    pipeline_params = {
        **pipeline_spec.parameters,
        "dataset_id": dataset_spec.id,
        "microscope_type": acquired.microscope_type,
    }

    results: list[BenchmarkResult] = []
    output_root = Path.cwd() / "benchmark_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    for adapter in tool_adapters:
        tool_output_dir = output_root / f"{adapter.name}_{dataset_spec.id}"
        tool_result = adapter.run(
            dataset_path=acquired.path,
            pipeline_name=pipeline_spec.name,
            pipeline_params=pipeline_params,
            metrics=list(metrics),
            output_dir=tool_output_dir,
        )
        results.append(tool_result)

    return results


def run_cellprofiler_compatibility_benchmark(
    dataset_spec: DatasetSpec,
    pipeline_name: str,
    metrics: Iterable,
    *,
    equivalence_reference_output_dir: Path | None = None,
    reuse_openhcs_cache: bool = True,
    cellprofiler_adapter: ToolAdapter | None = None,
    openhcs_adapter: ToolAdapter | None = None,
) -> CellProfilerCompatibilityResult:
    """Run native CellProfiler, then require OpenHCS converted output parity."""
    native_adapter = cellprofiler_adapter or CellProfilerAdapter()
    converted_adapter = openhcs_adapter or OpenHCSAdapter()
    if equivalence_reference_output_dir is None:
        native_adapter.validate_installation()

    acquired = acquire_dataset(dataset_spec)
    pipeline_spec = get_pipeline_spec(pipeline_name)
    pipeline_params = {
        **pipeline_spec.parameters,
        "dataset_id": dataset_spec.id,
        "microscope_type": acquired.microscope_type,
    }
    output_root = Path.cwd() / "benchmark_outputs"
    output_root.mkdir(parents=True, exist_ok=True)

    metric_collectors = list(metrics)
    if equivalence_reference_output_dir is None:
        native_result = native_adapter.run(
            dataset_path=acquired.path,
            pipeline_name=pipeline_spec.name,
            pipeline_params=pipeline_params,
            metrics=metric_collectors,
            output_dir=output_root / f"{native_adapter.name}_{dataset_spec.id}",
        )
    else:
        native_result = _cached_cellprofiler_reference_result(
            Path(equivalence_reference_output_dir),
            dataset_id=dataset_spec.id,
            pipeline_name=pipeline_spec.name,
            tool_name=native_adapter.name,
        )
    converted_params = {
        **pipeline_params,
        "equivalence_reference_output_dir": str(native_result.output_path),
    }
    converted_result = _run_or_reuse_cached_openhcs(
        converted_adapter,
        dataset_path=acquired.path,
        pipeline_name=pipeline_spec.name,
        pipeline_params=converted_params,
        metrics=metric_collectors,
        output_dir=output_root / f"{converted_adapter.name}_{dataset_spec.id}",
        reuse_cache=reuse_openhcs_cache,
    )
    return CellProfilerCompatibilityResult(
        native_cellprofiler=native_result,
        openhcs_converted=converted_result,
    )


def run_cellprofiler_cppipe_parity(
    dataset_path: Path,
    cppipe_path: Path,
    metrics: Iterable,
    *,
    dataset_id: str | None = None,
    pipeline_name: str | None = None,
    microscope_type: str | None = None,
    pipeline_params: Mapping[str, Any] | None = None,
    output_root: Path | None = None,
    equivalence_reference_output_dir: Path | None = None,
    reuse_openhcs_cache: bool = True,
    cellprofiler_adapter: ToolAdapter | None = None,
    openhcs_adapter: ToolAdapter | None = None,
) -> CellProfilerCompatibilityResult:
    """Run native CellProfiler, then require OpenHCS parity for one local .cppipe."""
    native_adapter = cellprofiler_adapter or CellProfilerAdapter()
    converted_adapter = openhcs_adapter or OpenHCSAdapter()
    if equivalence_reference_output_dir is None:
        native_adapter.validate_installation()

    resolved_dataset_path = Path(dataset_path)
    resolved_cppipe_path = Path(cppipe_path)
    resolved_dataset_id = dataset_id or resolved_dataset_path.name
    resolved_pipeline_name = pipeline_name or resolved_cppipe_path.stem
    resolved_output_root = output_root or Path.cwd() / "benchmark_outputs"
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    base_params: dict[str, Any] = {
        **dict(pipeline_params or {}),
        "dataset_id": resolved_dataset_id,
        "cppipe_path": str(resolved_cppipe_path),
    }
    if microscope_type is not None:
        base_params["microscope_type"] = microscope_type

    metric_collectors = list(metrics)
    run_slug = _benchmark_path_slug(f"{resolved_dataset_id}_{resolved_pipeline_name}")
    if equivalence_reference_output_dir is None:
        native_result = native_adapter.run(
            dataset_path=resolved_dataset_path,
            pipeline_name=resolved_pipeline_name,
            pipeline_params=base_params,
            metrics=metric_collectors,
            output_dir=resolved_output_root / f"{native_adapter.name}_{run_slug}",
        )
    else:
        native_result = _cached_cellprofiler_reference_result(
            Path(equivalence_reference_output_dir),
            dataset_id=resolved_dataset_id,
            pipeline_name=resolved_pipeline_name,
            tool_name=native_adapter.name,
        )
    converted_result = _run_or_reuse_cached_openhcs(
        converted_adapter,
        dataset_path=resolved_dataset_path,
        pipeline_name=resolved_pipeline_name,
        pipeline_params={
            **base_params,
            "equivalence_reference_output_dir": str(native_result.output_path),
        },
        metrics=metric_collectors,
        output_dir=resolved_output_root / f"{converted_adapter.name}_{run_slug}",
        reuse_cache=reuse_openhcs_cache,
    )
    return CellProfilerCompatibilityResult(
        native_cellprofiler=native_result,
        openhcs_converted=converted_result,
    )


def _cached_cellprofiler_reference_result(
    reference_output_dir: Path,
    *,
    dataset_id: str,
    pipeline_name: str,
    tool_name: str,
) -> BenchmarkResult:
    """Represent an already-produced native CellProfiler output as a result."""
    resolved_reference = Path(reference_output_dir)
    if not resolved_reference.exists():
        raise FileNotFoundError(
            "Cached CellProfiler reference output directory does not exist: "
            f"{resolved_reference}"
        )
    if not resolved_reference.is_dir():
        raise NotADirectoryError(
            "Cached CellProfiler reference output path is not a directory: "
            f"{resolved_reference}"
        )
    return BenchmarkResult(
        tool_name=tool_name,
        dataset_id=dataset_id,
        pipeline_name=pipeline_name,
        metrics={},
        output_path=resolved_reference,
        success=True,
        provenance={
            "pipeline_source": "native_cppipe",
            "reused_reference_output": True,
        },
    )


def _benchmark_path_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def _run_or_reuse_cached_openhcs(
    adapter: ToolAdapter,
    *,
    dataset_path: Path,
    pipeline_name: str,
    pipeline_params: Mapping[str, Any],
    metrics: list[Any],
    output_dir: Path,
    reuse_cache: bool,
) -> BenchmarkResult:
    """Run OpenHCS, or reuse a manifest-validated cached result."""
    resolved_output_dir = Path(output_dir)
    result_cache_key = _openhcs_cache_key(
        adapter,
        dataset_path=Path(dataset_path),
        pipeline_name=pipeline_name,
        pipeline_params=pipeline_params,
    )
    execution_cache_key = _openhcs_execution_cache_key(
        adapter,
        dataset_path=Path(dataset_path),
        pipeline_name=pipeline_name,
        pipeline_params=pipeline_params,
    )
    manifest_path = resolved_output_dir / _OPENHCS_CACHE_MANIFEST_NAME
    if reuse_cache:
        cached_result = _cached_benchmark_result(
            manifest_path,
            expected_cache_key=result_cache_key,
            requested_metrics=metrics,
        )
        if cached_result is not None:
            return cached_result

    execution_manifest_path = (
        resolved_output_dir / _OPENHCS_EXECUTION_CACHE_MANIFEST_NAME
    )
    execution_params = {
        **dict(pipeline_params),
        "runtime_execution_cache_manifest": str(execution_manifest_path),
        "runtime_execution_cache_key": execution_cache_key,
        "reuse_runtime_execution_cache": reuse_cache,
    }

    adapter.validate_installation()
    result = adapter.run(
        dataset_path=Path(dataset_path),
        pipeline_name=pipeline_name,
        pipeline_params=execution_params,
        metrics=metrics,
        output_dir=resolved_output_dir,
    )
    if result.success:
        _write_benchmark_result_cache(manifest_path, result_cache_key, result)
    return result


def _cached_benchmark_result(
    manifest_path: Path,
    *,
    expected_cache_key: Mapping[str, Any],
    requested_metrics: list[Any],
) -> BenchmarkResult | None:
    """Load a cached benchmark result when its manifest exactly matches."""
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("schema_version") != _BENCHMARK_CACHE_SCHEMA_VERSION:
        return None
    if manifest.get("cache_key") != expected_cache_key:
        return None

    result_payload = manifest.get("result")
    if not isinstance(result_payload, Mapping):
        return None
    output_path_value = result_payload.get("output_path")
    if not output_path_value:
        return None
    output_path = Path(str(output_path_value))
    if not output_path.exists():
        return None

    provenance = dict(result_payload.get("provenance") or {})
    provenance["reused_cached_output"] = True
    provenance["cache_manifest"] = str(manifest_path)
    return BenchmarkResult(
        tool_name=str(result_payload.get("tool_name", "OpenHCS")),
        dataset_id=str(result_payload.get("dataset_id", "")),
        pipeline_name=str(result_payload.get("pipeline_name", "")),
        metrics=_cached_metric_values(
            result_payload.get("metrics"),
            requested_metrics=requested_metrics,
        ),
        output_path=output_path,
        success=bool(result_payload.get("success", True)),
        error_message=result_payload.get("error_message"),
        provenance=provenance,
    )


def _write_benchmark_result_cache(
    manifest_path: Path,
    cache_key: Mapping[str, Any],
    result: BenchmarkResult,
) -> None:
    """Persist enough result metadata to safely reuse an existing output tree."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": _BENCHMARK_CACHE_SCHEMA_VERSION,
                "cache_key": cache_key,
                "result": {
                    "tool_name": result.tool_name,
                    "dataset_id": result.dataset_id,
                    "pipeline_name": result.pipeline_name,
                    "metrics": _json_ready(result.metrics),
                    "output_path": str(result.output_path),
                    "success": result.success,
                    "error_message": result.error_message,
                    "provenance": _json_ready(result.provenance or {}),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _cached_metric_values(
    cached_metrics: object,
    *,
    requested_metrics: list[Any],
) -> dict[str, Any]:
    """Return cached metric values only for metrics requested by this run."""
    if not isinstance(cached_metrics, Mapping):
        return {}
    requested_names = tuple(
        str(metric.name)
        for metric in requested_metrics
        if hasattr(metric, "name")
    )
    return {
        name: cached_metrics[name]
        for name in requested_names
        if name in cached_metrics
    }


def _openhcs_cache_key(
    adapter: ToolAdapter,
    *,
    dataset_path: Path,
    pipeline_name: str,
    pipeline_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the semantic OpenHCS execution cache identity."""
    reference_output = pipeline_params.get("equivalence_reference_output_dir")
    cppipe_path = pipeline_params.get("cppipe_path")
    return {
        "schema_version": _BENCHMARK_CACHE_SCHEMA_VERSION,
        "tool_name": adapter.name,
        "tool_version": getattr(adapter, "version", "unknown"),
        "pipeline_name": pipeline_name,
        "pipeline_params": _json_ready(dict(pipeline_params)),
        "dataset_tree": _tree_metadata_fingerprint(dataset_path),
        "cppipe_file": (
            _file_content_fingerprint(Path(str(cppipe_path)))
            if cppipe_path is not None
            else None
        ),
        "equivalence_reference_tree": (
            _tree_metadata_fingerprint(Path(str(reference_output)))
            if reference_output is not None
            else None
        ),
        "source_tree": _source_tree_fingerprint(),
    }


def _openhcs_execution_cache_key(
    adapter: ToolAdapter,
    *,
    dataset_path: Path,
    pipeline_name: str,
    pipeline_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the OpenHCS execution cache identity, excluding parity-only inputs."""
    cppipe_path = pipeline_params.get("cppipe_path")
    return {
        "schema_version": _BENCHMARK_CACHE_SCHEMA_VERSION,
        "tool_name": adapter.name,
        "tool_version": getattr(adapter, "version", "unknown"),
        "pipeline_name": pipeline_name,
        "pipeline_params": _json_ready(
            _execution_cache_pipeline_params(pipeline_params)
        ),
        "dataset_tree": _tree_metadata_fingerprint(dataset_path),
        "cppipe_file": (
            _file_content_fingerprint(Path(str(cppipe_path)))
            if cppipe_path is not None
            else None
        ),
        "execution_source_tree": _source_tree_fingerprint(
            excluded_cache_domains=_EXECUTION_SOURCE_CACHE_EXCLUDED_DOMAINS,
        ),
        _LEGACY_SOURCE_TREE_CACHE_KEY: _source_tree_fingerprint(),
    }


def _execution_cache_pipeline_params(
    pipeline_params: Mapping[str, Any],
) -> dict[str, Any]:
    """Return pipeline params that can affect OpenHCS execution outputs."""
    return {
        key: value
        for key, value in pipeline_params.items()
        if key not in _EXECUTION_CACHE_IGNORED_PARAM_KEYS
    }


def _source_tree_fingerprint(
    *,
    excluded_cache_domains: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Fingerprint relevant Python source files that affect converted execution."""
    repo_root = Path(__file__).resolve().parents[1]
    roots = tuple(repo_root / root for root in _SOURCE_CACHE_ROOTS)
    files = tuple(repo_root / file_name for file_name in _SOURCE_CACHE_FILES)
    digest = hashlib.sha256()
    file_count = 0
    for path in sorted((*roots, *files), key=lambda value: str(value)):
        if path.is_file():
            if (
                path.suffix in _SOURCE_CACHE_SUFFIXES
                and not _source_file_is_path_excluded(path, repo_root=repo_root)
                and not _source_file_has_excluded_cache_domain(
                    path,
                    excluded_cache_domains=excluded_cache_domains,
                )
            ):
                file_count += 1
                _update_digest_with_file_content(digest, path, relative_to=repo_root)
            continue
        if not path.is_dir():
            continue
        for child in sorted(path.rglob("*")):
            if (
                child.is_file()
                and child.suffix in _SOURCE_CACHE_SUFFIXES
                and not _source_file_is_path_excluded(child, repo_root=repo_root)
                and not _source_file_has_excluded_cache_domain(
                    child,
                    excluded_cache_domains=excluded_cache_domains,
                )
            ):
                file_count += 1
                _update_digest_with_file_content(
                    digest,
                    child,
                    relative_to=repo_root,
                )
    return {"file_count": file_count, "digest": digest.hexdigest()}


def _source_file_is_path_excluded(path: Path, *, repo_root: Path) -> bool:
    """Return whether a source file is outside runtime cache authority."""
    try:
        relative_path = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    return any(
        relative_path == excluded_dir or excluded_dir in relative_path.parents
        for excluded_dir in _SOURCE_CACHE_EXCLUDED_DIRS
    )


def _source_file_has_excluded_cache_domain(
    path: Path,
    *,
    excluded_cache_domains: frozenset[str],
) -> bool:
    if not excluded_cache_domains or path.suffix != ".py":
        return False
    try:
        stat = path.stat()
    except OSError:
        return False
    domains = _source_file_cache_domains(
        str(path),
        stat.st_size,
        stat.st_mtime_ns,
    )
    return bool(domains & excluded_cache_domains)


@lru_cache(maxsize=4096)
def _source_file_cache_domains(
    path: str,
    size: int,
    mtime_ns: int,
) -> frozenset[str]:
    del size, mtime_ns
    try:
        with tokenize_open(path) as handle:
            tree = ast.parse(handle.read(), filename=path)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return frozenset()

    for statement in tree.body:
        value_node: ast.AST | None = None
        targets: tuple[ast.AST, ...] = ()
        if isinstance(statement, ast.Assign):
            value_node = statement.value
            targets = tuple(statement.targets)
        elif isinstance(statement, ast.AnnAssign):
            value_node = statement.value
            targets = (statement.target,)
        if value_node is None:
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id == _BENCHMARK_CACHE_DOMAINS_SYMBOL
            for target in targets
        ):
            continue
        return _literal_string_set(value_node)
    return frozenset()


def _literal_string_set(node: ast.AST) -> frozenset[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return frozenset({node.value})
    if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
        values = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(
                element.value,
                str,
            ):
                return frozenset()
            values.append(element.value)
        return frozenset(values)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"frozenset", "set", "tuple", "list"}
        and len(node.args) == 1
        and not node.keywords
    ):
        return _literal_string_set(node.args[0])
    return frozenset()


def _tree_metadata_fingerprint(path: Path) -> dict[str, Any]:
    """Fingerprint a data/output tree using stable file metadata."""
    root = Path(path)
    if not root.exists():
        return {
            "path": str(root),
            "exists": False,
            "kind": "missing",
            "file_count": 0,
            "digest": None,
        }
    if root.is_file():
        return _file_metadata_fingerprint(root)

    digest = hashlib.sha256()
    file_count = 0
    for child in sorted(root.rglob("*")):
        if not child.is_file():
            continue
        file_count += 1
        _update_digest_with_file_metadata(digest, child, relative_to=root)
    return {
        "path": str(root.resolve()),
        "exists": True,
        "kind": "directory",
        "file_count": file_count,
        "digest": digest.hexdigest(),
    }


def _file_metadata_fingerprint(path: Path) -> dict[str, Any]:
    resolved = Path(path)
    stat = resolved.stat()
    digest = hashlib.sha256()
    _update_digest_with_file_metadata(digest, resolved, relative_to=resolved.parent)
    return {
        "path": str(resolved.resolve()),
        "exists": True,
        "kind": "file",
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "digest": digest.hexdigest(),
    }


def _file_content_fingerprint(path: Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        return {
            "path": str(resolved),
            "exists": False,
            "kind": "missing",
            "digest": None,
        }
    digest = hashlib.sha256()
    _update_digest_with_file_content(digest, resolved, relative_to=resolved.parent)
    return {
        "path": str(resolved.resolve()),
        "exists": True,
        "kind": "file",
        "digest": digest.hexdigest(),
    }


def _update_digest_with_file_metadata(
    digest: "hashlib._Hash",
    path: Path,
    *,
    relative_to: Path,
) -> None:
    stat = path.stat()
    relative_path = path.relative_to(relative_to).as_posix()
    digest.update(
        json.dumps(
            [relative_path, stat.st_size, stat.st_mtime_ns],
            separators=(",", ":"),
        ).encode()
    )


def _update_digest_with_file_content(
    digest: "hashlib._Hash",
    path: Path,
    *,
    relative_to: Path,
) -> None:
    relative_path = path.relative_to(relative_to).as_posix()
    digest.update(relative_path.encode())
    digest.update(b"\0")
    with path.open("rb") as handle:
        while chunk := handle.read(_TREE_METADATA_CHUNK_SIZE):
            digest.update(chunk)
    digest.update(b"\0")


def _json_ready(value: object) -> object:
    """Normalize arbitrary benchmark params into stable JSON-compatible values."""
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized_items = [_json_ready(item) for item in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)
