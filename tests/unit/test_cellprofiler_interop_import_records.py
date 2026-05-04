"""CellProfiler pipeline import record tests."""

from pathlib import Path

import pytest

from openhcs.constants import Backend
from openhcs.core.module_artifact_contract import ModuleArtifactContract
from openhcs.core.pipeline import Pipeline
from openhcs.core.pipeline_image_schema import PipelineImageSchema
from openhcs.interop.cellprofiler import (
    CellProfilerDialectCompiler,
    CellProfilerModuleReference,
    CellProfilerModuleRole,
    CellProfilerPipelineImportRequest,
    CellProfilerPipelineImportResult,
    CellProfilerPipelineProvenance,
    clear_cellprofiler_dialect_compiler,
    get_cellprofiler_dialect_compiler,
    register_cellprofiler_dialect_compiler,
)


def test_cellprofiler_pipeline_provenance_preserves_typed_module_roles() -> None:
    provenance = CellProfilerPipelineProvenance(
        cppipe_path=Path("example.cppipe"),
        modules=(
            CellProfilerModuleReference(
                name="Images",
                module_num=1,
                role=CellProfilerModuleRole.INFRASTRUCTURE,
            ),
            CellProfilerModuleReference(
                name="IdentifyPrimaryObjects",
                module_num=2,
                role=CellProfilerModuleRole.PROCESSING,
            ),
        ),
    )

    assert [module.name for module in provenance.modules] == [
        "Images",
        "IdentifyPrimaryObjects",
    ]
    assert [module.name for module in provenance.infrastructure_modules] == ["Images"]
    assert [module.name for module in provenance.processing_modules] == [
        "IdentifyPrimaryObjects"
    ]


def test_cellprofiler_pipeline_import_result_requires_openhcs_contracts() -> None:
    provenance = CellProfilerPipelineProvenance(
        cppipe_path=Path("example.cppipe"),
        modules=(
            CellProfilerModuleReference(
                name="IdentifyPrimaryObjects",
                module_num=1,
                role=CellProfilerModuleRole.PROCESSING,
            ),
        ),
    )
    result = CellProfilerPipelineImportResult(
        provenance=provenance,
        pipeline=Pipeline(steps=[], name="example"),
        source_schema=PipelineImageSchema.empty(),
        generated_source="# generated",
        generated_module_name="generated_example",
        generated_module_path=Path("generated_example.py"),
        artifact_contracts=(ModuleArtifactContract("IdentifyPrimaryObjects"),),
        registered_functions=("generated_example:run",),
    )

    assert result.provenance is provenance
    assert result.generated_module_path == Path("generated_example.py")
    assert result.artifact_contracts[0].module_name == "IdentifyPrimaryObjects"


def test_cellprofiler_pipeline_import_result_rejects_benchmark_contract_shape() -> None:
    provenance = CellProfilerPipelineProvenance(
        cppipe_path=Path("example.cppipe"),
        modules=(
            CellProfilerModuleReference(
                name="IdentifyPrimaryObjects",
                module_num=1,
                role=CellProfilerModuleRole.PROCESSING,
            ),
        ),
    )

    with pytest.raises(TypeError, match="ModuleArtifactContract"):
        CellProfilerPipelineImportResult(
            provenance=provenance,
            pipeline=Pipeline(steps=[], name="example"),
            source_schema=PipelineImageSchema.empty(),
            generated_source="# generated",
            generated_module_name="generated_example",
            generated_module_path=Path("generated_example.py"),
            artifact_contracts=("IdentifyPrimaryObjects",),  # type: ignore[arg-type]
        )


def test_cellprofiler_pipeline_import_request_preserves_typed_file_contract() -> None:
    filemanager = object()
    request = CellProfilerPipelineImportRequest(
        cppipe_path=Path("example.cppipe"),
        generated_pipeline_path=Path("example_generated.py"),
        prune_dead_unmaterialized_artifact_steps=True,
        filemanager=filemanager,
        cppipe_backend=Backend.MEMORY,
        generated_pipeline_backend=Backend.ZARR,
    )

    assert request.cppipe_path == Path("example.cppipe")
    assert request.generated_pipeline_path == Path("example_generated.py")
    assert request.prune_dead_unmaterialized_artifact_steps is True
    assert request.filemanager is filemanager
    assert request.cppipe_backend is Backend.MEMORY
    assert request.generated_pipeline_backend is Backend.ZARR


def test_cellprofiler_pipeline_import_request_rejects_wrong_file_roles() -> None:
    with pytest.raises(ValueError, match=r"\.cppipe"):
        CellProfilerPipelineImportRequest(
            cppipe_path=Path("example.py"),
            generated_pipeline_path=Path("example_generated.py"),
        )

    with pytest.raises(ValueError, match=r"\.py"):
        CellProfilerPipelineImportRequest(
            cppipe_path=Path("example.cppipe"),
            generated_pipeline_path=Path("example_generated.cppipe"),
        )


def test_cellprofiler_pipeline_import_request_rejects_untyped_backend() -> None:
    with pytest.raises(TypeError, match="cppipe_backend"):
        CellProfilerPipelineImportRequest(
            cppipe_path=Path("example.cppipe"),
            generated_pipeline_path=Path("example_generated.py"),
            cppipe_backend="memory",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="generated_pipeline_backend"):
        CellProfilerPipelineImportRequest(
            cppipe_path=Path("example.cppipe"),
            generated_pipeline_path=Path("example_generated.py"),
            generated_pipeline_backend="disk",  # type: ignore[arg-type]
        )


def test_cellprofiler_pipeline_importer_is_dialect_compiler_contract() -> None:
    from benchmark.converter.runtime_pipeline import (
        BenchmarkCellProfilerDialectCompiler,
    )

    assert issubclass(BenchmarkCellProfilerDialectCompiler, CellProfilerDialectCompiler)


def test_cellprofiler_dialect_compiler_registry_fails_loudly_without_provider() -> None:
    clear_cellprofiler_dialect_compiler()

    with pytest.raises(RuntimeError, match="No CellProfiler dialect compiler"):
        get_cellprofiler_dialect_compiler()


def test_cellprofiler_dialect_compiler_registry_requires_typed_provider() -> None:
    clear_cellprofiler_dialect_compiler()

    with pytest.raises(TypeError, match="CellProfilerDialectCompiler"):
        register_cellprofiler_dialect_compiler(object())  # type: ignore[arg-type]


def test_cellprofiler_dialect_compiler_registry_returns_explicit_provider() -> None:
    from benchmark.converter.runtime_pipeline import (
        BenchmarkCellProfilerDialectCompiler,
    )

    clear_cellprofiler_dialect_compiler()
    compiler = BenchmarkCellProfilerDialectCompiler()
    register_cellprofiler_dialect_compiler(compiler)

    assert get_cellprofiler_dialect_compiler() is compiler
