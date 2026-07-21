from openhcs.serialization.json import to_jsonable
from openhcs.agent.services.architecture_projection_service import (
    ArchitectureProjectionService,
)


def test_architecture_topics_are_listed_with_stable_ids():
    service = ArchitectureProjectionService()

    page = service.list_topics()
    topic_ids = {topic.topic_id for topic in page.topics}

    assert page.schema_version == "openhcs.agent.v1"
    assert "pipeline_model" in topic_ids
    assert "cellprofiler_translation" in topic_ids


def test_pipeline_architecture_topic_is_backed_by_real_symbols():
    service = ArchitectureProjectionService()

    topic = service.explain_topic("pipeline_model")
    function_step = next(
        symbol
        for symbol in topic.internal_symbols
        if symbol.symbol_id == "core.FunctionStep"
    )

    assert "FunctionStep" in topic.concepts[0]
    assert function_step.import_path == "openhcs.core.steps.function_step.FunctionStep"
    assert function_step.source_path == "openhcs/core/steps/function_step.py"
    assert function_step.line_number is not None
    assert function_step.symbol_kind == "class"


def test_cellprofiler_translation_topic_exposes_nominal_and_public_authorities():
    service = ArchitectureProjectionService()

    topic = service.explain_topic("cellprofiler_translation")
    symbol_ids = {symbol.symbol_id for symbol in topic.internal_symbols}

    assert symbol_ids == {
        "cellprofiler.CPPipeParser",
        "cellprofiler.ModuleBlock",
        "cellprofiler.CellProfilerModule",
        "cellprofiler.import_cellprofiler_pipeline",
        "core.FunctionStep",
        "core.PipelineConfig",
    }
    assert "ordinary public OpenHCS pipeline declarations" in topic.summary
    assert any(
        "CellProfilerModule is the nominal registry" in concept
        for concept in topic.concepts
    )
    assert any(
        "same FunctionStep and PipelineConfig declarations" in concept
        for concept in topic.concepts
    )
    assert "CellProfiler Image name" in topic.cellprofiler_translation_notes[0]


def test_cellprofiler_translation_symbols_resolve_to_current_source_owners():
    service = ArchitectureProjectionService()

    module = service.describe_internal_symbol("cellprofiler.CellProfilerModule")
    importer = service.describe_internal_symbol(
        "cellprofiler.import_cellprofiler_pipeline"
    )

    assert module.import_path == (
        "openhcs.interop.cellprofiler.module_declarations.CellProfilerModule"
    )
    assert module.source_path == "openhcs/interop/cellprofiler/module_declarations.py"
    assert importer.import_path == (
        "openhcs.interop.cellprofiler.pipeline_import.import_cellprofiler_pipeline"
    )
    assert importer.source_path == "openhcs/interop/cellprofiler/pipeline_import.py"
    assert "tuple[list[FunctionStep], PipelineConfig]" in importer.signature


def test_describe_internal_symbol_returns_json_safe_projection():
    service = ArchitectureProjectionService()

    symbol = service.describe_internal_symbol("core.FunctionStep")
    payload = to_jsonable(symbol)

    assert payload["symbol_id"] == "core.FunctionStep"
    assert payload["source_path"] == "openhcs/core/steps/function_step.py"
