from pathlib import Path

import pytest

from benchmark.converter.parser import ModuleBlock
from benchmark.converter.pipeline_generator import PipelineGenerator
from benchmark.converter.symbol_table import (
    CellProfilerSymbolKind,
    CellProfilerSymbolTable,
)
from openhcs.core.artifacts import ArtifactKind


def _module(
    module_num: int,
    name: str,
    settings: dict[str, str],
) -> ModuleBlock:
    return ModuleBlock(name=name, module_num=module_num, settings=settings)


def _identify_primary(module_num: int = 1) -> ModuleBlock:
    return _module(
        module_num,
        "IdentifyPrimaryObjects",
        {
            "Select the input image": "OrigBlue",
            "Name the primary objects to be identified": "Nuclei",
        },
    )


def _identify_secondary(module_num: int = 2) -> ModuleBlock:
    return _module(
        module_num,
        "IdentifySecondaryObjects",
        {
            "Select the input objects": "Nuclei",
            "Select the input image": "OrigGreen",
            "Name the objects to be identified": "Cells",
            "Name the new primary objects": "FilteredNuclei",
        },
    )


def _identify_tertiary(module_num: int = 3) -> ModuleBlock:
    return _module(
        module_num,
        "IdentifyTertiaryObjects",
        {
            "Select the larger identified objects": "Cells",
            "Select the smaller identified objects": "Nuclei",
            "Name the tertiary objects to be identified": "Cytoplasm",
        },
    )


def test_cellprofiler_symbol_table_compiles_object_measurement_graph():
    modules = [
        _identify_primary(),
        _identify_secondary(),
        _identify_tertiary(),
        _module(
            4,
            "MeasureObjectIntensity",
            {
                "Select images to measure": "OrigBlue, OrigGreen",
                "Select objects to measure": "Nuclei, Cells, Cytoplasm",
            },
        ),
        _module(
            5,
            "MeasureImageIntensity",
            {
                "Select images to measure": "OrigBlue",
                "Select input object sets": "",
            },
        ),
    ]

    table = CellProfilerSymbolTable.compile(modules)

    assert table.symbols["OrigBlue"].kind is CellProfilerSymbolKind.IMAGE
    assert table.symbols["OrigBlue"].producer_module_num is None
    assert table.symbols["Nuclei"].kind is CellProfilerSymbolKind.OBJECTS
    assert table.symbols["Nuclei"].producer_module_num == 1
    assert table.symbols["Cytoplasm"].kind is CellProfilerSymbolKind.OBJECTS
    assert table.symbols["MeasureObjectIntensity_4_measurements"].kind is (
        CellProfilerSymbolKind.MEASUREMENTS
    )

    primary_contract = table.contracts_by_module_num[1]
    assert [spec.kind for spec in primary_contract.inputs] == [ArtifactKind.IMAGE]
    assert primary_contract.external_image_inputs == ("OrigBlue",)
    assert primary_contract.runtime_artifact_inputs == ()
    assert primary_contract.outputs[0].kind is ArtifactKind.OBJECT_LABELS

    measure_contract = table.contracts_by_module_num[4]
    assert measure_contract.external_image_inputs == ("OrigBlue", "OrigGreen")
    assert [spec.name for spec in measure_contract.runtime_artifact_inputs] == [
        "Nuclei",
        "Cells",
        "Cytoplasm",
    ]
    assert measure_contract.outputs[0].kind is ArtifactKind.MEASUREMENTS


def test_cellprofiler_symbol_table_fails_for_unknown_object_input():
    modules = [
        _module(
            1,
            "MeasureObjectSizeShape",
            {"Select object sets to measure": "MissingObjects"},
        )
    ]

    with pytest.raises(ValueError, match="unknown objects symbol 'MissingObjects'"):
        CellProfilerSymbolTable.compile(modules)


def test_cellprofiler_symbol_table_fails_for_kind_conflict():
    modules = [
        _identify_primary(),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "Nuclei",
                "Name the primary objects to be identified": "OtherObjects",
            },
        ),
    ]

    with pytest.raises(ValueError, match="expects 'Nuclei' as image"):
        CellProfilerSymbolTable.compile(modules)


def test_cellprofiler_symbol_table_fails_for_duplicate_producer():
    modules = [
        _identify_primary(),
        _module(
            2,
            "IdentifyPrimaryObjects",
            {
                "Select the input image": "OrigGreen",
                "Name the primary objects to be identified": "Nuclei",
            },
        ),
    ]

    with pytest.raises(ValueError, match="already produced by module 1"):
        CellProfilerSymbolTable.compile(modules)


def test_pipeline_generator_emits_compiled_artifact_contracts():
    generator = PipelineGenerator()
    modules = [_identify_primary(), _identify_secondary()]

    generated = generator.generate_from_registry(
        pipeline_name="cp_graph",
        source_cppipe=Path("source.cppipe"),
        modules=modules,
    )

    assert len(generated.artifact_contracts) == 2
    assert "CELLPROFILER_ARTIFACT_CONTRACTS" in generated.code
    assert '"external_image_inputs": (\'OrigBlue\',)' in generated.code
    assert '"runtime_artifact_inputs": (ArtifactSpec(\'Nuclei\'' in generated.code
    assert "identify_primary_objects_1 = get_function" in generated.code
    assert "identify_secondary_objects_2 = get_function" in generated.code
