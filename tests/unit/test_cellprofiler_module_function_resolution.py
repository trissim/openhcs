"""Focused tests for declaration-owned CellProfiler callable selection."""

from dataclasses import replace

import pytest

from openhcs.constants.constants import AllComponents
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    InputStackBroadcastSourceRelation,
    ObjectLabelsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.config import StepSourceBindingsConfig
from openhcs.core.function_patterns import FunctionInvocationKey
from openhcs.core.source_bindings import NamedSourceBinding
from openhcs.interop.cellprofiler.module_declarations import CellProfilerModule
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.processing.backends.cellprofiler.crop import CropModule
from openhcs.processing.backends.cellprofiler.image_geometry import MaskImageModule
from openhcs.processing.backends.cellprofiler.morphology import (
    DilateObjectsModule,
    RemoveHolesModule,
    ShrinkToObjectCentersModule,
)


def _module(name: str, settings: dict[str, str]) -> ModuleBlock:
    return ModuleBlock(
        name=name,
        module_num=1,
        setting_records=[
            ModuleSetting(_setting_name, _setting_value)
            for (_setting_name, _setting_value) in settings.items()
        ],
    )


def _contract_and_source_bindings(
    module: ModuleBlock,
    *,
    object_input: bool = False,
    source_stack_components: tuple[AllComponents, ...] = (),
) -> tuple[CallableContract, StepSourceBindingsConfig]:
    module_type = CellProfilerModule.require_module(module.name)
    raw_contract = CallableContract.from_callable(module_type.require_callable())
    contract = replace(
        raw_contract,
        metadata=replace(
            raw_contract.metadata,
            artifact_inputs=(
                (ArtifactSpec.input("Nuclei", ObjectLabelsArtifactType),)
                if object_input
                else ()
            ),
        ),
    )
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(NamedSourceBinding(alias="Source"),),
        source_stack_components=source_stack_components,
    )
    return contract, source_bindings


def _resolved_callable(module: ModuleBlock, *, object_input: bool = False):
    module_type = CellProfilerModule.require_module(module.name)
    contract, source_bindings = _contract_and_source_bindings(
        module,
        object_input=object_input,
    )
    return module_type.resolve_function(
        module,
        contract=contract,
        source_bindings=source_bindings,
    )


def _assert_resolves(
    module: ModuleBlock,
    expected_name: str,
    *,
    object_input: bool = False,
) -> None:
    module_type = CellProfilerModule.require_module(module.name)
    assert _resolved_callable(
        module,
        object_input=object_input,
    ) is module_type.require_callable(expected_name)


def test_measure_texture_resolution_is_declared_on_module_class() -> None:
    _assert_resolves(
        _module(
            "MeasureTexture",
            {
                "Measure images or objects?": "Objects",
                "Select objects to measure": "Nuclei",
            },
        ),
        "measure_texture_objects",
        object_input=True,
    )
    _assert_resolves(
        _module(
            "MeasureTexture",
            {
                "Measure images or objects?": "Images",
                "Select objects to measure": "Nuclei",
            },
        ),
        "measure_texture",
    )


def test_measure_image_intensity_resolution_is_declared_on_module_class() -> None:
    _assert_resolves(
        _module(
            "MeasureImageIntensity",
            {"Select images to measure": "CropBlue"},
        ),
        "measure_image_intensity",
    )
    _assert_resolves(
        _module(
            "MeasureImageIntensity",
            {
                "Select images to measure": "CropBlue",
                "Select the input objects": "Nuclei",
            },
        ),
        "measure_image_intensity_objects",
        object_input=True,
    )


def test_measurement_variants_resolve_to_declared_callable_objects() -> None:
    _assert_resolves(
        _module(
            "MeasureColocalization",
            {
                "Select where to measure correlation": "Both",
                "Select objects to measure": "Nuclei",
            },
        ),
        "measure_colocalization_objects",
        object_input=True,
    )
    _assert_resolves(
        _module("MeasureGranularity", {"Select objects to measure": "Nuclei"}),
        "measure_granularity_objects",
        object_input=True,
    )


def test_resize_variants_resolve_to_declared_callable_objects() -> None:
    _assert_resolves(
        _module("Resize", {"Z Resizing factor": "1.0"}),
        "resize_volumetric",
    )
    _assert_resolves(
        _module("ResizeObjects", {"Planes (Z)": "10"}),
        "resize_objects_3d",
    )


@pytest.mark.parametrize(
    ("module_type", "object_input"),
    (
        (DilateObjectsModule, True),
        (RemoveHolesModule, False),
        (ShrinkToObjectCentersModule, True),
    ),
)
def test_source_stack_variants_use_declared_source_components(
    module_type: type[CellProfilerModule],
    object_input: bool,
) -> None:
    assert "resolve_function" not in module_type.__dict__
    module = _module(str(module_type.module_name), {})
    planar_contract, planar_source_bindings = _contract_and_source_bindings(
        module,
        object_input=object_input,
    )
    planar = module_type.resolve_function(
        module,
        contract=planar_contract,
        source_bindings=planar_source_bindings,
    )
    assert planar is module_type.require_callable()

    contract, source_bindings = _contract_and_source_bindings(
        module,
        object_input=object_input,
        source_stack_components=(AllComponents.Z_INDEX,),
    )
    resolved = module_type.resolve_function(
        module,
        contract=contract,
        source_bindings=source_bindings,
    )
    assert resolved is module_type.require_callable(module_type.function_variants[0])


@pytest.mark.parametrize("module_type", (CropModule, MaskImageModule))
def test_broadcast_image_outputs_inherit_primary_image_lineage(
    module_type: type[CellProfilerModule],
) -> None:
    assert "artifact_output_relations" not in module_type.__dict__
    primary = ArtifactSpec.input("Primary", ImageArtifactType)
    mask = ArtifactSpec.input("Mask", ImageArtifactType).with_group_scope_relation(
        InputStackBroadcastSourceRelation(source=primary.ref())
    )

    relations = module_type.artifact_output_relations(
        None,
        binding=module_type.output_image_binding,
        name="Output",
        invocation_key=FunctionInvocationKey(
            str(module_type.function_name),
            "default",
            0,
        ),
        step_context=None,
        artifact_inputs=ArtifactSpecCollection((primary, mask)),
        output_position=0,
    )

    assert relations == (SourceStackLineageSourceRelation(source=primary.ref()),)
