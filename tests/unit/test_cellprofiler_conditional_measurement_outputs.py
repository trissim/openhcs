from __future__ import annotations

import inspect
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from openhcs.core.artifacts import (
    ArtifactInputPlan,
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
    SourceStackLineageSourceRelation,
)
from openhcs.core.function_patterns import (
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.measurement_row_materialization import (
    MeasurementProjectedColumnarRows,
)
from openhcs.core.pipeline.artifact_planning import artifact_producers_for_outputs
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_measurements import MeasurementScope, MeasurementSubject
from openhcs.core.runtime_tabular_values import FieldSpec
from openhcs.processing.backends.cellprofiler.calculate_statistics import (
    CalculateStatisticsModule,
    calculate_statistics,
)
from openhcs.processing.backends.cellprofiler.flagging import (
    CombinationChoice,
    FlagImageModule,
    FlagImagePlan,
    MeasurementSource,
    flag_image,
)
from openhcs.interop.cellprofiler.parser import CPPipeParser
from openhcs.interop.cellprofiler.settings_binder import SettingsBinder
from openhcs.processing.backends.cellprofiler.intensity import (
    MeasureImageIntensityModule,
)
from openhcs.processing.backends.cellprofiler.object_overlap import (
    MeasureImageOverlapModule,
    measureimageoverlap,
    measureimageoverlap_with_emd,
)


def _invocation(func, kwargs=None):
    pattern = func if kwargs is None else (func, kwargs)
    return next(normalize_function_pattern(pattern).iter_items())


def _measurement_context() -> tuple[
    ArtifactDeclarationStepContext,
    ArtifactSpec,
    ArtifactSpec,
]:
    image = ArtifactSpec.output("DNA", ImageArtifactType)
    measurements = ArtifactSpec.output(
        "IntensityMeasurements",
        MeasurementsArtifactType,
        relations=(
            SourceStackLineageSourceRelation(
                source=image.for_plan_type(ArtifactInputPlan).ref()
            ),
        ),
    )
    producer_key = FunctionInvocationKey(
        str(MeasureImageIntensityModule.function_name),
        "default",
        0,
    )
    producers = artifact_producers_for_outputs(
        (measurements,),
        groups=(None,),
        invocation_keys=(producer_key,),
    )
    return (
        ArtifactDeclarationStepContext(
            step_index=1,
            available_artifacts=ArtifactSpecCollection((image, measurements)),
            main_flow_artifacts=ArtifactSpecCollection((image,)),
            available_artifact_producers=producers,
        ),
        image,
        measurements,
    )


def _table(
    name: str,
    columns: dict[str, tuple[object, ...]],
    subject: MeasurementSubject,
) -> MeasurementTable:
    fields = tuple(
        FieldSpec(
            field_name,
            int if field_name in {"slice_index", "object_label"} else float,
        )
        for field_name in columns
    )
    return MeasurementTable(
        name=name,
        rows=MeasurementProjectedColumnarRows(
            MappingProxyType(columns),
            fields=fields,
        ),
        subject=subject,
    )


def test_overlap_public_variants_reconstruct_conditional_contract_and_schema() -> None:
    ground_truth = ArtifactSpec.output("GroundTruth", ImageArtifactType)
    test_image = ArtifactSpec.output("Test", ImageArtifactType)
    base_invocation = _invocation(measureimageoverlap)
    emd_invocation = _invocation(measureimageoverlap_with_emd)
    context = ArtifactDeclarationStepContext(
        step_index=0,
        available_artifacts=ArtifactSpecCollection((ground_truth, test_image)),
        main_flow_artifacts=ArtifactSpecCollection(
            (
                ground_truth.for_plan_type(ArtifactInputPlan),
                test_image.for_plan_type(ArtifactInputPlan),
            )
        ),
        available_artifact_producers=artifact_producers_for_outputs(
            (ground_truth, test_image),
            groups=(None,),
            invocation_keys=(
                FunctionInvocationKey(
                    "fixture_producer",
                    base_invocation.key.group_key,
                    0,
                ),
            ),
        ),
    )

    base_blocks, _ = MeasureImageOverlapModule.module_blocks_for_invocation(
        invocation=base_invocation,
        step_context=context,
    )
    emd_blocks, _ = MeasureImageOverlapModule.module_blocks_for_invocation(
        invocation=emd_invocation,
        step_context=context,
    )
    (base_blocks,), next_module_num = (
        MeasureImageOverlapModule.number_step_invocation_blocks(
            (base_blocks,),
            first_module_num=1,
        )
    )
    (emd_blocks,), _ = MeasureImageOverlapModule.number_step_invocation_blocks(
        (emd_blocks,),
        first_module_num=next_module_num,
    )
    base_contract = MeasureImageOverlapModule.callable_contract(
        module=base_blocks[0],
        invocation_key=base_invocation.key,
        step_context=context,
    )
    emd_contract = MeasureImageOverlapModule.callable_contract(
        module=emd_blocks[0],
        invocation_key=emd_invocation.key,
        step_context=context,
    )

    for contract in (base_contract, emd_contract):
        assert contract.artifact_inputs.names() == ("GroundTruth", "Test")
        (measurement_output,) = contract.artifact_outputs.of_artifact_type(
            MeasurementsArtifactType
        )
        assert {relation.source.name for relation in measurement_output.relations} == {
            "GroundTruth",
            "Test",
        }
    assert not MeasureImageOverlapModule.emd_enabled(base_blocks[0])
    assert MeasureImageOverlapModule.emd_enabled(emd_blocks[0])

    image = np.stack(
        (
            np.asarray([[1, 0], [0, 1]], dtype=np.float32),
            np.asarray([[1, 0], [1, 0]], dtype=np.float32),
        )
    )
    base_rows = inspect.unwrap(measureimageoverlap)(image)[1]
    emd_rows = inspect.unwrap(measureimageoverlap_with_emd)(image)[1]
    assert "earth_movers_distance" not in {field.name for field in base_rows.fields}
    assert "earth_movers_distance" in {field.name for field in emd_rows.fields}
    assert (
        MeasureImageOverlapModule.overlap_measurement_feature_name(
            "earth_movers_distance", "Test"
        )
        == "Overlap_EarthMoversDistance_Test"
    )


def test_calculate_statistics_contract_and_exact_experiment_rows() -> None:
    context, _image, measurements = _measurement_context()
    invocation = _invocation(
        calculate_statistics,
        {
            "grouping_feature": "Metadata_Control",
            "dose_features": ("Metadata_Dose",),
            "log_transform_doses": (False,),
        },
    )
    blocks, _ = CalculateStatisticsModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = (
        CalculateStatisticsModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract = CalculateStatisticsModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=context,
    )
    assert any(
        spec.ref() == measurements.for_plan_type(ArtifactInputPlan).ref()
        for spec in contract.artifact_inputs
    )
    (measurement_output,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    assert measurements.for_plan_type(ArtifactInputPlan).ref() in {
        relation.source for relation in measurement_output.relations
    }

    image_table = _table(
        "ImageMeasurements",
        {
            "slice_index": (0, 1, 2, 3),
            "Metadata_Control": (-1.0, -1.0, 1.0, 1.0),
            "Metadata_Dose": (0.0, 0.0, 1.0, 1.0),
            "Intensity_Mean_DNA": (1.0, 2.0, 8.0, 9.0),
        },
        MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    object_table = _table(
        "CellMeasurements",
        {
            "slice_index": (0, 0, 1, 1, 2, 2, 3, 3),
            "object_label": (1, 2, 1, 2, 1, 2, 1, 2),
            "AreaShape_Area": (1.0, 3.0, 2.0, 4.0, 8.0, 10.0, 9.0, 11.0),
        },
        MeasurementSubject(MeasurementScope.OBJECT, "Cells"),
    )
    rows = inspect.unwrap(calculate_statistics)(
        np.zeros((4, 2, 2), dtype=np.float32),
        grouping_feature="Metadata_Control",
        dose_features=("Metadata_Dose",),
        log_transform_doses=(False,),
        measurement_tables=(image_table, object_table),
    )[1]
    names = {field.name for field in rows.fields}
    assert names == {
        f"{statistic}_{subject}_{feature}"
        for subject, feature in (
            ("Image", "Intensity_Mean_DNA"),
            ("Cells", "AreaShape_Area"),
        )
        for statistic in ("Zfactor", "Vfactor", "OneTailedZfactor", "EC50")
    }
    table = CalculateStatisticsModule.build_measurement_table(
        name="CalculateStatisticsMeasurements",
        rows=rows,
        object_name=None,
        source_image_name=None,
        source_metadata=None,
    )
    assert table.subject.scope is MeasurementScope.EXPERIMENT


def test_calculate_statistics_multiple_doses_qualify_only_ec50_rows() -> None:
    table = _table(
        "ImageMeasurements",
        {
            "slice_index": (0, 1, 2, 3),
            "Metadata_Control": (-1.0, -1.0, 1.0, 1.0),
            "Metadata_DoseA": (0.0, 0.0, 1.0, 1.0),
            "Metadata_DoseB": (1.0, 2.0, 3.0, 4.0),
            "Intensity_Mean_DNA": (1.0, 2.0, 8.0, 9.0),
        },
        MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    rows = inspect.unwrap(calculate_statistics)(
        np.zeros((4, 2, 2), dtype=np.float32),
        grouping_feature="Metadata_Control",
        dose_features=("Metadata_DoseA", "Metadata_DoseB"),
        log_transform_doses=(False, False),
        measurement_tables=(table,),
    )[1]
    names = {field.name for field in rows.fields}
    assert "EC50_Metadata_DoseA_Image_Intensity_Mean_DNA" in names
    assert "EC50_Metadata_DoseB_Image_Intensity_Mean_DNA" in names
    assert "EC50_Image_Intensity_Mean_DNA" not in names


def _flag_kwargs(*, wants_skip: tuple[bool, bool] = (False, False)):
    return {
        "flag_categories": ("Metadata", "QC"),
        "flag_names": ("AnyFailure", "AllFailures"),
        "combination_choices": (CombinationChoice.ANY, CombinationChoice.ALL),
        "wants_skip": wants_skip,
        "measurement_counts": (2, 2),
        "measurement_sources": (MeasurementSource.IMAGE,) * 4,
        "object_names": ("Image",) * 4,
        "measurement_features": ("Intensity_MeanIntensity_DNA",) * 4,
        "check_minimums": (False,) * 4,
        "minimum_values": (0.0,) * 4,
        "check_maximums": (True, False, True, False),
        "maximum_values": (1.0,) * 4,
        "ignore_flag_on_last": False,
    }


def test_flag_image_repeated_groups_reconstruct_contract_and_runtime_rows() -> None:
    context, _image, measurements = _measurement_context()
    invocation = _invocation(flag_image, _flag_kwargs())
    blocks, _ = FlagImageModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = (
        FlagImageModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=1,
        )
    )
    contract = FlagImageModule.callable_contract(
        module=numbered_blocks[0],
        invocation_key=invocation.key,
        step_context=context,
    )
    assert any(
        spec.ref() == measurements.for_plan_type(ArtifactInputPlan).ref()
        for spec in contract.artifact_inputs
    )
    (measurement_output,) = contract.artifact_outputs.of_artifact_type(
        MeasurementsArtifactType
    )
    assert measurements.for_plan_type(ArtifactInputPlan).ref() in {
        relation.source for relation in measurement_output.relations
    }

    table = _table(
        "ImageMeasurements",
        {
            "slice_index": (0,),
            "Intensity_MeanIntensity_DNA": (2.0,),
        },
        MeasurementSubject(MeasurementScope.IMAGE, "Image"),
    )
    rows = inspect.unwrap(flag_image)(
        np.zeros((2, 2), dtype=np.float32),
        **_flag_kwargs(),
        measurement_tables=(table,),
    )[1]
    assert rows.row_mappings() == (
        {
            "slice_index": 0,
            "Metadata_AnyFailure": 1,
            "QC_AllFailures": 0,
        },
    )


def test_flag_image_parsed_cppipe_uses_one_nominal_row_schema(tmp_path: Path) -> None:
    cppipe = tmp_path / "flag-image-rows.cppipe"
    cppipe.write_text(
        "\n".join(
            (
                "CellProfiler Pipeline: https://cellprofiler.org",
                "FlagImage:[module_num:9|enabled:True]",
                "    Flag count:2",
                "    Name the flag's category:Metadata",
                "    Name the flag:AnyFailure",
                "    How should measurements be linked?:Flag if any fail",
                "    Skip image set if flagged?:No",
                "    Measurement count:2",
                "    Flag is based on:Whole-image measurement",
                "    Select the object to be used for flagging:Image",
                "    Which measurement?:Intensity_MeanIntensity_DNA",
                "    Flag images based on low values?:No",
                "    Minimum value:0.0",
                "    Flag images based on high values?:Yes",
                "    Maximum value:1.0",
                "    Flag is based on:Whole-image measurement",
                "    Select the object to be used for flagging:Image",
                "    Which measurement?:Intensity_MeanIntensity_DNA",
                "    Flag images based on low values?:No",
                "    Minimum value:0.0",
                "    Flag images based on high values?:No",
                "    Maximum value:1.0",
                "    Name the flag's category:QC",
                "    Name the flag:AllFailures",
                "    How should measurements be linked?:Flag if all fail",
                "    Skip image set if flagged?:No",
                "    Measurement count:2",
                "    Flag is based on:Whole-image measurement",
                "    Select the object to be used for flagging:Image",
                "    Which measurement?:Intensity_MeanIntensity_DNA",
                "    Flag images based on low values?:No",
                "    Minimum value:0.0",
                "    Flag images based on high values?:Yes",
                "    Maximum value:1.0",
                "    Flag is based on:Whole-image measurement",
                "    Select the object to be used for flagging:Image",
                "    Which measurement?:Intensity_MeanIntensity_DNA",
                "    Flag images based on low values?:No",
                "    Minimum value:0.0",
                "    Flag images based on high values?:No",
                "    Maximum value:1.0",
                "    Ignore flag skips on last cycle?:No",
            )
        ),
        encoding="utf-8",
    )
    (module,) = CPPipeParser(cppipe).parse()

    plan = FlagImageModule.plan(module)
    bound = FlagImageModule.bind_settings(module, binder=SettingsBinder())

    assert FlagImageModule.setting_bindings == FlagImagePlan.setting_bindings()
    assert plan.public_kwargs() == _flag_kwargs()
    assert bound.kwargs == _flag_kwargs()


def test_flag_image_skip_setting_fails_without_disposition_authority() -> None:
    with pytest.raises(NotImplementedError, match="pipeline-disposition authority"):
        inspect.unwrap(flag_image)(
            np.zeros((2, 2), dtype=np.float32),
            **_flag_kwargs(wants_skip=(True, False)),
            measurement_tables=(
                _table(
                    "ImageMeasurements",
                    {
                        "slice_index": (0,),
                        "Intensity_MeanIntensity_DNA": (2.0,),
                    },
                    MeasurementSubject(MeasurementScope.IMAGE, "Image"),
                ),
            ),
        )
