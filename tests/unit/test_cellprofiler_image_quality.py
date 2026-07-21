from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import get_type_hints

import numpy as np

from openhcs.constants.constants import AllComponents
from openhcs.constants.input_source import InputSource
from openhcs.core.aligned_image_payload import ImagePayloadExecutionMode
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactSpecCollection,
    ImageArtifactType,
    MeasurementsArtifactType,
)
from openhcs.core.callable_contract import CallableContract
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.function_patterns import (
    DEFAULT_GROUP_KEY,
    FunctionInvocationKey,
    normalize_function_pattern,
)
from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
from openhcs.core.measurement_row_materialization import (
    ConcatenatedColumnarRows,
    DataclassMeasurementColumnarRows,
    MeasurementProjectedColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_image_values import ImagePayloadMetadata
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_artifact_queries import MeasurementTableUnion
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    MeasurementScope,
    MeasurementSubject,
)
from openhcs.core.runtime_plane_projection import (
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.source_image_provenance import SourceImageProvenancePlanes
from openhcs.core.source_bindings import NamedSourceBinding, StepSourceBindingsConfig
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.image_set_numbering import (
    CellProfilerImageSetNumbering,
)
from openhcs.interop.cellprofiler.runtime.function_contract_execution import (
    CellProfilerFunctionContractExecutor,
)
from openhcs.interop.cellprofiler.runtime.invocation import CellProfilerImageRequest
from openhcs.processing.backends.cellprofiler.image_quality import (
    ImageQualityBlurMetrics,
    ImageQualityBlurSummaryMetrics,
    ImageQualityIntensityMetrics,
    ImageQualityOtsuObjective,
    ImageQualityScalingMetrics,
    ImageQualityThresholdMethod,
    ImageQualityThresholdMetrics,
    MeasureImageQualityModule,
    image_quality_intensity_metrics,
    measure_image_quality,
)
from openhcs.processing.backends.cellprofiler.thresholding import (
    CellProfilerOtsuMethod,
    CellProfilerThresholdAssignment,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


def _project_feature_rows(rows: ColumnarRows) -> ColumnarRows:
    return MeasureImageQualityModule.MeasurementRows(
        results=rows,
        module_type=MeasureImageQualityModule,
        source_metadata=ImagePayloadMetadata(
            source_image_names=("OrigHoechst",),
        ),
        plane_projection=None,
    ).rows()


def _feature_rows(*records) -> ColumnarRows:
    return _project_feature_rows(
        ConcatenatedColumnarRows(
            tuple(
                DataclassMeasurementColumnarRows((record,), row_type=type(record))
                for record in records
            )
        )
    )


def _compiled_image_quality_contract() -> CallableContract:
    measurement_spec = ArtifactSpec.output(
        "quality_metrics",
        MeasurementsArtifactType,
    )
    contract = CallableContract.from_callable(measure_image_quality)
    return replace(
        contract,
        metadata=replace(
            contract.metadata,
            artifact_outputs=(measurement_spec,),
        ),
    )


def test_image_quality_total_area_uses_native_float_scalar() -> None:
    annotations = get_type_hints(ImageQualityIntensityMetrics, include_extras=True)
    metrics = image_quality_intensity_metrics(np.ones((2, 3), dtype=np.float32))
    empty_metrics = image_quality_intensity_metrics(np.empty((0, 3), dtype=np.float32))

    assert annotations["total_area"] is float
    assert metrics.total_area == 6.0
    assert type(metrics.total_area) is float
    assert empty_metrics.total_area == 0.0
    assert type(empty_metrics.total_area) is float


def test_measure_image_quality_declares_per_plane_processing_contract() -> None:
    contract = CallableContract.from_callable(measure_image_quality)

    assert contract.processing_contract is ProcessingContract.PURE_2D


def test_measure_image_quality_executes_once_per_declared_image_input() -> None:
    contract = CallableContract.from_callable(measure_image_quality)

    assert MeasureImageQualityModule.executes_per_image_measurements(
        contract.resolve_canonical_raw_callable(),
        (),
        callable_contract=contract,
    )


def test_measure_image_quality_image_selection_resolves_from_its_typed_scope() -> None:
    source_bindings = StepSourceBindingsConfig(
        enabled=True,
        bindings=(
            NamedSourceBinding(alias="DNA"),
            NamedSourceBinding(alias="RNA"),
        ),
    )
    source_specs = ArtifactSpecCollection(
        binding.input_spec() for binding in source_bindings.binding_declarations
    )
    context = ArtifactDeclarationStepContext(
        step_name=MeasureImageQualityModule.module_name,
        step_index=0,
        source_bindings=source_bindings,
        input_source=InputSource.PIPELINE_START,
        available_artifacts=source_specs,
        main_flow_artifacts=source_specs,
    )
    target_module = ModuleBlock(
        name=MeasureImageQualityModule.module_name,
        module_num=2,
        setting_records=[
            ModuleSetting(
                MeasureImageQualityModule.image_selection_setting,
                MeasureImageQualityModule.all_loaded_images_selection,
            )
        ],
    )
    invocation_key = FunctionInvocationKey(
        function_name=MeasureImageQualityModule.function_name,
        group_key=DEFAULT_GROUP_KEY,
        position=0,
    )

    target_inputs = MeasureImageQualityModule.artifact_inputs_for_binding(
        target_module,
        binding=MeasureImageQualityModule.image_measurement_binding,
        invocation_key=invocation_key,
        step_context=context,
    )
    explicit_module = replace(
        target_module,
        setting_records=[
            ModuleSetting(
                MeasureImageQualityModule.image_selection_setting,
                MeasureImageQualityModule.selected_images_selection,
            ),
            ModuleSetting(
                MeasureImageQualityModule.selected_images_setting,
                "RNA",
            ),
        ],
    )
    explicit_inputs = MeasureImageQualityModule.artifact_inputs_for_binding(
        explicit_module,
        binding=MeasureImageQualityModule.image_measurement_binding,
        invocation_key=invocation_key,
        step_context=context,
    )
    invocation = next(normalize_function_pattern(measure_image_quality).iter_items())
    blocks, consumed = MeasureImageQualityModule.module_blocks_for_invocation(
        invocation=invocation,
        step_context=context,
    )
    (numbered_blocks,), _next_module_num = (
        MeasureImageQualityModule.number_step_invocation_blocks(
            (blocks,),
            first_module_num=target_module.module_num,
        )
    )
    reconstructed, reconstructed_consumed = (
        MeasureImageQualityModule.invocation_callable_contract(
            invocation=invocation,
            numbered_module_blocks=numbered_blocks,
            consumed_kwarg_names=consumed,
            step_context=context,
        )
    )

    assert consumed == reconstructed_consumed == ()
    assert tuple(spec.name for spec in target_inputs) == ("DNA", "RNA")
    assert tuple(spec.name for spec in explicit_inputs) == ("RNA",)
    assert reconstructed.artifact_inputs.names_of_artifact_type(ImageArtifactType) == (
        "DNA",
        "RNA",
    )


def test_measure_image_quality_executes_blur_metrics_for_one_2d_plane() -> None:
    image = np.arange(64, dtype=np.float32).reshape((8, 8)) + 1.0

    _output, records = measure_image_quality.__wrapped__(
        image,
        include_scaling=False,
        calculate_saturation=False,
        calculate_intensity=False,
        calculate_threshold=False,
        blur_scales=(2,),
    )

    assert tuple(batch.row_type for batch in records.row_batches) == (
        ImageQualityBlurSummaryMetrics,
        ImageQualityBlurMetrics,
    )


def test_measure_image_quality_pure_2d_contract_slices_3d_site_batch() -> None:
    planes = np.stack(
        (
            np.arange(64, dtype=np.float32).reshape((8, 8)) + 1.0,
            np.arange(64, dtype=np.float32).reshape((8, 8)) + 2.0,
        )
    )
    image = ImagePayloadMetadata(
        plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
    ).payload_with(planes, None)
    contract = _compiled_image_quality_contract()

    _output, records = CellProfilerFunctionContractExecutor().execute(
        contract,
        contract.resolve_canonical_raw_callable(),
        image,
        {
            "include_scaling": False,
            "calculate_saturation": False,
            "calculate_intensity": False,
            "calculate_threshold": False,
            "blur_scales": (2,),
        },
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=RuntimePlaneAxisValueProjection.preserve(
            axis=RuntimePlaneAxis.RUNTIME_SLICE,
            axis_size=2,
        ),
    )

    assert tuple(row["slice_index"] for row in records.row_mappings()) == (0, 0, 1, 1)


def test_measure_image_quality_projects_named_source_axis_from_runtime_metadata() -> (
    None
):
    source_names = ("OrigBlue", "OrigGreen", "OrigRed")
    source_metadata = ImagePayloadMetadata(
        source_image_names=source_names,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/plate/A01_{name}.tif" for name in source_names),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )
    image = source_metadata.payload_with(
        np.stack(
            tuple(
                np.full((8, 8), channel_index + 1, dtype=np.float32)
                for channel_index in range(len(source_names))
            )
        )
    )
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=len(source_names),
        source_aliases=source_names,
    )
    contract = _compiled_image_quality_contract()

    _output, records = CellProfilerFunctionContractExecutor().execute(
        contract,
        contract.resolve_canonical_raw_callable(),
        image,
        {
            "include_scaling": False,
            "calculate_blur": False,
            "calculate_saturation": False,
            "calculate_intensity": True,
            "calculate_threshold": False,
        },
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=projection,
    )
    rows = MeasureImageQualityModule.MeasurementRows.for_request(
        MeasureImageQualityModule,
        SimpleNamespace(
            output_value=records,
            source=CellProfilerImageRequest(
                payload=image,
                source_image_name=None,
                source_aliases=source_names,
                image_count=len(source_names),
                execution_mode=ImagePayloadExecutionMode.NATURAL,
                plane_projection=projection,
            ),
        ),
    ).rows()

    assert rows.column_values(MeasurementRowAxisField.SOURCE_IMAGE_NAME.value) == (
        "OrigBlue",
        "OrigGreen",
        "OrigRed",
    )
    for source_name, row in zip(source_names, rows.row_mappings(), strict=True):
        assert f"ImageQuality_MeanIntensity_{source_name}" in row


def test_measure_image_quality_table_uses_row_qualified_source_inputs() -> None:
    source_names = ("OrigBlue", "OrigGreen", "OrigRed")
    source_specs = tuple(
        ArtifactSpec.input(
            source_name,
            ImageArtifactType,
            parameter_name="image",
        )
        for source_name in source_names
    )
    source_metadata = ImagePayloadMetadata(
        source_image_names=source_names,
        source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
            paths=tuple(f"/plate/A01_{name}.tif" for name in source_names),
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
    )
    image = source_metadata.payload_with(
        np.stack(
            tuple(
                np.full((4, 4), channel_index + 1, dtype=np.float32)
                for channel_index in range(len(source_names))
            )
        )
    )
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=len(source_names),
        source_aliases=source_names,
    )
    contract = _compiled_image_quality_contract()
    contract = replace(
        contract,
        metadata=replace(
            contract.metadata,
            artifact_inputs=source_specs,
        ),
    )
    _output, records = CellProfilerFunctionContractExecutor().execute(
        contract,
        contract.resolve_canonical_raw_callable(),
        image,
        {
            "include_scaling": False,
            "calculate_blur": False,
            "calculate_saturation": False,
            "calculate_intensity": True,
            "calculate_threshold": False,
        },
        execution_mode=ImagePayloadExecutionMode.NATURAL,
        plane_projection=projection,
    )
    requested_source_specs: list[tuple[ArtifactSpec, ...]] = []

    def measurement_source_metadata(
        specs: tuple[ArtifactSpec, ...],
    ) -> ImagePayloadMetadata:
        requested_source_specs.append(specs)
        return source_metadata

    table = MeasureImageQualityModule.measurement_table(
        SimpleNamespace(
            callable_contract=contract,
            output_value=records,
            source=CellProfilerImageRequest(
                payload=image,
                source_image_name=None,
                source_aliases=source_names,
                image_count=len(source_names),
                execution_mode=ImagePayloadExecutionMode.NATURAL,
                plane_projection=projection,
            ),
            spec=contract.artifact_outputs.specs[0],
            measurement_source_metadata=measurement_source_metadata,
        )
    )

    assert requested_source_specs == [source_specs]
    assert table.source_image_name is None
    assert table.subject == MeasurementSubject(
        MeasurementScope.IMAGE,
        MeasurementScope.IMAGE.value,
    )
    assert (
        tuple(table.rows.column_values(MeasurementRowAxisField.SOURCE_IMAGE_NAME.value))
        == source_names
    )


def test_quality_control_measurements_keep_distinct_site_image_identities() -> None:
    tables = tuple(
        MeasurementTable(
            name="quality_metrics",
            rows=MeasurementSparseColumnarRows.from_rows(
                (
                    {"slice_index": 0, "focus_score": float(channel)},
                    {"slice_index": 1, "focus_score": float(channel + 1)},
                ),
                fields=(
                    FieldSpec("slice_index", int),
                    FieldSpec("focus_score", float),
                ),
            ),
            source_image_name=f"Channel{channel}",
            subject=MeasurementSubject(
                MeasurementScope.IMAGE,
                f"Channel{channel}",
            ),
            measurement_feature_owner=MeasureImageQualityModule,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=(
                        f"/plate/A01_s001_w{channel}.tif",
                        f"/plate/A01_s002_w{channel}.tif",
                    ),
                    component_metadata=(
                        {
                            "well": "A01",
                            "site": "1",
                            "channel": str(channel),
                        },
                        {
                            "well": "A01",
                            "site": "2",
                            "channel": str(channel),
                        },
                    ),
                )
            ),
            source_image_names=(f"Channel{channel}", f"Channel{channel}"),
        )
        for channel in range(1, 6)
    )
    union = MeasurementTableUnion("quality_metrics", tables)
    table = MeasureImageQualityModule.build_measurement_table(
        name="quality_metrics",
        rows=union.rows(),
        object_name=None,
        source_image_name=None,
        source_metadata=union.source_metadata(),
    )

    projected_rows = CellProfilerImageSetNumbering(
        SourceImageSetIdentityPolicy(frozenset((AllComponents.CHANNEL,)))
    ).project_measurement_rows(
        scope=RuntimeExecutionAxisScope(axis_id="A01"),
        table=table,
    )

    assert tuple(projected_rows.column_values("slice_index")) == (
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
        1,
        2,
    )


def test_measure_image_quality_declares_exact_native_feature_names_and_scopes() -> None:
    rows = _feature_rows(
        ImageQualityBlurSummaryMetrics(
            focus_score=0.75,
            power_log_log_slope=-1.25,
        ),
        ImageQualityBlurMetrics(
            scale="20",
            local_focus_score=0.5,
            correlation=0.25,
        ),
        ImageQualityThresholdMetrics(
            slice_index=0,
            feature_name=ImageQualityThresholdMethod.OTSU.feature_field_name,
            scale=ImageQualityThresholdMethod.OTSU.descriptor_scale(
                otsu_class_count=CellProfilerOtsuMethod.TWO_CLASS,
                otsu_objective=ImageQualityOtsuObjective.WEIGHTED_VARIANCE,
                assign_middle_to_foreground=(
                    CellProfilerThresholdAssignment.FOREGROUND
                ),
            ),
            result_value=0.4,
        ),
    )

    assert {field.name for field in rows.fields} == {
        "slice_index",
        "source_image_name",
        "ImageQuality_FocusScore_OrigHoechst",
        "ImageQuality_PowerLogLogSlope_OrigHoechst",
        "ImageQuality_LocalFocusScore_OrigHoechst_20",
        "ImageQuality_Correlation_OrigHoechst_20",
        "ImageQuality_ThresholdOtsu_OrigHoechst_2W",
    }
    assert rows.row_mappings() == (
        {
            "slice_index": 0,
            "source_image_name": "OrigHoechst",
            "ImageQuality_FocusScore_OrigHoechst": 0.75,
            "ImageQuality_PowerLogLogSlope_OrigHoechst": -1.25,
            "ImageQuality_LocalFocusScore_OrigHoechst_20": 0.5,
            "ImageQuality_Correlation_OrigHoechst_20": 0.25,
            "ImageQuality_ThresholdOtsu_OrigHoechst_2W": 0.4,
        },
    )


def test_image_quality_finished_feature_rows_are_not_projected_twice() -> None:
    rows = _feature_rows(ImageQualityBlurSummaryMetrics(focus_score=0.75))

    prepared = MeasureImageQualityModule.prepare_measurement_record_rows(
        rows,
        source_image_name="OrigHoechst",
    )

    assert prepared.fields == rows.fields
    assert prepared.row_mappings() == rows.row_mappings()


def test_measure_image_quality_emits_only_enabled_nominal_feature_families() -> None:
    image = np.arange(64, dtype=np.uint16).reshape((8, 8))
    _output, records = measure_image_quality.__wrapped__(
        image,
        include_scaling=True,
        calculate_blur=False,
        calculate_saturation=False,
        calculate_intensity=True,
        calculate_threshold=False,
    )

    assert tuple(batch.row_type for batch in records.row_batches) == (
        ImageQualityScalingMetrics,
        ImageQualityIntensityMetrics,
    )
    rows = _project_feature_rows(records)
    row = rows.row_mappings()[0]
    assert row["ImageQuality_Scaling_OrigHoechst"] == 65535.0
    assert "ImageQuality_MADIntensity_OrigHoechst" in row
    assert "ImageQuality_MadIntensity_OrigHoechst" not in row


def test_image_quality_experiment_measurements_use_exact_columnar_schema() -> None:
    feature_name = "ImageQuality_ThresholdOtsu_OrigHoechst_2W"
    image_table = MeasurementTable(
        name="MeasureImageQuality_measurements",
        rows=MeasurementProjectedColumnarRows(
            {feature_name: (0.2, 0.4)},
            fields=(FieldSpec(feature_name, float, required=False),),
        ),
        subject=MeasurementSubject(MeasurementScope.IMAGE, "Image"),
        measurement_feature_owner=MeasureImageQualityModule,
    )

    (experiment_table,) = MeasureImageQualityModule.experiment_measurement_tables(
        (image_table,)
    )

    assert experiment_table.rows.fields == (
        FieldSpec(
            "ImageQuality_ThresholdMeanOtsu_OrigHoechst_2W",
            float,
            required=False,
        ),
        FieldSpec(
            "ImageQuality_ThresholdMedianOtsu_OrigHoechst_2W",
            float,
            required=False,
        ),
        FieldSpec(
            "ImageQuality_ThresholdStdOtsu_OrigHoechst_2W",
            float,
            required=False,
        ),
    )
    (experiment_row,) = experiment_table.rows.row_mappings()
    np.testing.assert_allclose(
        tuple(experiment_row.values()),
        (0.3, 0.3, 0.1),
    )
