"""
Converted from CellProfiler: MeasureImageAreaOccupied
Measures the total area in an image that is occupied by objects or foreground.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Annotated, ClassVar, Optional, Sequence, Tuple
from dataclasses import dataclass, replace
from enum import Enum
from openhcs.core.artifacts import (
    ArtifactSpec,
    ArtifactType,
    ImageArtifactType,
    ObjectLabelsArtifactType,
)
from openhcs.core.callable_contract import (
    KeywordRuntimeParameter,
    processing_prepare,
)
from openhcs.core.memory.decorators import numpy
from openhcs.core.measurement_row_materialization import (
    DataclassMeasurementColumnarRows,
    MeasurementSparseColumnarRows,
)
from openhcs.core.runtime_batch_contracts import SliceIndexRuntimeParameter
from openhcs.core.runtime_slice_projection import RuntimeSliceProjection
from openhcs.core.runtime_tabular_values import (
    FieldSpec,
)
from openhcs.core.runtime_measurements import (
    MeasurementRowAxisField,
    RuntimeMeasurementFeature,
)
from openhcs.core.runtime_plane_projection import (
    RuntimeSliceInvariantValue,
    RuntimePlaneAxis,
    RuntimePlaneAxisValueProjection,
)
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import image_payload_data, image_payload_metadata
from openhcs.core.runtime_object_labels import (
    ObjectLabelValue,
    object_label_dense_array,
)
from openhcs.processing.backends.analysis.region_properties import (
    binary_area_and_perimeter_2d,
    label_area_and_rounded_perimeter_2d,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract
from openhcs.core.pipeline.function_contracts import (
    ObjectLabelInputExecutionMode,
    composed_image_payload,
    object_label_input_execution_mode,
    resolved_callable_parameter,
    runtime_bound_parameters,
    special_inputs,
    )
from openhcs.interop.cellprofiler.module_settings import (
    BoundModuleSettings,
)
from openhcs.interop.cellprofiler.module_artifact_declarations import (
    MeasurementArtifactOutputModule,
    ObjectArtifactInputModule,
    SourceQualifiedMeasurementFeatureModule,
)
from openhcs.interop.cellprofiler.setting_names import (
    SettingNameFamily,
    block_setting_value,
    normalized_symbol_name,
    repeating_setting_blocks,
    setting_name_matches,
    setting_names,
    setting_values,
    split_symbol_names,
)
from openhcs.interop.cellprofiler.parser import ModuleBlock, ModuleSetting
from openhcs.interop.cellprofiler.settings_binder import (
    SettingToKeywordBinding,
    coerce_cellprofiler_enum,
)
from openhcs.interop.cellprofiler.runtime.object_input_policies import (
    CellProfilerObjectInputPolicyMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_recording import (
    FieldDerivedMeasurementFeatureModule,
    MeasurementFeatureRecord,
    NoObjectNameMeasurementRecordMixin,
    SourceQualifiedInputPayloadMeasurementRecordMixin,
)
from openhcs.interop.cellprofiler.runtime.measurement_rows import (
    ModuleOwnedResultMeasurementRows,
)

if TYPE_CHECKING:
    from openhcs.core.function_patterns import (
        FunctionInvocationKey,
        NormalizedFunctionItem,
    )
    from openhcs.core.invocation_artifacts import ArtifactDeclarationStepContext
    from openhcs.interop.cellprofiler.runtime.artifact_binding import (
        RuntimeInputBindingRequest,
    )


class OperandChoice(Enum):
    BINARY_IMAGE = "binary_image"
    OBJECTS = "objects"

    @classmethod
    def from_literal(cls, value: "OperandChoice | str") -> "OperandChoice":
        return coerce_cellprofiler_enum(cls, value)


@dataclass(frozen=True, slots=True)
class AreaOccupiedRow(RuntimeSliceInvariantValue):
    """One authoritative AreaOccupied operand and artifact-identity row."""

    operand: OperandChoice
    input_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.operand, OperandChoice):
            raise TypeError(
                "AreaOccupiedRow.operand must be OperandChoice, got "
                f"{type(self.operand).__name__}."
            )
        if not isinstance(self.input_name, str) or not self.input_name.strip():
            raise ValueError("AreaOccupiedRow.input_name must be a non-empty string.")
        if self.input_name != self.input_name.strip():
            raise ValueError("AreaOccupiedRow.input_name must already be normalized.")

    @property
    def artifact_type(self) -> type[ArtifactType]:
        if self.operand is OperandChoice.BINARY_IMAGE:
            return ImageArtifactType
        return ObjectLabelsArtifactType


class AreaOccupiedRowsRuntimeParameter(KeywordRuntimeParameter):
    """Runtime-bound AreaOccupied rows resolved from the compiled contract."""

    parameter_name = "area_occupied_rows"
    annotation_type = Sequence[AreaOccupiedRow]
    parameter_default = ()


class AreaOccupiedRuntimeInputPolicy(CellProfilerObjectInputPolicyMixin):
    """Bind AreaOccupied row identities from the declared artifact contract."""

    binds_without_declared_inputs = True

    @classmethod
    def bind_runtime_inputs(
        cls,
        request: "RuntimeInputBindingRequest",
    ) -> dict[str, object]:
        bound = super().bind_runtime_inputs(request)
        rows = cls._runtime_rows(request)
        return {
            **bound,
            AreaOccupiedRowsRuntimeParameter.require_parameter_name(): rows,
        }

    @classmethod
    def _runtime_rows(
        cls,
        request: "RuntimeInputBindingRequest",
    ) -> tuple[AreaOccupiedRow, ...]:
        operand_parameter_name = cls.operand_choices_binding.require_parameter_name()
        operand_literals = (
            request.kwargs[operand_parameter_name]
            if operand_parameter_name in request.kwargs
            else resolved_callable_parameter(
                request.func,
                operand_parameter_name,
            ).default
        )
        operands = tuple(
            OperandChoice.from_literal(value) for value in operand_literals
        )
        row_sources = cls._row_sources(request, operands)
        return tuple(
            AreaOccupiedRow(operand=operand, input_name=source.name)
            for operand, source in zip(operands, row_sources, strict=True)
        )

    @classmethod
    def _row_sources(
        cls,
        request: "RuntimeInputBindingRequest",
        operands: tuple[OperandChoice, ...],
    ) -> tuple[ArtifactSpec, ...]:
        image_inputs = iter(request.primary_image_inputs)
        object_inputs = iter(request.object_inputs)
        row_sources: list[ArtifactSpec] = []
        try:
            for operand in operands:
                if operand is OperandChoice.BINARY_IMAGE:
                    row_sources.append(next(image_inputs))
                elif operand is OperandChoice.OBJECTS:
                    row_sources.append(next(object_inputs))
        except StopIteration as exc:
            raise ValueError(
                "MeasureImageAreaOccupied operand choices do not match declared "
                "image/object artifact inputs."
            ) from exc
        remaining_images = tuple(image_inputs)
        remaining_objects = tuple(object_inputs)
        if remaining_images or remaining_objects:
            raise ValueError(
                "MeasureImageAreaOccupied declared artifact inputs exceed operand "
                f"rows: extra_images={[spec.name for spec in remaining_images]!r}, "
                f"extra_objects={[spec.name for spec in remaining_objects]!r}."
            )
        return tuple(row_sources)


class MeasureImageAreaOccupiedBinaryModule(
    AreaOccupiedRuntimeInputPolicy,
    NoObjectNameMeasurementRecordMixin,
    SourceQualifiedInputPayloadMeasurementRecordMixin,
    FieldDerivedMeasurementFeatureModule,
    SourceQualifiedMeasurementFeatureModule,
    ObjectArtifactInputModule,
    MeasurementArtifactOutputModule,
):
    module_name = "MeasureImageAreaOccupiedBinary"
    function_name = "measure_image_area_occupied"
    validated = True
    aliases = ("MeasureImageAreaOccupied",)
    function_variants = (
        "measure_image_volume_occupied_binary",
        "measure_image_volume_occupied_objects",
    )
    confidence = 1.0
    measurement_category_prefixes = (("area", "occupied"),)
    measurement_feature_family = "AreaOccupied"
    ignored_settings = ("Hidden",)

    class MeasurementFeature(RuntimeMeasurementFeature):
        """Feature families emitted by MeasureImageAreaOccupied."""

        AREA_OCCUPIED = "AreaOccupied"
        PERIMETER = "Perimeter"
        TOTAL_AREA = "TotalArea"

    @dataclass(frozen=True, slots=True)
    class MeasurementRows(ModuleOwnedResultMeasurementRows):
        """Project AreaOccupied result records into source-qualified CP features."""

        default_source_image_name: str | None = None

        @classmethod
        def for_request(cls, module_type, request):
            return cls(
                request.output_value,
                module_type=module_type,
                default_source_image_name=request.source.source_image_name,
            )

        def rows(self) -> MeasurementSparseColumnarRows:
            source_rows = self.source_rows()
            source_fields = {
                field_spec.name: field_spec for field_spec in source_rows.fields
            }
            axis_names = MeasurementRowAxisField.field_names()
            feature_fields = tuple(
                field_spec
                for field_spec in source_rows.fields
                if field_spec.name not in axis_names
            )
            projected_rows: list[dict[str, object]] = []
            slice_field_name = MeasurementRowAxisField.SLICE_INDEX.value
            source_field_name = MeasurementRowAxisField.SOURCE_IMAGE_NAME.value
            projected_fields: list[FieldSpec] = [
                source_fields[slice_field_name],
                FieldSpec(source_field_name, str, required=False),
            ]
            for source_record in source_rows.iter_row_mappings():
                source_image_name = source_record.get(
                    source_field_name,
                    self.default_source_image_name,
                )
                if source_image_name is None:
                    raise ValueError(
                        "MeasureImageAreaOccupied requires exact source-image "
                        "ownership for every measurement row."
                    )
                source_image_name = str(source_image_name)
                projected_row = {
                    slice_field_name: source_record[slice_field_name],
                    source_field_name: source_image_name,
                }
                for field_spec in feature_fields:
                    feature_name = self.module_type.measurement_feature_name(
                        field_spec.name,
                        source_image_name,
                    )
                    projected_row[feature_name] = source_record[field_spec.name]
                    projected_fields.append(
                        FieldSpec(feature_name, field_spec.dtype, required=False)
                    )
                projected_rows.append(projected_row)
            return MeasurementSparseColumnarRows.from_rows(
                projected_rows,
                fields=FieldSpec.merge_exact(
                    (projected_fields,),
                    context="MeasureImageAreaOccupied fields",
                ),
            )

    mode_setting = SettingNameFamily(
        "Measure the area occupied in a binary image, or in objects?",
        aliases=("Measure the area occupied by",),
    )
    binary_image_setting = SettingNameFamily(
        "Select a binary image to measure", aliases=("Select binary images to measure",)
    )
    objects_setting = SettingNameFamily(
        "Select objects to measure", aliases=("Select object sets to measure",)
    )
    operand_choices_binding = SettingToKeywordBinding(
        mode_setting,
        "operand_choices",
    )
    binary_image_binding = SettingToKeywordBinding.input(
        binary_image_setting, ImageArtifactType, repeated=True
    )
    objects_binding = SettingToKeywordBinding.input(
        objects_setting,
        ObjectLabelsArtifactType,
        runtime_parameter_name="object_labels",
        repeated=True,
    )
    setting_bindings: ClassVar[tuple[SettingToKeywordBinding, ...]] = (
        binary_image_binding,
        objects_binding,
        operand_choices_binding,
    )

    @classmethod
    def postprocess_bound_settings(
        cls,
        module: "ModuleBlock",
        bound: BoundModuleSettings,
    ) -> BoundModuleSettings:
        rows = cls.measurement_rows(module)
        kwargs: dict[str, object] = {
            cls.operand_choices_binding.require_parameter_name(): tuple(
                row.operand.value for row in rows
            ),
        }
        for binding, operand in (
            (cls.binary_image_binding, OperandChoice.BINARY_IMAGE),
            (cls.objects_binding, OperandChoice.OBJECTS),
        ):
            names = tuple(row.input_name for row in rows if row.operand is operand)
            if names:
                kwargs[binding.require_parameter_name()] = (
                    names[0] if len(names) == 1 else names
                )
        return bound.with_kwargs(kwargs)

    @classmethod
    def _operand_choices(
        cls,
        module: ModuleBlock,
    ) -> tuple[OperandChoice, ...]:
        operands = tuple(
            cls._operand_from_literal(value)
            for value in setting_values(module, cls.mode_setting)
        )
        if not operands:
            raise ValueError(
                "MeasureImageAreaOccupied requires at least one operand choice."
            )
        return operands

    @classmethod
    def active_artifact_bindings(
        cls,
        module: ModuleBlock | None = None,
        *,
        invocation_key: FunctionInvocationKey | None = None,
    ) -> tuple[SettingToKeywordBinding, ...]:
        bindings = super().active_artifact_bindings(
            module,
            invocation_key=invocation_key,
        )
        if module is None:
            return bindings
        operands = frozenset(cls._operand_choices(module))
        return tuple(
            binding
            for binding in bindings
            if OperandChoice.BINARY_IMAGE in operands
            or binding is not cls.binary_image_binding
            if OperandChoice.OBJECTS in operands
            or binding is not cls.objects_binding
        )

    @classmethod
    def finalize_module_blocks_for_invocation(
        cls,
        blocks, *,
        invocation: NormalizedFunctionItem,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ModuleBlock, ...]:
        """Reconstruct ordered AreaOccupied rows from public behavior and flow."""

        blocks = super().finalize_module_blocks_for_invocation(
            blocks, invocation=invocation,
            step_context=step_context,
        )
        reconstructed_blocks = tuple(
            reconstructed
            for block in blocks
            if (reconstructed := cls._block_with_measurement_rows(block))
            is not None
        )
        return reconstructed_blocks

    @classmethod
    def _block_with_measurement_rows(cls, block: ModuleBlock) -> ModuleBlock | None:
        operands = cls._operand_choices(block)
        binary_names = cls._row_names(
            block,
            setting=cls.binary_image_setting,
        )
        object_names = cls._row_names(
            block,
            setting=cls.objects_setting,
        )
        if (
            len(binary_names)
            != sum(operand is OperandChoice.BINARY_IMAGE for operand in operands)
            or len(object_names)
            != sum(operand is OperandChoice.OBJECTS for operand in operands)
        ):
            return None
        binary_iter = iter(binary_names)
        object_iter = iter(object_names)

        records: list[ModuleSetting] = []
        records.extend(
            record
            for record in block.iter_settings()
            if not (
                setting_name_matches(
                    record.name,
                    cls.operand_choices_binding.setting_name,
                )
                or setting_name_matches(
                    record.name,
                    cls.binary_image_binding.setting_name,
                )
                or setting_name_matches(
                    record.name,
                    cls.objects_binding.setting_name,
                )
            )
        )
        for operand in operands:
            input_name = (
                next(binary_iter)
                if operand is OperandChoice.BINARY_IMAGE
                else next(object_iter)
            )
            records.extend(
                cls._setting_records_for_row(
                    operand,
                    input_name=input_name,
                )
            )
        return replace(
            block,
            setting_records=records,
        )

    @classmethod
    def _row_names(
        cls,
        block: ModuleBlock,
        *,
        setting: str | SettingNameFamily,
    ) -> tuple[str, ...]:
        return tuple(
            name
            for value in setting_values(block, setting)
            if (name := normalized_symbol_name(value)) is not None
        )

    @classmethod
    def measurement_rows(cls, module: "ModuleBlock") -> tuple[AreaOccupiedRow, ...]:
        rows: list[AreaOccupiedRow] = []
        for block in repeating_setting_blocks(
            module.iter_settings(), start_name=cls.mode_setting
        ):
            rows.extend(cls._expanded_rows_from_block(module, block))
        return tuple(rows)

    @classmethod
    def _expanded_rows_from_block(
        cls, module: "ModuleBlock", block: Sequence["ModuleSetting"]
    ) -> tuple[AreaOccupiedRow, ...]:
        operand = cls._operand_from_literal(
            block_setting_value(block, cls.mode_setting)
        )
        binary_image_name = normalized_symbol_name(
            block_setting_value(block, cls.binary_image_setting)
        )
        objects_name = normalized_symbol_name(
            block_setting_value(block, cls.objects_setting)
        )
        input_name = cls._input_name_for_operand(
            module,
            operand,
            binary_image_name=binary_image_name,
            objects_name=objects_name,
        )
        expanded_input_names = cls._expanded_input_names(
            module, operand, input_name=input_name, block=block
        )
        return tuple(
            (
                AreaOccupiedRow(
                    operand=operand,
                    input_name=expanded_input_name,
                )
                for expanded_input_name in expanded_input_names
            )
        )

    @classmethod
    def _operand_from_literal(cls, value: str) -> OperandChoice:
        try:
            return OperandChoice.from_literal(value)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported MeasureImageAreaOccupied mode {value!r}."
            ) from exc

    @classmethod
    def _setting_records_for_row(
        cls,
        operand: OperandChoice,
        *,
        input_name: str,
    ) -> tuple[ModuleSetting, ...]:
        binary_image_name = (
            input_name if operand is OperandChoice.BINARY_IMAGE else "None"
        )
        objects_name = input_name if operand is OperandChoice.OBJECTS else "None"
        return (
            ModuleSetting(
                setting_names(cls.mode_setting)[0],
                "Binary image" if operand is OperandChoice.BINARY_IMAGE else "Objects",
            ),
            ModuleSetting(
                setting_names(cls.binary_image_setting)[0], binary_image_name
            ),
            ModuleSetting(setting_names(cls.objects_setting)[0], objects_name),
        )

    @classmethod
    def _input_name_for_operand(
        cls,
        module: "ModuleBlock",
        operand: OperandChoice,
        *,
        binary_image_name: str | None,
        objects_name: str | None,
    ) -> str:
        if operand is OperandChoice.BINARY_IMAGE:
            input_name = binary_image_name
            role = "binary image"
        else:
            input_name = objects_name
            role = "object"
        if input_name is None:
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an area-occupied row with no {role} input."
            )
        return input_name

    @classmethod
    def _expanded_input_names(
        cls,
        module: "ModuleBlock",
        operand: OperandChoice,
        *,
        input_name: str,
        block: Sequence["ModuleSetting"],
    ) -> tuple[str, ...]:
        if operand is OperandChoice.BINARY_IMAGE:
            names = split_symbol_names(
                block_setting_value(block, cls.binary_image_setting)
            )
        else:
            names = split_symbol_names(block_setting_value(block, cls.objects_setting))
        if not names:
            raise ValueError(
                f"Module {module.name}({module.module_num}) has an area-occupied "
                f"row whose selected input {input_name!r} cannot be reconstructed."
            )
        return names

    @classmethod
    def artifact_contract_inputs(
        cls,
        module: ModuleBlock,
        *,
        invocation_key: FunctionInvocationKey,
        step_context: ArtifactDeclarationStepContext,
    ) -> tuple[ArtifactSpec, ...]:
        """Preserve mixed image/object row order for runtime row alignment."""

        rows = cls.measurement_rows(module)
        if not rows:
            raise ValueError(
                f"Module {module.name}({module.module_num}) declares no MeasureImageAreaOccupied measurement rows."
            )
        return tuple(
            cls.require_available_artifact_input(
                module,
                binding=(
                    cls.binary_image_binding
                    if row.operand is OperandChoice.BINARY_IMAGE
                    else cls.objects_binding
                ),
                name=row.input_name,
                invocation_key=invocation_key,
                step_context=step_context,
            )
            for row in rows
        )


@dataclass
class AreaOccupiedMeasurement(MeasurementFeatureRecord):
    """Measurements for area occupied analysis."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    area_occupied: float
    perimeter: float
    total_area: float
    source_image_name: Annotated[
        str | None,
        MeasurementRowAxisField.SOURCE_IMAGE_NAME,
    ] = None

    @classmethod
    def from_area(
        cls,
        *,
        area_occupied: float,
        perimeter: float,
        total_area: float,
        slice_index: int = 0,
        source_image_name: str | None = None,
    ) -> "AreaOccupiedMeasurement":
        return cls(
            slice_index=slice_index,
            area_occupied=area_occupied,
            perimeter=perimeter,
            total_area=total_area,
            source_image_name=source_image_name,
        )


@dataclass(frozen=True, slots=True)
class BinaryAreaOccupiedRequest:
    """Measure occupied area for one binary image plane."""

    image: np.ndarray
    slice_index: int = 0
    source_image_name: str | None = None

    def measure(self) -> AreaOccupiedMeasurement:
        binary_mask = self.image > 0
        area_occupied, perimeter_value = binary_area_and_perimeter_2d(binary_mask)
        return AreaOccupiedMeasurement.from_area(
            area_occupied=area_occupied,
            perimeter=perimeter_value,
            total_area=float(np.prod(self.image.shape)),
            slice_index=self.slice_index,
            source_image_name=self.source_image_name,
        )


@dataclass(frozen=True, slots=True)
class ObjectLabelsAreaOccupiedRequest:
    """Measure occupied area for one object-label plane."""

    labels: ObjectLabelValue
    slice_index: int = 0
    source_image_name: str | None = None

    def measure(self) -> AreaOccupiedMeasurement:
        label_array = object_label_dense_array(self.labels)
        if label_array.ndim != 2:
            raise ValueError(
                "MeasureImageAreaOccupied requires object labels already projected "
                f"to one 2-D plane, got shape {label_array.shape!r}."
            )
        area_occupied, perimeter_value = label_area_and_perimeter(label_array)
        return AreaOccupiedMeasurement.from_area(
            area_occupied=area_occupied,
            perimeter=perimeter_value,
            total_area=float(np.prod(label_array.shape)),
            slice_index=self.slice_index,
            source_image_name=self.source_image_name,
        )


@composed_image_payload
@numpy(contract=ProcessingContract.FLEXIBLE)
@special_inputs("object_labels")
@runtime_bound_parameters(
    AreaOccupiedRowsRuntimeParameter,
    SliceIndexRuntimeParameter,
)
def measure_image_area_occupied(
    image: np.ndarray,
    *,
    operand_choices: Sequence[OperandChoice] = (OperandChoice.BINARY_IMAGE,),
    area_occupied_rows: Sequence[AreaOccupiedRow] = (),
    object_labels: Sequence[ObjectLabelValue] = (),
    slice_by_slice: bool = True,
    slice_index: int | None = None,
) -> tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """Measure area occupied for ordered binary-image and object rows.

    Args:
        object_labels: Object-label planes for rows whose operand is an object
            set, in the same order as those rows in ``area_occupied_rows``.
    """
    rows = tuple(area_occupied_rows)
    if not rows:
        raise ValueError(
            "MeasureImageAreaOccupied requires runtime-bound AreaOccupiedRow "
            "declarations."
        )
    if any(not isinstance(row, AreaOccupiedRow) for row in rows):
        raise TypeError(
            "MeasureImageAreaOccupied area_occupied_rows must contain only "
            "AreaOccupiedRow values."
        )
    configured_operands = tuple(operand_choices)
    row_operands = tuple(row.operand for row in rows)
    if configured_operands != row_operands:
        raise ValueError(
            "MeasureImageAreaOccupied runtime rows do not match configured "
            f"operand choices: {row_operands!r} != {configured_operands!r}."
        )
    binary_images = _binary_images_from_payload(
        image,
        tuple(
            row.input_name for row in rows if row.operand is OperandChoice.BINARY_IMAGE
        ),
    )
    expected_object_rows = sum((row.operand is OperandChoice.OBJECTS for row in rows))
    if len(object_labels) != expected_object_rows:
        raise ValueError(
            "MeasureImageAreaOccupied object_labels count must match object rows: "
            f"got {len(object_labels)} object label input(s) for "
            f"{expected_object_rows} object row(s) from "
            f"operand_choices={tuple(row.operand.value for row in rows)!r}, "
            f"row_sources={tuple(row.input_name for row in rows)!r}; "
            f"object_labels_type={type(object_labels).__name__}, "
            f"object_label_item_types={tuple(type(label).__name__ for label in object_labels)!r}."
        )
    measurements = []
    binary_index = 0
    object_index = 0
    measurement_slice_index = 0 if slice_index is None else int(slice_index)
    for row in rows:
        if row.operand is OperandChoice.BINARY_IMAGE:
            measurement = BinaryAreaOccupiedRequest(
                image=binary_images[binary_index],
                slice_index=measurement_slice_index,
                source_image_name=row.input_name,
            ).measure()
            binary_index += 1
        else:
            labels = object_labels[object_index]
            measurement = ObjectLabelsAreaOccupiedRequest(
                labels=labels,
                slice_index=measurement_slice_index,
                source_image_name=row.input_name,
            ).measure()
            object_index += 1
        measurements.append(measurement)
    measurement_rows = DataclassMeasurementColumnarRows(
        tuple(measurements),
        row_type=AreaOccupiedMeasurement,
    )
    return (image, measurement_rows)


def _binary_images_from_payload(
    image: RuntimeArrayData,
    binary_image_names: tuple[str, ...],
) -> tuple[np.ndarray, ...]:
    if not binary_image_names:
        return ()
    metadata = image_payload_metadata(image)
    if metadata.plane_axis is None:
        if len(binary_image_names) != 1:
            raise ValueError(
                "MeasureImageAreaOccupied requires a declared source-binding "
                "axis for multiple binary-image rows."
            )
        return (np.asarray(image_payload_data(image)),)
    if metadata.plane_axis is not RuntimePlaneAxis.SOURCE_BINDING:
        raise ValueError(
            "MeasureImageAreaOccupied binary-image composition requires a "
            "source-binding image axis, got "
            f"{metadata.plane_axis.value!r}."
        )
    source_aliases = metadata.source_image_names
    if source_aliases != binary_image_names:
        raise ValueError(
            "MeasureImageAreaOccupied binary-image rows must exactly match the "
            f"declared source aliases: {binary_image_names!r} != {source_aliases!r}."
        )
    projection = RuntimePlaneAxisValueProjection.preserve(
        axis=RuntimePlaneAxis.SOURCE_BINDING,
        axis_size=len(source_aliases),
        source_aliases=source_aliases,
    )
    projection.validate_shape(
        np.asarray(image_payload_data(image)).shape,
        value_name="MeasureImageAreaOccupied binary image payload",
    )
    return tuple(
        np.asarray(
            image_payload_data(
                RuntimeSliceProjection.value_for_slice(
                    image,
                    RuntimePlaneAxisValueProjection.from_selected_plane(
                        axis=RuntimePlaneAxis.SOURCE_BINDING,
                        plane_index=index,
                        axis_size=projection.axis_size,
                        source_aliases=projection.source_aliases,
                    ),
                )
            )
        )
        for index in range(projection.axis_size)
    )


def label_area_and_perimeter(labels: np.ndarray) -> tuple[float, float]:
    label_array = object_label_dense_array(labels)
    if label_array.ndim != 2:
        raise ValueError(
            "Area-occupied label measurement requires one projected 2-D plane, "
            f"got shape {label_array.shape!r}."
        )
    return _label_plane_area_and_perimeter(label_array)


def _label_plane_area_and_perimeter(labels: np.ndarray) -> tuple[float, float]:
    labels_array = object_label_dense_array(labels, dtype=np.int32)
    return label_area_and_rounded_perimeter_2d(labels_array)


@dataclass
class VolumeOccupiedMeasurement:
    """Measurements for volume occupied analysis (3D)."""

    slice_index: Annotated[int, MeasurementRowAxisField.SLICE_INDEX]
    volume_occupied: float
    surface_area: float
    total_volume: float

    @classmethod
    def from_volume(
        cls, *, volume_occupied: float, surface_area: float, total_volume: float
    ) -> "VolumeOccupiedMeasurement":
        return cls(
            slice_index=0,
            volume_occupied=volume_occupied,
            surface_area=surface_area,
            total_volume=total_volume,
        )


@dataclass(frozen=True, slots=True)
class SurfaceAreaRequest:
    """Compute rounded surface area for one 3D label image."""

    label_image: np.ndarray
    spacing: Optional[Tuple[float, ...]] = None

    def surface_area(self) -> float:
        from skimage.measure import marching_cubes, mesh_surface_area

        spacing = self.spacing
        label_image = np.asarray(self.label_image)
        if spacing is None:
            spacing = (1.0,) * label_image.ndim
        unique_labels = np.unique(label_image)
        unique_labels = unique_labels[unique_labels != 0]
        if len(unique_labels) == 0:
            return 0.0
        total_surface = 0.0
        for label in unique_labels:
            binary_mask = (label_image == label).astype(np.float32)
            try:
                verts, faces, _, _ = marching_cubes(
                    binary_mask, spacing=spacing, level=0.5, method="lorensen"
                )
                total_surface += mesh_surface_area(verts, faces)
            except (ValueError, RuntimeError):
                continue
        return float(np.round(total_surface))


@dataclass(frozen=True, slots=True)
class VolumeOccupiedRequest:
    """Materialize a volume-occupied measurement from voxel totals."""

    volume_occupied: float
    surface_area: float
    total_volume: float

    def measurement(self) -> VolumeOccupiedMeasurement:
        return VolumeOccupiedMeasurement.from_volume(
            volume_occupied=self.volume_occupied,
            surface_area=self.surface_area,
            total_volume=self.total_volume,
        )


VolumeOccupiedVoxelSpacingInput = Annotated[
    Optional[Tuple[float, float, float]],
    (
        "Physical voxel spacing in (z, y, x) order used to scale surface-area "
        "measurements."
    ),
]


@numpy(contract=ProcessingContract.PURE_3D)
def measure_image_volume_occupied_binary(
    image: np.ndarray, spacing: VolumeOccupiedVoxelSpacingInput = None
) -> Tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """
    Measure volume occupied by foreground in a 3D binary image.

    Args:
        image: 3D binary image (D, H, W) where foreground > 0

    Returns:
        Tuple of original image and exact volume-measurement rows.
    """
    binary_mask = image > 0
    volume_occupied = float(np.sum(binary_mask))
    if volume_occupied > 0:
        surface_area_value = SurfaceAreaRequest(
            binary_mask.astype(np.int32), spacing=spacing
        ).surface_area()
    else:
        surface_area_value = 0.0
    total_volume = float(np.prod(image.shape))
    measurement = VolumeOccupiedRequest(
        volume_occupied=volume_occupied,
        surface_area=surface_area_value,
        total_volume=total_volume,
    ).measurement()
    return (
        image,
        DataclassMeasurementColumnarRows(
            (measurement,),
            row_type=VolumeOccupiedMeasurement,
        ),
    )


@numpy(contract=ProcessingContract.PURE_3D)
@object_label_input_execution_mode(ObjectLabelInputExecutionMode.FULL_STACK)
@special_inputs(
    MeasureImageAreaOccupiedBinaryModule.objects_binding.require_runtime_parameter_name()
)
def measure_image_volume_occupied_objects(
    image: np.ndarray,
    object_labels: ObjectLabelValue,
    spacing: VolumeOccupiedVoxelSpacingInput = None,
) -> Tuple[np.ndarray, DataclassMeasurementColumnarRows]:
    """
    Measure volume occupied by labeled objects in 3D.

    Args:
        image: 3D intensity image (D, H, W)
        object_labels: 3D label image from segmentation (D, H, W)

    Returns:
        Tuple of original image and exact volume-measurement rows.
    """
    labels_array = object_label_dense_array(object_labels, dtype=np.int32)
    if labels_array.ndim != 3 or labels_array.shape != np.asarray(image).shape:
        raise ValueError(
            "MeasureImageVolumeOccupied requires labels already projected into "
            f"the 3-D image domain; got image {np.asarray(image).shape!r} and "
            f"labels {labels_array.shape!r}."
        )
    volume_occupied = float(np.count_nonzero(labels_array))
    if volume_occupied > 0:
        surface_area_value = SurfaceAreaRequest(
            labels_array, spacing=spacing
        ).surface_area()
    else:
        surface_area_value = 0.0
    total_volume = float(np.prod(labels_array.shape))
    measurement = VolumeOccupiedRequest(
        volume_occupied=volume_occupied,
        surface_area=surface_area_value,
        total_volume=total_volume,
    ).measurement()
    return (
        image,
        DataclassMeasurementColumnarRows(
            (measurement,),
            row_type=VolumeOccupiedMeasurement,
        ),
    )


@processing_prepare(measure_image_area_occupied)
def _prepare_measure_image_area_occupied() -> None:
    """Compile reusable area/perimeter kernels before timed execution."""
    binary = np.zeros((64, 64), dtype=np.float32)
    binary[8:40, 12:48] = 1.0
    labels = np.zeros((64, 64), dtype=np.int32)
    labels[8:24, 8:24] = 1
    labels[32:56, 32:56] = 2
    BinaryAreaOccupiedRequest(binary).measure()
    ObjectLabelsAreaOccupiedRequest(labels).measure()
