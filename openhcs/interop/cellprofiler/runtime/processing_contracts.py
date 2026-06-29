"""Processing-contract and PURE_2D slice-count authorities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from openhcs.core.callable_contract import CallableContract
from openhcs.core.measurement_row_materialization import (
    MEASUREMENT_OBJECT_ID_FIELD,
    MEASUREMENT_OBJECT_LABEL_FIELD,
    MEASUREMENT_OBJECT_NAME_FIELD,
    MEASUREMENT_OBJECT_NUMBER_FIELD,
)
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_semantics import (
    MeasurementRowAxisField,
    ParentChildRelationshipPayload,
    measurement_row_mapping,
)
from openhcs.core.runtime_slice_alignment import RuntimeSliceAlignedValueSet
from openhcs.core.runtime_slice_projection import (
    RuntimeSliceProjection,
    RuntimeSliceProjectionStrategy,
)
from openhcs.core.runtime_values import (
    MeasurementTable,
    ObjectRelationship,
    image_payload_data,
)
from openhcs.interop.cellprofiler.runtime.payload_types import (
    CellProfilerFunction,
    CellProfilerKwargs,
    CellProfilerRuntimeValue,
)
from openhcs.interop.cellprofiler.runtime.runtime_profile import (
    CellProfilerRuntimeProfileLogger,
)
from openhcs.processing.backends.cellprofiler.library import (
    coerce_registered_absorbed_processing_contract,
)
from openhcs.processing.backends.lib_registry.unified_registry import ProcessingContract


PROCESSING_CONTRACT_CACHE: dict[CellProfilerFunction, ProcessingContract] = {}


@dataclass(frozen=True, slots=True)
class RuntimeShapeInspection:
    """Diagnostic shape projection for dynamic runtime payload values."""

    value: CellProfilerRuntimeValue

    def shape_tuple(self) -> tuple[int, ...] | None:
        """Return a tuple shape for diagnostic payloads without reflection."""
        shape = np.shape(self.value)
        if not shape:
            return None
        return tuple(int(dimension) for dimension in shape)


class CellProfilerProcessingContractAuthority:
    """Resolve executable processing contracts for CellProfiler runtime calls."""

    @classmethod
    def for_callable(cls, func: CellProfilerFunction) -> ProcessingContract:
        cached = PROCESSING_CONTRACT_CACHE.get(func)
        if cached is not None:
            return cached
        contract = CallableContract.from_callable(func)
        if isinstance(contract.processing_contract, ProcessingContract):
            return cls.cache(func, contract.processing_contract)
        absorbed_contract = coerce_registered_absorbed_processing_contract(
            contract.function_name,
            func,
        )
        if absorbed_contract is not None:
            return cls.cache(func, absorbed_contract)
        raise TypeError(
            f"CellProfiler executable {contract.function_name!r} has no nominal "
            "__processing_contract__ metadata. Coerce the absorbed catalog contract "
            "before runtime execution."
        )

    @staticmethod
    def cache(
        func: CellProfilerFunction,
        contract: ProcessingContract,
    ) -> ProcessingContract:
        PROCESSING_CONTRACT_CACHE[func] = contract
        return contract


class Pure2DSliceCountPolicy:
    """Resolve runtime slice counts for PURE_2D CellProfiler execution."""

    @staticmethod
    def slice_count_from_kwargs(
        kwargs: CellProfilerKwargs,
        *,
        runtime_slice_sequence_parameter_names: frozenset[str] = frozenset(),
        measurement_table_parameter_names: frozenset[str] = frozenset(),
    ) -> int | None:
        if RuntimeProfileLogger.enabled():
            Pure2DSliceCountDiagnostics.log_kwargs(kwargs)
        return RuntimeSliceProjection.slice_count_from_kwargs(
            Pure2DSliceCountPolicy.slice_count_kwargs(
                kwargs,
                measurement_table_parameter_names=measurement_table_parameter_names,
            ),
            sequence_kwargs=runtime_slice_sequence_parameter_names,
        )

    @staticmethod
    def slice_count_kwargs(
        kwargs: CellProfilerKwargs,
        *,
        measurement_table_parameter_names: frozenset[str] = frozenset(),
    ) -> CellProfilerKwargs:
        """Return kwargs that should participate in execution slice-count choice."""
        if not measurement_table_parameter_names:
            return kwargs
        slice_kwargs = dict(kwargs)
        for parameter_name in measurement_table_parameter_names:
            if parameter_name not in slice_kwargs:
                continue
            tables = slice_kwargs[parameter_name]
            if not MeasurementTableFeatureRowsAuthority(tables).contains_features():
                slice_kwargs.pop(parameter_name)
        return slice_kwargs


@dataclass(frozen=True, slots=True)
class MeasurementTableFeatureRowsAuthority:
    """Feature-row semantics for measurement-table slice-count arbitration."""

    tables: CellProfilerRuntimeValue

    metadata_fields: ClassVar[frozenset[str]] = frozenset(
        (
            MeasurementRowAxisField.SLICE_INDEX.value,
            MeasurementRowAxisField.IMAGE_NUMBER.value,
            MEASUREMENT_OBJECT_NAME_FIELD,
            MEASUREMENT_OBJECT_LABEL_FIELD,
            MEASUREMENT_OBJECT_NUMBER_FIELD,
            MEASUREMENT_OBJECT_ID_FIELD,
        )
    )

    def contains_features(self) -> bool:
        return any(
            field_name not in self.metadata_fields
            for table in self.measurement_tables()
            for row in table.iter_rows()
            for field_name in measurement_row_mapping(row)
        )

    def measurement_tables(self) -> tuple[MeasurementTable, ...]:
        if self.tables is None:
            return ()
        if not isinstance(self.tables, Sequence) or isinstance(self.tables, (str, bytes)):
            raise TypeError(
                "measurement_tables must be a sequence of MeasurementTable values."
            )
        return tuple(self.measurement_table(table) for table in self.tables)

    @staticmethod
    def measurement_table(table: CellProfilerRuntimeValue) -> MeasurementTable:
        if not isinstance(table, MeasurementTable):
            raise TypeError(
                "measurement_tables must contain MeasurementTable values; "
                f"got {type(table).__name__}."
            )
        return table


class Pure2DSliceCountDiagnostics:
    """Profile slice-count arbitration candidates for PURE_2D execution."""

    @staticmethod
    def log_kwargs(kwargs: CellProfilerKwargs) -> None:
        for name, value in kwargs.items():
            stack_fields = []
            for stack in RuntimeSliceProjectionStrategy.strategy_for_value(
                value
            ).stack_views(value):
                stack_fields.append(
                    f"{type(stack).__name__}:{RuntimeShapeInspection(stack).shape_tuple()}"
                )
            stack_text = "|".join(stack_fields)
            slice_count_candidate = Pure2DSliceCountCandidate(value)
            measurement_count = RuntimeSliceProjection.measurement_table_slice_count(
                value
            )
            data = image_payload_data(value)
            CellProfilerRuntimeProfileLogger.log_module_profile(
                "cp_pure_2d_slice_count_candidate",
                0.0,
                kwarg=name,
                value_type=type(value).__name__,
                data_type=type(data).__name__,
                data_shape=RuntimeShapeInspection(data).shape_tuple(),
                stacks=stack_text or None,
                runtime_count=slice_count_candidate.runtime_count(),
                relationship_count=slice_count_candidate.relationship_count(),
                measurement_count=measurement_count,
            )


@dataclass(frozen=True, slots=True)
class Pure2DSliceCountCandidate:
    """Diagnostic count candidates for pure-2D slice arbitration."""

    value: CellProfilerRuntimeValue

    def runtime_count(self) -> int | None:
        match self.value:
            case RuntimeSliceAlignedValueSet() as value:
                return value.slice_count
            case _:
                return None

    def relationship_count(self) -> int | None:
        match self.value:
            case ParentChildRelationshipPayload() | ObjectRelationship():
                return RuntimeSliceProjection.relationship_slice_count(self.value)
            case _:
                return None
