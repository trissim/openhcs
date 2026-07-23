"""CellProfiler image numbering over typed OpenHCS source identities."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.measurement_row_materialization import (
    MeasurementRowsAxisProjection,
)
from openhcs.core.runtime_measurements import MeasurementScope
from openhcs.core.runtime_measurements import MeasurementTable
from openhcs.core.runtime_measurements import MeasurementRowAxisField
from openhcs.core.runtime_tabular_values import ColumnarRows
from openhcs.core.source_image_provenance import SourceImageProvenance
from openhcs.core.source_matching import (
    SourceImageSetIdentity,
    SourceImageSetIdentityPolicy,
)


@dataclass(slots=True)
class CellProfilerImageSetNumbering:
    """Assign stable one-based image numbers to exact source image sets."""

    identity_policy: SourceImageSetIdentityPolicy
    _numbers: OrderedDict[tuple[str, SourceImageSetIdentity], int] = field(
        default_factory=OrderedDict,
        init=False,
    )

    def for_source_slices(
        self,
        *,
        scope: RuntimeExecutionAxisScope,
        provenance: SourceImageProvenance,
        slice_indices: Sequence[int],
        owner: str,
    ) -> dict[int, int]:
        """Map one producer's local slices to CellProfiler image numbers."""

        return {
            slice_index: self._numbers.setdefault(
                (
                    scope.axis_id,
                    self._source_identity(
                        provenance,
                        slice_index,
                        owner=owner,
                    ),
                ),
                len(self._numbers) + 1,
            )
            for slice_index in slice_indices
        }

    def for_source_slice(
        self,
        *,
        scope: RuntimeExecutionAxisScope,
        provenance: SourceImageProvenance,
        slice_index: int,
        owner: str,
    ) -> int:
        """Return one producer slice's CellProfiler image number."""

        return self.for_source_slices(
            scope=scope,
            provenance=provenance,
            slice_indices=(slice_index,),
            owner=owner,
        )[slice_index]

    def project_measurement_rows(
        self,
        *,
        scope: RuntimeExecutionAxisScope,
        table: MeasurementTable,
    ) -> Sequence[object] | ColumnarRows:
        """Project OpenHCS row axes into exact CellProfiler image numbers."""

        slice_axis = MeasurementRowAxisField.SLICE_INDEX
        projection = MeasurementRowsAxisProjection.from_rows(table.rows)
        image_numbers_by_slice = self.for_source_slices(
            scope=scope,
            provenance=table.source_provenance,
            slice_indices=projection.present_axis_values(slice_axis.value),
            owner=table.name,
        )
        axisless_image_number = None
        if projection.has_axisless_rows(slice_axis):
            source_plane_indices = tuple(
                range(table.source_provenance.source_plane_count)
            ) or (0,)
            source_image_numbers = tuple(
                dict.fromkeys(
                    self.for_source_slices(
                        scope=scope,
                        provenance=table.source_provenance,
                        slice_indices=source_plane_indices,
                        owner=table.name,
                    ).values()
                )
            )
            if (
                len(source_image_numbers) != 1
                and table.subject.scope is not MeasurementScope.ARTIFACT
            ):
                raise ValueError(
                    f"CellProfiler export cannot bind axisless rows in "
                    f"{table.name!r} to one source image set; producer provenance "
                    f"resolves to image numbers {source_image_numbers!r}."
                )
            # Artifact-scoped rows summarize the complete produced stack rather
            # than any individual source plane. CellProfiler nevertheless
            # requires every exported row to carry one ImageNumber, so anchor
            # the artifact summary to the stack's first stable image-set number.
            # Image- and object-scoped axisless rows remain ambiguous and fail
            # above when their provenance spans multiple image sets.
            axisless_image_number = source_image_numbers[0]
        return projection.remap_runtime_slice_indices(
            image_numbers_by_slice,
            axisless_value=axisless_image_number,
        )

    def _source_identity(
        self,
        provenance: SourceImageProvenance,
        slice_index: int,
        *,
        owner: str,
    ) -> SourceImageSetIdentity:
        source_identity = provenance.for_source_plane(slice_index)
        image_set_identity = SourceImageSetIdentity.from_metadata(
            source_identity.source_component_metadata or {},
            fallback_source_path=source_identity.source_path or "",
            policy=self.identity_policy,
        )
        if image_set_identity.components == (("source_path", ""),):
            raise ValueError(
                f"CellProfiler export requires {owner!r} to carry producer-declared "
                f"source identity for slice_index={slice_index}; producer provenance "
                f"is {provenance.equality_identity!r}."
            )
        return image_set_identity
