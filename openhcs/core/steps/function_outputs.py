"""Output finalization for FunctionStep execution."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from polystore.streaming.identity import StreamProducerIdentity
from polystore.streaming.viewer_transport import ViewerStreamProducer

from openhcs.constants.constants import Backend
from openhcs.core.axis_filter import step_axis_allows_config
from openhcs.core.compiled_step_plan import (
    CompiledStepPlan,
    RuntimeArtifactMaterializationPlan,
)
from openhcs.core.context.processing_context import ProcessingContext
from openhcs.core.runtime_array_values import RuntimeArrayData
from openhcs.core.runtime_image_values import ImagePayloadMetadata, image_payload_data
from openhcs.core.runtime_profile import RuntimeProfileLogger
from openhcs.core.runtime_slice_projection import (
    RuntimeProjectedPayloadItem,
    RuntimeProjectionSourceIdentityRequest,
    RuntimeProjectionSourceIdentityRequirement,
)
from openhcs.core.source_image_provenance import (
    SourceComponentMetadata,
)
from openhcs.core.steps.function_artifact_materialization import (
    PersistentArtifactMaterializationTargetPlan,
    StreamingOnlyArtifactMaterializationTargetPlan,
    materialize_artifact_outputs,
)
from openhcs.core.steps.function_io import (
    prepare_storage_image_payloads,
    save_materialized_data,
    zarr_output_batch_layout,
)
from openhcs.core.steps.function_output_identity import (
    FunctionOutputIdentityAuthority,
    FunctionOutputParserContext,
    FunctionOutputPathAuthority,
)
from openhcs.core.steps.function_output_manifest import (
    ProducedOutputSemantics,
    step_output_manifest,
)
from openhcs.core.steps.stream_component_semantics import (
    StreamComponentMessageExtraAuthority,
    StreamImagePayloadMetadataProjector,
    StreamSourceComponentMetadataItems,
)
from openhcs.microscopes.microscope_interfaces import FilenameParser

logger = logging.getLogger(__name__)
StreamPayload = RuntimeArrayData


def stream_payload_summary(payload: StreamPayload) -> str:
    """Return bounded image payload facts for runtime streaming diagnostics."""
    data = image_payload_data(payload)
    if not isinstance(data, np.ndarray):
        return f"type={type(data).__name__}"

    summary = (
        f"shape={tuple(int(axis) for axis in data.shape)} "
        f"dtype={data.dtype} size={int(data.size)} "
        f"nonzero={int(np.count_nonzero(data))}"
    )
    if not data.size:
        return summary
    return f"{summary} min={data.min()} max={data.max()}"


class ProducedMemoryPathsAuthority:
    """Resolve absolute memory paths produced by the current step execution."""

    @classmethod
    def paths(
        cls,
        context: ProcessingContext,
        plan: CompiledStepPlan,
    ) -> list[str]:
        return [
            cls.memory_path(record, plan)
            for record in step_output_manifest(context).produced_records_for(plan)
            if record.is_image_payload
        ]

    @staticmethod
    def memory_path(
        record: ProducedOutputSemantics,
        plan: CompiledStepPlan,
    ) -> str:
        path = Path(record.output_path)
        if path.is_absolute():
            return str(path)
        return str(plan.output_dir / record.relative_output_path)


def finalize_function_step_outputs(
    context: ProcessingContext,
    plan: CompiledStepPlan,
) -> None:
    """Persist images, streams, metadata, and non-image artifacts for one step."""
    if not RuntimeProfileLogger.enabled():
        MemoryOutputWriter.write_if_needed(context, plan)
        MaterializedImageOutputWriter.write_if_needed(context, plan)
        StreamOutputsAuthority.stream_outputs(context, plan)
        OpenHCSMetadataWriter.write(context, plan)
        RuntimeArtifactMaterializationAuthority.materialize(context, plan)
        return

    _profile_finalization_phase(
        "finalize_memory_outputs",
        lambda: MemoryOutputWriter.write_if_needed(context, plan),
        plan,
    )
    _profile_finalization_phase(
        "finalize_materialized_images",
        lambda: MaterializedImageOutputWriter.write_if_needed(context, plan),
        plan,
    )
    _profile_finalization_phase(
        "finalize_stream_outputs",
        lambda: StreamOutputsAuthority.stream_outputs(context, plan),
        plan,
    )
    _profile_finalization_phase(
        "finalize_openhcs_metadata",
        lambda: OpenHCSMetadataWriter.write(context, plan),
        plan,
    )
    _profile_finalization_phase(
        "finalize_runtime_artifacts",
        lambda: RuntimeArtifactMaterializationAuthority.materialize(context, plan),
        plan,
    )


def _profile_finalization_phase(
    label: str,
    operation: Callable[[], None],
    plan: CompiledStepPlan,
) -> None:
    started_at = time.perf_counter()
    operation()
    RuntimeProfileLogger.log(
        logger,
        label,
        time.perf_counter() - started_at,
        step=plan.step_index,
        step_name=plan.step_name,
        axis_id=plan.axis_id,
    )


class MemoryOutputWriter:
    """Writes memory-backed step outputs to the configured write backend."""

    @classmethod
    def write_if_needed(
        cls,
        context: ProcessingContext,
        plan: CompiledStepPlan,
    ) -> None:
        if plan.write_backend == Backend.MEMORY.value:
            return

        produced_outputs = tuple(
            record
            for record in step_output_manifest(context).produced_records_for(plan)
            if record.is_image_payload
        )
        if not produced_outputs:
            return
        memory_paths = [
            ProducedMemoryPathsAuthority.memory_path(record, plan)
            for record in produced_outputs
        ]
        memory_data = context.filemanager.load_batch(
            memory_paths,
            Backend.MEMORY.value,
        )
        parser_context = FunctionOutputParserContext.from_processing_context(context)
        row, col = parser_context.parser.extract_component_coordinates(plan.axis_id)
        context.filemanager.ensure_directory(
            plan.output_dir,
            plan.write_backend,
        )
        context.filemanager.save_batch(
            cls.payloads(memory_data, memory_paths, plan),
            memory_paths,
            plan.write_backend,
            chunk_name=plan.axis_id,
            zarr_config=plan.zarr_config,
            batch_layout=zarr_output_batch_layout(produced_outputs),
            row=row,
            col=col,
            parser_name=parser_context.parser_name,
            microscope_type=parser_context.microscope_type,
        )

    @staticmethod
    def payloads(
        memory_data: list[StreamPayload],
        memory_paths: list[str],
        plan: CompiledStepPlan,
    ) -> list[StreamPayload]:
        return prepare_storage_image_payloads(
            memory_data,
            memory_paths,
            plan.write_backend,
        )


class MaterializedImageOutputWriter:
    """Materializes image outputs for steps configured with materialized output."""

    @staticmethod
    def write_if_needed(
        context: ProcessingContext,
        plan: CompiledStepPlan,
    ) -> None:
        materialized_output = plan.materialized_output
        if materialized_output is None:
            return

        produced_outputs = tuple(
            record
            for record in step_output_manifest(context).produced_records_for(plan)
            if record.is_image_payload
        )
        memory_paths = [
            ProducedMemoryPathsAuthority.memory_path(record, plan)
            for record in produced_outputs
        ]
        if not produced_outputs:
            return
        memory_data = context.filemanager.load_batch(
            memory_paths,
            Backend.MEMORY.value,
        )
        materialized_paths = [
            record.path_under(materialized_output.output_dir)
            for record in produced_outputs
        ]

        context.filemanager.ensure_directory(
            materialized_output.output_dir,
            materialized_output.backend,
        )
        save_materialized_data(
            context.filemanager,
            memory_data,
            materialized_paths,
            materialized_output.backend,
            plan.zarr_config,
            context,
            plan.axis_id,
            output_identities=produced_outputs,
        )
        logger.info(
            "Materialized %s files to %s",
            len(materialized_paths),
            materialized_output.output_dir,
        )


@dataclass(frozen=True, slots=True)
class StreamOutputProjectionRequest:
    """Nominal source of truth for projecting produced outputs into viewer streams."""

    parser: FilenameParser
    payloads: tuple[StreamPayload, ...]
    paths: tuple[str, ...]
    produced_outputs: tuple[ProducedOutputSemantics, ...]

    @classmethod
    def from_sequences(
        cls,
        *,
        parser: FilenameParser,
        payloads: list[StreamPayload],
        paths: list[str],
        produced_outputs: tuple[ProducedOutputSemantics, ...],
    ) -> "StreamOutputProjectionRequest":
        return cls(
            parser=parser,
            payloads=tuple(payloads),
            paths=tuple(paths),
            produced_outputs=produced_outputs,
        )

    def __post_init__(self) -> None:
        if len(self.payloads) != len(self.paths):
            raise ValueError(
                "Streaming payload/path cardinality mismatch: "
                f"{len(self.payloads)} payloads for {len(self.paths)} paths."
            )
        if len(self.payloads) != len(self.produced_outputs):
            raise ValueError(
                "Streaming payload/output-record cardinality mismatch: "
                f"{len(self.payloads)} payloads for "
                f"{len(self.produced_outputs)} output records."
            )
        if not self.produced_outputs:
            raise ValueError("Streaming requires at least one produced output record.")

    def require_single_projection(self) -> tuple[str, ...]:
        projection = self.produced_outputs[0].producer_identity.route_parts()
        for produced_output in self.produced_outputs[1:]:
            if produced_output.producer_identity.route_parts() != projection:
                raise ValueError(
                    "A viewer stream batch cannot mix producer projections."
                )
        return projection

    def runtime_projection_request(
        self,
        payload: StreamPayload,
        path: str,
        produced_output: ProducedOutputSemantics,
    ) -> RuntimeProjectionSourceIdentityRequest:
        return RuntimeProjectionSourceIdentityRequest(
            value=produced_output.contextualize_image_payload(payload),
            source_description=path,
        )

    def for_projection(
        self,
        projection: tuple[str, ...],
    ) -> "StreamOutputProjectionRequest":
        payloads: list[StreamPayload] = []
        paths: list[str] = []
        produced_outputs: list[ProducedOutputSemantics] = []
        for payload, path, produced_output in zip(
            self.payloads,
            self.paths,
            self.produced_outputs,
            strict=True,
        ):
            if produced_output.producer_identity.route_parts() == projection:
                payloads.append(payload)
                paths.append(path)
                produced_outputs.append(produced_output)
        return StreamOutputProjectionRequest.from_sequences(
            parser=self.parser,
            payloads=payloads,
            paths=paths,
            produced_outputs=tuple(produced_outputs),
        )


@dataclass(frozen=True, slots=True)
class StreamOutputItem:
    """One projected image payload and its stream-visible output path."""

    projected_payload: RuntimeProjectedPayloadItem
    output_path: str
    producer_identity: StreamProducerIdentity

    @property
    def data(self) -> StreamPayload:
        return self.projected_payload.data

    @property
    def metadata(self) -> ImagePayloadMetadata:
        return self.projected_payload.metadata

    @property
    def source_component_metadata(self) -> SourceComponentMetadata:
        return self.projected_payload.require_source_component_metadata()


@dataclass(frozen=True, slots=True)
class StreamOutputBatch:
    """Projected viewer stream items with one authoritative display projection."""

    items: tuple[StreamOutputItem, ...]
    producer: ViewerStreamProducer

    @classmethod
    def from_projection(
        cls,
        request: StreamOutputProjectionRequest,
    ) -> "StreamOutputBatch":
        request.require_single_projection()

        items: list[StreamOutputItem] = []
        for payload, path, produced_output in zip(
            request.payloads,
            request.paths,
            request.produced_outputs,
            strict=True,
        ):
            projected_items = tuple(
                cls.project_item(
                    request.runtime_projection_request(payload, path, produced_output)
                )
            )
            for projected_item in projected_items:
                stream_path = cls.stream_path_for_projected_item(
                    projected_item,
                    produced_path=path,
                    produced_output=produced_output,
                    parser=request.parser,
                    projected_item_count=len(projected_items),
                )
                source_metadata = projected_item.require_source_component_metadata()
                logger.info(
                    "🔬 STREAM SEND: path=%s components=%s %s",
                    stream_path,
                    dict(source_metadata),
                    stream_payload_summary(projected_item.data),
                )
                items.append(
                    StreamOutputItem(
                        projected_payload=projected_item,
                        output_path=stream_path,
                        producer_identity=produced_output.producer_identity,
                    )
                )

        return cls(
            items=tuple(items),
            producer=ViewerStreamProducer.from_identities(
                tuple(item.producer_identity for item in items)
            ),
        )

    @classmethod
    def from_projection_groups(
        cls,
        request: StreamOutputProjectionRequest,
    ) -> tuple["StreamOutputBatch", ...]:
        projections = tuple(
            dict.fromkeys(
                produced_output.producer_identity.route_parts()
                for produced_output in request.produced_outputs
            )
        )

        return tuple(
            cls.from_projection(request.for_projection(projection))
            for projection in projections
        )

    @property
    def is_empty(self) -> bool:
        return not self.items

    @property
    def data_list(self) -> list[StreamPayload]:
        return [item.data for item in self.items]

    @property
    def paths(self) -> list[str]:
        return [item.output_path for item in self.items]

    @property
    def source_metadata_items(self) -> StreamSourceComponentMetadataItems:
        return StreamSourceComponentMetadataItems.from_values(
            item.source_component_metadata for item in self.items
        )

    def item_fields(self, component_order: tuple[str, ...]) -> dict:
        fields_by_item = tuple(
            StreamImagePayloadMetadataProjector.item_fields(
                item.metadata,
                component_order,
            )
            for item in self.items
        )
        if not fields_by_item:
            return {}
        item_fields = fields_by_item[0]
        if any(fields != item_fields for fields in fields_by_item[1:]):
            raise ValueError(
                "One viewer stream batch cannot mix image-axis metadata fields."
            )
        return item_fields

    def partition_by_item_fields(
        self,
        component_order: tuple[str, ...],
    ) -> tuple["StreamOutputBatch", ...]:
        """Partition this producer projection into transport-homogeneous batches."""
        partitions: list[tuple[dict, list[StreamOutputItem]]] = []
        for item in self.items:
            item_fields = StreamImagePayloadMetadataProjector.item_fields(
                item.metadata,
                component_order,
            )
            for partition_fields, partition_items in partitions:
                if partition_fields == item_fields:
                    partition_items.append(item)
                    break
            else:
                partitions.append((item_fields, [item]))

        return tuple(
            type(self)(
                items=tuple(partition_items),
                producer=ViewerStreamProducer.from_identities(
                    tuple(item.producer_identity for item in partition_items)
                ),
            )
            for _partition_fields, partition_items in partitions
        )

    @staticmethod
    def project_item(
        request: RuntimeProjectionSourceIdentityRequest,
    ) -> tuple[RuntimeProjectedPayloadItem, ...]:
        return (
            RuntimeProjectionSourceIdentityRequirement.REQUIRED_COMPONENT_METADATA
        ).project_payload_items(request)

    @staticmethod
    def stream_path_for_projected_item(
        projected_item: RuntimeProjectedPayloadItem,
        *,
        produced_path: str,
        produced_output: ProducedOutputSemantics,
        parser: FilenameParser,
        projected_item_count: int,
    ) -> str:
        """Return the stream-visible path for one projected payload item."""
        if projected_item_count <= 1:
            return produced_path
        identity = FunctionOutputIdentityAuthority.identity_from_metadata(
            parser,
            projected_item.metadata,
            fallback_identity_path=produced_path,
        )
        if identity is None:
            return produced_path
        if produced_output.filename_qualifier is not None:
            identity = identity.with_filename_qualifier(
                produced_output.filename_qualifier
            )
        filename = FunctionOutputPathAuthority.filename_for_identity(
            parser,
            identity,
        )
        return str(Path(produced_path).parent / filename)


class StreamOutputsAuthority:
    """Streams step image outputs through viewer backends."""

    @staticmethod
    def stream_outputs(
        context: ProcessingContext,
        plan: CompiledStepPlan,
    ) -> None:
        for config_instance in plan.streaming_configs.values():
            if not step_axis_allows_config(
                context.step_axis_filters,
                step_index=plan.step_index,
                config=config_instance,
                axis_id=context.axis_id,
            ):
                logger.debug(
                    "Skipping %s streaming for step %s, axis %s (filtered out)",
                    type(config_instance).__name__,
                    plan.step_name,
                    context.axis_id,
                )
                continue
            produced_outputs = tuple(
                record
                for record in step_output_manifest(context).produced_records_for(plan)
                if record.is_image_payload
            )
            memory_paths = [
                ProducedMemoryPathsAuthority.memory_path(record, plan)
                for record in produced_outputs
            ]
            if not memory_paths:
                logger.info(
                    "No produced image outputs to stream for step %s.",
                    plan.step_name,
                )
                continue
            if plan.materialized_output is not None:
                streaming_paths = [
                    record.path_under(plan.materialized_output.output_dir)
                    for record in produced_outputs
                ]
            else:
                streaming_paths = memory_paths

            streaming_payloads: list[StreamPayload] = list(
                context.filemanager.load_batch(
                    memory_paths,
                    Backend.MEMORY.value,
                )
            )
            stream_batches = StreamOutputBatch.from_projection_groups(
                StreamOutputProjectionRequest.from_sequences(
                    parser=context.microscope_handler.parser,
                    payloads=streaming_payloads,
                    paths=list(streaming_paths),
                    produced_outputs=produced_outputs,
                )
            )
            stream_batches = tuple(
                stream_batch
                for stream_batch in stream_batches
                if not stream_batch.is_empty
            )
            if not stream_batches:
                logger.info(
                    "No streamable image outputs for step %s after stack projection.",
                    plan.step_name,
                )
                continue
            viewer_surface = config_instance.streaming_viewer_surface(context)
            for producer_batch in stream_batches:
                producer_metadata = StreamComponentMessageExtraAuthority.from_context(
                    viewer_surface,
                    context=context,
                    source_metadata_items=producer_batch.source_metadata_items,
                )
                for stream_batch in producer_batch.partition_by_item_fields(
                    producer_metadata.layout.component_order
                ):
                    stream_backend_kwargs = (
                        StreamComponentMessageExtraAuthority.from_context(
                            viewer_surface,
                            context=context,
                            source_metadata_items=stream_batch.source_metadata_items,
                        ).viewer_backend_kwargs(
                            producer=stream_batch.producer,
                        )
                    )
                    stream_backend_kwargs = stream_backend_kwargs.with_item_fields(
                        stream_batch.item_fields(
                            stream_backend_kwargs.stream_request.display_semantics.component_order
                        )
                    )
                    context.filemanager.save_batch(
                        stream_batch.data_list,
                        stream_batch.paths,
                        config_instance.backend.value,
                        **stream_backend_kwargs.to_kwargs(),
                    )


class OpenHCSMetadataWriter:
    """Writes OpenHCS metadata sidecars for primary and materialized outputs."""

    @dataclass(frozen=True, slots=True)
    class OutputTarget:
        """One compiled output location projected into OpenHCS metadata."""

        output_dir: Path
        backend: str
        is_main: bool
        plate_root: str
        sub_dir: str
        results_dir: str | None

        @classmethod
        def primary(
            cls, plan: CompiledStepPlan
        ) -> "OpenHCSMetadataWriter.OutputTarget | None":
            if plan.write_backend in [Backend.OMERO_LOCAL.value, Backend.MEMORY.value]:
                return None
            if plan.write_backend is None:
                raise ValueError(
                    f"Step {plan.step_index} ({plan.step_name}) has no write backend."
                )
            if plan.output_dir is None:
                raise ValueError(
                    f"Step {plan.step_index} ({plan.step_name}) has no output directory."
                )
            if plan.output_plate_root is None or plan.sub_dir is None:
                raise ValueError(
                    f"Step {plan.step_index} ({plan.step_name}) has incomplete "
                    "OpenHCS metadata output identity."
                )
            return cls(
                output_dir=plan.output_dir,
                backend=plan.write_backend,
                is_main=True,
                plate_root=plan.output_plate_root,
                sub_dir=plan.sub_dir,
                results_dir=plan.analysis_results_dir,
            )

        @classmethod
        def materialized(
            cls,
            plan: CompiledStepPlan,
        ) -> "OpenHCSMetadataWriter.OutputTarget | None":
            materialized_output = plan.materialized_output
            if materialized_output is None:
                return None
            if materialized_output.backend in [
                Backend.OMERO_LOCAL.value,
                Backend.MEMORY.value,
            ]:
                return None
            return cls(
                output_dir=materialized_output.output_dir,
                backend=materialized_output.backend,
                is_main=False,
                plate_root=materialized_output.plate_root,
                sub_dir=materialized_output.sub_dir,
                results_dir=materialized_output.analysis_results_dir,
            )

        def contains_images(self, context: ProcessingContext) -> bool:
            """Return whether the completed target contains image outputs."""

            if context.filemanager is None:
                raise ValueError("OpenHCS metadata requires a file manager.")
            return bool(
                context.filemanager.list_image_files(
                    self.output_dir,
                    self.backend,
                )
            )

        def write(self, context: ProcessingContext) -> None:
            """Project the target's current storage state into plate metadata."""

            from openhcs.microscopes.openhcs import OpenHCSMetadataGenerator

            if context.filemanager is None:
                raise ValueError("OpenHCS metadata requires a file manager.")
            OpenHCSMetadataGenerator(context.filemanager).create_metadata(
                context,
                str(self.output_dir),
                self.backend,
                is_main=self.is_main,
                plate_root=self.plate_root,
                sub_dir=self.sub_dir,
                results_dir=self.results_dir,
            )

    @classmethod
    def write(
        cls,
        context: ProcessingContext,
        plan: CompiledStepPlan,
    ) -> None:
        if not plan.create_openhcs_metadata:
            return
        cls.write_primary_metadata(context, plan)
        cls.write_materialized_metadata(context, plan)

    @classmethod
    def finalize_completed_plate(
        cls,
        compiled_contexts: Mapping[str, ProcessingContext],
    ) -> None:
        """Write each populated metadata target after all axis outputs exist."""

        target_contexts: dict[OpenHCSMetadataWriter.OutputTarget, ProcessingContext] = (
            {}
        )
        for context in compiled_contexts.values():
            for plan in context.step_plans.values():
                if not plan.create_openhcs_metadata:
                    continue
                for target in (
                    cls.OutputTarget.primary(plan),
                    cls.OutputTarget.materialized(plan),
                ):
                    if target is not None:
                        target_contexts.setdefault(target, context)

        for target, context in target_contexts.items():
            if target.contains_images(context):
                target.write(context)

    @staticmethod
    def write_primary_metadata(
        context: ProcessingContext,
        plan: CompiledStepPlan,
    ) -> None:
        if not ProducedMemoryPathsAuthority.paths(context, plan):
            return
        target = OpenHCSMetadataWriter.OutputTarget.primary(plan)
        if target is None:
            return
        target.write(context)

    @staticmethod
    def write_materialized_metadata(
        context: ProcessingContext,
        plan: CompiledStepPlan,
    ) -> None:
        target = OpenHCSMetadataWriter.OutputTarget.materialized(plan)
        if target is None:
            return
        target.write(context)


class RuntimeArtifactMaterializationAuthority:
    """Materializes runtime artifacts for persistent output and streaming."""

    @classmethod
    def materialize(
        cls,
        context: ProcessingContext,
        plan: CompiledStepPlan,
    ) -> None:
        if not plan.artifact_outputs:
            return
        materialization_plan = plan.runtime_artifact_materialization
        has_persistent_target = materialization_plan.has_persistent_target
        has_streaming_target = bool(plan.streaming_configs)
        if not has_persistent_target and not has_streaming_target:
            logger.info("Skipping runtime artifact materialization and streaming")
            return

        logger.info(
            "Starting materialization for %s artifact outputs",
            len(plan.artifact_outputs),
        )
        materialize_artifact_outputs(
            context.filemanager,
            plan,
            cls.target_plan(materialization_plan),
            context,
        )
        logger.info("Completed artifact materialization")

    @staticmethod
    def target_plan(
        materialization_plan: RuntimeArtifactMaterializationPlan,
    ) -> (
        PersistentArtifactMaterializationTargetPlan
        | StreamingOnlyArtifactMaterializationTargetPlan
    ):
        if not materialization_plan.has_persistent_target:
            logger.info("Skipping persistent runtime artifact materialization")
            return StreamingOnlyArtifactMaterializationTargetPlan()

        return PersistentArtifactMaterializationTargetPlan(
            materialization_plan.require_persistent_backend()
        )
