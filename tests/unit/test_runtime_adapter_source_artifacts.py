from pathlib import Path
from types import SimpleNamespace

import numpy as np
from polystore.base import ensure_storage_registry, storage_registry
from polystore.filemanager import FileManager
from scipy.io import savemat

from openhcs.constants.constants import AllComponents, Backend
from openhcs.constants.input_source import InputSource
from openhcs.core.artifacts import ImageArtifactType, ObjectLabelsArtifactType
from openhcs.core.component_group_scope import RuntimeExecutionAxisScope
from openhcs.core.runtime_adapters import RuntimeAdapterRequest
from openhcs.core.runtime_image_loading import ImagePayloadSourceMetadataContext
from openhcs.core.runtime_image_values import image_payload_data
from openhcs.core.runtime_object_labels import ObjectLabelSet
from openhcs.core.source_binding_workspace import SourceBindingWorkspaceProjector
from openhcs.core.source_bindings import (
    CompiledSourceBindingPlan,
    NamedSourceBinding,
    SourceBindingMatchMethod,
    SourceBindingMatchPlan,
    SourceBindingRuntimeContext,
    SourceBindingsConfig,
    SourceFilterClause,
    SourceFilterMatchType,
    SourceFilterSubject,
    SourceProjectionRole,
    SourceSelector,
    StepSourceBindingsConfig,
)
from openhcs.core.source_image_provenance import SourceImageIdentity
from openhcs.core.source_image_semantics import apply_source_binding_payload
from openhcs.core.source_matching import SourceImageSetIdentityPolicy
from openhcs.core.source_projection import SourcePlaneProjection
from openhcs.core.source_workspace_projection import (
    VirtualWorkspacePathLookup,
    VirtualWorkspaceSourceProjectionAuthority,
    VirtualWorkspaceSourceProjectionCache,
)
from openhcs.interop.cellprofiler.runtime.artifact_binding import (
    RuntimeArtifactInputRequest,
    RuntimeArtifactTypeStrategy,
)
from openhcs.microscopes.source_bindings_handler import SourceBindingsHandler
from openhcs.microscopes.source_schema import SourceSchemaFilenameParser


def _filemanager() -> FileManager:
    ensure_storage_registry()
    return FileManager(dict(storage_registry))


def test_source_artifact_inputs_share_workspace_vfs_and_contract_resolution(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    workspace_root = tmp_path / "workspace"
    source_root.mkdir()
    primary_path = source_root / "primary.npy"
    illumination_path = source_root / "illumination.mat"
    labels_path = source_root / "labels.npy"
    primary = np.arange(20, dtype=np.uint16).reshape(4, 5)
    illumination = np.full((4, 5), 0.25, dtype=np.float32)
    labels = np.array(
        [[0, 1, 1, 0, 0], [0, 1, 1, 0, 2], [0, 0, 0, 0, 2], [3, 3, 0, 0, 0]],
        dtype=np.uint16,
    )
    np.save(primary_path, primary)
    savemat(illumination_path, {"Image": illumination})
    np.save(labels_path, labels)

    def binding(
        alias: str,
        source_path: Path,
        *,
        artifact_kind=ImageArtifactType,
        projection_role=SourceProjectionRole.PRIMARY_PLANE,
    ) -> NamedSourceBinding:
        return NamedSourceBinding(
            alias=alias,
            artifact_kind=artifact_kind,
            projection_role=projection_role,
            selector=SourceSelector(
                filters=(
                    SourceFilterClause(
                        SourceFilterSubject.FILE,
                        SourceFilterMatchType.EQUALS,
                        source_path.name,
                    ),
                )
            ),
        )

    primary_binding = binding("Primary", primary_path)
    illumination_binding = binding(
        "Illumination",
        illumination_path,
        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
    )
    labels_binding = binding(
        "Labels",
        labels_path,
        artifact_kind=ObjectLabelsArtifactType,
        projection_role=SourceProjectionRole.SOURCE_ARTIFACT,
    )
    source_bindings = SourceBindingsConfig(
        bindings=(
            primary_binding,
            illumination_binding,
            labels_binding,
        ),
        match_plan=SourceBindingMatchPlan(SourceBindingMatchMethod.ORDER),
    )
    filemanager = _filemanager()
    SourceBindingWorkspaceProjector(
        source_bindings,
        parser=SourceSchemaFilenameParser(),
    ).materialize(
        source_root,
        workspace_root,
        filemanager=filemanager,
        source_backend=Backend.DISK,
        workspace_backend=Backend.DISK,
        source_files=(primary_path, illumination_path, labels_path),
    )
    microscope_handler = SourceBindingsHandler(filemanager, source_bindings)
    microscope_handler.initialize_workspace(workspace_root, filemanager)
    projection_cache = VirtualWorkspaceSourceProjectionCache()
    context = SimpleNamespace(
        plate_path=workspace_root,
        filemanager=filemanager,
        microscope_handler=microscope_handler,
        runtime_source_workspace_projection_cache=projection_cache,
        source_image_set_identity_policy=SourceImageSetIdentityPolicy(
            frozenset((AllComponents.CHANNEL,))
        ),
    )
    projection = VirtualWorkspaceSourceProjectionAuthority.from_context(
        context,
        cache=projection_cache,
    ).projection_or_empty()
    primary_virtual_path = next(
        virtual_path
        for virtual_path in projection.pipeline_start_files(axis_id="A01")
        if isinstance(
            projection.require_source_projection_for(
                VirtualWorkspacePathLookup.from_paths(
                    virtual_path,
                    virtual_path,
                )
            ),
            SourcePlaneProjection,
        )
    )
    primary_lookup = VirtualWorkspacePathLookup.from_paths(
        primary_virtual_path,
        primary_virtual_path,
    )
    primary_projection = projection.require_source_projection_for(primary_lookup)
    primary_payload = projection.project_payload(
        primary_lookup,
        filemanager.load(
            primary_virtual_path,
            Backend.VIRTUAL_WORKSPACE.value,
        ),
    )
    primary_payload = apply_source_binding_payload(
        primary_payload,
        primary_binding,
        ImagePayloadSourceMetadataContext(
            SourceImageIdentity(
                primary_virtual_path,
                projection.source_metadata_for(primary_lookup),
            ),
            primary_projection.ref.backend,
            filemanager,
            primary_projection.ref.backend_address,
        ),
    )
    runtime_context = SourceBindingRuntimeContext(
        step_input_source_paths={
            virtual_path: source_ref.backend_address
            for virtual_path, source_ref in projection.source_refs_by_virtual_path.items()
        },
        source_metadata_by_path=projection.source_metadata_by_path,
    )

    def request() -> RuntimeAdapterRequest:
        return RuntimeAdapterRequest(
            context=context,
            source_payload=primary_payload,
            source_binding_plan=CompiledSourceBindingPlan.from_config(
                StepSourceBindingsConfig(
                    bindings=source_bindings.binding_declarations,
                    match_plan=source_bindings.match_plan,
                    enabled=True,
                ),
            ),
            source_binding_context=runtime_context,
            axis_scope=RuntimeExecutionAxisScope.from_raw(
                "A01",
                component=None,
                value=None,
            ),
        )

    illumination_request = request()
    np.testing.assert_array_equal(
        image_payload_data(
            illumination_request.source_artifact_payload(
                illumination_binding.input_spec().ref()
            )
        ),
        illumination[np.newaxis, ...],
    )

    primary_auxiliary_request = request()
    assert (
        primary_auxiliary_request.source_binding_for_artifact_ref(
            primary_binding.input_spec().ref()
        )
        == primary_binding
    )

    labels_request = request()
    labels_spec = labels_binding.input_spec()
    labels_payload = labels_request.source_artifact_payload(labels_spec.ref())
    label_set = RuntimeArtifactTypeStrategy.for_artifact_type(
        ObjectLabelsArtifactType
    ).raw_runtime_input_value(
        RuntimeArtifactInputRequest(
            spec=labels_spec,
            value=labels_payload,
        )
    )
    assert isinstance(label_set, ObjectLabelSet)
    np.testing.assert_array_equal(label_set.labels, labels[np.newaxis, ...])
