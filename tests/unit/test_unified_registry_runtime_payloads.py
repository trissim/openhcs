from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Any

import numpy as np
import pytest
from arraybridge.decorators import (
    DtypeConversionConfig,
    PRESERVE_INPUT_DTYPE_CONFIG,
)
from python_introspect import (
    Enableable,
    UnifiedParameterAnalyzer,
    is_enableable,
    parameter_exclusions,
)

from openhcs.core.memory import MEMORY_TYPE_NUMPY
from openhcs.core.callable_contract import callable_request
from openhcs.core.config import LazyDtypeConfig
from openhcs.core.runtime_image_values import ImageMetadataPayload, ImagePayloadMetadata, image_payload_data, image_payload_metadata, image_payload_mask, with_image_payload_data
from openhcs.core.runtime_array_values import RuntimeArrayPayload
from openhcs.core.runtime_object_label_domains import (
    ObjectLabelDomain,
    ObjectLabelDomainScope,
)
from openhcs.core.source_image_provenance import (
    SourceImageProvenancePlanes,
)
from openhcs.core.runtime_object_labels import (
    ObjectLabelVariantData,
    ObjectLabelPayload,
)
from openhcs.core.runtime_plane_projection import RuntimePlaneAxis
from openhcs.processing.backends.processors.numpy_processor import (
    create_projection,
    gaussian_blur,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    LibraryRegistryBase,
    ProcessingContract,
    RuntimeCallableInvocation,
    RuntimeCallableView,
    RuntimeInvocationKwargPolicy,
)


class MinimalRegistry(LibraryRegistryBase):
    MODULES_TO_SCAN = []
    MEMORY_TYPE = MEMORY_TYPE_NUMPY
    FLOAT_DTYPE = np.float32

    def get_library_version(self) -> str:
        return "test"

    def is_library_available(self) -> bool:
        return True

    def discover_functions(self) -> dict[str, Any]:
        return {}

    def get_library_object(self) -> Any:
        return None

    def _preprocess_input(self, image: Any, func_name: str) -> Any:
        return image

    def _postprocess_output(
        self, result: Any, original_image: Any, func_name: str
    ) -> Any:
        return result

    def _check_first_parameter(self, first_param: Any, func_name: str) -> bool:
        return True


def test_catalog_availability_proves_the_declared_module_inventory(
    monkeypatch,
) -> None:
    """Package presence cannot admit a registry whose catalog cannot import."""

    registry = MinimalRegistry("minimal")

    def unavailable_inventory() -> None:
        raise ImportError("native dependency missing")

    monkeypatch.setattr(
        registry,
        "get_modules_to_scan",
        unavailable_inventory,
    )

    with pytest.raises(ImportError, match="native dependency missing"):
        registry.is_available_for_catalog()


def test_contract_wrapper_exposes_enableable_but_hides_runtime_parameters() -> None:
    """Enableable is an editable callable declaration, not a runtime exclusion."""

    def raw(image: np.ndarray, *, sigma: float = 1.0) -> np.ndarray:
        return image + sigma

    wrapped = MinimalRegistry("minimal").apply_contract_wrapper(
        raw,
        ProcessingContract.FLEXIBLE,
    )

    enabled_field = Enableable.require_parameter_name()
    params = UnifiedParameterAnalyzer.analyze(wrapped)
    exclusions = parameter_exclusions(wrapped)

    assert is_enableable(wrapped)
    assert enabled_field in params
    assert enabled_field not in exclusions
    assert "sigma" in params
    assert "dtype_config" not in params
    assert "slice_by_slice" not in params
    assert "dtype_config" not in exclusions
    assert "slice_by_slice" in exclusions


def test_contract_wrapper_exposes_registered_lazy_runtime_config_signature() -> None:
    import inspect

    def raw(
        image: np.ndarray,
        *,
        dtype_config: DtypeConversionConfig = PRESERVE_INPUT_DTYPE_CONFIG,
    ) -> np.ndarray:
        return image

    wrapped = MinimalRegistry("minimal").apply_contract_wrapper(
        raw,
        ProcessingContract.FLEXIBLE,
    )
    parameter = inspect.signature(wrapped).parameters["dtype_config"]

    assert parameter.annotation is LazyDtypeConfig
    assert isinstance(parameter.default, LazyDtypeConfig)


def test_pure_2d_contract_slices_image_metadata_payload_nominally() -> None:
    stack = np.arange(60, dtype=np.float32).reshape(2, 5, 6)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(paths=("z0.tif", "z1.tif"))
            ),
            source_plane_dtypes=("float32", "float32"),
        ),
    )
    seen_paths: list[str | None] = []

    def add_one(image: ImageMetadataPayload) -> ImageMetadataPayload:
        assert isinstance(image, ImageMetadataPayload)
        assert image_payload_data(image).shape == (5, 6)
        seen_paths.append(image_payload_metadata(image).source_path)
        return with_image_payload_data(image, image_payload_data(image) + 1)

    add_one.output_memory_type = MEMORY_TYPE_NUMPY

    result = ProcessingContract.PURE_2D.execute(
        MinimalRegistry("minimal"),
        add_one,
        payload,
    )

    assert isinstance(result, ImageMetadataPayload)
    np.testing.assert_array_equal(image_payload_data(result), stack + 1)
    assert seen_paths == ["z0.tif", "z1.tif"]
    assert image_payload_metadata(result).source_image_provenance_planes.paths == (
        "z0.tif",
        "z1.tif",
    )


def test_pure_2d_contract_projects_stack_shaped_kwargs_per_slice() -> None:
    stack_data = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    mask_data = np.zeros(stack_data.shape, dtype=bool)
    mask_data[0, 0, 0] = True
    mask_data[1, 1, 1] = True
    stack = ImageMetadataPayload(
        data=stack_data,
        metadata=ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    mask = ImageMetadataPayload(
        data=mask_data,
        metadata=ImagePayloadMetadata(plane_axis=RuntimePlaneAxis.RUNTIME_SLICE),
    )
    seen_shapes: list[tuple[int, ...]] = []
    seen_true_indices: list[tuple[int, int]] = []

    def apply_mask(
        image: ImageMetadataPayload,
        *,
        mask: ImageMetadataPayload,
    ) -> ImageMetadataPayload:
        image_data = image_payload_data(image)
        mask_data = image_payload_data(mask)
        seen_shapes.append(mask_data.shape)
        true_y, true_x = np.argwhere(mask_data)[0]
        seen_true_indices.append((int(true_y), int(true_x)))
        return with_image_payload_data(
            image,
            np.where(mask_data, image_data + 100, image_data),
        )

    apply_mask.output_memory_type = MEMORY_TYPE_NUMPY

    result = ProcessingContract.PURE_2D.execute(
        MinimalRegistry("minimal"),
        apply_mask,
        stack,
        mask=mask,
    )

    expected = stack_data.copy()
    expected[0, 0, 0] += 100
    expected[1, 1, 1] += 100
    np.testing.assert_array_equal(image_payload_data(result), expected)
    assert seen_shapes == [(4, 5), (4, 5)]
    assert seen_true_indices == [(0, 0), (1, 1)]


def test_pure_2d_contract_preserves_declared_source_binding_axis() -> None:
    source_aliases = ("OrigBlue", "OrigGreen")
    stack = ImageMetadataPayload(
        data=np.arange(24, dtype=np.float32).reshape(2, 3, 4),
        metadata=ImagePayloadMetadata(
            plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
            source_image_names=source_aliases,
            source_image_provenance_planes=SourceImageProvenancePlanes.from_components(
                paths=("blue.tif", "green.tif")
            ),
        ),
    )
    labels = ObjectLabelPayload(
        variant_data=ObjectLabelVariantData(
            labels=np.asarray(
                (
                    ((0, 1, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                    ((0, 2, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
                ),
                dtype=np.int32,
            )
        ),
        plane_axis=RuntimePlaneAxis.SOURCE_BINDING,
        source_image_names=source_aliases,
        domain=ObjectLabelDomain(
            declared_object_id_domains=((1,), (2,)),
            scope=ObjectLabelDomainScope.PLANE,
        ),
    )
    seen_label_shapes: list[tuple[int, ...]] = []

    def consume_labels(
        image: ImageMetadataPayload,
        *,
        labels: ObjectLabelPayload,
    ) -> ImageMetadataPayload:
        seen_label_shapes.append(labels.labels.shape)
        return image

    consume_labels.output_memory_type = MEMORY_TYPE_NUMPY

    result = ProcessingContract.PURE_2D.execute(
        MinimalRegistry("minimal"),
        consume_labels,
        stack,
        labels=labels,
    )

    assert seen_label_shapes == [(3, 4), (3, 4)]
    assert image_payload_metadata(result).plane_axis is RuntimePlaneAxis.SOURCE_BINDING
    assert image_payload_metadata(result).source_image_names == source_aliases


def test_pure_3d_contract_preserves_metadata_for_plain_numpy_processor() -> None:
    stack = np.arange(60, dtype=np.float32).reshape(2, 5, 6)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(paths=("z0.tif", "z1.tif"))
            ),
            source_plane_dtypes=("float32", "float32"),
        ),
    )

    result = ProcessingContract.PURE_3D.execute(
        MinimalRegistry("minimal"),
        gaussian_blur,
        payload,
        sigma=0.5,
    )

    assert isinstance(result, ImageMetadataPayload)
    assert image_payload_data(result).shape == stack.shape
    assert image_payload_data(result).dtype == stack.dtype
    assert image_payload_metadata(result).source_image_provenance_planes.paths == (
        "z0.tif",
        "z1.tif",
    )


def test_volumetric_projection_accepts_metadata_payload_array_methods() -> None:
    stack = np.arange(60, dtype=np.float32).reshape(2, 5, 6)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_dtype="float32",
            plane_axis=RuntimePlaneAxis.RUNTIME_SLICE,
        ),
    )

    result = ProcessingContract.VOLUMETRIC_TO_SLICE.execute(
        MinimalRegistry("minimal"),
        create_projection,
        payload,
    )

    assert isinstance(result, ImageMetadataPayload)
    assert image_payload_data(result).shape == (5, 6)
    np.testing.assert_array_equal(image_payload_data(result), stack.max(axis=0))
    result_metadata = image_payload_metadata(result)
    assert result_metadata.source_dtype == "float32"
    assert result_metadata.plane_axis is None


def test_with_image_payload_data_rejects_implicit_channel_mask_collapse() -> None:
    mask = np.zeros((4, 5, 2), dtype=bool)
    mask[:, :, 0] = True
    source = ImagePayloadMetadata(source_channel_axis=-1).payload_with(
        np.ones((4, 5, 2), dtype=np.float32), mask
    )

    with pytest.raises(ValueError, match="does not match declared image mask domain"):
        with_image_payload_data(
            source,
            np.ones((4, 5), dtype=np.float32),
        )


def test_runtime_callable_invocation_can_call_raw_signature_filtered_callable() -> None:
    source = ImagePayloadMetadata().payload_with(
        np.ones((4, 5), dtype=np.float32), np.ones((4, 5), dtype=bool)
    )

    def raw(image: RuntimeArrayPayload, *, scale: int) -> RuntimeArrayPayload:
        assert isinstance(image, RuntimeArrayPayload)
        return image_payload_metadata(image).payload_with(
            image_payload_data(image) * scale, image_payload_mask(image)
        )

    @wraps(raw)
    def decorated(image: Any, **kwargs: Any) -> np.ndarray:
        return np.asarray(image_payload_data(raw(image, **kwargs)))

    result = RuntimeCallableInvocation(
        decorated,
        args=(source,),
        kwargs={"scale": 3, "adapter_control": object()},
        callable_view=RuntimeCallableView.RAW,
        kwarg_policy=RuntimeInvocationKwargPolicy.SIGNATURE_FILTERED,
    ).call()

    assert isinstance(result, RuntimeArrayPayload)
    np.testing.assert_array_equal(image_payload_data(result), np.full((4, 5), 3.0))
    np.testing.assert_array_equal(
        image_payload_mask(result), np.ones((4, 5), dtype=bool)
    )


def test_raw_runtime_callable_preserves_request_binding_semantics() -> None:
    @dataclass(frozen=True, slots=True)
    class ScaleRequest:
        image: object
        scale: int

    @callable_request(ScaleRequest)
    def request_bound(request: ScaleRequest) -> object:
        return image_payload_metadata(request.image).payload_with(
            image_payload_data(request.image) * request.scale,
            image_payload_mask(request.image),
        )

    @wraps(request_bound)
    def decorated(*args: Any, **kwargs: Any) -> object:
        return request_bound(*args, **kwargs)

    source = ImagePayloadMetadata().payload_with(
        np.ones((3, 4), dtype=np.float32), np.ones((3, 4), dtype=bool)
    )

    result = RuntimeCallableInvocation(
        decorated,
        args=(source,),
        kwargs={"scale": 4, "adapter_control": object()},
        callable_view=RuntimeCallableView.RAW,
        kwarg_policy=RuntimeInvocationKwargPolicy.SIGNATURE_FILTERED,
    ).call()

    np.testing.assert_array_equal(image_payload_data(result), np.full((3, 4), 4.0))
