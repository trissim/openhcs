from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from openhcs.core.memory import MEMORY_TYPE_NUMPY
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    RuntimeArrayPayload,
    RuntimeImagePayloadContext,
    SourceImageProvenancePlanes,
    image_payload_data,
    image_payload_metadata,
    image_payload_mask,
    with_image_payload_data,
)
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

    def _postprocess_output(self, result: Any, original_image: Any, func_name: str) -> Any:
        return result

    def _check_first_parameter(self, first_param: Any, func_name: str) -> bool:
        return True


def test_pure_2d_contract_slices_image_metadata_payload_nominally() -> None:
    stack = np.arange(60, dtype=np.float32).reshape(2, 5, 6)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=("z0.tif", "z1.tif")
                )
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
    stack = np.arange(40, dtype=np.float32).reshape(2, 4, 5)
    mask = np.zeros(stack.shape, dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    seen_shapes: list[tuple[int, ...]] = []
    seen_true_indices: list[tuple[int, int]] = []

    def apply_mask(image: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
        seen_shapes.append(mask.shape)
        true_y, true_x = np.argwhere(mask)[0]
        seen_true_indices.append((int(true_y), int(true_x)))
        return np.where(mask, image + 100, image)

    apply_mask.output_memory_type = MEMORY_TYPE_NUMPY

    result = ProcessingContract.PURE_2D.execute(
        MinimalRegistry("minimal"),
        apply_mask,
        stack,
        mask=mask,
    )

    expected = stack.copy()
    expected[0, 0, 0] += 100
    expected[1, 1, 1] += 100
    np.testing.assert_array_equal(result, expected)
    assert seen_shapes == [(4, 5), (4, 5)]
    assert seen_true_indices == [(0, 0), (1, 1)]


def test_pure_3d_contract_preserves_metadata_for_plain_numpy_processor() -> None:
    stack = np.arange(60, dtype=np.float32).reshape(2, 5, 6)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(
            source_image_provenance_planes=(
                SourceImageProvenancePlanes.from_components(
                    paths=("z0.tif", "z1.tif")
                )
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


def test_pure_3d_projection_accepts_metadata_payload_array_methods() -> None:
    stack = np.arange(60, dtype=np.float32).reshape(2, 5, 6)
    payload = ImageMetadataPayload(
        data=stack,
        metadata=ImagePayloadMetadata(source_dtype="float32"),
    )

    result = ProcessingContract.PURE_3D.execute(
        MinimalRegistry("minimal"),
        create_projection,
        payload,
    )

    assert isinstance(result, ImageMetadataPayload)
    assert image_payload_data(result).shape == (1, 5, 6)
    np.testing.assert_array_equal(image_payload_data(result), stack.max(axis=0)[None])
    assert image_payload_metadata(result).source_dtype == "float32"


def test_with_image_payload_data_projects_channel_last_mask_to_grayscale() -> None:
    mask = np.zeros((4, 5, 2), dtype=bool)
    mask[:, :, 0] = True
    source = RuntimeImagePayloadContext(
        np.ones((4, 5, 2), dtype=np.float32),
        mask=mask,
        metadata=ImagePayloadMetadata(),
    ).payload()

    result = with_image_payload_data(
        source,
        np.ones((4, 5), dtype=np.float32),
    )

    assert image_payload_data(result).shape == (4, 5)
    np.testing.assert_array_equal(
        image_payload_mask(result),
        np.zeros((4, 5), dtype=bool),
    )


def test_runtime_callable_invocation_can_call_raw_signature_filtered_callable() -> None:
    source = RuntimeImagePayloadContext(
        np.ones((4, 5), dtype=np.float32),
        mask=np.ones((4, 5), dtype=bool),
        metadata=ImagePayloadMetadata(),
    ).payload()

    def raw(image: RuntimeArrayPayload, *, scale: int) -> RuntimeArrayPayload:
        assert isinstance(image, RuntimeArrayPayload)
        return RuntimeImagePayloadContext(
            image_payload_data(image) * scale,
            mask=image_payload_mask(image),
            metadata=image_payload_metadata(image),
        ).payload()

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
    np.testing.assert_array_equal(image_payload_mask(result), np.ones((4, 5), dtype=bool))
