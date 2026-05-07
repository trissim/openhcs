from __future__ import annotations

from typing import Any

import numpy as np

from openhcs.core.memory import MEMORY_TYPE_NUMPY
from openhcs.core.runtime_values import (
    ImageMetadataPayload,
    ImagePayloadMetadata,
    image_payload_data,
    image_payload_metadata,
    with_image_payload_data,
)
from openhcs.processing.backends.lib_registry.unified_registry import (
    LibraryRegistryBase,
    ProcessingContract,
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
            channel_source_paths=("z0.tif", "z1.tif"),
            channel_source_dtypes=("float32", "float32"),
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
    assert image_payload_metadata(result).channel_source_paths == ("z0.tif", "z1.tif")
