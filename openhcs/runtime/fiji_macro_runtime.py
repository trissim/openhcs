"""ImageJ macro execution through the managed OpenHCS Fiji server."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from openhcs.core.image_file_serialization import ImageFileFormat
from openhcs.runtime.viewer_protocol import (
    ViewerControlMessageRequest,
    ViewerControlResponseField,
    ViewerRuntimeEndpoint,
)
from openhcs.runtime.zmq_config import OPENHCS_ZMQ_CONFIG

if TYPE_CHECKING:
    from openhcs.core.config import FijiStreamingConfig


@dataclass(frozen=True, slots=True)
class FijiMacroExecutionRequest:
    """Exact file-group and variable ABI for one managed ImageJ macro run."""

    message_type: ClassVar[str] = "run_macro"

    macro_path: Path
    input_filenames: tuple[str, ...]
    output_filenames: tuple[str, ...]
    directory_variable: str
    macro_variables: Mapping[str, str]
    input_images: tuple[np.ndarray, ...]

    def __post_init__(self) -> None:
        if len(self.input_filenames) != len(self.input_images):
            raise ValueError(
                "Fiji macro input filenames and image payloads must have identical "
                "cardinality."
            )
        if not self.input_filenames or not self.output_filenames:
            raise ValueError("Fiji macro execution requires input and output groups.")
        if not self.directory_variable.strip():
            raise ValueError("Fiji macro directory variable cannot be blank.")

    @classmethod
    def from_arrays(
        cls,
        *,
        macro_path: str | Path,
        input_filenames: tuple[str, ...],
        output_filenames: tuple[str, ...],
        directory_variable: str,
        macro_variables: Mapping[str, str],
        input_images: tuple[np.ndarray, ...],
    ) -> "FijiMacroExecutionRequest":
        return cls(
            macro_path=Path(macro_path),
            input_filenames=input_filenames,
            output_filenames=output_filenames,
            directory_variable=directory_variable,
            macro_variables=dict(macro_variables),
            input_images=tuple(np.ascontiguousarray(image) for image in input_images),
        )

    def send(
        self,
        config: "FijiStreamingConfig",
        *,
        timeout: float = 300.0,
    ) -> tuple[np.ndarray, ...]:
        runtime_config = config.viewer_runtime_config()
        response = ViewerControlMessageRequest(
            endpoint=ViewerRuntimeEndpoint(
                transport=runtime_config.transport_endpoint,
                config=OPENHCS_ZMQ_CONFIG,
            ),
            message_type=self.message_type,
            payload=self,
            timeout=timeout,
        ).send()
        if not response.succeeded():
            raise RuntimeError(
                f"Managed Fiji macro execution failed: {response.payload!r}"
            )
        payload = response.payload.get(ViewerControlResponseField.PAYLOAD.value)
        if not isinstance(payload, FijiMacroExecutionResponse):
            raise TypeError(
                "Managed Fiji macro response is missing its nominal execution result."
            )
        return payload.outputs

    def execute(self, imagej_runtime) -> tuple[np.ndarray, ...]:
        """Run this request inside the managed PyImageJ server process."""

        if not self.macro_path.is_file():
            raise FileNotFoundError(f"ImageJ macro file not found: {self.macro_path}")
        with TemporaryDirectory(prefix="openhcs_imagej_macro_") as tempdir:
            directory = Path(tempdir)
            for filename, image in zip(
                self.input_filenames,
                self.input_images,
                strict=True,
            ):
                path = directory / filename
                ImageFileFormat.require_path(path).write(path, image)
            variables = {
                self.directory_variable: str(directory),
                **self.macro_variables,
            }
            imagej_runtime.py.run_script(
                self.macro_path.suffix.removeprefix(".") or "ijm",
                self.macro_path.read_text(encoding="utf-8"),
                variables,
            )
            missing = tuple(
                filename
                for filename in self.output_filenames
                if not (directory / filename).is_file()
            )
            if missing:
                raise FileNotFoundError(
                    f"ImageJ macro did not produce declared outputs: {missing!r}."
                )
            return tuple(
                ImageFileFormat.require_path(directory / filename).read(
                    directory / filename
                )
                for filename in self.output_filenames
            )


@dataclass(frozen=True, slots=True)
class FijiMacroExecutionResponse:
    """Nominal macro outputs returned by the managed Fiji server."""

    outputs: tuple[np.ndarray, ...]
