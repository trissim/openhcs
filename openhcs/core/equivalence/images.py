"""Image snapshot records for runtime equivalence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy


@dataclass(frozen=True, slots=True)
class RuntimeImageSnapshot:
    """Semantic snapshot of one exported runtime image."""

    path: Path
    shape: tuple[int, ...]
    dtype: str
    pixel_digest: str
    pixel_data: np.ndarray = field(repr=False, compare=False)

    @classmethod
    def from_image_file(cls, path: Path) -> "RuntimeImageSnapshot":
        """Read an image export into a decoded-pixel semantic snapshot."""
        array = (
            np.load(path)
            if path.suffix.lower() == ".npy"
            else np.asarray(imageio.imread(path))
        )
        return cls.from_array(path, array)

    @classmethod
    def from_array(
        cls,
        path: Path | str,
        array: object,
    ) -> "RuntimeImageSnapshot":
        """Build a semantic image snapshot from an in-memory runtime artifact."""
        contiguous = np.ascontiguousarray(array)
        return cls(
            path=Path(path),
            shape=tuple(int(axis) for axis in contiguous.shape),
            dtype=str(contiguous.dtype),
            pixel_digest=hashlib.sha256(contiguous.tobytes()).hexdigest(),
            pixel_data=contiguous.copy(),
        )

    def content_key(
        self,
        policy: RuntimeEquivalencePolicy,
    ) -> tuple[object, ...]:
        """Return image identity at the requested semantic strictness."""
        key: tuple[object, ...] = (self.shape, self.dtype)
        if policy.compare_image_pixels:
            key = (*key, self.pixel_digest)
        return key
