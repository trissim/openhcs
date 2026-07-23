from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RoiData:
    data: list[NDArray[np.number]] = field(default_factory=list)
    shape_type: list[str] = field(default_factory=list)
    names: list[str] | None = field(default_factory=lambda: None)
    features: dict[str, list[Any]] | None = field(default_factory=lambda: None)

    def __post_init__(self) -> None:
        if self.features is None:
            return
        row_count = len(self.data)
        for column, values in self.features.items():
            if not isinstance(column, str):
                raise TypeError("ROI feature column names must be strings")
            if isinstance(values, (str, bytes)) or len(values) != row_count:
                raise ValueError(
                    f"ROI feature column {column!r} must contain {row_count} values"
                )

    def to_json_dict(self) -> dict[str, Any]:
        """Convert RoiData to a JSON serializable dictionary."""
        data = [d.tolist() for d in self.data]
        out = {"data": data, "shape_type": self.shape_type}
        if self.names is not None:
            out["names"] = self.names
        if self.features is not None:
            out["features"] = {
                column: [_json_value(value) for value in values]
                for column, values in self.features.items()
            }
        return out

    @classmethod
    def from_json_dict(cls, js: dict[str, Any]) -> RoiData:
        """Create RoiData from a JSON serializable dictionary."""
        data = [np.array(d) for d in js["data"]]
        shape_type = js["shape_type"]
        names = js.get("names")
        features = js.get("features")
        if features is not None and not isinstance(features, dict):
            raise TypeError("ROI JSON features must be a column mapping")
        return RoiData(
            data,
            shape_type=shape_type,
            names=names,
            features=features,
        )

    def iter_shapes(self):
        """Iterate over shapes and their types."""
        for i in range(len(self.data)):
            yield RoiTuple(
                data=self.data[i],
                shape_type=self.shape_type[i],
                name=self.names[i] if self.names is not None else None,
            )


@dataclass
class RoiTuple:
    data: NDArray[np.number]
    shape_type: str
    name: str | None = None
    multidim: tuple[int, ...] = ()

    def __post_init__(self):
        self.data = np.asarray(self.data)


def _json_value(value: Any) -> Any:
    """Project native feature values into JSON without stringifying semantics."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(
        f"ROI feature value of type {type(value).__name__!r} is not JSON serializable"
    )
