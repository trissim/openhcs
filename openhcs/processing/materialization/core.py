"""Materialization core (writer-based, greenfield).

Key idea: the abstraction boundary is the output *format* (writers), not per-analysis handlers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from openhcs.processing.materialization.constants import MaterializationFormat, WriteMode
from openhcs.processing.materialization.options import (
    CsvOptions,
    FileOutputOptions,
    JsonOptions,
    ROIOptions,
    TextOptions,
    TiffStackOptions,
)
from openhcs.processing.materialization.utils import (
    discover_array_fields,
    expand_array_field,
    extract_fields,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Output:
    path: str
    content: Any


def _resolve_source(value: Any, source: Optional[str]) -> Any:
    if not source:
        return value

    cur = value
    for part in source.split("."):
        if isinstance(cur, dict):
            cur = cur[part]
        else:
            cur = getattr(cur, part)
    return cur


def _select_payload(data: Any, options: Any) -> Any:
    return _resolve_source(data, getattr(options, "source", None))


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    size = getattr(value, "size", None)
    if isinstance(size, int) and size == 0:
        return True
    try:
        return len(value) == 0  # type: ignore[arg-type]
    except Exception:
        return False


def _as_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


class PathHelper:
    """Path helper parameterized by output options."""

    def __init__(self, base_path: str, options: FileOutputOptions):
        self.base_path = self._strip_path(base_path, options)
        self.parent = self.base_path.parent
        self.name = self.base_path.name

    @staticmethod
    def _strip_path(path: str, options: FileOutputOptions) -> Path:
        p = Path(path)
        name = p.name

        if name.endswith(".roi.zip"):
            name = name[: -len(".roi.zip")]

        if options.strip_pkl and name.endswith(".pkl"):
            name = name[: -len(".pkl")]
        if options.strip_roi_suffix and name.endswith(".roi"):
            name = name[: -len(".roi")]

        return p.with_name(name)

    def with_suffix(self, suffix: str) -> str:
        return str(self.parent / f"{self.name}{suffix}")


class BackendSaver:
    """Centralized multi-backend saving."""

    def __init__(
        self,
        backends: list[str],
        filemanager: Any,
        backend_kwargs: dict[str, dict[str, Any]] | None,
        *,
        write_mode: WriteMode,
    ):
        self.backends = backends
        self.filemanager = filemanager
        self.backend_kwargs = backend_kwargs or {}
        self.write_mode = write_mode

    def save(self, content: Any, path: str) -> None:
        for backend in self.backends:
            self._prepare_path(backend, path)
            kwargs = self.backend_kwargs.get(backend, {})
            self.filemanager.save(content, path, backend, **kwargs)

    def _prepare_path(self, backend: str, path: str) -> None:
        backend_instance = self.filemanager._get_backend(backend)
        if not backend_instance.requires_filesystem_validation:
            return

        self.filemanager.ensure_directory(str(Path(path).parent), backend)

        if not self.filemanager.exists(path, backend):
            return

        if self.write_mode == WriteMode.OVERWRITE:
            self.filemanager.delete(path, backend)
            return

        if self.write_mode == WriteMode.ERROR:
            raise FileExistsError(f"Refusing to overwrite existing path: {path} ({backend})")

        raise ValueError(f"Unknown WriteMode: {self.write_mode!r}")


@dataclass(frozen=True)
class MaterializationContext:
    base_path: str
    backends: list[str]
    backend_kwargs: dict[str, dict[str, Any]]
    filemanager: Any
    extra_inputs: dict[str, Any]
    context: Any = None
    write_mode: WriteMode = WriteMode.OVERWRITE

    def paths(self, options: FileOutputOptions) -> PathHelper:
        return PathHelper(self.base_path, options)

    @property
    def saver(self) -> BackendSaver:
        return BackendSaver(
            self.backends,
            self.filemanager,
            self.backend_kwargs,
            write_mode=self.write_mode,
        )


@dataclass(frozen=True)
class WriterSpec:
    format: MaterializationFormat
    options_type: type
    write: Callable[[Any, Any, MaterializationContext], list[Output]]
    primary_path: Callable[[list[Output]], str]


_WRITERS_BY_OPTIONS: Dict[type, WriterSpec] = {}


def writer_for(
    options_type: type,
    fmt: MaterializationFormat,
    *,
    primary_path: Optional[Callable[[list[Output]], str]] = None,
):
    """Register a writer for a given options type.

    This is intentionally metaprogramming-friendly: adding a new format is
    defining one options dataclass and one function.
    """

    def decorator(fn: Callable[[Any, Any, MaterializationContext], list[Output]]):
        if options_type in _WRITERS_BY_OPTIONS:
            raise ValueError(f"Writer already registered for options type {options_type.__name__}")
        _WRITERS_BY_OPTIONS[options_type] = WriterSpec(
            format=fmt,
            options_type=options_type,
            write=fn,
            primary_path=primary_path or (lambda outs: outs[0].path if outs else ""),
        )
        return fn

    return decorator


def _wants_tabular(options: Any) -> bool:
    return bool(
        getattr(options, "fields", None)
        or getattr(options, "row_field", None)
        or getattr(options, "row_unpacker", None)
        or getattr(options, "row_columns", None)
    )


def _build_tabular_rows(data: Any, options: Any) -> list[dict[str, Any]]:
    # pandas support (optional): treat DataFrame as rows.
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.to_dict(orient="records")
        if isinstance(data, pd.Series):
            return [data.to_dict()]
    except ImportError:
        pass

    items = _as_sequence(data)
    rows: list[dict[str, Any]] = []

    for idx, item in enumerate(items):
        base_row = extract_fields(item, getattr(options, "fields", None))
        if "slice_index" not in base_row:
            base_row["slice_index"] = idx

        row_unpacker = getattr(options, "row_unpacker", None)
        if row_unpacker:
            for exp_row in row_unpacker(item):
                rows.append({**base_row, **exp_row})
            continue

        row_field = getattr(options, "row_field", None)
        if row_field:
            array_data = getattr(item, row_field)
            rows.extend(expand_array_field(array_data, base_row, getattr(options, "row_columns", {}) or {}))
            continue

        if array_fields := discover_array_fields(item):
            primary_field = array_fields[0]
            array_data = getattr(item, primary_field)
            rows.extend(expand_array_field(array_data, base_row, {}))
            continue

        rows.append(base_row)

    return rows


def _render_csv(data: Any, options: CsvOptions) -> str:
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return data.to_csv(index=False)

        rows = _build_tabular_rows(data, options)
        if not rows and options.fields:
            return pd.DataFrame(columns=options.fields).to_csv(index=False)
        return pd.DataFrame(rows).to_csv(index=False)
    except ImportError:
        raise ImportError("CSV materialization requires pandas")


def _render_json(data: Any, options: JsonOptions) -> str:
    # Make common OpenHCS outputs JSON-friendly:
    # - dataclass -> dict
    # - list[dataclass] -> list[dict]
    # - list[dict] unchanged
    # If the options look tabular, use the canonical tabular builder.
    payload: Any
    if _wants_tabular(options):
        payload = _build_tabular_rows(data, options)
    else:
        seq = _as_sequence(data)
        if len(seq) == 1 and seq[0] is data:
            # single element (non-list input)
            payload = extract_fields(data, getattr(options, "fields", None))
        else:
            payload = [extract_fields(item, getattr(options, "fields", None)) for item in seq]

    if options.wrap_list and isinstance(payload, list):
        payload = {"total_items": len(payload), "results": payload}

    return json.dumps(payload, indent=options.indent, default=str)


def _single_file_writer(
    render: Callable[[Any, Any], str],
    *,
    validate_payload: Optional[Callable[[Any, Any], None]] = None,
) -> Callable[[Any, Any, "MaterializationContext"], list[Output]]:
    def write(data: Any, options: Any, ctx: MaterializationContext) -> list[Output]:
        payload = _select_payload(data, options)
        if validate_payload is not None:
            validate_payload(payload, options)
        return [
            Output(
                path=ctx.paths(options).with_suffix(options.filename_suffix),
                content=render(payload, options),
            )
        ]

    return write


def register_single_file_writer(
    options_type: type,
    fmt: MaterializationFormat,
    *,
    render: Callable[[Any, Any], str],
    validate_payload: Optional[Callable[[Any, Any], None]] = None,
    primary_path: Optional[Callable[[list[Output]], str]] = None,
) -> None:
    writer_for(options_type, fmt, primary_path=primary_path)(
        _single_file_writer(render, validate_payload=validate_payload)
    )


register_single_file_writer(CsvOptions, MaterializationFormat.CSV, render=_render_csv)
register_single_file_writer(JsonOptions, MaterializationFormat.JSON, render=_render_json)


def _validate_text(payload: Any, options: TextOptions) -> None:
    if not isinstance(payload, str):
        raise TypeError(f"TextOptions expects a str payload, got {type(payload).__name__}")


register_single_file_writer(
    TextOptions,
    MaterializationFormat.TEXT,
    render=lambda payload, _options: payload,
    validate_payload=_validate_text,
)


def _roi_primary_path(outs: list[Output]) -> str:
    for out in outs:
        if out.path.endswith(".roi.zip"):
            return out.path
    return outs[0].path if outs else ""


@writer_for(ROIOptions, MaterializationFormat.ROI_ZIP, primary_path=_roi_primary_path)
def _write_roi_zip(data: Any, options: ROIOptions, ctx: MaterializationContext) -> list[Output]:
    from polystore.roi import extract_rois_from_labeled_mask

    data = _select_payload(data, options)
    paths = ctx.paths(options)
    roi_path = paths.with_suffix(options.roi_suffix)
    summary_path = paths.with_suffix(options.summary_suffix)

    if _is_empty(data):
        return [Output(path=summary_path, content="No segmentation masks generated (empty data)\n")]

    masks = _as_sequence(data)

    all_rois: list[Any] = []
    for mask in masks:
        rois = extract_rois_from_labeled_mask(
            mask,
            min_area=options.min_area,
            extract_contours=options.extract_contours,
        )
        all_rois.extend(rois)

    outs: list[Output] = []
    if all_rois:
        outs.append(Output(path=roi_path, content=all_rois))

    summary = f"Segmentation ROIs: {len(all_rois)} cells\nZ-planes: {len(masks)}\n"
    if all_rois:
        summary += f"ROI file: {roi_path}\n"
    else:
        summary += "No ROIs extracted (all regions below min_area threshold)\n"
    outs.append(Output(path=summary_path, content=summary))
    return outs


@writer_for(TiffStackOptions, MaterializationFormat.TIFF_STACK)
def _write_tiff_stack(data: Any, options: TiffStackOptions, ctx: MaterializationContext) -> list[Output]:
    data = _select_payload(data, options)
    paths = ctx.paths(options)
    base_name = paths.name

    if _is_empty(data):
        summary_path = paths.with_suffix(options.summary_suffix)
        return [Output(path=summary_path, content=options.empty_summary)]

    if isinstance(data, (list, tuple)):
        slices = list(data)
    else:
        ndim = getattr(data, "ndim", None)
        if ndim == 3:
            slices = [data[i] for i in range(data.shape[0])]  # type: ignore[index]
        else:
            slices = [data]

    outs: list[Output] = []
    for i, arr in enumerate(slices):
        filename = str(paths.parent / f"{base_name}{options.slice_pattern.format(index=i)}")
        out_arr = arr
        if options.normalize_uint8 and getattr(out_arr, "dtype", None) != "uint8":
            max_val = getattr(out_arr, "max", lambda: 0)()
            out_arr = (out_arr * 255).astype("uint8") if max_val <= 1.0 else out_arr.astype("uint8")
        outs.append(Output(path=filename, content=out_arr))

    summary_path = paths.with_suffix(options.summary_suffix)
    first = slices[0] if slices else None
    summary_content = (
        f"Images saved: {len(slices)} files\n"
        f"Base filename pattern: {base_name}{options.slice_pattern}\n"
        f"Image dtype: {getattr(first, 'dtype', 'unknown')}\n"
        f"Image shape: {getattr(first, 'shape', 'unknown')}\n"
    )
    outs.append(Output(path=summary_path, content=summary_content))
    return outs


@dataclass(frozen=True, init=False)
class MaterializationSpec:
    """Declarative materialization spec.

    The spec is a list of *writer options* objects. Writer dispatch is inferred
    from the option type.
    """

    outputs: Tuple[Any, ...]
    allowed_backends: Optional[List[str]]
    primary: int

    def __init__(self, *outputs: Any, allowed_backends: Optional[List[str]] = None, primary: int = 0):
        if len(outputs) == 1 and isinstance(outputs[0], (list, tuple)):
            outputs = tuple(outputs[0])

        if not outputs:
            raise ValueError("MaterializationSpec requires at least one output options object")

        for opt in outputs:
            if isinstance(opt, dict):
                raise TypeError("dict-based materialization options are not supported")
            if type(opt) not in _WRITERS_BY_OPTIONS:
                raise ValueError(
                    f"No writer registered for options type {type(opt).__name__}. "
                    f"Registered: {[t.__name__ for t in _WRITERS_BY_OPTIONS.keys()]}"
                )

        if primary < 0 or primary >= len(outputs):
            raise IndexError("MaterializationSpec.primary out of range")

        object.__setattr__(self, "outputs", tuple(outputs))
        object.__setattr__(self, "allowed_backends", allowed_backends)
        object.__setattr__(self, "primary", primary)

    @classmethod
    def __objectstate_rebuild__(
        cls,
        *,
        outputs: Tuple[Any, ...],
        allowed_backends: Optional[List[str]] = None,
        primary: int = 0,
    ) -> "MaterializationSpec":
        # Rebuild via the normal constructor to keep validation behavior.
        return cls(*outputs, allowed_backends=allowed_backends, primary=primary)


def _normalize_backends(backends: Sequence[str] | str) -> list[str]:
    if isinstance(backends, str):
        return [backends]
    return list(backends)


def _validate_allowed_backends(spec: MaterializationSpec, backends: list[str]) -> None:
    if not spec.allowed_backends:
        return
    invalid = [b for b in backends if b not in spec.allowed_backends]
    if invalid:
        raise ValueError(f"Backend(s) {invalid} not in allowed backends for this spec: {spec.allowed_backends}")


def materialize(
    spec: MaterializationSpec,
    data: Any,
    path: str,
    filemanager: Any,
    backends: Sequence[str] | str,
    backend_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    context: Any = None,
    extra_inputs: Optional[Dict[str, Any]] = None,
    *,
    write_mode: WriteMode = WriteMode.OVERWRITE,
) -> str:
    """Materialize data to one or more backends."""

    normalized_backends = _normalize_backends(backends)
    _validate_allowed_backends(spec, normalized_backends)

    ctx = MaterializationContext(
        base_path=path,
        backends=normalized_backends,
        backend_kwargs=backend_kwargs or {},
        filemanager=filemanager,
        extra_inputs=extra_inputs or {},
        context=context,
        write_mode=write_mode,
    )

    primary_path = ""

    for i, opt in enumerate(spec.outputs):
        writer = _WRITERS_BY_OPTIONS[type(opt)]
        outs = writer.write(data, opt, ctx)
        for out in outs:
            ctx.saver.save(out.content, out.path)
        if i == spec.primary:
            primary_path = writer.primary_path(outs)

    return primary_path
