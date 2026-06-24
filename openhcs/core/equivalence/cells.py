"""Scalar cell signatures for runtime equivalence."""

from __future__ import annotations

import hashlib
import math
import pickle
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from types import MappingProxyType
from typing import Any

from metaclass_registry import AutoRegisterMeta

from openhcs.core.equivalence.arrays import canonical_scalar, semantic_array_payload
from openhcs.core.equivalence.policy import RuntimeEquivalencePolicy


class RuntimeCellValueKind(str, Enum):
    """Canonical scalar families used for exported table comparison."""

    EMPTY = "empty"
    NUMBER = "number"
    TEXT = "text"


@dataclass(frozen=True, slots=True)
class RuntimeCellSignature:
    """Canonical scalar value for exported table comparison."""

    kind: RuntimeCellValueKind
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            (
                self.kind
                if isinstance(self.kind, RuntimeCellValueKind)
                else RuntimeCellValueKind(self.kind)
            ),
        )

    @property
    def sort_key(self) -> tuple[str, str]:
        """Return a stable ordering key for mixed scalar families."""
        return (self.kind.value, self.value)

    def to_cache_payload(self) -> tuple[str, str]:
        """Return a pickle/JSON-stable semantic cache payload."""
        return (self.kind.value, self.value)

    @classmethod
    def from_cache_payload(cls, payload: object) -> "RuntimeCellSignature":
        """Rebuild a cell signature from a semantic cache payload."""
        kind, value = payload  # type: ignore[misc]
        return cls(RuntimeCellValueKind(str(kind)), str(value))


class RuntimeCellMissingStrategy(ABC, metaclass=AutoRegisterMeta):
    """Closed RuntimeCellValueKind strategy for missing-value semantics."""

    __registry_key__ = "kind"
    __skip_if_no_key__ = True
    kind: RuntimeCellValueKind | None = None

    @classmethod
    def for_kind(
        cls,
        kind: RuntimeCellValueKind,
    ) -> "RuntimeCellMissingStrategy":
        strategy_type = _RUNTIME_CELL_MISSING_STRATEGY_BY_KIND[kind]
        return strategy_type()

    @abstractmethod
    def is_missing(self, value: RuntimeCellSignature) -> bool:
        """Return whether this cell signature should be treated as missing."""


class EmptyRuntimeCellMissingStrategy(RuntimeCellMissingStrategy):
    kind = RuntimeCellValueKind.EMPTY

    def is_missing(self, value: RuntimeCellSignature) -> bool:
        return True


class NumberRuntimeCellMissingStrategy(RuntimeCellMissingStrategy):
    kind = RuntimeCellValueKind.NUMBER

    def is_missing(self, value: RuntimeCellSignature) -> bool:
        try:
            return math.isnan(float(value.value))
        except ValueError:
            return False


class TextRuntimeCellMissingStrategy(RuntimeCellMissingStrategy):
    kind = RuntimeCellValueKind.TEXT

    def is_missing(self, value: RuntimeCellSignature) -> bool:
        return False


_RUNTIME_CELL_MISSING_STRATEGY_BY_KIND = MappingProxyType(
    dict(RuntimeCellMissingStrategy.__registry__)
)
if set(_RUNTIME_CELL_MISSING_STRATEGY_BY_KIND) != set(RuntimeCellValueKind):
    missing_kinds = sorted(
        set(RuntimeCellValueKind) - set(_RUNTIME_CELL_MISSING_STRATEGY_BY_KIND),
        key=lambda kind: kind.value,
    )
    extra_kinds = sorted(
        set(_RUNTIME_CELL_MISSING_STRATEGY_BY_KIND) - set(RuntimeCellValueKind),
        key=lambda kind: kind.value,
    )
    raise ValueError(
        "Runtime cell missing strategies must cover RuntimeCellValueKind exactly: "
        f"missing={missing_kinds!r}, extra={extra_kinds!r}."
    )


@dataclass(frozen=True, slots=True)
class _SparseNumericInstability:
    """Policy fragment for sparse numeric runtime-equivalence instability."""

    abs_tolerance: float
    rel_tolerance: float
    max_unstable_values: int
    max_unstable_fraction: float

    @classmethod
    def from_values(
        cls,
        abs_tolerance: float,
        rel_tolerance: float,
        max_unstable_values: int,
        max_unstable_fraction: float,
    ) -> "_SparseNumericInstability":
        return cls(
            abs_tolerance=abs_tolerance,
            rel_tolerance=rel_tolerance,
            max_unstable_values=max_unstable_values,
            max_unstable_fraction=max_unstable_fraction,
        )

    def unstable_cap(self, stable_value_count: int) -> int:
        return max(
            self.max_unstable_values,
            math.ceil(stable_value_count * self.max_unstable_fraction),
        )

    def relaxed_policy(
        self,
        base_policy: RuntimeEquivalencePolicy,
    ) -> RuntimeEquivalencePolicy:
        return RuntimeEquivalencePolicy(
            numeric_decimal_places=base_policy.numeric_decimal_places,
            numeric_abs_tolerance=self.abs_tolerance,
            numeric_rel_tolerance=self.rel_tolerance,
            measurement_feature_name_mode=base_policy.measurement_feature_name_mode,
        )


def runtime_cell_signature(
    value: str,
    policy: RuntimeEquivalencePolicy,
) -> RuntimeCellSignature:
    """Return a canonical scalar signature for runtime table comparison."""
    text = value.strip()
    if not text:
        return RuntimeCellSignature(RuntimeCellValueKind.EMPTY, "")
    try:
        numeric = float(text)
    except ValueError:
        return RuntimeCellSignature(RuntimeCellValueKind.TEXT, text)
    if math.isnan(numeric):
        canonical = "nan"
    elif math.isinf(numeric):
        canonical = "inf" if numeric > 0 else "-inf"
    else:
        rounded = round(numeric, policy.numeric_decimal_places)
        canonical = repr(0.0 if rounded == 0 else rounded)
    return RuntimeCellSignature(RuntimeCellValueKind.NUMBER, canonical)


def measurement_table_cell_payload(value: object) -> object:
    """Return a hashable exact payload for measurement-table cells."""
    value = canonical_scalar(value)
    if value is None:
        return None
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float) and math.isnan(value):
        return ("float", "nan")
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (
                    measurement_table_cell_payload(key),
                    measurement_table_cell_payload(nested_value),
                )
                for key, nested_value in value.items()
            ),
        )
    if isinstance(value, (tuple, list)):
        return (
            type(value).__name__,
            tuple(measurement_table_cell_payload(item) for item in value),
        )
    array_payload = semantic_array_payload(value)
    if array_payload is not None:
        return array_payload
    return (type(value).__name__, repr(value))


def update_measurement_table_cell_hash(digest: Any, value: object) -> None:
    """Update an exact cell digest without materializing nested payload trees."""
    value = canonical_scalar(value)
    if value is None:
        digest.update(pickle.dumps(("none",), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, str):
        digest.update(pickle.dumps(("str", value), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, bool):
        digest.update(pickle.dumps(("bool", value), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, int):
        digest.update(pickle.dumps(("int", value), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, float) and math.isnan(value):
        digest.update(pickle.dumps(("float", "nan"), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, float):
        digest.update(
            pickle.dumps(("float", repr(value)), protocol=pickle.HIGHEST_PROTOCOL)
        )
        return
    array_payload = semantic_array_payload(value)
    if array_payload is not None:
        digest.update(pickle.dumps(array_payload, protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, Mapping):
        digest.update(
            pickle.dumps(("mapping", len(value)), protocol=pickle.HIGHEST_PROTOCOL)
        )
        for key, nested_value in value.items():
            update_measurement_table_cell_hash(digest, key)
            update_measurement_table_cell_hash(digest, nested_value)
        digest.update(
            pickle.dumps(("mapping_end",), protocol=pickle.HIGHEST_PROTOCOL)
        )
        return
    if isinstance(value, tuple):
        digest.update(
            pickle.dumps(("tuple", len(value)), protocol=pickle.HIGHEST_PROTOCOL)
        )
        for item in value:
            update_measurement_table_cell_hash(digest, item)
        digest.update(pickle.dumps(("tuple_end",), protocol=pickle.HIGHEST_PROTOCOL))
        return
    if isinstance(value, list):
        digest.update(
            pickle.dumps(("list", len(value)), protocol=pickle.HIGHEST_PROTOCOL)
        )
        for item in value:
            update_measurement_table_cell_hash(digest, item)
        digest.update(pickle.dumps(("list_end",), protocol=pickle.HIGHEST_PROTOCOL))
        return
    digest.update(
        pickle.dumps((type(value).__name__, repr(value)), protocol=pickle.HIGHEST_PROTOCOL)
    )


@lru_cache(maxsize=256)
def _runtime_value_type_is_mapping(value_type: type[object]) -> bool:
    return issubclass(value_type, Mapping)


def runtime_value_is_mapping(value: object) -> bool:
    return _runtime_value_type_is_mapping(type(value))


_RUNTIME_NUMERIC_TEXT_RE = re.compile(
    r"^[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|inf(?:inity)?|nan)$",
    re.IGNORECASE,
)


def runtime_numeric_text_value(text: str) -> float | None:
    stripped = text.strip()
    if not stripped or _RUNTIME_NUMERIC_TEXT_RE.fullmatch(stripped) is None:
        return None
    return float(stripped)


@lru_cache(maxsize=131072)
def _cached_runtime_cell_signature(
    text: str,
    numeric_decimal_places: int,
) -> RuntimeCellSignature:
    stripped = text.strip()
    if not stripped:
        return RuntimeCellSignature(RuntimeCellValueKind.EMPTY, "")
    numeric = runtime_numeric_text_value(stripped)
    if numeric is None:
        return RuntimeCellSignature(RuntimeCellValueKind.TEXT, stripped)
    if math.isnan(numeric):
        canonical = "nan"
    elif math.isinf(numeric):
        canonical = "inf" if numeric > 0 else "-inf"
    else:
        rounded = round(numeric, numeric_decimal_places)
        canonical = repr(0.0 if rounded == 0 else rounded)
    return RuntimeCellSignature(RuntimeCellValueKind.NUMBER, canonical)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellSignatureProjection:
    """Project runtime measurement payloads into comparison cell signatures."""

    value: object
    policy: RuntimeEquivalencePolicy

    def signature(self) -> RuntimeCellSignature:
        value = canonical_scalar(self.value)
        if value is None or isinstance(value, (str, bool, int, float)):
            return _cached_runtime_cell_signature(
                str(value),
                self.policy.numeric_decimal_places,
            )
        array_payload = semantic_array_payload(value)
        if array_payload is not None:
            dtype, shape, digest = array_payload[1:]
            return RuntimeCellSignature(
                RuntimeCellValueKind.TEXT,
                f"array:{dtype}:{'x'.join(str(axis) for axis in shape)}:{digest}",
            )
        if runtime_value_is_mapping(value) or isinstance(value, (tuple, list)):
            value_digest = hashlib.blake2b(digest_size=32)
            update_measurement_table_cell_hash(value_digest, value)
            return RuntimeCellSignature(
                RuntimeCellValueKind.TEXT,
                f"{type(value).__name__}:{value_digest.hexdigest()}",
            )
        return runtime_cell_signature(str(value), self.policy)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementCellPresence:
    """Presence semantics for runtime measurement cell payloads."""

    value: object

    def is_present(self) -> bool:
        value = canonical_scalar(self.value)
        if value is None:
            return False
        array_payload = semantic_array_payload(value)
        if array_payload is not None:
            return any(axis > 0 for axis in array_payload[2])
        if runtime_value_is_mapping(value) or isinstance(value, (tuple, list)):
            return bool(value)
        text = str(value).strip()
        if not text:
            return False
        numeric = runtime_numeric_text_value(text)
        if numeric is None:
            return True
        return not math.isnan(numeric)


@dataclass(frozen=True, slots=True)
class RuntimeMeasurementValuePresence:
    """Presence semantics for scalar or nested runtime measurement values."""

    value: object

    def is_present(self) -> bool:
        if runtime_value_is_mapping(self.value):
            return any(
                RuntimeMeasurementValuePresence(nested).is_present()
                for nested in self.value.values()
            )
        return RuntimeMeasurementCellPresence(self.value).is_present()


def measurement_numeric_runtime_value(
    value: object,
    policy: RuntimeEquivalencePolicy,
) -> float | None:
    numeric_value = runtime_numeric_text_value(str(value))
    if numeric_value is None:
        return None
    if math.isnan(numeric_value):
        return None
    if math.isfinite(numeric_value):
        return float(round(numeric_value, policy.numeric_decimal_places))
    return numeric_value


def cell_signature_numeric_value(value: RuntimeCellSignature) -> float | None:
    if value.kind is not RuntimeCellValueKind.NUMBER:
        return None
    numeric_value = float(value.value)
    if math.isnan(numeric_value):
        return None
    return numeric_value


def runtime_cell_signature_counters_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    """Return whether two canonical cell multisets are equivalent."""
    if reference == candidate:
        return True
    if policy.numeric_abs_tolerance == 0 and policy.numeric_rel_tolerance == 0:
        return False

    reference_exact, reference_numbers = split_approximate_numeric_signatures(reference)
    candidate_exact, candidate_numbers = split_approximate_numeric_signatures(candidate)
    return (
        reference_exact == candidate_exact
        and finite_numeric_values_equivalent(
            reference_numbers,
            candidate_numbers,
            policy,
        )
    )


def sparse_numeric_counters_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    policy: RuntimeEquivalencePolicy,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
    max_unstable_values: int,
    max_unstable_fraction: float,
) -> bool:
    """Return whether sparse numeric instability is within policy bounds."""
    instability = _SparseNumericInstability.from_values(
        abs_tolerance,
        rel_tolerance,
        max_unstable_values,
        max_unstable_fraction,
    )
    reference_exact, reference_numbers = split_approximate_numeric_signatures(reference)
    candidate_exact, candidate_numbers = split_approximate_numeric_signatures(candidate)
    if not _sparse_nonfinite_numeric_counters_equivalent(
        reference_exact,
        candidate_exact,
        instability=instability,
    ):
        return False
    return sparse_numeric_values_equivalent(
        reference_numbers,
        candidate_numbers,
        policy,
        instability=instability,
    )


def absolute_numeric_counters_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    """Return whether absolute finite numeric values are equivalent."""
    reference_exact, reference_numbers = split_approximate_numeric_signatures(reference)
    candidate_exact, candidate_numbers = split_approximate_numeric_signatures(candidate)
    if reference_exact != candidate_exact:
        return False
    return finite_numeric_values_equivalent(
        tuple(abs(value) for value in reference_numbers),
        tuple(abs(value) for value in candidate_numbers),
        policy,
    )


def sparse_absolute_numeric_counters_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    policy: RuntimeEquivalencePolicy,
    *,
    abs_tolerance: float,
    rel_tolerance: float,
    max_unstable_values: int,
    max_unstable_fraction: float,
) -> bool:
    """Return whether sparse absolute numeric instability is policy-equivalent."""
    instability = _SparseNumericInstability.from_values(
        abs_tolerance,
        rel_tolerance,
        max_unstable_values,
        max_unstable_fraction,
    )
    reference_exact, reference_numbers = split_approximate_numeric_signatures(reference)
    candidate_exact, candidate_numbers = split_approximate_numeric_signatures(candidate)
    if reference_exact != candidate_exact:
        return False
    return sparse_numeric_values_equivalent(
        tuple(abs(value) for value in reference_numbers),
        tuple(abs(value) for value in candidate_numbers),
        policy,
        instability=instability,
    )


def split_approximate_numeric_signatures(
    signatures: Counter[RuntimeCellSignature],
) -> tuple[Counter[RuntimeCellSignature], tuple[float, ...]]:
    """Split exact non-finite/text signatures from finite numeric values."""
    exact: Counter[RuntimeCellSignature] = Counter()
    numbers: list[float] = []
    for signature, count in signatures.items():
        numeric = finite_signature_number(signature)
        if numeric is None:
            exact[signature] = count
            continue
        numbers.extend([numeric] * count)
    return exact, tuple(numbers)


def finite_signature_number(signature: RuntimeCellSignature) -> float | None:
    """Return a finite numeric value carried by a cell signature."""
    if signature.kind is not RuntimeCellValueKind.NUMBER:
        return None
    try:
        numeric = float(signature.value)
    except ValueError:
        return None
    return numeric if math.isfinite(numeric) else None


def finite_numeric_values_equivalent(
    reference: tuple[float, ...],
    candidate: tuple[float, ...],
    policy: RuntimeEquivalencePolicy,
) -> bool:
    """Return whether finite numeric values can be one-to-one tolerance matched."""
    if len(reference) != len(candidate):
        return False
    unmatched_reference, unmatched_candidate = unmatched_numeric_values(
        reference,
        candidate,
        policy,
    )
    return not unmatched_reference and not unmatched_candidate


def sparse_numeric_values_equivalent(
    reference_numbers: tuple[float, ...],
    candidate_numbers: tuple[float, ...],
    policy: RuntimeEquivalencePolicy,
    *,
    instability: _SparseNumericInstability,
) -> bool:
    """Return whether sparse finite numeric differences fit instability bounds."""
    unstable_cap = instability.unstable_cap(len(reference_numbers))
    unstable_policy = instability.relaxed_policy(policy)
    unmatched_reference, unmatched_candidate = unmatched_numeric_values(
        reference_numbers,
        candidate_numbers,
        policy,
    )
    relaxed_unmatched_reference, relaxed_unmatched_candidate = unmatched_numeric_values(
        unmatched_reference,
        unmatched_candidate,
        unstable_policy,
    )
    return (
        max(len(relaxed_unmatched_reference), len(relaxed_unmatched_candidate))
        <= unstable_cap
    )


def unmatched_numeric_values(
    reference: tuple[float, ...],
    candidate: tuple[float, ...],
    policy: RuntimeEquivalencePolicy,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return numeric values left after stable one-to-one tolerance matching."""
    reference_values = sorted(reference)
    candidate_values = sorted(candidate)
    unmatched_reference: list[float] = []
    unmatched_candidate: list[float] = []
    reference_index = 0
    candidate_index = 0
    while (
        reference_index < len(reference_values)
        and candidate_index < len(candidate_values)
    ):
        reference_value = reference_values[reference_index]
        candidate_value = candidate_values[candidate_index]
        if numbers_equivalent(reference_value, candidate_value, policy):
            reference_index += 1
            candidate_index += 1
            continue
        tolerance = max(
            policy.numeric_abs_tolerance,
            policy.numeric_rel_tolerance
            * max(abs(reference_value), abs(candidate_value)),
        )
        if candidate_value < reference_value - tolerance:
            unmatched_candidate.append(candidate_value)
            candidate_index += 1
            continue
        unmatched_reference.append(reference_value)
        reference_index += 1

    unmatched_reference.extend(reference_values[reference_index:])
    unmatched_candidate.extend(candidate_values[candidate_index:])
    return tuple(unmatched_reference), tuple(unmatched_candidate)


def numbers_equivalent(
    reference: float,
    candidate: float,
    policy: RuntimeEquivalencePolicy,
) -> bool:
    """Return whether two scalar numbers are within policy tolerance."""
    tolerance = max(
        policy.numeric_abs_tolerance,
        policy.numeric_rel_tolerance * max(abs(reference), abs(candidate)),
    )
    return abs(reference - candidate) <= tolerance


def _sparse_nonfinite_numeric_counters_equivalent(
    reference: Counter[RuntimeCellSignature],
    candidate: Counter[RuntimeCellSignature],
    *,
    instability: _SparseNumericInstability,
) -> bool:
    if reference == candidate:
        return True
    if any(not _signature_is_nonfinite_number(signature) for signature in reference):
        return False
    if any(not _signature_is_nonfinite_number(signature) for signature in candidate):
        return False

    unstable_cap = instability.unstable_cap(sum(reference.values()))
    missing = sum((reference - candidate).values())
    extra = sum((candidate - reference).values())
    return max(missing, extra) <= unstable_cap


def _signature_is_nonfinite_number(signature: RuntimeCellSignature) -> bool:
    if signature.kind is not RuntimeCellValueKind.NUMBER:
        return False
    try:
        numeric = float(signature.value)
    except ValueError:
        return False
    return not math.isfinite(numeric)
