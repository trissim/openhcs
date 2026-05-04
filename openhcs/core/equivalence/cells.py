"""Scalar cell signatures for runtime equivalence."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum

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
        canonical = repr(round(numeric, policy.numeric_decimal_places))
    return RuntimeCellSignature(RuntimeCellValueKind.NUMBER, canonical)


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
