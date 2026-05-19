"""Shared enum metadata helpers for CellProfiler-compatible backend families."""

from __future__ import annotations

from typing import ClassVar


class CellProfilerEnumAttributeMixin:
    """Attach tuple-valued enum residue to named member attributes."""

    __cellprofiler_attribute_names__: ClassVar[tuple[str, ...]] = ()

    def __new__(cls, value: object, *attribute_values: object):
        member = object.__new__(cls)
        member._value_ = value
        attribute_names = cls.__cellprofiler_attribute_names__
        if len(attribute_values) != len(attribute_names):
            raise TypeError(
                f"{cls.__name__} expected {len(attribute_names)} metadata "
                f"value(s), got {len(attribute_values)}."
            )
        for attribute_name, attribute_value in zip(attribute_names, attribute_values):
            setattr(member, attribute_name, attribute_value)
        return member
