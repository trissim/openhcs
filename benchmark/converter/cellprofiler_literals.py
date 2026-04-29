"""CellProfiler escaped setting-name/value literal decoding."""

from __future__ import annotations

import warnings


def decode_cellprofiler_setting_literal(value: str) -> str:
    """Decode escaped CellProfiler setting labels and values.

    Some official CP3 example files store text as escaped byte literals, including
    UTF-16LE strings with BOMs.  The converter should expose normalized Python
    text before semantic matching sees the setting.
    """
    if "\\" not in value:
        return value
    decoded = _decode_escape_sequences(value)
    return _decode_utf16_text(decoded)


def _decode_escape_sequences(value: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return bytes(value, "utf-8").decode("unicode_escape")


def _decode_utf16_text(value: str) -> str:
    if not value.startswith(("\xff\xfe", "\xfe\xff")):
        return value
    try:
        return value.encode("latin-1").decode("utf-16")
    except UnicodeError:
        return value
