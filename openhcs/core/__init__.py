"""Core module for openhcs."""

from enum import Enum


class CoreExport(Enum):
    PROCESSING_CONTEXT = "ProcessingContext"
    STEP = "Step"


def _load_processing_context():
    from openhcs.core.context.processing_context import ProcessingContext

    return ProcessingContext


def _load_step():
    from openhcs.core.steps.abstract import AbstractStep

    return AbstractStep


_LAZY_EXPORTS = {
    CoreExport.PROCESSING_CONTEXT: _load_processing_context,
    CoreExport.STEP: _load_step,
}


def __getattr__(name):
    """Lazy public re-exports without import-time pipeline/runtime side effects."""

    try:
        export = CoreExport(name)
    except ValueError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    try:
        return _LAZY_EXPORTS[export]()
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

__all__ = [
    'ProcessingContext',
    'Step',
]
