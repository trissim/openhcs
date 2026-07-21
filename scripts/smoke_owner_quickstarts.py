#!/usr/bin/env python3
"""Exercise one dependency-light quick-start path for each first-party owner."""

from __future__ import annotations

from dataclasses import dataclass
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np


@dataclass
class SmokeConfig:
    name: str = "default"
    workers: int = 4


def smoke_objectstate() -> None:
    from objectstate import LazyDataclassFactory, set_base_config_type

    set_base_config_type(SmokeConfig)
    lazy_config = LazyDataclassFactory.make_lazy_simple(SmokeConfig)
    projected = lazy_config.from_config(SmokeConfig(name="production", workers=8))
    assert projected.name == "production"
    assert projected.workers == 8


def smoke_metaclass_registry() -> None:
    from metaclass_registry import AutoRegisterMeta, RegistryFamily

    class Plugin(metaclass=AutoRegisterMeta):
        __registry_family__ = RegistryFamily("name", registry_name="plugin")
        __registry__ = {}
        name = None

    class Email(Plugin):
        name = "email"

    assert Plugin.__registry__["email"] is Email


def smoke_arraybridge() -> None:
    from arraybridge import (
        convert_memory,
        detect_memory_type,
        numpy,
        stack_slices,
        unstack_slices,
    )

    value = np.arange(6).reshape(2, 3)
    assert detect_memory_type(value) == "numpy"
    assert np.array_equal(convert_memory(value, "numpy", "numpy", 0), value)

    @numpy
    def double(image):
        return image * 2

    assert np.array_equal(double(value), value * 2)
    stack = stack_slices([value, value], "numpy", 0)
    assert len(unstack_slices(stack, "numpy", 0)) == 2


def smoke_polystore() -> None:
    from polystore import FileManager, MemoryBackend

    value = np.arange(6).reshape(2, 3)
    files = FileManager({"memory": MemoryBackend()})
    files.ensure_directory("/docs-smoke", backend="memory")
    files.save(value, "/docs-smoke/value.npy", backend="memory")
    assert np.array_equal(
        files.load("/docs-smoke/value.npy", backend="memory"),
        value,
    )


def smoke_python_introspect() -> None:
    from python_introspect import SignatureAnalyzer

    def example(a: int, b: str = "default") -> bool:
        return bool(a and b)

    parameters = SignatureAnalyzer().analyze(example)
    assert tuple(parameters) == ("a", "b")


def smoke_zmqruntime() -> None:
    from zmqruntime import TransportMode, ZMQClient, ZMQConfig, ZMQServer
    from zmqruntime.execution import BatchSubmitWaitEngine, ExecutionStatusPoller

    assert ZMQClient and ZMQServer and ZMQConfig and TransportMode
    assert BatchSubmitWaitEngine and ExecutionStatusPoller


def smoke_pycodify() -> None:
    from pycodify import Assignment, generate_python_source

    source = generate_python_source(
        Assignment("config", SmokeConfig(name="production"))
    )
    assert "config" in source
    assert "SmokeConfig" in source


def smoke_pyqt_reactive() -> None:
    from PyQt6.QtWidgets import QApplication
    from objectstate import ObjectState
    from pyqt_reactive.forms.parameter_form_manager import (
        FormManagerConfig,
        ParameterFormManager,
    )
    from pyqt_reactive.theming import ColorScheme

    app = QApplication.instance() or QApplication([])
    state = ObjectState(SmokeConfig(), scope_id="docs-smoke")
    form = ParameterFormManager(
        state,
        config=FormManagerConfig(color_scheme=ColorScheme()),
    )
    form.close()
    assert app is not None


def main() -> None:
    smoke_objectstate()
    smoke_metaclass_registry()
    smoke_arraybridge()
    smoke_polystore()
    smoke_python_introspect()
    smoke_zmqruntime()
    smoke_pycodify()
    smoke_pyqt_reactive()
    print("first-party owner quick-start smoke tests passed")


if __name__ == "__main__":
    main()
