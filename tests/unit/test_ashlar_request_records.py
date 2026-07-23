import importlib

import numpy as np

from openhcs.processing.backends.pos_gen.ashlar_config import (
    AshlarAlignmentConfig,
    AshlarPositionRequest,
)
from openhcs.processing.backends.pos_gen import ashlar_main_cpu


def test_ashlar_alignment_config_defaults_are_shared():
    config = AshlarAlignmentConfig()

    assert config.pixel_size == 1.0
    assert config.max_shift == 30.0
    assert config.stitch_alpha == 0.05
    assert config.window_size_factor == 0.15


def test_cpu_public_function_projects_alignment_config(monkeypatch):
    captured = {}

    class FakeAligner:
        def __init__(self, image_stack, positions, tile_size, alignment_config):
            captured["image_stack"] = image_stack
            captured["positions"] = positions
            captured["tile_size"] = tile_size
            captured["alignment_config"] = alignment_config
            self.final_positions = positions

        def run(self):
            captured["ran"] = True

    image_stack = np.zeros((4, 8, 10), dtype=np.float32)
    monkeypatch.setattr(ashlar_main_cpu, "ArrayEdgeAligner", FakeAligner)

    result_stack, positions = ashlar_main_cpu.ashlar_compute_tile_positions_cpu.__wrapped__(
        image_stack,
        (2, 2),
        overlap_ratio=0.25,
        pixel_size=0.5,
        max_shift=12.0,
        stitch_alpha=0.2,
        max_error=3.0,
        randomize=True,
        verbose=True,
        upsample_factor=7,
        permutation_upsample=2,
        permutation_samples=11,
        min_permutation_samples=3,
        max_permutation_tries=13,
        window_size_factor=0.4,
    )

    assert result_stack is image_stack
    assert positions == [(0.0, 0.0), (7.5, 0.0), (0.0, 6.0), (7.5, 6.0)]
    assert captured["ran"] is True
    np.testing.assert_array_equal(captured["tile_size"], np.array([8, 10]))
    assert captured["alignment_config"] == AshlarAlignmentConfig(
        pixel_size=0.5,
        max_shift=12.0,
        stitch_alpha=0.2,
        max_error=3.0,
        randomize=True,
        verbose=True,
        upsample_factor=7,
        permutation_upsample=2,
        permutation_samples=11,
        min_permutation_samples=3,
        max_permutation_tries=13,
        window_size_factor=0.4,
    )


def test_cpu_aligner_consumes_alignment_config():
    image_stack = np.zeros((4, 8, 10), dtype=np.float32)
    positions = np.array([[0.0, 0.0], [7.0, 0.0], [0.0, 7.0], [7.0, 7.0]])
    config = AshlarAlignmentConfig(
        pixel_size=0.25,
        max_shift=8.0,
        stitch_alpha=0.15,
        max_error=1.5,
        randomize=True,
        verbose=True,
        upsample_factor=9,
        permutation_upsample=3,
        permutation_samples=17,
        min_permutation_samples=5,
        max_permutation_tries=19,
        window_size_factor=0.35,
    )

    aligner = ashlar_main_cpu.ArrayEdgeAligner(
        image_stack=image_stack,
        positions=positions,
        tile_size=np.array([8, 10]),
        alignment_config=config,
    )

    assert aligner.pixel_size == 0.25
    assert aligner.max_shift_pixels == 32.0
    assert aligner.alpha == 0.15
    assert aligner.max_error == 1.5
    assert aligner.randomize is True
    assert aligner.verbose is True
    assert aligner.upsample_factor == 9
    assert aligner.permutation_upsample == 3
    assert aligner.permutation_samples == 17
    assert aligner.min_permutation_samples == 5
    assert aligner.max_permutation_tries == 19
    assert aligner.window_size_factor == 0.35


def test_gpu_public_function_projects_alignment_config(monkeypatch):
    gpu_module = importlib.import_module("openhcs.processing.backends.pos_gen.ashlar_main_gpu")

    class FakeCuPy:
        ndarray = np.ndarray
        float64 = np.float64

        @staticmethod
        def asarray(value):
            return np.asarray(value)

        @staticmethod
        def array(value, dtype=None):
            return np.array(value, dtype=dtype)

        @staticmethod
        def asnumpy(value):
            return np.asarray(value)

    monkeypatch.setattr(gpu_module, "cp", FakeCuPy)

    captured = {}

    class FakeGPUAligner:
        def __init__(self, image_stack, positions, tile_size, alignment_config):
            captured["image_stack"] = image_stack
            captured["positions"] = positions
            captured["tile_size"] = tile_size
            captured["alignment_config"] = alignment_config
            self.final_positions = positions

        def run(self):
            captured["ran"] = True

    image_stack = np.zeros((4, 8, 10), dtype=np.float32)
    monkeypatch.setattr(gpu_module, "ArrayEdgeAlignerGPU", FakeGPUAligner)

    result_stack, positions = gpu_module.ashlar_compute_tile_positions_gpu.__wrapped__(
        image_stack,
        (2, 2),
        overlap_ratio=0.25,
        pixel_size=0.5,
        max_shift=12.0,
        stitch_alpha=0.2,
        max_error=3.0,
        randomize=True,
        verbose=True,
        upsample_factor=7,
        permutation_upsample=2,
        permutation_samples=11,
        min_permutation_samples=3,
        max_permutation_tries=13,
        window_size_factor=0.4,
    )

    assert isinstance(result_stack, np.ndarray)
    assert positions == [(0.0, 0.0), (7.5, 0.0), (0.0, 6.0), (7.5, 6.0)]
    assert captured["ran"] is True
    assert captured["alignment_config"] == AshlarAlignmentConfig(
        pixel_size=0.5,
        max_shift=12.0,
        stitch_alpha=0.2,
        max_error=3.0,
        randomize=True,
        verbose=True,
        upsample_factor=7,
        permutation_upsample=2,
        permutation_samples=11,
        min_permutation_samples=3,
        max_permutation_tries=13,
        window_size_factor=0.4,
    )


def test_position_request_groups_payload_and_alignment():
    image_stack = np.zeros((1, 2, 3))
    config = AshlarAlignmentConfig(max_shift=4.0)
    request = AshlarPositionRequest(
        image_stack=image_stack,
        grid_dimensions=(1, 1),
        overlap_ratio=0.2,
        alignment=config,
    )

    assert request.image_stack is image_stack
    assert request.grid_dimensions == (1, 1)
    assert request.alignment is config
