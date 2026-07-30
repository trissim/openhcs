"""Shared nominal contracts for self-supervised deconvolution backends."""

from enum import Enum


class DeconvolutionBlurMode(Enum):
    """Blur model optimized jointly with a deconvolution network."""

    LEARNED = "learned"
    FFT = "fft"
    GAUSSIAN = "gaussian"

    @property
    def uses_fixed_kernel(self) -> bool:
        return self in {self.FFT, self.GAUSSIAN}
