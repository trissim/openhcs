"""Benchmark report and figure generation."""

from benchmark.reports.cppipe_figures import SummarySource
from benchmark.reports.cppipe_figures import generate_cppipe_benchmark_figures
from benchmark.reports.cppipe_figures import parse_summary_source
from benchmark.reports.cppipe_scaling_figures import generate_cppipe_scaling_figures

__all__ = (
    "SummarySource",
    "generate_cppipe_benchmark_figures",
    "generate_cppipe_scaling_figures",
    "parse_summary_source",
)
