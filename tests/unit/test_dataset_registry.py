from benchmark.datasets.registry import BBBC021_SINGLE_PLATE


def test_bbbc021_dataset_spec_exposes_reference_cppipe_urls() -> None:
    assert BBBC021_SINGLE_PLATE.reference_cppipe_urls == (
        "https://data.broadinstitute.org/bbbc/BBBC021/analysis.cppipe",
        "https://data.broadinstitute.org/bbbc/BBBC021/illum.cppipe",
    )
