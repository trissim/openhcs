from benchmark.datasets.registry import BBBC021_SINGLE_PLATE, dataset_specs
from openhcs.constants.constants import Microscope


def test_bbbc021_dataset_spec_exposes_reference_cppipe_urls() -> None:
    assert BBBC021_SINGLE_PLATE.reference_cppipe_urls == (
        "https://data.broadinstitute.org/bbbc/BBBC021/analysis.cppipe",
        "https://data.broadinstitute.org/bbbc/BBBC021/illum.cppipe",
    )


def test_dataset_ingestion_types_are_registered_microscope_declarations() -> None:
    registered_types = {
        microscope.value
        for microscope in Microscope
        if microscope is not Microscope.AUTO
    }

    assert {spec.microscope_type for spec in dataset_specs()} <= registered_types
