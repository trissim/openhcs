import sys
import gc
from unittest.mock import patch
from dq_dock_engine.benchmark.benchmark_pdb import run_benchmark
from dq_dock_engine.benchmark.benchmark_pdb import PDBEntry, BenchmarkScope

def mock_casf_entries():
    return [
        PDBEntry(pdb_id="2qwd", target_name="2qwd_target", scope=BenchmarkScope.LIGAND),
        PDBEntry(pdb_id="2qwe", target_name="2qwe_target", scope=BenchmarkScope.LIGAND),
    ]

if __name__ == "__main__":
    with patch("dq_dock_engine.benchmark.benchmark_pdb.get_casf_2007_entries", new=mock_casf_entries):
        run_benchmark(
            n_complexes=2,
            use_rich_exact_rescoring=True,
            dataset="casf2007",
            competitors="none",
        )
