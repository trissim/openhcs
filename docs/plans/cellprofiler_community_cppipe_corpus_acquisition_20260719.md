# CellProfiler community cppipe corpus acquisition and provenance

Date: 2026-07-19

Status: acquisition complete; runtime acceptance not started

## Scope and non-claims

This workstream acquired traceable public CellProfiler `.cppipe` material into OpenHCS's existing persistent cache roots. It did not edit the CellProfiler importer, runtime, tests, current comparison manifests, knowledge-base manifest, or native references. It did not run CellProfiler, import a pipeline, regenerate a reference, compile over ZMQ, execute a pipeline, or judge semantic coherence. Therefore no acquired pipeline is claimed to be an OpenHCS-usable example.

The licensed acquisition contains 367 physical `.cppipe` files from 22 revision-pinned source snapshots. Eleven macOS resource-fork files and one explicitly named generated backup are excluded, leaving 355 qualified path records and 275 globally unique SHA-256 content objects. Four official CellProfiler publication archives add nine physical files: two resource forks are excluded and seven otherwise valid pipelines are quarantined because the archives do not state an artifact reuse license. The complete acquired universe is therefore 376 physical files: 355 licensed path records, 7 license-quarantined records, and 14 exclusions.

## Authority inventory

- `benchmark/converter/cppipe_corpus.py` is the current corpus projection. `CPPipeCorpusCase` represents a path and coarse support status, and it derives cases from the official examples cache plus `ComparisonManifest`; it has no source revision, license, citation, artifact checksum, dependency, or staged-acceptance fields.
- `benchmark/datasets/registry.py` and `BenchmarkDatasetDeclaration` are the nominal acquisition declarations. `DatasetSpec`/`DatasetSourceSpec` in `benchmark/contracts/dataset.py` own URLs, git source/ref, sparse paths, validation, reference pipeline URLs, and benchmark cases. They cannot currently represent per-pipeline provenance.
- `benchmark/datasets/acquire.py`, `benchmark/contracts/manifest_acquisition.py`, and `benchmark/datasets/cache.py` own acquisition and persistent cache layout. This work reused `~/.cache/openhcs/benchmark_datasets` and `~/.cache/openhcs/cellprofiler_examples`; no source data was deleted.
- `benchmark/native_refs` is output/reference storage, not a source corpus. It was inspected and not changed.
- `docs/source/development/mcp_knowledge_base_manifest.json` and `KnowledgeBaseService` dynamically project manifest/native-reference corpus material. No static knowledge-base pipeline list was added.

## Cache layout

- Official examples baseline: `/home/ts/.cache/openhcs/cellprofiler_examples`.
- Revision-pinned community source checkouts: `/home/ts/.cache/openhcs/cellprofiler_examples/community_sources/<owner>--<repo>`.
- Dataset and official publication-archive acquisitions: `/home/ts/.cache/openhcs/benchmark_datasets/<dataset-id>/data`.
- The CellProfiler tutorials acquisition is `/home/ts/.cache/openhcs/benchmark_datasets/CellProfiler_tutorials/data`; tracked archives are expanded in place by existing acquisition machinery, which explains untracked extracted tutorial content under that cache checkout.

## Licensed source snapshots

Each source link is canonical. Each revision link is the direct immutable repository snapshot. Except for archive-expanded tutorial rows, an inventory path's direct artifact URL is `https://raw.githubusercontent.com/<owner>/<repo>/<revision>/<relative-path>`. Tutorial paths extracted from a committed ZIP use the immutable revision tree as the direct archive source. License links point to the license at the recorded revision.

| Source | Revision | License | Raw | Excluded | Qualified paths | In-source unique SHA |
|---|---|---:|---:|---:|---:|---:|
| [CellProfiler/examples](https://github.com/CellProfiler/examples) | [4972b59e](https://github.com/CellProfiler/examples/tree/4972b59e670a4ae96c3d453803c92eeff378d054) | [BSD-3](https://github.com/CellProfiler/examples/blob/4972b59e670a4ae96c3d453803c92eeff378d054/LICENSE) | 43 | 0 | 43 | 43 |
| [CellProfiler/tutorials](https://github.com/CellProfiler/tutorials) | [264a8155](https://github.com/CellProfiler/tutorials/tree/264a8155da21a2d468051f78211bed2e580a8934) | [BSD-3](https://github.com/CellProfiler/tutorials/blob/264a8155da21a2d468051f78211bed2e580a8934/LICENSE) | 47 | 11 | 36 | 16 |
| [CellProfiler/CellProfiler](https://github.com/CellProfiler/CellProfiler) | [673225fb](https://github.com/CellProfiler/CellProfiler/tree/673225fb664a5214c24479ce6329256d046220fa) | [BSD-3](https://github.com/CellProfiler/CellProfiler/blob/673225fb664a5214c24479ce6329256d046220fa/LICENSE) | 8 | 0 | 8 | 7 |
| [CellProfiler/core](https://github.com/CellProfiler/core) | [9fbb218b](https://github.com/CellProfiler/core/tree/9fbb218b018196569b413298db51e72de6c6a013) | [BSD-3](https://github.com/CellProfiler/core/blob/9fbb218b018196569b413298db51e72de6c6a013/LICENSE) | 5 | 0 | 5 | 5 |
| [broadinstitute/2022_09_07_New_phenotypic_dye_testing_CDoT_Broad](https://github.com/broadinstitute/2022_09_07_New_phenotypic_dye_testing_CDoT_Broad) | [6389ddcc](https://github.com/broadinstitute/2022_09_07_New_phenotypic_dye_testing_CDoT_Broad/tree/6389ddcc8b4cb3e61a95b3de39b725389489c29b) | [BSD-3](https://github.com/broadinstitute/2022_09_07_New_phenotypic_dye_testing_CDoT_Broad/blob/6389ddcc8b4cb3e61a95b3de39b725389489c29b/LICENSE) | 2 | 0 | 2 | 2 |
| [broadinstitute/NeuroPainting](https://github.com/broadinstitute/NeuroPainting) | [8e50a7c6](https://github.com/broadinstitute/NeuroPainting/tree/8e50a7c63dd7f303aebaa7f003fbd51043872366) | [BSD-3](https://github.com/broadinstitute/NeuroPainting/blob/8e50a7c63dd7f303aebaa7f003fbd51043872366/LICENSE) | 32 | 0 | 32 | 20 |
| [broadinstitute/cell-health](https://github.com/broadinstitute/cell-health) | [30ea5de3](https://github.com/broadinstitute/cell-health/tree/30ea5de393eb9cfc10b575582aa9f0f857b44c59) | [MIT](https://github.com/broadinstitute/cell-health/blob/30ea5de393eb9cfc10b575582aa9f0f857b44c59/LICENSE) | 2 | 0 | 2 | 2 |
| [broadinstitute/nf-pooled-cellpainting](https://github.com/broadinstitute/nf-pooled-cellpainting) | [c91e4004](https://github.com/broadinstitute/nf-pooled-cellpainting/tree/c91e4004865280a9b7085e0746edd51d44b2493f) | [BSD-3; infrastructure portions MIT](https://github.com/broadinstitute/nf-pooled-cellpainting/blob/c91e4004865280a9b7085e0746edd51d44b2493f/LICENSE) | 12 | 0 | 12 | 12 |
| [broadinstitute/pooled-cell-painting-image-processing](https://github.com/broadinstitute/pooled-cell-painting-image-processing) | [8027f225](https://github.com/broadinstitute/pooled-cell-painting-image-processing/tree/8027f2257f9ed169bedf16697555c1c737a21b19) | [BSD-3](https://github.com/broadinstitute/pooled-cell-painting-image-processing/blob/8027f2257f9ed169bedf16697555c1c737a21b19/LICENSE) | 36 | 0 | 36 | 35 |
| [broadinstitute/profiling-resistance-mechanisms](https://github.com/broadinstitute/profiling-resistance-mechanisms) | [a0e32d6c](https://github.com/broadinstitute/profiling-resistance-mechanisms/tree/a0e32d6ca8be2ca34e6e5698791b87beaacd9b79) | [BSD-3](https://github.com/broadinstitute/profiling-resistance-mechanisms/blob/a0e32d6ca8be2ca34e6e5698791b87beaacd9b79/LICENSE) | 25 | 0 | 25 | 24 |
| [broadinstitute/scripts_notebooks_fossa](https://github.com/broadinstitute/scripts_notebooks_fossa) | [ed1ae529](https://github.com/broadinstitute/scripts_notebooks_fossa/tree/ed1ae52907260d899edbb2fd00f4d4aa812ebb8a) | [MIT](https://github.com/broadinstitute/scripts_notebooks_fossa/blob/ed1ae52907260d899edbb2fd00f4d4aa812ebb8a/LICENSE) | 6 | 0 | 6 | 6 |
| [broadinstitute/starrynight](https://github.com/broadinstitute/starrynight) | [1807217b](https://github.com/broadinstitute/starrynight/tree/1807217bc28a5c0335bd0bbb67c48de29c19fee5) | [BSD-3](https://github.com/broadinstitute/starrynight/blob/1807217bc28a5c0335bd0bbb67c48de29c19fee5/LICENSE) | 43 | 0 | 43 | 34 |
| [carpenterlab/2018_Rohban_NatComm](https://github.com/carpenterlab/2018_Rohban_NatComm) | [53430fe9](https://github.com/carpenterlab/2018_Rohban_NatComm/tree/53430fe9f8198801315b1f9a03ba669f3bdb7481) | [BSD-3](https://github.com/carpenterlab/2018_Rohban_NatComm/blob/53430fe9f8198801315b1f9a03ba669f3bdb7481/LICENSE) | 9 | 1 | 8 | 8 |
| [carpenterlab/2019_Doan_PNAS](https://github.com/carpenterlab/2019_Doan_PNAS) | [1901347d](https://github.com/carpenterlab/2019_Doan_PNAS/tree/1901347d90abb12264dc40f4092009ebf8692995) | [BSD-3](https://github.com/carpenterlab/2019_Doan_PNAS/blob/1901347d90abb12264dc40f4092009ebf8692995/LICENSE) | 1 | 0 | 1 | 1 |
| [carpenterlab/2021_Stirling_BMCBioInformatics](https://github.com/carpenterlab/2021_Stirling_BMCBioInformatics) | [40abc2e6](https://github.com/carpenterlab/2021_Stirling_BMCBioInformatics/tree/40abc2e600fd46b74c213999dd25c5245048dc92) | [BSD-3](https://github.com/carpenterlab/2021_Stirling_BMCBioInformatics/blob/40abc2e600fd46b74c213999dd25c5245048dc92/LICENSE) | 5 | 0 | 5 | 5 |
| [cytomining/CytoDataFrame](https://github.com/cytomining/CytoDataFrame) | [16b79232](https://github.com/cytomining/CytoDataFrame/tree/16b7923254ac50729019e9e32824634b4ea78e90) | [BSD-3](https://github.com/cytomining/CytoDataFrame/blob/16b7923254ac50729019e9e32824634b4ea78e90/LICENSE) | 3 | 0 | 3 | 2 |
| [cytomining/CytoTable](https://github.com/cytomining/CytoTable) | [a3a8c0b0](https://github.com/cytomining/CytoTable/tree/a3a8c0b059c6b884f60df9ee2d2a3b5c22f1949e) | [BSD-3](https://github.com/cytomining/CytoTable/blob/a3a8c0b059c6b884f60df9ee2d2a3b5c22f1949e/LICENSE) | 1 | 0 | 1 | 1 |
| [cytomining/coSMicQC](https://github.com/cytomining/coSMicQC) | [ae60310c](https://github.com/cytomining/coSMicQC/tree/ae60310cc2511eb476c0841afd8e629b92f1f389) | [BSD-3](https://github.com/cytomining/coSMicQC/blob/ae60310cc2511eb476c0841afd8e629b92f1f389/LICENSE) | 3 | 0 | 3 | 2 |
| [jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1) | [56845c7d](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1/tree/56845c7d4dc322652952783d91dae0ffef47829f) | [BSD-3 code; CC0 data](https://github.com/jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1/blob/56845c7d4dc322652952783d91dae0ffef47829f/LICENSE_BSD3.md) | 12 | 0 | 12 | 2 |
| [jump-cellpainting/jump-scope](https://github.com/jump-cellpainting/jump-scope) | [9f582198](https://github.com/jump-cellpainting/jump-scope/tree/9f58219829413e34ede073092a26fe53357eb06f) | [BSD-3](https://github.com/jump-cellpainting/jump-scope/blob/9f58219829413e34ede073092a26fe53357eb06f/LICENSE) | 70 | 0 | 70 | 65 |
| [rgomez-AI/3DChromTrans](https://github.com/rgomez-AI/3DChromTrans) | [2749b4a9](https://github.com/rgomez-AI/3DChromTrans/tree/2749b4a9c3ec55e7f7eb2a9f39ae5d25750740b8) | [MIT](https://github.com/rgomez-AI/3DChromTrans/blob/2749b4a9c3ec55e7f7eb2a9f39ae5d25750740b8/LICENSE) | 1 | 0 | 1 | 1 |
| [rgomez-AI/CellOrientation](https://github.com/rgomez-AI/CellOrientation) | [39f4c9df](https://github.com/rgomez-AI/CellOrientation/tree/39f4c9df1b194a271ca3011d6bce6d2633ad5b07) | [MIT](https://github.com/rgomez-AI/CellOrientation/blob/39f4c9df1b194a271ca3011d6bce6d2633ad5b07/LICENSE) | 1 | 0 | 1 | 1 |
| **Total** | 22 snapshots | BSD-3: 345 qualified paths; MIT: 10 | **367** | **12** | **355** | **294 before global dedupe** |

## Citations and dependencies

- CellProfiler official sources and the CellProfiler 4 supplement: Stirling et al., *CellProfiler 4: improvements in speed, utility and usability*, BMC Bioinformatics (2021), [doi:10.1186/s12859-021-04344-9](https://doi.org/10.1186/s12859-021-04344-9). Official example and tutorial trees include sibling images; tutorial bonus pipelines additionally depend on ilastik, Cellpose, or plugin modules.
- `2018_Rohban_NatComm`: Rohban et al., *Systematic morphological profiling of human gene and allele function via Cell Painting*, [doi:10.1038/s41467-019-10154-8](https://doi.org/10.1038/s41467-019-10154-8). The repository points to large Cell Painting data now in `s3://cellpainting-gallery/cpg0015-heterogeneity`; the named `_backup.cppipe` is excluded.
- `2019_Doan_PNAS`: Doan et al., *Objective assessment of stored blood quality by deep learning*, [doi:10.1073/pnas.2001227117](https://doi.org/10.1073/pnas.2001227117). External imaging-flow-cytometry data and associated metadata are required.
- `profiling-resistance-mechanisms`: Kelley et al., *High-content microscopy reveals a morphological signature of bortezomib resistance*, [doi:10.7554/eLife.91362](https://doi.org/10.7554/eLife.91362); immutable project archive [doi:10.5281/zenodo.8170152](https://doi.org/10.5281/zenodo.8170152). Raw Cell Painting images are external.
- `cell-health`: Way et al., *Predicting cell health phenotypes using image-based morphology profiling*, [doi:10.1091/mbc.E20-12-0784](https://doi.org/10.1091/mbc.E20-12-0784); images [doi:10.17867/10000153](https://doi.org/10.17867/10000153); profiles [doi:10.35092/yhjc.9995672.v5](https://doi.org/10.35092/yhjc.9995672.v5). Raw images are not in the source checkout.
- `CPJUMP1`: Chandrasekaran et al., *Three million images and morphological profiles of cells treated with matched chemical and genetic perturbations*, [doi:10.1038/s41592-024-02241-6](https://doi.org/10.1038/s41592-024-02241-6). Images and load-data metadata are external under `s3://cellpainting-gallery/cpg0000-jump-pilot`; the six experiment copies collapse to two content objects.
- `jump-scope`: Tromans-Coia and Jamali et al., preprint [doi:10.1101/2023.02.15.528711](https://doi.org/10.1101/2023.02.15.528711), plus the Cell Painting Gallery citation [doi:10.1038/s41592-024-02399-z](https://doi.org/10.1038/s41592-024-02399-z). Images/profiles are external under `cpg0002-jump-scope` and are multi-terabyte resources.
- `CellOrientation`: Colomer-Molera et al., *eIF2A regulates cell migration in a translation-independent manner*, [doi:10.1126/sciadv.adu5668](https://doi.org/10.1126/sciadv.adu5668); raw microscopy [doi:10.5281/zenodo.13773035](https://doi.org/10.5281/zenodo.13773035). The existing acquisition also fetched the declared sample archive; the wider workflow uses Snakemake and R.
- `3DChromTrans`: the repository requests citation of the project URL and names no associated workflow paper. Its declared sample archive is acquired. The wider workflow additionally requires Snakemake, Fiji/ImageJ macros, Cellpose, and R.
- `NeuroPainting` includes archived/failed experimental pipelines alongside later workflows and external S3 profiles. `pooled-cell-painting-image-processing`, `starrynight`, `nf-pooled-cellpainting`, and the CDoT template require experiment-specific images, LoadData metadata, illumination files, and/or cloud workflow assets. Their repositories do not identify one pipeline-specific paper, so no DOI is invented.
- `scripts_notebooks_fossa` includes course/example pipelines and a RunCellpose workflow that needs CellProfiler plugins, a Cellpose model/runtime, input images, and generated LoadData CSVs.
- `CytoDataFrame`, `CytoTable`, `coSMicQC`, and the CellProfiler main/core rows are primarily parser/test fixtures. They are useful discovery inputs, but they need semantic review before they can be presented as scientific examples.

## Official publication archives held in license quarantine

The canonical index is [CellProfiler Published Pipelines](https://cellprofiler.org/published-pipelines). The direct ZIP URLs below are the existing `BenchmarkDatasetDeclaration.urls` authority. The page supplies publication/version context, and SHA-256 records the acquired snapshot, but neither the page nor these ZIPs supplies a pipeline-artifact license. These seven records fail the provenance/license gate and must not enter runtime or knowledge-base projections unless reuse status is clarified.

| Dataset | Direct artifact | CellProfiler version | Publication | Paths |
|---|---|---:|---|---:|
| `Singh_2014_illumination_correction` | [JMicroscopy_Singh_2014.zip](https://cellprofiler-published-pipelines.s3.amazonaws.com/JMicroscopy_Singh_2014.zip) | 2.1.0 | [doi:10.1111/jmi.12178](https://doi.org/10.1111/jmi.12178) | 3 |
| `Sanz_2019_histology` | [Sanz_JAP_2019.zip](https://cellprofiler-published-pipelines.s3.amazonaws.com/Sanz_JAP_2019.zip) | 3.1.5 and 3.1.9 in filenames | [doi:10.1152/japplphysiol.00257.2019](https://doi.org/10.1152/japplphysiol.00257.2019) | 2 plus 2 resource forks excluded |
| `Tian_2019_neurons` | [Tian_Neuron_2019.zip](https://cellprofiler-published-pipelines.s3.amazonaws.com/Tian_Neuron_2019.zip) | not stated on index | [doi:10.1016/j.neuron.2019.07.014](https://doi.org/10.1016/j.neuron.2019.07.014) | 1 |
| `Sokolov_2023_neurons` | [AM Sokolov archive](https://cellprofiler-published-pipelines.s3.amazonaws.com/AM+Sokolov+Cell+Morphology+pipeline.zip) | 4.2.1 | [doi:10.1523/ENEURO.0160-23.2023](https://doi.org/10.1523/ENEURO.0160-23.2023) | 1 |

Only the Sanz archive contains sample images (8 image files in the acquired snapshot). The other three acquired publication archives do not contain runnable source images. No native CellProfiler run was attempted.

## Reproducible acquisition

Existing dataset authority was used for the publication archives and the two repositories with declared companion data archives:

```bash
cd /home/ts/code/projects/openhcs
.venv/bin/python scripts/prepare_cellprofiler_benchmark_datasets.py acquire \
  --dataset-id Singh_2014_illumination_correction \
  --dataset-id Sanz_2019_histology \
  --dataset-id Tian_2019_neurons \
  --dataset-id Sokolov_2023_neurons \
  --no-only-with-cases

.venv/bin/python scripts/prepare_cellprofiler_benchmark_datasets.py acquire \
  --dataset-id CellOrientation_wound_healing \
  --dataset-id ChromTrans_3d_fish \
  --no-only-with-cases
```

The two `rgomez-AI` declarations currently set `tls_verify=False` for their companion archives. That is existing registry behavior and remains a provenance/security caveat.

Revision-pinned source-only checkouts used this cache-preserving sparse-clone template, with `<revision>` taken from the source table:

```bash
repo=owner/name
revision=40_hex_commit
dest=/home/ts/.cache/openhcs/cellprofiler_examples/community_sources/${repo//\//--}
git clone --depth 1 --filter=blob:none --no-checkout \
  "https://github.com/$repo.git" "$dest"
git -C "$dest" sparse-checkout init --no-cone
git -C "$dest" sparse-checkout set --no-cone \
  '/*' '!/*/' '/**/*.cppipe' '/**/LICENSE*' '/**/COPYING*' \
  '/**/README*' '/**/CITATION*' '/**/*.bib'
git -C "$dest" checkout --detach "$revision"
```

All source trees remain intact. Deduplication is logical by SHA-256; files were not removed from source checkouts. The qualified inventory below groups every source-relative alias under one global content hash.

## Provenance schema gap and exact owner extension

No provenance manifest was added because the current manifest/corpus schema cannot represent the required fields without becoming a parallel semantic authority.

The extension should be made at the existing acquisition owner:

1. Add a frozen `PipelineArtifactSpec` value object in `benchmark/contracts/dataset.py` with `relative_path`, `sha256`, `canonical_source_url`, `direct_artifact_url`, `source_revision`, `license_spdx`, `license_url`, optional publication title/DOI, acquisition status, and typed external-resource dependencies.
2. Add `pipeline_artifacts: tuple[PipelineArtifactSpec, ...]` to `DatasetSpec` and the corresponding class variable to `BenchmarkDatasetDeclaration`; materialize it in `to_spec()`. For source-only corpora, add a registered dataset declaration instead of another registry.
3. Let acquisition validate declared hashes and resolved git revisions. Do not put runtime support judgments in acquisition provenance.
4. Extend `CPPipeCorpusCase` only as a derived acceptance projection with explicit gate results for import, pycodify/reload, ZMQ compile, execution, semantic coherence, and knowledge-base eligibility. It must derive identity/provenance from `DatasetSpec.pipeline_artifacts` and must not mirror source metadata.
5. Keep `ComparisonManifest` as the run-ready case authority. `KnowledgeBaseService` should continue dynamic projection and include only artifacts whose final knowledge-base gate passes; no hardcoded pipeline list is needed.

## Acceptance funnel

| Gate | Current state | Promotion rule |
|---|---|---|
| discover | complete | path is traceable to a canonical source snapshot or official archive |
| provenance/license | passed for 355 path records; failed/pending for 7 quarantined records | explicit source license covers artifact and revision/archive identity is recorded |
| import | not run | OpenHCS imports without unsupported silent loss; diagnostics recorded |
| pycodify/reload | not run | generated Python reloads and preserves pipeline structure/settings |
| compile over ZMQ | not run | compiler resolves modules/artifacts against the execution server |
| execute where source data exists | not run | execution completes against acquired or declared source data |
| semantic coherence review | not run | outputs and workflow intent are scientifically coherent, not merely non-crashing |
| knowledge-base inclusion | not run | all applicable prior gates pass and the dynamic corpus projection exposes the example |

Import and later gates should run only after official30 parity is restored. Sources marked as parser/test fixtures, archived/failed experiments, or missing practical input data may pass early structural gates but still fail execution or semantic-coherence review.

## Unresolved sources

Current primary-source searches found additional `.cppipe` files that were not acquired as candidates because no explicit repository/artifact license was present at the inspected revision:

- [CellProfiler/CellProfiler-plugins at fd82c0a0](https://github.com/CellProfiler/CellProfiler-plugins/tree/fd82c0a0bfc9a54eca766a7bb44efda030a398b3): 3 paths.
- [carpenterlab/2018_McQuin_PLOSBio at 86b05872](https://github.com/carpenterlab/2018_McQuin_PLOSBio/tree/86b05872514216dcc3890cbd7a5fba14cccbd4a2), [2019_Caicedo_CytometryA at b33debf5](https://github.com/carpenterlab/2019_Caicedo_CytometryA/tree/b33debf5503cdf0926072c740eec4d4fc626c156), and [2019_caicedo_dsb at 7ce8dd12](https://github.com/carpenterlab/2019_caicedo_dsb/tree/7ce8dd12be9d5ca3fe719a3a2e62bba42b63e3bb): 11, 4, and 9 paths.
- [broadinstitute/imaging-platform-pipelines at 7b107db5](https://github.com/broadinstitute/imaging-platform-pipelines/tree/7b107db5b54cbe3f0cce4e6488dc24e374d235a0): 119 paths.
- [broadinstitute/nf-pooled-cellpainting-assets at 0828a2ac](https://github.com/broadinstitute/nf-pooled-cellpainting-assets/tree/0828a2acea34ec5d173f2e192da60ff7425b7351): 15 paths.
- Broad project repositories [2016_10_19_InSitu_Expression_Hacohen_Lab_Paul_Hoover_Partners at 4eef7de2](https://github.com/broadinstitute/2016_10_19_InSitu_Expression_Hacohen_Lab_Paul_Hoover_Partners/tree/4eef7de2a16689c12a7adefc01dde03796da6994), [Dymecki-neurons at 77e8ade9](https://github.com/broadinstitute/Dymecki-neurons/tree/77e8ade934c2d6f4fffc319dbf738e9041c121e5), [LeLiu_Projects at cc608637](https://github.com/broadinstitute/LeLiu_Projects/tree/cc608637e91f6ea2786aab1f571d9f2657e4af6c), and [PaulaLlanos_Projects at f6ec6696](https://github.com/broadinstitute/PaulaLlanos_Projects/tree/f6ec6696917ed665cc5ac53d72cafb1a6113cf96): 3, 1, 1, and 13 paths.
- The official Published Pipelines index has many more downloadable archives, but it does not state an artifact license and its S3 URLs are mutable. The four archives already declared by OpenHCS were retained in quarantine; mass acquisition of the remaining unlicensed archives was intentionally not performed.

Forks and search-result mirrors were not acquired. A source can be reconsidered after the owner adds an explicit license or gives artifact-specific reuse permission, at which point the exact revision and SHA-256 inventory must be recorded.

## Qualified SHA-256 inventory

The 355 qualified source-relative paths below collapse to 275 global content objects. A line with multiple paths is an exact-content duplicate retained only as a provenance alias.

- `000f8242281adabcb246545efc8e1eb6a1e7a060a04a8fd29f6db6318a582c30` - `broadinstitute/nf-pooled-cellpainting:assets/cpg0032_test_cppipes/1_CP_Illum.cppipe`
- `00d2a7539ad29c86ed9efc433376096aab9d5cb39380d1e6f3677635a9903468` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleYeastPatches.cppipe`
- `01551775cb1c8b603e9d74e6e97485ed214e34f724832e39ea669f531e4e5b7e` - `jump-cellpainting/jump-scope:pipelines/2020_11_24_Scope1_Nikon/Analysis_10X_Nikon_without_batchfile.cppipe`
- `027a3a8d6914d8e3624f0c6199aa2207a76090d6f3a4eda953a7169ba3436ec3` - `jump-cellpainting/jump-scope:pipelines/2020_11_24_Scope1_Nikon/Erin_tifsaver.cppipe`
- `03d13a73e909525603254d9ab579b38dea1dc9999961cd1d8fe1e739405ba29c` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch8.cppipe`
- `03d30f3ad99cd596c541e95c076801a213dee6cca6c3fd87713322f65316243a` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/3_CP_SegmentationCheck/3_CP_SegmentationCheck_Plate1_Plate2.cppipe`
- `0498a58a3abebc535ba4d9ad7179dd9ae94ec88abc30318bb2873b19cadc5593` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Analysis_Scope1_Yokogawa_US_20X_6Ch_BRO01177034-20x_164856.cppipe`
- `0623bf444db44b06d1b43d813ab3e00831c4311e0b1f02537a8220bcdfd47d73` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/6_BC_Apply_Illum_DebrisMask.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/6_BC_Apply_Illum_DebrisMask.cppipe`
- `075dce2b5877bd0eee3d5202c5aea87c50bf94e456a81289bc5951d486f30c89` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/6B_Apply_Illum_forBarcoding_TI2_AlignToA.cppipe`
- `07728b9540e6a26a20c53e0cb20b30581b49ffb0eb3246b4d75b0f0f91a59bc4` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/Analysis_Bin1_without_batchfile407.cppipe`; `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/Analysis_PE413_Bin1_without_batchfile.cppipe`
- `07c3851b7f48ae5570ff3061df3cbebc0c2a8503668c85b1d1eabf656349bdd7` - `carpenterlab/2021_Stirling_BMCBioInformatics:CombineObjects/CombineObjectsDemo.cppipe`
- `094c4576bdc7f9ce4432b57f93d747da5f89def0159fe6c7228dc39fa371e3ab` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/pipelines_CP4/12cycles/9_Analysis.cppipe`
- `0985b7fac6b513a4bb183385723a261e946d4ffe799b4a1507dc09b764ab7428` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Scope1_Yokogawa_US_AssayDev_20X_6Ch_BRO0117033_151718.cppipe`
- `09fd1f5ff4e2fe93a2ed967eb0154e724fb5ae08820b4dd96b2440ce14111fc8` - `carpenterlab/2019_Doan_PNAS:CellProfiler_Feature_extraction/Scripts/STEP2_ObjectDetection_FeatureExtraction_CellProfiler.cppipe`
- `0a97a4850352d17973f7cc3ac4efa4b25ac861d6950bb29b5cf1b9b5816f7d97` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Scope1_Yokogawa_US_AssayDev_BRO0117014_10x_20201026_113115.cppipe`
- `0b200410a93ca4d5e40bfdf679e3074961e855d5ef418872369f1b686fe80724` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/1_Illum_forCP_TI2.cppipe`
- `0b43b6d087aab12290e63982e78dfc295b54bdcbfe61d87e11e4becf96806cdb` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/9_Analysis/9_Analysis_Rerun.cppipe`
- `0d5d19668b09ca52e31c5d350de1ed21256b20dcc39295308fe507e5244fbf5a` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/assaydev.cppipe`
- `1040a337fb1999211bafae5047fda6b0e317cada8adace59fa163987cc6e3b95` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch5.cppipe`
- `106b82e6350e12f3d77a9b000cff2e6d7a1f718283d2d8679bd4272603afed10` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/7_BC_Preprocess/7_BC_Preprocess_2strict.cppipe`
- `11d03e78bdee9e80bf05e1a233ae16aa224a399855fd9d608a325127a09ea051` - `CellProfiler/tutorials:PixelBasedClassification/Archive_PT/pixel_based_classification_cho.cppipe`; `CellProfiler/tutorials:PixelBasedClassification/pixel_based_classification_cho.cppipe`
- `11e767abe88de00310ab0fb09dc0799e630d89eaf0865bef512d21b604c61d79` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_10X_assayDev_Plate3_CresetZ.cppipe`
- `12782c79bcd9e9a35075e40bbd2335417ff8a83f2dbd3fbd49be391e8c4c7bbb` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleCometAssay.cppipe`
- `12fc66d146716adb3c2a4dcaaa14609288b9184216bb54372b6a2f16d75bdbbe` - `CellProfiler/examples:ExampleIlluminationCorrection/ExampleIlluminationCorrection_Example3.cppipe`
- `14e469a2b9e21173d697f4a85362ee7e8c246bfc22435ce665ac70c2913854ad` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch22.cppipe`
- `15819d4eb4456c91381eb2c75eaa17ac7c320907c85225b28d0a31b559f9a4dc` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleFlyURL.cppipe`
- `1581ef56788e38d5e3c6944971eb60a96fc9aa413c8507290d518c81aa31fa9b` - `CellProfiler/tutorials:BeginnerSegmentation/bonus_1_import_masks.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/bonus_materials/bonus_1_import_masks.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/bonus_materials_PT/bonus_1_import_masks.cppipe`
- `16b3e94474a96baae09e85649662f9a5f64299a4a77a6cff43dc01c8dce38899` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleUntangleWormsBrightField.cppipe`
- `17368891f8f8ed633f0a850e17bb1a93c76adaf08b21379813e385e265726a79` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleImagingFlowCytometryObjectsInGrid.cppipe`
- `178ffb0df262abcbed25898fda5faf448fe4a18e83e1c5e44a5b8f2703b9d4f6` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/9cycles/9_Analysis.cppipe`
- `1809456cb77d8f0c2ff75c26f41374a77ed5d026a1056cc0a64f5d1f53d6075f` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/maxproj.cppipe`
- `1ac3ae0877cd1886a59a9dfca09b3705301d9b8e3ac7c46a0fafa141e09dd57e` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch26.cppipe`
- `1b60d5f204540e0b11a538e0e62fd2b01ff436fdebfa83d1c3783aae150bcebd` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/9_Analysis/9_Analysis.cppipe`
- `1c673596cd56c0dc4bc50b982e5ddb85602b10a9c90f8cec8f2c66b989a1c956` - `CellProfiler/tutorials:QualityControl/BBBC022_QC.cppipe`
- `1cb536af1dc5b5d9c09093395ea38c87d59b798098e7c560181473d3e247c8e2` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_20x/assaydev_orig.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_63x/assaydev.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PROGENITORS_1/assaydev.cppipe`
- `1d97ffb61d5bc51c143bbb300f1f8d79b223f2035dc5df614e0c2843f837914b` - `broadinstitute/scripts_notebooks_fossa:cellprofiler/CellPainting_pipeline/analysis.cppipe`
- `1f0bf795f0b9bbbcc9e069c32e3f9468adf0d155bdc6c1936aba232b3e82831e` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/Scope1PE_assayDev.cppipe`
- `1f3fb3a9b359d8f116757f4f822f6f6b791c11ae9fdd8b9a104b622eb6ae3635` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/2022_03_03_NCP_NEURONS_2_20x/ProjectionPipeline.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/2022_03_03_NCP_NEURONS_2_63x/ProjectionPipeline.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_20x/ProjectionPipeline.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_63x/ProjectionPipeline.cppipe`
- `1f628b88ae53234c2528c5e6773cac52ed97b226b6f6d5c988fd8475a84de9f2` - `CellProfiler/tutorials:BeginnerSegmentation/segmentation_start.cppipe`
- `21e511076af2ca2175451250bc6e2e0524106739105836f45d8d8f6ff6353bd2` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Analysis_Scope1_Yokogawa_US_BRO0117014_10x_20201026_113115.cppipe`
- `223c44c8146a75dbd38290efa5d908829a7ad0f2595f8fbd5da2fe25629a5277` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch14.cppipe`
- `23e0f75d3c0f3a2b8ccf36b50980f504e9a944e3809b98b719be35eef833ea61` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleIlluminationCorrection_Example2.cppipe`
- `2432e103e1a22c940e48f6e9e8bd681b971614684ebd9995ae6c62170d793f29` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/ref_5_BC_Illum.cppipe`; `broadinstitute/starrynight:starrynight/src/starrynight/templates/cppipe/ref_5_BC_Illum.cppipe`
- `244ea8e82a2aa3d65c5760959c67e4546ede8a1530e2103dd9179fa481cf4c05` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_assatdev_20X.cppipe`
- `2457396d9b106db2a6127f8e7a84b697e910d2c25ce6b4d563ae190ff9d70706` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/7_BC_Preprocess/7_BC_Preprocess.cppipe`
- `24664af4866e625387388e218ba6b70cadfa21a858ebd71f78b11dede27faced` - `CellProfiler/CellProfiler:src/frontend/cellprofiler/data/examples/ExampleFly/ExampleFly.cppipe`; `CellProfiler/CellProfiler:tests/core/data/pipeline/v5_ExampleFly.cppipe`; `CellProfiler/core:tests/data/pipeline/v5_ExampleFly.cppipe`
- `251e627ab09f64a78cba30fe6c9a9ce6dffd3bde9955ac627459babcb8af94ec` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/9cycles/6_BC_Apply_Illum.cppipe`
- `25a1ed77969264d49df7477efbbf4a550f795638153251432b3dee188015b86d` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/2_CP_Apply_Illum/2_CP_Apply_Illum_Plate3_Plate4.cppipe`
- `26fe0cc1d732762a13e7c5dafbe3c290974f55414b3608e8f31e04ecc1ec4e2d` - `broadinstitute/cell-health:0.download-data/cellprofiler_pipelines/illum.cppipe`
- `278086eeeb45c86d553d64071c0b20cd34d1066954f5d005443c7a47a930ce8b` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleNeighbors.cppipe`
- `29de79937ec4839480b54090d8c031137356ab63451e1a7adf5c8c0ded54523f` - `CellProfiler/examples:CellProfiler3Pipelines/ExamplePercentPositive.cppipe`
- `2a42c3f3213e026f810f7cc3b953b0664a4a6ee56c414e35862eef0665cc2f09` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/TA-ORF/qc.cppipe`
- `2b32717ac09a722c0b3efd95ad1a023f32b7ac0afce7d951ff513fdac120a8be` - `CellProfiler/examples:ExampleCometAssay/ExampleCometAssay.cppipe`
- `2bc085810de51a209d3deb41fa9853e56e518dca27e305ba195f3878eace3e0e` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/ref_9_Analysis.cppipe`; `broadinstitute/starrynight:starrynight/src/starrynight/templates/cppipe/ref_9_Analysis.cppipe`
- `2e9083558d3a6e44c6c146904eba66bf3972ab6e23c022230bdc35d0db32b8fe` - `carpenterlab/2021_Stirling_BMCBioInformatics:CellPainting/analysis_CP4.cppipe`
- `3332dfdaa4b6fba2409b26cfafad93d01aece310a0d1499bbb1d1a217b66a503` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_10X_Analysis413_Plate3_CresetZ_NoMeasureObjSize.cppipe`
- `335d36a4087d1f323b274b47233aabf4ceb0ecb850e65ed61f171fdb67f6ac4d` - `CellProfiler/CellProfiler:tests/frontend/resources/ExampleFlyURL.cppipe`
- `34de88d6fbb42fcf643c998685b01e7cce94f9047a1ea03bc951a48b99d881d7` - `jump-cellpainting/jump-scope:pipelines/2020_11_24_Scope1_Nikon/Analysis_Nikon_20X_without_batchfile.cppipe`
- `35276d417ae5a76867bdd03543a7c40697a5913dbe6ed1118f7bae02c02ade65` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/7A_BarcodePreprocessing_Troubleshooting.cppipe`
- `357687fb4ad27891a45555dca36fbfd53cb14361781e35ace14b56a1c32a3529` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleIlluminationCorrection_Example1_EachMethod.cppipe`
- `35e6d2f00a337bd8ec5d961a0d5dc1db9e1de9353dba7ef50a031e1f58d03b31` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch10.cppipe`; `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch12.cppipe`
- `37a3771df59af22d47ee0b59ee31ea10ed3638c9e854363fd2af3d5de9586218` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_20X_Analysis413_Crest_Plate2.cppipe`
- `37cf30f8dd0dbdda0d278a52c2add1ec95411a2553744c9ad126cff6858e7260` - `CellProfiler/tutorials:BeginnerSegmentation/bonus_2_ilastik.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/bonus_materials/bonus_2_ilastik.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/bonus_materials_PT/bonus_2_ilastik.cppipe`
- `3b684cc8117d91f025fe071f681c52b7205f9e422a2e5f77d7ad85a253dd9912` - `cytomining/coSMicQC:docs/manuscripts/cosmicqc_paper_v1/figure_3/whole_FOV_examples/whole_FOV_outlines.cppipe`
- `3d1335d9aa58a4cefb13fe41214d60a0f3491ab5fa6fe520fbd038ce50565b36` - `CellProfiler/examples:ExampleVitraImages/ExampleVitra.cppipe`
- `3e57712620f58a7f72290cb3d20ebbae39b6817edaa2125de609a121d7e018b0` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleYeastColonies.cppipe`
- `3f16dcd253e2ed8206fdafe7e7bb9ff2babdb131c455ded4b2ecd90a6f84b260` - `carpenterlab/2021_Stirling_BMCBioInformatics:CellPainting/illum_CP3.cppipe`
- `3f388cb655932c773f042cdb91fa0567d8fc85f2d1b22fc6cd1d42ddf3a5e996` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/2022_03_03_NCP_NEURONS_2_20x/analysis_with_branching.cppipe`
- `3fc7ace9921e5591f6455ee04ba4cff3754ea70a9f8035ce5cd8eeffe74bce3f` - `CellProfiler/examples:ExampleHuman/ExampleHuman.cppipe`; `cytomining/CytoTable:tests/data/cellprofiler/ExampleHuman/ExampleHuman.cppipe`
- `410710691a24fe91cd44a5c4e9df14f0b15a5ad5c1438244f86356fb078e7174` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/7_BC_Preprocess/7_BC_Preprocess_2.cppipe`
- `44e16777aba0f7c93837265bb20e7c58f863da176ca7bd93ed8f2d782cbd6f1d` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_20X_Analysis413_Adaptive_CrestZ_NoMeasureObjSizeShape.cppipe`
- `44efab83cd268b85c41daad21080765295f708cd0c7ccf2343e493d50d94455f` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch7.cppipe`
- `46c9874ad5b647522feda76372fa0cc48a8ab86160625a30a01cabaa2e077f6b` - `CellProfiler/CellProfiler:tests/core/data/ExampleSBSImages/ExampleSBSIllumination.cppipe`; `CellProfiler/core:tests/data/ExampleSBSImages/ExampleSBSIllumination.cppipe`
- `47600b80b9606dd4f0b78be7fd91e5f02fa1d1ba140d265097f1f16537f1bf04` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/CDRP/illum_8x_without_batchfile.cppipe`
- `476e604a5c3c1e4f8137f15cf5913dfcb353ad1c58003eae4ae124ed8f89959f` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch15.cppipe`
- `49061499bf93951163befad246c308f4e07ef5290ae2c580b2dd544c24acffb0` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/8cycles/7_BC_Preprocess.cppipe`
- `49a15341cb3387e4784da67b36299e7bd314191f09cf872ad6c8cc158796d170` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/illum_without_batchfile.cppipe`
- `4a72de68211c2ab23a2f5389d05491c3aea462d8ad1cda19c0ff78644f4bd381` - `broadinstitute/cell-health:0.download-data/cellprofiler_pipelines/analysis.cppipe`
- `4aa9b40a0e2974c29c2c82797625bed4768677e54e149fa1842ae6fbd7544eb4` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/8cycles/2_CP_Apply_Illum.cppipe`
- `4b3c8fb7f7cdc061dc8f1362f23b1e88d80d6519c7e120ffc233f75e8916c840` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/1_CP_Illum/1_Illum_Plate1_Plate2.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/ref_1_CP_Illum.cppipe`; `broadinstitute/starrynight:starrynight/src/starrynight/templates/cppipe/ref_1_CP_Illum.cppipe`
- `4ca0818ceb047d93b6d5fa80d38baa635569cefeb66d263e702285eaef924a62` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/6_Apply_Illum_forBarcoding_TI2_fast.cppipe`
- `4d6f80edb2c1ab1da317611c61212ce7e8d2790dca1516057d1d80323cf2b7a5` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/9_Analysis.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/9_Analysis.cppipe`
- `4dfc6c0ce2697395db7b596b2dbd427ffabf11c5042f05ae07e222bbf615059c` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_10X_Analysis413_Plate3_CresetZ.cppipe`
- `4ee5f1895d9fba8b0c435aab0e4993129d18d5ea2544851c226e1cb383104ef6` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleColocalization.cppipe`
- `4f5cc19a08032c00dcdd5ce8ba6c136cae1cf3bb8f7c0eba553e6feeda912d7b` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/8cycles/6_BC_Apply_Illum.cppipe`
- `4fc9be36b60bd02423f024d71b42d408c10ede3b2b6b6ab94c9e437e02ee279d` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/6_BC_Apply_Illum/6_BC_Apply_Illum.cppipe`
- `5061c143e554892932d9a4cfa290cd84fca668c4f84afe8598c0dd8588d68b1d` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_AssayDev_20X_Adaptive_CrestZ.cppipe`
- `516309047c0513bc7ed2f6b5c961347f16eaa40a8fbeb2360130a21fb8103f65` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/7_BC_Preprocess/7_BC_Preprocess_3.cppipe`
- `519dda94f5eca8f4b7fb067ed9dd05eabf3d09b287a33c4dc73c1a23641ccbba` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/7A_BC_Preprocess_Troubleshooting.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/7A_BC_Preprocess_Troubleshooting.cppipe`
- `5301af17e4057f2b44f1ec9c034fb94db243537a5ea3cf56d8050fb5f7b39f10` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/9_Analysis/9_Analysis_Plate1_Plate2.cppipe`
- `537999c15da0035f72975ae930884a578835061119b8cff4c971ac59f6193c15` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/Scope1_YokogawaJapan_413assaydev_20x.cppipe`
- `548d56a4a18d0b22e092cb79e26703c9b154271cab16b345c2dcbfa6fd87d619` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_10X_Analysis413_Plate2_NoMeasureObjSizeShape.cppipe`
- `5497bd4788f94719a519e5c0b5b7be2726a701c43aa7dd51de279de38a92944c` - `CellProfiler/examples:ExamplePercentPositive/ExamplePercentPositive.cppipe`
- `55a49ed06becfce520a086d6cf52c5aea9a9c48e2388e6a733d3360dfa7f6c61` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Scope1_Yokogawa_US_AssayDev_BRO0117059-40x_6Ch_20201013_155609.cppipe`
- `56547847e688f73928b86914b0e2eda00f072230f20b4e59e254c5185e899c2d` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/2022_03_03_NCP_NEURONS_2_63x/assaydev.cppipe`
- `5762d67e743d938239d543059a9e8a4cb3bb570d216be75a0b945c63819894ac` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/QC_without_batchfile.cppipe`
- `583e36992b2446b1dde2b44b8900924c924326a34033353e6696062551681f0a` - `CellProfiler/examples:ExampleTumor/ExampleTumor.cppipe`
- `59fab67b71f61edcadd473fc38f06efecc4b0368d9a9d754c4c4d8ee6f57463d` - `broadinstitute/scripts_notebooks_fossa:cellprofiler/2023_CourseBioImageAnalysis_pipelines/assaydev.cppipe`
- `5c43d60afad9d3b48de5e2661c858f9674005c4c1208ae4a41bb3198e1598a6a` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/6C_Apply_Illum_forBarcoding_TI2_SqRt.cppipe`
- `5f6b507c80846a53869f2889710dae38015b1c0fa8fb40e6b55a2cac3a4622b2` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_20X_Analysis413_Crest_Plate2_NoMeasureObjSizeShape.cppipe`
- `5f99a0bdbad23ff64d41b1d62e5a7feec11d82892f0546de1262b630c7c1986f` - `broadinstitute/nf-pooled-cellpainting:assets/cellprofiler/combined_analysis.cppipe`
- `60f6d117f0d334424aa310550ee5caa109d7718a2cbeb5f6174a3afb6884fff5` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_10X_assayDev.cppipe`
- `61a7fcfa999d4e4354d1ef3fe289a6bed6932dbd25086c0bfda114fa6ba3a184` - `broadinstitute/nf-pooled-cellpainting:assets/cpg0032_test_cppipes/3_CP_SegmentationCheck.cppipe`
- `61da2bdd29daec73813645fef978fe237ecd92229433106d80e9bb0c65a3c8e6` - `CellProfiler/examples:ExampleWoundHealing/ExampleWoundHealing.cppipe`
- `624380b2a89edf4ee62714b82883c022791bd83155b9011adf9c1d8a1a093ce8` - `rgomez-AI/CellOrientation:workflow/scripts/Orientation.cppipe`
- `626f887d57ca7fcb878ad65fc2539247e9ff9308ab4245b4a0a797c1884f38d8` - `CellProfiler/tutorials:BeginnerSegmentation/Archive_EN/segmentation_final.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/Archive_PT/segmentation_final.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/segmentation_final.cppipe`
- `62bb0e8696194a08862955986caa491d288ebeaa027bb259ce85eb0ff901ee30` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_20x/analysis_with_branching.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_63x/analysis_with_branching.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PROGENITORS_1/analysis_with_branching.cppipe`
- `62f17467f61081ec0934c4716381351a1dd690ec1cfa250393e9768b3ce0c2f4` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch16.cppipe`
- `63db43119d44caa19f5a6bc467dfb178606c9963b046c35f6fc6adc6be7a6a18` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/QC_without_batchfile.cppipe`; `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/QC_Pipeline_General.cppipe`; `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/QC_without_batchfile.cppipe`; `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/QC_without_batchfile.cppipe`
- `640815f44b695ac6446c15f5df0b64acee0f5e4f451cf3d538e3e207bd465242` - `CellProfiler/examples:ExampleTrackObjects/ExampleTrackObjects.cppipe`
- `640e5c936dddb1a4f030c78db6b8ff038ff41ff322804439dc0cda914f2a688c` - `CellProfiler/examples:ExampleColocalization/ExampleColocalization.cppipe`
- `648e594739356be4dcce938744495d533725b0022973afbeb7dbb69c6ded4d8c` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/6A_Apply_Illum_forBarcoding_TI2_NormCC.cppipe`
- `67aa9b97a4c156cd04e318e38f98c52fdc7f4cef6e3499c9a4b787fd9871f7e4` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Analysis_Scope1_Yokogawa_US_5Ch_20X_BRO0117056__173956.cppipe`
- `68570eebce9bc359029aa90788acfa257aa9d2680aeaadf0ad2c576fac335d91` - `cytomining/CytoDataFrame:tests/data/CP_tutorial_3D_noise_nuclei_segmentation/3d-nuclei-profiling.cppipe`
- `68a6bf955a374536f5c142ec9d23c146c316dc3d97e571422b7c92821070e85a` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Scope1_Yokogawa_US_AssayDev_20X_6Ch_BRO01177034-20x_164856.cppipe`
- `693f3dcc7783b40320e88aca96aaf12fb939f702255077764b921292b2a52fbc` - `CellProfiler/tutorials:Translocation/Archive_EN/Translocation_start.cppipe`; `CellProfiler/tutorials:Translocation/Archive_ES/Translocation_start.cppipe`; `CellProfiler/tutorials:Translocation/Archive_PT/Translocation_start.cppipe`; `CellProfiler/tutorials:Translocation/Translocation_start.cppipe`
- `694d785acfb7a8283234c7d05655f6bb768daaae88790a94a64ff352b17ebec4` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleTrackObjects.cppipe`
- `6954e574e84247a6409a13d2a16ef5aa8e6e1e23df24179874b84284562f8f8d` - `CellProfiler/CellProfiler:tests/core/data/pipeline/v5_coreOnly.cppipe`; `CellProfiler/core:tests/data/pipeline/v5_coreOnly.cppipe`
- `69dc9a5a70acaa0480895b8ebc55651bcf2a6b7204e15fadecb7a02babe4d6f9` - `broadinstitute/pooled-cell-painting-image-processing:qc/QC_OneCycle.cppipe`
- `69e94ed359eaa5ee95c0326679584bf8a9226a64b52986e4560efd362b3cb7d2` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Analysis_Scope1_Yokogawa_US_5Ch_BRO0117033_20x_20201015_161709.cppipe`
- `6a0380ca356f4e7d13120b37eefb14705fd548a5453bece419db30ca3531ad22` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/2022_03_03_NCP_NEURONS_2_20x/assaydev.cppipe`
- `6c44ec8ed600d26b3a33560d94d15dde0320196d6429db94b42a6fc707f28f56` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_10X_assayDev_Pl3_Creset.cppipe`
- `6de0c46434d4f2845289fd4e4862e833d8ec9501b0926a826723f341b11b8223` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch4.cppipe`
- `6ef373fdaaea17ff0f171005f23ec6a16faf46e045ef42ee03aff2bf323673f8` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_20x/assaydev.cppipe`
- `6f1f0a6c2a26c99ade3a258b8dfe1cbbd741757400f632960f66bded1a7fb11c` - `CellProfiler/examples:ExampleSpeckles/ExampleSpeckles.cppipe`
- `6f62179bccf163793f19fa643b7e7a9c7bdfb8b54c36a1f9ef39a047b6938427` - `CellProfiler/CellProfiler:tests/frontend/resources/pipelineV3.cppipe`
- `6fb1813194bd81b256239d4069f4ed159fec8ee0b88344b9dc78b4e88892bcbf` - `broadinstitute/nf-pooled-cellpainting:assets/cpg0032_test_cppipes/6_BC_Apply_Illum.cppipe`
- `71c4de5bd057dc8fbb0dc29c8d289e409f978a257a504530378dd71f5072cbe9` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/ref_6_BC_Apply_Illum.cppipe`; `broadinstitute/starrynight:starrynight/src/starrynight/templates/cppipe/ref_6_BC_Apply_Illum.cppipe`
- `7239c6f9ba438aad236c8b617274bf64fde3a505a6945087c3a5c6b6f485c9d1` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/7_Barcoding_Preprocessing.cppipe`
- `73b7c139ae47531ba0635eb24caafaed03614d900bf07481e66139dab37ec206` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PILOT_3B/illum_without_batchfile.cppipe`
- `7457ea0cf565ad5339cfc4ee49089a6f7af882537469dc8903c9d7044fcf075a` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/illum_without_batchfile.cppipe`
- `74a2b380dc4f8b41192419b705e0c962037a79852485391e29b746e3c2f3007a` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_20X_Analysis413_Adaptive_CrestZ.cppipe`
- `74e9ed1589b2fdff2d84b3b7185f0b9b2f8904a03f3f3ea00bdacd78f7be9b25` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PROGENITORS_1/analysis_without_batchfile.cppipe`
- `7610e4c73fc9a2d2524b7dfddbdbcf3e02dd121bbecf8992c019cb4024901e34` - `broadinstitute/nf-pooled-cellpainting:assets/cpg0032_test_cppipes/2_CP_Apply_Illum.cppipe`
- `7785615e19ad210ee45c8493f622eb3954782389006aaafaaa575189f79dd0b4` - `broadinstitute/nf-pooled-cellpainting:assets/cpg0032_test_cppipes/7_BC_Preprocess.cppipe`
- `78a31f478dfcf7ae5ad407eac32ad5a573611c2c2f67db996a54c635c7f903d4` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/7_BC_Preprocess.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/7_BC_Preprocess.cppipe`
- `78f691e9aa52c63691c3a6229e8aa1ed719180d06d275f4aa07f7acd4a19de6b` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/3_SegmentationCheck_TI2.cppipe`
- `7914b1cc855d235066cf3e4eb43ca5e54d59fac28ceb9cce094df6b0d8604097` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_10X_Analysis413_PL3_Creset.cppipe`
- `7a8a25acac687ad272b5bba329e5baddfc57394e29efce3ae473d6412b97202e` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_10X_Analysis413_Plate3_CresetZ.cppipe`
- `7be25cf0d372a36f9d23beeea67fe7beecf6d23541185dbfe58c26071e6c938a` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/1_SABER_CP_Illum.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/1_SABER_CP_Illum.cppipe`
- `7bf6cbecec4dbbd79b14e9e21cadb2d8c095abdc5c20a74648d3ce7168cb9086` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_20X_Analysis413_Adaptive_CrestZ.cppipe`
- `7e15b80b6b7ab324504b1d12461b13383ee3e957798f11e5a88b946fec3e0087` - `broadinstitute/nf-pooled-cellpainting:assets/cellprofiler/painting_segcheck.cppipe`
- `805146550f128163e6fc1d5280876d6bf6031e3f7c4755f674fa7b766f2fd745` - `CellProfiler/examples:ExampleIlluminationCorrection/ExampleIlluminationCorrection_Example1_AllMethod.cppipe`
- `8131416d46c1334e58cbb404bd6ea7f248013b3e5efcf5631405c24361be29d6` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch23.cppipe`
- `81fb80ca6e0806ff7aede734b237857ad9d98b156df7a104853f77cb617d9887` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Scope1_Yokogawa_US_AssayDev_20X_5Ch_BRO0117056__173956.cppipe`
- `83280a402b508b5bece629d5586606069dc676f29d1bb2ea6000ca3285071874` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch21.cppipe`
- `83a5b139853aa1f3614da388ffe01a4404d644e11f39f9dc53efa102fa9a0ca2` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleUntangleWorms.cppipe`
- `83ee1c010c074cc3ba611c0e3d9e318780295ef588056e26f8fdb47d24dd75c7` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/illumination_calculation.cppipe`
- `89b1edde42cdbe2a8871d2e64db984918b4acfeccb2b0f577e43a6f72989c178` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch19.cppipe`
- `8aba08f4b440025069b6df8af4aff9f580efff25b4ef023072caf48f32b6f661` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleFly.cppipe`
- `8b52cca11387f5ebebefb04e3b503184f0baec4f1c09d8c765cec9befd39786e` - `broadinstitute/nf-pooled-cellpainting:assets/cellprofiler/barcoding_illumapply.cppipe`
- `8c1bca4c33f3bb2b8aa8b81a70475ac849a11d85fc6d092e80bbe1acce930592` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/3_CP_SegmentationCheck/3_CP_SegmentationCheck_Plate3_Plate4.cppipe`
- `8df02997660e8b9b13652a0f2d47f96c5a79edd651e49d0039be84d89f992db2` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Scope1_Yokogawa_US_AssayDev_20X_6Ch_BRO0117059_20X_140319.cppipe`
- `8e67eb9fd22043f95a1563cc39f99741a081e37dd68be0f3087f3de51beb0eed` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/maxproj.cppipe`
- `8e6b594082717a9e074859c3f4ca172807ee3c74f9816e8a70d4bb38277396f1` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/8cycles/3_CP_SegmentationCheck.cppipe`
- `8e965efe3eaffa18b7c76195b4ccb3319fc78ff2435c0baaf2efeecbe3a98a8e` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PILOT_3B/projection_without_batchfile.cppipe`
- `8ea0b19b584444af0aaf6f2421ec2dd71824a4e6e24c7af80b49e1c7c5c1eb59` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/1_CP_Illum/1_CP_Illum.cppipe`
- `8f94f48ac8eb927a49eb93763772db439cffd5b5d4213025d966ed28ae2a7b6a` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/8cycles/1_CP_Illum.cppipe`
- `90663694d8388719d2a510ce87e0c8cad9730159895d6b6cbf4f5a98877b4c3d` - `CellProfiler/CellProfiler:tests/core/data/ExampleSBSImages/ExampleSBS.cppipe`; `CellProfiler/core:tests/data/ExampleSBSImages/ExampleSBS.cppipe`
- `957008f4f128db499fbe7e8a7916d7d4cc869562f1d566d4898e7c5e8d84e5a5` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/CDRP/segmentation_without_batchfile.cppipe`
- `98c31cb3a3f52fd3d96c8190e5cb5ec8ba5ab9b31ab2a71d81cb09a47d66465d` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_10X_assayDev_Plate2.cppipe`
- `98f895d354f6564c5b921871816182cb69ce8a8054ed3a8acafa31d0028a63cf` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/JUANCHI_PILOT_1/analysis.cppipe`
- `99260b546f01685e0a2a54888b06471404dcaeb935ac3a69ff0ad6ff0b1ebfde` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/Scope1_YokogawaJapan_20x_assaydev407_Bin1.cppipe`
- `99375c5c3b86321bc7b410271c484256265062c17cfe58a60016e62c3d79cc77` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PILOT_3B/analysis_without_batchfile.cppipe`
- `998390feb38232dd4feb5262f0ee92b2dbcac647c1848218cbd57adafe82a1d5` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/ref_7_BC_Preprocess.cppipe`
- `9a428ed4606db6247b7b7d57fc3fdf3d9d85c75b8ede0afc5a6333b956a649fe` - `CellProfiler/tutorials:AdvancedSegmentation/BBBC022_Analysis_Final.cppipe`
- `9ce19d9f4a37f4eab59bccaa5e6f4a88c9ea3c753779925ab30b5b4b73edc78f` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/9_Analysis/9_Analysis_foci.cppipe`
- `9f0de0630d2d07210c4ffc640b0e64c637565ea6850a3746c7f6759744830500` - `CellProfiler/examples:ExampleImagingFlowCytometryObjectsInGrid/ExampleImagingFlowCytometryObjectsInGrid.cppipe`
- `9f2f18524ecc1b881996876164524ae432f032dd114750cba9e3515be8ff2a79` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/2_Apply_Illum_forCP_TI2.cppipe`
- `9fb02f506a640868c22048ef11d3c17ef5abf9b476dcafcaef4b8c669d79c646` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/8cycles/7A_BC_PreprocessTroubleshoot.cppipe`
- `9fcd24a1be8372b5c2e87637a055032c3362b9ad8729e361095f93d8bbe71838` - `CellProfiler/tutorials:3d_monolayer/3d_monolayer_final.cppipe`; `CellProfiler/tutorials:3d_monolayer/EN_3D_Monolayer/3d_monolayer_final.cppipe`; `CellProfiler/tutorials:3d_monolayer/ES_Monocapa_3D/3d_monolayer_final.cppipe`; `CellProfiler/tutorials:3d_monolayer/PT_3D_Monolayer/3d_monolayer_final.cppipe`
- `9fcd5c660fbe2a1d3a2e40716a34abe06461c2c5d215f494538f917bae71013e` - `broadinstitute/2022_09_07_New_phenotypic_dye_testing_CDoT_Broad:pipelines/analysis_batch1.cppipe`
- `a04f321134344018e1f24b7ca8adc0de7ad65631a5b6eefe21ba3f8a8bb26716` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/6_BC_Apply_Illum.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/6_BC_Apply_Illum.cppipe`
- `a08d8ce7d1b49d737b0c3d974b28afd416fe4a1245dba4517d9457d67cf45e43` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/2022_03_03_NCP_NEURONS_2_63x/analysis_with_branching.cppipe`
- `a14fcf885e5ddd5095b75bc1feb983ceb5006a919f4777ccb6dd44d8efaceea8` - `broadinstitute/nf-pooled-cellpainting:assets/cellprofiler/painting_illumapply.cppipe`
- `a1600b567d0212ab29d4e86cc4da52bf278500fc820775ccd0d4ed4761b4e2de` - `CellProfiler/tutorials:Translocation/Archive_EN/Translocation_final.cppipe`; `CellProfiler/tutorials:Translocation/Archive_ES/Translocation_final.cppipe`; `CellProfiler/tutorials:Translocation/Archive_PT/Translocation_final.cppipe`; `CellProfiler/tutorials:Translocation/Translocation_final.cppipe`
- `a64c6bd2a57b4642f3427b3247a501ff9682408fe00bd174cb64e45fd06f7888` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch6.cppipe`
- `a6cd48b2bf61531533daf1282eb6ea2d6e6dc8a2dcbe940bdaf2fd9af2ce2030` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleTumor.cppipe`
- `aa6a1920c8fec957224c2b21b757b4150afe2d69a7a9561bd7541f0267e80a53` - `jump-cellpainting/jump-scope:pipelines/2020_11_24_Scope1_Nikon/illum_without_batchfile.cppipe`
- `aa6f79d131384a4735b27e89b0e8b5ed5cc1b293369ded5f4127cbea1ea750a7` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch24.cppipe`
- `aa81c14143f590371c0b786cc73df638888cc9f2d89ec5f05dd24188b0d9d42f` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/TA-ORF/save_illum_corrected_files_without_batchfile.cppipe`
- `aae9b6c3a42bbfb77ac795f5c7e357052a8a765c23453d41798a762e03e5fd01` - `broadinstitute/scripts_notebooks_fossa:cellprofiler/LiveCellPainting_pipeline/analysis.cppipe`
- `abd648ee4fbb35a6b18b6a381ee5547a03f35b6d5018fe7bd2ad71a0122d53b6` - `CellProfiler/CellProfiler:tests/core/data/pipeline/v3.cppipe`; `CellProfiler/core:tests/data/pipeline/v3.cppipe`
- `ad149c2a1a6978b57c89b0c296a5867412814c13b79e4fc2ce52e6d11838c955` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/TA-ORF/illum.cppipe`
- `ad3a6f3c857e064ca15c2c73d85f6894884e598dde2b2f59d3aab8367d283189` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleWoundHealing.cppipe`
- `ad600550c2f97cefc3b3f4bf54b9e5dab9d5dbe22c279f94f5ece3ec3a349baf` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/2_SABER_CP_Apply_Illum.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/2_SABER_CP_Apply_Illum.cppipe`
- `b04991754b6a571f6c98ecbe80306692644e555bae6a20c364c31b31714ea999` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Analysis_Scope1_Yokogawa_US_BRO0117059-40x_20201013_155609.cppipe`
- `b05a0603da3cfeabc3445fdf7b0455abd9e7ecbbc85e24fb8bf1232376a5ed57` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch27.cppipe`
- `b1081e795cfda04c49f9c98a6418e3c20d51f9371e59d2b38a06147373137936` - `broadinstitute/nf-pooled-cellpainting:assets/cpg0032_test_cppipes/5_BC_Illum.cppipe`; `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/5_BC_Illum.cppipe`; `broadinstitute/pooled-cell-painting-image-processing:pipelines/8cycles/5_BC_Illum.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/5_BC_Illum.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/5_BC_Illum/5_BC_Illum.cppipe`
- `b119263dd46e77c92bd15911d52a39c7b29f763b4a0787bbd463ada215ce61be` - `CellProfiler/examples:ExampleIlluminationCorrection/ExampleIlluminationCorrection_Example2.cppipe`
- `b13c4e5180a1493d53a28b3b6aa738e3ff92f411143e38d7e69172b0e156c474` - `jump-cellpainting/jump-scope:pipelines/2020_11_24_Scope1_Nikon/Scope1_Nikon_AssayDev_413_10x.cppipe`
- `b1961a8dd8a9debbf4a0a50ff110f238b51bce60cd3bf930b7de02d0713fc7b4` - `CellProfiler/examples:ExampleYeastPatches/ExampleYeastPatches.cppipe`
- `b2661d63bffc6ea09cc35fdd4a800491e38b9c3a1c5b96d0d477a42e31dc4bae` - `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_11_04_CPJUMP1/CPJUMP1_analysis_without_batchfile_406.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_11_18_CPJUMP1_TimepointDay1/CPJUMP1_analysis_without_batchfile_406.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_11_19_TimepointDay4/CPJUMP1_analysis_without_batchfile_406.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_12_02_CPJUMP1_2WeeksTimePoint/CPJUMP1_analysis_without_batchfile_406.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_12_07_CPJUMP1_4WeeksTimePoint/CPJUMP1_analysis_without_batchfile_406.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_12_08_CPJUMP1_Bleaching/CPJUMP1_analysis_without_batchfile_406.cppipe`
- `b2ee868954c787a43a70763873f06b34d8910391790d8ed804d32719f7e696a3` - `jump-cellpainting/jump-scope:pipelines/2020_11_24_Scope1_Nikon/Scope1_AssayDev_Nikon20X_413.cppipe`
- `b3db8ced2d5e1c6cce338fe1e9b4e475341f8606c121f2eaa9fbe803fdef13eb` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/Scope1_YokogawaJapan_413assaydev_Bin1_40x.cppipe`
- `b4fb262cbdf6e1711ecfeb3d4091dab402c1c733335a0d7c1673c7d20b3c156e` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_10X_Analysis413_Plate2.cppipe`
- `b4fd02634e5d99d1fd4bb3b6f97b4c130eb90a96d0544b3259d377b3ad1da584` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/illum_without_batchfile_6channel.cppipe`
- `b8bd67daad9413b511b62ea5b431a7448cb7e56ffe2811f05011cd75872f7fb5` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch11.cppipe`
- `b9524823ee94a96e6a8fd839310d0df359c98ae0f56d47f84501d9ce15f0f924` - `broadinstitute/scripts_notebooks_fossa:cellprofiler/2023_CourseBioImageAnalysis_pipelines/analysis.cppipe`
- `b982f7fdc5e7e226896baf7c2271c1e30c55ea981861296dcd8f0ce1c0058693` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/1_CP_Illum.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/1_CP_Illum.cppipe`
- `b9f3f543b7128466b9e3a1bfd3ac80c1ef3b11648e11b561ef91013322218aac` - `CellProfiler/tutorials:AdvancedSegmentation/BBBC022_Analysis_Start.cppipe`
- `ba5fdcad6541572451c1299699180026c954f8f2d86942420b286d85b355764d` - `CellProfiler/examples:ExampleYeastColonies/ExampleYeastColonies.cppipe`
- `bba9973e7c832941aad88606783e80c502491644b4d6767a7d8eb468db57f2d2` - `broadinstitute/scripts_notebooks_fossa:cellprofiler/ToxPath_pipeline/analysis.cppipe`
- `bcaa8a0242f612adf5c69fc51d15f2efd8a396ba733d9f067a67a424d6213387` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/Analysis_without_batchfile.cppipe`
- `bcec079a8e5ec38291f602860a7335ede3ff33001bd0ac481a883c41ed7923a9` - `broadinstitute/nf-pooled-cellpainting:assets/cpg0032_test_cppipes/9_Analysis.cppipe`
- `bcf0316e4e10a291bb6aefbf7ec42e440db947c34156a9298941c66b49ae62d9` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/9cycles/7_BC_Preprocess.cppipe`
- `bd5e3a1ed6c437060d6ee4fb33ec62758782d92770ed9a506894c25ffe7e8ae9` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleSpeckles.cppipe`
- `bf85742bc9cf1d6624e94a64e6c381cde979e331d7baac0ed0ddb42ee8881d81` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Analysis_Scope1_Yokogawa_US_20X_6Ch_BRO0117059_20X_140319.cppipe`
- `bfc9b3790639d7afff9fdc188ccd1fc54c3a7d8c755d976612985c49c566b28c` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/Analysis_20X_17Bin1_413_without_batchfile.cppipe`
- `c43a48da5fad1f932ac38ec61dfac1e51234ed807c41276a2a10cac0061dea6a` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/CDRP/analysis_without_batchfile.cppipe`
- `c456574b71dd4d9e511659a229512717c6fca6d142d0336f8bd0937f3218082d` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch25.cppipe`
- `c97f7b1031d6187d6572be93b2f85adb94765bd2e0e95addab320ea70f9d4c44` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_20X_Analysis413_Crest_Plate2.cppipe`
- `ca13e640fe5981ccff352e0d410e712a74d0f7687e45407d0d802bdb3bed2385` - `CellProfiler/examples:ExampleUntangleWorms/ExampleUntangleWorms.cppipe`
- `ca423ce8e5b96a2c09452ecd27ae75ec3145657ea1d3b5e9fac1989e687dc57c` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/JUANCHI_PILOT_1/illum_without_batchfile.cppipe`
- `ca7e03f45aedc2ee407983ffbb3d20227187a4bf93c1c3ae399d2290fa7a66c3` - `CellProfiler/tutorials:BeginnerSegmentation/bonus_3_cellpose.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/bonus_materials/bonus_3_cellpose.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/bonus_materials_PT/bonus_3_cellpose.cppipe`
- `cb0b9216f19527593ec2dad047ce60ee9c0a6c7b49acfaac4bacb69355296ab0` - `broadinstitute/nf-pooled-cellpainting:assets/cellprofiler/barcoding_preprocess.cppipe`
- `cba05d96f872bf2216935a612a7089908abd127f90114db57a7c240522e7eb26` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/Scope1_YokogawaJapan__assayDev407_Bin1.cppipe`
- `cbb08f5b2904722d6693e079d2eb1c3d3d83f8e10c4325d910fce7e34a65bada` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/illum_without_batchfile.cppipe`; `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/illum_without_batchfile_5channel.cppipe`
- `cbda1739c67e0b1f30267a5e8918826d96221166ac7c4641bf199574df648111` - `rgomez-AI/3DChromTrans:workflow/scripts/3D_Distance_LowResolution.cppipe`
- `cc78e8f79b89a5295e06d31039cd6a292d08cc829eae06acaf1aa0a0c521ce69` - `jump-cellpainting/jump-scope:pipelines/2020_11_24_Scope1_Nikon/QC_without_batchfile.cppipe`
- `ccec056b6acc5fdc2549ae376e33343ebc3458a2927f49f2d865177c5a447f34` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/7_BC_Preprocess/7_BC_Preprocess_4.cppipe`
- `cd26c7a4c8df2a250a03dd3888c19c459aa33bb2166b301c399331a5483414cd` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/JUANCHI_PILOT_1/analysis_without_batchfile.cppipe`
- `cd6b9515713c2e26bb39f98aa90d8bd1c53ea98c44dcec65de840078713890ad` - `broadinstitute/2022_09_07_New_phenotypic_dye_testing_CDoT_Broad:pipelines/analysis_batch5.cppipe`
- `ce9b56b5c1f5acfad923a02d8bcf24e6cf1223c9ee26774e02d8a1f58e86d984` - `CellProfiler/tutorials:3DNoiseNuclei/3DNucleiPipelineComputeConsumingFinal.cppipe`; `CellProfiler/tutorials:3DNoiseNuclei/Archive_PT/3DNucleiPipelineComputeConsumingFinal.cppipe`
- `cf53339c00d9bf1aded1014db0abe4f3e099079efafa48c9c1c23451fdd18428` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/Analysis_Bin1_without_batchfile407_Correct.cppipe`
- `d059c048f62ed0e0f3f1fb95acf9ae65ef6f5d142c7323d50a80744f8c73a892` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/2_CP_Apply_Illum/2_CP_Apply_Illum.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/ref_2_CP_Apply_Illum.cppipe`; `broadinstitute/starrynight:starrynight/src/starrynight/templates/cppipe/ref_2_CP_Apply_Illum.cppipe`
- `d0acd84320bf6c09380d46ca57fe1ac56c8e2c4d34244195fd13dfb172bf1d6b` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/Analysis_40X_Bin1_413_without_batchfile.cppipe`
- `d0b72b38c627aab92f5ac769efd409b79e8a20c80902f7b201fe687dfce7d67a` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/ref_3_CP_SegmentationCheck.cppipe`; `broadinstitute/starrynight:starrynight/src/starrynight/templates/cppipe/ref_3_CP_SegmentationCheck.cppipe`
- `d29f5ecf65450d4645ddb6f1f2efa3f688152ffa8ba48f97ef07529279b8b786` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PROGENITORS_1/branch_analysis_without_batchfile.cppipe`
- `d3a6b9d4b9d7590b1f10a4cf68fffa755aad06fd318da739e0013087d9522949` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PILOT_3/analysis_without_batchfile.cppipe`
- `d54afdfdae18d8107412673ac2e21f76c86da63dbf84c76b81bcbbc475081cf0` - `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_11_04_CPJUMP1/illum_without_batchfile.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_11_18_CPJUMP1_TimepointDay1/illum_without_batchfile.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_11_19_TimepointDay4/illum_without_batchfile.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_12_02_CPJUMP1_2WeeksTimePoint/illum_without_batchfile.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_12_07_CPJUMP1_4WeeksTimePoint/illum_without_batchfile.cppipe`; `jump-cellpainting/2024_Chandrasekaran_NatureMethods_CPJUMP1:pipelines/2020_12_08_CPJUMP1_Bleaching/illum_without_batchfile.cppipe`
- `d82420fc328e2f23cda9b9619e520920b49221bd8ab8b8d167fcfd1b9d58d320` - `broadinstitute/starrynight:starrynight/src/starrynight/templates/cppipe/ref_7_BC_Preprocess.cppipe`
- `dac5e69c81fa1a2f3df7c6c2166b5b4ee623d21a0fa4588c4fdbc6a3f6958175` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/3_CP_SegmentationCheck.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/3_CP_SegmentationCheck.cppipe`
- `db59fa93f00175f9e9b4bbe47aa7f99dbc19b1f256e9b0626485ba2429a80adb` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_10X_Analysis413_Plate2.cppipe`
- `dd38d85a172a5e0bca98e582ea412294604b871d6df655fa50980206e651f4c5` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch18.cppipe`
- `de4225b7ad03db635d9f2d1196dc08888120bd011edfaf14c358ecebe2ddd121` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/Scope1_MolDevices_AssayDev_20X_Plate2_Crest.cppipe`
- `dfb995dd3096a8ded9e11616b8d5956cb82600d46033333b2f7f6dc395056c60` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Scope1_Yokogawa_US_AssayDev_20X_5Ch_BRO0117056_110522.cppipe`
- `e0480c94882de760ef93bb2a6181fb123ed172547cdde459b5ecd2941f15031c` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/12cycles/2_CP_Apply_Illum.cppipe`; `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_pcpip_12cycles/2_CP_Apply_Illum.cppipe`
- `e2e799d385446c2f6c534ca3ad5fe74bc88dfcca79a17d350005b096f51eefdd` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleHuman.cppipe`
- `e3b80dd524555cdde6b59eaea7134fd0b7536322fb1486203b277c328e0ad37b` - `carpenterlab/2021_Stirling_BMCBioInformatics:CellPainting/illum_CP4.cppipe`
- `e3cd5b104e2479f2b67d9c697cd86e011111aac984d6ea81231c02db37f8d0e1` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch13.cppipe`
- `e4202696f3f4e83d91eb7cb3383dd11cccfa11257c83263794a748338373d8ba` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/101919_Neuronal_cellpainting_analysis.cppipe`
- `e44e2d4f4dad91515bb004074284c431188eb0e6f7b2aee432e5bf47eb8f544e` - `CellProfiler/tutorials:BeginnerSegmentation/Archive_ES/segmentation_final.cppipe`
- `e4f12f5376542a6b093527d2cc76fb7512f66c6e72f189d74a06dbf1b806d4bc` - `CellProfiler/examples:ExampleFly/ExampleFly.cppipe`
- `e6c59a4ba0d3bd98e64d64af13dbe058a7bad820048798e603f47c15c14ceaec` - `CellProfiler/examples:ExampleUntangleWormsBrightField/ExampleUntangleWormsBrightField.cppipe`
- `e735fb53cdffd6388bb67d9f2d45cec4b974050bacf7f647c2da06e5fefd767e` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch20.cppipe`
- `e7999998f357f81bac6525eceaa2582a1789bb8f25f2321bed52c0c1e8b0352f` - `broadinstitute/pooled-cell-painting-image-processing:pipelines/9cycles/7A_BC_PreprocessTroubleshoot.cppipe`
- `e87afa9d55b3baa66ae6df8104013beb6fc532db814f2f84cd146c24bc166774` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleVitra.cppipe`
- `e88c4e6bf304f674a0e8cfaaa12d6c2e64122a68e1323326d667b5017d59b907` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/6_Apply_Illum_forBarcoding_TI2_slow.cppipe`
- `e95e55fa74b685a882ac625884d2e7d67a8d99b5b05fbaeb184678c15b58000b` - `CellProfiler/examples:ExampleStraightenWorms/ExampleUntangleAndStraightenWorms.cppipe`
- `ea10febcb5b67a55912bc196d25308c2fa19984aa167935248d9a57b2cf22996` - `carpenterlab/2021_Stirling_BMCBioInformatics:CellPainting/analysis_CP3.cppipe`
- `ececef1e5d2e401444e3ef177d8fd4c206dc6e121410fabee2432a346a1021a3` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleIlluminationCorrection_Example3.cppipe`
- `ed0ac9ccab4183f1e3370c91f0ef4fefcfb2be40d28184cfae7a37bd3c167e7a` - `broadinstitute/starrynight:docs/developer/legacy/pcpip-pipelines/_refsource/5_BC_Illum/5_BC_Illum_byWell.cppipe`
- `ee0ebab39ff3e621dd66bbac2ff59e8f7d6506010b21897f19a9f151256f8c66` - `CellProfiler/examples:ExampleNeighbors/ExampleNeighbors.cppipe`
- `ee16201f184cd53109dceada3fa19f2cb4e805cc6265399cb95d90899fd00aaa` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch17.cppipe`
- `ee793b00c40f5adc0aebb914b39f8e121af7f1153812886a8d2f9faa5a9d28e4` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_PE/Scope1PE_assayDev407_Bin1.cppipe`
- `ef8fc7c4ef5d01602c60ea5b64686532dbfb8a008e194e8f9a70198b00dfe5dc` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_10X_Analysis413_PL3_Creset_NoMeasureSizeShape.cppipe`
- `f07e9030e413c399782f0e348f4506c0608f94fae1a9c0a4083b0493122eba91` - `CellProfiler/tutorials:BeginnerSegmentation/Archive_EN/segmentation_start.cppipe`; `CellProfiler/tutorials:BeginnerSegmentation/Archive_PT/segmentation_start.cppipe`
- `f0e669f5bf3019a0cdd11f167c87f5d1efd2688db9edfc7a7aaf077911ac081e` - `broadinstitute/NeuroPainting:1.run-workflows/pipeline/2022_03_03_NCP_NEURONS_2_20x/illum_without_batchfile.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/2022_03_03_NCP_NEURONS_2_63x/illum_without_batchfile.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_20x/illum_without_batchfile.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_NEURONS_1_63x/illum_without_batchfile.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PILOT_3/illum_without_batchfile.cppipe`; `broadinstitute/NeuroPainting:1.run-workflows/pipeline/NCP_PROGENITORS_1/illum_without_batchfile.cppipe`
- `f30b65aaf27550d61709719a05cff6c5d9997769eb02e3620a527e0dd0e2ecbe` - `broadinstitute/profiling-resistance-mechanisms:0.generate-profiles/pipelines/analysis_batch9.cppipe`
- `f325d8913ce46d885e44a509e0c52519209ef0e1279f24d8c0dec585152b8a85` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Analysis_Scope1_Yokogawa_US_5Ch_20X_BRO0117056_110522.cppipe`
- `f7037b5c902cf9ee70b107db5097deca56c722e865e331ba0f05b097e5fae308` - `jump-cellpainting/jump-scope:pipelines/2020_10_27_Scope1_YokogawaJapan/Scope1_YokogawaJapan__assaydev407_Bin1_40x.cppipe`
- `f70ebe867ce4017715059f0bec525df13695db5585ddd8d80ed8bc36d293858d` - `jump-cellpainting/jump-scope:pipelines/2020_11_06_Scope1_MolDev/previousAnalysis/Scope1_MolDevices_10X_Analysis413_PL3_Creset.cppipe`
- `f7ec453d0b24a64a5bd38c9fcc6203619009412a8d6f160e6f09309ff3e51048` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/TA-ORF/analysis.cppipe`
- `f7f6c9762e70d5f30f61171e23aa906ca5d2cba528484794ab09b9c63bca75d8` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleIlluminationCorrection_Example1_AllMethod.cppipe`
- `f89905419591d34b2e454b33037b1d18e78d7d09b17d6c742abead9550d3655f` - `CellProfiler/examples:ExampleIlluminationCorrection/ExampleIlluminationCorrection_Example1_EachMethod.cppipe`
- `f961cdf32d43de2dc8185f1047b14da9924bc8e7d4f399c3dfd83b7cb48f97a1` - `broadinstitute/pooled-cell-painting-image-processing:python2/pipelines/12cycles/5_Illum_forBarcoding_TI2.cppipe`
- `f9925726af63131c2ddd5b33c4a44497af8ddbab27727a9043a62c3d82debb79` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/TA-ORF/analysis_without_batchfile.cppipe`
- `fa61e0a30dab60e7c7b32283dc109b8f662f17e57c38b033b43c57e1d5cc5470` - `CellProfiler/examples:CellProfiler3Pipelines/ExampleUntangleAndStraightenWorms.cppipe`
- `fce906dd738e78637354a180bd9c6f9a6466cc2cc31d5609a524e105953d7728` - `CellProfiler/tutorials:BeginnerSegmentation/Archive_ES/segmentation_start.cppipe`
- `fd7cfc55c9cb784c34c23af1690af331d906157b67c373e6c2d3ef117eeb8548` - `cytomining/CytoDataFrame:tests/data/cytotable/NF1_cellpainting_data/NF1_plate2_export_masks.cppipe`; `cytomining/CytoDataFrame:tests/data/cytotable/NF1_cellpainting_data_shrunken/NF1_plate2_export_masks.cppipe`; `cytomining/coSMicQC:tests/data/cytotable/NF1_cellpainting_data/NF1_plate2_export_masks.cppipe`; `cytomining/coSMicQC:tests/data/cytotable/NF1_cellpainting_data_shrunken/NF1_plate2_export_masks.cppipe`
- `fe424b1e198b06a40c8d6155a0ed906e52c5737e970b783f77afbd511d962fec` - `broadinstitute/scripts_notebooks_fossa:cellprofiler/mimic_nuclei_pipeline/MimicNuclei.cppipe`
- `fe547bade0f1e40f50e0a9916d64a88a89a05ba84b7b347061846cbc4345fcee` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Analysis_Scope1_Yokogawa_US_20X_6Ch_BRO0117033_151718.cppipe`
- `fec8f41afcd8a2a0a2179524a8c340eef2424cd21a02019c8d4d151df769278d` - `broadinstitute/pooled-cell-painting-image-processing:qc/QC_EightCycle.cppipe`
- `ffa4e452664bf1596a0d3e98ba56cb807242354eb0c83b947a38cb23dd9cbb4f` - `jump-cellpainting/jump-scope:pipelines/2020_11_16_Scope1_YokogawaUS/Scope1_Yokogawa_US_AssayDev_20X_5Ch_BRO0117033_20x_20201015_161709.cppipe`

## License-quarantined SHA-256 inventory

- `ddc0dad50c178150201d0f123208345a1394010d14abf2fd3666d118ca08fa39` - `published/Sanz_2019_histology:Sanz_JAP_2019/Muscle2View pipeline v3.1.5.cppipe` (quarantined_license)
- `bd4b2af9164d0efc213130a5ef83afe6a8ca84204eca09d10a3d367d2d1c295f` - `published/Sanz_2019_histology:Sanz_JAP_2019/Muscle2View pipeline v3.1.9.cppipe` (quarantined_license)
- `cce83598d0c48f9bf4945c6d29d684d81e80a33d9c6c86e840f38ceaf89ddd6a` - `published/Singh_2014_illumination_correction:pipelines/Analysis_posneg_withICFs_Median500M.cppipe` (quarantined_license)
- `33b0a8fd35f421e0dd752c6288fd43c9c8938c560540872d9829408ca92b30f8` - `published/Singh_2014_illumination_correction:pipelines/Analysis_posneg_withoutICFs.cppipe` (quarantined_license)
- `dbe9f4a27c1bfe88ae9fc1070ddf847c969b03a59bc976fcd754b235ae9fa282` - `published/Singh_2014_illumination_correction:pipelines/Illum_Median500M.cppipe` (quarantined_license)
- `ca0ee40ce60ed039b197926d685707c366df600f8b7a57f251a5cafa1e83b88e` - `published/Sokolov_2023_neurons:AM Sokolov Cell Morphology pipeline.cppipe` (quarantined_license)
- `b453803b589c1c0c822f65ee7ab97b43531d75206c4b491e87a44a3c41d3993e` - `published/Tian_2019_neurons:measure_neurites.cppipe` (quarantined_license)

## Excluded physical files

- `af8b64521f7dc3aa44d3e0518c2dd16ba9e979311b1bf9fb06938df1250ac74b` - `carpenterlab/2018_Rohban_NatComm:CP_pipelines/CDRP/illum_8x_STILLWITH_batchfile_backup.cppipe` (generated_backup)
- `3dcc241ba2e662029deef179c35028f6d01ca12a51369ad1335f77a7ec30d636` - `CellProfiler/tutorials:3d_monolayer/__MACOSX/EN_3D_Monolayer/._3d_monolayer_final.cppipe` (resource_fork)
- `3dcc241ba2e662029deef179c35028f6d01ca12a51369ad1335f77a7ec30d636` - `CellProfiler/tutorials:3d_monolayer/__MACOSX/ES_Monocapa_3D/._3d_monolayer_final.cppipe` (resource_fork)
- `3dcc241ba2e662029deef179c35028f6d01ca12a51369ad1335f77a7ec30d636` - `CellProfiler/tutorials:3d_monolayer/__MACOSX/PT_3D_Monolayer/._3d_monolayer_final.cppipe` (resource_fork)
- `c49720f1c0e272ec212b1ef11e10461902d50134ced310a15d80183c5d7ff694` - `CellProfiler/tutorials:BeginnerSegmentation/__MACOSX/._bonus_1_import_masks.cppipe` (resource_fork)
- `7db29df5a12d6905c9b563d0d627d706319cab11e6a3612f047f5edfca1079fe` - `CellProfiler/tutorials:BeginnerSegmentation/__MACOSX/._bonus_2_ilastik.cppipe` (resource_fork)
- `ca7f9f3883989b6fb066d9a1c7499fa4650070a7b68009cefe009439e5239ba7` - `CellProfiler/tutorials:BeginnerSegmentation/__MACOSX/._bonus_3_cellpose.cppipe` (resource_fork)
- `ad06ad895c8c428a64dc6df6adccb1e57b484276f198f3153692a0a381d3b58c` - `CellProfiler/tutorials:BeginnerSegmentation/__MACOSX/Archive_ES/._segmentation_final.cppipe` (resource_fork)
- `ad06ad895c8c428a64dc6df6adccb1e57b484276f198f3153692a0a381d3b58c` - `CellProfiler/tutorials:BeginnerSegmentation/__MACOSX/Archive_ES/._segmentation_start.cppipe` (resource_fork)
- `c49720f1c0e272ec212b1ef11e10461902d50134ced310a15d80183c5d7ff694` - `CellProfiler/tutorials:BeginnerSegmentation/__MACOSX/bonus_materials_PT/._bonus_1_import_masks.cppipe` (resource_fork)
- `7db29df5a12d6905c9b563d0d627d706319cab11e6a3612f047f5edfca1079fe` - `CellProfiler/tutorials:BeginnerSegmentation/__MACOSX/bonus_materials_PT/._bonus_2_ilastik.cppipe` (resource_fork)
- `ca7f9f3883989b6fb066d9a1c7499fa4650070a7b68009cefe009439e5239ba7` - `CellProfiler/tutorials:BeginnerSegmentation/__MACOSX/bonus_materials_PT/._bonus_3_cellpose.cppipe` (resource_fork)
- `bb7a57c1fb41da9b12c2234ae71d1b4ce575bdba16be5ff5708ea70cdd55882b` - `published/Sanz_2019_histology:__MACOSX/Sanz_JAP_2019/._Muscle2View pipeline v3.1.5.cppipe` (resource_fork)
- `bb7a57c1fb41da9b12c2234ae71d1b4ce575bdba16be5ff5708ea70cdd55882b` - `published/Sanz_2019_histology:__MACOSX/Sanz_JAP_2019/._Muscle2View pipeline v3.1.9.cppipe` (resource_fork)

## Verification

- Every licensed git source is detached at the full revision recorded above.
- Root license text was inspected at each source snapshot. `nf-pooled-cellpainting` is BSD-3 with explicitly identified MIT-derived infrastructure; CPJUMP1 separates BSD-3 code and CC0 data.
- SHA-256 was computed from the acquired bytes. Global grouping produced 275 qualified objects from 355 source-relative paths.
- Exclusion rules were path-based only for `__MACOSX`/`._*` resource forks and the explicit `_backup.cppipe`; exact-content duplicates were grouped by checksum rather than deleted.
- No CellProfiler process, importer, pycodifier, compiler, ZMQ execution, or reference generation command was run.
- Repository scope is this report only; cache acquisitions are outside the git worktree.

## Changed paths

- Repository: `docs/plans/cellprofiler_community_cppipe_corpus_acquisition_20260719.md`.
- Persistent cache: 18 revision-pinned community checkouts under `/home/ts/.cache/openhcs/cellprofiler_examples/community_sources/`.
- Existing dataset cache paths acquired or confirmed: `Singh_2014_illumination_correction`, `Sanz_2019_histology`, `Tian_2019_neurons`, `Sokolov_2023_neurons`, `CellOrientation_wound_healing`, and `ChromTrans_3d_fish` under `/home/ts/.cache/openhcs/benchmark_datasets/`.
