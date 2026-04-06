---
title: "PolyStore: Unified Storage Abstraction with Streaming Backends for Scientific Python"
tags:
  - Python
  - storage
  - scientific computing
  - microscopy
  - streaming
authors:
  - name: Tristan Simas
    orcid: 0000-0002-6526-3149
    affiliation: 1
affiliations:
  - name: McGill University
    index: 1
date: 15 January 2026
bibliography: paper.bib
---

# Summary

PolyStore provides a unified API for heterogeneous storage backends—disk, memory, Zarr, and live streaming to Napari or Fiji—through a single interface. The key insight: **streaming viewers are just backends**:

```python
from polystore import FileManager, BackendRegistry

fm = FileManager(BackendRegistry())

# Same API for persistent storage, cache, and live visualization
fm.save(image, "result.npy", backend="disk")
fm.save(image, "result.npy", backend="memory")
fm.save(image, "result.npy", backend="napari_stream")  # Appears in Napari
```

The `FileManager` routes operations to explicitly selected backends with no implicit fallback. Backends auto-register via metaclass, support lazy imports for optional dependencies, and provide atomic file operations for concurrent metadata updates.

# Statement of Need

Scientific pipelines move data between arrays, files, chunked formats, and visualization tools. Each destination has different I/O conventions:

```python
# Without PolyStore: per-backend code everywhere
np.save("result.npy", data)                    # Disk
memory_store["result.npy"] = data              # Memory
zarr.save("result.zarr", data)                 # Zarr
socket.send(msgpack.packb({"data": data}))    # Streaming
```

With PolyStore, one call handles all backends. The explicit `backend=` parameter ensures deterministic behavior—no silent fallbacks, no hidden resolution logic.

# State of the Field

| Feature | PolyStore | fsspec | zarr | xarray |
|---------|:---------:|:------:|:----:|:------:|
| Unified storage API | ✓ | ✓ | — | — |
| Streaming backends | ✓ | — | — | — |
| Multi-framework I/O | ✓ | — | — | ✓ |
| Atomic concurrent writes | ✓ | — | — | — |
| Explicit backend selection | ✓ | — | — | — |
| Zero implicit fallback | ✓ | — | — | — |

**fsspec** [@fsspec] provides unified filesystem access but cannot support streaming because its abstraction is *filesystems*—everything must behave like a file. PolyStore's abstraction is *data sinks*, which includes destinations that consume data without persisting it. This distinction is fundamental: a Napari viewer is not a filesystem, but it is a valid data sink.

**zarr** [@zarr] handles chunked arrays but is a single format, not a storage abstraction. **xarray** [@xarray] provides multi-dimensional arrays with NetCDF/Zarr backends but no streaming or explicit backend routing.

# Software Design

**Backend Hierarchy**: The key architectural decision is the base abstraction. `DataSink` (write-only) is the root interface—not `StorageBackend`. This allows streaming backends that consume data without supporting reads:

```python
class StreamingBackend(DataSink):      # Write-only sink
    def save_batch(self, data, paths, **kwargs): ...
    # No load() method - streaming is one-way

class StorageBackend(DataSink):        # Read/write storage
    def save_batch(self, data, paths, **kwargs): ...
    def load_batch(self, paths, **kwargs): ...
```

The `FileManager` routes to any `DataSink`. Pipeline code doesn't know whether data goes to disk, memory, or a live Napari viewer—and doesn't need to.

**Streaming Internals Hidden**: Streaming backends handle substantial complexity internally—GPU tensor conversion, shared memory allocation, ZMQ socket management, ROI serialization—all behind the same `save_batch()` interface. The orchestrator remains backend-agnostic.

**Atomic Operations**: Cross-platform file locking (`fcntl` on Unix, `portalocker` on Windows) with `atomic_update_json()` for concurrent metadata writes from multiple pipeline workers. This is critical for OpenHCS where multiple worker processes write metadata simultaneously—without atomic operations, race conditions corrupt JSON files.

**Lazy Backend Instantiation**: Backends auto-register via `metaclass-registry` [@metaclassregistry] and are lazily instantiated, keeping optional dependencies unloaded until used. For example, the Napari streaming backend only imports `napari` when first used, avoiding dependency bloat for users who don't need visualization.

**Batch Operations**: The `save_batch()` and `load_batch()` interfaces accept lists of paths and data, enabling backends to optimize I/O. The Zarr backend can write multiple arrays in a single transaction; the Napari backend can batch ROI updates into a single viewer refresh. This is more efficient than per-file operations.

# Research Application

PolyStore was developed for OpenHCS (Open High-Content Screening) where microscopy pipelines process thousands of images per experiment. A typical workflow:

1. **Load**: Read raw images from disk (TIFF, OME-TIFF) or virtual workspace (lazy-loaded)
2. **Process**: Apply filters, segmentation, feature extraction in memory
3. **Save**: Write results to Zarr (chunked, compressed for efficient storage)
4. **Stream**: Send intermediate results to Napari for live preview and quality control

All through one interface:

```python
# Load → process → save → stream: same API
images = fm.load_batch(paths, backend="disk")
processed = pipeline(images)
fm.save_batch(processed, paths, backend="zarr")
fm.save_batch(processed, paths, backend="napari_stream")
```

**Concrete Example**: A user processes 10,000 images. Without PolyStore, the pipeline code would contain:
- `np.load()` for disk reads
- `zarr.open_array()` for Zarr writes
- `napari.Viewer.add_image()` for visualization
- Custom socket code for streaming to remote Fiji instances

With PolyStore, all I/O goes through `FileManager`, and the user can switch backends by changing a config parameter—no code changes needed.

**Bug Prevention**: The explicit backend model eliminated an entire class of bugs where code assumed disk storage but ran against memory or streaming backends. For example, a function that called `os.path.exists()` would fail silently against a memory backend. With PolyStore, the backend is explicit, and such mismatches are caught immediately.

# AI Usage Disclosure

Generative AI (Claude) assisted with code generation and documentation. All content was reviewed and tested.

# Acknowledgements

This work was supported in part by the Fournier lab at the Montreal Neurological Institute, McGill University.

# References
