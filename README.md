# Wildfire dNBR CNN Audit

This repository contains the code for a burned-area mapping benchmark that
tests dNBR-supervised convolutional models across sensors, regions, and
independent perimeter products.

The benchmark separates two questions that are often mixed together:

1. how well a model reproduces dNBR-derived proxy labels; and
2. whether that model agrees better than dNBR with independent fire products.

The main result is intentionally framed as an audit rather than a claim that a
single CNN is a global replacement for dNBR. The code is designed to make that
distinction reproducible.

## Repository Contents

- `src/` - reusable Python package for data preparation, models, training,
  metrics, sliding-window inference, and explainability checks.
- `scripts/` - command-line workflows for fetching scenes, creating labels,
  extracting patches, training models, validating against independent products,
  and aggregating tables.
- `configs/` - event definitions and dataset configuration.
- `data/manifests/` - lightweight CSV manifests with event IDs, sensor IDs,
  dates, STAC item IDs, and generated relative paths.
- `tests/` - unit and smoke tests, including two embedded patch fixtures.
- `requirements.txt` and `environment.yml` - Python environment definitions.
- `REPRODUCTION.md` - command-by-command guide for rerunning the benchmark.

Large files are intentionally not committed: raw satellite scenes, harmonized
GeoTIFFs, extracted patch arrays, trained checkpoints, validation rasters, and
paper figures belong in the external artifact archive named in the paper.

## Data Manifest Note

The public manifests keep the metadata required to refetch scenes, especially
the STAC item IDs stored in the `notes` column. Raw local cache columns such as
`pre_image_path`, `post_image_path`, and `label_mask_path` are intentionally
blank because those paths are machine-specific. Running
`scripts/fetch_and_populate_manifests.py` recreates local paths for a new
checkout.

Generated harmonized columns use relative paths such as
`data/processed_full/...` so they can be paired with the external artifact
archive when available.

## Quick Start

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the local test suite:

```bash
python -m pytest -q
```

Run only the embedded-fixture smoke test:

```bash
python -m pytest -q tests/test_demo_pipeline.py
```

The smoke test does not download data. It loads the two small fixtures in
`tests/fixtures/`, applies the same normalization path used by the training
dataset, instantiates all model families, and checks the core metrics.

## Full Benchmark

The full workflow is documented in `REPRODUCTION.md`. In summary:

1. fetch Sentinel-2 and Landsat scenes from public STAC sources;
2. generate reflectance-space dNBR labels;
3. harmonize imagery and extract overlapping patches;
4. train the corrected 64-channel model matrix;
5. evaluate Australia and Mediterranean transfer;
6. compare against MTBS, NSW Fire History, Copernicus EMS, MCD64A1, and DEA;
7. run CaBuAr, calibration, boundary-buffer, latency, and explainability
   diagnostics; and
8. aggregate the reported tables and statistics.

## License

See `LICENSE`.
