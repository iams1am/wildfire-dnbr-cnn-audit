# Reproduction Guide

Run all commands from the repository root. Paths use forward slashes so the
commands are easy to copy on Linux and macOS; Python also accepts these paths on
Windows. If a Windows shell requires backslashes, only the path separators need
to be changed.

The reported benchmark uses the corrected `reflectance64` protocol:

- reflectance-space dNBR labels for Sentinel-2 and Landsat;
- per-sensor reflectance normalization before model inference;
- 64 base channels for learned models;
- dynamic positive-class weighting in the BCE term plus Dice loss;
- pixel-pooled headline metrics;
- event-level k-fold diagnostics;
- QA-clipped independent validation; and
- mean-reflectance baseline for Integrated Gradients.

Older narrow-width or raw-DN label runs are historical and should not be used
for the final results.

Output directories such as `data/paper_assets/` are created locally by these
commands and are intentionally not tracked by Git.

## 1. Environment

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

For GPU experiments, install a CUDA-compatible PyTorch build before running the
training sweep. On Windows, if multiprocessing workers produce access-denied
errors, add `--num-workers 0`. On Linux workstations, `--num-workers 2` or
higher is usually appropriate.

## 2. Fetch Scenes, Build Labels, and Extract Patches

The checked-in manifests keep STAC scene IDs and event metadata. Raw local path
columns are blank until the fetch step recreates them for the current machine.

```bash
python scripts/fetch_and_populate_manifests.py \
  --sensors sentinel2 landsat8 landsat9 \
  --skip-missing-pairs

python scripts/regenerate_dnbr_labels.py \
  --manifests \
  data/manifests/california_train_val_manifest.csv \
  data/manifests/australia_external_test_manifest.csv

python scripts/preprocess_manifest.py \
  --manifest data/manifests/california_train_val_manifest.csv \
  --output-root data/processed_full \
  --overwrite

python scripts/preprocess_manifest.py \
  --manifest data/manifests/australia_external_test_manifest.csv \
  --output-root data/processed_full \
  --overwrite

python scripts/extract_patches.py \
  --manifest data/manifests/california_train_val_manifest_harmonized.csv \
  --output-root data/patches/california_full

python scripts/extract_patches.py \
  --manifest data/manifests/australia_external_test_manifest_harmonized.csv \
  --output-root data/patches/australia_full
```

QA-off manifests and patches:

```bash
python scripts/create_noqa_manifests.py \
  --input-manifest \
  data/manifests/california_train_val_manifest.csv \
  data/manifests/australia_external_test_manifest.csv \
  --suffix _noqa_full

python scripts/regenerate_dnbr_labels.py \
  --manifests \
  data/manifests/california_train_val_manifest_noqa_full.csv \
  data/manifests/australia_external_test_manifest_noqa_full.csv

python scripts/preprocess_manifest.py \
  --manifest data/manifests/california_train_val_manifest_noqa_full.csv \
  --output-root data/processed_noqa_full \
  --overwrite

python scripts/preprocess_manifest.py \
  --manifest data/manifests/australia_external_test_manifest_noqa_full.csv \
  --output-root data/processed_noqa_full \
  --overwrite

python scripts/extract_patches.py \
  --manifest data/manifests/california_train_val_manifest_noqa_full_harmonized.csv \
  --output-root data/patches_noqa/california_full

python scripts/extract_patches.py \
  --manifest data/manifests/australia_external_test_manifest_noqa_full_harmonized.csv \
  --output-root data/patches_noqa/australia_full
```

## 3. Full Corrected Sweep

Use the dry run first to inspect which commands will execute:

```bash
python scripts/run_post_sweep.py \
  --dry-run \
  --run-suffix _reflectance64 \
  --eval-suffix _reflectance64
```

Run the full corrected matrix:

```bash
python scripts/run_post_sweep.py \
  --epochs 20 \
  --batch-size 8 \
  --base-channels 64 \
  --num-workers 0 \
  --device cuda \
  --amp \
  --amp-exclude change_transformer \
  --run-suffix _reflectance64 \
  --eval-suffix _reflectance64 \
  --run-kfold \
  --run-mtbs-model \
  --run-cabuar \
  --run-latency \
  --run-xai \
  --run-per-event \
  --run-paired-qa \
  --run-stitched-scenes
```

The sweep trains and evaluates the corrected model matrix, then runs the
configured MTBS, CaBuAr, latency, per-event, QA, stitched-scene, and
explainability diagnostics.

## 4. Single-Model Commands

Train one DeepLabv3+ checkpoint:

```bash
python scripts/train_model.py \
  --model-name deeplab \
  --patch-index data/patches/california_full/patch_index.csv \
  --output-dir data/runs/deeplab_qaon_seed42_reflectance64 \
  --epochs 20 \
  --batch-size 8 \
  --base-channels 64 \
  --seed 42 \
  --device cuda \
  --num-workers 0 \
  --amp \
  --normalize \
  --augment \
  --dynamic-pos-weight \
  --val-split spatial \
  --block-size 1024
```

Evaluate one checkpoint on Australia:

```bash
python scripts/evaluate_model.py \
  --model-name deeplab \
  --checkpoint data/runs/deeplab_qaon_seed42_reflectance64/best_model.pt \
  --patch-index data/patches/australia_full/patch_index.csv \
  --output-dir data/evaluation/australia_full_seed42_reflectance64 \
  --base-channels 64 \
  --device cuda \
  --normalize
```

Run scene-level stitched inference:

```bash
python scripts/evaluate_scene_stitched.py \
  --model-name deeplab \
  --checkpoint data/runs/deeplab_qaon_seed42_reflectance64/best_model.pt \
  --manifest data/manifests/australia_external_test_manifest_harmonized.csv \
  --output-dir data/evaluation/australia_full_seed42_reflectance64_stitched/deeplab \
  --base-channels 64 \
  --patch-size 256 \
  --stride 128 \
  --device cuda \
  --normalize
```

## 5. dNBR Baseline

```bash
python -c "from pathlib import Path; import numpy as np; from src.evaluation.dnbr_baseline import sweep_dnbr_thresholds; sweep_dnbr_thresholds(Path('data/patches/australia_full/patch_index.csv'), [round(float(v), 3) for v in np.arange(0.02, 0.301, 0.02)], Path('data/evaluation/australia_full_dnbr_reflectance64/dnbr_sweep.csv'), normalize=True)"
```

## 6. Event-Level K-Fold and MTBS Validation

Run the Concat U-Net and Siamese U-Net k-fold source experiments:

```bash
python scripts/run_kfold_in_domain.py \
  --patch-index data/patches/california_full/patch_index.csv \
  --output-dir data/paper_assets/stats_reflectance64 \
  --n-splits 5 \
  --epochs 10 \
  --batch-size 8 \
  --base-channels 64 \
  --device cuda \
  --num-workers 0 \
  --normalize \
  --augment \
  --dynamic-pos-weight
```

Run the DeepLabv3+ k-fold source experiment:

```bash
python scripts/run_kfold_in_domain.py \
  --models deeplab \
  --patch-index data/patches/california_full/patch_index.csv \
  --output-dir data/paper_assets/stats_reflectance64_deeplab \
  --n-splits 5 \
  --epochs 10 \
  --batch-size 4 \
  --base-channels 64 \
  --device cuda \
  --num-workers 0 \
  --normalize \
  --augment \
  --dynamic-pos-weight \
  --amp \
  --drop-last-train
```

Validate held-out DeepLabv3+ checkpoints against MTBS:

```bash
python scripts/mtbs_kfold_model_validation.py \
  --model-name deeplab \
  --kfold-dir data/paper_assets/stats_reflectance64_deeplab \
  --output-dir data/paper_assets/mtbs_model_validation_reflectance64 \
  --base-channels 64 \
  --device cuda \
  --normalize \
  --no-write-rasters

python scripts/aggregate_mtbs_model_comparison.py \
  --input-dir data/paper_assets/mtbs_model_validation_reflectance64 \
  --output-dir data/paper_assets/mtbs_model_validation_reflectance64
```

## 7. Independent Perimeter Products

Copernicus EMS validation:

```bash
python scripts/evaluate_scene_stitched.py \
  --model-name deeplab \
  --checkpoint data/runs/deeplab_qaoff_seed42_reflectance64/best_model.pt \
  --manifest data/manifests/copernicus_ems_harmonized_noqa.csv \
  --output-dir data/evaluation/copernicus_ems_deeplab_stitched_seed42_reflectance64 \
  --base-channels 64 \
  --patch-size 256 \
  --stride 128 \
  --device cuda \
  --normalize \
  --write-rasters

python scripts/copernicus_ems_validation.py \
  --manifest data/manifests/copernicus_ems_harmonized_noqa.csv \
  --case-config data/external/copernicus_ems/copernicus_ems_cases.json

python scripts/copernicus_ems_boundary_buffer.py
```

NSW Fire History validation:

```bash
python scripts/nsw_fire_history_validation.py --refresh
```

MCD64A1 and DEA validation:

```bash
python scripts/mcd64a1_independent_validation.py
python scripts/dea_independent_validation.py
```

Pilbara failure-case figure:

```bash
python scripts/create_pilbara_failure_map.py --skip-dea
```

## 8. CaBuAr, Calibration, and Explainability Diagnostics

CaBuAr extraction and evaluation:

```bash
python scripts/cabuar_extract_patches.py \
  --output-root data/patches_cabuar_full

python scripts/cabuar_evaluate_with_ci.py \
  --patch-index data/patches_cabuar_full/patch_index.csv \
  --run-suffix _reflectance64 \
  --base-channels 64 \
  --batch-size 8 \
  --device cuda \
  --no-normalize
```

CaBuAr domain diagnostics:

```bash
python scripts/cabuar_domain_diagnosis.py
python scripts/cabuar_threshold_calibration.py
python scripts/cabuar_finetune_upper_bound.py
```

Explainability faithfulness for one model/seed:

```bash
python scripts/run_xai_faithfulness.py \
  --model-name siamese \
  --checkpoint data/runs/siamese_qaoff_seed42_reflectance64/best_model.pt \
  --patch-index data/patches_noqa/australia_full/patch_index.csv \
  --output-json data/paper_assets/xai_full/faithfulness_n200_seed42_reflectance64.json \
  --output-csv data/paper_assets/xai_full/faithfulness_n200_seed42_reflectance64.csv \
  --num-samples 200 \
  --ig-steps 32 \
  --occlusion-window 32 \
  --occlusion-stride 16 \
  --base-channels 64 \
  --device cuda \
  --normalize

python scripts/aggregate_xai_3seed.py
```

## 9. Mediterranean Third-Region Diagnostic

The Mediterranean extension evaluates the same QA-off `reflectance64`
checkpoints on additional event-sensor pairs. It is a region-shift diagnostic
using dNBR proxy labels, not an independent perimeter claim by itself.

```bash
python scripts/fetch_mediterranean_only.py \
  --skip-existing \
  --sensors landsat8 sentinel2

python scripts/preprocess_manifest.py \
  --manifest data/manifests/mediterranean_external_test_manifest.csv \
  --output-root data/processed

python scripts/create_noqa_manifests.py \
  --input-manifest data/manifests/mediterranean_external_test_manifest_harmonized.csv

python scripts/extract_patches.py \
  --manifest data/manifests/mediterranean_external_test_manifest_harmonized_noqa.csv \
  --output-root data/patches_noqa/mediterranean_full \
  --patch-size 256 \
  --overlap-fraction 0.5

python scripts/eval_mediterranean_3seed.py
```

## 10. Tests

```bash
python -m pytest -q
python -m pytest tests/test_demo_pipeline.py -q
```

Expected status: 19 total local tests pass, including 11 embedded-fixture
tests in `tests/test_demo_pipeline.py`.

## Checkpoint Scope

The main cross-region matrix contains six learned architectures, two QA states,
and three seeds, plus the deterministic dNBR rule. Checkpoints, extracted patch
arrays, validation rasters, and large external products are not stored in Git;
they are rebuilt with the commands above or supplied through the separate
artifact archive named in the paper.
