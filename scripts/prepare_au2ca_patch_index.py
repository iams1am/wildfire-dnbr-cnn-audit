"""Prepare an Australia-as-train patch index for the AU->CA reverse-transfer experiment.

The original australia_full patch_index.csv has split='external_test' for all 4456
patches (they were generated for inference on California-trained models). For the
AU->CA reverse transfer, we relabel these patches as 'train_val' so the standard
training script can consume them, holding out one Australia event for validation
to enable early stopping.

Output: data/patches/australia_full_train/patch_index.csv (a sibling index file
        pointing at the same .npz patches).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_INDEX = PROJECT_ROOT / "data" / "patches" / "australia_full" / "patch_index.csv"
DST_DIR = PROJECT_ROOT / "data" / "patches" / "australia_full_train"
DST_INDEX = DST_DIR / "patch_index.csv"

# Use Kangaroo Island as the validation event (single event, contains both Landsat-8 and Sentinel-2 pairs and includes the burn-majority prior-shift failure mode, so it stresses the val signal).
VAL_EVENT = "kangaroo_island_fire_2020"


def main() -> None:
    src = pd.read_csv(SRC_INDEX)
    out = src.copy()
    out["split"] = out["event_id"].apply(
        lambda e: "val" if str(e) == VAL_EVENT else "train_val"
    )
    DST_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(DST_INDEX, index=False)
    print(f"Wrote {DST_INDEX}")
    print(f"  train_val patches: {(out['split']=='train_val').sum()}")
    print(f"  val patches      : {(out['split']=='val').sum()}")
    print(f"  total events     : {out['event_id'].nunique()}")
    print(f"  val event        : {VAL_EVENT}")


if __name__ == "__main__":
    main()
