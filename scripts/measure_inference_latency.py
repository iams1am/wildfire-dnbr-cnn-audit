"""Measure inference latency for each corrected reflectance64 seed-42 model."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.factory import build_model

MODELS = ["baseline", "siamese", "siamese_fcn_conc", "siamese_fcn_diff", "deeplab", "change_transformer"]
device = torch.device("cuda")
rows = []

for m in MODELS:
    ckpt = ROOT / "data" / "runs" / f"{m}_qaoff_seed42_reflectance64" / "best_model.pt"
    model = build_model(m, channels_per_image=5, base_channels=64)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    # warm up
    with torch.no_grad():
        for _ in range(5):
            _ = model(torch.zeros(1, 5, 256, 256, device=device), torch.zeros(1, 5, 256, 256, device=device))
        torch.cuda.synchronize()

        # batch=1
        t0 = time.perf_counter()
        for _ in range(50):
            _ = model(torch.zeros(1, 5, 256, 256, device=device), torch.zeros(1, 5, 256, 256, device=device))
        torch.cuda.synchronize()
        dt1 = (time.perf_counter() - t0) / 50 * 1000

        # batch=8
        t0 = time.perf_counter()
        for _ in range(20):
            _ = model(torch.zeros(8, 5, 256, 256, device=device), torch.zeros(8, 5, 256, 256, device=device))
        torch.cuda.synchronize()
        dt8 = (time.perf_counter() - t0) / 20 / 8 * 1000

    n_params = sum(p.numel() for p in model.parameters())
    rows.append(
        {
            "model": m,
            "params": n_params,
            "params_m": n_params / 1e6,
            "batch1_ms_per_patch": dt1,
            "batch8_ms_per_patch": dt8,
            "batch8_patches_per_second": 1000.0 / dt8,
        }
    )
    print(f"{m:30s}  params={n_params/1e6:5.2f}M  bs1={dt1:6.2f} ms/patch  bs8={dt8:6.2f} ms/patch")

out_csv = ROOT / "data" / "paper_assets" / "tables" / "inference_latency_reflectance64.csv"
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(f"Wrote {out_csv}")
