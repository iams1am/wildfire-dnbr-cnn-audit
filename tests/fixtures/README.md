# Demo Fixtures

Two small `.npz` patches are embedded so the smoke test can run without
downloading external data:

```bash
python -m pytest -q tests/test_demo_pipeline.py
```

Each file contains:

- `pre`: `(5, 256, 256)` pre-event bands `{B, G, R, NIR, SWIR2}`;
- `post`: `(5, 256, 256)` post-event bands with the same band order; and
- `label`: `(256, 256)` binary dNBR-derived burn mask at `tau=0.10`.

| File | Source event | Burned fraction | Size |
|---|---|---:|---:|
| `demo_california_landsat8.npz` | Camp Fire 2018, California, Landsat-8 | 0.26 | ~1.3 MB |
| `demo_australia_landsat8.npz` | Australia external test, Landsat-8 | 0.35 | ~0.6 MB |

The arrays are real Landsat patch samples from the same data path as the main
benchmark. They are stored as compact raw digital-number patch arrays; the
dataset loader applies the Landsat reflectance scale and offset during tests,
matching the normalization used before model inference.
