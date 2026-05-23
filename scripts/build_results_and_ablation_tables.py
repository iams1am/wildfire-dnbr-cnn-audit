from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_summary(model: str, qa_masking: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "qa_masking": qa_masking,
        "IoU": float(summary["iou"]),
        "F1": float(summary["f1"]),
        "Precision": float(summary["precision"]),
        "Recall": float(summary["recall"]),
        "Area_MAE_sq_km": float(summary["area_mae_sq_km"]),
        "Area_RMSE_sq_km": float(summary["area_rmse_sq_km"]),
        "Patch_Count": int(round(float(summary["patch_count"]))),
    }


def _write_table(df: pd.DataFrame, csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    markdown_path.write_text(df.to_markdown(index=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build results and ablation tables from summary JSON files.")
    parser.add_argument(
        "--qa-on-baseline",
        type=Path,
        default=Path("data/evaluation/australia/baseline_summary.json"),
    )
    parser.add_argument(
        "--qa-on-siamese",
        type=Path,
        default=Path("data/evaluation/australia/siamese_summary.json"),
    )
    parser.add_argument(
        "--qa-off-baseline",
        type=Path,
        default=Path("data/evaluation/australia_noqa/baseline_summary.json"),
    )
    parser.add_argument(
        "--qa-off-siamese",
        type=Path,
        default=Path("data/evaluation/australia_noqa/siamese_summary.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/paper_assets/tables"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    qa_on_rows = [
        _row_from_summary("baseline", "on", _read_summary(args.qa_on_baseline)),
        _row_from_summary("siamese", "on", _read_summary(args.qa_on_siamese)),
    ]
    results_df = pd.DataFrame(qa_on_rows).sort_values(by="model").reset_index(drop=True)
    _write_table(
        results_df,
        args.output_dir / "results_table.csv",
        args.output_dir / "results_table.md",
    )

    ablation_rows = qa_on_rows + [
        _row_from_summary("baseline", "off", _read_summary(args.qa_off_baseline)),
        _row_from_summary("siamese", "off", _read_summary(args.qa_off_siamese)),
    ]
    ablation_df = (
        pd.DataFrame(ablation_rows)
        .sort_values(by=["model", "qa_masking"], ascending=[True, False])
        .reset_index(drop=True)
    )
    _write_table(
        ablation_df,
        args.output_dir / "ablation_table.csv",
        args.output_dir / "ablation_table.md",
    )

    print(f"Wrote {args.output_dir / 'results_table.csv'}")
    print(f"Wrote {args.output_dir / 'results_table.md'}")
    print(f"Wrote {args.output_dir / 'ablation_table.csv'}")
    print(f"Wrote {args.output_dir / 'ablation_table.md'}")


if __name__ == "__main__":
    main()
