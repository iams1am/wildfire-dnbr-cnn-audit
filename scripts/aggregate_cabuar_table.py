from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the CaBuAr summary table.")
    parser.add_argument("--input-csv", type=Path, default=ROOT / "data" / "evaluation" / "cabuar_full_reflectance64" / "cabuar_summary_ci.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "paper_assets" / "tables")
    parser.add_argument("--output-stem", type=str, default="cabuar_results_reflectance64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Missing CaBuAr summary: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.output_stem}.csv"
    md_path = args.output_dir / f"{args.output_stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
