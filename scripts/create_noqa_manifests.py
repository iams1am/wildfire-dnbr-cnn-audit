from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create no-QA manifest copies by removing QA paths/mode.")
    parser.add_argument("--input-manifest", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--suffix",
        type=str,
        default="_noqa",
        help="Suffix to append before .csv in output manifest names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for manifest_path in args.input_manifest:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        df = pd.read_csv(manifest_path)
        for col in ("pre_qa_path", "post_qa_path", "qa_mode"):
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = ""
        output_path = manifest_path.with_name(f"{manifest_path.stem}{args.suffix}.csv")
        df.to_csv(output_path, index=False)
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
