"""Small command-line entrypoints for local learning loops."""

from __future__ import annotations

import argparse
import json

from opjax.isft.dataset import build_fake_click_isft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="opjax")
    subparsers = parser.add_subparsers(dest="command", required=True)

    click_isft = subparsers.add_parser("build-click-isft", help="Build a synthetic click iSFT/RGT JSONL dataset.")
    click_isft.add_argument("--output-dir", default="data/smoke/click_isft", help="Output directory for images and records.jsonl.")
    click_isft.add_argument("--count-per-tier", type=int, default=2, help="Number of examples per curriculum tier.")
    click_isft.add_argument("--seed", type=int, default=0, help="Initial deterministic task seed.")

    args = parser.parse_args()
    if args.command == "build-click-isft":
        result = build_fake_click_isft_dataset(
            args.output_dir,
            count_per_tier=args.count_per_tier,
            seed=args.seed,
        )
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
