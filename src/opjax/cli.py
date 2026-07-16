"""Small command-line entrypoints for local learning loops."""

from __future__ import annotations

import argparse
import json
import os

from opjax.isft.claude import ClaudeRepairer
from opjax.isft.dataset import (
    build_fake_click_isft_dataset,
    build_real_click_isft_dataset,
)
from opjax.isft.fake import FakeClickRepairer


def main() -> None:
    parser = argparse.ArgumentParser(prog="opjax")
    subparsers = parser.add_subparsers(dest="command", required=True)

    click_isft = subparsers.add_parser(
        "build-click-isft",
        help="Build a synthetic click iSFT/RGT JSONL dataset with the fake repairer.",
    )
    click_isft.add_argument("--output-dir", default="data/smoke/click_isft")
    click_isft.add_argument("--count-per-tier", type=int, default=2)
    click_isft.add_argument("--seed", type=int, default=0)

    real_isft = subparsers.add_parser(
        "build-real-click-isft",
        help="Refine a JSONL of Gemma 4 base-model attempts into an iSFT dataset.",
    )
    real_isft.add_argument(
        "--attempts-jsonl",
        required=True,
        help="Path to JSONL produced by Modal gemma4_batch_click_attempts_cli.",
    )
    real_isft.add_argument("--output-dir", default="data/click_isft")
    real_isft.add_argument(
        "--repairer",
        choices=["claude", "fake"],
        default="claude",
        help="Use Claude (real, requires ANTHROPIC_API_KEY) or fake (oracle).",
    )
    real_isft.add_argument(
        "--include-image",
        action="store_true",
        default=True,
        help="Pass the click image to Claude (recommended for grounding).",
    )
    real_isft.add_argument("--max-rounds", type=int, default=3)
    real_isft.add_argument("--claude-model", default="claude-sonnet-4-6")

    hub_push = subparsers.add_parser(
        "push-isft-to-hub",
        help="Push a local iSFT dataset directory to the Hugging Face Hub.",
    )
    hub_push.add_argument("--local-dir", required=True)
    hub_push.add_argument("--repo-id", default="sudarshan/opjax-click-isft")
    hub_push.add_argument("--private", action="store_true", default=False)
    hub_push.add_argument(
        "--token-env",
        default="HF_TOKEN",
        help="Environment variable name carrying the HF write token.",
    )
    hub_push.add_argument(
        "--commit-message",
        default="phase 1c: opjax click iSFT/RGT dataset",
    )

    factory = subparsers.add_parser(
        "factory",
        help="Model Factory data path (scrub / render-tinker / audit / preflight).",
    )
    factory.add_argument(
        "factory_args",
        nargs=argparse.REMAINDER,
        help="Args forwarded to `python -m opjax.factory` (e.g. preflight --dataset ...).",
    )

    args = parser.parse_args()

    if args.command == "build-click-isft":
        result = build_fake_click_isft_dataset(
            args.output_dir,
            count_per_tier=args.count_per_tier,
            seed=args.seed,
        )
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "build-real-click-isft":
        repairer = (
            ClaudeRepairer(model=args.claude_model, include_image=args.include_image)
            if args.repairer == "claude"
            else FakeClickRepairer()
        )
        result = build_real_click_isft_dataset(
            args.output_dir,
            attempts_jsonl=args.attempts_jsonl,
            repairer=repairer,
            max_rounds=args.max_rounds,
        )
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return

    if args.command == "push-isft-to-hub":
        from opjax.isft.hub import push_dataset_to_hub

        token = os.environ.get(args.token_env)
        if not token:
            raise SystemExit(
                f"Environment variable {args.token_env} is empty; set it to a "
                "HuggingFace write token before pushing."
            )
        result = push_dataset_to_hub(
            local_dir=args.local_dir,
            repo_id=args.repo_id,
            token=token,
            private=args.private,
            commit_message=args.commit_message,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.command == "factory":
        from opjax.factory.cli import main as factory_main

        forwarded = list(args.factory_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        factory_main(forwarded)
        return

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
