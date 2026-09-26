"""Command line entrypoint: python -m mua.cli <command>."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable

from mua.config import load_config

logger = logging.getLogger("mua")

STUB_COMMANDS = ("train", "causal", "auction", "report")


def _run_generate(args: argparse.Namespace) -> None:
    from mua.sim.generate import run

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.scale == "small":
        cfg["target_rows"] = int(cfg.get("small_rows", 50_000))
        logger.info("Small scale: %s rows", cfg["target_rows"])
    if args.rows is not None:
        cfg["target_rows"] = args.rows
    run(cfg)


def _run_features(args: argparse.Namespace) -> None:
    from mua.features.build import build

    cfg = load_config(args.config)
    build(cfg)


def _stub(name: str) -> Callable[[argparse.Namespace], None]:
    def _run(args: argparse.Namespace) -> None:
        logger.warning("%s: not implemented yet", name)

    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mua", description="Marketplace uplift modeling and auction pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate simulated marketplace data")
    gen.add_argument("--config", default="configs/sim.yaml")
    gen.add_argument("--seed", type=int, default=None, help="Override the config seed")
    gen.add_argument(
        "--scale", choices=["full", "small"], default="full", help="small = 50k rows for CI"
    )
    gen.add_argument("--rows", type=int, default=None, help="Override target row count")
    gen.set_defaults(func=_run_generate)

    feat = sub.add_parser("features", help="Join dims, encode, split into train/valid/test")
    feat.add_argument("--config", default="configs/features.yaml")
    feat.set_defaults(func=_run_features)

    for name in STUB_COMMANDS:
        stub = sub.add_parser(name, help=f"Placeholder for the {name} stage")
        stub.set_defaults(func=_stub(name))

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
