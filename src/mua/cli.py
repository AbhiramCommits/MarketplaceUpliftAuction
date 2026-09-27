"""Command line entrypoint: python -m mua.cli <command>."""

from __future__ import annotations

import argparse
import logging
import sys

from mua.config import load_config

logger = logging.getLogger("mua")


def _run_generate(args: argparse.Namespace) -> int:
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
    return 0


def _run_features(args: argparse.Namespace) -> int:
    from mua.features.build import build

    cfg = load_config(args.config)
    build(cfg)
    return 0


def _run_train_ranker(args: argparse.Namespace) -> int:
    from mua.ranker.train import run

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.in_memory:
        cfg["data"]["in_memory"] = True
    _, ok = run(cfg)
    return 0 if ok else 1


def _run_score(args: argparse.Namespace) -> int:
    from mua.ranker.predict import run

    cfg = load_config(args.config)
    run(cfg)
    return 0


def _run_causal(args: argparse.Namespace) -> int:
    from mua.causal.run import run

    cfg = load_config(args.config)
    _, ok = run(cfg, selected=args.estimator)
    return 0 if ok else 1


def _run_refute(args: argparse.Namespace) -> int:
    from mua.causal.refute import run

    cfg = load_config(args.config)
    result = run(cfg)
    return 0 if result["ok"] else 1


def _run_auction(args: argparse.Namespace) -> int:
    from mua.auction.simulator import run

    cfg = load_config(args.config)
    if args.mechanism is not None:
        cfg["auction"]["mechanism"] = args.mechanism
    if args.rounds is not None:
        cfg["auction"]["rounds"] = args.rounds
    run(cfg)
    return 0


def _run_experiment(args: argparse.Namespace) -> int:
    from mua.reporting.experiment import run

    cfg = load_config(args.config)
    run(cfg, scale=args.scale)
    return 0


def _run_report(args: argparse.Namespace) -> int:
    from mua.reporting.report import run

    cfg = load_config(args.config)
    run(cfg, scale=args.scale)
    return 0


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

    rank = sub.add_parser("train-ranker", help="Train + calibrate the CTR ranker")
    rank.add_argument("--config", default="configs/ranker.yaml")
    rank.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    rank.add_argument("--in-memory", action="store_true", help="Load datasets in memory")
    rank.set_defaults(func=_run_train_ranker)

    score = sub.add_parser("score", help="Batch-score splits and write pctr to data/scored/")
    score.add_argument("--config", default="configs/ranker.yaml")
    score.set_defaults(func=_run_score)

    causal = sub.add_parser(
        "causal", help="Fit and evaluate ATE/CATE estimators against ground truth"
    )
    causal.add_argument("--config", default="configs/causal.yaml")
    causal.add_argument(
        "--estimator",
        default="all",
        choices=["all", "naive", "iptw", "s", "t", "x", "dr", "forest"],
        help="Which estimator(s) to fit (default: all)",
    )
    causal.set_defaults(func=_run_causal)

    refute = sub.add_parser("refute", help="Run the DoWhy refutation battery")
    refute.add_argument("--config", default="configs/causal.yaml")
    refute.set_defaults(func=_run_refute)

    auction = sub.add_parser("auction", help="Simulate the ads auction")
    auction.add_argument("--config", default="configs/auction.yaml")
    auction.add_argument(
        "--mechanism",
        choices=["first_price", "second_price", "gsp", "vcg"],
        default=None,
        help="Auction mechanism (default: config value)",
    )
    auction.add_argument("--rounds", type=int, default=None, help="Number of auction rounds")
    auction.set_defaults(func=_run_auction)

    experiment = sub.add_parser(
        "experiment", help="Run the mechanism x targeting policy experiment grid"
    )
    experiment.add_argument("--config", default="configs/experiment.yaml")
    experiment.add_argument("--scale", choices=["full", "small"], default="full")
    experiment.set_defaults(func=_run_experiment)

    report = sub.add_parser("report", help="Generate reports/REPORT.md")
    report.add_argument("--config", default="configs/experiment.yaml")
    report.add_argument("--scale", choices=["full", "small"], default="full")
    report.set_defaults(func=_run_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
