from __future__ import annotations

import pytest

from mua import cli

COMMANDS = [
    ("generate", "mua.sim.generate", "run"),
    ("features", "mua.features.build", "build"),
    ("train-ranker", "mua.ranker.train", "run"),
    ("score", "mua.ranker.predict", "run"),
    ("causal", "mua.causal.run", "run"),
    ("refute", "mua.causal.refute", "run"),
    ("auction", "mua.auction.simulator", "run"),
    ("experiment", "mua.reporting.experiment", "run"),
    ("report", "mua.reporting.report", "run"),
]


@pytest.mark.parametrize("command,module,attr", COMMANDS)
def test_command_dispatch(monkeypatch, command, module, attr):

    if command == "causal":
        monkeypatch.setattr(f"{module}.{attr}", lambda cfg, selected="all": ({"ok": True}, True))
    elif command == "train-ranker":
        monkeypatch.setattr(f"{module}.{attr}", lambda cfg: ({"ok": True}, True))
    elif command == "refute":
        monkeypatch.setattr(f"{module}.{attr}", lambda cfg: {"ok": True})
    elif command in ("experiment", "report"):
        monkeypatch.setattr(f"{module}.{attr}", lambda cfg, scale="full": None)
    else:
        monkeypatch.setattr(f"{module}.{attr}", lambda cfg: None)

    args = [command]
    if command == "generate":
        args += ["--scale", "small"]
    rc = cli.main(args)
    assert rc == 0


def test_parser_options():
    parser = cli.build_parser()
    args = parser.parse_args(["generate", "--scale", "small", "--seed", "7", "--rows", "100"])
    assert args.scale == "small"
    assert args.seed == 7
    assert args.rows == 100
    auction = parser.parse_args(["auction", "--mechanism", "vcg", "--rounds", "10"])
    assert auction.mechanism == "vcg"
    assert auction.rounds == 10
    causal = parser.parse_args(["causal", "--estimator", "dr"])
    assert causal.estimator == "dr"


def test_config_helpers(tmp_path):
    from mua import config

    assert config.repo_root().name == "MarketplaceUpliftAuction"
    path = tmp_path / "cfg.yaml"
    path.write_text("a: 1\nb: [2, 3]\n")
    assert config.load_config(path) == {"a": 1, "b": [2, 3]}
    assert config.resolve(path) == path
    assert config.resolve("data/raw") == config.repo_root() / "data/raw"
