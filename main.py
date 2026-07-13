"""Minesweeper RL project entry point.

Examples:
    python main.py train --alg=dqn --h=6 --w=5 --m=5 --n-episodes=500
    python main.py test  --alg=ppo --ckpt=checkpoints/ppo/best/best.pt --h=6 --w=6 --m=5
    python main.py game  --h=9 --w=9 --m=10
"""

import argparse
import importlib
import sys

import yaml

from common.paths import resolve_project_path

DEFAULT_CONFIG_PATH = resolve_project_path("config.yaml")

# Config keys that get a short CLI alias in addition to their full
# dashed-key name (e.g. board_height also becomes --h / --height).
ALIASES = {
    "board_height": ["--h", "--height"],
    "board_width": ["--w", "--width"],
    "n_mines": ["--m", "--mines"],
    "checkpoint_path": ["--ckpt"],
    "height": ["--h"],
    "width": ["--w"],
    "mines": ["--m"],
}


def load_yaml(path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def build_bootstrap_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["train", "test", "game"])
    parser.add_argument("--alg", choices=["dqn", "ppo"], default=None)
    parser.add_argument("--config", default=None)
    return parser


def key_to_flag(key: str) -> str:
    return "--" + key.replace("_", "-")


def build_full_parser(command: str, subtree: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"main.py {command}")
    parser.add_argument("command")
    parser.add_argument("--alg", choices=["dqn", "ppo"], default=None)
    parser.add_argument("--config", default=None)

    for key in subtree:
        flags = [key_to_flag(key)] + ALIASES.get(key, [])
        parser.add_argument(*flags, dest=key, default=None, type=yaml.safe_load)

    if command == "train":
        parser.add_argument("--auto-test", dest="auto_test", action="store_true", default=None)
        parser.add_argument("--no-auto-test", dest="auto_test", action="store_false", default=None)

    return parser


def _import_train(alg: str):
    return importlib.import_module(f"train.{alg}")


def _import_test(alg: str):
    return importlib.import_module(f"test.{alg}")


def main() -> None:
    bootstrap_args, _ = build_bootstrap_parser().parse_known_args()

    if bootstrap_args.command in ("train", "test") and bootstrap_args.alg is None:
        print(f"error: --alg is required for '{bootstrap_args.command}'", file=sys.stderr)
        raise SystemExit(2)

    config = load_yaml(DEFAULT_CONFIG_PATH)
    if bootstrap_args.config:
        config = deep_merge(config, load_yaml(resolve_project_path(bootstrap_args.config)))

    if bootstrap_args.command == "game":
        subtree = config["game"]
    else:
        subtree = config[bootstrap_args.alg][bootstrap_args.command]

    args = build_full_parser(bootstrap_args.command, subtree).parse_args()

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in subtree and value is not None
    }
    run_config = {**subtree, **overrides}

    if bootstrap_args.command == "game":
        from environment import manual_play

        manual_play.play(run_config)
        return

    if bootstrap_args.command == "train":
        result = _import_train(bootstrap_args.alg).run(run_config)

        auto_test = args.auto_test
        if auto_test is None:
            auto_test = config.get("main", {}).get("auto_test", True)

        if auto_test:
            test_subtree = dict(config[bootstrap_args.alg]["test"])
            test_subtree["checkpoint_path"] = str(result["best_checkpoint_path"])
            test_subtree["board_height"] = result["board_height"]
            test_subtree["board_width"] = result["board_width"]
            test_subtree["n_mines"] = result["n_mines"]
            _import_test(bootstrap_args.alg).run(test_subtree)
        return

    if bootstrap_args.command == "test":
        _import_test(bootstrap_args.alg).run(run_config)


if __name__ == "__main__":
    main()
