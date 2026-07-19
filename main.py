"""Minesweeper RL project entry point.

Examples:
    python main.py train --alg=dqn --h=6 --w=6 --m=5 --n-episodes=500
    python main.py test  --alg=ppo --ckpt=checkpoints/*-best.pt --h=6 --w=6 --m=5
    python main.py game  --h=9 --w=9 --m=10
    python main.py stats --alpha 0.01
"""

import argparse
import sys
from datetime import datetime

import wandb
import yaml

from common.paths import resolve_project_path
from common.config_merge import inject_algorithm_root_fields
from tracking.wandb_logger import make_live_validation_callback, run_group
from common import utility
import train.dqn
import train.ppo
import evaluation.dqn
import evaluation.ppo
import evaluation.statistical_testing as stat_testing
from environment import manual_play

DEFAULT_CONFIG_PATH = resolve_project_path("config.yaml")

# quale modulo train/test usare in base al valore di --alg
TRAIN_MODULES = {"dqn": train.dqn, "ppo": train.ppo}
TEST_MODULES = {"dqn": evaluation.dqn, "ppo": evaluation.ppo}

# alcune chiavi di config hanno anche una versione abbreviata del flag
# da riga di comando, ad esempio board_height diventa sia --board-height
# che --h e --height
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
    # legge un file YAML e lo trasforma in un dizionario python
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def deep_merge(base: dict, override: dict) -> dict:
    # unisce due dizionari annidati: per ogni chiave di override, se il
    # valore è a sua volta un dizionario in entrambi i dizionari viene fatto
    # il merge ricorsivamente, altrimenti il valore di override sovrascrive
    # quello di base
    merged = dict(base)

    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value

    return merged


def build_bootstrap_parser() -> argparse.ArgumentParser:
    # questo è un primo parser "leggero", usato solo per scoprire quale
    # comando (train/test/game/stats) e quale algoritmo (dqn/ppo) sono stati
    # richiesti, prima di sapere quali altri flag accettare: i flag validi
    # per --n-episodes, --learning-rate ecc dipendono infatti da comando e
    # algoritmo, quindi non possono essere definiti tutti insieme in un
    # unico parser statico
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("command", choices=["train", "test", "game", "stats"])
    parser.add_argument("--alg", choices=["dqn", "ppo"], default=None)
    parser.add_argument("--config", default=None)
    return parser


def key_to_flag(key: str) -> str:
    # trasforma una chiave di config tipo board_height nel flag --board-height
    # questo ci permette di generare automaticamente i flag da riga di comando
    return "--" + key.replace("_", "-")


def build_full_parser(command: str, subtree: dict) -> argparse.ArgumentParser:
    # questo secondo parser viene costruito dopo aver letto la config, e
    # genera automaticamente un flag da riga di comando per ogni chiave
    # presente nella sezione di config scelta (es. dqn.train), così non
    # serve scrivere ogni flag a mano e config e CLI restano sempre allineati
    parser = argparse.ArgumentParser(prog=f"main.py {command}")
    parser.add_argument("command")
    parser.add_argument("--alg", choices=["dqn", "ppo"], default=None)
    parser.add_argument("--config", default=None)

    for key in subtree:
        flags = [key_to_flag(key)] + ALIASES.get(key, [])
        # default=None ci permette di distinguere "flag non passato" da
        # "flag passato con il valore di default"; type=yaml.safe_load
        # interpreta il valore passato da riga di comando come farebbe
        # il parser YAML (numeri, bool, null, stringhe...) invece di
        # trattarlo sempre come stringa
        parser.add_argument(*flags, dest=key, default=None, type=yaml.safe_load)

    if command == "train":
        parser.add_argument("--auto-test", dest="auto_test", action="store_true", default=None)
        parser.add_argument("--no-auto-test", dest="auto_test", action="store_false", default=None)

    return parser


def main() -> None:
    bootstrap_args, _ = build_bootstrap_parser().parse_known_args()

    if bootstrap_args.command in ("train", "test") and bootstrap_args.alg is None:
        print(f"error: --alg is required for '{bootstrap_args.command}'", file=sys.stderr)
        raise SystemExit(2)

    # ordine di priorità della config, dal più basso al più alto:
    # config.yaml di default < file passato con --config < flag da riga di comando
    config = load_yaml(DEFAULT_CONFIG_PATH)
    if bootstrap_args.config:
        config = deep_merge(config, load_yaml(resolve_project_path(bootstrap_args.config)))

    # subtree è la sezione di config rilevante per il comando corrente,
    # ad esempio config["dqn"]["train"] per "python main.py train --alg=dqn"
    if bootstrap_args.command in ("game", "stats"):
        subtree = config[bootstrap_args.command]
    else:
        subtree = config[bootstrap_args.alg][bootstrap_args.command]

    args = build_full_parser(bootstrap_args.command, subtree).parse_args()

    # tra tutti i flag del parser, teniamo solo quelli effettivamente
    # passati da riga di comando (valore diverso da None) e che
    # corrispondono a una chiave della config: questi sono gli override
    # da applicare sopra ai valori di default della config
    overrides = {}
    for key, value in vars(args).items():
        if key in subtree and value is not None:
            overrides[key] = value

    # il dizionario finale di config da usare per l'esecuzione è
    # ottenuto facendo il merge della config di default con gli override
    run_config = {**subtree, **overrides}

    if bootstrap_args.command == "stats":
        output_wilcoxon = stat_testing.pairwise_wilcoxon_returns(
            run_config["csv_a"], run_config["csv_b"], alpha=run_config["alpha"])
        output_mcnemar = stat_testing.pairwise_mcnemar_winrate(
            run_config["csv_a"], run_config["csv_b"], alpha=run_config["alpha"]
        )
        print(output_wilcoxon)
        print(output_mcnemar)
        utility.write_txt(
            output_wilcoxon + "\n" + output_mcnemar,
            run_config["save_path"],
            run_config["csv_a"],
            run_config["csv_b"],
        )
        return

    if bootstrap_args.command != "game":
        inject_algorithm_root_fields(run_config, config, bootstrap_args.alg)

    if bootstrap_args.command == "game":
        manual_play.play(run_config)
        return

    if bootstrap_args.command == "train":
        # timestamp per il nome della run
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        # apriamo la run wandb prima di iniziare il training per il
        # logging live se richiesto
        wandb.init(
            project=run_config["wandb_project"],
            entity=run_config["wandb_entity"],
            job_type=bootstrap_args.alg,
            group=run_group(
                bootstrap_args.alg,
                run_config["architecture_name"],
                run_config["board_height"],
                run_config["board_width"],
                run_config["n_mines"],
            ),
            name=f"{bootstrap_args.alg}-{timestamp}-train",
            config=run_config,
        )

        # l'effettiva callback per il logging live
        on_validation = make_live_validation_callback(bootstrap_args.alg)
        result = TRAIN_MODULES[bootstrap_args.alg].run(run_config, on_validation=on_validation)

        # chiudo la run wandb
        wandb.finish()

        # auto-test dopo il training se richiesto
        auto_test = args.auto_test
        if auto_test is None:
            auto_test = config.get("main", {}).get("auto_test", True)

        if auto_test:
            # il test viene lanciato sulla stessa combinazione di board
            # usato per il training, sul checkpoint migliore appena prodotto
            test_subtree = dict(config[bootstrap_args.alg]["test"])
            test_subtree["checkpoint_path"] = str(result["best_checkpoint_path"])
            test_subtree["board_height"] = result["board_height"]
            test_subtree["board_width"] = result["board_width"]
            test_subtree["n_mines"] = result["n_mines"]
            test_subtree["architecture_name"] = config[bootstrap_args.alg]["architecture_name"]
            test_subtree["device"] = run_config["device"]
            TEST_MODULES[bootstrap_args.alg].run(test_subtree)
        return

    if bootstrap_args.command == "test":
        TEST_MODULES[bootstrap_args.alg].run(run_config)


if __name__ == "__main__":
    main()
