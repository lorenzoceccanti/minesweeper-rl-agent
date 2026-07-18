"""interfaccia da terminale per la ricerca distribuita di iperparametri/architetture

sottocomandi disponibili:
    validate  -- controlla la campagna yaml e stima quanti trial/episodi servirebbero
    register  -- crea uno sweep su wandb per ogni tripla algoritmo/task/architettura
    worker    -- fa girare i trial degli sweep gia' registrati (qui si allena davvero)
    promote   -- prende i finalisti, li conferma e li testa su episodi mai visti
    report    -- genera un report markdown a partire dai risultati salvati
"""

from __future__ import annotations

import argparse
import functools
import sys
from pathlib import Path
from typing import Any

import yaml

from common.paths import resolve_project_path
from sweeps.campaign_schema import validate_campaign
from sweeps.config_space import iter_algorithm_architecture_pairs
from sweeps.promote import PromotionStore, format_report, promote_sweep, promotion_path_for
from sweeps.registry import Registry, build_code_state, code_state_mismatches
from sweeps.sweep_builder import build_sweep_config, sweep_name
from sweeps.train_entrypoint import run_trial

DEFAULT_GRID_TRIAL_WARNING_THRESHOLD = 100


# legge il file yaml della campagna e lo trasforma in un dict python
def load_campaign(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


# dove viene salvato il json che tiene traccia degli sweep gia' registrati per questa campagna
def registry_path_for(campaign_name: str) -> Path:
    return resolve_project_path(f"sweeps/registry/{campaign_name}.json")


def expand_triples(campaign: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    """Every (task, algorithm, architecture) triple a campaign generates one sweep for."""
    # fa il prodotto cartesiano tra task e coppie algoritmo/architettura
    # quindi se ho 2 task e 3 combinazioni algoritmo-architettura, ottengo 6 sweep totali
    return [
        (task, algorithm, architecture_name)
        for task in campaign["tasks"]
        for algorithm, architecture_name in iter_algorithm_architecture_pairs(campaign["architectures"])
    ]


# se lo sweep e' grid search calcola quante combinazioni totali verrebbero provate
def grid_trial_count(sweep_config: dict[str, Any]) -> int | None:
    """Total number of trials a `grid` sweep would run; None for non-grid methods."""
    if sweep_config["method"] != "grid":
        return None

    total = 1
    for spec in sweep_config["parameters"].values():
        if "values" in spec:
            total *= len(spec["values"])
        elif "value" not in spec:
            raise ValueError("grid search requires every parameter to use 'values' or a fixed 'value'")
    return total


# subcommand "validate": controlla la campagna e stampa una stima di quanti episodi/trial costerebbe
# senza registrare niente su wandb, serve per controllare prima di spendere risorse vere
def cmd_validate(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.campaign)
    errors = validate_campaign(campaign)
    if errors:
        print("Campaign is invalid:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    triples = expand_triples(campaign)
    print(f"{len(triples)} sweep(s) will be created:")
    total_screening_episodes = 0
    for task, algorithm, architecture_name in triples:
        sweep_config = build_sweep_config(campaign, task, algorithm, architecture_name)
        episode_budget = sweep_config["parameters"]["n_episodes"]["value"]
        total_screening_episodes += episode_budget
        trial_count = grid_trial_count(sweep_config)
        name = sweep_config["name"]

        # safety net so a grid search with a huge combinatorial explosion doesn't silently burn compute
        if (
            trial_count is not None
            and trial_count > DEFAULT_GRID_TRIAL_WARNING_THRESHOLD
            and not args.allow_large_grid
        ):
            print(
                f"error: sweep {name!r} is a grid search with {trial_count} trials "
                f"(> {DEFAULT_GRID_TRIAL_WARNING_THRESHOLD}); pass --allow-large-grid to proceed",
                file=sys.stderr,
            )
            return 1

        trial_note = f", {trial_count} grid trials" if trial_count is not None else ""
        print(f"  - {name}: {episode_budget} screening episodes/trial{trial_note}")

    print(f"Total screening episode budget (one trial per sweep): {total_screening_episodes}")

    # rough worst-case estimate assuming every single sweep promotes the max number of finalists
    promotion = campaign["promotion"]
    confirm_cost = (
        len(triples) * promotion["finalists_per_sweep"]
        * len(promotion["confirm_seeds"]) * promotion["confirm_episodes"]
    )
    print(
        f"Confirmation cost if every sweep promotes {promotion['finalists_per_sweep']} finalists: "
        f"{confirm_cost} episodes ({len(promotion['confirm_seeds'])} seeds x "
        f"{promotion['confirm_episodes']} episodes each)"
    )
    return 0


# subcommand "register": crea davvero gli sweep su wandb, uno per ogni tripla task/algoritmo/architettura
# e li salva nel registry cosi' i worker sanno quali sweep esistono
def cmd_register(args: argparse.Namespace) -> int:
    import wandb

    campaign = load_campaign(args.campaign)
    errors = validate_campaign(campaign)
    if errors:
        print("Campaign is invalid, aborting registration:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    registry = Registry(registry_path_for(campaign["campaign_name"]))
    # facciamo lo snapshot del codice una volta sola qui, tutti gli sweep sotto avranno lo stesso stato
    code_state = build_code_state()

    for task, algorithm, architecture_name in expand_triples(campaign):
        sweep_config = build_sweep_config(campaign, task, algorithm, architecture_name)
        # this actually talks to the w&b servers and creates the sweep remotely
        sweep_id = wandb.sweep(sweep_config, project=args.project, entity=args.entity)
        registry.register(
            sweep_id,
            task_id=task["id"],
            algorithm=algorithm,
            architecture_name=architecture_name,
            code_state=code_state,
        )
        print(f"registered {sweep_config['name']!r} -> sweep_id={sweep_id}")

    registry.save()
    print(f"registry saved to {registry.path}")
    return 0


# subcommand "worker": questa e' la macchina che davvero allena i modelli
# pesca gli sweep gia' registrati compatibili col profilo passato e ci fa girare wandb.agent sopra
def cmd_worker(args: argparse.Namespace) -> int:
    import wandb

    campaign = load_campaign(args.campaign)
    profile = campaign["worker_profiles"].get(args.profile)
    if profile is None:
        print(f"error: unknown worker profile {args.profile!r}", file=sys.stderr)
        return 1

    registry = Registry(registry_path_for(campaign["campaign_name"]))
    entries = registry.entries_for_algorithms(profile["algorithms"])
    if args.sweep_id is not None:
        # utile per lanciare uno sweep alla volta in processi separati (uno per
        # sweep, in parallelo su una sola macchina) invece che in sequenza
        entries = {sweep_id: entry for sweep_id, entry in entries.items() if sweep_id == args.sweep_id}
    if not entries:
        print(f"no registered sweeps match algorithms {profile['algorithms']}", file=sys.stderr)
        return 1

    current_code_state = build_code_state()

    for sweep_id, entry in entries.items():
        # controllo di sicurezza: se il codice e' cambiato da quando lo sweep e' stato registrato
        # meglio bloccare tutto, altrimenti i risultati non sono piu' confrontabili tra loro
        mismatches = code_state_mismatches(entry, current_code_state)
        if mismatches and not args.allow_code_mismatch:
            print(
                f"error: sweep {sweep_id} code mismatch, refusing to run "
                "(use --allow-code-mismatch to override):",
                file=sys.stderr,
            )
            for mismatch in mismatches:
                print(f"  - {mismatch}", file=sys.stderr)
            return 1
        if mismatches:
            print(f"warning: sweep {sweep_id} code mismatch overridden by --allow-code-mismatch:")
            for mismatch in mismatches:
                print(f"  - {mismatch}")

        print(f"running sweep {sweep_id} (algorithm={entry['algorithm']})")
        # wandb.agent si blocca qui e continua a chiedere nuovi trial al controller dello sweep
        # functools.partial pre-riempie l'argomento algorithm perche' run_trial lo vuole per primo
        wandb.agent(
            sweep_id,
            function=functools.partial(run_trial, entry["algorithm"]),
            project=args.project,
            entity=args.entity,
            count=args.count,
        )

    return 0


# subcommand "promote": prende i migliori risultati di ogni sweep, li ri-allena su piu' seed
# per confermare che non sia stata fortuna, e testa il vincitore su episodi mai visti
def cmd_promote(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.campaign)
    registry = Registry(registry_path_for(campaign["campaign_name"]))
    store = PromotionStore(promotion_path_for(campaign["campaign_name"]))

    sweep_ids = [args.sweep_id] if args.sweep_id else registry.sweep_ids()
    if not sweep_ids:
        print("no registered sweeps to promote", file=sys.stderr)
        return 1

    for sweep_id in sweep_ids:
        entry = registry.get(sweep_id)
        print(f"promoting {sweep_id} (algorithm={entry['algorithm']}, task={entry['task_id']})")
        # this is the heavy part: finalists -> multi-seed confirm -> held-out test, see promote.py
        result = promote_sweep(
            sweep_id,
            entry["algorithm"],
            entry["task_id"],
            entry["architecture"],
            campaign["promotion"],
            campaign["test"],
            project=args.project,
            entity=args.entity,
        )
        store.set_result(sweep_id, result)
        if result["winner"] is None:
            print(f"  no winner: no config completed every confirmation seed")
        else:
            print(f"  winner: config {result['winner']['config_id']!r}")

    store.save()
    print(f"promotion results saved to {store.path}")
    return 0


# subcommand "report": genera un file markdown leggibile con i risultati salvati dal promote
def cmd_report(args: argparse.Namespace) -> int:
    campaign = load_campaign(args.campaign)
    store = PromotionStore(promotion_path_for(campaign["campaign_name"]))
    if not store.results():
        print("no promotion results found; run 'promote' first", file=sys.stderr)
        return 1

    report = format_report(campaign["campaign_name"], store.results())
    output_path = resolve_project_path(f"sweeps/reports/{campaign['campaign_name']}.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(f"report written to {output_path}")
    return 0


# qui si costruisce il parser argparse con tutti i subcommand e le loro opzioni
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sweeps.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # each block below just wires up one subcommand's flags and points it at its cmd_ function
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--campaign", required=True)
    validate_parser.add_argument("--allow-large-grid", action="store_true")
    validate_parser.set_defaults(func=cmd_validate)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--campaign", required=True)
    register_parser.add_argument("--project", default="minesweeper-rl")
    register_parser.add_argument("--entity", default=None)
    register_parser.set_defaults(func=cmd_register)

    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--campaign", required=True)
    worker_parser.add_argument("--profile", required=True)
    worker_parser.add_argument("--count", type=int, default=None)
    worker_parser.add_argument("--sweep-id", default=None)
    worker_parser.add_argument("--project", default="minesweeper-rl")
    worker_parser.add_argument("--entity", default=None)
    worker_parser.add_argument("--allow-code-mismatch", action="store_true")
    worker_parser.set_defaults(func=cmd_worker)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument("--campaign", required=True)
    promote_parser.add_argument("--sweep-id", default=None)
    promote_parser.add_argument("--project", default="minesweeper-rl")
    promote_parser.add_argument("--entity", default=None)
    promote_parser.set_defaults(func=cmd_promote)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--campaign", required=True)
    report_parser.set_defaults(func=cmd_report)

    return parser


# entry point: legge gli argomenti da linea di comando e chiama la funzione giusta
# in base al subcommand (validate/register/worker/promote/report)
def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
