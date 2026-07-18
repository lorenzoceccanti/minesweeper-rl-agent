"""funzioni per dare un identita', ordinare e aggregare i risultati della ricerca

config_id hasha un dict di iperparametri per raggruppare i run della stessa
config. attenzione: nel dict vanno sempre inclusi anche i campi di
architettura (hidden_channels, global_features_dim, critic_hidden_size,
architecture_name), altrimenti due trial che differiscono solo nella forma
della rete finiscono con lo stesso id
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Any


# turns a hyperparameter dict into a short deterministic id, used to group runs of the same config
def config_id(config: dict[str, Any]) -> str:
    """Stable short hash identifying a configuration.

    Callers must pass every field that makes two configs meaningfully
    different otherwise distinct configs collide under the same id.
    """
    # serializza il dict in json ordinato (cosi' l'ordine delle chiavi non cambia l'hash)
    # e ne fa uno sha1 troncato a 10 caratteri, solo per avere un id corto da leggere
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:10]


# chiave di ordinamento: solo il win rate. validation_mean_return non viene mai
# loggato durante screening/conferma (e' sempre 0.0), quindi non serve come
# tie-break: usarlo non cambierebbe mai l'ordine reale dei run
def score(record: dict[str, Any]) -> float:
    """Sort key for ranking screening/confirmation records: win rate."""
    return record["validation_win_rate"]


# picks the best `count` configs coming out of the screening sweep, one per config_id
def select_finalists(
    records: list[dict[str, Any]],
    algorithm: str,
    count: int,
) -> list[dict[str, Any]]:
    """Top `count` *unique* config_ids for `algorithm` among completed screening records.

    If the same config_id appears more than once (e.g. re-run after a
    resume), only its best-scoring record is kept before ranking, so the
    result always contains at most one record per config_id.
    """
    # filtro solo i record di screening completati per l'algoritmo che mi interessa
    candidates = [
        record for record in records
        if record.get("stage") == "screen"
        and record.get("algorithm") == algorithm
        and record.get("status") == "completed"
    ]
    # se un config_id compare piu' volte tengo solo il migliore, cosi' non conto due volte lo stesso
    best_by_config: dict[str, dict[str, Any]] = {}
    for record in candidates:
        identifier = record["config_id"]
        current_best = best_by_config.get(identifier)
        if current_best is None or score(record) > score(current_best):
            best_by_config[identifier] = record
    # ordino dal migliore al peggiore e prendo solo i primi "count"
    return sorted(best_by_config.values(), key=score, reverse=True)[:count]


# combines the multi-seed confirmation runs into one row per config with mean/std/CI, ready for reporting
def aggregate_confirmation(records: list[dict[str, Any]], algorithm: str) -> list[dict[str, Any]]:
    """Group completed confirmation records by config_id and compute mean/CI stats.

    Configs that did not complete every required confirmation seed are the
    caller's responsibility to filter out beforehand (or after, by checking
    `len(aggregate["records"])` against the expected seed count) -- this
    function only aggregates whatever completed records it is given.
    """
    # raggruppo tutti i run di conferma per config_id, cosi' ogni gruppo e' "stesso config, seed diversi"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if (
            record.get("stage") == "confirm"
            and record.get("algorithm") == algorithm
            and record.get("status") == "completed"
        ):
            grouped[record["config_id"]].append(record)

    aggregates = []
    for identifier, group in grouped.items():
        rates = [record["validation_win_rate"] for record in group]
        returns = [record["validation_mean_return"] for record in group]
        # deviazione standard e intervallo di confidenza al 95% (formula classica 1.96 * std / sqrt(n))
        # con un solo seed non ha senso calcolarla quindi resta 0
        standard_deviation = stdev(rates) if len(rates) > 1 else 0.0
        ci_half = 1.96 * standard_deviation / math.sqrt(len(rates)) if len(rates) > 1 else 0.0
        aggregates.append({
            "config_id": identifier,
            "records": group,
            "hyperparameters": group[0]["hyperparameters"],
            "mean_win_rate": mean(rates),
            "std_win_rate": standard_deviation,
            "ci_low": mean(rates) - ci_half,
            "ci_high": mean(rates) + ci_half,
            "mean_return": mean(returns),
        })
    # best config first, stesso criterio di score() sopra (solo win rate)
    return sorted(aggregates, key=lambda item: item["mean_win_rate"], reverse=True)
