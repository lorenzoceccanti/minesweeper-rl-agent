"""Identity, ranking and aggregation logic for hyperparameter/architecture search.

Extracted and revalidated from ``tmp/random_search.py`` (config_id, score,
select_finalists, aggregate_confirmation). The original script only ever
hashed training hyperparameters (learning rate, gamma, batch size, ...) since
it predates the network-architecture parametrization (hidden_channels,
global_features_dim, critic_hidden_size, architecture_name). ``config_id``
itself is generic and unchanged: the fix is that every caller in this new
pipeline must include the architecture fields in the dict it hashes, so two
trials that only differ in network shape never collide under the same id.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from statistics import mean, stdev
from typing import Any


def config_id(config: dict[str, Any]) -> str:
    """Stable short hash identifying a configuration.

    Callers must pass every field that makes two configs meaningfully
    different -- training hyperparameters AND architecture_name /
    hidden_channels / global_features_dim / critic_hidden_size -- otherwise
    distinct configs collide under the same id.
    """
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:10]


def score(record: dict[str, Any]) -> tuple[float, float]:
    """Sort key for ranking screening/confirmation records: win rate, then mean return."""
    return record["validation_win_rate"], record["validation_mean_return"]


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
    candidates = [
        record for record in records
        if record.get("stage") == "screen"
        and record.get("algorithm") == algorithm
        and record.get("status") == "completed"
    ]
    best_by_config: dict[str, dict[str, Any]] = {}
    for record in candidates:
        identifier = record["config_id"]
        current_best = best_by_config.get(identifier)
        if current_best is None or score(record) > score(current_best):
            best_by_config[identifier] = record
    return sorted(best_by_config.values(), key=score, reverse=True)[:count]


def aggregate_confirmation(records: list[dict[str, Any]], algorithm: str) -> list[dict[str, Any]]:
    """Group completed confirmation records by config_id and compute mean/CI stats.

    Configs that did not complete every required confirmation seed are the
    caller's responsibility to filter out beforehand (or after, by checking
    `len(aggregate["records"])` against the expected seed count) -- this
    function only aggregates whatever completed records it is given.
    """
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
    return sorted(aggregates, key=lambda item: (item["mean_win_rate"], item["mean_return"]), reverse=True)
