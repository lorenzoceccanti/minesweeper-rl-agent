"""qui si costruisce lo spazio di ricerca per ogni combinazione algoritmo+architettura

ogni sweep di wandb punta a una sola architettura, quindi i parametri devono
essere "piatti": non deve mai comparire un parametro che non ha senso per
quell'architettura (tipo global_features_dim, che serve solo alla rete con
skip connection). questo modulo si occupa solo di unire i parametri "common"
dello yaml con quelli specifici dell'architettura e controllare che non ci
siano errori nella divisione
"""

from __future__ import annotations

from typing import Any

# le due architetture di rete neurale che il progetto sa costruire (vedi models/factory.py)
KNOWN_ARCHITECTURES = frozenset({
    "fully_conv_3layer_64ch_11in",
    "global_skip_conv_3layer_64ch_11in",
})

# these keys can only be set for the architecture listed, not shared across all archs
ARCHITECTURE_RESTRICTED_PARAMS: dict[str, frozenset[str]] = {
    "global_skip_conv_3layer_64ch_11in": frozenset({"global_features_dim"}),
}


# generatore semplice: scorre il dict {algoritmo: [lista architetture]} e tira fuori tutte le coppie
def iter_algorithm_architecture_pairs(architectures: dict[str, list[str]]):
    """Yield (algorithm, architecture) for every combination declared in a campaign."""
    for algorithm, archs in architectures.items():
        for architecture_name in archs:
            if architecture_name not in KNOWN_ARCHITECTURES:
                raise ValueError(
                    f"Unknown architecture {architecture_name!r} for algorithm {algorithm!r}; "
                    f"known architectures: {sorted(KNOWN_ARCHITECTURES)}"
                )
            yield algorithm, architecture_name


# main function of this file, merges common + architecture-specific hyperparams into one flat dict
def build_parameter_space(
    algorithm: str,
    architecture_name: str,
    search_space: dict[str, Any],
) -> dict[str, Any]:
    """Merge `common` and architecture-specific search space into one flat W&B `parameters` dict.

    Raises ValueError if the campaign YAML misconfigures the split (an
    architecture-restricted param placed under `common`, or the same key
    declared in both `common` and the architecture-specific section) so
    mistakes fail at `validate` time instead of silently wasting sweep trials.
    """
    if architecture_name not in KNOWN_ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture_name!r}")

    # prendo i parametri comuni a tutte le architetture per questo algoritmo
    algo_space = search_space.get(algorithm, {})
    common = dict(algo_space.get("common", {}))

    # controllo che nessun parametro "riservato" ad altre architetture sia finito per sbaglio in common
    for other_architecture, restricted_keys in ARCHITECTURE_RESTRICTED_PARAMS.items():
        if other_architecture == architecture_name:
            continue
        leaked = restricted_keys & common.keys()
        if leaked:
            raise ValueError(
                f"search_space.{algorithm}.common must not include {sorted(leaked)} "
                f"(restricted to architecture {other_architecture!r}); move under "
                f"search_space.{algorithm}.{other_architecture}"
            )

    # e aggiungo i parametri specifici solo di questa architettura
    arch_specific = dict(algo_space.get(architecture_name, {}))
    overlap = common.keys() & arch_specific.keys()
    if overlap:
        raise ValueError(
            f"search_space.{algorithm} keys {sorted(overlap)} declared in both "
            f"'common' and {architecture_name!r}"
        )

    # unisco i due dizionari, questo e' quello che poi diventa il "parameters" di wandb.sweep
    merged = {**common, **arch_specific}
    if not merged:
        raise ValueError(
            f"empty search space for algorithm={algorithm!r} architecture={architecture_name!r}"
        )
    return merged
