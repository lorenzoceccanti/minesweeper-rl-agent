"""Per-architecture flat search spaces for W&B sweeps.

Each sweep targets exactly one (algorithm, task, architecture) triple, so its
W&B `parameters` dict must be flat: no parameter that is meaningless for the
chosen architecture should ever appear (e.g. `global_features_dim`, which
only affects the global_skip backbone -- see models/factory.py, where
fully_conv networks never receive it). Splitting sweeps by architecture
(instead of sampling architecture *inside* one sweep) is what keeps this
flat: this module only merges the campaign YAML's `common` search space with
the architecture-specific one and validates the split was done correctly.
"""

from __future__ import annotations

from typing import Any

KNOWN_ARCHITECTURES = frozenset({
    "fully_conv_3layer_64ch_11in",
    "global_skip_conv_3layer_64ch_11in",
})

# Parameters that only make sense for a specific architecture. hidden_channels
# and critic_hidden_size are deliberately absent here: per models/factory.py
# both apply to every known architecture (critic_hidden_size sizes the critic
# MLP head regardless of backbone), so they belong under `common`.
ARCHITECTURE_RESTRICTED_PARAMS: dict[str, frozenset[str]] = {
    "global_skip_conv_3layer_64ch_11in": frozenset({"global_features_dim"}),
}


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

    algo_space = search_space.get(algorithm, {})
    common = dict(algo_space.get("common", {}))

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

    arch_specific = dict(algo_space.get(architecture_name, {}))
    overlap = common.keys() & arch_specific.keys()
    if overlap:
        raise ValueError(
            f"search_space.{algorithm} keys {sorted(overlap)} declared in both "
            f"'common' and {architecture_name!r}"
        )

    merged = {**common, **arch_specific}
    if not merged:
        raise ValueError(
            f"empty search space for algorithm={algorithm!r} architecture={architecture_name!r}"
        )
    return merged
