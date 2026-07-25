"""Species-name canonicalization shared across classification stages."""
from __future__ import annotations

import re
from collections import defaultdict


def _normalized_species_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.casefold())


_CANONICAL_SPECIES_NAMES = {
    _normalized_species_key("Tit Mouse"): "Tufted Titmouse",
    _normalized_species_key("Tufted Titmouse"): "Tufted Titmouse",
}


def canonical_species_name(name: str) -> str:
    """Return the preferred display name for a classifier species label."""
    cleaned = " ".join(name.strip().split())
    return _CANONICAL_SPECIES_NAMES.get(_normalized_species_key(cleaned), cleaned)


def canonical_species_key(name: str) -> str:
    """Return a stable identity key shared by aliases of the same species."""
    return _normalized_species_key(canonical_species_name(name))


def collapse_species_aliases(
    predictions: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Merge prediction scores that refer to the same canonical species."""
    scores: defaultdict[str, float] = defaultdict(float)
    labels: dict[str, str] = {}

    for label, score in predictions:
        canonical_name = canonical_species_name(label)
        key = canonical_species_key(canonical_name)
        labels[key] = canonical_name
        scores[key] += score

    return sorted(
        ((labels[key], score) for key, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )
