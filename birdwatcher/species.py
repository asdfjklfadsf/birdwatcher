"""Species-name canonicalization shared across classification stages.

This module is the single source of truth for species aliases. The broad
classifier ships labels that are misspelled or spelled differently from the
regional checklists (for example ``Tit Mouse`` for ``Tufted Titmouse``). Every
stage that compares species -- regional priors, classifier blending, and the
final plausibility gate -- must agree on one identity per species, otherwise
evidence for one bird is split across two labels or a correct identification is
rejected as out of region.
"""
from __future__ import annotations

import re
from collections import defaultdict


def normalize_species_key(name: str) -> str:
    """Return a case- and punctuation-insensitive key for one species label."""
    return re.sub(r"[^a-z0-9]", "", name.casefold())


# Broad-classifier label -> preferred display name. Add new aliases here only;
# every direction of the mapping is derived from this one table.
_SPECIES_ALIASES = {
    "Tit Mouse": "Tufted Titmouse",
}

_CANONICAL_BY_KEY = {
    normalize_species_key(source): canonical for source, canonical in _SPECIES_ALIASES.items()
}
_CANONICAL_BY_KEY.update(
    {normalize_species_key(canonical): canonical for canonical in _SPECIES_ALIASES.values()}
)


def canonical_species_name(name: str) -> str:
    """Return the preferred display name for a classifier species label."""
    cleaned = " ".join(name.strip().split())
    return _CANONICAL_BY_KEY.get(normalize_species_key(cleaned), cleaned)


def canonical_species_key(name: str) -> str:
    """Return a stable identity key shared by every alias of the same species."""
    return normalize_species_key(canonical_species_name(name))


def canonical_species_keys(names) -> set[str]:
    """Return the canonical identity keys for an iterable of species labels."""
    return {canonical_species_key(name) for name in names}


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
