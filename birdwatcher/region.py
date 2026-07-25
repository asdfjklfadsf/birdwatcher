"""Regional species priors and multi-frame identification resolution."""
from __future__ import annotations

import math
from collections import Counter, defaultdict

from .domain import IdentificationResult
from .species import (
    canonical_species_key,
    canonical_species_keys,
    collapse_species_aliases,
    normalize_species_key,
)


def season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    if month in {9, 10, 11}:
        return "fall"
    raise ValueError("month must be between 1 and 12")


def species_key(name: str) -> str:
    """Normalize one label without resolving aliases.

    Prefer :func:`birdwatcher.species.canonical_species_key` when comparing two
    species: this helper keeps aliases distinct.
    """
    return normalize_species_key(name)


_NORTHERN_NJ_RESIDENTS = {
    "American Goldfinch", "American Robin", "Black-Capped Chickadee", "Blue Jay",
    "Carolina Wren", "Cedar Waxwing", "Common Starling", "Downy Woodpecker",
    "Eastern Bluebird", "Hairy Woodpecker", "House Finch", "House Sparrow",
    "Mourning Dove", "Northern Cardinal", "Northern Flicker", "Northern Mockingbird",
    "Red-Bellied Woodpecker", "Red-Shouldered Hawk", "Red-Tailed Hawk", "Rock Dove",
    "Song Sparrow", "Tufted Titmouse", "Turkey Vulture", "White-Breasted Nuthatch",
    "Wild Turkey",
}

_NORTHERN_NJ_SEASONAL = {
    "winter": {
        "American Tree Sparrow", "Dark-Eyed Junco", "Evening Grosbeak", "Pine Siskin",
        "Purple Finch", "Red-Breasted Nuthatch", "Ruby-Crowned Kinglet", "White-Throated Sparrow",
    },
    "spring": {
        "Baltimore Oriole", "Barn Swallow", "Brown-Headed Cowbird", "Chipping Sparrow",
        "Common Grackle", "Gray Catbird", "Rose-Breasted Grosbeak", "Ruby-Throated Hummingbird",
        "Scarlet Tanager", "Tree Swallow",
    },
    "summer": {
        "Baltimore Oriole", "Barn Swallow", "Blue Grosbeak", "Brown-Headed Cowbird",
        "Chipping Sparrow", "Common Grackle", "Gray Catbird", "Rose-Breasted Grosbeak",
        "Ruby-Throated Hummingbird", "Scarlet Tanager", "Tree Swallow",
    },
    "fall": {
        "Brown-Headed Cowbird", "Common Grackle", "Dark-Eyed Junco", "Gray Catbird",
        "Purple Finch", "Ruby-Crowned Kinglet", "White-Throated Sparrow",
    },
}

# Labels from the broad classifier that intersect the New Jersey all-years checklist.
# This is a plausibility boundary, not a statement that every listed species is common.
_NEW_JERSEY_MODEL_SPECIES = frozenset(
    name.strip()
    for name in """
American Avocet
American Bittern
American Coot
American Flamingo
American Goldfinch
American Kestrel
American Pipit
American Redstart
American Robin
American Wigeon
Anhinga
Bald Eagle
Baltimore Oriole
Bar-Tailed Godwit
Barn Swallow
Barrows Goldeneye
Bay-Breasted Warbler
Belted Kingfisher
Black Necked Stilt
Black Skimmer
Black Swan
Black Throated Warbler
Black Vulture
Black-Capped Chickadee
Black-Necked Grebe
Black-Throated Sparrow
Blackburniam Warbler
Blue Gray Gnatcatcher
Blue Grosbeak
Blue Heron
Bobolink
Brewers Blackbird
Brown Crepper
Brown Headed Cowbird
Brown Thrasher
Bufflehead
California Gull
California Quail
Canvasback
Cape May Warbler
Caspian Tern
Cedar Waxwing
Cerulean Warbler
Chipping Sparrow
Cinnamon Teal
Common Grackle
Common Loon
Common Starling
Crested Caracara
Dark Eyed Junco
Downy Woodpecker
Dunlin
Eastern Bluebird
Eastern Meadowlark
Eastern Towee
Egyptian Goose
European Goldfinch
Evening Grosbeak
Glossy Ibis
Gold Wing Warbler
Golden Eagle
Golden Pheasant
Gray Catbird
Gray Kingbird
Gray Partridge
Grey Plover
Gyrfalcon
Harlequin Duck
Hooded Merganser
Horned Lark
House Finch
House Sparrow
Indigo Bunting
Ivory Gull
Java Sparrow
Killdear
King Eider
Lark Bunting
Laughing Gull
Lazuli Bunting
Limpkin
Loggerhead Shrike
Long-Eared Owl
Mallard Duck
Masked Booby
Merlin
Mourning Dove
Northern Cardinal
Northern Flicker
Northern Fulmar
Northern Gannet
Northern Mockingbird
Northern Parula
Northern Red Bishop
Northern Shoveler
Osprey
Ovenbird
Oyster Catcher
Painted Bunting
Peregrine Falcon
Pomarine Jaeger
Purple Finch
Purple Gallinule
Purple Martin
Razorbill
Red Billed Tropicbird
Red Crossbill
Red Headed Woodpecker
Red Knot
Red Shouldered Hawk
Red Tailed Hawk
Red Winged Blackbird
Ring-Necked Pheasant
Rose Breasted Grosbeak
Roseate Spoonbill
Rosy Faced Lovebird
Rough Leg Buzzard
Ruby Crowned Kinglet
Ruby Throated Hummingbird
Ruddy Shelduck
Sandhill Crane
Says Phoebe
Scarlet Ibis
Scarlet Tanager
Short Billed Dowitcher
Smiths Longspur
Snow Goose
Snowy Egret
Snowy Owl
Sora
Surf Scoter
Tit Mouse
Townsends Warbler
Tree Swallow
Tropical Kingbird
Trumpter Swan
Turkey Vulture
Varied Thrush
Veery
Violet Green Swallow
White Necked Raven
Wild Turkey
Wood Duck
Wood Thrush
Yellow Breasted Chat
Yellow Headed Blackbird
""".splitlines()
    if name.strip()
)


def _validate_region_profile(profile: str) -> str:
    normalized = profile.strip().casefold().replace("-", "_")
    if normalized != "northern_nj":
        raise ValueError(f"Unsupported REGION_PROFILE: {profile}")
    return normalized


def preferred_regional_species_names(profile: str, month: int) -> set[str]:
    _validate_region_profile(profile)
    return _NORTHERN_NJ_RESIDENTS | _NORTHERN_NJ_SEASONAL[season_for_month(month)]


def preferred_regional_species(profile: str, month: int) -> set[str]:
    """Canonical keys for locally preferred species the broad model can name."""
    preferred = canonical_species_keys(preferred_regional_species_names(profile, month))
    return preferred & canonical_species_keys(_NEW_JERSEY_MODEL_SPECIES)


def regional_species(profile: str, month: int) -> set[str]:
    _validate_region_profile(profile)
    season_for_month(month)
    return canonical_species_keys(_NEW_JERSEY_MODEL_SPECIES)


def identification_plausible_species(profile: str, month: int) -> set[str]:
    return regional_species(profile, month) | canonical_species_keys(
        preferred_regional_species_names(profile, month)
    )


def apply_regional_prior(
    predictions: list[tuple[str, float]],
    preferred_species: set[str],
    weight: float,
    plausible_species: set[str] | None = None,
    plausible_weight: float = 1.5,
) -> list[tuple[str, float]]:
    if not predictions:
        raise ValueError("Predictions cannot be empty")
    if not math.isfinite(weight) or weight < 1:
        raise ValueError("REGIONAL_PRIOR_WEIGHT must be finite and at least 1")
    if not math.isfinite(plausible_weight) or not 1 <= plausible_weight <= weight:
        raise ValueError("Broad regional weight must be between 1 and the preferred weight")

    plausible_species = plausible_species or set()
    weighted = []
    for label, score in predictions:
        key = canonical_species_key(label)
        factor = weight if key in preferred_species else plausible_weight if key in plausible_species else 1.0
        weighted.append((label, score * factor))

    total = sum(score for _, score in weighted)
    if total <= 0:
        raise ValueError("Prediction scores must have a positive sum")
    return sorted(((label, score / total) for label, score in weighted), key=lambda item: -item[1])


def combine_classifier_predictions(
    local_predictions: list[tuple[str, float]],
    global_predictions: list[tuple[str, float]],
    local_weight: float,
) -> list[tuple[str, float]]:
    """Blend BioCLIP and broad-classifier scores under one identity per species.

    Aliases are collapsed first so that the two classifiers, which spell some
    species differently, contribute to the same candidate instead of splitting
    one bird's evidence across two labels.
    """
    if not local_predictions or not global_predictions:
        raise ValueError("Both classifier prediction lists must be nonempty")
    if not math.isfinite(local_weight) or not 0 <= local_weight <= 1:
        raise ValueError("LOCAL_CLASSIFIER_WEIGHT must be between 0 and 1")

    scores: defaultdict[str, float] = defaultdict(float)
    labels: dict[str, str] = {}
    for predictions, weight in (
        (collapse_species_aliases(global_predictions), 1.0 - local_weight),
        (collapse_species_aliases(local_predictions), local_weight),
    ):
        for label, score in predictions:
            key = canonical_species_key(label)
            labels[key] = label
            scores[key] += weight * score

    total = sum(scores.values())
    if total <= 0:
        raise ValueError("Combined classifier scores must have a positive sum")
    return sorted(
        ((labels[key], score / total) for key, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def resolve_identification(
    frame_predictions: list[list[tuple[str, float]]],
    min_votes: int,
    min_confidence: float,
    min_margin: float,
    plausible_species: set[str] | None = None,
) -> IdentificationResult:
    if not frame_predictions or any(not predictions for predictions in frame_predictions):
        raise ValueError("At least one nonempty prediction list is required")

    votes = Counter(predictions[0][0] for predictions in frame_predictions)
    aggregate_scores: defaultdict[str, float] = defaultdict(float)
    for predictions in frame_predictions:
        for label, score in predictions:
            aggregate_scores[label] += score

    frame_count = len(frame_predictions)
    ranked_scores = sorted(aggregate_scores.items(), key=lambda item: (-item[1], item[0]))
    candidate_name, candidate_total = ranked_scores[0]
    confidence = candidate_total / frame_count
    runner_up_confidence = ranked_scores[1][1] / frame_count if len(ranked_scores) > 1 else 0.0
    margin = confidence - runner_up_confidence
    vote_count = votes[candidate_name]
    # Winners arrive canonicalized, so the plausibility set must be compared on
    # canonical keys too; matching raw labels here would reject any aliased
    # species whose preferred spelling is absent from the regional checklists.
    out_of_region = (
        bool(plausible_species) and canonical_species_key(candidate_name) not in plausible_species
    )
    uncertain = (
        vote_count < min_votes
        or confidence < min_confidence
        or margin < min_margin
        or out_of_region
    )
    display_name = candidate_name if not uncertain else "Uncertain bird"
    top_candidates = tuple((label, total / frame_count) for label, total in ranked_scores[:3])
    return IdentificationResult(
        display_name=display_name,
        candidate_name=candidate_name,
        confidence=confidence,
        margin=margin,
        votes=vote_count,
        frame_count=frame_count,
        uncertain=uncertain,
        top_candidates=top_candidates,
    )
