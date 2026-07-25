"""BioCLIP and broad-classifier orchestration for Bird Watcher."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable

import cv2

from .constants import BROAD_REGIONAL_PRIOR_WEIGHT, TEXT_FEATURE_CACHE_SIZE
from .region import apply_regional_prior, combine_classifier_predictions
from .species import canonical_species_name
from .tracking import TrackedDetection


@dataclass(frozen=True)
class EncodedCrop:
    detection: TrackedDetection
    embedding: Any
    presence_score: float


def encode_bioclip_image(models, bird_image):
    from PIL import Image

    rgb_image = cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB)
    tensor = models.local_preprocess(Image.fromarray(rgb_image)).unsqueeze(0).to(models.device)
    with models.torch.inference_mode():
        features = models.local_classifier.encode_image(tensor)
        return features / features.norm(dim=-1, keepdim=True)


def _species_text_features(models, names: tuple[str, ...]):
    # The candidate set includes the broad model's top-k, so it changes almost
    # every frame. Bound the cache or a long-running watcher grows without limit.
    cache = getattr(models, "_runtime_text_cache", None)
    if cache is None:
        cache = OrderedDict()
        models._runtime_text_cache = cache
    if names in cache:
        cache.move_to_end(names)
        return cache[names]

    templates = (
        "a photo of a {}.",
        "a cropped photo of a {}.",
        "a close-up photo of a {}.",
        "a photo of the {}.",
        "a blurry photo of a {}.",
    )
    prompts = [template.format(name) for name in names for template in templates]
    with models.torch.inference_mode():
        features = models.local_classifier.encode_text(models.local_tokenizer(prompts).to(models.device))
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.reshape(len(names), len(templates), -1).mean(dim=1)
        features = features / features.norm(dim=-1, keepdim=True)
        cache[names] = features.T.contiguous()
    while len(cache) > TEXT_FEATURE_CACHE_SIZE:
        cache.popitem(last=False)
    return cache[names]


def bird_presence_from_embedding(models, embedding) -> float:
    features = getattr(models, "_runtime_presence_text_features", None)
    if features is None:
        groups = (
            (
                "a photograph of a real bird at a bird feeder",
                "a real bird with feathers, head, beak, body and legs",
                "a perched wild bird",
                "a small bird eating seeds",
            ),
            (
                "an empty bird feeder with no bird",
                "a clear plastic feeder dish and seeds",
                "bird feeder hardware with no animal",
                "an empty feeding tray, foliage and reflections",
            ),
        )
        prompts = [prompt for group in groups for prompt in group]
        with models.torch.inference_mode():
            features = models.local_classifier.encode_text(models.local_tokenizer(prompts).to(models.device))
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.reshape(2, len(groups[0]), -1).mean(dim=1)
            features = features / features.norm(dim=-1, keepdim=True)
            features = features.T.contiguous()
        models._runtime_presence_text_features = features
    with models.torch.inference_mode():
        probabilities = models.torch.softmax(100.0 * embedding @ features, dim=-1)[0]
    return float(probabilities[0])


def local_predictions_from_embedding(models, embedding, species_names: set[str]):
    names = tuple(sorted(species_names))
    if not names:
        raise ValueError("BioCLIP requires at least one candidate species")
    features = _species_text_features(models, names)
    with models.torch.inference_mode():
        logits = models.local_classifier.logit_scale.exp() * embedding @ features
        probabilities = models.torch.softmax(logits, dim=-1)[0]
    return sorted(
        ((name, float(probabilities[index])) for index, name in enumerate(names)),
        key=lambda item: -item[1],
    )


def candidate_species_names(
    preferred_names: set[str], raw_global_predictions: list[tuple[str, float]]
) -> set[str]:
    return {canonical_species_name(name) for name in preferred_names} | {
        canonical_species_name(name) for name, _ in raw_global_predictions
    }


def hybrid_predictions(
    models,
    bird_image,
    embedding,
    preferred_names: set[str],
    preferred_keys: set[str],
    broad_keys: set[str],
    settings,
):
    raw_global = models.identify_species_candidates(bird_image, top_k=20)
    global_predictions = apply_regional_prior(
        raw_global,
        preferred_keys,
        settings.regional_prior_weight,
        plausible_species=broad_keys,
        # Never exceed the preferred weight: REGIONAL_PRIOR_WEIGHT=1.0 means
        # "no regional preference at all", which must stay expressible.
        plausible_weight=min(BROAD_REGIONAL_PRIOR_WEIGHT, settings.regional_prior_weight),
    )
    local_predictions = local_predictions_from_embedding(
        models,
        embedding,
        candidate_species_names(preferred_names, raw_global),
    )
    return combine_classifier_predictions(
        local_predictions,
        global_predictions,
        settings.local_classifier_weight,
    )


def encode_accepted_crops(
    detections: list[TrackedDetection],
    models,
    min_presence_score: float,
    *,
    encoder: Callable | None = None,
    presence_scorer: Callable | None = None,
) -> list[EncodedCrop]:
    encoder = encoder or (lambda image: encode_bioclip_image(models, image))
    presence_scorer = presence_scorer or (
        lambda embedding: bird_presence_from_embedding(models, embedding)
    )
    encoded: list[EncodedCrop] = []
    for detection in detections:
        embedding = encoder(detection.crop)
        presence_score = float(presence_scorer(embedding))
        if presence_score >= min_presence_score:
            encoded.append(EncodedCrop(detection, embedding, presence_score))
    return encoded
