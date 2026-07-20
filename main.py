"""Continuously detect birds, classify their species, save images, and email alerts."""

from __future__ import annotations

import logging
import math
import os
import re
import smtplib
import ssl
import time
import hashlib
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.message import EmailMessage
from pathlib import Path
from statistics import median
from typing import Any

import cv2
from dotenv import load_dotenv

LOG = logging.getLogger("bird_watcher")


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
    return re.sub(r"[^a-z0-9]", "", name.casefold())


_NORTHERN_NJ_RESIDENTS = {
    "American Goldfinch",
    "American Robin",
    "Black-Capped Chickadee",
    "Blue Jay",
    "Carolina Wren",
    "Cedar Waxwing",
    "Common Starling",
    "Downy Woodpecker",
    "Eastern Bluebird",
    "Hairy Woodpecker",
    "House Finch",
    "House Sparrow",
    "Mourning Dove",
    "Northern Cardinal",
    "Northern Flicker",
    "Northern Mockingbird",
    "Red-Bellied Woodpecker",
    "Red-Shouldered Hawk",
    "Red-Tailed Hawk",
    "Rock Dove",
    "Song Sparrow",
    "Tufted Titmouse",
    "Turkey Vulture",
    "White-Breasted Nuthatch",
    "Wild Turkey",
}
_NORTHERN_NJ_SEASONAL = {
    "winter": {
        "American Tree Sparrow",
        "Dark-Eyed Junco",
        "Evening Grosbeak",
        "Pine Siskin",
        "Purple Finch",
        "Red-Breasted Nuthatch",
        "Ruby-Crowned Kinglet",
        "White-Throated Sparrow",
    },
    "spring": {
        "Baltimore Oriole",
        "Barn Swallow",
        "Brown-Headed Cowbird",
        "Chipping Sparrow",
        "Common Grackle",
        "Gray Catbird",
        "Rose-Breasted Grosbeak",
        "Ruby-Throated Hummingbird",
        "Scarlet Tanager",
        "Tree Swallow",
    },
    "summer": {
        "Baltimore Oriole",
        "Barn Swallow",
        "Blue Grosbeak",
        "Brown-Headed Cowbird",
        "Chipping Sparrow",
        "Common Grackle",
        "Gray Catbird",
        "Rose-Breasted Grosbeak",
        "Ruby-Throated Hummingbird",
        "Scarlet Tanager",
        "Tree Swallow",
    },
    "fall": {
        "Brown-Headed Cowbird",
        "Common Grackle",
        "Dark-Eyed Junco",
        "Gray Catbird",
        "Purple Finch",
        "Ruby-Crowned Kinglet",
        "White-Throated Sparrow",
    },
}


# Model labels that intersect the eBird New Jersey all-years checklist. This broad
# state-level set is a safety boundary, not a claim that every species is common.
_NEW_JERSEY_MODEL_SPECIES = frozenset(
    name.strip()
    for name in
    """
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
    normalized_profile = profile.strip().casefold().replace("-", "_")
    if normalized_profile != "northern_nj":
        raise ValueError(f"Unsupported REGION_PROFILE: {profile}")
    return normalized_profile


def preferred_regional_species_names(profile: str, month: int) -> set[str]:
    """Return human-readable common/resident species for local visual retrieval."""
    _validate_region_profile(profile)
    return _NORTHERN_NJ_RESIDENTS | _NORTHERN_NJ_SEASONAL[season_for_month(month)]


def preferred_regional_species(profile: str, month: int) -> set[str]:
    """Return supported common/resident labels that receive the seasonal boost."""
    names = preferred_regional_species_names(profile, month)
    aliases = {species_key("Tufted Titmouse"): species_key("Tit Mouse")}
    preferred = {aliases.get(species_key(name), species_key(name)) for name in names}
    supported = {species_key(name) for name in _NEW_JERSEY_MODEL_SPECIES}
    return preferred & supported


def regional_species(profile: str, month: int) -> set[str]:
    """Return broad NJ-documented legacy-model labels used for regional evidence."""
    _validate_region_profile(profile)
    season_for_month(month)
    return {species_key(name) for name in _NEW_JERSEY_MODEL_SPECIES}


def identification_plausible_species(profile: str, month: int) -> set[str]:
    """Return the union of broad legacy labels and valid local BioCLIP names."""
    return regional_species(profile, month) | {
        species_key(name) for name in preferred_regional_species_names(profile, month)
    }


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
        key = species_key(label)
        if key in preferred_species:
            factor = weight
        elif key in plausible_species:
            factor = plausible_weight
        else:
            factor = 1.0
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
    """Blend local-feeder and broad classifier evidence without deleting either."""
    if not local_predictions or not global_predictions:
        raise ValueError("Both classifier prediction lists must be nonempty")
    if not math.isfinite(local_weight) or not 0 <= local_weight <= 1:
        raise ValueError("LOCAL_CLASSIFIER_WEIGHT must be between 0 and 1")

    scores: defaultdict[str, float] = defaultdict(float)
    labels: dict[str, str] = {}
    for predictions, weight in (
        (global_predictions, 1.0 - local_weight),
        (local_predictions, local_weight),
    ):
        for label, score in predictions:
            key = species_key(label)
            labels[key] = label
            scores[key] += weight * score

    total = sum(scores.values())
    if total <= 0:
        raise ValueError("Combined classifier scores must have a positive sum")
    return sorted(
        ((labels[key], score / total) for key, score in scores.items()),
        key=lambda item: (-item[1], item[0]),
    )


def hybrid_species_predictions(
    models,
    bird_image,
    preferred_names: set[str],
    preferred_keys: set[str],
    plausible_keys: set[str],
    regional_weight: float,
    local_weight: float,
) -> list[tuple[str, float]]:
    """Blend BioCLIP local evidence with the broad 525-label classifier."""
    global_predictions = apply_regional_prior(
        models.identify_species_candidates(bird_image, top_k=20),
        preferred_keys,
        regional_weight,
        plausible_species=plausible_keys,
        plausible_weight=1.5,
    )
    local_predictions = models.identify_local_species_candidates(
        bird_image,
        preferred_names,
        top_k=len(preferred_names),
    )
    return combine_classifier_predictions(
        local_predictions,
        global_predictions,
        local_weight,
    )


@dataclass(frozen=True)
class EmailSettings:
    host: str
    port: int
    username: str
    password: str
    sender: str
    recipient: str
    use_ssl: bool
    use_starttls: bool
    allow_insecure: bool


@dataclass(frozen=True)
class AppSettings:
    camera: int | str
    camera_width: int
    camera_height: int
    camera_fps: float
    image_dir: Path
    cooldown: timedelta
    scan_interval: float
    detection_confidence: float
    crop_padding: float
    burst_frames: int
    burst_frame_interval: float
    sharpest_frames: int
    min_valid_bird_frames: int
    min_event_detector_confidence: float
    detection_floor_confidence: float
    max_bird_crop_aspect_ratio: float
    min_bird_presence_score: float
    min_bird_presence_frames: int
    consensus_min_votes: int
    species_min_confidence: float
    species_min_margin: float
    region_profile: str
    regional_prior_weight: float
    email: EmailSettings
    detector_model: str
    detector_sha256: str
    classifier_model: str
    classifier_revision: str
    local_classifier_model: str
    local_classifier_revision: str
    local_classifier_weight: float


@dataclass(frozen=True)
class IdentificationResult:
    display_name: str
    candidate_name: str
    confidence: float
    margin: float
    votes: int
    frame_count: int
    uncertain: bool
    top_candidates: tuple[tuple[str, float], ...]


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
    ranked_scores = sorted(
        aggregate_scores.items(), key=lambda item: (-item[1], item[0])
    )
    candidate_name, candidate_total = ranked_scores[0]
    confidence = candidate_total / frame_count
    runner_up_confidence = ranked_scores[1][1] / frame_count if len(ranked_scores) > 1 else 0.0
    margin = confidence - runner_up_confidence
    vote_count = votes[candidate_name]
    out_of_region = bool(plausible_species) and species_key(candidate_name) not in plausible_species
    uncertain = (
        vote_count < min_votes
        or confidence < min_confidence
        or margin < min_margin
        or out_of_region
    )
    display_name = candidate_name if not uncertain else "Uncertain bird"
    top_candidates = tuple(
        (label, total / frame_count) for label, total in ranked_scores[:3]
    )
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


def image_sharpness(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_sharpest_crops(detections: list[tuple[object, float]], count: int):
    if count <= 0:
        raise ValueError("Sharpest crop count must be positive")
    return sorted(detections, key=lambda item: image_sharpness(item[0]), reverse=True)[:count]


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, not {value!r}")


def validate_settings(settings: AppSettings) -> None:
    if settings.camera_width <= 0 or settings.camera_height <= 0:
        raise ValueError("CAMERA_WIDTH and CAMERA_HEIGHT must be positive")
    if not math.isfinite(settings.camera_fps) or settings.camera_fps <= 0:
        raise ValueError("CAMERA_FPS must be finite and greater than zero")
    if not 1 <= settings.email.port <= 65535:
        raise ValueError("SMTP_PORT must be between 1 and 65535")
    if settings.cooldown <= timedelta(0):
        raise ValueError("COOLDOWN_MINUTES must be greater than zero")
    if not math.isfinite(settings.scan_interval) or settings.scan_interval < 0:
        raise ValueError("SCAN_INTERVAL_SECONDS must be finite and nonnegative")
    if not math.isfinite(settings.detection_confidence) or not 0 < settings.detection_confidence <= 1:
        raise ValueError("DETECTION_CONFIDENCE must be greater than 0 and at most 1")
    if not math.isfinite(settings.crop_padding) or not 0 <= settings.crop_padding <= 1:
        raise ValueError("DETECTION_CROP_PADDING must be between 0 and 1")
    if settings.burst_frames < 1:
        raise ValueError("BURST_FRAMES must be at least 1")
    if not math.isfinite(settings.burst_frame_interval) or settings.burst_frame_interval < 0:
        raise ValueError("BURST_FRAME_INTERVAL_SECONDS must be finite and nonnegative")
    if not 1 <= settings.sharpest_frames <= settings.burst_frames:
        raise ValueError("SHARPEST_FRAMES must be between 1 and BURST_FRAMES")
    if not 1 <= settings.min_valid_bird_frames <= settings.burst_frames:
        raise ValueError("MIN_VALID_BIRD_FRAMES must be between 1 and BURST_FRAMES")
    if not 0 < settings.min_event_detector_confidence <= 1:
        raise ValueError("MIN_EVENT_DETECTOR_CONFIDENCE must be greater than 0 and at most 1")
    if not 0 < settings.detection_floor_confidence <= 1:
        raise ValueError("DETECTION_FLOOR_CONFIDENCE must be greater than 0 and at most 1")
    if settings.detection_floor_confidence > settings.min_event_detector_confidence:
        raise ValueError(
            "DETECTION_FLOOR_CONFIDENCE must not exceed MIN_EVENT_DETECTOR_CONFIDENCE"
        )
    if (
        not math.isfinite(settings.max_bird_crop_aspect_ratio)
        or settings.max_bird_crop_aspect_ratio < 1
    ):
        raise ValueError("MAX_BIRD_CROP_ASPECT_RATIO must be finite and at least 1")
    if not 0 <= settings.min_bird_presence_score <= 1:
        raise ValueError("MIN_BIRD_PRESENCE_SCORE must be between 0 and 1")
    if not 1 <= settings.min_bird_presence_frames <= settings.min_valid_bird_frames:
        raise ValueError(
            "MIN_BIRD_PRESENCE_FRAMES must be between 1 and MIN_VALID_BIRD_FRAMES"
        )
    if not 1 <= settings.consensus_min_votes <= settings.sharpest_frames:
        raise ValueError("CONSENSUS_MIN_VOTES must be between 1 and SHARPEST_FRAMES")
    if not 0 <= settings.species_min_confidence <= 1:
        raise ValueError("SPECIES_MIN_CONFIDENCE must be between 0 and 1")
    if not 0 <= settings.species_min_margin <= 1:
        raise ValueError("SPECIES_MIN_MARGIN must be between 0 and 1")
    if not math.isfinite(settings.regional_prior_weight) or settings.regional_prior_weight < 1:
        raise ValueError("REGIONAL_PRIOR_WEIGHT must be finite and at least 1")
    regional_species(settings.region_profile, 7)
    if settings.email.use_ssl and settings.email.use_starttls:
        raise ValueError("Enable only one of SMTP_USE_SSL and SMTP_USE_STARTTLS")
    if not settings.email.use_ssl and not settings.email.use_starttls and not settings.email.allow_insecure:
        raise ValueError("Refusing plaintext SMTP unless SMTP_ALLOW_INSECURE=true")
    if bool(settings.email.username) != bool(settings.email.password):
        raise ValueError("SMTP_USERNAME and SMTP_PASSWORD must either both be set or both be empty")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", settings.detector_sha256):
        raise ValueError("DETECTOR_MODEL_SHA256 must contain exactly 64 hexadecimal characters")
    if not settings.classifier_revision:
        raise ValueError("CLASSIFIER_REVISION is required")
    if not settings.local_classifier_model or not settings.local_classifier_revision:
        raise ValueError("LOCAL_CLASSIFIER_MODEL and LOCAL_CLASSIFIER_REVISION are required")
    if not math.isfinite(settings.local_classifier_weight) or not 0 <= settings.local_classifier_weight <= 1:
        raise ValueError("LOCAL_CLASSIFIER_WEIGHT must be between 0 and 1")


def load_settings() -> AppSettings:
    load_dotenv()
    camera_value = os.getenv("CAMERA", "1").strip()
    camera: int | str = int(camera_value) if camera_value.isdigit() else camera_value
    email = EmailSettings(
        host=os.getenv("SMTP_HOST", "").strip(),
        port=int(os.getenv("SMTP_PORT", "587")),
        username=os.getenv("SMTP_USERNAME", "").strip(),
        password=os.getenv("SMTP_PASSWORD", ""),
        sender=os.getenv("EMAIL_FROM", "").strip(),
        recipient=os.getenv("EMAIL_TO", "").strip(),
        use_ssl=env_bool("SMTP_USE_SSL", False),
        use_starttls=env_bool("SMTP_USE_STARTTLS", True),
        allow_insecure=env_bool("SMTP_ALLOW_INSECURE", False),
    )
    required = {
        "SMTP_HOST": email.host,
        "EMAIL_FROM": email.sender,
        "EMAIL_TO": email.recipient,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ValueError(f"Missing required .env settings: {', '.join(missing)}")
    settings = AppSettings(
        camera=camera,
        camera_width=int(os.getenv("CAMERA_WIDTH", "1280")),
        camera_height=int(os.getenv("CAMERA_HEIGHT", "960")),
        camera_fps=float(os.getenv("CAMERA_FPS", "5")),
        image_dir=Path(os.getenv("BIRD_IMAGE_DIR", "bird_images")).expanduser(),
        cooldown=timedelta(minutes=float(os.getenv("COOLDOWN_MINUTES", "10"))),
        scan_interval=float(os.getenv("SCAN_INTERVAL_SECONDS", "1")),
        detection_confidence=float(os.getenv("DETECTION_CONFIDENCE", "0.35")),
        crop_padding=float(os.getenv("DETECTION_CROP_PADDING", "0.20")),
        burst_frames=int(os.getenv("BURST_FRAMES", "9")),
        burst_frame_interval=float(os.getenv("BURST_FRAME_INTERVAL_SECONDS", "1.0")),
        sharpest_frames=int(os.getenv("SHARPEST_FRAMES", "7")),
        min_valid_bird_frames=int(os.getenv("MIN_VALID_BIRD_FRAMES", "4")),
        min_event_detector_confidence=float(
            os.getenv("MIN_EVENT_DETECTOR_CONFIDENCE", "0.45")
        ),
        detection_floor_confidence=float(
            os.getenv("DETECTION_FLOOR_CONFIDENCE", "0.05")
        ),
        max_bird_crop_aspect_ratio=float(os.getenv("MAX_BIRD_CROP_ASPECT_RATIO", "2.5")),
        min_bird_presence_score=float(os.getenv("MIN_BIRD_PRESENCE_SCORE", "0.50")),
        min_bird_presence_frames=int(os.getenv("MIN_BIRD_PRESENCE_FRAMES", "2")),
        consensus_min_votes=int(os.getenv("CONSENSUS_MIN_VOTES", "4")),
        species_min_confidence=float(os.getenv("SPECIES_MIN_CONFIDENCE", "0.60")),
        species_min_margin=float(os.getenv("SPECIES_MIN_MARGIN", "0.20")),
        region_profile=os.getenv("REGION_PROFILE", "northern_nj").strip(),
        regional_prior_weight=float(os.getenv("REGIONAL_PRIOR_WEIGHT", "3.0")),
        email=email,
        detector_model=os.getenv("DETECTOR_MODEL", "yolo11n.pt"),
        detector_sha256=os.getenv(
            "DETECTOR_MODEL_SHA256",
            "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        ),
        classifier_model=os.getenv("CLASSIFIER_MODEL", "chriamue/bird-species-classifier"),
        classifier_revision=os.getenv(
            "CLASSIFIER_REVISION",
            "558944ca4448f5b311af8393c8b894eff20a06da",
        ),
        local_classifier_model=os.getenv("LOCAL_CLASSIFIER_MODEL", "imageomics/bioclip"),
        local_classifier_revision=os.getenv(
            "LOCAL_CLASSIFIER_REVISION",
            "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c",
        ),
        local_classifier_weight=float(os.getenv("LOCAL_CLASSIFIER_WEIGHT", "0.65")),
    )
    validate_settings(settings)
    return settings


class CooldownTracker:
    """Conservatively block all repeat alerts during the cooldown window."""

    def __init__(self, cooldown: timedelta, state_path: Path | None = None):
        self.cooldown = cooldown
        self.state_path = state_path
        self._last_sent: datetime | None = None
        if state_path is not None and state_path.exists():
            try:
                self._last_sent = datetime.fromisoformat(state_path.read_text().strip())
            except (OSError, ValueError):
                LOG.warning("Ignoring invalid cooldown state file: %s", state_path)

    def is_ready(self, species: str, now: datetime) -> bool:
        return self._last_sent is None or now - self._last_sent >= self.cooldown

    def mark_sent(self, species: str, now: datetime) -> None:
        self._last_sent = now
        if self.state_path is not None:
            temporary = self.state_path.with_suffix(".tmp")
            try:
                temporary.write_text(now.isoformat())
                temporary.replace(self.state_path)
            except OSError:
                LOG.exception("Could not persist cooldown state to %s", self.state_path)
                temporary.unlink(missing_ok=True)


def format_species_name(label: str) -> str:
    return label.strip().title()


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_") or "bird"


def save_bird_image(image, directory: Path, species: str, observed_at: datetime) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    # Upscale small crops so tiny/distant birds are actually visible in the
    # saved file and email attachment. Crops below the target short side are
    # enlarged with LANCZOS; the scale is capped to keep files reasonable.
    # Skipped when cv2 lacks resize (e.g. lightweight test stubs).
    resize = getattr(cv2, "resize", None)
    if resize is not None and hasattr(image, "shape"):
        height, width = image.shape[:2]
        short_side = min(height, width)
        target_short_side = 320
        max_long_side = 1280
        if short_side < target_short_side:
            scale = target_short_side / max(1, short_side)
            if max(height, width) * scale > max_long_side:
                scale = max_long_side / max(height, width)
            new_width = max(1, int(round(width * scale)))
            new_height = max(1, int(round(height * scale)))
            image = resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    filename = f"{observed_at:%Y%m%d_%H%M%S_%f}_{safe_filename(species)}.jpg"
    path = directory / filename
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not save image to {path}")
    return path


def send_email(
    settings: EmailSettings,
    identification: IdentificationResult,
    observed_at: datetime,
    image_path: Path,
) -> None:
    message = EmailMessage()
    if identification.uncertain:
        message["Subject"] = (
            f"Bird spotted: Uncertain bird (possible {identification.candidate_name})"
        )
        candidates = ", ".join(
            f"{name} {score:.1%}" for name, score in identification.top_candidates
        )
        message.set_content(
            "Identification: Uncertain bird\n"
            f"Approximate guess: {identification.candidate_name}\n"
            f"Approximate-guess score: {identification.confidence:.1%}\n"
            f"Agreement: {identification.votes} of {identification.frame_count} frames\n"
            f"Top candidates: {candidates}\n"
            "This approximate guess did not meet the certainty requirements and may be incorrect.\n"
            f"Time: {observed_at:%Y-%m-%d %H:%M:%S}\n"
        )
    else:
        message["Subject"] = f"Bird spotted: {identification.display_name}"
        message.set_content(
            f"Bird: {identification.display_name}\n"
            f"Confidence: {identification.confidence:.1%}\n"
            f"Time: {observed_at:%Y-%m-%d %H:%M:%S}\n"
        )
    message["From"] = settings.sender
    message["To"] = settings.recipient
    message.add_attachment(
        image_path.read_bytes(),
        maintype="image",
        subtype="jpeg",
        filename=image_path.name,
    )

    smtp_class = smtplib.SMTP_SSL if settings.use_ssl else smtplib.SMTP
    context = ssl.create_default_context()
    with smtp_class(settings.host, settings.port, timeout=30) as smtp:
        if not settings.use_ssl:
            smtp.ehlo()
            if settings.use_starttls:
                smtp.starttls(context=context)
                smtp.ehlo()
        if settings.username:
            smtp.login(settings.username, settings.password)
        smtp.send_message(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_detector_model(model_name: str, expected_sha256: str) -> Path:
    path = Path(model_name).expanduser()
    if not path.exists():
        if model_name != "yolo11n.pt":
            raise FileNotFoundError(f"Detector model does not exist: {path}")
        url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
        temporary = path.with_suffix(path.suffix + ".download")
        LOG.info("Downloading pinned detector model from %s", url)
        try:
            urllib.request.urlretrieve(url, temporary)
            if file_sha256(temporary) != expected_sha256.lower():
                raise ValueError("Downloaded detector model checksum does not match DETECTOR_MODEL_SHA256")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if file_sha256(path) != expected_sha256.lower():
        raise ValueError(f"Detector model checksum mismatch: {path}")
    return path


class BirdModels:
    def __init__(
        self,
        detector_name: str,
        detector_sha256: str,
        classifier_name: str,
        classifier_revision: str,
        local_classifier_name: str = "imageomics/bioclip",
        local_classifier_revision: str = "ce901ab3c6a913f9e9ef94ce6d27761069f4f01c",
    ):
        import open_clip
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        from ultralytics import YOLO

        detector_path = prepare_detector_model(detector_name, detector_sha256)
        LOG.info("Loading verified bird detector: %s", detector_path)
        self.detector = YOLO(str(detector_path))
        LOG.info("Loading species classifier: %s", classifier_name)
        self.processor = AutoImageProcessor.from_pretrained(
            classifier_name, revision=classifier_revision, use_fast=False
        )
        self.classifier = AutoModelForImageClassification.from_pretrained(
            classifier_name, revision=classifier_revision, use_safetensors=True
        )
        self.classifier.eval()
        self.torch = torch
        self.device = torch.device("cpu")
        self.classifier.to(self.device)
        LOG.info("Loading local BioCLIP classifier: %s", local_classifier_name)
        local_snapshot = snapshot_download(
            repo_id=local_classifier_name,
            revision=local_classifier_revision,
            allow_patterns=[
                "open_clip_config.json",
                "open_clip_pytorch_model.bin",
                "merges.txt",
                "vocab.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ],
        )
        local_source = f"local-dir:{local_snapshot}"
        self.local_classifier, _, self.local_preprocess = open_clip.create_model_and_transforms(
            local_source,
            require_pretrained=True,
            weights_only=True,
        )
        self.local_tokenizer = open_clip.get_tokenizer(local_source)
        self.local_classifier.eval().to(self.device)
        self._local_text_cache = {}
        LOG.info("Models ready; species classifiers device: %s", self.device)

    def _collect_bird_boxes(self, frame, imgsz: int, confidence: float):
        """Run YOLO on one view of the frame and return (x1, y1, x2, y2, score)."""
        results = self.detector.predict(
            source=frame,
            classes=[14],  # COCO class 14 is bird.
            conf=confidence,
            imgsz=imgsz,
            verbose=False,
        )
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            score = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, score))
        return boxes

    def find_best_bird(
        self,
        frame,
        confidence: float,
        crop_padding: float = 0.08,
        max_aspect_ratio: float = math.inf,
        multiscale: bool = True,
        tile_grid: int = 3,
        high_res: int = 1280,
        low_confidence: float = 0.05,
    ):
        """Return (cropped bird image, detector confidence), or None.

        The cheap full-frame pass at ``confidence`` runs first. When it finds
        nothing, a higher-resolution full-frame pass plus a ``tile_grid`` x
        ``tile_grid`` tile sweep at ``low_confidence`` hunts small or distant
        birds that the coarse pass misses. The multi-scale sweep only runs when
        the cheap pass is empty, so close birds cost a single inference.
        """
        if not math.isfinite(crop_padding) or not 0 <= crop_padding <= 1:
            raise ValueError("Bird crop padding must be between 0 and 1")
        if max_aspect_ratio < 1:
            raise ValueError("Maximum bird-box aspect ratio must be at least 1")
        if tile_grid < 1:
            raise ValueError("Detection tile grid must be at least 1")
        if high_res < 1:
            raise ValueError("Detection high-res image size must be positive")
        if not math.isfinite(low_confidence) or not 0 <= low_confidence <= 1:
            raise ValueError("Low-confidence detection floor must be between 0 and 1")

        height, width = frame.shape[:2]
        candidates = []

        def accept(x1, y1, x2, y2, score):
            box_width = max(0, x2 - x1)
            box_height = max(0, y2 - y1)
            if box_width == 0 or box_height == 0:
                return None
            aspect_ratio = max(box_width / box_height, box_height / box_width)
            if aspect_ratio > max_aspect_ratio:
                return None
            return (box_width * box_height * score, score, x1, y1, x2, y2)

        # Cheap primary pass at the configured confidence.
        for x1, y1, x2, y2, score in self._collect_bird_boxes(frame, 640, confidence):
            candidate = accept(x1, y1, x2, y2, score)
            if candidate is not None:
                candidates.append(candidate)

        # Expensive multi-scale sweep only when the cheap pass found nothing.
        if not candidates and multiscale:
            for x1, y1, x2, y2, score in self._collect_bird_boxes(frame, high_res, low_confidence):
                candidate = accept(x1, y1, x2, y2, score)
                if candidate is not None:
                    candidates.append(candidate)
            if tile_grid > 1:
                for ty in range(tile_grid):
                    for tx in range(tile_grid):
                        y0 = int(ty * height / tile_grid)
                        y1t = int((ty + 1) * height / tile_grid)
                        x0 = int(tx * width / tile_grid)
                        x1t = int((tx + 1) * width / tile_grid)
                        tile = frame[y0:y1t, x0:x1t]
                        if tile.size == 0:
                            continue
                        for lx1, ly1, lx2, ly2, score in self._collect_bird_boxes(
                            tile, 640, low_confidence
                        ):
                            candidate = accept(x0 + lx1, y0 + ly1, x0 + lx2, y0 + ly2, score)
                            if candidate is not None:
                                candidates.append(candidate)

        if not candidates:
            return None

        _, score, x1, y1, x2, y2 = max(candidates)
        box_w = x2 - x1
        box_h = y2 - y1
        box_dim = max(box_w, box_h)
        # Size-adaptive padding: small/distant birds get a larger relative
        # margin so the crop is not a postage stamp, while well-framed birds
        # keep the configured padding. Relative padding grows as the box shrinks
        # and never exceeds 1.0.
        relative_padding = min(1.0, crop_padding + 0.6 * (1.0 - min(1.0, box_dim / 220.0)))
        padding = int(relative_padding * box_dim)
        x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
        x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
        crop = frame[y1:y2, x1:x2].copy()
        if crop.size == 0:
            raise ValueError("Detector returned an empty bird crop")
        return crop, score

    def identify_species_candidates(self, bird_image, top_k: int = 3) -> list[tuple[str, float]]:
        from PIL import Image

        rgb_image = cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=Image.fromarray(rgb_image), return_tensors="pt")
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        with self.torch.inference_mode():
            logits = self.classifier(**inputs).logits
            probabilities = self.torch.softmax(logits, dim=-1)[0]
            count = min(top_k, int(probabilities.shape[0]))
            confidences, class_ids = self.torch.topk(probabilities, count)
        return [
            (
                format_species_name(self.classifier.config.id2label[int(class_id)]),
                float(confidence),
            )
            for confidence, class_id in zip(confidences.tolist(), class_ids.tolist())
        ]

    def identify_species(self, bird_image) -> tuple[str, float]:
        return self.identify_species_candidates(bird_image, top_k=1)[0]

    def identify_local_species_candidates(
        self,
        bird_image,
        species_names: set[str],
        top_k: int = 20,
    ) -> list[tuple[str, float]]:
        """Classify among common local/seasonal species with pinned BioCLIP."""
        from PIL import Image

        names = tuple(sorted(species_names))
        if not names:
            raise ValueError("Local classifier requires at least one species name")
        text_features = self._local_text_cache.get(names)
        if text_features is None:
            templates = (
                "a photo of a {}.",
                "a cropped photo of a {}.",
                "a close-up photo of a {}.",
                "a photo of the {}.",
                "a blurry photo of a {}.",
            )
            prompts = [template.format(name) for name in names for template in templates]
            with self.torch.inference_mode():
                features = self.local_classifier.encode_text(
                    self.local_tokenizer(prompts).to(self.device)
                )
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.reshape(len(names), len(templates), -1).mean(dim=1)
                features = features / features.norm(dim=-1, keepdim=True)
                text_features = features.T.contiguous()
            self._local_text_cache[names] = text_features

        rgb_image = cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB)
        image_tensor = self.local_preprocess(Image.fromarray(rgb_image)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            image_features = self.local_classifier.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = self.local_classifier.logit_scale.exp() * image_features @ text_features
            probabilities = self.torch.softmax(logits, dim=-1)[0]
            count = min(top_k, len(names))
            confidences, class_ids = self.torch.topk(probabilities, count)
        return [
            (names[int(class_id)], float(confidence))
            for confidence, class_id in zip(confidences.tolist(), class_ids.tolist())
        ]

    def bird_presence_score(self, bird_image) -> float:
        """Return local BioCLIP evidence that a crop contains a real bird.

        This is intentionally separate from species identification. It compares
        explicit bird prompts against empty-feeder/dish prompts so a YOLO false
        positive cannot become an email merely because a closed-set species
        classifier must choose some bird label.
        """
        from PIL import Image

        text_features = getattr(self, "_bird_presence_text_features", None)
        if text_features is None:
            prompt_groups = (
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
            prompts = [prompt for group in prompt_groups for prompt in group]
            with self.torch.inference_mode():
                features = self.local_classifier.encode_text(
                    self.local_tokenizer(prompts).to(self.device)
                )
                features = features / features.norm(dim=-1, keepdim=True)
                features = features.reshape(2, len(prompt_groups[0]), -1).mean(dim=1)
                text_features = features / features.norm(dim=-1, keepdim=True)
            self._bird_presence_text_features = text_features.T.contiguous()

        rgb_image = cv2.cvtColor(bird_image, cv2.COLOR_BGR2RGB)
        image_tensor = self.local_preprocess(Image.fromarray(rgb_image)).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            image_features = self.local_classifier.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = 100.0 * image_features @ self._bird_presence_text_features
            probabilities = self.torch.softmax(logits, dim=-1)[0]
        return float(probabilities[0])


def collect_bird_crops(
    capture,
    models: BirdModels,
    initial_detection,
    burst_frames: int,
    detection_confidence: float,
    frame_interval: float = 0.0,
    crop_padding: float = 0.08,
    max_aspect_ratio: float = math.inf,
    detection_floor_confidence: float = 0.05,
):
    if not math.isfinite(frame_interval) or frame_interval < 0:
        raise ValueError("BURST_FRAME_INTERVAL_SECONDS must be finite and nonnegative")
    detections = [initial_detection]
    for _ in range(burst_frames - 1):
        if frame_interval:
            time.sleep(frame_interval)
        ok, frame = capture.read()
        if not ok or frame is None:
            LOG.warning("Camera read failed during bird burst; skipping frame")
            continue
        try:
            detection = models.find_best_bird(
                frame,
                detection_confidence,
                crop_padding,
                max_aspect_ratio,
                low_confidence=detection_floor_confidence,
            )
        except Exception:
            LOG.exception("Bird detection failed on a burst frame; skipping frame")
            continue
        if detection is not None:
            detections.append(detection)
    return detections


def validate_bird_event(
    detections: list[tuple[Any, float]],
    min_frames: int,
    min_median_confidence: float,
    max_aspect_ratio: float,
) -> tuple[list[tuple[Any, float]], str | None]:
    """Reject sparse, weak, or implausibly shaped detector events before email."""
    if min_frames < 1:
        raise ValueError("Minimum bird-event frame count must be positive")
    if not 0 < min_median_confidence <= 1:
        raise ValueError("Minimum event detector confidence must be in (0, 1]")
    if not math.isfinite(max_aspect_ratio) or max_aspect_ratio < 1:
        raise ValueError("Maximum bird-crop aspect ratio must be finite and at least 1")

    plausible = []
    rejected_shapes = 0
    for crop, score in detections:
        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            rejected_shapes += 1
            continue
        aspect_ratio = max(width / height, height / width)
        if aspect_ratio > max_aspect_ratio:
            rejected_shapes += 1
            continue
        plausible.append((crop, score))

    if len(plausible) < min_frames:
        reason = (
            f"only {len(plausible)} valid detection(s), require {min_frames}"
        )
        if rejected_shapes:
            reason += f"; rejected {rejected_shapes} crop(s) with implausible shape"
        return [], reason

    event_confidence = median(score for _, score in plausible)
    if event_confidence < min_median_confidence:
        return [], (
            f"median detector confidence {event_confidence:.1%} is below "
            f"{min_median_confidence:.1%}"
        )
    return plausible, None


def validate_visual_bird_presence(
    detections: list[tuple[Any, float]],
    scorer,
    min_score: float,
    min_frames: int,
) -> tuple[list[tuple[Any, float]], str | None, list[float]]:
    """Require repeated BioCLIP evidence of a real bird before side effects."""
    if not 0 <= min_score <= 1:
        raise ValueError("Minimum bird-presence score must be between 0 and 1")
    if min_frames < 1:
        raise ValueError("Minimum bird-presence frame count must be positive")

    accepted = []
    scores = []
    for crop, detector_score in detections:
        score = float(scorer(crop))
        scores.append(score)
        if score >= min_score:
            accepted.append((crop, detector_score))

    if len(accepted) < min_frames:
        median_score = median(scores) if scores else 0.0
        return [], (
            f"only {len(accepted)} crop(s) passed bird-presence score {min_score:.1%}, "
            f"require {min_frames}; median presence score {median_score:.1%}"
        ), scores
    return accepted, None, scores


def open_camera(camera: int | str, width: int, height: int, fps: float):
    if isinstance(camera, str) and camera.startswith("/dev/"):
        capture = cv2.VideoCapture(camera, cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(camera)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if not capture.isOpened():
        capture.release()
        return None
    return capture


def run(settings: AppSettings) -> None:
    settings.image_dir.mkdir(parents=True, exist_ok=True)
    models = BirdModels(
        settings.detector_model,
        settings.detector_sha256,
        settings.classifier_model,
        settings.classifier_revision,
        settings.local_classifier_model,
        settings.local_classifier_revision,
    )
    cooldown = CooldownTracker(settings.cooldown, settings.image_dir / ".last_alert")
    capture = None
    LOG.info("Watching camera %r; images will be saved in %s", settings.camera, settings.image_dir.resolve())
    startup_season = season_for_month(datetime.now().astimezone().month)
    LOG.info(
        "Regional identification prior: %s (%s, automatic by month; %d preferred at %.1fx / %d broad at 1.5x)",
        settings.region_profile,
        startup_season,
        len(preferred_regional_species(settings.region_profile, datetime.now().astimezone().month)),
        settings.regional_prior_weight,
        len(regional_species(settings.region_profile, datetime.now().astimezone().month)),
    )
    LOG.info(
        "Hybrid species identification: %.0f%% BioCLIP local/seasonal + %.0f%% broad 525-label classifier",
        settings.local_classifier_weight * 100,
        (1.0 - settings.local_classifier_weight) * 100,
    )

    try:
        while True:
            if capture is None:
                capture = open_camera(
                    settings.camera,
                    settings.camera_width,
                    settings.camera_height,
                    settings.camera_fps,
                )
                if capture is None:
                    LOG.error("Could not open camera %r; retrying in 5 seconds", settings.camera)
                    time.sleep(5)
                    continue
                LOG.info(
                    "Camera opened at %.0fx%.0f, requested %.1f FPS",
                    capture.get(cv2.CAP_PROP_FRAME_WIDTH),
                    capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    settings.camera_fps,
                )
            ok, frame = capture.read()
            if not ok or frame is None:
                LOG.error("Camera read failed; reopening camera")
                capture.release()
                capture = None
                time.sleep(2)
                continue

            try:
                detection = models.find_best_bird(
                    frame,
                    settings.detection_confidence,
                    settings.crop_padding,
                    settings.max_bird_crop_aspect_ratio,
                    low_confidence=settings.detection_floor_confidence,
                )
                if detection is not None:
                    now = datetime.now().astimezone()
                    if not cooldown.is_ready("bird", now):
                        LOG.debug("Cooldown active; skipping extended bird identification")
                        time.sleep(max(0.0, settings.scan_interval))
                        continue
                    broad = regional_species(settings.region_profile, now.month)
                    plausible = identification_plausible_species(
                        settings.region_profile, now.month
                    )
                    preferred = preferred_regional_species(settings.region_profile, now.month)
                    preferred_names = preferred_regional_species_names(
                        settings.region_profile, now.month
                    )
                    burst_detections = collect_bird_crops(
                        capture,
                        models,
                        detection,
                        settings.burst_frames,
                        settings.detection_confidence,
                        settings.burst_frame_interval,
                        settings.crop_padding,
                        settings.max_bird_crop_aspect_ratio,
                        settings.detection_floor_confidence,
                    )
                    valid_detections, rejection_reason = validate_bird_event(
                        burst_detections,
                        settings.min_valid_bird_frames,
                        settings.min_event_detector_confidence,
                        settings.max_bird_crop_aspect_ratio,
                    )
                    if not valid_detections:
                        LOG.info(
                            "Suppressed false bird event: %s; no image saved or email sent",
                            rejection_reason,
                        )
                        time.sleep(max(0.0, settings.scan_interval))
                        continue
                    presence_detections, presence_reason, presence_scores = (
                        validate_visual_bird_presence(
                            valid_detections,
                            models.bird_presence_score,
                            settings.min_bird_presence_score,
                            settings.min_bird_presence_frames,
                        )
                    )
                    if not presence_detections:
                        LOG.info(
                            "Suppressed empty-feeder event: %s; scores=%s; "
                            "no image saved or email sent",
                            presence_reason,
                            ", ".join(f"{score:.1%}" for score in presence_scores),
                        )
                        time.sleep(max(0.0, settings.scan_interval))
                        continue
                    LOG.info(
                        "Bird-presence gate passed: %d/%d crop(s), median %.1f%%",
                        len(presence_detections),
                        len(valid_detections),
                        median(presence_scores) * 100,
                    )
                    selected = select_sharpest_crops(
                        presence_detections, settings.sharpest_frames
                    )
                    frame_predictions = []
                    for bird_image, _ in selected:
                        frame_predictions.append(
                            hybrid_species_predictions(
                                models,
                                bird_image,
                                preferred_names,
                                preferred,
                                broad,
                                settings.regional_prior_weight,
                                settings.local_classifier_weight,
                            )
                        )
                    identification = resolve_identification(
                        frame_predictions,
                        settings.consensus_min_votes,
                        settings.species_min_confidence,
                        settings.species_min_margin,
                        plausible,
                    )
                    bird_image = selected[0][0]
                    detector_confidence = max(score for _, score in selected)
                    candidate_text = ", ".join(
                        f"{name} {score:.1%}" for name, score in identification.top_candidates
                    )
                    LOG.info(
                        "Bird identification: %s; votes %d/%d, confidence %.1f%%, "
                        "margin %.1f%%, detector %.1f%%, region %s/%s; aggregate candidates: %s",
                        identification.display_name,
                        identification.votes,
                        identification.frame_count,
                        identification.confidence * 100,
                        identification.margin * 100,
                        detector_confidence * 100,
                        settings.region_profile,
                        season_for_month(now.month),
                        candidate_text,
                    )
                    cooldown_key = identification.candidate_name
                    if cooldown.is_ready(cooldown_key, now):
                        image_path = save_bird_image(
                            bird_image,
                            settings.image_dir,
                            identification.candidate_name,
                            now,
                        )
                        LOG.info("Saved bird image: %s", image_path)
                        cooldown.mark_sent(cooldown_key, now)
                        try:
                            send_email(
                                settings.email,
                                identification,
                                now,
                                image_path,
                            )
                        except Exception:
                            LOG.exception("Email failed; the image remains saved and retry waits for the cooldown")
                        else:
                            LOG.info("Email alert sent for %s", identification.display_name)
                    else:
                        LOG.info("Cooldown active for %s; no repeat email sent", cooldown_key)
            except Exception:
                LOG.exception("Detection failed; continuing")

            time.sleep(max(0.0, settings.scan_interval))
    except KeyboardInterrupt:
        LOG.info("Stopping bird watcher")
    finally:
        if capture is not None:
            capture.release()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        settings = load_settings()
        run(settings)
    except Exception:
        LOG.exception("Bird watcher could not start")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
