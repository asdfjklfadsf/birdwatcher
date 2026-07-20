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


def preferred_regional_species(profile: str, month: int) -> set[str]:
    """Return supported common/resident labels that receive the seasonal boost."""
    _validate_region_profile(profile)
    names = _NORTHERN_NJ_RESIDENTS | _NORTHERN_NJ_SEASONAL[season_for_month(month)]
    aliases = {species_key("Tufted Titmouse"): species_key("Tit Mouse")}
    preferred = {aliases.get(species_key(name), species_key(name)) for name in names}
    supported = {species_key(name) for name in _NEW_JERSEY_MODEL_SPECIES}
    return preferred & supported


def regional_species(profile: str, month: int) -> set[str]:
    """Return broad NJ-documented model labels used only as a plausibility gate."""
    _validate_region_profile(profile)
    season_for_month(month)
    return {species_key(name) for name in _NEW_JERSEY_MODEL_SPECIES}


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
    burst_frames: int
    sharpest_frames: int
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
    confidence_sums: defaultdict[str, float] = defaultdict(float)
    aggregate_scores: defaultdict[str, float] = defaultdict(float)
    for predictions in frame_predictions:
        confidence_sums[predictions[0][0]] += predictions[0][1]
        for label, score in predictions:
            aggregate_scores[label] += score

    candidate_name = max(votes, key=lambda label: (votes[label], confidence_sums[label], label))
    winning_frames = [predictions for predictions in frame_predictions if predictions[0][0] == candidate_name]
    confidence = sum(predictions[0][1] for predictions in winning_frames) / len(winning_frames)
    margins = [
        predictions[0][1] - (predictions[1][1] if len(predictions) > 1 else 0.0)
        for predictions in winning_frames
    ]
    margin = sum(margins) / len(margins)
    vote_count = votes[candidate_name]
    out_of_region = bool(plausible_species) and species_key(candidate_name) not in plausible_species
    uncertain = (
        vote_count < min_votes
        or confidence < min_confidence
        or margin < min_margin
        or out_of_region
    )
    display_name = candidate_name if not uncertain else "Uncertain bird"
    frame_count = len(frame_predictions)
    top_candidates = tuple(
        (label, total / frame_count)
        for label, total in sorted(
            aggregate_scores.items(), key=lambda item: (-item[1], item[0])
        )[:3]
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
    if settings.burst_frames < 1:
        raise ValueError("BURST_FRAMES must be at least 1")
    if not 1 <= settings.sharpest_frames <= settings.burst_frames:
        raise ValueError("SHARPEST_FRAMES must be between 1 and BURST_FRAMES")
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
        burst_frames=int(os.getenv("BURST_FRAMES", "7")),
        sharpest_frames=int(os.getenv("SHARPEST_FRAMES", "3")),
        consensus_min_votes=int(os.getenv("CONSENSUS_MIN_VOTES", "2")),
        species_min_confidence=float(os.getenv("SPECIES_MIN_CONFIDENCE", "0.70")),
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
    ):
        import torch
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
        LOG.info("Models ready; species classifier device: %s", self.device)

    def find_best_bird(self, frame, confidence: float):
        """Return (cropped bird image, detector confidence), or None."""
        results = self.detector.predict(
            source=frame,
            classes=[14],  # COCO class 14 is bird.
            conf=confidence,
            verbose=False,
        )
        candidates = []
        height, width = frame.shape[:2]
        for box in results[0].boxes:
            x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
            score = float(box.conf[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            candidates.append((area * score, score, x1, y1, x2, y2))
        if not candidates:
            return None

        _, score, x1, y1, x2, y2 = max(candidates)
        padding = int(0.08 * max(x2 - x1, y2 - y1))
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


def collect_bird_crops(
    capture,
    models: BirdModels,
    initial_detection,
    burst_frames: int,
    detection_confidence: float,
):
    detections = [initial_detection]
    for _ in range(burst_frames - 1):
        ok, frame = capture.read()
        if not ok or frame is None:
            LOG.warning("Camera read failed during bird burst; skipping frame")
            continue
        try:
            detection = models.find_best_bird(frame, detection_confidence)
        except Exception:
            LOG.exception("Bird detection failed on a burst frame; skipping frame")
            continue
        if detection is not None:
            detections.append(detection)
    return detections


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
                detection = models.find_best_bird(frame, settings.detection_confidence)
                if detection is not None:
                    now = datetime.now().astimezone()
                    plausible = regional_species(settings.region_profile, now.month)
                    preferred = preferred_regional_species(settings.region_profile, now.month)
                    burst_detections = collect_bird_crops(
                        capture,
                        models,
                        detection,
                        settings.burst_frames,
                        settings.detection_confidence,
                    )
                    selected = select_sharpest_crops(burst_detections, settings.sharpest_frames)
                    frame_predictions = []
                    for bird_image, _ in selected:
                        raw_predictions = models.identify_species_candidates(bird_image, top_k=20)
                        frame_predictions.append(
                            apply_regional_prior(
                                raw_predictions,
                                preferred,
                                settings.regional_prior_weight,
                                plausible_species=plausible,
                                plausible_weight=1.5,
                            )[:3]
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
