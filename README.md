# Local Bird Watcher

A local Python 3.11+ application that watches a camera, detects birds with YOLO, tracks one bird through a short observation window, verifies that the crop contains a real bird, identifies the species with BioCLIP plus a broad 525-species classifier, saves the clearest crop, and emails an alert through SMTP.

After model files are downloaded once, inference runs locally. The app does not capture audio and does not require a paid inference API.

## Canonical entrypoint

Run the application with:

```bash
python main.py
```

`main.py` directly launches `birdwatcher.app`. `birdwatcher_improved.py` and `legacy_main.py` are compatibility launchers only. **The production `birdwatcher` package does not import or depend on `legacy_main.py`.**

Current runtime layout:

```text
main.py
  -> birdwatcher.app             application loop and event processing
     -> birdwatcher.config       .env loading and validation
     -> birdwatcher.models       pinned YOLO and classifier loading
     -> birdwatcher.tracking     detection association and active-event deduplication
     -> birdwatcher.validation   repeated-detection validation
     -> birdwatcher.classification  BioCLIP embedding reuse and hybrid scoring
     -> birdwatcher.region       regional priors and consensus resolution
     -> birdwatcher.species      species aliases and canonical identity keys
     -> birdwatcher.constants    shared tuning constants
     -> birdwatcher.media        camera and image helpers
     -> birdwatcher.emailer      SMTP delivery
     -> birdwatcher.alerts       persistent retry queue
```

## How it works

```text
Camera
  -> YOLO bird detection
  -> one-to-one matching against active spatial events
  -> choose a new bird event
  -> follow that bird across the observation window
  -> repeated-detection validation
  -> encode each accepted crop once with BioCLIP
       -> bird-vs-empty-feeder validation
       -> reuse the same embedding for species comparison
  -> broad 525-species classifier proposes candidates
  -> BioCLIP evaluates local/seasonal species plus broad-model candidates
  -> hybrid score + regional evidence + multi-frame consensus
  -> mark the confirmed spatial event active
  -> save the sharpest crop
  -> email alert or persist it for retry
  -> refresh the active event after expensive processing completes
```

Tracking uses bounding-box overlap, center distance, and short motion prediction. Active-event matching is one-to-one within each scan, so a single existing event cannot consume multiple nearby detections.

## Detection sweeps

The normal pass runs one detector inference per scan. When it finds nothing, a full-frame low-confidence pass runs, and only then a nine-tile sweep for small or distant birds.

- The nine-tile sweep costs nine extra inferences, so it is rate-limited by `TILE_SWEEP_INTERVAL_SECONDS`.
- Top-level scans and tracked observation bursts draw from one shared budget. This matters most during a burst: samples are taken on a wall-clock schedule, so sweeps that overrun `BURST_FRAME_INTERVAL_SECONDS` stretch the observation window rather than merely costing CPU.
- The interval is spent only when a sweep actually runs. Frames where the cheaper passes already found the bird cost nothing, so the budget stays available for the frames that lost it.
- Overlapping boxes from the different passes are collapsed by non-maximum suppression, so one bird cannot become several detections.

## Species naming

The broad classifier spells some species differently from the regional checklists (for example `Tit Mouse` for `Tufted Titmouse`). `birdwatcher/species.py` holds the single alias table, and every comparison — regional priors, classifier blending, and the final plausibility gate — uses canonical identity keys. Without this, one bird's evidence splits across two labels and can lose to a runner-up.

## Active-event deduplication

Confirmed birds create active spatial events rather than species-name cooldown keys.

- Matching detections are filtered before burst collection and classification.
- A spatially different bird can start a new event while another bird remains visible.
- One active event can match at most one detection per scan.
- An event closes after `EVENT_CLEAR_SECONDS` with no matching detection.
- Confirmed events are refreshed after classification and alert processing so slow CPU inference does not immediately create a duplicate event.
- `ACTIVE_EVENT_MAX_MINUTES` prevents one continuously visible event from remaining suppressed forever.
- `COOLDOWN_MINUTES` remains a backwards-compatible fallback for `ACTIVE_EVENT_MAX_MINUTES`.

The default `EVENT_CLEAR_SECONDS` is **6 seconds**.

## Email retry behavior

If immediate SMTP delivery fails, the saved alert is written to `BIRD_IMAGE_DIR/.email_retry_queue/` and retried with bounded exponential backoff.

- Successful retries remove the queue item.
- Retry state is written atomically.
- Items are abandoned after the configured maximum attempt count.
- Corrupt payloads, missing-image items, and exhausted retries are moved to `.email_retry_queue/failed/` instead of being retried forever.

## Models

- **Bird detection:** Ultralytics `yolo11n.pt`, pretrained on COCO. COCO class 14 is used for birds. The pinned detector download is SHA-256 verified before loading.
- **Broad species classifier:** Hugging Face `chriamue/bird-species-classifier`, an EfficientNet-B2 model covering 525 bird species, pinned to a fixed revision and safetensors weights.
- **Visual retrieval and validation:** `imageomics/bioclip` (ViT-B/16), pinned to a fixed revision. It provides bird-presence evidence and local/candidate species comparison.
- **Regional evidence:** a broad New Jersey plausibility tier is combined with Northern New Jersey resident and seasonal preferences.

The ML models were not changed by the runtime refactor. Species identification remains probabilistic. The email confidence is a hybrid evidence score, not a calibrated probability or guarantee.

## Fresh install

```bash
git clone https://github.com/asdfjklfadsf/birdwatcher.git
cd birdwatcher

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
```

Edit `.env` before running. At minimum, configure the camera and SMTP settings.

### Camera configuration

For a normal USB webcam, start with:

```dotenv
CAMERA=0
```

On Linux, prefer a stable `/dev/v4l/by-id/...` path when available. The default requested capture mode is 1280x960 at 5 FPS.

### SMTP configuration

For Gmail with STARTTLS:

```dotenv
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EMAIL_FROM=your_email@gmail.com
EMAIL_TO=recipient@example.com
SMTP_USE_SSL=false
SMTP_USE_STARTTLS=true
```

Use an app password rather than a normal Gmail password. `EMAIL_TO` may contain one address or a comma-separated recipient list.

For implicit TLS, commonly on port 465:

```dotenv
SMTP_PORT=465
SMTP_USE_SSL=true
SMTP_USE_STARTTLS=false
```

Plaintext SMTP is rejected unless `SMTP_ALLOW_INSECURE=true` is explicitly enabled for a trusted local relay.

## Run

```bash
python main.py
```

Press `Ctrl+C` to stop. The first run requires internet access to download the pinned model files. Later inference uses the local cache.

## Test

Run the unit and integration suite with:

```bash
python -m unittest -v
```

The suite covers regional and consensus logic, event validation, motion-aware tracking, one-to-one active-event matching, event clearing and post-processing refresh, `.env` overrides, broad-candidate expansion, BioCLIP embedding reuse, retry persistence/backoff/quarantine, canonical species identity across every comparison stage, detection deduplication and tile-sweep rate limiting, the bounded prompt-embedding cache, and the canonical modular entrypoint.

GitHub Actions has two jobs:

1. A lightweight unit/integration job.
2. A full `requirements.txt` runtime smoke job that imports and exercises the real dependency stack, validates configuration, checks critical third-party APIs, and then removes `legacy_main.py` before importing the production runtime to prove the package has no legacy dependency.

Real-camera validation is still recommended because CI does not download all model weights or attach physical camera hardware.

## Important configuration defaults

| Setting | Default | Description |
|---|---:|---|
| `CAMERA` | `0` | OpenCV camera index or Linux device path |
| `CAMERA_WIDTH` | `1280` | Requested capture width |
| `CAMERA_HEIGHT` | `960` | Requested capture height |
| `CAMERA_FPS` | `5` | Requested camera frame rate |
| `SCAN_INTERVAL_SECONDS` | `1` | Delay between normal scan iterations |
| `EVENT_CLEAR_SECONDS` | `6` | Absence period before a spatial bird event is considered finished |
| `ACTIVE_EVENT_MAX_MINUTES` | `10` | Maximum lifetime of one continuously active event |
| `COOLDOWN_MINUTES` | `10` fallback | Backwards-compatible fallback when `ACTIVE_EVENT_MAX_MINUTES` is absent |
| `DETECTION_CONFIDENCE` | `0.35` | Normal YOLO bird-detection threshold |
| `TILE_SWEEP_INTERVAL_SECONDS` | `5` | Minimum delay between nine-tile low-confidence sweeps |
| `DETECTION_CROP_PADDING` | `0.20` | Base crop context around the raw bird box |
| `MAX_BIRD_CROP_ASPECT_RATIO` | `2.5` | Rejects implausibly elongated detection boxes |
| `BURST_FRAMES` | `9` | Maximum observation samples in one tracked event |
| `BURST_FRAME_INTERVAL_SECONDS` | `1.0` | Target spacing between observation samples |
| `SHARPEST_FRAMES` | `7` | Maximum accepted crops used for identification |
| `MIN_VALID_BIRD_FRAMES` | `4` | Required repeated tracked detections |
| `MIN_EVENT_DETECTOR_CONFIDENCE` | `0.07` | Minimum median detector confidence for the event |
| `DETECTION_FLOOR_CONFIDENCE` | `0.05` | Low-confidence floor used by the small/distant-bird sweep |
| `MIN_BIRD_PRESENCE_SCORE` | `0.50` | Minimum BioCLIP bird-presence score for a crop |
| `MIN_BIRD_PRESENCE_FRAMES` | `2` | Required crops passing the bird-presence gate |
| `CONSENSUS_MIN_VOTES` | `4` | Required agreeing top species predictions for a definite name |
| `SPECIES_MIN_CONFIDENCE` | `0.60` | Minimum aggregate hybrid evidence score |
| `SPECIES_MIN_MARGIN` | `0.20` | Minimum winner-versus-runner-up score margin |
| `REGION_PROFILE` | `northern_nj` | Active regional profile |
| `REGIONAL_PRIOR_WEIGHT` | `3.0` | Preferred local/seasonal regional multiplier; `1.0` disables the regional preference |
| `LOCAL_CLASSIFIER_WEIGHT` | `0.65` | BioCLIP share of the hybrid species score |

An event may pass the bird-presence gate with fewer frames than `CONSENSUS_MIN_VOTES`. In that case the app can report `Uncertain bird` because there is not enough multi-frame evidence for a definite species name.

## Automated installation note

For an automated installer or AI agent: clone the repository, create a Python 3.11 virtual environment, install `requirements.txt`, copy `.env.example` to `.env`, configure a valid camera and SMTP credentials, run `python -m unittest -v`, and start the application with `python main.py`.
