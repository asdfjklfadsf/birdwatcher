# Local Bird Watcher

A local Python 3.11+ bird-watching app that continuously reads a camera, detects birds with YOLO, follows one bird through a short observation window, verifies that the crop actually contains a bird, identifies the species with BioCLIP plus a broad 525-species classifier, saves the clearest crop, and emails an alert through SMTP.

After the models are downloaded once, inference runs locally. The app does not capture or analyze audio, and it does not use a paid inference API.

## Canonical entrypoint

**Run the application with `python main.py`.**

`main.py` directly launches the modular runtime in `birdwatcher.app`. There is no runtime `sys.modules` rewriting and the application no longer routes through `birdwatcher_improved.py`. That file remains only as a compatibility launcher for older service definitions.

Current runtime layout:

```text
main.py
  -> birdwatcher.app          application loop and event processing
     -> birdwatcher.config    .env loading and validation
     -> birdwatcher.tracking  detection association and active-event deduplication
     -> birdwatcher.classification  BioCLIP embedding reuse and hybrid classification
```

`legacy_main.py` is retained for the established low-level model, regional-prior, image, and SMTP helper implementations while the current application logic lives in the package modules above. Its old standalone run loop is not the supported entrypoint.

## How it works

```text
Camera
  -> YOLO bird detection
  -> remove detections that belong to an already-active bird event
  -> choose a new bird event
  -> track that bird across the observation window
  -> repeated-detection validation
  -> encode each accepted crop once with BioCLIP
       -> bird-vs-empty-feeder validation
       -> reuse the same embedding for species comparison
  -> broad 525-species classifier proposes candidates
  -> BioCLIP evaluates local/seasonal species plus broad-model candidates
  -> hybrid score + regional evidence + multi-frame consensus
  -> mark the confirmed spatial event active
  -> save the sharpest crop
  -> email alert
```

Tracking uses bounding-box overlap, center distance, and short motion prediction. This reduces accidental switching when another bird enters or crosses near the tracked subject. It is still lightweight spatial tracking rather than biometric identity tracking.

## Active-event deduplication

The current runtime does **not** suppress alerts by predicted species name. Instead, confirmed birds create active spatial events.

- Matching detections of the same active bird are filtered before the expensive burst and classification pipeline.
- A spatially different bird can start a new event immediately, even while another bird remains visible.
- An event closes after `EVENT_CLEAR_SECONDS` with no matching detection.
- `ACTIVE_EVENT_MAX_MINUTES` prevents one continuously visible event from remaining suppressed forever.
- A changing species prediction for the same lingering bird does not create a new cooldown key and therefore does not cause repeat alerts by itself.
- SMTP failure does not cause repeated immediate processing while the same event remains active.

`COOLDOWN_MINUTES` is accepted only as a backwards-compatible fallback for `ACTIVE_EVENT_MAX_MINUTES` when the new setting is absent.

## Models

- **Bird detection:** Ultralytics `yolo11n.pt`, pretrained on COCO. COCO class 14 is used for birds. The pinned detector download is SHA-256 verified before loading.
- **Broad species classifier:** Hugging Face `chriamue/bird-species-classifier`, an EfficientNet-B2 model covering 525 bird species, pinned to a fixed revision and safetensors weights.
- **Visual retrieval and validation:** `imageomics/bioclip` (ViT-B/16), pinned to a fixed revision. It is used for bird-presence evidence and for comparison against the local/seasonal list plus candidates proposed by the broad classifier.
- **Regional evidence:** a broad New Jersey safety tier is stored locally and combined with Northern New Jersey resident and seasonal preferences.

The ML models themselves were not changed during the runtime refactor. Species identification remains probabilistic. The email confidence is a hybrid evidence score, not a calibrated probability or guarantee that the species name is correct.

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

Edit `.env` before running the application. At minimum, configure the camera and SMTP settings. Values explicitly set in `.env`, including `MIN_EVENT_DETECTOR_CONFIDENCE`, are honored by the current configuration loader.

### Camera configuration

For a normal USB webcam, start with:

```dotenv
CAMERA=0
```

On Linux, a persistent device path under `/dev/v4l/by-id/` is preferable when available because `/dev/videoN` numbering can change. To inspect available persistent camera paths:

```bash
ls -l /dev/v4l/by-id/
```

Set `CAMERA` to the correct path when one is available, for example:

```dotenv
CAMERA=/dev/v4l/by-id/your-camera-video-index0
```

The configured capture defaults are 1280x960 at 5 FPS. Change them in `.env` when the camera does not support that mode.

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

Use an app password rather than a normal Gmail password. `EMAIL_TO` may contain one recipient or a comma-separated recipient list.

For implicit TLS, commonly on port 465:

```dotenv
SMTP_PORT=465
SMTP_USE_SSL=true
SMTP_USE_STARTTLS=false
```

Plaintext SMTP is rejected unless `SMTP_ALLOW_INSECURE=true` is explicitly set for a trusted local relay.

## Run

With the virtual environment active:

```bash
python main.py
```

Press `Ctrl+C` to stop. The first run requires internet access to download the pinned YOLO and species models. Later inference uses the locally cached models.

## Test

Run the repository tests with:

```bash
python -m unittest -v
```

The tests cover core regional and consensus logic, event validation, motion-aware tracking, active-event suppression and clearing, `.env` override behavior, broad-candidate expansion, BioCLIP one-encode-per-crop behavior, and the canonical modular entrypoint.

GitHub Actions also compiles the Python sources and runs these tests on pushes to `main` and on pull requests. The full ML path still depends on the actual camera and downloaded model weights, so real-camera validation is recommended after installation.

## Important configuration defaults

| Setting | Default | Description |
|---|---:|---|
| `CAMERA` | `0` | OpenCV camera index or Linux device path |
| `CAMERA_WIDTH` | `1280` | Requested capture width |
| `CAMERA_HEIGHT` | `960` | Requested capture height |
| `CAMERA_FPS` | `5` | Requested camera frame rate |
| `SCAN_INTERVAL_SECONDS` | `1` | Delay between normal scan iterations |
| `EVENT_CLEAR_SECONDS` | `3` | Absence period before a spatial bird event is considered finished |
| `ACTIVE_EVENT_MAX_MINUTES` | `10` | Maximum lifetime of one continuously active event |
| `COOLDOWN_MINUTES` | `10` fallback | Backwards-compatible alias used only when `ACTIVE_EVENT_MAX_MINUTES` is absent |
| `DETECTION_CONFIDENCE` | `0.35` | Normal YOLO bird-detection threshold |
| `DETECTION_CROP_PADDING` | `0.20` | Base crop context around the raw bird box |
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
| `REGIONAL_PRIOR_WEIGHT` | `3.0` | Preferred local/seasonal regional multiplier |
| `LOCAL_CLASSIFIER_WEIGHT` | `0.65` | BioCLIP share of the hybrid species score |

An event may pass the bird-presence gate with fewer frames than `CONSENSUS_MIN_VOTES`. In that case the app can correctly report that a bird was seen while labeling the species as `Uncertain bird` because there was not enough multi-frame evidence for a definite name.

## Runtime behavior

- A cheap full-frame YOLO pass runs first. When that finds nothing, a higher-resolution full-frame pass and tiled sweep look for small or distant birds.
- Active-event filtering happens before extended burst collection and species classification, reducing repeated CPU work for a bird that remains in view.
- Once a new event begins, detections are associated across frames using overlap, center proximity, and short motion prediction.
- Repeated detections must satisfy the event frame-count and median-confidence gates before classification or alerting proceeds.
- Each accepted crop is encoded once by BioCLIP; the embedding is reused for bird-presence and species scoring.
- The broad classifier proposes unusual candidates. BioCLIP evaluates those candidates together with the active Northern New Jersey resident and seasonal species list before the two evidence streams are blended.
- Weak agreement, insufficient confidence, a small top-two margin, or an implausible regional winner produces an `Uncertain bird` alert with an approximate guess and leading candidates.
- The sharpest accepted bird crop is saved and attached to the email.
- Camera-read, detection, classification, image-save, and SMTP failures are logged rather than terminating the watcher where recovery is possible.

## Automated installation note

For an automated installer or AI agent, the intended sequence is: clone the repository, create a Python 3.11 virtual environment, install `requirements.txt`, copy `.env.example` to `.env`, configure a valid camera and SMTP credentials, run `python -m unittest -v`, and start the application with **`python main.py`**.
