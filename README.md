# Local Bird Watcher

A local Python 3.11+ bird-watching app that continuously reads a camera, detects birds with YOLO, tracks the same bird across a short observation window, verifies that the crop actually contains a bird, identifies the species with a hybrid BioCLIP and 525-species classifier, saves the clearest crop, and emails an alert through SMTP.

After the models are downloaded once, inference runs locally. The app does not capture or analyze audio.

## Canonical entrypoint

**Run the application with `python main.py`.**

`main.py` is the supported entrypoint and launches the current tracked-event runtime. `legacy_main.py` preserves the older implementation for compatibility with existing imports and tests. `birdwatcher_improved.py` contains the tracked runtime used by `main.py`; installers and service definitions should still invoke `main.py`.

## How it works

```text
Camera
  -> YOLO bird detection
  -> track the same bird across the observation window
  -> repeated-detection validation
  -> BioCLIP bird-vs-empty-feeder validation
  -> select the sharpest accepted crops
  -> broad 525-species classifier proposes candidates
  -> BioCLIP evaluates local/seasonal species plus broad-model candidates
  -> hybrid score + regional evidence + multi-frame consensus
  -> save the sharpest crop
  -> email alert
```

The tracking step prevents a second bird entering the frame from silently replacing the original subject during one event. BioCLIP image embeddings are reused for presence and species scoring to reduce duplicate CPU work.

## Models

- **Bird detection:** Ultralytics `yolo11n.pt`, pretrained on COCO. COCO class 14 is used for birds. The pinned detector download is SHA-256 verified before loading.
- **Broad species classifier:** Hugging Face `chriamue/bird-species-classifier`, an EfficientNet-B2 model covering 525 bird species, pinned to a fixed revision and safetensors weights.
- **Visual retrieval and validation:** `imageomics/bioclip` (ViT-B/16), pinned to a fixed revision. It is used both for bird-presence evidence and for visual comparison against the local/seasonal list plus candidates proposed by the broad classifier.
- **Regional evidence:** a broad New Jersey safety tier is stored locally and combined with Northern New Jersey resident and seasonal preferences.

Species identification is probabilistic. The email confidence is a hybrid evidence score, not a calibrated probability or guarantee that the species name is correct.

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

Edit `.env` before running the application. At minimum, configure the camera and SMTP settings.

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

Press `Ctrl+C` to stop. The first run requires internet access to download the pinned YOLO and species models. Later inference uses the locally cached models and does not require a paid API.

## Test

Run the repository unit tests with:

```bash
python -m unittest
```

The ML-heavy integration path still depends on the actual camera, installed model packages, and downloaded model weights, so real-camera validation is recommended after installation.

## Important configuration defaults

| Setting | Default | Description |
|---|---:|---|
| `CAMERA` | `0` in example config | OpenCV camera index or Linux device path |
| `CAMERA_WIDTH` | `1280` | Requested capture width |
| `CAMERA_HEIGHT` | `960` | Requested capture height |
| `CAMERA_FPS` | `5` | Requested camera frame rate |
| `COOLDOWN_MINUTES` | `10` | Per-species alert cooldown in the current runtime |
| `SCAN_INTERVAL_SECONDS` | `1` | Delay between normal scan iterations |
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
| `CONSENSUS_MIN_VOTES` | `4` | Required agreeing top species predictions |
| `SPECIES_MIN_CONFIDENCE` | `0.60` | Minimum aggregate hybrid evidence score |
| `SPECIES_MIN_MARGIN` | `0.20` | Minimum winner-versus-runner-up score margin |
| `REGION_PROFILE` | `northern_nj` | Active regional profile |
| `REGIONAL_PRIOR_WEIGHT` | `3.0` | Preferred local/seasonal regional multiplier |
| `LOCAL_CLASSIFIER_WEIGHT` | `0.65` | BioCLIP share of the hybrid species score |

## Runtime behavior

- A cheap full-frame YOLO pass runs first. When that finds nothing, a higher-resolution full-frame pass and tiled sweep look for small or distant birds.
- Once an event begins, detections are associated across frames using overlap and center proximity so the event does not intentionally switch to a different bird.
- Repeated detections must satisfy the event frame-count and median-confidence gates before classification or alerting proceeds.
- BioCLIP separately verifies bird presence to suppress feeder dishes, seeds, glare, and hardware that YOLO may mistake for a bird.
- The broad classifier proposes unusual candidates. BioCLIP evaluates those candidates together with the active Northern New Jersey resident and seasonal species list before the two evidence streams are blended.
- Weak agreement, insufficient confidence, a small top-two margin, or an implausible regional winner produces an `Uncertain bird` alert with an approximate guess and leading candidates.
- The sharpest accepted bird crop is saved and attached to the email.
- Cooldowns are persisted per species so one species does not automatically suppress a different species for the full cooldown period.
- Camera-read, detection, classification, image-save, and SMTP failures are logged rather than terminating the watcher where recovery is possible.

## Automated installation note

For an automated installer or AI agent, the intended sequence is: clone the repository, create the Python 3.11 virtual environment, install `requirements.txt`, copy `.env.example` to `.env`, configure a valid camera and SMTP credentials, run `python -m unittest`, and start the application with **`python main.py`**.
