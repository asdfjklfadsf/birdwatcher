"""Alert delivery with a small persistent retry queue."""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import legacy_main as core

LOG = logging.getLogger("bird_watcher")


class EmailRetryQueue:
    """Persist failed email alerts and retry them with bounded backoff."""

    def __init__(self, directory: Path, max_attempts: int = 5, base_delay_seconds: float = 30.0):
        self.directory = directory
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.directory.mkdir(parents=True, exist_ok=True)

    def enqueue(self, identification, observed_at: datetime, image_path: Path) -> Path:
        payload = {
            "identification": asdict(identification),
            "observed_at": observed_at.isoformat(),
            "image_path": str(image_path),
            "attempts": 0,
            "next_attempt_at": time.time() + self.base_delay_seconds,
        }
        target = self.directory / f"{int(time.time() * 1000)}_{uuid.uuid4().hex}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, target)
        return target

    def _load_identification(self, data: dict):
        identification = dict(data)
        identification["top_candidates"] = tuple(
            (str(name), float(score)) for name, score in identification.get("top_candidates", [])
        )
        return core.IdentificationResult(**identification)

    def retry_due(self, email_settings, now: float | None = None) -> tuple[int, int]:
        now = time.time() if now is None else now
        sent = 0
        failed = 0
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if float(payload.get("next_attempt_at", 0)) > now:
                    continue
                image_path = Path(payload["image_path"])
                if not image_path.exists():
                    LOG.error("Dropping queued alert because image is missing: %s", image_path)
                    path.unlink(missing_ok=True)
                    failed += 1
                    continue
                identification = self._load_identification(payload["identification"])
                observed_at = datetime.fromisoformat(payload["observed_at"])
                core.send_email(email_settings, identification, observed_at, image_path)
                path.unlink(missing_ok=True)
                sent += 1
            except Exception:
                failed += 1
                LOG.exception("Queued email retry failed: %s", path)
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    attempts = int(payload.get("attempts", 0)) + 1
                    if attempts >= self.max_attempts:
                        LOG.error("Dropping queued alert after %d attempts: %s", attempts, path)
                        path.unlink(missing_ok=True)
                        continue
                    payload["attempts"] = attempts
                    payload["next_attempt_at"] = now + self.base_delay_seconds * (2 ** (attempts - 1))
                    temporary = path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(payload), encoding="utf-8")
                    os.replace(temporary, path)
                except Exception:
                    LOG.exception("Could not update failed queue item: %s", path)
        return sent, failed


def send_or_queue(queue: EmailRetryQueue, email_settings, identification, observed_at, image_path) -> bool:
    try:
        core.send_email(email_settings, identification, observed_at, image_path)
    except Exception:
        LOG.exception("Email failed; queueing alert for retry")
        queue.enqueue(identification, observed_at, image_path)
        return False
    return True
