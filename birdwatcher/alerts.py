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

from .domain import IdentificationResult
from .emailer import send_email

LOG = logging.getLogger("bird_watcher")


class EmailRetryQueue:
    """Persist failed alerts, retry with bounded backoff, and quarantine bad items."""

    def __init__(
        self,
        directory: Path,
        max_attempts: int = 5,
        base_delay_seconds: float = 30.0,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        self.directory = directory
        self.failed_directory = directory / "failed"
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.directory.mkdir(parents=True, exist_ok=True)
        self.failed_directory.mkdir(parents=True, exist_ok=True)

    def enqueue(self, identification, observed_at: datetime, image_path: Path) -> Path:
        payload = {
            "identification": asdict(identification),
            "observed_at": observed_at.isoformat(),
            "image_path": str(image_path),
            "attempts": 0,
            "next_attempt_at": time.time() + self.base_delay_seconds,
        }
        target = self.directory / f"{int(time.time() * 1000)}_{uuid.uuid4().hex}.json"
        self._write_payload(target, payload)
        return target

    @staticmethod
    def _write_payload(path: Path, payload: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _load_identification(data: dict) -> IdentificationResult:
        identification = dict(data)
        identification["top_candidates"] = tuple(
            (str(name), float(score)) for name, score in identification.get("top_candidates", [])
        )
        return IdentificationResult(**identification)

    def _dead_letter(self, path: Path, reason: str) -> Path:
        target = self.failed_directory / path.name
        if target.exists():
            target = self.failed_directory / f"{path.stem}_{uuid.uuid4().hex}{path.suffix}"
        try:
            os.replace(path, target)
        except OSError:
            LOG.exception("Could not quarantine failed email queue item: %s", path)
            path.unlink(missing_ok=True)
        else:
            LOG.error("Quarantined email queue item (%s): %s", reason, target)
        return target

    def _read_payload(self, path: Path) -> dict:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("queue payload must be a JSON object")
        required = {"identification", "observed_at", "image_path"}
        missing = required - payload.keys()
        if missing:
            raise ValueError(f"queue payload missing fields: {', '.join(sorted(missing))}")
        return payload

    def retry_due(self, email_settings, now: float | None = None) -> tuple[int, int]:
        now = time.time() if now is None else now
        sent = 0
        failed = 0
        for path in sorted(self.directory.glob("*.json")):
            try:
                payload = self._read_payload(path)
                next_attempt_at = float(payload.get("next_attempt_at", 0))
                attempts = int(payload.get("attempts", 0))
                identification = self._load_identification(payload["identification"])
                observed_at = datetime.fromisoformat(payload["observed_at"])
                image_path = Path(payload["image_path"])
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                failed += 1
                LOG.exception("Invalid queued email alert: %s", path)
                self._dead_letter(path, f"invalid payload: {exc}")
                continue

            if next_attempt_at > now:
                continue
            if not image_path.exists():
                failed += 1
                self._dead_letter(path, f"missing image: {image_path}")
                continue

            try:
                send_email(email_settings, identification, observed_at, image_path)
            except Exception:
                failed += 1
                attempts += 1
                LOG.exception("Queued email retry failed: %s", path)
                if attempts >= self.max_attempts:
                    self._dead_letter(path, f"exhausted after {attempts} attempts")
                    continue
                payload["attempts"] = attempts
                payload["next_attempt_at"] = now + self.base_delay_seconds * (2 ** (attempts - 1))
                try:
                    self._write_payload(path, payload)
                except OSError as exc:
                    self._dead_letter(path, f"could not persist retry state: {exc}")
            else:
                path.unlink(missing_ok=True)
                sent += 1
        return sent, failed


def send_or_queue(queue: EmailRetryQueue, email_settings, identification, observed_at, image_path) -> bool:
    try:
        send_email(email_settings, identification, observed_at, image_path)
    except Exception:
        LOG.exception("Email failed; queueing alert for retry")
        queue.enqueue(identification, observed_at, image_path)
        return False
    return True
