import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from birdwatcher.alerts import EmailRetryQueue, send_or_queue
from birdwatcher.domain import IdentificationResult


class EmailRetryQueueTests(unittest.TestCase):
    def _identification(self):
        return IdentificationResult(
            display_name="Northern Cardinal",
            candidate_name="Northern Cardinal",
            confidence=0.9,
            margin=0.7,
            votes=4,
            frame_count=4,
            uncertain=False,
            top_candidates=(("Northern Cardinal", 0.9),),
        )

    def test_send_or_queue_persists_failed_alert(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "bird.jpg"
            image.write_bytes(b"jpg")
            queue = EmailRetryQueue(root / "queue", base_delay_seconds=1)
            with patch("birdwatcher.alerts.send_email", side_effect=OSError("smtp down")):
                sent = send_or_queue(queue, object(), self._identification(), datetime.now(), image)
            self.assertFalse(sent)
            self.assertEqual(len(list(queue.directory.glob("*.json"))), 1)

    def test_due_retry_sends_and_deletes_item(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "bird.jpg"
            image.write_bytes(b"jpg")
            queue = EmailRetryQueue(root / "queue", base_delay_seconds=1)
            path = queue.enqueue(self._identification(), datetime(2026, 7, 22, 12, 0, 0), image)
            payload = json.loads(path.read_text())
            payload["next_attempt_at"] = 0
            path.write_text(json.dumps(payload))
            with patch("birdwatcher.alerts.send_email") as send:
                sent, failed = queue.retry_due(object(), now=10)
            self.assertEqual((sent, failed), (1, 0))
            send.assert_called_once()
            self.assertFalse(path.exists())

    def test_corrupt_item_is_quarantined_once(self):
        with TemporaryDirectory() as temp:
            queue = EmailRetryQueue(Path(temp) / "queue")
            corrupt = queue.directory / "broken.json"
            corrupt.write_text("{not-json", encoding="utf-8")
            sent, failed = queue.retry_due(object(), now=10)
            self.assertEqual((sent, failed), (0, 1))
            self.assertFalse(corrupt.exists())
            self.assertEqual(len(list(queue.failed_directory.glob("*.json"))), 1)
            self.assertEqual(queue.retry_due(object(), now=20), (0, 0))

    def test_max_attempts_moves_item_to_dead_letter(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            image = root / "bird.jpg"
            image.write_bytes(b"jpg")
            queue = EmailRetryQueue(root / "queue", max_attempts=2, base_delay_seconds=1)
            path = queue.enqueue(self._identification(), datetime(2026, 7, 22, 12, 0, 0), image)
            payload = json.loads(path.read_text())
            payload["next_attempt_at"] = 0
            path.write_text(json.dumps(payload))
            with patch("birdwatcher.alerts.send_email", side_effect=OSError("still down")):
                self.assertEqual(queue.retry_due(object(), now=10), (0, 1))
                payload = json.loads(path.read_text())
                self.assertEqual(payload["attempts"], 1)
                self.assertEqual(queue.retry_due(object(), now=11), (0, 1))
            self.assertFalse(path.exists())
            self.assertEqual(len(list(queue.failed_directory.glob("*.json"))), 1)


if __name__ == "__main__":
    unittest.main()
