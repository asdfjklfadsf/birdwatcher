"""SMTP message construction and delivery."""
from __future__ import annotations

import smtplib
import ssl
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

from .domain import EmailSettings, IdentificationResult


def send_email(
    settings: EmailSettings,
    identification: IdentificationResult,
    observed_at: datetime,
    image_path: Path,
) -> None:
    message = EmailMessage()
    if identification.uncertain:
        message["Subject"] = f"Bird spotted: Uncertain bird (possible {identification.candidate_name})"
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
