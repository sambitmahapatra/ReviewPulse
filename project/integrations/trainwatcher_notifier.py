from __future__ import annotations

import json
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainWatcherNotificationSettings:
    enabled: bool = False
    milestones_enabled: bool = False
    email: str = ""
    credentials_path: str = ""
    base_url: str = ""


def get_session_credentials_path(session_id: str) -> str:
    root = Path(tempfile.gettempdir()) / "reviewpulse_trainwatcher" / session_id
    root.mkdir(parents=True, exist_ok=True)
    return str(root / "credentials.json")


def get_register_status_message(settings: TrainWatcherNotificationSettings) -> str:
    if not settings.enabled:
        return "Notifications are disabled."
    if settings.credentials_path and Path(settings.credentials_path).exists():
        return f"Notifications will be sent to {settings.email or 'the verified email'}."
    return "Verify an email to enable TrainWatcher notifications for this session."


def send_verification_code(email: str, base_url: str = "") -> str:
    from trainwatcher.cloud import get_base_url
    from trainwatcher.exceptions import TrainWatcherError

    resolved_base_url = get_base_url(base_url or None)
    payload = json.dumps({"email": email}).encode("utf-8")
    request = urllib.request.Request(
        f"{resolved_base_url}/register",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ReviewPulse/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            raw = response.read().decode("utf-8")
    except Exception as exc:
        raise TrainWatcherError(f"Failed to send verification code: {exc}") from exc

    if raw:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("error"):
            raise TrainWatcherError(str(data["error"]))

    return resolved_base_url


def verify_session_email(email: str, code: str, credentials_path: str, base_url: str = "") -> str:
    from trainwatcher import verify_email

    return verify_email(
        email=email,
        code=code,
        base_url=base_url or None,
        api_key_path=credentials_path,
    )


def clear_session_credentials(credentials_path: str) -> None:
    if credentials_path and Path(credentials_path).exists():
        Path(credentials_path).unlink()


def trainwatcher_available() -> tuple[bool, str]:
    try:
        import trainwatcher  # noqa: F401

        return True, ""
    except Exception as exc:
        return False, str(exc)


def build_milestone_message(analysis_label: str, stage: str, detail: str = "") -> str:
    message = f"ReviewPulse milestone for {analysis_label}: {stage}."
    if detail.strip():
        message += f"\n\n{detail.strip()}"
    return message


def build_completion_message(
    analysis_label: str,
    source_name: str,
    review_count: int,
    runtime_seconds: float,
    positive_share: float,
    negative_share: float,
) -> str:
    return (
        f"ReviewPulse analysis completed for {analysis_label}.\n\n"
        f"Source: {source_name}\n"
        f"Reviews analyzed: {review_count:,}\n"
        f"Runtime: {runtime_seconds:.1f}s\n"
        f"Positive sentiment: {positive_share * 100:.1f}%\n"
        f"Negative sentiment: {negative_share * 100:.1f}%"
    )


def build_failure_message(analysis_label: str, runtime_seconds: float, error_text: str) -> str:
    return (
        f"ReviewPulse analysis failed for {analysis_label or 'the current run'}.\n\n"
        f"Runtime before failure: {runtime_seconds:.1f}s\n"
        f"Error: {error_text.strip() or 'Unknown error'}"
    )


class TrainWatcherNotifier:
    def __init__(self, settings: TrainWatcherNotificationSettings) -> None:
        self.settings = settings
        self._start_time = time.perf_counter()
        self.last_error = ""
        self._cloud = None

        if not settings.enabled:
            return

        try:
            from trainwatcher import cloud

            self._cloud = cloud
        except Exception as exc:
            self.last_error = str(exc)

    @property
    def is_ready(self) -> bool:
        return bool(
            self.settings.enabled
            and self._cloud is not None
            and self.settings.credentials_path
            and Path(self.settings.credentials_path).exists()
        )

    def _send(self, message: str, subject: str) -> None:
        if not self.is_ready:
            return
        try:
            self._cloud.send_notification(
                message=message,
                subject=subject,
                base_url=self.settings.base_url or None,
                api_key_path=self.settings.credentials_path,
            )
        except Exception as exc:
            self.last_error = str(exc)

    def milestone(self, analysis_label: str, stage: str, detail: str = "") -> None:
        if not self.settings.milestones_enabled:
            return
        self._send(
            build_milestone_message(analysis_label, stage, detail),
            subject=f"ReviewPulse Milestone: {stage}",
        )

    def complete(
        self,
        analysis_label: str,
        source_name: str,
        review_count: int,
        positive_share: float,
        negative_share: float,
    ) -> None:
        runtime_seconds = time.perf_counter() - self._start_time
        self._send(
            build_completion_message(
                analysis_label=analysis_label,
                source_name=source_name,
                review_count=review_count,
                runtime_seconds=runtime_seconds,
                positive_share=positive_share,
                negative_share=negative_share,
            ),
            subject="ReviewPulse Analysis Completed",
        )

    def fail(self, analysis_label: str, error_text: str) -> None:
        runtime_seconds = time.perf_counter() - self._start_time
        self._send(
            build_failure_message(
                analysis_label=analysis_label,
                runtime_seconds=runtime_seconds,
                error_text=error_text,
            ),
            subject="ReviewPulse Analysis Failed",
        )
