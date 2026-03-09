from pathlib import Path
import uuid

from project.integrations.trainwatcher_notifier import (
    TrainWatcherNotificationSettings,
    build_completion_message,
    build_failure_message,
    build_milestone_message,
    get_register_status_message,
    get_session_credentials_path,
)


def test_get_session_credentials_path_is_scoped_and_stable():
    path = get_session_credentials_path("session123")

    assert path.endswith("credentials.json")
    assert "session123" in path


def test_register_status_message_reflects_verification_state():
    credentials_root = Path("E:/A_6/exports") / f"tw_{uuid.uuid4().hex}"
    credentials_root.mkdir(parents=True, exist_ok=True)
    ready_settings = TrainWatcherNotificationSettings(
        enabled=True,
        email="user@example.com",
        credentials_path=str(credentials_root / "credentials.json"),
    )
    Path(ready_settings.credentials_path).write_text("{}", encoding="utf-8")

    assert "user@example.com" in get_register_status_message(ready_settings)
    assert get_register_status_message(TrainWatcherNotificationSettings(enabled=False)) == "Notifications are disabled."


def test_notification_messages_contain_expected_context():
    milestone = build_milestone_message("Hotstar", "Sentiment complete", "Positive 60%")
    completed = build_completion_message("Hotstar", "Google Play Store", 500, 12.5, 0.6, 0.2)
    failed = build_failure_message("Hotstar", 4.3, "Timeout")

    assert "Sentiment complete" in milestone
    assert "500" in completed
    assert "60.0%" in completed
    assert "Timeout" in failed
