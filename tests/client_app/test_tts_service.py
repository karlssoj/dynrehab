import time
from unittest.mock import MagicMock
from client_app.services.tts_service import TTSService


def test_speak_updates_last_spoken_at(mocker):
    mocker.patch("client_app.services.tts_service.subprocess.Popen",
                 return_value=MagicMock(poll=lambda: None))
    svc = TTSService(cooldown_seconds=0.0)
    before = svc._last_spoken_at
    svc.speak("Hello")
    assert svc._last_spoken_at > before


def test_cooldown_blocks_rapid_messages(mocker):
    mocker.patch("client_app.services.tts_service.subprocess.Popen",
                 return_value=MagicMock(poll=lambda: None))
    svc = TTSService(cooldown_seconds=10.0)
    svc.speak("First")
    ts1 = svc._last_spoken_at
    svc.speak("Second")
    assert svc._last_spoken_at == ts1


def test_cooldown_allows_after_wait(mocker):
    mocker.patch("client_app.services.tts_service.subprocess.Popen",
                 return_value=MagicMock(poll=lambda: None))
    svc = TTSService(cooldown_seconds=0.05)
    svc.speak("First")
    ts1 = svc._last_spoken_at
    time.sleep(0.1)
    svc.speak("Second")
    assert svc._last_spoken_at > ts1
