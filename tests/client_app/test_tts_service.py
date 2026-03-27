import time
from unittest.mock import MagicMock
from client_app.services.tts_service import TTSService


def test_speak_calls_tts_engine(mocker):
    mock_engine = MagicMock()
    mocker.patch("client_app.services.tts_service.pyttsx3.init", return_value=mock_engine)
    svc = TTSService(cooldown_seconds=0.0)
    svc.speak("Hello")
    time.sleep(0.1)
    mock_engine.say.assert_called_once_with("Hello")


def test_cooldown_blocks_rapid_messages(mocker):
    mock_engine = MagicMock()
    mocker.patch("client_app.services.tts_service.pyttsx3.init", return_value=mock_engine)
    svc = TTSService(cooldown_seconds=10.0)
    svc.speak("First")
    svc.speak("Second")  # should be blocked by cooldown
    time.sleep(0.2)
    assert mock_engine.say.call_count == 1


def test_cooldown_allows_after_wait(mocker):
    mock_engine = MagicMock()
    mocker.patch("client_app.services.tts_service.pyttsx3.init", return_value=mock_engine)
    svc = TTSService(cooldown_seconds=0.05)
    svc.speak("First")
    time.sleep(0.15)
    svc.speak("Second")
    time.sleep(0.2)
    assert mock_engine.say.call_count == 2
