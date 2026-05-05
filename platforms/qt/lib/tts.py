import threading
import queue as _queue_mod
import rospy
from qt_robot_interface.srv import speech_say

_q: _queue_mod.Queue = _queue_mod.Queue()
_say = None


def _worker():
    global _say
    while True:
        text = _q.get()
        if text is None:
            _q.task_done()
            break
        try:
            if _say is None:
                rospy.wait_for_service('/qt_robot/speech/say', timeout=5.0)
                _say = rospy.ServiceProxy('/qt_robot/speech/say', speech_say)
            _say(str(text))
        except Exception as e:
            rospy.logwarn(f"[tts] speech failed: {e}")
            _say = None
        finally:
            _q.task_done()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


def speak(text: str) -> None:
    """Queue text for speech. Returns immediately."""
    _q.put(str(text))


def speak_sync(text: str) -> None:
    """Queue text and block until the worker has spoken it.

    Must only be called serially from a single thread and must not be
    interleaved with concurrent speak() calls — Queue.join() waits for
    all outstanding items, not only the item enqueued by this call.
    """
    _q.put(str(text))
    _q.join()


def stop() -> None:
    """Signal the worker thread to exit. Safe to call only once."""
    if _worker_thread.is_alive():
        _q.put(None)
    _worker_thread.join(timeout=2.0)
