from core.pose_engine import _should_sample_spine, _SPINE_SAMPLE_INTERVAL_SECS


def test_should_sample_spine_true_when_interval_elapsed():
    assert _should_sample_spine(last_sample_time=0.0, now=0.3, interval=0.25) is True


def test_should_sample_spine_false_when_interval_not_elapsed():
    assert _should_sample_spine(last_sample_time=0.0, now=0.1, interval=0.25) is False


def test_should_sample_spine_true_on_exact_boundary():
    assert _should_sample_spine(last_sample_time=1.0, now=1.25, interval=0.25) is True


def test_should_sample_spine_true_for_first_ever_call():
    # last_sample_time starts at 0.0 (PoseEngine.__init__ default); real
    # time.time() values are always far larger, so the very first call in a
    # live session always samples immediately.
    assert _should_sample_spine(last_sample_time=0.0, now=1_700_000_000.0, interval=0.25) is True


def test_should_sample_spine_uses_default_interval():
    assert _should_sample_spine(last_sample_time=0.0, now=_SPINE_SAMPLE_INTERVAL_SECS) is True
    assert _should_sample_spine(last_sample_time=0.0, now=_SPINE_SAMPLE_INTERVAL_SECS - 0.01) is False
