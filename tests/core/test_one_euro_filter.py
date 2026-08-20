import pytest

from core.one_euro_filter import OneEuroFilter


def test_first_sample_returned_unfiltered():
    f = OneEuroFilter()
    assert f.filter(5.0, 0.0) == 5.0


def test_smooths_noisy_constant_signal():
    raw = []
    filtered = []
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    t = 0.0
    for i in range(20):
        value = 5.0 + (0.5 if i % 2 == 0 else -0.5)
        raw.append(value)
        filtered.append(f.filter(value, t))
        t += 1.0 / 30.0

    raw_range = max(raw[5:]) - min(raw[5:])
    filtered_range = max(filtered[5:]) - min(filtered[5:])
    assert filtered_range < raw_range


def test_tracks_step_change_within_tolerance():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    t = 0.0
    for _ in range(10):
        f.filter(0.0, t)
        t += 1.0 / 30.0

    last = None
    for _ in range(60):
        last = f.filter(10.0, t)
        t += 1.0 / 30.0

    assert last == pytest.approx(10.0, abs=0.1)


def test_reset_clears_lag_state():
    f = OneEuroFilter()
    f.filter(5.0, 0.0)
    f.filter(6.0, 1.0)
    f.reset()
    assert f.filter(100.0, 2.0) == 100.0


def test_duplicate_timestamp_does_not_raise():
    f = OneEuroFilter()
    f.filter(5.0, 1.0)
    result = f.filter(6.0, 1.0)
    assert result == 5.0


@pytest.mark.parametrize("kwargs", [{"min_cutoff": 0}, {"d_cutoff": 0}, {"min_cutoff": -1}])
def test_rejects_non_positive_cutoffs(kwargs):
    with pytest.raises(ValueError):
        OneEuroFilter(**kwargs)
