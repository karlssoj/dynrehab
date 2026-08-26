"""One Euro Filter (Casiez, Casalta, Roussel, Vogel, 2012) — adaptive
low-latency low-pass filter for jittery real-time signals.

Reference: https://cristal.univ-lille.fr/~casiez/1euro/
"""
from __future__ import annotations

import math


def _smoothing_factor(t_e: float, cutoff: float) -> float:
    r = 2.0 * math.pi * cutoff * t_e
    return r / (r + 1.0)


def _exponential_smoothing(a: float, x: float, x_prev: float) -> float:
    return a * x + (1.0 - a) * x_prev


class OneEuroFilter:
    """Adaptive low-pass filter for a single scalar signal.

    Parameters
    ----------
    min_cutoff:
        Minimum cutoff frequency (Hz). Lower values mean more smoothing
        at low speeds, at the cost of more lag.
    beta:
        Speed coefficient. Higher values reduce lag during fast motion,
        at the cost of less smoothing while moving quickly.
    d_cutoff:
        Cutoff frequency used to filter the derivative (speed) estimate.
    """

    __slots__ = ("min_cutoff", "beta", "d_cutoff", "_x_prev", "_dx_prev", "_t_prev")

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        if min_cutoff <= 0:
            raise ValueError("min_cutoff must be > 0")
        if d_cutoff <= 0:
            raise ValueError("d_cutoff must be > 0")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def filter(self, value: float, timestamp: float) -> float:
        """Filter one new sample. `timestamp` must be monotonically
        increasing (e.g. time.time() or time.monotonic()).

        The first call (or the first call after reset()) stores the raw
        value and returns it unmodified, so there is no startup lag.
        """
        if self._t_prev is None:
            self._x_prev = value
            self._dx_prev = 0.0
            self._t_prev = timestamp
            return value

        t_e = timestamp - self._t_prev
        if t_e <= 0.0:
            return self._x_prev

        a_d = _smoothing_factor(t_e, self.d_cutoff)
        dx = (value - self._x_prev) / t_e
        dx_hat = _exponential_smoothing(a_d, dx, self._dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = _smoothing_factor(t_e, cutoff)
        x_hat = _exponential_smoothing(a, value, self._x_prev)

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = timestamp
        return x_hat

    def reset(self) -> None:
        """Clear all state so the next filter() call behaves like the
        very first sample ever seen."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None
