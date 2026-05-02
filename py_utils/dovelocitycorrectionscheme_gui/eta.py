"""ETA estimator for a running simulation.

Each time a new checkpoint is dropped, call :meth:`record(idx, t_sim)`. The
estimator keeps the last ``window`` points and fits a linear ``wall_time = a *
t_sim + b``. ``eta_seconds`` extrapolates to a target final sim time.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass


@dataclass
class ETASample:
    t_sim: float
    wall: float


class ETAEstimator:
    def __init__(self, window: int = 12):
        self.window = window
        self._samples: deque[ETASample] = deque(maxlen=window)
        self._t0: float | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._t0 = None

    def start(self) -> None:
        self._t0 = time.monotonic()

    def record(self, t_sim: float) -> None:
        if self._t0 is None:
            self._t0 = time.monotonic()
        self._samples.append(ETASample(t_sim=t_sim, wall=time.monotonic() - self._t0))

    @property
    def n(self) -> int:
        return len(self._samples)

    def slope(self) -> float | None:
        if self.n < 2:
            return None
        n = self.n
        sx = sum(s.t_sim for s in self._samples)
        sy = sum(s.wall for s in self._samples)
        sxx = sum(s.t_sim * s.t_sim for s in self._samples)
        sxy = sum(s.t_sim * s.wall for s in self._samples)
        denom = n * sxx - sx * sx
        if denom <= 0:
            return None
        return (n * sxy - sx * sy) / denom

    def eta_seconds(self, t_sim_target: float) -> float | None:
        if not self._samples:
            return None
        last = self._samples[-1]
        if t_sim_target <= last.t_sim:
            return 0.0
        sl = self.slope()
        if sl is None or sl <= 0:
            if last.t_sim <= 0:
                return None
            sl = last.wall / last.t_sim
            if sl <= 0:
                return None
        return sl * (t_sim_target - last.t_sim)

    def progress(self, t_sim_target: float) -> float:
        if not self._samples or t_sim_target <= 0:
            return 0.0
        return min(1.0, self._samples[-1].t_sim / t_sim_target)


def fmt_seconds(s: float | None) -> str:
    if s is None:
        return "—"
    s = max(0.0, float(s))
    if s < 60:
        return f"{s:.0f}s"
    m, ss = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m {int(ss):02d}s"
    h, mm = divmod(m, 60)
    return f"{int(h)}h {int(mm):02d}m"
