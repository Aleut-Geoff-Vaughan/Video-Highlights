from __future__ import annotations

import numpy as np

from VideoHighlights import detect_review_candidate_intervals


def test_review_candidate_fallback_selects_strong_motion_windows() -> None:
    times = np.arange(0, 60, dtype=float)
    speed = np.zeros_like(times)
    direction = np.zeros_like(times)
    speed[10] = 4.0
    speed[35] = 6.0
    direction[35] = np.pi

    intervals = detect_review_candidate_intervals(
        times=times,
        speed=speed,
        direction_changes=direction,
        pre=2.0,
        post=4.0,
        max_candidates=2,
    )

    assert len(intervals) == 2
    assert intervals[0] == (8.0, 14.0)
    assert intervals[1] == (33.0, 39.0)


def test_review_candidate_fallback_stays_empty_without_signal() -> None:
    times = np.arange(0, 10, dtype=float)
    intervals = detect_review_candidate_intervals(
        times=times,
        speed=np.zeros_like(times),
        direction_changes=np.zeros_like(times),
        pre=2.0,
        post=4.0,
    )

    assert intervals == []
