from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.services.player_focus import choose_target_track_id, resolve_player_roi_box, stitch_target_track


@dataclass
class _Point:
    t: float
    xy: tuple[float, float]
    bbox: tuple[float, float, float, float]


def test_resolve_player_roi_box_from_normalized_coords() -> None:
    roi = {
        "normalized": True,
        "x1_norm": 0.25,
        "y1_norm": 0.10,
        "x2_norm": 0.50,
        "y2_norm": 0.60,
    }

    result = resolve_player_roi_box(roi, frame_width=200, frame_height=100)
    assert result == (50.0, 10.0, 100.0, 60.0)


def test_resolve_player_roi_box_from_xywh_normalized() -> None:
    roi = {
        "normalized": True,
        "x": 0.20,
        "y": 0.30,
        "w": 0.15,
        "h": 0.25,
    }

    result = resolve_player_roi_box(roi, frame_width=400, frame_height=200)
    assert result == pytest.approx((80.0, 60.0, 140.0, 110.0))


def test_resolve_player_roi_box_clamps_and_rejects_tiny_boxes() -> None:
    assert resolve_player_roi_box({"normalized": True, "x1_norm": 0.2, "y1_norm": 0.2, "x2_norm": 0.201, "y2_norm": 0.21}, 100, 100) is None
    clamped = resolve_player_roi_box({"x": -10, "y": -5, "w": 40, "h": 50}, 100, 100)
    assert clamped == (0.0, 0.0, 30.0, 45.0)


def test_choose_target_track_id_prefers_track_with_strong_roi_overlap() -> None:
    user_box = (40.0, 10.0, 70.0, 90.0)
    tracks = {
        3: [
            _Point(t=0.0, xy=(88.0, 50.0), bbox=(76.0, 18.0, 100.0, 88.0)),
            _Point(t=0.5, xy=(90.0, 50.0), bbox=(78.0, 18.0, 102.0, 88.0)),
        ],
        7: [
            _Point(t=0.0, xy=(55.0, 50.0), bbox=(43.0, 16.0, 68.0, 88.0)),
            _Point(t=0.5, xy=(57.0, 50.0), bbox=(45.0, 16.0, 70.0, 88.0)),
        ],
    }

    assert choose_target_track_id(tracks, user_box, window_t=1.0) == 7


def test_stitch_target_track_merges_adjacent_track_fragments() -> None:
    tracks = {
        11: [
            _Point(t=0.0, xy=(100.0, 80.0), bbox=(88.0, 42.0, 112.0, 118.0)),
            _Point(t=0.5, xy=(102.0, 80.0), bbox=(90.0, 42.0, 114.0, 118.0)),
            _Point(t=1.0, xy=(104.0, 80.0), bbox=(92.0, 42.0, 116.0, 118.0)),
        ],
        21: [
            _Point(t=1.2, xy=(106.0, 80.0), bbox=(94.0, 42.0, 118.0, 118.0)),
            _Point(t=1.7, xy=(109.0, 80.0), bbox=(97.0, 42.0, 121.0, 118.0)),
        ],
        99: [
            _Point(t=1.1, xy=(240.0, 90.0), bbox=(225.0, 45.0, 255.0, 125.0)),
            _Point(t=1.6, xy=(244.0, 90.0), bbox=(229.0, 45.0, 259.0, 125.0)),
        ],
    }

    stitched_ids, stitched = stitch_target_track(tracks, 11, max_gap_seconds=0.5)

    assert stitched_ids == [11, 21]
    assert [round(point.t, 2) for point in stitched] == [0.0, 0.5, 1.0, 1.2, 1.7]
