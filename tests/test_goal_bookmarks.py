from __future__ import annotations

from VideoHighlights import build_analysis_bookmarks


def test_goal_event_upgrades_overlapping_bookmark() -> None:
    bookmarks = build_analysis_bookmarks(
        original_intervals=[(10.0, 20.0), (40.0, 48.0)],
        speed_intervals=[(11.0, 13.0)],
        audio_intervals=[(14.0, 15.0)],
        requested_targets=[],
        goal_events=[
            {"t": 15.5, "side": "left", "confidence": 0.9, "reason": "ball observed inside left goal"}
        ],
        game_states=[
            {"start_s": 0.0, "end_s": 15.0, "state": "in_play"},
            {"start_s": 15.0, "end_s": 25.0, "state": "goal_left"},
            {"start_s": 25.0, "end_s": 60.0, "state": "in_play"},
        ],
    )

    assert len(bookmarks) == 2
    goal_bm = bookmarks[0]
    assert goal_bm["event_type"] == "goal"
    assert goal_bm["label"] == "goal_detected"
    assert goal_bm["confidence"] >= 0.9
    assert "ball_tracking" in goal_bm["sources"]
    assert goal_bm["occurred_at_s"] == 15.5
    assert goal_bm["signals"]["goal_side"] == "left"
    assert goal_bm["game_state"] == "goal_left"

    other = bookmarks[1]
    assert other["event_type"] != "goal" or other["label"] != "goal_detected"
    assert other["game_state"] == "in_play"


def test_bookmarks_without_goal_events_unchanged() -> None:
    bookmarks = build_analysis_bookmarks(
        original_intervals=[(5.0, 12.0)],
        speed_intervals=[(6.0, 7.0)],
        audio_intervals=[],
        requested_targets=["shot"],
    )
    assert len(bookmarks) == 1
    assert bookmarks[0]["event_type"] == "shot"
    assert bookmarks[0]["game_state"] is None
