"""Tests for analysis.shot_detection.detect_shots."""

import numpy as np
import pytest

from analysis.shot_detection import detect_shots, ShotEvent, _evaluate_attempt, _is_ball_near_rim
from tests.conftest import make_result, STANDARD_NAMES


# ---------------------------------------------------------------------------
# Helper: build a sequence of frames from simple specs
# ---------------------------------------------------------------------------

def _frames(specs):
    """Turn a list of (ball_cy, rim_cy) or None into mock result objects.

    ``None`` means no detections in that frame.
    Ball box is always 20×20 centred on the given y (x=300 for all frames).
    Rim box is always 40×20 centred on rim_cy (x=290), giving a realistic
    height so the proximity threshold (3× height = 60px) catches typical
    approach trajectories.
    """
    results = []
    for spec in specs:
        if spec is None:
            results.append(make_result(STANDARD_NAMES, []))
        else:
            ball_cy, rim_cy = spec
            results.append(make_result(
                STANDARD_NAMES,
                [
                    ("ball", 290, ball_cy - 10, 310, ball_cy + 10, 0.9),
                    ("rim",  270, rim_cy  - 10, 310, rim_cy  + 10, 0.9),
                ],
            ))
    return results


# ---------------------------------------------------------------------------
# _is_ball_near_rim
# ---------------------------------------------------------------------------

class TestIsBallNearRim:
    def test_ball_directly_on_rim_is_near(self):
        rim_box = [270, 295, 310, 305]  # centre (290, 300), 40×10
        assert _is_ball_near_rim(290, 300, rim_box) is True

    def test_ball_very_far_away_is_not_near(self):
        rim_box = [270, 295, 310, 305]
        assert _is_ball_near_rim(1000, 1000, rim_box) is False

    def test_ball_just_inside_horizontal_threshold(self):
        rim_box = [0, 90, 40, 110]  # centre (20, 100), w=40
        # w_scale=2 → threshold = 80, ball at cx=99 → |99-20|=79 < 80
        assert _is_ball_near_rim(99, 100, rim_box) is True

    def test_ball_just_outside_horizontal_threshold(self):
        rim_box = [0, 90, 40, 110]  # centre (20, 100), w=40
        # threshold = 80, ball at cx=101 → |101-20|=81 > 80
        assert _is_ball_near_rim(101, 100, rim_box) is False


# ---------------------------------------------------------------------------
# _evaluate_attempt
# ---------------------------------------------------------------------------

class TestEvaluateAttempt:
    def _make_frames(self, ball_ys, rim_cy=300):
        return [
            {
                "frame": i,
                "ball_center": (300.0, float(by)),
                "rim_center": (290.0, float(rim_cy)),
            }
            for i, by in enumerate(ball_ys)
        ]

    def test_empty_frames_returns_none(self):
        assert _evaluate_attempt([]) is None

    def test_ball_above_to_below_rim_is_made(self):
        # Ball starts at y=250 (above rim at y=300) and ends at y=350 (below)
        frames = self._make_frames([250, 275, 300, 325, 350])
        event = _evaluate_attempt(frames)
        assert event is not None
        assert event.result == "made"

    def test_ball_stays_above_rim_is_missed(self):
        frames = self._make_frames([250, 260, 270, 265, 255])
        event = _evaluate_attempt(frames)
        assert event is not None
        assert event.result == "missed"

    def test_ball_starts_below_rim_is_missed(self):
        # Ball already below rim — not coming from above
        frames = self._make_frames([350, 360, 370])
        event = _evaluate_attempt(frames)
        assert event.result == "missed"

    def test_single_frame_attempt(self):
        frames = self._make_frames([280])
        event = _evaluate_attempt(frames)
        assert event is not None
        assert event.frame_index == 0

    def test_representative_frame_is_midpoint(self):
        frames = self._make_frames([250, 270, 290, 310, 330])
        event = _evaluate_attempt(frames)
        assert event.frame_index == 2  # index 2 is midpoint of 0–4


# ---------------------------------------------------------------------------
# detect_shots – edge cases
# ---------------------------------------------------------------------------

class TestDetectShotsEdgeCases:
    def test_empty_results_returns_empty(self):
        assert detect_shots([]) == []

    def test_no_detections_at_all_returns_empty(self):
        results = _frames([None, None, None])
        assert detect_shots(results) == []

    def test_ball_never_near_rim_returns_empty(self):
        # Ball at y=100, rim at y=800 — far apart
        results = _frames([(100, 800)] * 5)
        assert detect_shots(results) == []

    def test_fewer_than_min_frames_ignored(self):
        # Only 2 near-rim frames; default min is 3
        results = _frames([(280, 300), (290, 300)])
        assert detect_shots(results, min_approach_frames=3) == []

    def test_missing_ball_mid_sequence_splits_attempt(self):
        # 4 near-rim frames, then None (gap), then 4 more → should yield 2 events
        near = [(280, 300), (290, 300), (310, 300), (320, 300)]
        specs = near + [None] + near
        results = _frames(specs)
        events = detect_shots(results, min_approach_frames=3)
        assert len(events) == 2


# ---------------------------------------------------------------------------
# detect_shots – shot classification
# ---------------------------------------------------------------------------

class TestDetectShotsClassification:
    def test_made_shot_detected(self):
        # Ball travels from above rim to below rim over 5 frames
        rim_cy = 300
        specs = [
            (250, rim_cy),
            (270, rim_cy),
            (295, rim_cy),
            (320, rim_cy),
            (340, rim_cy),
        ]
        events = detect_shots(_frames(specs), min_approach_frames=3)
        assert len(events) == 1
        assert events[0].result == "made"

    def test_missed_shot_detected(self):
        # Ball approaches rim from above but bounces back up
        rim_cy = 300
        specs = [
            (260, rim_cy),
            (275, rim_cy),
            (285, rim_cy),
            (275, rim_cy),
            (260, rim_cy),
        ]
        events = detect_shots(_frames(specs), min_approach_frames=3)
        assert len(events) == 1
        assert events[0].result == "missed"

    def test_multiple_shots_in_sequence(self):
        rim_cy = 300
        made_attempt  = [(250, rim_cy), (275, rim_cy), (310, rim_cy), (340, rim_cy)]
        missed_attempt = [(260, rim_cy), (280, rim_cy), (270, rim_cy), (255, rim_cy)]
        # Gap between attempts: ball moves far from rim
        gap = [(100, rim_cy)]  # ball far above rim — not near
        specs = made_attempt + gap + missed_attempt
        events = detect_shots(_frames(specs), min_approach_frames=3)
        assert len(events) == 2
        assert events[0].result == "made"
        assert events[1].result == "missed"

    def test_attempt_at_end_of_video_is_captured(self):
        # Attempt runs right to the last frame with no trailing non-rim frames
        rim_cy = 300
        specs = [(250, rim_cy), (275, rim_cy), (310, rim_cy), (340, rim_cy)]
        events = detect_shots(_frames(specs), min_approach_frames=3)
        assert len(events) == 1

    def test_event_contains_correct_positions(self):
        rim_cy = 300
        specs = [(250, rim_cy), (275, rim_cy), (310, rim_cy), (340, rim_cy)]
        events = detect_shots(_frames(specs), min_approach_frames=3)
        assert len(events) == 1
        event = events[0]
        # Ball position and rim position should be finite real numbers
        assert all(np.isfinite(v) for v in event.ball_position)
        assert all(np.isfinite(v) for v in event.rim_position)

    def test_custom_min_approach_frames(self):
        rim_cy = 300
        specs = [(260, rim_cy), (280, rim_cy)]  # only 2 frames
        # With min=2 this should register; with min=3 it should not
        assert len(detect_shots(_frames(specs), min_approach_frames=2)) == 1
        assert len(detect_shots(_frames(specs), min_approach_frames=3)) == 0
