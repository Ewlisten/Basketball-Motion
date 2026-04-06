"""Tests for analysis.possession.detect_possession."""

import pytest

from analysis.possession import detect_possession, _iou, _center_distance
from tests.conftest import make_result, STANDARD_NAMES


# ---------------------------------------------------------------------------
# Helper geometry
# ---------------------------------------------------------------------------

class TestIou:
    def test_identical_boxes_return_one(self):
        box = [0, 0, 10, 10]
        assert _iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_boxes_return_zero(self):
        assert _iou([0, 0, 5, 5], [10, 10, 20, 20]) == pytest.approx(0.0)

    def test_partial_overlap(self):
        # Two 10×10 boxes overlapping by a 5×5 square
        iou = _iou([0, 0, 10, 10], [5, 5, 15, 15])
        # inter=25, union=200-25=175 → 25/175
        assert iou == pytest.approx(25 / 175)

    def test_one_box_inside_other(self):
        outer = [0, 0, 10, 10]
        inner = [2, 2, 5, 5]
        iou = _iou(outer, inner)
        assert 0.0 < iou < 1.0


class TestCenterDistance:
    def test_same_box_zero_distance(self):
        box = [0, 0, 10, 10]
        assert _center_distance(box, box) == pytest.approx(0.0)

    def test_known_distance(self):
        # centres at (5,5) and (5,15) → distance 10
        assert _center_distance([0, 0, 10, 10], [0, 10, 10, 20]) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# detect_possession – edge cases
# ---------------------------------------------------------------------------

class TestDetectPossessionEdgeCases:
    def test_no_detections_returns_none(self):
        result = make_result(STANDARD_NAMES, [])
        assert detect_possession(result) is None

    def test_ball_only_returns_none(self):
        result = make_result(
            STANDARD_NAMES,
            [("ball", 100, 100, 120, 120, 0.9)],
        )
        assert detect_possession(result) is None

    def test_players_only_returns_none(self):
        result = make_result(
            STANDARD_NAMES,
            [("player", 0, 0, 50, 150, 0.9)],
        )
        assert detect_possession(result) is None

    def test_unknown_class_in_names_does_not_crash(self):
        names = {0: "unknown_class", 1: "player", 2: "rim"}
        result = make_result(names, [("player", 0, 0, 50, 150, 0.9)])
        assert detect_possession(result) is None


# ---------------------------------------------------------------------------
# detect_possession – IoU branch
# ---------------------------------------------------------------------------

class TestDetectPossessionOverlap:
    def test_single_player_overlapping_ball(self):
        # Ball overlaps with player_0
        result = make_result(
            STANDARD_NAMES,
            [
                ("ball",   10, 80, 30, 100, 0.95),   # ball at centre ~(20, 90)
                ("player",  0, 50, 60, 200, 0.90),   # player covers ball
            ],
        )
        assert detect_possession(result) == 0

    def test_two_players_returns_higher_iou(self):
        # ball overlaps player_0 more than player_1
        result = make_result(
            STANDARD_NAMES,
            [
                ("ball",    10, 80, 30, 100, 0.9),
                ("player",  0, 60, 40, 150, 0.9),   # large overlap
                ("player", 200, 200, 250, 350, 0.9), # no overlap
            ],
        )
        assert detect_possession(result) == 0

    def test_second_player_overlapping_ball(self):
        result = make_result(
            STANDARD_NAMES,
            [
                ("ball",   210, 210, 240, 240, 0.9),
                ("player",   0,   0,  50, 150, 0.9),  # far away
                ("player", 200, 200, 260, 360, 0.9),  # overlapping
            ],
        )
        assert detect_possession(result) == 1


# ---------------------------------------------------------------------------
# detect_possession – proximity (centroid) branch
# ---------------------------------------------------------------------------

class TestDetectPossessionProximity:
    def test_nearest_player_within_threshold(self):
        # Ball at (200, 200), player_0 tall box centred near ball, player_1 far
        # Player height = 100 px, ball 40 px away → within 1.5 * 100
        result = make_result(
            STANDARD_NAMES,
            [
                ("ball",   190, 190, 210, 210, 0.9),   # centre (200, 200)
                ("player", 180, 160, 220, 260, 0.9),   # centre (200, 210), height 100
                ("player", 500, 500, 550, 650, 0.9),   # far away
            ],
        )
        assert detect_possession(result) == 0

    def test_all_players_too_far_returns_none(self):
        # Ball at (200, 200), only player is far away (500 px) relative to its height (50 px)
        result = make_result(
            STANDARD_NAMES,
            [
                ("ball",   190, 190, 210, 210, 0.9),
                ("player", 700, 700, 750, 750, 0.9),   # height 50, distance >> 75
            ],
        )
        assert detect_possession(result) is None

    def test_proximity_picks_closest_of_two_distant_players(self):
        # Neither player overlaps ball; player_1 is closer
        result = make_result(
            STANDARD_NAMES,
            [
                ("ball",   100, 100, 120, 120, 0.9),   # centre (110, 110)
                ("player",   0,   0,  40, 200, 0.9),   # centre (20, 100), height 200
                ("player",  90,  80, 130, 280, 0.9),   # centre (110, 180), height 200 – closer
            ],
        )
        assert detect_possession(result) == 1
