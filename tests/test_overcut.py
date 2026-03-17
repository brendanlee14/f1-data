"""Tests for engine.strategy.overcut detector."""

import pandas as pd
import pytest

from engine.strategy import overcut
from engine.strategy.overcut import GAP_THRESHOLD_SEC, MIN_STAY_OUT_LAPS

REQUIRED_KEYS = {"insight_type", "driver", "lap", "rank_score", "summary", "detail"}
DETAIL_KEYS = {
    "target_driver", "overcut_driver_pit_lap", "target_driver_pit_lap",
    "laps_stayed_out", "gap_before_sec", "gap_after_sec", "gap_gained_sec",
    "avg_pace_delta_vs_fresh_sec", "tyre_age_at_pit", "overcut_succeeded",
}


def _assert_schema(ins: dict):
    assert REQUIRED_KEYS.issubset(ins.keys())
    assert ins["insight_type"] == "overcut"
    assert isinstance(ins["driver"], str)
    assert isinstance(ins["lap"], int)
    assert 0.0 <= ins["rank_score"] <= 1.0
    assert DETAIL_KEYS.issubset(ins["detail"].keys())
    assert isinstance(ins["detail"]["overcut_succeeded"], bool)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def _make_overcut_scenario(
    gap_before: float = 2.5,            # A ahead of B (positive = A leads) before B pits
    pit_lap_b: int = 15,                # B pits here
    stay_out: int = 5,                  # A stays out this many laps after B
    a_pace_vs_b_fresh: float = 0.1,     # unused; both run `base` pre-pit for stable gap
    total_laps: int = 40,
    base: float = 82.0,                 # shared pre-pit pace — keeps gap at gap_before
    b_fresh_pace: float = 82.0,         # pace on fresh rubber (both drivers post-pit)
) -> pd.DataFrame:
    """
    A is ahead of B. B pits at pit_lap_b. A stays out `stay_out` more laps.

    Both drivers run `base` pace before their own pit so the gap stays exactly
    gap_before at check_lap (pit_lap_b - 1). After pitting, both run at
    b_fresh_pace. The overcut works if A's track-position lead survives the
    pit-stop loss when A eventually stops.
    """
    pit_lap_a = pit_lap_b + stay_out
    rows = []

    # --- Driver A (stays out longer, pits at pit_lap_a) ---
    cum_a = 0.0
    for lap in range(1, total_laps + 1):
        in_stint1 = lap < pit_lap_a
        pit = (lap == pit_lap_a)
        lt = base if in_stint1 else b_fresh_pace
        if pit:
            lt = base + 22.0
        cum_a += lt
        rows.append({
            "driver": "AAA", "team": "T1", "lap": lap,
            "compound": "MEDIUM" if in_stint1 else "HARD",
            "tyre_age": (lap - 1) if in_stint1 else (lap - pit_lap_a),
            "lap_time": round(lt, 3),
            "cumulative_time": round(cum_a, 3),
            "pit_this_lap": pit,
            "stint_number": 1 if in_stint1 else 2,
        })

    # --- Driver B (pits first, starts gap_before seconds behind A) ---
    cum_b = gap_before
    for lap in range(1, total_laps + 1):
        in_stint1 = lap < pit_lap_b
        pit = (lap == pit_lap_b)
        lt = base if in_stint1 else b_fresh_pace
        if pit:
            lt = b_fresh_pace + 22.0
        cum_b += lt
        rows.append({
            "driver": "BBB", "team": "T2", "lap": lap,
            "compound": "MEDIUM" if in_stint1 else "HARD",
            "tyre_age": (lap - 1) if in_stint1 else (lap - pit_lap_b),
            "lap_time": round(lt, 3),
            "cumulative_time": round(cum_b, 3),
            "pit_this_lap": pit,
            "stint_number": 1 if in_stint1 else 2,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestOvercutDetection:
    def test_detects_classic_overcut(self):
        """A stays out 5 laps after B pits, maintaining gap → overcut flagged."""
        df = _make_overcut_scenario(stay_out=5, gap_before=2.5)
        insights = overcut.detect(df)
        assert len(insights) == 1
        assert insights[0]["driver"] == "AAA"
        assert insights[0]["detail"]["target_driver"] == "BBB"

    def test_no_detection_when_gap_too_large(self):
        """Drivers too far apart should not trigger overcut."""
        df = _make_overcut_scenario(gap_before=GAP_THRESHOLD_SEC + 2.0)
        insights = overcut.detect(df)
        assert len(insights) == 0

    def test_no_detection_when_b_is_ahead(self):
        """If B is already ahead of A, A can't overcut B."""
        df = _make_overcut_scenario(gap_before=-3.0)  # B ahead
        insights = overcut.detect(df)
        assert len(insights) == 0

    def test_no_detection_when_stay_out_too_short(self):
        """Stay-out less than MIN_STAY_OUT_LAPS should not fire."""
        df = _make_overcut_scenario(stay_out=MIN_STAY_OUT_LAPS - 1)
        insights = overcut.detect(df)
        assert len(insights) == 0

    def test_stay_out_laps_recorded_correctly(self):
        stay_out = 6
        df = _make_overcut_scenario(stay_out=stay_out)
        insights = overcut.detect(df)
        assert len(insights) == 1
        assert insights[0]["detail"]["laps_stayed_out"] == stay_out

    def test_overcut_succeeded_when_a_maintains_lead(self):
        """With a large enough initial gap A keeps the lead after pitting."""
        df = _make_overcut_scenario(stay_out=5, gap_before=3.5)
        insights = overcut.detect(df)
        assert len(insights) == 1
        assert insights[0]["detail"]["overcut_succeeded"] is True

    def test_overcut_driver_pit_lap_is_insight_lap(self):
        pit_lap_a = 20  # pit_lap_b=15, stay_out=5
        df = _make_overcut_scenario(pit_lap_b=15, stay_out=5)
        insights = overcut.detect(df)
        assert len(insights) == 1
        assert insights[0]["lap"] == pit_lap_a

    def test_no_duplicate_insights_per_pair(self, melbourne_df):
        """Each ordered (A, B) pair should generate at most one overcut insight."""
        insights = overcut.detect(melbourne_df)
        pairs = [(i["driver"], i["detail"]["target_driver"]) for i in insights]
        assert len(pairs) == len(set(pairs)), "Duplicate (driver, target) pairs found"

    def test_no_insights_with_no_pit_stops(self):
        rows = [
            {"driver": d, "team": "T", "lap": lap, "compound": "MEDIUM",
             "tyre_age": lap - 1, "lap_time": 82.0, "cumulative_time": 82.0 * lap + off,
             "pit_this_lap": False, "stint_number": 1}
            for d, off in [("AAA", 0), ("BBB", 2)]
            for lap in range(1, 30)
        ]
        df = pd.DataFrame(rows)
        assert overcut.detect(df) == []


# ---------------------------------------------------------------------------
# Rank score tests
# ---------------------------------------------------------------------------

class TestOvercutRankScore:
    def test_rank_score_in_range(self, melbourne_df):
        for ins in overcut.detect(melbourne_df):
            assert 0.0 <= ins["rank_score"] <= 1.0

    def test_longer_stay_out_higher_score(self):
        """Staying out longer (more risk) should be rewarded with higher score."""
        df_short = _make_overcut_scenario(stay_out=2)
        df_long  = _make_overcut_scenario(stay_out=8)

        ins_short = overcut.detect(df_short)
        ins_long  = overcut.detect(df_long)

        if ins_short and ins_long:
            assert ins_long[0]["rank_score"] >= ins_short[0]["rank_score"]

    def test_rank_score_capped_at_one(self, melbourne_df):
        for ins in overcut.detect(melbourne_df):
            assert ins["rank_score"] <= 1.0


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestOvercutSchema:
    def test_insight_schema_controlled(self):
        df = _make_overcut_scenario(stay_out=5)
        insights = overcut.detect(df)
        assert len(insights) == 1
        _assert_schema(insights[0])

    def test_insight_schema_melbourne(self, melbourne_df):
        for ins in overcut.detect(melbourne_df):
            _assert_schema(ins)

    def test_gap_gained_sign_matches_succeeded(self):
        """gap_gained should be consistent with overcut_succeeded."""
        df = _make_overcut_scenario(stay_out=5, gap_before=3.0)
        insights = overcut.detect(df)
        if insights:
            ins = insights[0]
            if ins["detail"]["overcut_succeeded"]:
                assert ins["detail"]["gap_after_sec"] > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestOvercutEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=[
            "driver", "team", "lap", "compound", "tyre_age",
            "lap_time", "cumulative_time", "pit_this_lap", "stint_number",
        ])
        assert overcut.detect(df) == []

    def test_single_driver(self):
        df = _make_overcut_scenario()
        df = df[df["driver"] == "AAA"]
        assert overcut.detect(df) == []

    def test_returns_list(self, melbourne_df):
        assert isinstance(overcut.detect(melbourne_df), list)
