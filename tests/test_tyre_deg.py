"""Tests for engine.strategy.tyre_deg detector."""

import pandas as pd
import pytest

from engine.strategy import tyre_deg
from engine.strategy.tyre_deg import DEG_SLOPE_THRESHOLD, SLOPE_MAX, WINDOW, MIN_STINT_LAPS
from tests.conftest import make_race_df

RACE_LAPS = 58


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _single_stint_df(driver: str, lap_times: list[float], compound: str = "MEDIUM") -> pd.DataFrame:
    """Build a single-stint DataFrame for one driver."""
    stints = [{
        "compound": compound,
        "laps": [(lap, lt, age) for age, (lap, lt) in enumerate(
            zip(range(1, len(lap_times) + 1), lap_times)
        )],
    }]
    return make_race_df([{"driver": driver, "stints": stints}])


def _degrading_lap_times(n: int, base: float = 82.0, slope: float = 0.20) -> list[float]:
    """Lap times with a clear linear degradation."""
    return [base + slope * i for i in range(n)]


def _stable_lap_times(n: int, base: float = 82.0, noise: float = 0.05) -> list[float]:
    """Flat lap times with tiny noise (well below threshold)."""
    import numpy as np
    rng = _rng()
    return [base + rng.uniform(-noise, noise) for _ in range(n)]


def _rng():
    import numpy as np
    return np.random.default_rng(999)


# ---------------------------------------------------------------------------
# Schema validation helper
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"insight_type", "driver", "lap", "rank_score", "summary", "detail"}
DETAIL_KEYS = {
    "compound", "tyre_age", "stint_number", "deg_slope_sec_per_lap",
    "window_laps", "window_lap_numbers", "window_lap_times", "stint_laps",
}


def _assert_schema(insight: dict):
    assert REQUIRED_KEYS.issubset(insight.keys()), f"Missing keys: {REQUIRED_KEYS - insight.keys()}"
    assert insight["insight_type"] == "tyre_degradation"
    assert isinstance(insight["driver"], str) and len(insight["driver"]) == 3
    assert isinstance(insight["lap"], int)
    assert 0.0 <= insight["rank_score"] <= 1.0
    assert isinstance(insight["summary"], str) and len(insight["summary"]) > 0
    assert DETAIL_KEYS.issubset(insight["detail"].keys())


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestTyreDegDetection:
    def test_detects_clear_degradation(self):
        """A steep slope well above threshold should produce an insight."""
        times = _degrading_lap_times(30, slope=0.25)
        df = _single_stint_df("VER", times)
        insights = tyre_deg.detect(df)
        assert len(insights) == 1
        assert insights[0]["driver"] == "VER"

    def test_no_detection_on_stable_stint(self):
        """Flat lap times should not trigger a degradation insight."""
        times = _stable_lap_times(30)
        df = _single_stint_df("VER", times)
        insights = tyre_deg.detect(df)
        assert len(insights) == 0

    def test_no_detection_below_min_stint_laps(self):
        """Stint shorter than MIN_STINT_LAPS + WINDOW should never fire."""
        times = _degrading_lap_times(MIN_STINT_LAPS + WINDOW - 1, slope=0.30)
        df = _single_stint_df("VER", times)
        insights = tyre_deg.detect(df)
        assert len(insights) == 0

    def test_at_most_one_insight_per_stint(self, melbourne_df):
        """Each stint should produce at most one degradation insight."""
        insights = tyre_deg.detect(melbourne_df)
        from collections import Counter
        counts = Counter((i["driver"], i["detail"]["stint_number"]) for i in insights)
        assert max(counts.values()) == 1

    def test_multiple_drivers_detected_independently(self):
        """Each degrading driver should produce their own insight."""
        times = _degrading_lap_times(30, slope=0.25)
        df = pd.concat([
            _single_stint_df("VER", times),
            _single_stint_df("NOR", times),
        ], ignore_index=True)
        insights = tyre_deg.detect(df)
        drivers = {i["driver"] for i in insights}
        assert "VER" in drivers
        assert "NOR" in drivers


# ---------------------------------------------------------------------------
# Rank score tests
# ---------------------------------------------------------------------------

class TestTyreDegRankScore:
    def test_rank_score_in_range(self, melbourne_df):
        insights = tyre_deg.detect(melbourne_df)
        for i in insights:
            assert 0.0 <= i["rank_score"] <= 1.0, f"rank_score out of range: {i}"

    def test_steeper_slope_higher_rank(self):
        """Higher deg slope should produce a higher rank score."""
        shallow = _degrading_lap_times(30, slope=DEG_SLOPE_THRESHOLD + 0.02)
        steep   = _degrading_lap_times(30, slope=SLOPE_MAX)

        df_shallow = _single_stint_df("AAA", shallow)
        df_steep   = _single_stint_df("BBB", steep)

        ins_shallow = tyre_deg.detect(df_shallow)
        ins_steep   = tyre_deg.detect(df_steep)

        assert len(ins_shallow) == 1
        assert len(ins_steep) == 1
        assert ins_steep[0]["rank_score"] >= ins_shallow[0]["rank_score"]

    def test_max_rank_score_capped_at_one(self):
        """Slope much greater than SLOPE_MAX should still cap at 1.0."""
        times = _degrading_lap_times(30, slope=SLOPE_MAX * 3)
        df = _single_stint_df("VER", times)
        insights = tyre_deg.detect(df)
        assert insights[0]["rank_score"] == 1.0


# ---------------------------------------------------------------------------
# Insight schema tests
# ---------------------------------------------------------------------------

class TestTyreDegSchema:
    def test_insight_schema(self, melbourne_df):
        insights = tyre_deg.detect(melbourne_df)
        assert len(insights) > 0, "No insights to validate schema against"
        for ins in insights:
            _assert_schema(ins)

    def test_lap_is_within_race(self, melbourne_df):
        insights = tyre_deg.detect(melbourne_df)
        for ins in insights:
            assert 1 <= ins["lap"] <= RACE_LAPS

    def test_detail_window_laps_matches_constant(self, melbourne_df):
        insights = tyre_deg.detect(melbourne_df)
        for ins in insights:
            assert ins["detail"]["window_laps"] == WINDOW

    def test_detail_compound_is_valid(self, melbourne_df):
        insights = tyre_deg.detect(melbourne_df)
        for ins in insights:
            assert ins["detail"]["compound"] in {"SOFT", "MEDIUM", "HARD"}

    def test_detail_window_lap_times_length_and_type(self, melbourne_df):
        """window_lap_times must have exactly WINDOW entries of floats."""
        insights = tyre_deg.detect(melbourne_df)
        for ins in insights:
            times = ins["detail"]["window_lap_times"]
            numbers = ins["detail"]["window_lap_numbers"]
            assert len(times) == WINDOW, f"Expected {WINDOW} lap times, got {len(times)}"
            assert len(numbers) == WINDOW
            assert all(isinstance(t, float) for t in times)
            assert all(isinstance(n, int) for n in numbers)

    def test_detail_window_lap_times_are_degrading(self):
        """For a clearly degrading stint, window_lap_times should increase."""
        times = _degrading_lap_times(30, slope=0.25)
        df = _single_stint_df("VER", times)
        insights = tyre_deg.detect(df)
        assert len(insights) == 1
        wt = insights[0]["detail"]["window_lap_times"]
        assert wt[-1] > wt[0], "Last lap in window should be slower than first"

    def test_detail_window_lap_numbers_match_lap_field(self, melbourne_df):
        """The last entry of window_lap_numbers should equal the insight's lap."""
        insights = tyre_deg.detect(melbourne_df)
        for ins in insights:
            assert ins["detail"]["window_lap_numbers"][-1] == ins["lap"]

    def test_stint_laps_structure(self, melbourne_df):
        """stint_laps must contain all stint laps with required keys and valid flag values."""
        insights = tyre_deg.detect(melbourne_df)
        valid_flags = {None, "SC", "VSC"}
        for ins in insights:
            sl = ins["detail"]["stint_laps"]
            assert isinstance(sl, list) and len(sl) > 0
            for entry in sl:
                assert {"lap", "time", "flag", "in_window"} == entry.keys()
                assert isinstance(entry["lap"], int)
                assert isinstance(entry["time"], float)
                assert entry["flag"] in valid_flags
                assert isinstance(entry["in_window"], bool)

    def test_stint_laps_window_coverage(self, melbourne_df):
        """Laps marked in_window must exactly match window_lap_numbers."""
        insights = tyre_deg.detect(melbourne_df)
        for ins in insights:
            d = ins["detail"]
            in_window_laps = {e["lap"] for e in d["stint_laps"] if e["in_window"]}
            assert in_window_laps == set(d["window_lap_numbers"])

    def test_stint_laps_sc_flags_injected(self):
        """SC laps (>25s above median) must be tagged with flag='SC'."""
        times = _degrading_lap_times(30, slope=0.25)
        # Inject a safety-car lap in the middle
        times[5] = times[5] + 40.0
        df = _single_stint_df("VER", times)
        insights = tyre_deg.detect(df)
        assert len(insights) == 1
        sc_entries = [e for e in insights[0]["detail"]["stint_laps"] if e["flag"] == "SC"]
        assert len(sc_entries) == 1
        assert sc_entries[0]["lap"] == 6  # 1-indexed lap


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestTyreDegEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=[
            "driver", "team", "lap", "compound", "tyre_age",
            "lap_time", "cumulative_time", "pit_this_lap", "stint_number",
        ])
        assert tyre_deg.detect(df) == []

    def test_single_driver_single_lap(self):
        df = make_race_df([{
            "driver": "TST",
            "stints": [{"compound": "SOFT", "laps": [(1, 82.0, 0)]}],
        }])
        assert tyre_deg.detect(df) == []

    def test_returns_list(self, melbourne_df):
        assert isinstance(tyre_deg.detect(melbourne_df), list)
