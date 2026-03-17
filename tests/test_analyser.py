"""Tests for engine.strategy.analyser.StrategyAnalyser orchestrator."""

import pandas as pd
import pytest

from engine.strategy import StrategyAnalyser

INSIGHT_TYPES = {"tyre_degradation", "undercut", "overcut"}
REQUIRED_KEYS = {"insight_type", "driver", "lap", "rank_score", "summary", "detail"}


# ---------------------------------------------------------------------------
# load_race
# ---------------------------------------------------------------------------

class TestLoadRace:
    def test_returns_dataframe(self):
        df = StrategyAnalyser().load_race()
        assert isinstance(df, pd.DataFrame)

    def test_deterministic(self):
        df1 = StrategyAnalyser().load_race(rng_seed=42)
        df2 = StrategyAnalyser().load_race(rng_seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = StrategyAnalyser().load_race(rng_seed=1)
        df2 = StrategyAnalyser().load_race(rng_seed=2)
        assert not df1["lap_time"].equals(df2["lap_time"])

    def test_expected_columns(self):
        df = StrategyAnalyser().load_race()
        for col in ("driver", "lap", "compound", "tyre_age", "lap_time",
                    "cumulative_time", "pit_this_lap", "stint_number"):
            assert col in df.columns


# ---------------------------------------------------------------------------
# run — structure and ordering
# ---------------------------------------------------------------------------

class TestAnalyserRun:
    def test_returns_list(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df)
        assert isinstance(result, list)

    def test_all_three_insight_types_present(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df)
        types = {i["insight_type"] for i in result}
        assert types == INSIGHT_TYPES, f"Missing types: {INSIGHT_TYPES - types}"

    def test_sorted_by_rank_score_descending(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df)
        scores = [i["rank_score"] for i in result]
        assert scores == sorted(scores, reverse=True)

    def test_all_insights_have_required_keys(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df)
        for ins in result:
            missing = REQUIRED_KEYS - ins.keys()
            assert not missing, f"Insight missing keys: {missing}\n{ins}"

    def test_rank_score_all_in_range(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df)
        for ins in result:
            assert 0.0 <= ins["rank_score"] <= 1.0, f"rank_score out of range: {ins}"

    def test_insight_type_is_string(self, melbourne_df):
        for ins in StrategyAnalyser().run(melbourne_df):
            assert isinstance(ins["insight_type"], str)

    def test_driver_is_3_char_string(self, melbourne_df):
        for ins in StrategyAnalyser().run(melbourne_df):
            assert isinstance(ins["driver"], str)
            assert len(ins["driver"]) == 3, f"Driver code not 3 chars: {ins['driver']}"

    def test_lap_is_integer(self, melbourne_df):
        for ins in StrategyAnalyser().run(melbourne_df):
            assert isinstance(ins["lap"], int)

    def test_summary_is_non_empty_string(self, melbourne_df):
        for ins in StrategyAnalyser().run(melbourne_df):
            assert isinstance(ins["summary"], str) and len(ins["summary"]) > 0

    def test_detail_is_dict(self, melbourne_df):
        for ins in StrategyAnalyser().run(melbourne_df):
            assert isinstance(ins["detail"], dict)


# ---------------------------------------------------------------------------
# run — min_rank_score filter
# ---------------------------------------------------------------------------

class TestMinRankScoreFilter:
    def test_filter_removes_low_scores(self, melbourne_df):
        threshold = 0.5
        result = StrategyAnalyser().run(melbourne_df, min_rank_score=threshold)
        for ins in result:
            assert ins["rank_score"] >= threshold

    def test_zero_threshold_returns_all(self, melbourne_df):
        all_insights = StrategyAnalyser().run(melbourne_df, min_rank_score=0.0)
        filtered = StrategyAnalyser().run(melbourne_df, min_rank_score=0.3)
        assert len(all_insights) >= len(filtered)

    def test_threshold_one_returns_only_perfect_scores(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df, min_rank_score=1.0)
        for ins in result:
            assert ins["rank_score"] == 1.0

    def test_threshold_above_one_returns_empty(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df, min_rank_score=1.1)
        assert result == []

    def test_higher_threshold_fewer_or_equal_results(self, melbourne_df):
        a = StrategyAnalyser()
        low  = len(a.run(melbourne_df, min_rank_score=0.2))
        high = len(a.run(melbourne_df, min_rank_score=0.6))
        assert high <= low


# ---------------------------------------------------------------------------
# run — insight counts are non-trivial
# ---------------------------------------------------------------------------

class TestAnalyserInsightVolume:
    def test_produces_insights(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df)
        assert len(result) > 0

    def test_produces_tyre_deg_insights(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df)
        assert any(i["insight_type"] == "tyre_degradation" for i in result)

    def test_produces_overcut_insights(self, melbourne_df):
        result = StrategyAnalyser().run(melbourne_df)
        assert any(i["insight_type"] == "overcut" for i in result)

    def test_empty_dataframe_returns_empty_list(self):
        df = pd.DataFrame(columns=[
            "driver", "team", "lap", "compound", "tyre_age",
            "lap_time", "cumulative_time", "pit_this_lap", "stint_number",
        ])
        result = StrategyAnalyser().run(df)
        assert result == []
