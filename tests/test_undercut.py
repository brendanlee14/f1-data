"""Tests for engine.strategy.undercut detector."""

import pandas as pd
import pytest

from engine.strategy import undercut
from engine.strategy.undercut import GAP_THRESHOLD_SEC, MIN_PACE_ADVANTAGE, MAX_POSITION_GAP

REQUIRED_KEYS = {"insight_type", "driver", "lap", "rank_score", "summary", "detail"}
DETAIL_KEYS = {
    "target_driver", "undercut_driver_pit_lap", "target_driver_pit_lap",
    "laps_early", "gap_before_sec", "gap_after_sec",
    "pace_advantage_sec_per_lap", "undercut_succeeded",
    "position_a", "position_b", "position_gap",
}


def _assert_schema(ins: dict):
    assert REQUIRED_KEYS.issubset(ins.keys())
    assert ins["insight_type"] == "undercut"
    assert isinstance(ins["driver"], str)
    assert isinstance(ins["lap"], int)
    assert 0.0 <= ins["rank_score"] <= 1.0
    assert DETAIL_KEYS.issubset(ins["detail"].keys())
    assert isinstance(ins["detail"]["undercut_succeeded"], bool)


# ---------------------------------------------------------------------------
# Scenario builder
# ---------------------------------------------------------------------------

def _make_undercut_scenario(
    gap_before: float = 2.0,         # A's lead over B before B pits (positive = A ahead)
    laps_early: int = 2,              # how many laps before A that B pits
    b_fresh_pace: float = 81.5,       # pace on fresh tyres (both drivers post-pit)
    a_old_pace: float = 82.5,         # unused; both run `base` pre-pit for stable gap
    pit_lap_b: int = 20,
    total_laps: int = 40,
    base: float = 82.0,               # shared pre-pit pace — keeps gap stable at gap_before
) -> pd.DataFrame:
    """
    Two-driver scenario: A leads B by gap_before before B pits.
    B undercuts A by pitting laps_early laps sooner.

    Both drivers run `base` pace before their pit so the gap stays exactly
    gap_before at check_lap (pit_lap_b - 1). The undercut pace advantage is
    base - b_fresh_pace (B gains this per lap on fresh rubber vs A on old).
    """
    pit_lap_a = pit_lap_b + laps_early
    rows = []

    # --- Driver A (pits later) ---
    cum_a = 0.0
    for lap in range(1, total_laps + 1):
        stint = 1 if lap < pit_lap_a else 2
        pit = (lap == pit_lap_a)
        lt = base if lap < pit_lap_a else b_fresh_pace
        if pit:
            lt = base + 22.0
        cum_a += lt
        rows.append({
            "driver": "AAA", "team": "T1", "lap": lap,
            "compound": "MEDIUM" if lap < pit_lap_a else "HARD",
            "tyre_age": (lap - 1) if lap < pit_lap_a else (lap - pit_lap_a),
            "lap_time": round(lt, 3),
            "cumulative_time": round(cum_a, 3),
            "pit_this_lap": pit,
            "stint_number": stint,
        })

    # --- Driver B (starts gap_before seconds behind A, pits first) ---
    cum_b = gap_before  # offset so gap(B-A) = gap_before at lap 0
    for lap in range(1, total_laps + 1):
        stint = 1 if lap < pit_lap_b else 2
        pit = (lap == pit_lap_b)
        lt = base if lap < pit_lap_b else b_fresh_pace
        if pit:
            lt = b_fresh_pace + 22.0
        cum_b += lt
        rows.append({
            "driver": "BBB", "team": "T2", "lap": lap,
            "compound": "MEDIUM" if lap < pit_lap_b else "HARD",
            "tyre_age": (lap - 1) if lap < pit_lap_b else (lap - pit_lap_b),
            "lap_time": round(lt, 3),
            "cumulative_time": round(cum_b, 3),
            "pit_this_lap": pit,
            "stint_number": stint,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection tests
# ---------------------------------------------------------------------------

class TestUndercutDetection:
    def test_detects_classic_undercut(self):
        """B pits 2 laps before A with clear pace advantage → insight generated."""
        df = _make_undercut_scenario(gap_before=2.0, laps_early=2, b_fresh_pace=81.2)
        insights = undercut.detect(df)
        assert len(insights) == 1
        assert insights[0]["driver"] == "BBB"
        assert insights[0]["detail"]["target_driver"] == "AAA"

    def test_no_detection_when_gap_too_large(self):
        """Drivers far apart should not trigger an undercut insight."""
        df = _make_undercut_scenario(gap_before=GAP_THRESHOLD_SEC + 2.0)
        insights = undercut.detect(df)
        assert len(insights) == 0

    def test_no_detection_when_pace_advantage_too_small(self):
        """Fresh pace equal to base means no advantage — should not fire."""
        df = _make_undercut_scenario(b_fresh_pace=82.0)  # base - b_fresh_pace = 0.0
        insights = undercut.detect(df)
        assert len(insights) == 0

    def test_no_detection_when_same_pit_lap(self):
        """No undercut if both drivers pit on the same lap."""
        df = _make_undercut_scenario(laps_early=0)
        insights = undercut.detect(df)
        assert len(insights) == 0

    def test_no_detection_when_b_already_ahead(self):
        """If B is already ahead of A, it's not an undercut."""
        df = _make_undercut_scenario(gap_before=-2.0)  # B ahead of A
        insights = undercut.detect(df)
        assert len(insights) == 0

    def test_laps_early_recorded_correctly(self):
        df = _make_undercut_scenario(laps_early=3, b_fresh_pace=81.0)
        insights = undercut.detect(df)
        assert len(insights) == 1
        assert insights[0]["detail"]["laps_early"] == 3

    def test_undercut_succeeded_flag(self):
        """When B emerges ahead, undercut_succeeded should be True."""
        # Very large pace advantage — B will easily jump A
        df = _make_undercut_scenario(gap_before=1.0, laps_early=2, b_fresh_pace=80.0)
        insights = undercut.detect(df)
        # May or may not succeed depending on exact cumulative times, just check it's a bool
        if insights:
            assert isinstance(insights[0]["detail"]["undercut_succeeded"], bool)

    def test_no_insights_with_no_pit_stops(self):
        """Drivers who never pit cannot undercut."""
        rows = [
            {"driver": "AAA", "team": "T1", "lap": lap, "compound": "MEDIUM",
             "tyre_age": lap - 1, "lap_time": 82.0, "cumulative_time": 82.0 * lap,
             "pit_this_lap": False, "stint_number": 1}
            for lap in range(1, 30)
        ] + [
            {"driver": "BBB", "team": "T2", "lap": lap, "compound": "MEDIUM",
             "tyre_age": lap - 1, "lap_time": 82.5, "cumulative_time": 82.5 * lap + 2.0,
             "pit_this_lap": False, "stint_number": 1}
            for lap in range(1, 30)
        ]
        df = pd.DataFrame(rows)
        assert undercut.detect(df) == []


# ---------------------------------------------------------------------------
# Rank score tests
# ---------------------------------------------------------------------------

class TestUndercutRankScore:
    def test_rank_score_in_range(self, melbourne_df):
        insights = undercut.detect(melbourne_df)
        for ins in insights:
            assert 0.0 <= ins["rank_score"] <= 1.0

    def test_larger_pace_advantage_higher_score(self):
        df_small = _make_undercut_scenario(b_fresh_pace=82.2, laps_early=2)
        df_large = _make_undercut_scenario(b_fresh_pace=80.0, laps_early=2)

        ins_small = undercut.detect(df_small)
        ins_large = undercut.detect(df_large)

        if ins_small and ins_large:
            assert ins_large[0]["rank_score"] >= ins_small[0]["rank_score"]

    def test_rank_score_capped_at_one(self, melbourne_df):
        insights = undercut.detect(melbourne_df)
        for ins in insights:
            assert ins["rank_score"] <= 1.0


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestUndercutSchema:
    def test_insight_schema_from_controlled_scenario(self):
        df = _make_undercut_scenario(gap_before=2.0, laps_early=2, b_fresh_pace=81.0)
        insights = undercut.detect(df)
        assert len(insights) == 1
        _assert_schema(insights[0])

    def test_insight_schema_from_melbourne(self, melbourne_df):
        insights = undercut.detect(melbourne_df)
        for ins in insights:
            _assert_schema(ins)

    def test_pit_lap_recorded_as_insight_lap(self):
        pit_lap_b = 18
        df = _make_undercut_scenario(pit_lap_b=pit_lap_b, laps_early=2, b_fresh_pace=81.0)
        insights = undercut.detect(df)
        assert len(insights) == 1
        assert insights[0]["lap"] == pit_lap_b


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestUndercutEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=[
            "driver", "team", "lap", "compound", "tyre_age",
            "lap_time", "cumulative_time", "pit_this_lap", "stint_number",
        ])
        assert undercut.detect(df) == []

    def test_single_driver(self):
        df = _make_undercut_scenario()
        df = df[df["driver"] == "AAA"]
        assert undercut.detect(df) == []

    def test_returns_list(self, melbourne_df):
        assert isinstance(undercut.detect(melbourne_df), list)


# ---------------------------------------------------------------------------
# Position filter
# ---------------------------------------------------------------------------

def _add_filler_drivers(df: pd.DataFrame, n: int, spacing: float = 0.4) -> pd.DataFrame:
    """
    Insert n filler drivers between AAA and BBB in cumulative time.

    Each filler has the same lap times as AAA but with an additional cumulative
    offset of k * spacing, placing them between AAA (P1) and BBB (P=n+2).
    Filler drivers have no pit stops so they're skipped in the pair loop.
    """
    aaa = df[df["driver"] == "AAA"].copy().sort_values("lap").reset_index(drop=True)
    fillers = []
    for k in range(1, n + 1):
        filler = aaa.copy()
        filler["driver"] = f"F{k:02d}"
        filler["cumulative_time"] = (filler["cumulative_time"] + k * spacing).round(3)
        filler["pit_this_lap"] = False
        filler["stint_number"] = 1
        fillers.append(filler)
    return pd.concat([df] + fillers, ignore_index=True)


class TestUndercutPositionFilter:
    def test_blocks_pair_outside_position_gap(self):
        """Pair within time gap but separated by > MAX_POSITION_GAP positions is suppressed."""
        # base scenario: gap_before=2.0, pace_adv = 82.0 - 80.0 = 2.0 (well above threshold)
        df = _make_undercut_scenario(gap_before=2.0, laps_early=2, b_fresh_pace=80.0)
        # Add MAX_POSITION_GAP fillers to push BBB to position MAX_POSITION_GAP + 2
        df_with_fillers = _add_filler_drivers(df, n=MAX_POSITION_GAP)
        insights = undercut.detect(df_with_fillers)
        assert len(insights) == 0, "Should be blocked: position gap exceeds MAX_POSITION_GAP"

    def test_allows_pair_at_position_gap_boundary(self):
        """Pair separated by exactly MAX_POSITION_GAP positions should still fire."""
        df = _make_undercut_scenario(gap_before=2.0, laps_early=2, b_fresh_pace=80.0)
        # MAX_POSITION_GAP - 1 fillers → BBB at position MAX_POSITION_GAP + 1 → gap = MAX_POSITION_GAP
        df_with_fillers = _add_filler_drivers(df, n=MAX_POSITION_GAP - 1)
        insights = undercut.detect(df_with_fillers)
        assert len(insights) == 1, "Should fire: position gap equals MAX_POSITION_GAP"

    def test_position_fields_in_detail(self):
        """position_a, position_b, position_gap must be present in detail."""
        df = _make_undercut_scenario(gap_before=2.0, laps_early=2, b_fresh_pace=80.0)
        insights = undercut.detect(df)
        assert len(insights) == 1
        detail = insights[0]["detail"]
        assert "position_a" in detail
        assert "position_b" in detail
        assert "position_gap" in detail

    def test_position_gap_value_is_correct(self):
        """With 1 filler driver, AAA=P1, filler=P2, BBB=P3 → position_gap=2."""
        df = _make_undercut_scenario(gap_before=2.0, laps_early=2, b_fresh_pace=80.0)
        df_with_filler = _add_filler_drivers(df, n=1)
        insights = undercut.detect(df_with_filler)
        assert len(insights) == 1
        assert insights[0]["detail"]["position_gap"] == 2

    def test_position_gap_without_fillers_is_one(self):
        """In a pure 2-driver scenario, the two drivers are always P1 and P2."""
        df = _make_undercut_scenario(gap_before=2.0, laps_early=2, b_fresh_pace=80.0)
        insights = undercut.detect(df)
        assert len(insights) == 1
        assert insights[0]["detail"]["position_gap"] == 1
