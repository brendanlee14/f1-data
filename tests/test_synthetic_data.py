"""Tests for engine.strategy.synthetic_data."""

import numpy as np
import pandas as pd
import pytest

from engine.strategy.synthetic_data import (
    simulate_race,
    MELBOURNE_2026_STRATEGIES,
    RACE_LAPS,
    BASE_LAP_TIME,
    PIT_STOP_LOSS,
    COMPOUND_PARAMS,
    _tyre_deg_penalty,
)


# ---------------------------------------------------------------------------
# _tyre_deg_penalty
# ---------------------------------------------------------------------------

class TestTyreDegPenalty:
    def test_zero_age_is_zero(self):
        assert _tyre_deg_penalty("MEDIUM", 0) == 0.0

    def test_increases_with_age(self):
        assert _tyre_deg_penalty("SOFT", 5) < _tyre_deg_penalty("SOFT", 10)

    def test_hard_slower_than_soft_at_low_age(self):
        # Hard compound accumulates less deg per lap
        assert _tyre_deg_penalty("HARD", 10) < _tyre_deg_penalty("SOFT", 10)

    def test_cliff_accelerates_deg(self):
        """Deg per lap should be higher post-cliff than pre-cliff."""
        cliff = COMPOUND_PARAMS["SOFT"]["cliff"]
        pre_cliff = _tyre_deg_penalty("SOFT", cliff) - _tyre_deg_penalty("SOFT", cliff - 1)
        post_cliff = _tyre_deg_penalty("SOFT", cliff + 2) - _tyre_deg_penalty("SOFT", cliff + 1)
        assert post_cliff > pre_cliff

    def test_all_compounds_defined(self):
        for compound in ("SOFT", "MEDIUM", "HARD"):
            assert _tyre_deg_penalty(compound, 15) >= 0


# ---------------------------------------------------------------------------
# simulate_race — DataFrame shape and schema
# ---------------------------------------------------------------------------

EXPECTED_COLUMNS = {
    "driver", "team", "lap", "compound", "tyre_age",
    "lap_time", "cumulative_time", "pit_this_lap", "stint_number",
}


class TestSimulateRaceSchema:
    def test_columns_present(self, melbourne_df):
        assert EXPECTED_COLUMNS.issubset(set(melbourne_df.columns))

    def test_row_count(self, melbourne_df):
        # 15 drivers × 58 laps
        assert len(melbourne_df) == 15 * RACE_LAPS

    def test_driver_count(self, melbourne_df):
        assert melbourne_df["driver"].nunique() == 15

    def test_lap_range(self, melbourne_df):
        assert melbourne_df["lap"].min() == 1
        assert melbourne_df["lap"].max() == RACE_LAPS

    def test_compounds_valid(self, melbourne_df):
        assert set(melbourne_df["compound"].unique()).issubset({"SOFT", "MEDIUM", "HARD"})

    def test_stint_number_positive(self, melbourne_df):
        assert (melbourne_df["stint_number"] >= 1).all()

    def test_tyre_age_non_negative(self, melbourne_df):
        assert (melbourne_df["tyre_age"] >= 0).all()


# ---------------------------------------------------------------------------
# simulate_race — lap time plausibility
# ---------------------------------------------------------------------------

class TestSimulateRaceLapTimes:
    def test_normal_lap_times_reasonable(self, melbourne_df):
        """Non-pit laps should be within ±10 s of base lap time."""
        normal = melbourne_df[~melbourne_df["pit_this_lap"]]["lap_time"]
        assert (normal > BASE_LAP_TIME - 5).all()
        assert (normal < BASE_LAP_TIME + 15).all()

    def test_pit_laps_include_loss(self, melbourne_df):
        """Pit laps must be significantly longer than non-pit laps."""
        pit = melbourne_df[melbourne_df["pit_this_lap"]]["lap_time"].mean()
        non_pit = melbourne_df[~melbourne_df["pit_this_lap"]]["lap_time"].mean()
        assert pit > non_pit + PIT_STOP_LOSS * 0.8

    def test_cumulative_time_monotonic_per_driver(self, melbourne_df):
        for driver, df in melbourne_df.groupby("driver"):
            times = df.sort_values("lap")["cumulative_time"].values
            assert (np.diff(times) > 0).all(), f"{driver} cumulative time not monotonic"

    def test_deterministic_with_same_seed(self):
        df1 = simulate_race(rng_seed=42)
        df2 = simulate_race(rng_seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = simulate_race(rng_seed=1)
        df2 = simulate_race(rng_seed=2)
        assert not df1["lap_time"].equals(df2["lap_time"])


# ---------------------------------------------------------------------------
# simulate_race — tyre age logic
# ---------------------------------------------------------------------------

class TestTyreAge:
    def test_tyre_age_increments_within_stint(self, melbourne_df):
        for (driver, stint), df in melbourne_df.groupby(["driver", "stint_number"]):
            df = df.sort_values("lap")
            ages = df["tyre_age"].values
            # Each lap tyre_age should increase by 1 (after recording)
            diffs = np.diff(ages)
            assert (diffs == 1).all(), f"{driver} stint {stint}: age not incrementing"

    def test_pit_lap_resets_age(self, melbourne_df):
        """First lap of a new stint (pit lap) should have low tyre age."""
        pit_rows = melbourne_df[melbourne_df["pit_this_lap"]]
        assert (pit_rows["tyre_age"] == 0).all()


# ---------------------------------------------------------------------------
# simulate_race — pit stop logic
# ---------------------------------------------------------------------------

class TestPitStops:
    def test_first_stint_has_no_pit(self, melbourne_df):
        first_laps = melbourne_df[melbourne_df["lap"] == 1]
        assert (~first_laps["pit_this_lap"]).all()

    def test_each_driver_pits_at_least_once(self, melbourne_df):
        """All drivers on a 1-stop strategy must have ≥ 1 pit."""
        pit_counts = melbourne_df.groupby("driver")["pit_this_lap"].sum()
        assert (pit_counts >= 1).all()

    def test_multi_stop_drivers_pit_multiple_times(self, melbourne_df):
        """PIA, ANT, STR, SAR are on 2-stop strategies."""
        two_stoppers = ["PIA", "ANT", "STR", "SAR"]
        pit_counts = melbourne_df.groupby("driver")["pit_this_lap"].sum()
        for driver in two_stoppers:
            assert pit_counts[driver] == 2, f"{driver} expected 2 pit stops"
