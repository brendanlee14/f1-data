"""
Shared fixtures for Strategy Analyser tests.
"""

import numpy as np
import pandas as pd
import pytest

from engine.strategy.synthetic_data import simulate_race, MELBOURNE_2026_STRATEGIES


@pytest.fixture(scope="session")
def melbourne_df():
    """Full Melbourne 2026 simulated race — shared across tests."""
    return simulate_race(strategies=MELBOURNE_2026_STRATEGIES, rng_seed=2026)


def make_race_df(drivers: list[dict]) -> pd.DataFrame:
    """
    Build a minimal race DataFrame from a list of driver stint specs.

    Each driver dict:
        driver, team, stints: [{compound, laps: [(lap_time, tyre_age)], pit_on_first: bool}]
    """
    rows = []
    for d in drivers:
        cumulative_time = 0.0
        for stint_num, stint in enumerate(d["stints"], start=1):
            for i, (lap_num, lap_time, tyre_age) in enumerate(stint["laps"]):
                pit_this_lap = (i == 0 and stint_num > 1)
                lt = lap_time + (22.0 if pit_this_lap else 0.0)
                cumulative_time += lt
                rows.append({
                    "driver": d["driver"],
                    "team": d.get("team", "Test"),
                    "lap": lap_num,
                    "compound": stint["compound"],
                    "tyre_age": tyre_age,
                    "lap_time": round(lt, 3),
                    "cumulative_time": round(cumulative_time, 3),
                    "pit_this_lap": pit_this_lap,
                    "stint_number": stint_num,
                })
    return pd.DataFrame(rows)
