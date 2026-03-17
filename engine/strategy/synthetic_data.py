"""
Synthetic race data generator for Melbourne 2026 (Australian GP).

Albert Park characteristics:
- 58 laps, ~5.278 km
- Base lap time ~82.0 s (1:22.0)
- Typical 1-stop or 2-stop strategies
- Medium deg track — softs fall off after ~15 laps, mediums after ~25
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# --- Circuit constants -------------------------------------------------------

RACE_LAPS = 58
BASE_LAP_TIME = 82.0  # seconds

# Tyre compound configs: (deg_per_lap, initial_delta_vs_medium, cliff_lap)
COMPOUND_PARAMS = {
    "SOFT":   {"deg": 0.095, "delta": -0.6, "cliff": 16},
    "MEDIUM": {"deg": 0.055, "delta":  0.0, "cliff": 26},
    "HARD":   {"deg": 0.030, "delta":  0.5, "cliff": 40},
}

PIT_STOP_LOSS = 22.0  # seconds lost in a pit stop


# --- Driver stint builder ----------------------------------------------------

@dataclass
class Stint:
    compound: str
    start_lap: int
    end_lap: int  # inclusive
    start_tyre_age: int = 0  # pre-used tyres (qualifying sets, etc.)


@dataclass
class DriverStrategy:
    driver: str
    team: str
    stints: List[Stint]
    base_pace_delta: float = 0.0   # seconds vs field base (negative = faster)
    noise_std: float = 0.12


# --- 2026 Melbourne grid (fictional but plausible) ---------------------------

MELBOURNE_2026_STRATEGIES: List[DriverStrategy] = [
    # Front runners — 1-stop (Medium → Hard)
    DriverStrategy("VER", "Red Bull",    [Stint("MEDIUM", 1, 26), Stint("HARD", 27, 58)],  base_pace_delta=-0.55),
    DriverStrategy("NOR", "McLaren",     [Stint("MEDIUM", 1, 25), Stint("HARD", 26, 58)],  base_pace_delta=-0.45),
    DriverStrategy("PIA", "McLaren",     [Stint("SOFT",   1, 14), Stint("MEDIUM", 15, 36), Stint("HARD", 37, 58)], base_pace_delta=-0.40),
    DriverStrategy("LEC", "Ferrari",     [Stint("MEDIUM", 1, 24), Stint("HARD", 25, 58)],  base_pace_delta=-0.35),
    DriverStrategy("SAI", "Ferrari",     [Stint("SOFT",   1, 13), Stint("HARD",   14, 58)], base_pace_delta=-0.30),
    # Midfield — various strategies
    DriverStrategy("RUS", "Mercedes",    [Stint("MEDIUM", 1, 28), Stint("HARD", 29, 58)],  base_pace_delta=-0.15),
    DriverStrategy("ANT", "Mercedes",    [Stint("SOFT",   1, 15), Stint("MEDIUM", 16, 40), Stint("HARD", 41, 58)], base_pace_delta=-0.10),
    DriverStrategy("ALO", "Aston Martin",[Stint("MEDIUM", 1, 22), Stint("HARD", 23, 58)],  base_pace_delta= 0.05),
    DriverStrategy("STR", "Aston Martin",[Stint("SOFT",   1, 12), Stint("MEDIUM", 13, 38), Stint("HARD", 39, 58)], base_pace_delta= 0.20),
    DriverStrategy("TSU", "Red Bull",    [Stint("MEDIUM", 1, 27), Stint("HARD", 28, 58)],  base_pace_delta= 0.25),
    # Backmarkers
    DriverStrategy("GAS", "Alpine",      [Stint("MEDIUM", 1, 23), Stint("HARD", 24, 58)],  base_pace_delta= 0.55),
    DriverStrategy("OCO", "Haas",        [Stint("SOFT",   1, 14), Stint("HARD",   15, 58)], base_pace_delta= 0.65),
    DriverStrategy("ALB", "Williams",    [Stint("MEDIUM", 1, 30), Stint("HARD", 31, 58)],  base_pace_delta= 0.70),
    DriverStrategy("SAR", "Williams",    [Stint("SOFT",   1, 11), Stint("MEDIUM", 12, 35), Stint("HARD", 36, 58)], base_pace_delta= 0.90),
    DriverStrategy("HUL", "Sauber",      [Stint("MEDIUM", 1, 25), Stint("HARD", 26, 58)],  base_pace_delta= 1.10),
]


# --- Lap time calculator -----------------------------------------------------

def _tyre_deg_penalty(compound: str, age: int) -> float:
    """Returns cumulative deg penalty in seconds for a given tyre age."""
    p = COMPOUND_PARAMS[compound]
    if age <= p["cliff"]:
        return p["deg"] * age
    else:
        # Exponential fall-off after cliff
        cliff_penalty = p["deg"] * p["cliff"]
        post_cliff_age = age - p["cliff"]
        return cliff_penalty + p["deg"] * post_cliff_age * (1.0 + 0.04 * post_cliff_age)


def simulate_race(
    strategies: Optional[List[DriverStrategy]] = None,
    rng_seed: int = 2026,
) -> pd.DataFrame:
    """
    Simulate a race and return a DataFrame of lap-by-lap data.

    Columns:
        driver, team, lap, compound, tyre_age, lap_time, cumulative_time,
        pit_this_lap, stint_number
    """
    if strategies is None:
        strategies = MELBOURNE_2026_STRATEGIES

    rng = np.random.default_rng(rng_seed)
    rows = []

    for ds in strategies:
        cumulative_time = 0.0
        stint_number = 0

        # Build a lap → stint lookup
        stint_map: dict = {}
        for stint in ds.stints:
            for lap in range(stint.start_lap, stint.end_lap + 1):
                stint_map[lap] = stint

        for lap in range(1, RACE_LAPS + 1):
            stint = stint_map.get(lap)
            if stint is None:
                continue  # driver didn't finish (not modelled here)

            is_first_lap_of_stint = (lap == stint.start_lap)
            if is_first_lap_of_stint:
                stint_number += 1
                tyre_age = stint.start_tyre_age

            compound = stint.compound
            params = COMPOUND_PARAMS[compound]

            # Lap time = base + compound delta + degradation + pace delta + noise
            deg_penalty = _tyre_deg_penalty(compound, tyre_age)
            noise = rng.normal(0, ds.noise_std)
            lap_time = (
                BASE_LAP_TIME
                + params["delta"]
                + deg_penalty
                + ds.base_pace_delta
                + noise
            )

            # Pit stop on the last lap of a stint (except the final stint)
            pit_this_lap = False
            if is_first_lap_of_stint and stint_number > 1:
                lap_time += PIT_STOP_LOSS
                pit_this_lap = True

            # Lap 1 first-lap chaos
            if lap == 1:
                lap_time += rng.uniform(0.5, 3.0)

            cumulative_time += lap_time

            rows.append({
                "driver": ds.driver,
                "team": ds.team,
                "lap": lap,
                "compound": compound,
                "tyre_age": tyre_age,
                "lap_time": round(lap_time, 3),
                "cumulative_time": round(cumulative_time, 3),
                "pit_this_lap": pit_this_lap,
                "stint_number": stint_number,
            })

            tyre_age += 1

    return pd.DataFrame(rows)
