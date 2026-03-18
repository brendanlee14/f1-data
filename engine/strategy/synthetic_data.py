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
    "SOFT":   {"deg": 0.14,  "delta": -0.6, "cliff": 16},
    "MEDIUM": {"deg": 0.075, "delta":  0.0, "cliff": 26},
    "HARD":   {"deg": 0.040, "delta":  0.5, "cliff": 30},
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
        return cliff_penalty + p["deg"] * post_cliff_age * (1.0 + 0.08 * post_cliff_age)


def simulate_race(
    strategies: Optional[List[DriverStrategy]] = None,
    rng_seed: int = 2026,
    base_lap_time: float = BASE_LAP_TIME,
    total_laps: int = RACE_LAPS,
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

        for lap in range(1, total_laps + 1):
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
                base_lap_time
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


# ---------------------------------------------------------------------------
# 2026 Full-season calendar & multi-race simulation
# ---------------------------------------------------------------------------

# Full 20-driver 2026 grid: (code, team, base_pace_delta, noise_std)
# base_pace_delta is relative to field median; negative = faster
DRIVERS_2026 = [
    ("ANT", "Mercedes",      -0.65, 0.11),
    ("RUS", "Mercedes",      -0.50, 0.10),
    ("HAM", "Ferrari",       -0.40, 0.10),
    ("LEC", "Ferrari",       -0.35, 0.11),
    ("NOR", "McLaren",       -0.25, 0.11),
    ("PIA", "McLaren",       -0.20, 0.12),
    ("LAW", "Red Bull",      -0.10, 0.12),
    ("TSU", "Red Bull",       0.05, 0.12),
    ("GAS", "Alpine",         0.15, 0.12),
    ("ALO", "Aston Martin",   0.20, 0.11),
    ("STR", "Aston Martin",   0.40, 0.13),
    ("COL", "Alpine",         0.45, 0.14),
    ("HAD", "Racing Bulls",   0.55, 0.13),
    ("BEA", "Haas",           0.60, 0.13),
    ("OCO", "Haas",           0.65, 0.13),
    ("DOO", "Racing Bulls",   0.80, 0.14),
    ("SAI", "Williams",       0.85, 0.12),
    ("ALB", "Williams",       1.00, 0.13),
    ("HUL", "Audi",           1.15, 0.13),
    ("BOR", "Audi",           1.30, 0.14),
]

# 2026 calendar — 24 rounds (corrected)
# Sources: formula1.com/en/racing/2026
RACES_2026 = [
    {"key": "australia",  "name": "Australian GP",    "circuit": "Albert Park",   "round":  1, "laps": 58, "base_time": 82.0},
    {"key": "china",      "name": "Chinese GP",       "circuit": "Shanghai",      "round":  2, "laps": 56, "base_time": 93.5},
    {"key": "japan",      "name": "Japanese GP",      "circuit": "Suzuka",        "round":  3, "laps": 53, "base_time": 91.0},
    {"key": "bahrain",    "name": "Bahrain GP",       "circuit": "Sakhir",        "round":  4, "laps": 57, "base_time": 91.5},
    {"key": "saudi_arabia","name":"Saudi Arabian GP", "circuit": "Jeddah",        "round":  5, "laps": 50, "base_time": 87.5},
    {"key": "miami",      "name": "Miami GP",         "circuit": "Miami",         "round":  6, "laps": 57, "base_time": 90.0},
    {"key": "canada",     "name": "Canadian GP",      "circuit": "Montreal",      "round":  7, "laps": 70, "base_time": 75.5},
    {"key": "monaco",     "name": "Monaco GP",        "circuit": "Monaco",        "round":  8, "laps": 78, "base_time": 73.0},
    {"key": "spain",      "name": "Spanish GP",       "circuit": "Barcelona",     "round":  9, "laps": 66, "base_time": 80.0},
    {"key": "austria",    "name": "Austrian GP",      "circuit": "Red Bull Ring", "round": 10, "laps": 71, "base_time": 68.0},
    {"key": "britain",    "name": "British GP",       "circuit": "Silverstone",   "round": 11, "laps": 52, "base_time": 88.0},
    {"key": "belgium",    "name": "Belgian GP",       "circuit": "Spa",           "round": 12, "laps": 44, "base_time": 105.0},
    {"key": "hungary",    "name": "Hungarian GP",     "circuit": "Budapest",      "round": 13, "laps": 70, "base_time": 79.0},
    {"key": "netherlands","name": "Dutch GP",         "circuit": "Zandvoort",     "round": 14, "laps": 72, "base_time": 74.0},
    {"key": "italy",      "name": "Italian GP",       "circuit": "Monza",         "round": 15, "laps": 53, "base_time": 82.0},
    {"key": "madrid",     "name": "Madrid GP",        "circuit": "Madrid",        "round": 16, "laps": 55, "base_time": 86.0},
    {"key": "azerbaijan", "name": "Azerbaijan GP",    "circuit": "Baku",          "round": 17, "laps": 51, "base_time": 103.0},
    {"key": "singapore",  "name": "Singapore GP",     "circuit": "Marina Bay",    "round": 18, "laps": 62, "base_time": 95.0},
    {"key": "usa",        "name": "United States GP", "circuit": "Austin",        "round": 19, "laps": 56, "base_time": 97.0},
    {"key": "mexico",     "name": "Mexico City GP",   "circuit": "Mexico City",   "round": 20, "laps": 71, "base_time": 78.5},
    {"key": "brazil",     "name": "São Paulo GP",     "circuit": "Interlagos",    "round": 21, "laps": 71, "base_time": 72.0},
    {"key": "las_vegas",  "name": "Las Vegas GP",     "circuit": "Las Vegas",     "round": 22, "laps": 50, "base_time": 96.0},
    {"key": "qatar",      "name": "Qatar GP",         "circuit": "Lusail",        "round": 23, "laps": 57, "base_time": 84.5},
    {"key": "abu_dhabi",  "name": "Abu Dhabi GP",     "circuit": "Yas Marina",    "round": 24, "laps": 58, "base_time": 88.0},
]

# Circuit strategy profiles.
# compound_pool_1s / compound_pool_2s: list of (c1, c2[, c3]) tuples.
# two_stop_pct: fraction of the grid expected to do 2 stops.
_CIRCUIT_PROFILES: dict = {
    "australia":      {"two_stop_pct": 0.20, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD"),("SOFT","MEDIUM"),("HARD","MEDIUM")],      "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM")]},
    "china":          {"two_stop_pct": 0.30, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD"),("HARD","MEDIUM")],                         "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","SOFT","HARD")]},
    "japan":          {"two_stop_pct": 0.15, "pool_1s": [("MEDIUM","HARD"),("HARD","MEDIUM"),("SOFT","HARD")],                         "pool_2s": [("SOFT","MEDIUM","HARD")]},
    "bahrain":        {"two_stop_pct": 0.55, "pool_1s": [("SOFT","HARD"),("MEDIUM","HARD")],                                           "pool_2s": [("SOFT","MEDIUM","HARD"),("SOFT","HARD","MEDIUM"),("MEDIUM","HARD","MEDIUM")]},
    "saudi_arabia":   {"two_stop_pct": 0.20, "pool_1s": [("SOFT","MEDIUM"),("SOFT","HARD"),("MEDIUM","HARD")],                         "pool_2s": [("SOFT","MEDIUM","HARD")]},
    "miami":          {"two_stop_pct": 0.25, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD"),("SOFT","MEDIUM")],                         "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM")]},
    "monaco":         {"two_stop_pct": 0.05, "pool_1s": [("SOFT","MEDIUM"),("SOFT","SOFT"),("MEDIUM","HARD"),("SOFT","HARD")],          "pool_2s": [("SOFT","SOFT","MEDIUM")]},
    "spain":          {"two_stop_pct": 0.60, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD")],                                           "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM"),("SOFT","HARD","MEDIUM")]},
    "canada":         {"two_stop_pct": 0.20, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD"),("HARD","MEDIUM"),("SOFT","MEDIUM")],        "pool_2s": [("SOFT","MEDIUM","HARD")]},
    "austria":        {"two_stop_pct": 0.55, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD")],                                           "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM"),("SOFT","HARD","MEDIUM")]},
    "britain":        {"two_stop_pct": 0.55, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD")],                                           "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM"),("SOFT","HARD","MEDIUM")]},
    "belgium":        {"two_stop_pct": 0.15, "pool_1s": [("MEDIUM","HARD"),("HARD","MEDIUM"),("SOFT","HARD")],                         "pool_2s": [("SOFT","MEDIUM","HARD")]},
    "hungary":        {"two_stop_pct": 0.60, "pool_1s": [("MEDIUM","HARD"),("SOFT","MEDIUM")],                                         "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM"),("SOFT","HARD","MEDIUM")]},
    "netherlands":    {"two_stop_pct": 0.50, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD")],                                           "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM")]},
    "italy":          {"two_stop_pct": 0.10, "pool_1s": [("MEDIUM","HARD"),("HARD","MEDIUM"),("SOFT","HARD"),("HARD","SOFT")],          "pool_2s": [("SOFT","MEDIUM","HARD")]},
    "madrid":         {"two_stop_pct": 0.25, "pool_1s": [("MEDIUM","HARD"),("SOFT","MEDIUM"),("SOFT","HARD")],                         "pool_2s": [("SOFT","MEDIUM","HARD"),("SOFT","HARD","MEDIUM")]},
    "azerbaijan":     {"two_stop_pct": 0.20, "pool_1s": [("SOFT","MEDIUM"),("SOFT","HARD"),("MEDIUM","HARD"),("HARD","MEDIUM")],        "pool_2s": [("SOFT","MEDIUM","HARD"),("SOFT","HARD","MEDIUM")]},
    "singapore":      {"two_stop_pct": 0.30, "pool_1s": [("MEDIUM","HARD"),("SOFT","MEDIUM"),("SOFT","HARD")],                         "pool_2s": [("SOFT","MEDIUM","HARD"),("SOFT","HARD","MEDIUM")]},
    "usa":            {"two_stop_pct": 0.60, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD")],                                           "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM"),("SOFT","HARD","MEDIUM")]},
    "mexico":         {"two_stop_pct": 0.10, "pool_1s": [("MEDIUM","HARD"),("HARD","MEDIUM"),("SOFT","HARD")],                         "pool_2s": [("SOFT","MEDIUM","HARD")]},
    "brazil":         {"two_stop_pct": 0.35, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD"),("SOFT","MEDIUM")],                         "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM")]},
    "las_vegas":      {"two_stop_pct": 0.10, "pool_1s": [("MEDIUM","HARD"),("HARD","MEDIUM"),("SOFT","HARD")],                         "pool_2s": [("SOFT","MEDIUM","HARD")]},
    "qatar":          {"two_stop_pct": 0.70, "pool_1s": [("SOFT","MEDIUM"),("MEDIUM","HARD")],                                         "pool_2s": [("SOFT","MEDIUM","HARD"),("SOFT","HARD","MEDIUM"),("MEDIUM","SOFT","HARD")]},
    "abu_dhabi":      {"two_stop_pct": 0.20, "pool_1s": [("MEDIUM","HARD"),("SOFT","HARD"),("HARD","MEDIUM")],                         "pool_2s": [("SOFT","MEDIUM","HARD"),("MEDIUM","HARD","MEDIUM")]},
}


def _build_strategies_for_race(race_key: str, laps: int, rng: "np.random.Generator") -> List[DriverStrategy]:
    """Generate a plausible grid of strategies for a given race."""
    profile = _CIRCUIT_PROFILES.get(race_key, _CIRCUIT_PROFILES["australia"])
    pool_1s = profile["pool_1s"]
    pool_2s = profile["pool_2s"]
    two_stop_pct = profile["two_stop_pct"]

    strategies = []
    n_two_stop = int(len(DRIVERS_2026) * two_stop_pct)
    # Assign 2-stop to the back of the grid (slower cars gain more from 2-stop)
    two_stop_indices = set(range(len(DRIVERS_2026) - n_two_stop, len(DRIVERS_2026)))

    for i, (driver, team, base_delta, noise_std) in enumerate(DRIVERS_2026):
        # Slight per-circuit pace shuffle (±0.25 s) so the order isn't always identical
        circuit_delta = float(rng.uniform(-0.25, 0.25))
        adj_delta = base_delta + circuit_delta

        # Flip 2-stop assignment ~25% of the time to add variety
        use_two_stop = (i in two_stop_indices) ^ (rng.random() < 0.25)

        if use_two_stop and len(pool_2s) > 0:
            compounds = pool_2s[int(rng.integers(len(pool_2s)))]
            pit1 = max(5,        min(laps - 22, int(laps * float(rng.uniform(0.22, 0.33)))))
            pit2 = max(pit1 + 10, min(laps - 8,  int(laps * float(rng.uniform(0.54, 0.68)))))
            stints = [
                Stint(compounds[0], 1,        pit1),
                Stint(compounds[1], pit1 + 1, pit2),
                Stint(compounds[2], pit2 + 1, laps),
            ]
        else:
            compounds = pool_1s[int(rng.integers(len(pool_1s)))]
            pit1 = max(5, min(laps - 8, int(laps * float(rng.uniform(0.38, 0.56)))))
            stints = [
                Stint(compounds[0], 1,        pit1),
                Stint(compounds[1], pit1 + 1, laps),
            ]

        strategies.append(DriverStrategy(
            driver=driver,
            team=team,
            stints=stints,
            base_pace_delta=round(adj_delta, 3),
            noise_std=noise_std,
        ))

    return strategies


def simulate_race_for_round(race_key: str) -> pd.DataFrame:
    """Simulate a full race weekend for a 2026 calendar round."""
    race = next((r for r in RACES_2026 if r["key"] == race_key), None)
    if race is None:
        raise ValueError(f"Unknown race key: {race_key!r}")

    seed = race["round"] * 997 + 2026  # deterministic, unique per round
    rng  = np.random.default_rng(seed)
    strategies = _build_strategies_for_race(race_key, race["laps"], rng)

    return simulate_race(
        strategies=strategies,
        rng_seed=seed,
        base_lap_time=race["base_time"],
        total_laps=race["laps"],
    )
