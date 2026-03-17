"""
FastF1 loader — transforms a real race session into the standard race DataFrame
schema expected by the Strategy Analyser detectors.

Required columns:
    driver, team, lap, compound, tyre_age, lap_time,
    cumulative_time, pit_this_lap, stint_number
"""

import logging
import os
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(os.getenv("FASTF1_CACHE", "/tmp/fastf1_cache"))


def load_fastf1_race(year: int, gp: str) -> pd.DataFrame:
    """
    Load a real race session via FastF1 and return a cleaned DataFrame
    matching the synthetic data schema.

    Parameters
    ----------
    year : int   e.g. 2025
    gp   : str   e.g. "Australia" or "Melbourne" or full "Australian Grand Prix"
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    log.info("Loading FastF1 session: %s %s Race", year, gp)
    session = fastf1.get_session(year, gp, "R")
    session.load(telemetry=False, weather=False, messages=False)

    laps: pd.DataFrame = session.laps.copy()

    # Drop laps with no valid lap time (in/out laps, red flags, etc.)
    laps = laps.dropna(subset=["LapTime", "Compound", "TyreLife"])
    laps = laps[laps["LapTime"].dt.total_seconds() > 0]

    # Normalise compound to uppercase string
    laps["Compound"] = laps["Compound"].str.upper()

    # Build output rows
    rows = []
    for driver, grp in laps.groupby("Driver", sort=False):
        grp = grp.sort_values("LapNumber").copy()

        lap_times_s = grp["LapTime"].dt.total_seconds().values
        cumulative  = np.cumsum(lap_times_s)

        # pit_this_lap: True when PitInTime is non-null (driver entered pits)
        pit_flag = grp["PitInTime"].notna().values

        team = grp["Team"].iloc[0] if "Team" in grp.columns else ""

        for i, (_, row) in enumerate(grp.iterrows()):
            rows.append({
                "driver":          driver,
                "team":            team,
                "lap":             int(row["LapNumber"]),
                "compound":        row["Compound"],
                "tyre_age":        int(row["TyreLife"]),
                "lap_time":        lap_times_s[i],
                "cumulative_time": cumulative[i],
                "pit_this_lap":    bool(pit_flag[i]),
                "stint_number":    int(row["Stint"]) if not pd.isna(row.get("Stint", np.nan)) else 1,
            })

    df = pd.DataFrame(rows)
    log.info("Loaded %d laps for %d drivers", len(df), df["driver"].nunique())
    return df
