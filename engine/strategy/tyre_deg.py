"""
Tyre degradation detector.

Detects when a driver's lap times are degrading significantly within a stint,
signalling that a tyre change is overdue or that they're on a cliff.

Algorithm:
  - For each driver, group laps into stints.
  - Compute a rolling linear regression slope over a window of laps.
  - If the slope (seconds/lap) exceeds DEG_SLOPE_THRESHOLD and the stint has
    run long enough, flag a degradation insight.
  - rank_score = min(1.0, slope / SLOPE_MAX)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

DEG_SLOPE_THRESHOLD = 0.06   # sec/lap — minimum slope to flag
SLOPE_MAX = 0.40              # sec/lap — slope that earns rank_score 1.0
WINDOW = 5                    # laps in the regression window
MIN_STINT_LAPS = 5            # don't flag before this many laps in a stint


def _slope(lap_times: np.ndarray) -> float:
    """Linear regression slope of lap times over sequential laps."""
    x = np.arange(len(lap_times), dtype=float)
    return float(np.polyfit(x, lap_times, 1)[0])


def detect(race_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns a list of tyre_degradation insights.

    Parameters
    ----------
    race_df : DataFrame produced by synthetic_data.simulate_race()
    """
    insights: List[Dict[str, Any]] = []

    for driver, driver_df in race_df.groupby("driver"):
        for stint_num, stint_df in driver_df.groupby("stint_number"):
            stint_df = stint_df.sort_values("lap").reset_index(drop=True)

            if len(stint_df) < MIN_STINT_LAPS + WINDOW:
                continue

            times = stint_df["lap_time"].values
            # Exclude the pit lap itself (first lap of stint has pit loss added)
            start_idx = 1 if stint_df["pit_this_lap"].iloc[0] else 0

            for i in range(start_idx + MIN_STINT_LAPS, len(times) - WINDOW + 1):
                window_times = times[i : i + WINDOW]
                s = _slope(window_times)

                if s >= DEG_SLOPE_THRESHOLD:
                    lap = int(stint_df["lap"].iloc[i + WINDOW - 1])
                    compound = stint_df["compound"].iloc[i]
                    tyre_age = int(stint_df["tyre_age"].iloc[i + WINDOW - 1])
                    rank_score = round(min(1.0, s / SLOPE_MAX), 3)

                    window_lap_numbers = [
                        int(stint_df["lap"].iloc[j]) for j in range(i, i + WINDOW)
                    ]
                    window_lap_times = [round(float(t), 3) for t in window_times]

                    insights.append({
                        "insight_type": "tyre_degradation",
                        "driver": driver,
                        "lap": lap,
                        "rank_score": rank_score,
                        "summary": (
                            f"{driver} on {compound} (age {tyre_age}) "
                            f"degrading at {s:.3f} s/lap"
                        ),
                        "detail": {
                            "compound": compound,
                            "tyre_age": tyre_age,
                            "stint_number": int(stint_num),
                            "deg_slope_sec_per_lap": round(s, 4),
                            "window_laps": WINDOW,
                            "window_lap_numbers": window_lap_numbers,
                            "window_lap_times": window_lap_times,
                        },
                    })
                    break  # one insight per stint is sufficient

    return insights
