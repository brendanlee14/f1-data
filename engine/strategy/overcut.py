"""
Overcut detector.

An overcut occurs when Driver A stays out longer than Driver B (who has already
pitted), uses clean air and a tyre delta to build a gap, and comes out ahead of
B after their own pit stop.

Algorithm:
  1. Find driver pairs where B has already pitted and A is still on old tyres.
  2. In the laps between B's pit and A's pit, measure if A's lap times are
     competitive despite older rubber (clean air compensation).
  3. Check whether A emerges ahead of B after pitting.
  4. rank_score reflects the gap gained and the tyre age A extended to.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

GAP_THRESHOLD_SEC = 4.0     # max gap before B's pit to consider a battle
OVERCUT_WINDOW = 3          # laps to measure A's pace after B pits
CLEAN_AIR_BENEFIT = 0.20    # assumed sec/lap from clean air (used in scoring)
MIN_STAY_OUT_LAPS = 2       # A must stay out at least this many laps after B pits


def _cumulative_gap(race_df: pd.DataFrame, lap: int, driver_a: str, driver_b: str) -> float:
    """Gap: positive value means A's cumulative time is less (A is ahead on track)."""
    def cum(d):
        row = race_df[(race_df["driver"] == d) & (race_df["lap"] == lap)]
        return float(row["cumulative_time"].iloc[0]) if not row.empty else np.nan
    # A ahead means A has lower cumulative time → gap = cum_b - cum_a > 0
    a_row = race_df[(race_df["driver"] == driver_a) & (race_df["lap"] == lap)]
    b_row = race_df[(race_df["driver"] == driver_b) & (race_df["lap"] == lap)]
    if a_row.empty or b_row.empty:
        return np.nan
    return float(b_row["cumulative_time"].iloc[0]) - float(a_row["cumulative_time"].iloc[0])


def detect(race_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns a list of overcut insights.
    """
    insights: List[Dict[str, Any]] = []
    drivers = race_df["driver"].unique().tolist()

    pit_laps: Dict[str, List[int]] = {}
    for driver, df in race_df.groupby("driver"):
        pit_laps[driver] = df[df["pit_this_lap"]]["lap"].tolist()

    for i, _da in enumerate(drivers):
        for _db in drivers[i + 1:]:
            # Use local variables so swapping doesn't corrupt the loop state
            driver_a, driver_b = _da, _db
            pits_a = pit_laps.get(driver_a, [])
            pits_b = pit_laps.get(driver_b, [])

            if not pits_a or not pits_b:
                continue

            pit_a = pits_a[0]
            pit_b = pits_b[0]

            # A must pit AFTER B by at least MIN_STAY_OUT_LAPS
            stay_out = pit_a - pit_b
            if stay_out < MIN_STAY_OUT_LAPS:
                # Try swapped roles
                stay_out_swap = pit_b - pit_a
                if stay_out_swap < MIN_STAY_OUT_LAPS:
                    continue
                driver_a, driver_b = driver_b, driver_a
                pit_a, pit_b = pit_b, pit_a
                stay_out = stay_out_swap

            # Check gap just before B pits — they should be close (A ahead)
            check_lap = pit_b - 1
            if check_lap < 1:
                continue
            gap_before = _cumulative_gap(race_df, check_lap, driver_a, driver_b)
            if np.isnan(gap_before):
                continue
            if gap_before < 0:
                continue  # A is behind — overcut not relevant
            if gap_before > GAP_THRESHOLD_SEC:
                continue  # too far apart

            # Measure A's pace while B is on fresh tyres (between pit_b and pit_a)
            measure_start = pit_b + 1
            measure_end = min(pit_b + OVERCUT_WINDOW, pit_a - 1)
            if measure_end < measure_start:
                continue

            pace_a = race_df[
                (race_df["driver"] == driver_a)
                & (race_df["lap"].between(measure_start, measure_end))
            ]["lap_time"].values

            pace_b = race_df[
                (race_df["driver"] == driver_b)
                & (race_df["lap"].between(measure_start, measure_end))
            ]["lap_time"].values

            if len(pace_a) == 0 or len(pace_b) == 0:
                continue

            avg_a = float(np.mean(pace_a))
            avg_b = float(np.mean(pace_b))
            # B on fresh tyres should theoretically be faster; if A keeps up, overcut is on
            pace_delta = avg_b - avg_a  # positive = A surprisingly fast vs B on fresher tyres

            # Check gap after A pits
            gap_after = _cumulative_gap(race_df, pit_a, driver_a, driver_b)
            if np.isnan(gap_after):
                continue

            overcut_succeeded = gap_after > 0  # A still ahead after pitting
            gap_gained = gap_after - gap_before  # positive = A extended lead

            # Only flag if A stayed competitive (didn't just give up the gap passively)
            tyre_age_at_pit = race_df[
                (race_df["driver"] == driver_a) & (race_df["lap"] == pit_a)
            ]["tyre_age"].values
            tyre_age = int(tyre_age_at_pit[0]) if len(tyre_age_at_pit) > 0 else 0

            # rank_score: penalise if A lost significant ground during the stay-out
            tyre_extension_score = min(1.0, stay_out / 8.0)
            success_bonus = 0.3 if overcut_succeeded else 0.0
            pace_score = min(0.7, max(0.0, 0.35 + pace_delta / 2.0))
            rank_score = round(min(1.0, pace_score + tyre_extension_score * 0.3 + success_bonus), 3)

            insights.append({
                "insight_type": "overcut",
                "driver": driver_a,
                "lap": pit_a,
                "rank_score": rank_score,
                "summary": (
                    f"{driver_a} overcut {driver_b}: stayed out until lap {pit_a} "
                    f"({stay_out} laps after {driver_b} pitted), "
                    f"tyre age {tyre_age}"
                    + (" — succeeded" if overcut_succeeded else " — did not succeed")
                ),
                "detail": {
                    "target_driver": driver_b,
                    "overcut_driver_pit_lap": pit_a,
                    "target_driver_pit_lap": pit_b,
                    "laps_stayed_out": stay_out,
                    "gap_before_sec": round(gap_before, 3),
                    "gap_after_sec": round(gap_after, 3),
                    "gap_gained_sec": round(gap_gained, 3),
                    "avg_pace_delta_vs_fresh_sec": round(pace_delta, 3),
                    "tyre_age_at_pit": tyre_age,
                    "overcut_succeeded": overcut_succeeded,
                },
            })

    return insights
