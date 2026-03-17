"""
Undercut detector.

An undercut occurs when Driver B pits *before* Driver A (who is ahead on track),
returns on fresh tyres, and uses the pace advantage to jump Driver A when they
eventually pit.

Algorithm:
  1. For each pair of drivers that are within a gap threshold before the first
     pit stop, check if Driver B pits 1–4 laps earlier than Driver A.
  2. After Driver B's pit, measure whether B's lap times on fresh rubber are
     faster than A's degrading times.
  3. Check whether B emerges ahead of (or within DRS range of) A after A pits.
  4. rank_score is derived from the position delta and pace advantage.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .utils import build_position_index

GAP_THRESHOLD_SEC = 3.5    # max gap to consider a battle
UNDERCUT_WINDOW = 3        # laps after B's pit to measure pace advantage
MIN_PACE_ADVANTAGE = 0.15  # sec/lap advantage B must have to count
DRS_WINDOW_SEC = 1.0       # gap within which DRS is relevant post-jump
MAX_POSITION_GAP = 3       # only consider drivers within this many positions


def _cumulative_gap(race_df: pd.DataFrame, lap: int, driver_a: str, driver_b: str) -> float:
    """Gap between A and B at end of `lap` (positive = A ahead)."""
    def cum(d):
        row = race_df[(race_df["driver"] == d) & (race_df["lap"] == lap)]
        return float(row["cumulative_time"].iloc[0]) if not row.empty else np.nan
    return cum(driver_b) - cum(driver_a)


def detect(race_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Returns a list of undercut insights.
    """
    insights: List[Dict[str, Any]] = []
    drivers = race_df["driver"].unique().tolist()

    # Pre-index pit events: driver → list of pit laps
    pit_laps: Dict[str, List[int]] = {}
    for driver, df in race_df.groupby("driver"):
        pit_laps[driver] = df[df["pit_this_lap"]]["lap"].tolist()

    positions = build_position_index(race_df)

    for i, driver_a in enumerate(drivers):
        for driver_b in drivers[i + 1 :]:
            pits_a = pit_laps.get(driver_a, [])
            pits_b = pit_laps.get(driver_b, [])

            if not pits_a or not pits_b:
                continue

            # Check first pit of each driver
            pit_a = pits_a[0]
            pit_b = pits_b[0]

            # B must pit 1–4 laps before A
            lap_diff = pit_a - pit_b
            if not (1 <= lap_diff <= 4):
                # Try swapped roles
                lap_diff_swap = pit_b - pit_a
                if not (1 <= lap_diff_swap <= 4):
                    continue
                # Swap so that B undercuts A
                driver_a, driver_b = driver_b, driver_a
                pit_a, pit_b = pit_b, pit_a
                lap_diff = lap_diff_swap

            # Check gap before B pits
            check_lap = pit_b - 1
            if check_lap < 1:
                continue
            gap_before = _cumulative_gap(race_df, check_lap, driver_a, driver_b)
            if abs(gap_before) > GAP_THRESHOLD_SEC:
                continue  # not in a battle
            if gap_before < 0:
                continue  # B is already ahead — not an undercut scenario

            # Position filter: only flag drivers actually racing each other
            lap_positions = positions.get(check_lap, {})
            pos_a = lap_positions.get(driver_a)
            pos_b = lap_positions.get(driver_b)
            if pos_a is None or pos_b is None:
                continue
            position_gap = abs(pos_a - pos_b)
            if position_gap > MAX_POSITION_GAP:
                continue

            # Measure B's pace advantage on fresh tyres vs A's degrading laps.
            # Exclude any lap where a driver is doing their pit stop (lap_time inflated).
            a_window = race_df[
                (race_df["driver"] == driver_a)
                & (race_df["lap"].between(pit_b, pit_b + UNDERCUT_WINDOW - 1))
                & (~race_df["pit_this_lap"])
            ]
            b_window = race_df[
                (race_df["driver"] == driver_b)
                & (race_df["lap"].between(pit_b + 1, pit_b + UNDERCUT_WINDOW))
                & (~race_df["pit_this_lap"])
            ]

            if a_window.empty or b_window.empty:
                continue

            avg_a = float(a_window["lap_time"].mean())
            avg_b = float(b_window["lap_time"].mean())
            pace_adv = avg_a - avg_b  # positive = B faster

            if pace_adv < MIN_PACE_ADVANTAGE:
                continue

            # Check gap after A pits (did the undercut actually work?)
            gap_after = _cumulative_gap(race_df, pit_a, driver_a, driver_b)
            undercut_succeeded = gap_after < 0  # B now ahead of A

            # rank_score: blend of pace advantage and success
            pace_score = min(1.0, pace_adv / 1.5)
            success_bonus = 0.25 if undercut_succeeded else 0.0
            rank_score = round(min(1.0, pace_score + success_bonus), 3)

            insights.append({
                "insight_type": "undercut",
                "driver": driver_b,
                "lap": pit_b,
                "rank_score": rank_score,
                "summary": (
                    f"{driver_b} undercut {driver_a}: pitted lap {pit_b} "
                    f"({lap_diff} laps early), "
                    f"pace adv {pace_adv:.2f} s/lap"
                    + (" — succeeded" if undercut_succeeded else " — did not succeed")
                ),
                "detail": {
                    "target_driver": driver_a,
                    "undercut_driver_pit_lap": pit_b,
                    "target_driver_pit_lap": pit_a,
                    "laps_early": lap_diff,
                    "gap_before_sec": round(gap_before, 3),
                    "gap_after_sec": round(gap_after, 3),
                    "pace_advantage_sec_per_lap": round(pace_adv, 3),
                    "undercut_succeeded": undercut_succeeded,
                    "position_a": pos_a,
                    "position_b": pos_b,
                    "position_gap": position_gap,
                },
            })

    return insights
