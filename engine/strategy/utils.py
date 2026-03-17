"""Shared utilities for Strategy Analyser detectors."""

import pandas as pd


def build_position_index(race_df: pd.DataFrame) -> dict[int, dict[str, int]]:
    """
    Build a lap-by-lap position index from cumulative race times.

    Returns {lap: {driver: position}} where position 1 = race leader
    (lowest cumulative time).
    """
    index: dict[int, dict[str, int]] = {}
    for lap, lap_df in race_df.groupby("lap"):
        sorted_drivers = lap_df.sort_values("cumulative_time")["driver"].tolist()
        index[int(lap)] = {driver: pos + 1 for pos, driver in enumerate(sorted_drivers)}
    return index
