"""
StrategyAnalyser — top-level orchestrator for the Strategy Analyser module.

Usage:
    from engine.strategy import StrategyAnalyser

    analyser = StrategyAnalyser()
    race_df = analyser.load_race()      # loads synthetic Melbourne 2026 data
    insights = analyser.run(race_df)    # returns list of Insight dicts
"""

import pandas as pd
from typing import List, Dict, Any, Optional

from .synthetic_data import simulate_race, MELBOURNE_2026_STRATEGIES
from . import tyre_deg, undercut, overcut


class StrategyAnalyser:
    """Orchestrates tyre_deg, undercut, and overcut detectors."""

    def load_race(self, rng_seed: int = 2026) -> pd.DataFrame:
        """Return a simulated Melbourne 2026 race DataFrame."""
        return simulate_race(strategies=MELBOURNE_2026_STRATEGIES, rng_seed=rng_seed)

    def run(
        self,
        race_df: pd.DataFrame,
        min_rank_score: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Run all detectors and return a merged, rank-sorted list of insights.

        Parameters
        ----------
        race_df : DataFrame from load_race() or a custom simulation
        min_rank_score : discard insights below this threshold (0.0 = keep all)
        """
        all_insights: List[Dict[str, Any]] = []

        all_insights.extend(tyre_deg.detect(race_df))
        all_insights.extend(undercut.detect(race_df))
        all_insights.extend(overcut.detect(race_df))

        # Filter and sort
        filtered = [i for i in all_insights if i["rank_score"] >= min_rank_score]
        filtered.sort(key=lambda x: x["rank_score"], reverse=True)

        return filtered
