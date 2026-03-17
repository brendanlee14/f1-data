"""
Quick smoke-test: run the Strategy Analyser against Melbourne 2026 synthetic data
and print all insights sorted by rank_score.
"""

import json
from engine.strategy import StrategyAnalyser


def main():
    analyser = StrategyAnalyser()
    race_df = analyser.load_race()

    print(f"Race data: {len(race_df)} laps across {race_df['driver'].nunique()} drivers\n")

    insights = analyser.run(race_df, min_rank_score=0.1)

    print(f"{'='*60}")
    print(f"Melbourne 2026 — Strategy Insights ({len(insights)} total)")
    print(f"{'='*60}\n")

    for ins in insights:
        print(f"[{ins['insight_type'].upper():20s}] rank={ins['rank_score']:.3f}  lap={ins['lap']:3d}  {ins['summary']}")

    print(f"\n--- Full JSON (top 5) ---")
    print(json.dumps(insights[:5], indent=2))


if __name__ == "__main__":
    main()
