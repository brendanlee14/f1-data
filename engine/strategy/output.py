"""
Output formatting and persistence for Strategy Analyser insights.

Provides:
  - write_json()     — save insights to a JSON file
  - load_json()      — load a previously saved insights file
  - to_summary_text() — plain-text summary for logging / CLI
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# Schema version — bump when the Insight JSON schema changes
SCHEMA_VERSION = "1.0"


def _build_envelope(
    insights: List[Dict[str, Any]],
    race_label: str,
    generated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Wrap insights in a metadata envelope."""
    return {
        "schema_version": SCHEMA_VERSION,
        "race": race_label,
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "insight_count": len(insights),
        "insights": insights,
    }


def write_json(
    insights: List[Dict[str, Any]],
    path: str | os.PathLike,
    race_label: str = "Melbourne 2026",
    indent: int = 2,
) -> Path:
    """
    Serialise insights to a JSON file.

    Parameters
    ----------
    insights  : list returned by StrategyAnalyser.run()
    path      : destination file path (created if it doesn't exist)
    race_label: human-readable race name stored in the envelope
    indent    : JSON indentation (set to None for compact output)

    Returns
    -------
    Path to the written file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    envelope = _build_envelope(insights, race_label)
    with out.open("w", encoding="utf-8") as f:
        json.dump(envelope, f, indent=indent, ensure_ascii=False)
    return out


def load_json(path: str | os.PathLike) -> Dict[str, Any]:
    """
    Load an insights file previously written by write_json().

    Returns the full envelope dict (including metadata).
    Raises FileNotFoundError if the file doesn't exist.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_summary_text(
    insights: List[Dict[str, Any]],
    race_label: str = "Melbourne 2026",
    min_rank_score: float = 0.0,
) -> str:
    """
    Format insights as a plain-text summary string.

    Parameters
    ----------
    insights       : list returned by StrategyAnalyser.run()
    race_label     : header label
    min_rank_score : only include insights at or above this threshold
    """
    filtered = [i for i in insights if i["rank_score"] >= min_rank_score]

    lines = [
        f"{'='*60}",
        f"{race_label} — Strategy Insights ({len(filtered)} shown)",
        f"{'='*60}",
    ]

    type_order = ["undercut", "overcut", "tyre_degradation"]
    grouped: Dict[str, List] = {t: [] for t in type_order}
    for ins in filtered:
        grouped.setdefault(ins["insight_type"], []).append(ins)

    section_titles = {
        "undercut": "UNDERCUTS",
        "overcut": "OVERCUTTS",
        "tyre_degradation": "TYRE DEGRADATION",
    }

    for t in type_order:
        group = grouped.get(t, [])
        if not group:
            continue
        lines.append(f"\n{section_titles[t]}")
        lines.append("-" * 40)
        for ins in group:
            lines.append(
                f"  Lap {ins['lap']:3d}  [{ins['rank_score']:.2f}]  {ins['summary']}"
            )

    return "\n".join(lines)
