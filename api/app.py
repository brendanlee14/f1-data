"""
F1 Strategy Insights API.

Serves insights from the Strategy Analyser module over HTTP.

Run locally:
    uvicorn api.app:app --reload --port 8000

Endpoints:
    GET /health
    GET /insights
    GET /insights/{insight_type}
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import os

from engine.strategy import StrategyAnalyser
from engine.strategy.fastf1_loader import load_fastf1_race
from engine.strategy.output import _build_envelope, SCHEMA_VERSION
from engine.strategy.synthetic_data import RACES_2026, simulate_race_for_round

STATIC_DIR = Path(__file__).parent.parent / "static"

app = FastAPI(
    title="F1 Strategy Insights",
    description="Race strategy insights powered by the Strategy Analyser engine.",
    version="0.1.0",
)

VALID_INSIGHT_TYPES = {"tyre_degradation", "undercut", "overcut"}

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ---------------------------------------------------------------------------
# Shared state — race data and insights loaded once at startup
# ---------------------------------------------------------------------------

_analyser = StrategyAnalyser()

_F1_YEAR = os.getenv("F1_YEAR")
_F1_GP   = os.getenv("F1_GP")

if _F1_YEAR and _F1_GP:
    _race_df   = load_fastf1_race(int(_F1_YEAR), _F1_GP)
    RACE_LABEL = f"{_F1_GP} {_F1_YEAR}"
else:
    _race_df   = _analyser.load_race()
    RACE_LABEL = "Melbourne 2026 (synthetic)"

_all_insights: List[Dict[str, Any]] = _analyser.run(_race_df, min_rank_score=0.0)

# Pre-compute synthetic race DataFrames for all 2026 rounds (used by race-pace view)
_race_pace_cache: Dict[str, Any] = {
    r["key"]: simulate_race_for_round(r["key"]) for r in RACES_2026
}

# Pre-compute strategy insights for all 2026 rounds
_insights_cache: Dict[str, List[Dict[str, Any]]] = {
    r["key"]: _analyser.run(_race_pace_cache[r["key"]], min_rank_score=0.0)
    for r in RACES_2026
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    """Serve the frontend."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
def health() -> Dict[str, str]:
    """Liveness check."""
    return {"status": "ok", "schema_version": SCHEMA_VERSION}


@app.get("/insights")
def get_insights(
    min_rank_score: float = Query(default=0.0, ge=0.0, le=1.0),
    insight_type: Optional[str] = Query(default=None),
    driver: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    race: str = Query(default="australia"),
) -> JSONResponse:
    """
    Return all insights, optionally filtered.

    Query parameters
    ----------------
    min_rank_score : float (0.0–1.0) — minimum rank score to include
    insight_type   : one of tyre_degradation | undercut | overcut
    driver         : 3-letter driver code (e.g. VER, NOR)
    limit          : max number of insights to return (default 100)
    race           : 2026 race key (e.g. australia, monaco). Defaults to australia.
    """
    if race not in _insights_cache:
        raise HTTPException(status_code=404, detail=f"Unknown race key: {race!r}")

    if insight_type and insight_type not in VALID_INSIGHT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"insight_type must be one of {sorted(VALID_INSIGHT_TYPES)}",
        )

    race_meta = next(r for r in RACES_2026 if r["key"] == race)
    race_label = f"{race_meta['name']} · {race_meta['circuit']} 2026"

    results = _insights_cache[race]

    if min_rank_score > 0.0:
        results = [i for i in results if i["rank_score"] >= min_rank_score]

    if insight_type:
        results = [i for i in results if i["insight_type"] == insight_type]

    if driver:
        results = [i for i in results if i["driver"] == driver.upper()]

    results = results[:limit]

    envelope = _build_envelope(results, race_label)
    return JSONResponse(content=envelope)


@app.get("/race-pace")
def race_pace_page() -> FileResponse:
    """Serve the race pace visualisation page."""
    return FileResponse(str(STATIC_DIR / "race_pace.html"))


@app.get("/api/races")
def get_races() -> JSONResponse:
    """Return the full 2026 calendar (key, name, circuit, round)."""
    return JSONResponse(content={"races": RACES_2026})


@app.get("/api/race-pace")
def get_race_pace(race: str = Query(default="australia")) -> JSONResponse:
    """
    Return aggregated race pace data for all drivers.

    Query parameters
    ----------------
    race : 2026 race key (e.g. australia, china, monaco). Defaults to australia.

    Per driver: box stats (Q1/median/Q3/whiskers/outliers) from clean laps,
    mean lap time, delta to leader, strategy stints, and per-lap times for
    the smoothed trace chart.
    """
    import numpy as np

    if race not in _race_pace_cache:
        raise HTTPException(status_code=404, detail=f"Unknown race key: {race!r}")

    race_meta = next(r for r in RACES_2026 if r["key"] == race)
    active_df = _race_pace_cache[race]

    result_drivers = []

    for driver, driver_df in active_df.groupby("driver"):
        driver_df = driver_df.sort_values("lap").reset_index(drop=True)
        team = str(driver_df["team"].iloc[0])

        # Flag anomalous laps (pit laps and SC/VSC laps >10s above stint median)
        median_all = float(np.median(driver_df["lap_time"].values))
        is_pit    = driver_df["pit_this_lap"].astype(bool)
        is_flagged = driver_df["lap_time"] > median_all + 10.0
        clean_mask = (~is_pit) & (~is_flagged)
        clean_times = sorted(driver_df.loc[clean_mask, "lap_time"].tolist())

        if len(clean_times) < 4:
            continue

        q1     = float(np.percentile(clean_times, 25))
        median = float(np.percentile(clean_times, 50))
        q3     = float(np.percentile(clean_times, 75))
        iqr    = q3 - q1
        wlo    = float(max(min(clean_times), q1 - 1.5 * iqr))
        whi    = float(min(max(clean_times), q3 + 1.5 * iqr))
        outliers = [round(t, 3) for t in clean_times if t < wlo or t > whi]
        mean_clean = float(np.mean(clean_times))

        # Strategy stints
        stints = []
        for stint_num, s_df in driver_df.groupby("stint_number"):
            stints.append({
                "stint":    int(stint_num),
                "compound": str(s_df["compound"].iloc[0]),
                "start_lap": int(s_df["lap"].iloc[0]),
                "end_lap":   int(s_df["lap"].iloc[-1]),
            })

        # Per-lap times (null for pit/flagged — used by trace chart)
        lap_times: dict = {}
        for _, row in driver_df.iterrows():
            lap_num = int(row["lap"])
            if bool(row["pit_this_lap"]) or float(row["lap_time"]) > median_all + 10.0:
                lap_times[str(lap_num)] = None
            else:
                lap_times[str(lap_num)] = round(float(row["lap_time"]), 3)

        result_drivers.append({
            "driver":        driver,
            "team":          team,
            "mean":          round(mean_clean, 3),
            "compounds_used": list(dict.fromkeys(  # ordered, deduped
                s["compound"] for s in stints
            )),
            "box": {
                "min":      round(wlo, 3),
                "q1":       round(q1, 3),
                "median":   round(median, 3),
                "q3":       round(q3, 3),
                "max":      round(whi, 3),
                "outliers": outliers,
            },
            "stints":     stints,
            "lap_times":  lap_times,
        })

    # Sort fastest → slowest by mean clean lap time
    result_drivers.sort(key=lambda d: d["mean"])

    # Delta to leader
    if result_drivers:
        leader_mean = result_drivers[0]["mean"]
        for d in result_drivers:
            d["delta"] = round(d["mean"] - leader_mean, 3)

    return JSONResponse(content={
        "race":     race_meta["name"],
        "race_key": race_meta["key"],
        "circuit":  race_meta["circuit"],
        "round":    race_meta["round"],
        "drivers":  result_drivers,
    })


@app.get("/insights/{insight_type}")
def get_insights_by_type(
    insight_type: str,
    min_rank_score: float = Query(default=0.0, ge=0.0, le=1.0),
    driver: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    race: str = Query(default="australia"),
) -> JSONResponse:
    """
    Return insights of a specific type.

    Path parameter
    --------------
    insight_type : tyre_degradation | undercut | overcut

    Query parameters
    ----------------
    race : 2026 race key. Defaults to australia.
    """
    if insight_type not in VALID_INSIGHT_TYPES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown insight type '{insight_type}'. "
                   f"Valid types: {sorted(VALID_INSIGHT_TYPES)}",
        )

    if race not in _insights_cache:
        raise HTTPException(status_code=404, detail=f"Unknown race key: {race!r}")

    race_meta = next(r for r in RACES_2026 if r["key"] == race)
    race_label = f"{race_meta['name']} · {race_meta['circuit']} 2026"

    results = [i for i in _insights_cache[race] if i["insight_type"] == insight_type]

    if min_rank_score > 0.0:
        results = [i for i in results if i["rank_score"] >= min_rank_score]

    if driver:
        results = [i for i in results if i["driver"] == driver.upper()]

    results = results[:limit]

    envelope = _build_envelope(results, race_label)
    return JSONResponse(content=envelope)
