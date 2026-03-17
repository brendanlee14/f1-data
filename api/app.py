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
) -> JSONResponse:
    """
    Return all insights, optionally filtered.

    Query parameters
    ----------------
    min_rank_score : float (0.0–1.0) — minimum rank score to include
    insight_type   : one of tyre_degradation | undercut | overcut
    driver         : 3-letter driver code (e.g. VER, NOR)
    limit          : max number of insights to return (default 100)
    """
    if insight_type and insight_type not in VALID_INSIGHT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"insight_type must be one of {sorted(VALID_INSIGHT_TYPES)}",
        )

    results = _all_insights

    if min_rank_score > 0.0:
        results = [i for i in results if i["rank_score"] >= min_rank_score]

    if insight_type:
        results = [i for i in results if i["insight_type"] == insight_type]

    if driver:
        results = [i for i in results if i["driver"] == driver.upper()]

    results = results[:limit]

    envelope = _build_envelope(results, RACE_LABEL)
    return JSONResponse(content=envelope)


@app.get("/insights/{insight_type}")
def get_insights_by_type(
    insight_type: str,
    min_rank_score: float = Query(default=0.0, ge=0.0, le=1.0),
    driver: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> JSONResponse:
    """
    Return insights of a specific type.

    Path parameter
    --------------
    insight_type : tyre_degradation | undercut | overcut
    """
    if insight_type not in VALID_INSIGHT_TYPES:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown insight type '{insight_type}'. "
                   f"Valid types: {sorted(VALID_INSIGHT_TYPES)}",
        )

    results = [i for i in _all_insights if i["insight_type"] == insight_type]

    if min_rank_score > 0.0:
        results = [i for i in results if i["rank_score"] >= min_rank_score]

    if driver:
        results = [i for i in results if i["driver"] == driver.upper()]

    results = results[:limit]

    envelope = _build_envelope(results, RACE_LABEL)
    return JSONResponse(content=envelope)
