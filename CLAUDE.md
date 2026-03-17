# F1 Data Storytelling Platform — CLAUDE.md

## Project Overview
An F1 data storytelling platform that surfaces race strategy insights via a structured Insight Engine.

## Phase Status
- **Phase 1:** Insight Engine — In Progress
  - **Strategy Analyser module:** Planning stage

## Architecture (planned)
- `/engine/strategy/` — Strategy Analyser module (tyre deg, undercut, overcut detection)

## Insight JSON Schema
Insights produced by all algorithms must conform to:
```json
{
  "insight_type": "string",
  "driver": "string (3-letter code)",
  "lap": "integer",
  "rank_score": "float (0.0–1.0)",
  "summary": "string",
  "detail": "object (algorithm-specific fields)"
}
```

## Environment Notes
- Python 3.11.14
- fastf1 3.8.1, pandas 2.3.3, numpy 2.4.3
- **Network constraint:** `livetiming.formula1.com` and `api.openf1.org` are blocked by the sandbox proxy. Live FastF1 data loading is not available in this environment. A network-enabled environment is required to run algorithms against real race data.
- FastF1 cache directory: `/home/user/f1-data/cache/` (gitignored)

## Session Log

### Session 1 (2026-03-17)
- Confirmed Python + fastf1/pandas/numpy installed
- Discovered network proxy blocks all F1 data API endpoints
- Created `.gitignore` and this `CLAUDE.md`
- **Status:** Awaiting decision on how to proceed (live data vs synthetic mock)
- Tyre degradation detection: not started
- Undercut detection: not started
- Overcut detection: not started
