"""Tests for engine.strategy.output and api.app."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from engine.strategy import StrategyAnalyser
from engine.strategy.output import (
    SCHEMA_VERSION,
    _build_envelope,
    load_json,
    to_summary_text,
    write_json,
)
from api.app import app, RACE_LABEL, VALID_INSIGHT_TYPES

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def insights(melbourne_df):
    return StrategyAnalyser().run(melbourne_df, min_rank_score=0.0)


@pytest.fixture(scope="module")
def api_client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# output.py — _build_envelope
# ---------------------------------------------------------------------------

class TestBuildEnvelope:
    def test_required_keys(self, insights):
        env = _build_envelope(insights, "Test Race")
        for key in ("schema_version", "race", "generated_at", "insight_count", "insights"):
            assert key in env

    def test_schema_version(self, insights):
        env = _build_envelope(insights, "Test Race")
        assert env["schema_version"] == SCHEMA_VERSION

    def test_insight_count_matches(self, insights):
        env = _build_envelope(insights, "Test Race")
        assert env["insight_count"] == len(insights)
        assert len(env["insights"]) == len(insights)

    def test_race_label_stored(self, insights):
        env = _build_envelope(insights, "Specific Race Name")
        assert env["race"] == "Specific Race Name"

    def test_custom_generated_at(self, insights):
        ts = "2026-03-17T12:00:00+00:00"
        env = _build_envelope(insights, "Test", generated_at=ts)
        assert env["generated_at"] == ts

    def test_generated_at_iso_format_by_default(self, insights):
        env = _build_envelope(insights, "Test")
        # Must be parseable as ISO datetime
        from datetime import datetime
        datetime.fromisoformat(env["generated_at"])  # raises if not valid


# ---------------------------------------------------------------------------
# output.py — write_json / load_json
# ---------------------------------------------------------------------------

class TestWriteLoadJson:
    def test_write_creates_file(self, insights):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "insights.json"
            result = write_json(insights, path)
            assert result == path
            assert path.exists()

    def test_written_file_is_valid_json(self, insights):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "insights.json"
            write_json(insights, path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)

    def test_write_creates_parent_dirs(self, insights):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "insights.json"
            write_json(insights, path)
            assert path.exists()

    def test_roundtrip_preserves_insights(self, insights):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "insights.json"
            write_json(insights, path, race_label="Round Trip Test")
            loaded = load_json(path)
            assert loaded["race"] == "Round Trip Test"
            assert loaded["insight_count"] == len(insights)
            assert len(loaded["insights"]) == len(insights)

    def test_roundtrip_insight_fields_preserved(self, insights):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "insights.json"
            write_json(insights, path)
            loaded = load_json(path)
            for orig, saved in zip(insights, loaded["insights"]):
                assert orig["insight_type"] == saved["insight_type"]
                assert orig["driver"] == saved["driver"]
                assert orig["lap"] == saved["lap"]
                assert orig["rank_score"] == saved["rank_score"]

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_json("/tmp/does_not_exist_xyz.json")

    def test_returns_path_object(self, insights):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_json(insights, Path(tmpdir) / "out.json")
            assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# output.py — to_summary_text
# ---------------------------------------------------------------------------

class TestToSummaryText:
    def test_returns_string(self, insights):
        assert isinstance(to_summary_text(insights), str)

    def test_contains_race_label(self, insights):
        text = to_summary_text(insights, race_label="My Race")
        assert "My Race" in text

    def test_contains_all_insight_types(self, insights):
        text = to_summary_text(insights)
        assert "UNDERCUT" in text.upper()
        assert "OVERCUT" in text.upper()
        assert "TYRE" in text.upper()

    def test_min_rank_score_filters(self, insights):
        full = to_summary_text(insights, min_rank_score=0.0)
        filtered = to_summary_text(insights, min_rank_score=0.9)
        assert len(filtered) < len(full)

    def test_empty_insights_returns_string(self):
        text = to_summary_text([])
        assert isinstance(text, str)

    def test_insight_summary_present_in_output(self, insights):
        text = to_summary_text(insights)
        for ins in insights[:3]:
            assert ins["driver"] in text


# ---------------------------------------------------------------------------
# api/app.py — GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, api_client):
        r = api_client.get("/health")
        assert r.status_code == 200

    def test_status_ok(self, api_client):
        r = api_client.get("/health")
        assert r.json()["status"] == "ok"

    def test_schema_version_present(self, api_client):
        r = api_client.get("/health")
        assert r.json()["schema_version"] == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# api/app.py — GET /insights
# ---------------------------------------------------------------------------

class TestInsightsEndpoint:
    def test_returns_200(self, api_client):
        assert api_client.get("/insights").status_code == 200

    def test_envelope_structure(self, api_client):
        data = api_client.get("/insights").json()
        for key in ("schema_version", "race", "generated_at", "insight_count", "insights"):
            assert key in data

    def test_race_label(self, api_client):
        assert api_client.get("/insights").json()["race"] == RACE_LABEL

    def test_returns_list_of_insights(self, api_client):
        data = api_client.get("/insights").json()
        assert isinstance(data["insights"], list)
        assert len(data["insights"]) > 0

    def test_insight_count_matches_list(self, api_client):
        data = api_client.get("/insights").json()
        assert data["insight_count"] == len(data["insights"])

    def test_filter_by_insight_type(self, api_client):
        for t in VALID_INSIGHT_TYPES:
            data = api_client.get(f"/insights?insight_type={t}").json()
            for ins in data["insights"]:
                assert ins["insight_type"] == t

    def test_filter_by_driver(self, api_client):
        data = api_client.get("/insights?driver=VER").json()
        for ins in data["insights"]:
            assert ins["driver"] == "VER"

    def test_driver_filter_case_insensitive(self, api_client):
        upper = api_client.get("/insights?driver=VER").json()["insight_count"]
        lower = api_client.get("/insights?driver=ver").json()["insight_count"]
        assert upper == lower

    def test_min_rank_score_filter(self, api_client):
        all_ins = api_client.get("/insights?min_rank_score=0").json()["insights"]
        high = api_client.get("/insights?min_rank_score=0.8").json()["insights"]
        assert len(high) <= len(all_ins)
        for ins in high:
            assert ins["rank_score"] >= 0.8

    def test_limit_parameter(self, api_client):
        data = api_client.get("/insights?limit=3").json()
        assert len(data["insights"]) <= 3

    def test_invalid_insight_type_returns_422(self, api_client):
        r = api_client.get("/insights?insight_type=banana")
        assert r.status_code == 422

    def test_min_rank_score_out_of_range_returns_422(self, api_client):
        assert api_client.get("/insights?min_rank_score=1.5").status_code == 422
        assert api_client.get("/insights?min_rank_score=-0.1").status_code == 422


# ---------------------------------------------------------------------------
# api/app.py — GET /insights/{insight_type}
# ---------------------------------------------------------------------------

class TestInsightsByTypeEndpoint:
    def test_each_valid_type_returns_200(self, api_client):
        for t in VALID_INSIGHT_TYPES:
            assert api_client.get(f"/insights/{t}").status_code == 200

    def test_results_are_correct_type(self, api_client):
        for t in VALID_INSIGHT_TYPES:
            data = api_client.get(f"/insights/{t}").json()
            for ins in data["insights"]:
                assert ins["insight_type"] == t

    def test_unknown_type_returns_404(self, api_client):
        assert api_client.get("/insights/banana").status_code == 404

    def test_driver_filter_on_type_route(self, api_client):
        data = api_client.get("/insights/tyre_degradation?driver=VER").json()
        for ins in data["insights"]:
            assert ins["driver"] == "VER"
            assert ins["insight_type"] == "tyre_degradation"

    def test_min_rank_score_on_type_route(self, api_client):
        data = api_client.get("/insights/overcut?min_rank_score=0.9").json()
        for ins in data["insights"]:
            assert ins["rank_score"] >= 0.9

    def test_limit_on_type_route(self, api_client):
        data = api_client.get("/insights/tyre_degradation?limit=2").json()
        assert len(data["insights"]) <= 2
