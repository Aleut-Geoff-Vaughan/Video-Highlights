from __future__ import annotations

from backend.config import settings
from backend.models import Event, Match
from backend.services.llm_agent import AgentService


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


def test_agent_service_uses_ollama_provider(monkeypatch) -> None:
    service = AgentService()
    monkeypatch.setattr(service, "_get_match_events", lambda session, tenant_id, match_id, limit: [])

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_model", "llama3.2:3b")
    monkeypatch.setattr(settings, "llm_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr(settings, "llm_timeout_seconds", 5.0)

    def _fake_post(url, json, timeout):  # noqa: ANN001
        assert url == "http://127.0.0.1:11434/api/chat"
        assert json["model"] == "llama3.2:3b"
        assert json["stream"] is False
        assert timeout == 5.0
        return _FakeResponse({"message": {"content": "Local model summary"}})

    monkeypatch.setattr("backend.services.llm_agent.requests.post", _fake_post)

    result = service.query_match(
        session=None,  # type: ignore[arg-type]
        tenant_id="tenant_local",
        match_id="match_local",
        query="summarize the match",
        limit=20,
    )
    assert result["provider"] == "ollama"
    assert result["model"] == "llama3.2:3b"
    assert result["answer"] == "Local model summary"


def test_agent_service_falls_back_when_ollama_returns_no_text(monkeypatch) -> None:
    service = AgentService()
    monkeypatch.setattr(service, "_get_match_events", lambda session, tenant_id, match_id, limit: [])

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_model", "llama3.2:3b")
    monkeypatch.setattr(settings, "llm_base_url", "http://127.0.0.1:11434")

    def _fake_post(url, json, timeout):  # noqa: ANN001
        return _FakeResponse({"message": {"content": ""}})

    monkeypatch.setattr("backend.services.llm_agent.requests.post", _fake_post)

    result = service.query_match(
        session=None,  # type: ignore[arg-type]
        tenant_id="tenant_local",
        match_id="match_local",
        query="summarize the match",
        limit=20,
    )
    assert result["provider"] == "fallback"
    assert result["model"] is None


def test_prepare_llm_payload_includes_match_and_signal_context() -> None:
    service = AgentService()
    match = Match(
        id="match_local",
        tenant_id="tenant_local",
        name="U12 Semifinal",
        home_team_name="Blue FC",
        away_team_name="Red FC",
        match_date="2026-05-09",
        source_video_path="C:/tmp/match.mp4",
        metadata_json={"requested_targets": ["goal", "save"]},
    )
    event = Event(
        id="evt_local",
        tenant_id="tenant_local",
        match_id="match_local",
        event_type="shot",
        confidence=0.88,
        occurred_at_ms=125000,
        start_ms=123000,
        end_ms=128000,
        team_id="blue",
        player_id="player_10",
        jersey_number="10",
        source_json={
            "detector": "videohighlights-multi-factor",
            "camera_mode": "follow_action",
            "zoom_factor": 1.8,
            "sources": ["motion", "audio"],
        },
        evidence_json={"tracking_manifest_path": "C:/tmp/analysis_tracking.json"},
        explanations_json=[{"signal": "speed_overlap_s", "value": 2.1}],
    )

    system, user = service._prepare_llm_payload(match, [event], query="Summarize the best moments.")

    assert "soccer video analysis" in system
    assert "U12 Semifinal" in user
    assert '"requested_targets": [' in user
    assert '"camera_mode": "follow_action"' in user
    assert '"signal": "speed_overlap_s"' in user
