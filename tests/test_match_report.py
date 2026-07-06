from __future__ import annotations

from types import SimpleNamespace

from backend.config import settings
from backend.services.match_report import generate_match_report

SUMMARY = {
    "goal_events": [{"t": 487.0, "side": "left", "team": "Hawks", "confidence": 0.86}],
    "card_events": [{"t": 1201.0, "kind": "yellow_card", "confidence": 0.71}],
    "team_stats": {"possession_pct": {"Lions": 58.0, "Hawks": 42.0}},
}


def test_fallback_report_without_llm(monkeypatch) -> None:
    monkeypatch.setattr(settings, "llm_provider", "none")
    report = generate_match_report(SUMMARY)
    assert report.startswith("# Match Report")
    assert "Goals flagged: 1" in report
    assert "Hawks" in report
    assert "Cards flagged: 1" in report
    assert "Possession" in report


def test_llm_report_uses_ollama_endpoint_and_model(monkeypatch) -> None:
    import openai

    calls: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            message = SimpleNamespace(content="# Hawks edge it late\nA tight match.")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    class FakeClient:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(openai, "OpenAI", FakeClient)
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_model", "llama3.1:8b")
    monkeypatch.setattr(settings, "llm_base_url", None)

    report = generate_match_report(SUMMARY)

    assert report.startswith("# Hawks edge it late")
    assert calls["client"]["base_url"] == "http://localhost:11434/v1"
    assert calls["create"]["model"] == "llama3.1:8b"
    # The structured data must be in the prompt - the LLM narrates, YOLO decides.
    prompt = calls["create"]["messages"][0]["content"]
    assert "goal_events" in prompt and "487" in prompt

    # A base URL configured without /v1 (as the agent settings use it) is
    # normalized to Ollama's OpenAI-compatible endpoint.
    monkeypatch.setattr(settings, "llm_base_url", "http://192.168.1.50:11434")
    generate_match_report(SUMMARY)
    assert calls["client"]["base_url"] == "http://192.168.1.50:11434/v1"


def test_llm_failure_falls_back_to_template(monkeypatch) -> None:
    import openai

    class BrokenClient:
        def __init__(self, **kwargs):
            raise ConnectionError("ollama is not running")

    monkeypatch.setattr(openai, "OpenAI", BrokenClient)
    monkeypatch.setattr(settings, "llm_provider", "ollama")

    report = generate_match_report(SUMMARY)
    assert report.startswith("# Match Report")
    assert "Goals flagged: 1" in report
