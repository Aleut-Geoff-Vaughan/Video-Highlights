from __future__ import annotations

import json
from collections import Counter
from typing import Any
from typing import Dict, List, Optional, Tuple

import requests
from sqlmodel import Session, select

from ..config import settings
from ..models import Event, Match


class AgentService:
    @property
    def provider(self) -> str:
        raw = str(settings.llm_provider or "none").strip().lower()
        aliases = {
            "local": "ollama",
            "lmstudio": "openai_compatible",
            "openai-compatible": "openai_compatible",
        }
        return aliases.get(raw, raw or "none")

    @property
    def model(self) -> str:
        return str(settings.llm_model or "").strip() or "gpt-4o-mini"

    def status(self) -> Dict[str, object]:
        provider = self.provider
        status: Dict[str, object] = {
            "provider": provider,
            "model": self.model if provider != "none" else None,
            "configured": provider in {"openai", "openai_compatible", "ollama"},
            "base_url": settings.llm_base_url,
            "reachable": None,
            "message": "",
        }
        if provider == "none":
            status["message"] = "LLM provider is not configured; assistant will use deterministic fallback answers."
            return status
        if provider == "openai" and not settings.openai_api_key:
            status["configured"] = False
            status["reachable"] = False
            status["message"] = "OPENAI_API_KEY is not set."
            return status
        if provider == "openai_compatible" and not settings.llm_base_url:
            status["configured"] = False
            status["reachable"] = False
            status["message"] = "VH_LLM_BASE_URL is required for OpenAI-compatible local servers."
            return status
        if provider == "ollama":
            base_url = str(settings.llm_base_url or "http://127.0.0.1:11434").rstrip("/")
            if base_url.endswith("/v1"):
                base_url = base_url[:-3].rstrip("/")
            status["base_url"] = base_url
            try:
                response = requests.get(f"{base_url}/api/tags", timeout=min(5.0, float(settings.llm_timeout_seconds)))
                response.raise_for_status()
                payload = response.json()
                models = [str(item.get("name")) for item in list(payload.get("models", []) or []) if isinstance(item, dict)]
                status["reachable"] = True
                status["models"] = models
                if self.model not in models:
                    status["configured"] = False
                    status["message"] = f"Ollama is reachable, but model '{self.model}' is not installed. Run `ollama pull {self.model}`."
                    return status
                status["message"] = "Ollama is reachable."
            except Exception as exc:
                status["reachable"] = False
                status["message"] = f"Ollama is not reachable: {exc}"
            return status

        status["message"] = "Provider configuration will be checked when a request is sent."
        return status

    def query_match(
        self,
        session: Session,
        tenant_id: str,
        match_id: str,
        query: str,
        limit: int = 50,
    ) -> Dict[str, object]:
        match = self._get_match(session, tenant_id, match_id)
        events = self._get_match_events(session, tenant_id, match_id, limit=limit)
        fallback = self._fallback_query_answer(match, match_id, events, query)
        referenced_ids = [event.id for event in events[: min(len(events), 20)]]

        llm_answer = self._try_llm_query(match, events, query)
        if llm_answer:
            return {
                "provider": self.provider,
                "model": self.model,
                "answer": llm_answer,
                "referenced_event_ids": referenced_ids,
            }

        return {
            "provider": "fallback",
            "model": None,
            "answer": fallback,
            "referenced_event_ids": referenced_ids,
        }

    def explain_event(
        self,
        session: Session,
        tenant_id: str,
        match_id: str,
        event_id: str,
        question: Optional[str] = None,
    ) -> Dict[str, object]:
        match = self._get_match(session, tenant_id, match_id)
        event = session.exec(
            select(Event)
            .where(Event.tenant_id == tenant_id)
            .where(Event.match_id == match_id)
            .where(Event.id == event_id)
        ).first()
        if not event:
            return {
                "provider": "fallback",
                "model": None,
                "answer": f"Event {event_id} was not found for match {match_id}.",
                "referenced_event_ids": [],
            }

        fallback_answer = self._fallback_event_explanation(match, event, question)
        llm_answer = self._try_llm_explanation(match, event, question)
        if llm_answer:
            return {
                "provider": self.provider,
                "model": self.model,
                "answer": llm_answer,
                "referenced_event_ids": [event.id],
            }

        return {
            "provider": "fallback",
            "model": None,
            "answer": fallback_answer,
            "referenced_event_ids": [event.id],
        }

    def _get_match_events(self, session: Session, tenant_id: str, match_id: str, limit: int) -> List[Event]:
        stmt = (
            select(Event)
            .where(Event.tenant_id == tenant_id)
            .where(Event.match_id == match_id)
            .order_by(Event.occurred_at_ms.desc())
            .limit(max(1, min(limit, 200)))
        )
        return list(session.exec(stmt))

    def _get_match(self, session: Session, tenant_id: str, match_id: str) -> Optional[Match]:
        if session is None:
            return None
        match = session.get(Match, match_id)
        if not match or match.tenant_id != tenant_id:
            return None
        return match

    def _match_label(self, match: Optional[Match], fallback_match_id: Optional[str] = None) -> str:
        if match is None:
            return fallback_match_id or "this match"
        name = str(match.name or "").strip()
        if name:
            return name
        teams = " vs ".join(part for part in [str(match.home_team_name or "").strip(), str(match.away_team_name or "").strip()] if part)
        if teams:
            return teams
        return fallback_match_id or str(match.id)

    def _format_clock_ms(self, value: int) -> str:
        total_seconds = max(0, int(round(float(value) / 1000.0)))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def _event_context_row(self, event: Event) -> Dict[str, object]:
        source = dict(event.source_json or {})
        evidence = dict(event.evidence_json or {})
        explanations = list(event.explanations_json or [])
        duration_ms = max(0, int(event.end_ms) - int(event.start_ms))
        return {
            "event_id": event.id,
            "event_type": event.event_type,
            "status": event.status,
            "confidence": round(float(event.confidence), 3),
            "time": self._format_clock_ms(int(event.occurred_at_ms)),
            "occurred_at_ms": int(event.occurred_at_ms),
            "start_ms": int(event.start_ms),
            "end_ms": int(event.end_ms),
            "duration_s": round(duration_ms / 1000.0, 3),
            "team_id": event.team_id,
            "player_id": event.player_id,
            "jersey_number": event.jersey_number,
            "detector": source.get("detector"),
            "camera_mode": source.get("camera_mode"),
            "zoom_factor": source.get("zoom_factor"),
            "bookmark_label": source.get("bookmark_label"),
            "sources": list(source.get("sources", []) or []),
            "signals": [
                {
                    "signal": str(item.get("signal") or ""),
                    "value": item.get("value"),
                }
                for item in explanations[:4]
                if isinstance(item, dict) and item.get("signal")
            ],
            "has_tracking_manifest": bool(evidence.get("tracking_manifest_path")),
        }

    def _fallback_query_answer(self, match: Optional[Match], match_id: str, events: List[Event], query: str) -> str:
        if not events:
            return f"No detected events are available yet for {self._match_label(match, match_id)}."

        counts = Counter(event.event_type for event in events)
        total = len(events)
        top_three = ", ".join(f"{k}: {v}" for k, v in counts.most_common(3))
        strongest = sorted(events, key=lambda event: (float(event.confidence), int(event.occurred_at_ms)), reverse=True)[:3]
        strongest_text = ", ".join(
            f"{event.event_type} at {self._format_clock_ms(int(event.occurred_at_ms))} ({float(event.confidence):.2f})"
            for event in strongest
        )
        label = self._match_label(match, match_id)

        return (
            f"{label} has {total} detected events. Most common event types: {top_three}. "
            f"Highest-confidence moments: {strongest_text}. "
            f"Query received: '{query}'. Use the event list for exact slices by team, player, or time range."
        )

    def _fallback_event_explanation(self, match: Optional[Match], event: Event, question: Optional[str]) -> str:
        parts = [
            f"Match: {self._match_label(match, event.match_id)}.",
            f"Event {event.id} is labeled as '{event.event_type}' with confidence {event.confidence:.2f}.",
            (
                f"Timestamp window: {self._format_clock_ms(int(event.start_ms))} to "
                f"{self._format_clock_ms(int(event.end_ms))} "
                f"(occurred_at={self._format_clock_ms(int(event.occurred_at_ms))})."
            ),
        ]

        if event.team_id:
            parts.append(f"Team: {event.team_id}.")
        if event.player_id:
            parts.append(f"Player: {event.player_id}.")
        if event.explanations_json:
            top_signals = ", ".join(
                f"{item.get('signal')}={item.get('value')}" for item in event.explanations_json[:3]
            )
            parts.append(f"Signals: {top_signals}.")
        if question:
            parts.append(f"Question received: '{question}'.")
        return " ".join(parts)

    def _try_llm_query(self, match: Optional[Match], events: List[Event], query: str) -> Optional[str]:
        payload = self._prepare_llm_payload(match, events, query=query)
        return self._call_llm(payload)

    def _try_llm_explanation(self, match: Optional[Match], event: Event, question: Optional[str]) -> Optional[str]:
        query = question or "Explain why this event was detected and what confidence caveats apply."
        payload = self._prepare_llm_payload(match, [event], query=query)
        return self._call_llm(payload)

    def _prepare_llm_payload(self, match: Optional[Match], events: List[Event], query: str) -> Tuple[str, str]:
        ordered_events = sorted(events[:100], key=lambda event: int(event.occurred_at_ms))
        summary_rows = [self._event_context_row(event) for event in ordered_events]
        counts = Counter(event.event_type for event in ordered_events)
        strongest_events = sorted(
            ordered_events,
            key=lambda event: (float(event.confidence), int(event.occurred_at_ms)),
            reverse=True,
        )[:8]

        match_context = {
            "match_id": str(match.id) if match is not None else None,
            "name": str(match.name or "").strip() if match is not None else None,
            "home_team_name": str(match.home_team_name or "").strip() if match is not None else None,
            "away_team_name": str(match.away_team_name or "").strip() if match is not None else None,
            "match_date": str(match.match_date or "").strip() if match is not None else None,
            "requested_targets": (
                list((match.metadata_json or {}).get("requested_targets", []) or [])
                if match is not None and isinstance(match.metadata_json, dict)
                else []
            ),
        }

        system = (
            "You are an assistant for soccer video analysis. "
            "Use only the provided match and event context. "
            "Do not claim you watched the video. "
            "When asked about missing moments, frame them as review candidates or hypotheses, not facts. "
            "Be concise, practical, and mention uncertainty when event evidence looks weak."
        )
        user = (
            f"Query: {query}\n\n"
            "Match context JSON:\n"
            f"{json.dumps(match_context, indent=2)}\n\n"
            "Event counts by type:\n"
            f"{json.dumps(dict(counts), indent=2)}\n\n"
            "Highest-confidence events JSON:\n"
            f"{json.dumps([self._event_context_row(event) for event in strongest_events], indent=2)}\n\n"
            "Event context JSON:\n"
            f"{json.dumps(summary_rows, indent=2)}"
        )
        return system, user

    def _call_llm(self, payload: Tuple[str, str]) -> Optional[str]:
        if self.provider == "openai":
            return self._call_openai_chat(payload, api_key=settings.openai_api_key, base_url=None)
        if self.provider == "openai_compatible":
            if not settings.llm_base_url:
                return None
            api_key = settings.llm_api_key or "local-dev-key"
            return self._call_openai_chat(payload, api_key=api_key, base_url=settings.llm_base_url)
        if self.provider == "ollama":
            return self._call_ollama(payload)
        return None

    def _call_openai_chat(
        self,
        payload: Tuple[str, str],
        api_key: Optional[str],
        base_url: Optional[str],
    ) -> Optional[str]:
        if not api_key:
            return None

        try:
            from openai import OpenAI
        except Exception:
            return None

        system, user = payload
        try:
            client_kwargs: Dict[str, Any] = {"api_key": api_key}
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            if not response.choices:
                return None
            message = response.choices[0].message
            text = self._extract_message_text(getattr(message, "content", None))
            return text.strip() or None
        except Exception:
            return None

    def _call_ollama(self, payload: Tuple[str, str]) -> Optional[str]:
        base_url = str(settings.llm_base_url or "http://127.0.0.1:11434").rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3].rstrip("/")
        endpoint = f"{base_url}/api/chat"
        system, user = payload
        body = {
            "model": self.model,
            "stream": False,
            "keep_alive": settings.llm_keep_alive,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = requests.post(endpoint, json=body, timeout=float(settings.llm_timeout_seconds))
            response.raise_for_status()
            payload_json = response.json()
            message = payload_json.get("message", {}) if isinstance(payload_json, dict) else {}
            text = message.get("content") if isinstance(message, dict) else None
            if isinstance(text, str):
                return text.strip() or None
            return None
        except Exception:
            return None

    def _extract_message_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    else:
                        nested = item.get("content")
                        if isinstance(nested, str):
                            parts.append(nested)
            return "\n".join(part for part in parts if part).strip()
        return ""


agent_service = AgentService()
