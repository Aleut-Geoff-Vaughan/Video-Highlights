from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

from sqlmodel import Session, select

from ..config import settings
from ..models import Event


class AgentService:
    def __init__(self) -> None:
        self.provider = settings.llm_provider
        self.model = settings.llm_model

    def query_match(
        self,
        session: Session,
        tenant_id: str,
        match_id: str,
        query: str,
        limit: int = 50,
    ) -> Dict[str, object]:
        events = self._get_match_events(session, tenant_id, match_id, limit=limit)
        fallback = self._fallback_query_answer(events, query)
        referenced_ids = [event.id for event in events[: min(len(events), 20)]]

        llm_answer = self._try_llm_query(events, query)
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

        fallback_answer = self._fallback_event_explanation(event, question)
        llm_answer = self._try_llm_explanation(event, question)
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

    def _fallback_query_answer(self, events: List[Event], query: str) -> str:
        if not events:
            return "No events are available yet for this match."

        counts = Counter(event.event_type for event in events)
        total = len(events)
        top_three = ", ".join(f"{k}: {v}" for k, v in counts.most_common(3))

        return (
            f"Using the latest {total} events, the most common event types are {top_three}. "
            f"Query received: '{query}'. Use event filters for exact slices (team, player, period, time range)."
        )

    def _fallback_event_explanation(self, event: Event, question: Optional[str]) -> str:
        parts = [
            f"Event {event.id} is labeled as '{event.event_type}' with confidence {event.confidence:.2f}.",
            f"Timestamp window: {event.start_ms}ms to {event.end_ms}ms (occurred_at={event.occurred_at_ms}ms).",
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

    def _try_llm_query(self, events: List[Event], query: str) -> Optional[str]:
        payload = self._prepare_llm_payload(events, query=query)
        return self._call_openai(payload)

    def _try_llm_explanation(self, event: Event, question: Optional[str]) -> Optional[str]:
        query = question or "Explain why this event was detected and what confidence caveats apply."
        payload = self._prepare_llm_payload([event], query=query)
        return self._call_openai(payload)

    def _prepare_llm_payload(self, events: List[Event], query: str) -> Tuple[str, str]:
        summary_rows = []
        for event in events[:100]:
            summary_rows.append(
                {
                    "event_id": event.id,
                    "event_type": event.event_type,
                    "confidence": event.confidence,
                    "occurred_at_ms": event.occurred_at_ms,
                    "team_id": event.team_id,
                    "player_id": event.player_id,
                }
            )

        system = (
            "You are an assistant for soccer video analysis. "
            "Do not fabricate unseen facts. Use only provided event context."
        )
        user = f"Query: {query}\n\nEvent context:\n{summary_rows}"
        return system, user

    def _call_openai(self, payload: Tuple[str, str]) -> Optional[str]:
        if self.provider != "openai" or not settings.openai_api_key:
            return None

        try:
            from openai import OpenAI
        except Exception:
            return None

        system, user = payload
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            text = (response.output_text or "").strip()
            return text or None
        except Exception:
            return None


agent_service = AgentService()
