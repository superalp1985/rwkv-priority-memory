"""Deterministic probe for a persisted, four-factor memory priority policy."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from typing import Iterable


POLICY_VERSION = "priority-v2-model-appraised"


def _unit(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MemorySignals:
    memory_id: str
    text: str
    first_seen_at: datetime
    last_user_mention_at: datetime
    distinct_user_mentions: int
    explicit_user_importance: float | None
    model_assessed_importance: float
    model_assessed_emotion_intensity: float
    model_assessed_repetition: float
    model_assessed_recency: float
    model_confidence: float = 1.0
    token_count: int = 1

    def __post_init__(self) -> None:
        if not self.memory_id or not self.text:
            raise ValueError("memory_id and text must be non-empty")
        first = _utc(self.first_seen_at)
        last = _utc(self.last_user_mention_at)
        if last < first:
            raise ValueError("last_user_mention_at cannot precede first_seen_at")
        if isinstance(self.distinct_user_mentions, bool) or self.distinct_user_mentions < 1:
            raise ValueError("distinct_user_mentions must be a positive integer")
        if int(self.distinct_user_mentions) != self.distinct_user_mentions:
            raise ValueError("distinct_user_mentions must be an integer")
        if isinstance(self.token_count, bool) or int(self.token_count) != self.token_count or self.token_count < 1:
            raise ValueError("token_count must be a positive integer")
        for name in (
            "model_assessed_importance",
            "model_assessed_emotion_intensity",
            "model_assessed_repetition",
            "model_assessed_recency",
            "model_confidence",
        ):
            _unit(getattr(self, name), name)
        if self.explicit_user_importance is not None:
            _unit(self.explicit_user_importance, "explicit_user_importance")


@dataclass(frozen=True)
class PrioritySnapshot:
    memory_id: str
    priority_score: float
    importance: float
    explicit_user_importance: float | None
    model_assessed_importance: float
    model_confidence: float
    model_assessed_emotion_intensity: float
    model_assessed_repetition: float
    model_assessed_recency: float
    distinct_user_mentions: int
    last_user_mention_at: str
    scored_at: str
    policy_version: str = POLICY_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "priority_score": round(self.priority_score, 6),
            "importance": round(self.importance, 6),
            "explicit_user_importance": (
                round(self.explicit_user_importance, 6)
                if self.explicit_user_importance is not None
                else None
            ),
            "model_assessed_importance": round(self.model_assessed_importance, 6),
            "model_confidence": round(self.model_confidence, 6),
            "model_assessed_emotion_intensity": round(self.model_assessed_emotion_intensity, 6),
            "model_assessed_repetition": round(self.model_assessed_repetition, 6),
            "model_assessed_recency": round(self.model_assessed_recency, 6),
            "distinct_user_mentions": self.distinct_user_mentions,
            "last_user_mention_at": self.last_user_mention_at,
            "scored_at": self.scored_at,
            "policy_version": self.policy_version,
        }


def score_memory(memory: MemorySignals, now: datetime) -> PrioritySnapshot:
    """Average four model judgments; counts and timestamps remain audit evidence."""
    now_utc = _utc(now)
    last = _utc(memory.last_user_mention_at)
    if last > now_utc:
        raise ValueError("last_user_mention_at cannot be in the future")

    explicit = memory.explicit_user_importance
    model_importance = memory.model_assessed_importance
    importance = max(explicit if explicit is not None else 0.0, model_importance)
    emotion = memory.model_assessed_emotion_intensity
    repetition = memory.model_assessed_repetition
    recency = memory.model_assessed_recency
    score = 25.0 * (importance + emotion + repetition + recency)

    return PrioritySnapshot(
        memory_id=memory.memory_id,
        priority_score=score,
        importance=importance,
        explicit_user_importance=explicit,
        model_assessed_importance=model_importance,
        model_confidence=memory.model_confidence,
        model_assessed_emotion_intensity=emotion,
        model_assessed_repetition=repetition,
        model_assessed_recency=recency,
        distinct_user_mentions=memory.distinct_user_mentions,
        last_user_mention_at=last.isoformat(),
        scored_at=now_utc.isoformat(),
    )


def rank_memories(
    memories: Iterable[MemorySignals], now: datetime
) -> list[tuple[MemorySignals, PrioritySnapshot]]:
    scored = [(memory, score_memory(memory, now)) for memory in memories]
    scored.sort(key=lambda pair: (-pair[1].priority_score, pair[0].memory_id))
    return scored


def select_seed(
    ranked: Iterable[tuple[MemorySignals, PrioritySnapshot]], token_budget: int
) -> list[tuple[MemorySignals, PrioritySnapshot]]:
    """Pack a score-ranked prefix of cards with known lengths; no compression here."""
    if isinstance(token_budget, bool) or token_budget < 0:
        raise ValueError("token_budget must be a non-negative integer")
    if int(token_budget) != token_budget:
        raise ValueError("token_budget must be an integer")
    chosen: list[tuple[MemorySignals, PrioritySnapshot]] = []
    used = 0
    for memory, snapshot in ranked:
        if used + memory.token_count <= token_budget:
            chosen.append((memory, snapshot))
            used += memory.token_count
        else:
            break
    return chosen


def demo() -> list[dict[str, object]]:
    now = datetime(2026, 11, 18, 8, tzinfo=timezone.utc)
    samples = [
        MemorySignals(
            memory_id="plan",
            text="The user explicitly asked to retain project K47.",
            first_seen_at=now,
            last_user_mention_at=now,
            distinct_user_mentions=3,
            explicit_user_importance=1.0,
            model_assessed_importance=0.8,
            model_assessed_emotion_intensity=0.2,
            model_assessed_repetition=0.4,
            model_assessed_recency=0.9,
            token_count=12,
        ),
        MemorySignals(
            memory_id="dog",
            text="The user's dog's name is Pip.",
            first_seen_at=now,
            last_user_mention_at=now,
            distinct_user_mentions=1,
            explicit_user_importance=0.0,
            model_assessed_importance=0.7,
            model_assessed_emotion_intensity=0.8,
            model_assessed_repetition=0.1,
            model_assessed_recency=0.9,
            token_count=8,
        ),
        MemorySignals(
            memory_id="old-detail",
            text="A low-stakes detail from months ago.",
            first_seen_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            last_user_mention_at=datetime(2026, 5, 1, tzinfo=timezone.utc),
            distinct_user_mentions=1,
            explicit_user_importance=0.0,
            model_assessed_importance=0.2,
            model_assessed_emotion_intensity=0.0,
            model_assessed_repetition=0.0,
            model_assessed_recency=0.1,
            token_count=10,
        ),
    ]
    ranked = rank_memories(samples, now)
    seed = select_seed(ranked, token_budget=20)
    return [
        {**snapshot.as_dict(), "selected_for_seed": memory.memory_id in {m.memory_id for m, _ in seed}}
        for memory, snapshot in ranked
    ]


if __name__ == "__main__":
    print(json.dumps({"scope": "synthetic deterministic score probe; not an LLM or RWKV result", "memories": demo()}, indent=2))
