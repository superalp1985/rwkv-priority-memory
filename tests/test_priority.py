import unittest
from datetime import datetime, timedelta, timezone

from src.priority import MemorySignals, POLICY_VERSION, rank_memories, score_memory, select_seed


NOW = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)


def memory(**overrides):
    values = {
        "memory_id": "m",
        "text": "A synthetic memory",
        "first_seen_at": NOW - timedelta(days=90),
        "last_user_mention_at": NOW - timedelta(days=10),
        "distinct_user_mentions": 2,
        "explicit_user_importance": None,
        "model_assessed_importance": 0.3,
        "model_assessed_emotion_intensity": 0.2,
        "model_assessed_repetition": 0.25,
        "model_assessed_recency": 0.5,
        "token_count": 1,
    }
    values.update(overrides)
    return MemorySignals(**values)


class PriorityPolicyTests(unittest.TestCase):
    def test_priority_is_only_the_mean_of_four_model_appraisals(self):
        signals = memory()
        snapshot = score_memory(signals, NOW)
        self.assertEqual(POLICY_VERSION, "priority-v2-model-appraised")
        self.assertEqual(snapshot.priority_score, 31.25)
        self.assertEqual(snapshot.model_assessed_repetition, 0.25)
        self.assertEqual(snapshot.model_assessed_recency, 0.5)

    def test_every_model_appraised_dimension_is_monotonic(self):
        fields = (
            "model_assessed_importance",
            "model_assessed_emotion_intensity",
            "model_assessed_repetition",
            "model_assessed_recency",
        )
        for field in fields:
            with self.subTest(field=field):
                low = score_memory(memory(**{field: 0.2}), NOW)
                high = score_memory(memory(**{field: 0.8}), NOW)
                self.assertGreater(high.priority_score, low.priority_score)

    def test_count_and_timestamps_are_evidence_not_implicit_score_inputs(self):
        baseline = score_memory(memory(), NOW)
        changed_metadata = score_memory(memory(
            distinct_user_mentions=80,
            first_seen_at=NOW - timedelta(days=900),
            last_user_mention_at=NOW,
        ), NOW)
        self.assertEqual(changed_metadata.priority_score, baseline.priority_score)
        self.assertEqual(changed_metadata.distinct_user_mentions, 80)
        self.assertEqual(changed_metadata.last_user_mention_at, NOW.isoformat())

    def test_each_model_dimension_can_decide_the_single_available_slot(self):
        fields = (
            "model_assessed_importance",
            "model_assessed_emotion_intensity",
            "model_assessed_repetition",
            "model_assessed_recency",
        )
        for field in fields:
            with self.subTest(field=field):
                a = memory(memory_id="A", **{field: 0.25})
                b = memory(memory_id="B", **{field: 0.75})
                selected = select_seed(rank_memories([a, b], NOW), token_budget=1)
                self.assertEqual([m.memory_id for m, _ in selected], ["B"])

    def test_model_recency_is_not_recomputed_from_wall_clock(self):
        old = score_memory(memory(
            first_seen_at=NOW - timedelta(days=400),
            model_assessed_recency=0.9,
            last_user_mention_at=NOW - timedelta(days=365),
        ), NOW)
        recent = score_memory(memory(
            model_assessed_recency=0.1,
            last_user_mention_at=NOW,
        ), NOW)
        self.assertGreater(old.priority_score, recent.priority_score)

    def test_snapshot_records_model_dimensions_and_audit_metadata(self):
        snapshot = score_memory(memory(), NOW)
        data = snapshot.as_dict()
        self.assertEqual(data["policy_version"], POLICY_VERSION)
        self.assertEqual(data["model_assessed_emotion_intensity"], 0.2)
        self.assertEqual(data["model_assessed_repetition"], 0.25)
        self.assertEqual(data["model_assessed_recency"], 0.5)
        self.assertEqual(data["distinct_user_mentions"], 2)
        self.assertEqual(data["last_user_mention_at"], (NOW - timedelta(days=10)).isoformat())

    def test_rank_and_seed_respect_model_score_and_token_budget(self):
        high = memory(memory_id="high", model_assessed_importance=0.9, token_count=2)
        mid = memory(memory_id="mid", model_assessed_importance=0.5, token_count=2)
        low = memory(memory_id="low", model_assessed_importance=0.1, token_count=1)
        ranked = rank_memories([low, mid, high], NOW)
        self.assertEqual([m.memory_id for m, _ in ranked], ["high", "mid", "low"])
        selected = select_seed(ranked, token_budget=2)
        self.assertEqual([m.memory_id for m, _ in selected], ["high"])
        self.assertEqual(len(ranked), 3)

    def test_invalid_input_is_rejected(self):
        for field, value in (
            ("model_assessed_emotion_intensity", float("nan")),
            ("model_assessed_recency", 1.1),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                memory(**{field: value})
        with self.assertRaises(ValueError):
            score_memory(memory(), NOW - timedelta(days=20))
        with self.assertRaises(ValueError):
            select_seed([], -1)


if __name__ == "__main__":
    unittest.main()
