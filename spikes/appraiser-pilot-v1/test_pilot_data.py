import csv
import io
import json
import tarfile
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pilot_data import EMOTIONS, load_brighter, load_memora, mention_evidence, temporal_contract_probe


class PilotDataTest(unittest.TestCase):
    def test_memora_excludes_delete_and_uses_past_context_grouped_by_persona(self):
        groups = {"train": ("first",), "dev": ("second",), "test": ("third",)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.tar.gz"
            with tarfile.open(path, "w:gz") as archive:
                for split, people in groups.items():
                    persona, = people
                    horizon = "quarterly" if split == "train" else "weekly"
                    for session, operation in enumerate(("add", "delete"), 1):
                        dialogues = [
                            dict(turn=1, speaker="user_agent", message="Shared boilerplate", share_memory=False),
                            dict(turn=2, speaker="ai_agent", message="Which item?", share_memory=False),
                            dict(turn=3, speaker="user_agent", message=f"Item from {persona} session {session}", share_memory=True),
                            dict(turn=4, speaker="ai_agent", message="FUTURE", share_memory=False),
                            dict(turn=5, speaker="user_agent", message=f"Thanks, from {persona}", share_memory=False),
                        ]
                        raw = json.dumps(dict(persona=persona, session_id=session, operation=operation, conversation=dialogues)).encode()
                        member = tarfile.TarInfo(f"root/data/{horizon}/{persona}/conversations/session_{session:04d}.json")
                        member.size = len(raw)
                        archive.addfile(member, io.BytesIO(raw))
            rows, audit = load_memora(path, groups=groups)
        for split, records in rows.items():
            self.assertEqual(sorted(row.labels for row in records), [(0,), (1,)])
            positive = next(row for row in records if row.labels == (1,))
            self.assertEqual(positive.context, (("user", "Shared boilerplate"), ("assistant", "Which item?")))
            self.assertNotIn("FUTURE", repr(positive))
            self.assertEqual(audit[split]["cross_split_targets_removed"], 1)
            self.assertEqual(audit[split]["delete_sessions_excluded"], 1)

    def test_brighter_duplicate_text_is_removed_from_all_splits_without_repartition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            for split in ("train", "dev", "test"):
                with (path / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["id", "text", *EMOTIONS])
                    writer.writerow([split + "1", "Same text", 0, 0, 0, 0, 0, 0])
                    writer.writerow([split + "2", "Unique " + split, 0, 0, 0, 3, 0, 0])
            rows, audit = load_brighter(path, expected={s: 2 for s in ("train", "dev", "test")})
        self.assertEqual([x[0].labels for x in rows.values()], [(0, 0, 0, 3, 0, 0)] * 3)
        self.assertTrue(all(x["cross_split_targets_removed"] == 1 for x in audit.values()))

    def test_repetition_and_elapsed_time_use_only_real_user_mentions(self):
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        now = start + timedelta(days=7)
        row = dict(turn="u1", role="user", origin="dialogue", item="v1", at=start)
        extras = [dict(row, turn="a", role="assistant", at=now),
                  dict(row, turn="r", origin="retrieval", at=now),
                  dict(row, turn="u2", item="v2", at=now)]
        self.assertEqual(mention_evidence([row, dict(row), *extras], "v1", now), (1, start, 604800.0))
        self.assertEqual(mention_evidence([row, dict(row, turn="u3", at=now)], "v1", now), (2, now, 0.0))
        self.assertIsNone(mention_evidence(extras, "v1", now))
        with self.assertRaises(ValueError):
            mention_evidence([dict(row, at=now + timedelta(seconds=1))], "v1", now)

    def test_temporal_contract_probe_excludes_replay_and_advances_utc_age(self):
        self.assertEqual(temporal_contract_probe(), dict(replay_count=1,
                         replay_last_mention_seconds=259200.0,
                         replay_does_not_reinforce=True,
                         time_advance_does_not_refresh=True,
                         new_user_turn_increments=True))


if __name__ == "__main__":
    unittest.main()