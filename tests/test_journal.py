import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.journal import ScoreJournal


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
LATER = START + timedelta(days=30)
RATING = dict(importance=.75, emotion_intensity=.25, repetition=.25, recency=1.0,
              confidence=.8, reason_code='commitment', model='fixture-not-a-model',
              prompt='appraisal-v1', quote='nine')


class JournalContract(unittest.TestCase):
    def test_reopen_decay_remention_and_score_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'score.sqlite'
            with ScoreJournal(p) as j:
                j.record_turn('u1', 'user', 'Weekly handoff at nine.', START, consent=True)
                self.assertTrue(j.note_mention('handoff', 'u1', RATING))
                original, = j.history('handoff')
            with ScoreJournal(p) as j:
                self.assertEqual(j.history('handoff'), [original])
                self.assertEqual(original['assessment']['source_turn'], 'u1')
                self.assertEqual(original['assessment']['quote'], 'nine')
                idle = j.checkpoint('handoff', LATER)
                assert idle is not None
                self.assertEqual(idle['model_assessed_recency'], 1.0)
                self.assertEqual(idle['priority_score'], original['priority_score'])
                j.record_turn('u2', 'user', 'Same handoff time this week.', LATER, consent=True)
                self.assertTrue(j.note_mention('handoff', 'u2',
                                               dict(RATING, quote='handoff time', repetition=.5, recency=1.0)))
                self.assertFalse(j.note_mention('handoff', 'u2'))
                renewed = j.history('handoff')[-1]
                self.assertEqual(renewed['distinct_user_mentions'], 2)
                self.assertEqual(renewed['model_assessed_recency'], 1.0)
                self.assertEqual(renewed['assessment']['source_turn'], 'u2')
                self.assertGreater(renewed['priority_score'], original['priority_score'])
                self.assertEqual(len(j.history('handoff')), 3)

    def test_replay_never_reinforces_but_real_assistant_turn_is_archived(self):
        with ScoreJournal(':memory:') as j:
            j.record_turn('u', 'user', 'At nine.', START, consent=True)
            j.note_mention('time', 'u', RATING)
            j.record_turn('a', 'assistant', 'At nine.', LATER, consent=True)
            self.assertFalse(j.note_mention('time', 'a'))
            self.assertFalse(j.record_turn('r', 'user', 'At nine.', LATER, consent=True, origin='retrieval'))
            self.assertFalse(j.record_turn('x', 'user', 'private', LATER))
            self.assertEqual(j.transcript(), [('u', 'At nine.'), ('a', 'At nine.')])
            idle = j.checkpoint('time', LATER)
            assert idle is not None
            self.assertEqual(idle['distinct_user_mentions'], 1)
            self.assertEqual(j.history('time')[-1]['last_user_mention_at'], START.isoformat())

    def test_pending_and_rejected_appraisal_do_not_lose_raw_text(self):
        with ScoreJournal(':memory:') as j:
            j.record_turn('p', 'user', 'Just an example.', START, consent=True)
            with self.assertRaises(ValueError):
                j.note_mention('example', 'p', RATING)  # cited quote not in this message
            self.assertTrue(j.note_mention('example', 'p'))
            self.assertIsNone(j.checkpoint('example', LATER))
            self.assertEqual(j.history('example'), [])
            self.assertEqual(j.transcript(), [('p', 'Just an example.')])
            self.assertTrue(j.note_mention('example', 'p', dict(RATING, quote='example')))
            self.assertEqual(j.history('example')[-1]['distinct_user_mentions'], 1)

    def test_new_context_can_lower_importance_without_erasing_previous_judgment(self):
        with ScoreJournal(':memory:') as j:
            j.record_turn('u1', 'user', 'Weekly handoff at nine.', START, consent=True)
            j.note_mention('handoff', 'u1', RATING)
            j.record_turn('u2', 'user', 'The handoff is now optional.', LATER, consent=True)
            j.note_mention('handoff', 'u2', dict(RATING, importance=.25, quote='optional'))
            old, new = j.history('handoff')
            self.assertEqual((old['importance'], new['importance']), (.75, .25))
            self.assertEqual(new['assessment']['source_turn'], 'u2')
            self.assertLess(new['priority_score'], old['priority_score'])