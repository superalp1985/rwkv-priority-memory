"""Offline score journal. One file per consent scope; caller supplies item/version IDs.

Not a dialogue service, LLM, semantic deduplicator, or deletion implementation.
The small interface accepts a normalized appraisal fixture; no network calls.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from .priority import MemorySignals, _unit, _utc, score_memory

REASONS = {'commitment', 'constraint', 'preference', 'personal_event', 'incidental', 'ambiguous'}


class ScoreJournal:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS turns(
                id TEXT PRIMARY KEY NOT NULL, role TEXT NOT NULL, body TEXT NOT NULL, at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS mentions(
                item TEXT NOT NULL, turn TEXT NOT NULL REFERENCES turns(id), floor REAL, rating TEXT,
                PRIMARY KEY(item,turn));
            CREATE TABLE IF NOT EXISTS snapshots(
                seq INTEGER PRIMARY KEY, item TEXT NOT NULL, at TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS by_item ON snapshots(item,seq);
        ''')

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()

    def record_turn(self, turn, role, body, at, *, consent=False, origin='dialogue'):
        if consent is not True or origin != 'dialogue':
            return False
        if role not in ('user', 'assistant') or not all(isinstance(s, str) and s.strip() for s in (turn, body)):
            raise ValueError('a real dialogue turn, nonempty id and text are required')
        row = (role, body, _utc(at).isoformat())
        with self.db:
            prev = self.db.execute('SELECT role,body,at FROM turns WHERE id=?', (turn,)).fetchone()
            if prev is not None:
                if prev != row:
                    raise ValueError('turn id already used with different content or time')
                return False
            self.db.execute('INSERT INTO turns VALUES(?,?,?,?)', (turn, *row))
        return True

    def transcript(self):
        return self.db.execute('SELECT id,body FROM turns ORDER BY at,rowid').fetchall()

    @staticmethod
    def _rating(report, turn, body):
        model = report.get('model')
        prompt = report.get('prompt')
        quote = report.get('quote')
        reason_code = report.get('reason_code')
        if not all(isinstance(value, str) and value.strip()
                   for value in (model, prompt, quote, reason_code)):
            raise ValueError('missing appraisal provenance field')
        if quote not in body or reason_code not in REASONS:
            raise ValueError('unanchored quote or unknown reason code')
        norm = {k: report[k] for k in ('model', 'prompt', 'quote', 'reason_code')}
        for k in ('importance', 'emotion_intensity', 'repetition', 'recency', 'confidence'):
            value = report.get(k)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f'invalid appraisal number: {k}')
            norm[k] = _unit(value, k)
        for k in ('importance', 'emotion_intensity', 'repetition', 'recency'):
            if norm[k] * 4 != int(norm[k] * 4):
                raise ValueError(f'{k} must be a normalized 0..4 grade')
        return json.dumps(dict(norm, source_turn=turn), sort_keys=True, allow_nan=False)

    def note_mention(self, item, turn, appraisal=None, *, explicit_floor=None, as_of=None):
        """Caller links one genuine mention to one fact version. Replay is idempotent.

        A pending mention can later acquire a rating without becoming a second mention.
        Only the trusted caller, not the appraisal output, can supply an explicit user floor.
        """
        if not isinstance(item, str) or not item.strip():
            raise ValueError('empty item/version identifier')
        row = self.db.execute('SELECT role,body,at FROM turns WHERE id=?', (turn,)).fetchone()
        if not row or row[0] != 'user':
            return False
        rating = None if appraisal is None else self._rating(appraisal, turn, row[1])
        floor = None if explicit_floor is None else _unit(explicit_floor, 'explicit floor')
        with self.db:
            prev = self.db.execute('SELECT floor,rating FROM mentions WHERE item=? AND turn=?', (item, turn)).fetchone()
            merged = (floor if floor is not None else prev[0] if prev else None,
                      rating if rating is not None else prev[1] if prev else None)
            if prev == merged:
                return False
            self.db.execute('INSERT INTO mentions VALUES(?,?,?,?) ON CONFLICT(item,turn) DO UPDATE SET '
                            'floor=excluded.floor,rating=excluded.rating', (item, turn, *merged))
            clock = _utc(as_of).isoformat() if as_of else max(row[2], self._last_clock(item) or row[2], self._bounds(item)[1])
            self._snapshot(item, clock, 'mention' if prev is None else 'appraisal_update', turn)
        return True

    def _last_clock(self, item):
        return self.db.execute('SELECT max(at) FROM snapshots WHERE item=?', (item,)).fetchone()[0]

    def _bounds(self, item):
        return self.db.execute('SELECT min(t.at),max(t.at),count(*),max(m.floor) '
                               'FROM mentions m JOIN turns t ON t.id=m.turn WHERE m.item=?', (item,)).fetchone()

    def _snapshot(self, item, clock, trigger, cause=None):
        first, last, n, floor = self._bounds(item)
        if not n:
            return None
        if clock < last or clock < (self._last_clock(item) or clock):
            raise ValueError('checkpoint cannot precede known mentions or the last checkpoint')
        row = self.db.execute('SELECT m.rating FROM mentions m JOIN turns t ON t.id=m.turn '
                              'WHERE m.item=? AND m.rating IS NOT NULL ORDER BY t.at DESC,t.rowid DESC LIMIT 1',
                              (item,)).fetchone()
        if not row:
            return None   # Pending is distinct from importance=0.
        r = json.loads(row[0])
        signals = MemorySignals(
            item, item, datetime.fromisoformat(first), datetime.fromisoformat(last), n,
            floor, r['importance'], r['emotion_intensity'], r['repetition'], r['recency'], r['confidence'])
        snap = score_memory(signals, datetime.fromisoformat(clock)).as_dict()
        snap.update(assessment=r, trigger=trigger, trigger_turn=cause, first_seen_at=first, last_user_mention_at=last)
        self.db.execute('INSERT INTO snapshots(item,at,payload) VALUES(?,?,?)',
                        (item, clock, json.dumps(snap, sort_keys=True, allow_nan=False)))
        return snap

    def checkpoint(self, item, now):
        with self.db:
            return self._snapshot(item, _utc(now).isoformat(), 'clock')

    def history(self, item):
        return [json.loads(r[0]) for r in self.db.execute('SELECT payload FROM snapshots WHERE item=? ORDER BY seq', (item,))]


def demo():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = start + timedelta(days=30)
    rating = dict(model='synthetic-fixture', prompt='appraisal-v2', quote='nine', reason_code='commitment',
                  importance=.75, emotion_intensity=.25, repetition=.25, recency=1.0, confidence=.8)
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / 'demo.sqlite'
        with ScoreJournal(p) as j:
            j.record_turn('u1', 'user', 'Weekly handoff at nine.', start, consent=True)
            j.note_mention('weekly-handoff:v1', 'u1', rating)
            before = j.history('weekly-handoff:v1')
        with ScoreJournal(p) as j:
            reopened = before == j.history('weekly-handoff:v1')
            j.record_turn('a', 'assistant', 'Weekly handoff at nine.', later, consent=True)
            reinforced = j.note_mention('weekly-handoff:v1', 'a')
            j.checkpoint('weekly-handoff:v1', later)
            j.record_turn('u2', 'user', 'Still the usual handoff time.', later, consent=True)
            j.note_mention('weekly-handoff:v1', 'u2', dict(rating, quote='usual handoff', repetition=.5))
            return dict(scope='invented dialogue, manually supplied rating; no model', reopened_equal=reopened,
                        assistant_counted=reinforced, archived_turns=len(j.transcript()), history=j.history('weekly-handoff:v1'))


if __name__ == '__main__':
    print(json.dumps(demo(), ensure_ascii=False, indent=2))
