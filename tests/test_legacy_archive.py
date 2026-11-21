import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from probes.legacy_spikes.archive import Archive


class ArchiveContract(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'sample.sqlite'
        self.store = Archive(self.path)
        self.addCleanup(lambda: self.store.close())

    def put(self, user='u', session='s', key='1', text='The label is COBALT'):
        return self.store.put(user, session, key, text, consent=True)

    def test_pending_survives_reopen_and_replay_is_idempotent(self):
        self.put()
        self.assertEqual(self.store.lookup('u', 's', 'COBALT'), [])
        self.store.close()
        self.store = Archive(self.path)
        self.assertEqual(self.store.pending(), 1)
        self.assertFalse(self.put())
        self.assertTrue(self.store.index_one())
        self.assertFalse(self.store.index_one())
        self.assertEqual(self.store.lookup('u', 's', 'COBALT'), ['1'])
        with self.assertRaises(ValueError):
            self.put(text='different payload under the same key')

    def test_consent_and_scope(self):
        self.assertFalse(self.store.put('u', 's', 'denied', 'COBALT'))
        self.put(user='other', key='foreign')
        self.put(session='other', key='another-session')
        self.put()
        while self.store.index_one():
            pass
        self.assertEqual(self.store.lookup('u', 's', 'COBALT'), ['1'])

    def test_deletion_before_and_after_index_prevents_replay(self):
        self.put(key='A'); self.put(key='B')
        self.store.index_one()
        self.store.forget('u', 's', 'A'); self.store.forget('u', 's', 'B')
        self.assertEqual(self.store.pending(), 0)
        self.assertEqual(self.store.lookup('u', 's', 'COBALT'), [])
        with self.assertRaises(ValueError):
            self.put(key='B')

    def test_archive_and_job_rollback_together_on_write_error(self):
        self.store.db.execute("CREATE TRIGGER reject BEFORE INSERT ON jobs BEGIN SELECT RAISE(ABORT,'failpoint'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            self.put()
        self.assertEqual(self.store.db.execute('SELECT count(*) FROM events').fetchone()[0], 0)
        self.assertEqual(self.store.pending(), 0)


if __name__ == '__main__':
    unittest.main()
