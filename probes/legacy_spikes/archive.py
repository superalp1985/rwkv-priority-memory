"""Local, single-writer SQLite outbox/FTS experiment. Demo uses invented text only."""

import json
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory


class Archive:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY, u TEXT NOT NULL, s TEXT NOT NULL,
                k TEXT NOT NULL, body TEXT, UNIQUE(u,s,k));
            CREATE TABLE IF NOT EXISTS jobs(
                id INTEGER PRIMARY KEY REFERENCES events(id));
            CREATE VIRTUAL TABLE IF NOT EXISTS lexical USING fts5(body);
        ''')

    def close(self):
        self.db.close()

    def put(self, user, session, key, text, *, consent=False):
        if not consent:
            return False
        if not all(isinstance(s, str) and s for s in (user, session, key, text)):
            raise ValueError('non-empty identifiers and text required')
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT body FROM events WHERE u=? AND s=? AND k=?',
                                  (user, session, key)).fetchone()
            if row is not None:
                if row[0] != text:
                    raise ValueError('deleted event or idempotency key reused with different content')
                return False
            cur = self.db.execute('INSERT INTO events(u,s,k,body) VALUES(?,?,?,?)',
                                  (user, session, key, text))
            self.db.execute('INSERT INTO jobs(id) VALUES(?)', (cur.lastrowid,))
        return True

    def pending(self):
        return self.db.execute('SELECT count(*) FROM jobs').fetchone()[0]

    def index_one(self):
        # Worker transaction is run explicitly; no thread, vector store or network.
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT e.id,e.body FROM jobs j JOIN events e ON j.id=e.id '
                                  'WHERE e.body IS NOT NULL ORDER BY e.id LIMIT 1').fetchone()
            if row is None:
                return False
            self.db.execute('INSERT INTO lexical(rowid,body) VALUES(?,?)', row)
            self.db.execute('DELETE FROM jobs WHERE id=?', (row[0],))
        return True

    def forget(self, user, session, key):
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT id FROM events WHERE u=? AND s=? AND k=?',
                                  (user, session, key)).fetchone()
            if row is not None:
                self.db.execute('UPDATE events SET body=NULL WHERE id=?', row)
                self.db.execute('DELETE FROM lexical WHERE rowid=?', row)
                self.db.execute('DELETE FROM jobs WHERE id=?', row)

    def lookup(self, user, session, query):
        if not query.strip():
            return []
        phrase = '"' + query.replace('"', '""') + '"'
        sql = ('SELECT e.k FROM lexical JOIN events e ON e.id=lexical.rowid '
               'WHERE e.u=? AND e.s=? AND e.body IS NOT NULL AND lexical MATCH ? '
               'ORDER BY bm25(lexical),e.id LIMIT 3')
        return [row[0] for row in self.db.execute(sql, (user, session, phrase))]


def main():
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / 'demo.sqlite'
        a = Archive(path)
        a.put('demo', 'A', 'turn-1', 'The fictional label is COBALT', consent=True)
        before = a.lookup('demo', 'A', 'cobalt')
        a.close()
        a = Archive(path)
        try:
            pending = a.pending()
            a.index_one()
            after = a.lookup('demo', 'A', 'cobalt')
            foreign = a.lookup('other', 'A', 'cobalt')
            a.forget('demo', 'A', 'turn-1')
            print(json.dumps({'before_index': before, 'pending_after_reopen': pending,
                              'after_index': after, 'other_user': foreign,
                              'after_delete': a.lookup('demo', 'A', 'cobalt')}, indent=2))
        finally:
            a.close()


if __name__ == '__main__':
    main()
