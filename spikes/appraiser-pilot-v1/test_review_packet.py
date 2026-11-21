import csv
import tempfile
import unittest
from pathlib import Path

from review_packet import DIMS, RATING_COLUMNS, RATIONALE_COLUMNS, make_packet, compare, records


class PacketTest(unittest.TestCase):
    def test_known_evidence_from_new_stimuli(self):
        rows = {r['编号']: r for r in records()}
        self.assertEqual((len(rows), len({r['对话'] for r in rows.values()})), (12, 10))
        self.assertEqual(rows['K47']['独立用户提及数'], '1')
        self.assertEqual(rows['K47']['最后用户提及UTC'], '2026-06-10T09:00:00+00:00')
        self.assertEqual(rows['K47']['距最后提及秒'], '7200')
        self.assertEqual(rows['K06']['独立用户提及数'], '0')
        self.assertEqual(rows['K06']['距最后提及秒'], '无')
        self.assertEqual(rows['K71']['独立用户提及数'], '2')

    def test_blank_disagreement_and_abstentions_do_not_become_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'packet'
            a, b = make_packet(out)
            with self.assertRaises(FileExistsError):
                make_packet(out)
            def read(path):
                with path.open(encoding='utf-8-sig', newline='') as f:
                    return list(csv.DictReader(f))
            def put(path, rows):
                with path.open('w', encoding='utf-8-sig', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            ra, rb = read(a), read(b)
            self.assertEqual({r['编号'] for r in ra}, {r['编号'] for r in rb})
            self.assertNotEqual([r['编号'] for r in ra], [r['编号'] for r in rb])
            self.assertTrue(all(not r[column] for r in ra for column in RATING_COLUMNS.values()))
            self.assertTrue(all(not r[column] for r in rb for column in RATING_COLUMNS.values()))
            blank_result = compare(a, b)
            self.assertEqual(blank_result['complete_submissions'], 0)
            self.assertEqual(blank_result['complete_on_all_four_dimensions'], 0)
            self.assertEqual(blank_result['items_missing_rationale'], 12)
            self.assertEqual(blank_result['missing_rationale_by_dimension'], dict.fromkeys(DIMS, 12))
            self.assertEqual(blank_result['dimensions']['I']['unfilled_ratings'], 12)
            for row in ra + rb:
                row.update({column: '待评；测试空白' for column in RATIONALE_COLUMNS.values()})
            put(a, ra); put(b, rb)
            self.assertEqual(compare(a, b)['dimensions']['I']['unfilled_ratings'], 12)
            with self.assertRaises(ValueError):
                compare(a, a)
            for table, values in ((ra, ('0','X','3','X')), (rb, ('2','X','3','1'))):
                row = next(r for r in table if r['编号'] == 'K47')
                row.update(dict(zip(RATING_COLUMNS.values(), values)))
                row.update({column: '有逐项依据' for column in RATIONALE_COLUMNS.values()})
            put(a, ra); put(b, rb)
            result = compare(a, b)
            self.assertEqual(result['dimensions']['I']['gap_ge_2'], 1)
            self.assertEqual(result['dimensions']['E']['both_X'], 1)
            self.assertEqual(result['dimensions']['F']['exact_agreement'], 1)
            self.assertEqual(result['dimensions']['R']['one_X'], 1)
            self.assertEqual(result['complete_on_all_four_dimensions'], 1)
            self.assertEqual(result['complete_submissions'], 0)
            self.assertIn('K47', result['disagreements'])
            row = next(r for r in ra if r['编号'] == 'K47')
            row[RATING_COLUMNS['I']] = '0.5'; put(a, ra)
            with self.assertRaises(ValueError): compare(a, b)
            row[RATING_COLUMNS['I']] = '0'; row['事项版本'] += '改过'; put(a, ra)
            with self.assertRaises(ValueError): compare(a, b)


if __name__ == '__main__':
    unittest.main()