import json
from pathlib import Path
import random
import unittest
from typing import Any

from probes.human_reply_packet import ARMS, build


ROOT = Path(__file__).resolve().parents[1]


class HumanReplyPacketTest(unittest.TestCase):
    report: dict[str, Any]

    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'reports/capacity-holdout-v1-cpu.json').read_bytes())

    def test_packet_blinds_all_five_arms_and_expected_labels(self):
        packet, key = build(self.report, random.Random(17))
        self.assertEqual(len(key['panels']), 27)
        self.assertEqual(sum(len(row['group_to_arm']) for row in key['panels']), 135)
        for row in key['panels']:
            self.assertEqual(set(row['group_to_arm']), set('ABCDE'))
            self.assertEqual(set(row['group_to_arm'].values()), set(ARMS))
        for forbidden in (*ARMS, 'expected_marker', 'obsolete_marker', 'archive_evidence', 'h01', 'h08'):
            self.assertNotIn(forbidden, packet)

    def test_each_panel_gets_an_independent_group_mapping(self):
        _, key = build(self.report, random.Random(31))
        mappings = [tuple(row['group_to_arm'][letter] for letter in 'ABCDE') for row in key['panels']]
        self.assertGreater(len(set(mappings)), 1)
        self.assertEqual(len({row['panel'] for row in key['panels']}), 27)

    def test_panel_order_and_history_are_preserved_without_future_replies(self):
        packet, key = build(self.report, random.Random(7))
        self.assertIn('不是多轮互动录像', packet)
        self.assertIn('Please look up my original message', packet)
        self.assertEqual(len(key['panels']), packet.count('### P'))
        self.assertEqual(sum(len(row['archive_evidence']) for row in key['panels']), 8)


if __name__ == '__main__':
    unittest.main()