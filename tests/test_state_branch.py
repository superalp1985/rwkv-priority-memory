"""Opt-in pinned CPU integration. The default stdlib suite needs no torch."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


@unittest.skipUnless(os.environ.get('RWKV_BRANCH_TEST_SOURCE'), 'set RWKV_BRANCH_TEST_SOURCE to the pinned ChatRWKV model.py')
class StateBranchTest(unittest.TestCase):
    def test_carried_context_fork_and_next_turn_feedback(self):
        source = Path(os.environ['RWKV_BRANCH_TEST_SOURCE']).resolve()
        probe = subprocess.run(
            [sys.executable, '-m', 'probes.state_branch', '--source', str(source)],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(probe.returncode, 0, probe.stderr or probe.stdout)
        result = json.loads(probe.stdout)
        self.assertTrue(result['same_history_same_logits'])
        self.assertTrue(result['independent_tensor_storage'])
        self.assertTrue(result['live_matches_no_appraisal'])
        self.assertGreater(result['changed_history_same_suffix_max_logit_delta'], 0)
        self.assertNotEqual(*result['grade_logit_margin_history_a_b'])
        self.assertTrue(result['fixture_grade_flipped_selected_card'])
        self.assertGreater(result['next_probe_different_card_max_logit_delta'], 0)
        self.assertNotEqual(*result['next_grade_logit_margin_forced_low_high'])
        self.assertGreater(result['selected_card_vs_no_card_max_logit_delta'], 0)
        self.assertTrue(result['same_card_same_next_logits'])
        self.assertTrue(result['all_finite'])
