import math
import unittest

from probes.legacy_spikes.decay import next_trace, retention, kernel_log_decay


class DecayContract(unittest.TestCase):
    def test_off_and_no_trace_are_exact_identity(self):
        for w in (0.55, 0.81, 0.999999, 1.0):
            self.assertEqual(retention(w, 1.0, strength=0.0), w)
            self.assertEqual(retention(w, 0.0), w)
            self.assertEqual(retention(w, 1.0, mask=0.0), w)

    def test_protection_is_bounded_and_monotonic(self):
        vals = [retention(0.8, x) for x in (0.0, 0.25, 0.5, 1.0)]
        self.assertEqual(vals, sorted(set(vals)))
        self.assertLess(vals[-1], 1.0)
        self.assertGreater(vals[1], 0.8)

    def test_new_fact_affects_only_later_steps_and_halves(self):
        prev = 0.0
        self.assertEqual(retention(0.8, prev), 0.8)
        p = next_trace(prev, 1.0, half_life=64)
        self.assertGreater(retention(0.8, p), 0.8)
        for _ in range(64):
            p = next_trace(p, 0.0, half_life=64)
        self.assertAlmostEqual(p, 0.5)

    def test_packed_kernel_argument_matches_effective_decay(self):
        for z in (-0.5, -2.0, -12.0):
            w = math.exp(-math.exp(z))
            for p in (0.0, 0.25, 1.0):
                packed = kernel_log_decay(z, p, strength=0.1, mask=0.5)
                self.assertAlmostEqual(math.exp(-math.exp(packed)),
                                       retention(w, p, strength=0.1, mask=0.5), places=14)

    def test_one_shot_decay_at_empty_state_cannot_deepen_a_write(self):
        for power in (0.0, 0.1):
            self.assertEqual(0.0 * retention(0.8, 1.0, strength=power) + 1.0, 1.0)
        self.assertGreater(1.0 * retention(0.8, 1.0), 1.0 * 0.8)

    def test_rejects_invalid_policy_inputs(self):
        for args in ((0.0, 0.5), (0.8, float('nan')), (0.8, 2.0), (1.1, 0.5)):
            with self.assertRaises(ValueError):
                retention(*args)
        with self.assertRaises(ValueError):
            next_trace(0.0, 0.8, half_life=0)
        with self.assertRaises(ValueError):
            kernel_log_decay(float('nan'), 0.4)


if __name__ == '__main__':
    unittest.main()
