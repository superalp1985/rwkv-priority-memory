import unittest

import numpy as np

from pilot_eval import admission_metrics, contrast_uses_context, emotion_metrics, pair_result


class PilotMetricsTest(unittest.TestCase):
    def test_admission_ties_are_invariant_and_not_counted_as_separation(self):
        self.assertEqual(admission_metrics([0, 1, 1, 0], [.5] * 4),
                         dict(n=4, positives=2, prevalence=.5, average_precision=.5, auroc=.5))
        self.assertEqual(admission_metrics([1, 0], [.9, .1])["average_precision"], 1.0)
        self.assertEqual(admission_metrics([1, 0], [.1, .9])["auroc"], 0.0)

    def test_emotion_metrics_separate_positive_error_and_undefined_correlation(self):
        labels = np.tile([0, 1, 3], (6, 1)).T
        scores = labels.astype(float)
        scores[:, 0] = 0
        result = emotion_metrics(labels, scores)
        self.assertIsNone(result["per_emotion"]["anger"]["pearson"])
        self.assertAlmostEqual(result["macro_pearson"], 5 / 6)
        self.assertEqual(result["per_emotion"]["anger"]["positive_mae"], 2.0)

    def test_pair_margin_is_required(self):
        self.assertFalse(pair_result("same", .71, .71)["pass"])
        self.assertFalse(pair_result("too-close", .73, .71)["pass"])
        self.assertTrue(pair_result("separated", .81, .71)["pass"])

    def test_context_contrast_requires_distinct_context_effects(self):
        self.assertFalse(contrast_uses_context([.7], [.7], [.1], [.1]))
        self.assertTrue(contrast_uses_context([.7], [.3], [.1], [.1]))
        with self.assertRaises(ValueError):
            contrast_uses_context([np.nan], [.3], [.1], [.1])


if __name__ == "__main__":
    unittest.main()