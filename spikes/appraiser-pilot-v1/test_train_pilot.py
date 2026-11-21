import unittest
import json
from pathlib import Path

from pilot_data import Example
from train_pilot import encode, pair_examples


class TinyTokenizer:
    bos_token_id = 0
    eos_token_id = 2
    pad_token_id = 1

    def __call__(self, texts, **_kwargs):
        table = {
            "same words": [11, 12],
            "User: real agreement": [21, 22],
            "Assistant: fictional line": [31, 32],
            "long target": list(range(1, 201)),
        }
        return {"input_ids": [table.get(text, [3 + ord(char) for char in text]) for text in texts]}


class PilotEncodingTest(unittest.TestCase):
    def test_context_is_preserved_for_both_source_tasks_and_identical_targets(self):
        tokenizer = TinyTokenizer()
        first = Example("a", "same words", (("user", "real agreement"),), (0,))
        second = Example("b", "same words", (("assistant", "fictional line"),), (1,))
        for task in ("importance", "emotion"):
            with self.subTest(task=task):
                records, clipping = encode([first, second], tokenizer, task)
                self.assertEqual(records[0][0], [0, 21, 22, 2, 2, 11, 12, 2])
                self.assertEqual(records[1][0], [0, 31, 32, 2, 2, 11, 12, 2])
                self.assertEqual(records[0][0][-4:], records[1][0][-4:])
                self.assertEqual(clipping, {"target": 0, "context": 0})

    def test_target_only_ablation_keeps_target_and_drops_only_context(self):
        row = Example("a", "same words", (("user", "real agreement"),), (0,))
        records, clipping = encode([row], TinyTokenizer(), "emotion", with_context=False)
        self.assertEqual(records[0][0], [0, 11, 12, 2])
        self.assertEqual(clipping, {"target": 0, "context": 0})

    def test_emotion_context_and_target_are_both_bounded_without_overflow(self):
        row = Example("a", "long target", (("user", "real agreement"),), (0, 0, 0, 0, 0, 0))
        records, clipping = encode([row], TinyTokenizer(), "emotion")
        self.assertLessEqual(len(records[0][0]), 192)
        self.assertEqual(clipping, {"target": 1, "context": 0})

    def test_authored_context_pairs_reach_the_same_encoder(self):
        pairs = json.loads((Path(__file__).parent / "pairs.json").read_text(encoding="utf-8"))
        importance = pair_examples("importance", pairs["importance"])
        emotion = pair_examples("emotion", pairs["emotion"])
        self.assertTrue(importance[0].context)
        self.assertEqual(importance[0].target, importance[1].target)
        self.assertNotEqual(importance[0].context, importance[1].context)
        self.assertTrue(emotion[0].context)
        self.assertEqual(emotion[0].target, emotion[1].target)
        self.assertNotEqual(emotion[0].context, emotion[1].context)
        for task, rows in (("importance", importance), ("emotion", emotion)):
            encoded, _ = encode(rows[:2], TinyTokenizer(), task)
            self.assertNotEqual(encoded[0][0], encoded[1][0])

if __name__ == "__main__":
    unittest.main()