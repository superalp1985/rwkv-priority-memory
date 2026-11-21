from datetime import timedelta
from pathlib import Path
import json
import unittest

from probes.state_replay import DIMENSIONS, START, evaluate_case, summarize


def appraisal(importance=1, emotion_intensity=0, repetition=0, recency=4):
    return dict(importance=importance, emotion_intensity=emotion_intensity,
                repetition=repetition, recency=recency)


class MutableRuntime:
    def __init__(self, grade):
        self.grade = grade
        self.assessments = []
        self.forward_calls = 0
        self.input_tokens = 0
        self.appraisal_calls = 0

    @staticmethod
    def encode(text):
        return [text]

    def forward(self, tokens, state):
        self.forward_calls += 1
        self.input_tokens += len(tokens)
        state = [] if state is None else state
        state.extend(tokens)
        return tuple(state), state

    def appraise(self, state, card, mentions, as_of):
        self.appraisal_calls += 1
        self.assessments.append((tuple(state), card, tuple(mentions), as_of))
        result = self.grade(state, card, mentions, as_of)
        state.append("disposable rubric")
        return result

    @staticmethod
    def equal(a, b):
        return a == b


def turn(user, mentions=None, **extra):
    return {"user": user, "assistant": "Noted.", "mentions": mentions or {}, **extra}


class StateReplayTest(unittest.TestCase):
    def test_runtime_work_counts_forwarded_tokens_and_appraisal_calls(self):
        engine = MutableRuntime(lambda *_: appraisal())
        case = {"id": "work-count", "cards": {"a": "Card A"}, "turns": [
            turn("A matters.", {"a": "A matters."}),
        ]}
        result = evaluate_case(engine, case, budget=1)
        self.assertEqual(result["runtime_work"], {
            "forward_calls": 7,
            "input_tokens": 7,
            "appraisal_calls": 3,
        })

    def test_replacement_and_feedback_use_prior_cards_but_read_only_does_not(self):
        def grade(state, card, _, __):
            carries_a = any("[Working memory card]\nCard A" in segment for segment in state)
            return appraisal(importance=3 if card == "Card A" else (4 if carries_a else 1))

        engine = MutableRuntime(grade)
        case = {"id": "state-switch", "cards": {"a": "Card A", "b": "Card B"}, "turns": [
            turn("A and B", {"a": "A", "b": "B"}, preferred=["a"]),
            turn("Next turn.", preferred=["b"]),
        ]}
        result = evaluate_case(engine, case, budget=1)
        first, second = result["rounds"]
        self.assertEqual(first["read_only"]["selected"], ["a"])
        self.assertEqual(first["replacement"]["selected"], ["a"])
        self.assertEqual(first["feedback"]["selected"], ["a"])
        self.assertEqual(second["read_only"]["selected"], ["a"])
        self.assertEqual(second["replacement"]["selected"], ["b"])
        self.assertEqual(second["feedback"]["selected"], ["b"])
        self.assertEqual(second["feedback_writes"], ["b"])
        self.assertEqual(second["replacement_rebuild_items"], ["a"])
        self.assertEqual(second["rebuilds"], ["feedback_eviction"])
        self.assertTrue(result["live_replay_exact"])
        self.assertFalse(any("disposable rubric" in state for state, *_ in engine.assessments))
        totals = summarize([result])
        self.assertEqual(totals["primary_endpoint"], "human_judged_memory_use_behavior_and_strategy_preference")
        self.assertFalse(totals["primary_human_endpoint_measured"])
        self.assertFalse(totals["answer_accuracy_measured"])
        self.assertFalse(totals["human_ratings_collected"])
        self.assertEqual(totals["pilot_endpoint"], "curator_preferred_retention_hypothesis_before_queries")
        self.assertEqual(totals["arms"]["feedback"]["curator_preferred_coverage_mean"], 1.0)
        self.assertEqual(totals["curator_hypothesis_deltas"]["feedback_vs_read_only"]["wins"], 1)
        self.assertEqual(totals["secondary_endpoint"], "future_question_answer_accuracy_not_measured")

    def test_three_strategies_receive_each_users_evidence_and_do_not_write_appraisal(self):
        engine = MutableRuntime(lambda *_: appraisal())
        case = {"id": "evidence", "cards": {"a": "Card A"}, "turns": [
            turn("A matters.", {"a": "A matters."}),
            turn("A still matters.", {"a": "A still matters."}, day=2),
        ]}
        result = evaluate_case(engine, case, budget=1)
        self.assertEqual(len(engine.assessments), 6)
        second_turn = engine.assessments[3:]
        self.assertEqual({state.count("\n\nUser: A still matters.") for state, *_ in second_turn}, {1})
        self.assertEqual({tuple(m["quote"] for m in mentions) for _, _, mentions, _ in second_turn},
                         {("A matters.", "A still matters.")})
        self.assertEqual({card for _, card, _, _ in second_turn}, {"Card A"})
        self.assertEqual({as_of for *_, as_of in second_turn}, {START + timedelta(days=2)})
        self.assertTrue(all("disposable rubric" not in state for state, *_ in engine.assessments))
        self.assertEqual(set(DIMENSIONS) | {"evidence", "rating_source_turn", "as_of"},
                         set(result["rounds"][1]["feedback"]["appraisal"]["a"]))

    def test_superseded_card_is_removed_from_both_persistent_strategies(self):
        engine = MutableRuntime(lambda *_: appraisal(importance=4))
        case = {"id": "correction", "cards": {"old": "Old time", "new": "New time"}, "turns": [
            turn("The old time is nine.", {"old": "old time is nine"}, preferred=["old"]),
            turn("Correction: ten replaces nine.", {"new": "ten replaces nine"},
                 retire=["old"], preferred=["new"]),
        ]}
        result = evaluate_case(engine, case, budget=1)
        second = result["rounds"][1]
        self.assertEqual(second["superseded_cards_removed"],
                         {"read_only": [], "replacement": ["old"], "feedback": ["old"]})
        self.assertNotIn("old", second["replacement"]["selected"])
        self.assertNotIn("old", second["feedback"]["selected"])
        self.assertEqual(second["rebuilds"], ["feedback_retirement"])
        self.assertEqual(summarize([result])["superseded_card_handling"]["feedback"]["prior_cards_removed"], 1)

    def test_uncertain_appraisal_stays_pending_instead_of_becoming_zero(self):
        def grade(_, card, __, ___):
            result = appraisal(importance=4)
            if card == "Card B":
                result["emotion_intensity"] = None
            return result

        case = {"id": "uncertain", "cards": {"a": "Card A", "b": "Card B"}, "turns": [
            turn("A is first and B is new.", {"a": "A is first", "b": "B is new"}),
        ]}
        result = evaluate_case(MutableRuntime(grade), case, budget=1)["rounds"][0]["feedback"]
        self.assertEqual(result["selected"], ["a"])
        self.assertEqual(result["pending"], ["b"])

    def test_absent_quote_cannot_be_used_to_create_an_item(self):
        engine = MutableRuntime(lambda *_: appraisal(importance=4))
        case = {"id": "invalid", "cards": {"a": "Card A"}, "turns": [
            turn("Nothing here.", {"a": "invented"})
        ]}
        with self.assertRaisesRegex(ValueError, "quote"):
            evaluate_case(engine, case, budget=1)
        self.assertEqual(engine.assessments, [])

    def test_report_counts_cards_actually_forwarded_including_a_retirement_restore(self):
        engine = MutableRuntime(lambda *_: appraisal(importance=4))
        case = {"id": "write-counts", "cards": {"keep": "Keep", "old": "Old", "new": "New"}, "turns": [
            turn("Keep and Old.", {"keep": "Keep", "old": "Old"}),
            turn("Replace Old with New.", {"new": "New"}, retire=["old"]),
        ]}
        first, second = evaluate_case(engine, case, budget=2)["rounds"]
        self.assertEqual((first["replacement_card_write_tokens"], first["feedback_card_write_tokens"]), (0, 2))
        self.assertEqual((second["replacement_card_write_tokens"], second["feedback_card_write_tokens"]), (1, 2))

    def test_irrelevant_turn_can_change_ratings_without_changing_selection(self):
        def grade(state, *_):
            return appraisal(importance=1 + int(any("unrelated" in part for part in state)))

        case = {"id": "quiet-drift", "cards": {"a": "A"}, "turns": [
            turn("Keep A.", {"a": "A"}), turn("Something unrelated.", irrelevant=True),
        ]}
        stats = summarize([evaluate_case(MutableRuntime(grade), case, 1)])
        for arm in stats["irrelevant_turn_stability"].values():
            self.assertEqual((arm["selection_stable"], arm["appraisal_items_stable"]), (1, 0))

    def test_feedback_state_reorders_cards_when_rank_order_changes(self):
        def grade(state, card, *_):
            flipped = any("Flip priority." in part for part in state)
            importance = (3 if flipped else 4) if card == "Card A" else (4 if flipped else 3)
            return appraisal(importance=importance)

        engine = MutableRuntime(grade)
        case = {"id": "order-change", "cards": {"a": "Card A", "b": "Card B"}, "turns": [
            turn("A and B both matter.", {"a": "A", "b": "B"}),
            turn("Flip priority."),
            turn("Continue."),
        ]}
        result = evaluate_case(engine, case, budget=2)
        second = result["rounds"][1]
        self.assertEqual(second["feedback"]["selected"], ["b", "a"])
        self.assertEqual(second["rebuilds"], ["feedback_reorder"])
        self.assertEqual(second["cards_forwarded"]["feedback"], ["b", "a"])
        next_feedback_state = engine.assessments[-1][0]
        card_a = "\n\n[Working memory card]\nCard A"
        card_b = "\n\n[Working memory card]\nCard B"
        self.assertLess(next_feedback_state.index(card_b), next_feedback_state.index(card_a))
        stats = summarize([result])
        self.assertEqual(stats["card_order_changes_between_turns"]["feedback"], 1)

    def test_equal_score_ties_are_reported_as_lexical_id_ties(self):
        case = {"id": "zero-tie", "cards": {"z": "Fact Z", "a": "Fact A"}, "turns": [
            turn("Fact A and fact Z.", {"z": "fact Z", "a": "Fact A"}),
        ]}
        result = evaluate_case(MutableRuntime(lambda *_: appraisal(importance=0, recency=0)), case, 1)
        stats = summarize([result])
        self.assertEqual(result["rounds"][0]["feedback"]["selected"], ["a"])
        self.assertEqual(stats["score_tie_groups"]["feedback"],
                         {"groups": 1, "zero_score_groups": 1, "affected_turns": 1})
        self.assertEqual(stats["tie_break"], "descending unrounded score, then lexicographic memory_id")

    def test_window_pair_changes_only_the_older_dialogue_not_candidate_evidence(self):
        fixture = Path(__file__).resolve().parents[1] / 'fixtures' / 'state-window-pair-v1.json'
        cases = json.loads(fixture.read_text())['cases']
        traces = []
        for case in cases:
            runtime = MutableRuntime(lambda *_: appraisal())
            result = evaluate_case(runtime, case, 1)
            traces.append([call for call in runtime.assessments if call[1] == case['cards']['handoff']
                           and call[0].count('\n\nUser: Switch to the next slide.') == 0])
            self.assertTrue(result['live_replay_exact'])
        actual, fictional = traces
        self.assertEqual((len(actual), len(fictional)), (3, 3))
        self.assertEqual(actual[1], fictional[1])  # rebuilt states and evidence match exactly
        for i in (0, 2):
            self.assertNotEqual(actual[i][0], fictional[i][0])
            self.assertEqual(actual[i][1:], fictional[i][1:])

    def test_feedback_tracks_assistant_turn_even_when_no_card_is_carried(self):
        engine = MutableRuntime(lambda *_: appraisal())
        case = {"id": "assistant-history", "cards": {"a": "Card A"}, "turns": [
            turn("Unrelated opener."),
            turn("A matters.", {"a": "A matters."}),
        ]}
        evaluate_case(engine, case, budget=1)
        feedback_assessment = engine.assessments[-1][0]
        self.assertIn("\n\nAssistant: Noted.", feedback_assessment)

    def test_irrelevant_turn_stability_is_scored_as_a_habit_metric(self):
        engine = MutableRuntime(lambda *_: appraisal())
        case = {"id": "quiet", "cards": {"a": "A stable preference"}, "turns": [
            turn("I prefer a simple layout.", {"a": "prefer a simple layout"}, preferred=["a"]),
            turn("What is two plus two?", irrelevant=True),
        ]}
        result = evaluate_case(engine, case, budget=1)
        self.assertEqual(result["rounds"][1]["read_only"]["selected"], ["a"])
        self.assertEqual(result["rounds"][1]["feedback"]["selected"], ["a"])
        stability = summarize([result])["irrelevant_turn_stability"]
        self.assertEqual(stability["read_only"]["compared_turns"], 1)
        self.assertEqual(stability["read_only"]["selection_stable"], 1)
        self.assertEqual(stability["read_only"]["appraisal_items_compared"], 1)
        self.assertEqual(stability["read_only"]["appraisal_items_stable"], 1)
        self.assertEqual(stability["feedback"]["selection_stable"], 1)


if __name__ == "__main__":
    unittest.main()
