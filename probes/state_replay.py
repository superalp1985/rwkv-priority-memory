"""Paired, sequential retention pilot. No retrieval or generated replies."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, TypedDict

from src.priority import POLICY_VERSION, MemorySignals, rank_memories, select_seed


START = datetime(2026, 1, 1, tzinfo=timezone.utc)
ARMS = ('read_only', 'replacement', 'feedback')
DIMENSIONS = ('importance', 'emotion_intensity', 'repetition', 'recency')
RUBRIC_DIMENSIONS = DIMENSIONS


@dataclass(frozen=True)
class Mention:
    turn: int
    at: datetime
    quote: str


class Appraisal(TypedDict, total=False):
    importance: int | None
    emotion_intensity: int | None
    repetition: int | None
    recency: int | None
    confidence: float
    model: str
    prompt: str
    label_probabilities: dict[str, list[float]]
    evidence: list[dict[str, object]]
    rating_source_turn: int
    as_of: str


class Selection(TypedDict):
    selected: list[str]
    used_tokens: int
    scores: list[dict[str, object]]
    pending: list[str]
    appraisal: dict[str, Appraisal]


def _score(runtime, boundary, cards, mentions, previous, now, budget) -> Selection:
    signals, pending, appraisal = [], [], {}
    for item, history in sorted(mentions.items()):
        first, last = history[0], history[-1]
        card, tokens = cards[item]
        evidence = [dict(turn=mention.turn, at=mention.at, quote=mention.quote)
                    for mention in history]
        current = runtime.appraise(deepcopy(boundary), card, evidence, now)
        if not isinstance(current, dict) or any(name not in current for name in RUBRIC_DIMENSIONS):
            raise ValueError(f'incomplete model appraisal for {item}')
        grades = {name: current[name] for name in RUBRIC_DIMENSIONS}
        if any(value is not None and (type(value) is not int or not 0 <= value <= 4)
               for value in grades.values()):
            raise ValueError(f'invalid rating for {item}')
        if any(value is None for value in grades.values()):
            pending.append(item)
        else:
            current_confidence = current.get('confidence', 1.0)
            if not isinstance(current_confidence, (int, float)) or isinstance(current_confidence, bool) \
                    or not 0 <= current_confidence <= 1:
                raise ValueError(f'invalid appraisal confidence for {item}')
            previous[item] = (grades, last, float(current_confidence))
        kept = previous.get(item)
        appraisal[item] = dict(current)
        appraisal[item].update(grades)
        appraisal[item].update(evidence=[dict(turn=mention.turn, at=mention.at.isoformat(), quote=mention.quote)
                                         for mention in history])
        appraisal[item].update(as_of=now.isoformat())
        if kept is None:
            continue
        used_grades, source, used_confidence = kept
        signals.append(MemorySignals(
            memory_id=item,
            text=card,
            first_seen_at=first.at,
            last_user_mention_at=last.at,
            distinct_user_mentions=len(history),
            explicit_user_importance=None,
            model_assessed_importance=used_grades['importance'] / 4,
            model_assessed_emotion_intensity=used_grades['emotion_intensity'] / 4,
            model_assessed_repetition=used_grades['repetition'] / 4,
            model_assessed_recency=used_grades['recency'] / 4,
            model_confidence=used_confidence,
            token_count=len(tokens),
        ))
        appraisal[item].update(rating_source_turn=source.turn)
    ranked = rank_memories(signals, now)
    selected = [memory.memory_id for memory, _ in select_seed(ranked, budget)]
    scores = [snapshot.as_dict() for _, snapshot in ranked]
    return Selection(selected=selected, used_tokens=sum(len(cards[item][1]) for item in selected),
                     scores=scores, pending=pending, appraisal=appraisal)


def evaluate_case(runtime, case, budget):
    """Replay one dialogue across all three strategies, scoring before each reply."""
    work_start = {name: getattr(runtime, name, 0)
                  for name in ('forward_calls', 'input_tokens', 'appraisal_calls')}
    if type(budget) is not int or budget < 1:
        raise ValueError('budget must be a positive integer')
    cards = {item: (text, runtime.encode(f'\n\n[Working memory card]\n{text}'))
             for item, text in case['cards'].items()}
    if not case['turns']:
        raise ValueError('at least one dialogue round is required')
    native, feedback_state, live_logits = None, None, None
    mentions, retired = {}, set()
    carried = {arm: [] for arm in ARMS}
    ratings = {arm: {} for arm in ARMS}
    live_segments, dialogue_history, rounds = [], [], []
    now = START

    def write_cards(state, items, written):
        # Encode each framed segment once so the token budget matches what is fed.
        for item in items:
            _, state = runtime.forward(cards[item][1], state)
            written.append(item)
        return state

    for turn, row in enumerate(case['turns'], start=1):
        at = START + timedelta(days=row['day']) if 'day' in row else now
        if at < now:
            raise ValueError('dialogue dates cannot go backwards')
        now = at
        user, answer = row['user'], row['assistant']
        invalidated = set(row.get('retire', ()))
        retired_carried = {arm: sorted(set(carried[arm]) & invalidated) for arm in ARMS}
        if not invalidated <= mentions.keys():
            raise ValueError('only active item versions can be retired')
        for item, quote in row.get('mentions', {}).items():
            if item not in cards or item in retired | invalidated:
                raise ValueError('mention must reference a non-retired fact version')
            if not quote.strip() or quote not in user:
                raise ValueError('mention quote is not in the user message')
            mentions.setdefault(item, []).append(Mention(turn, at, quote))
        for item in invalidated:
            del mentions[item]
            for arm in ARMS:
                ratings[arm].pop(item, None)
        retired.update(invalidated)
        carried['replacement'] = [item for item in carried['replacement'] if item not in invalidated]
        preferred = row.get('preferred', [])
        if len(set(preferred)) != len(preferred) or not set(preferred) <= mentions.keys():
            raise ValueError('preferred items must be unique active fact versions')

        user_tokens = runtime.encode(f'\n\nUser: {user}')
        answer_tokens = runtime.encode(f'\n\nAssistant: {answer}')
        live_segments.extend((user_tokens, answer_tokens))
        _, native = runtime.forward(user_tokens, native)
        rebuilds = []
        written = {'replacement': [], 'feedback': []}
        old_feedback_cards = [item for item in carried['feedback'] if item not in invalidated]
        if invalidated.intersection(carried['feedback']):
            feedback_state = write_cards(deepcopy(native), old_feedback_cards, written['feedback'])
            rebuilds.append('feedback_retirement')
        elif feedback_state is None:
            feedback_state = deepcopy(native)
        else:
            _, feedback_state = runtime.forward(user_tokens, feedback_state)

        base = _score(runtime, native, cards, mentions, ratings['read_only'], at, budget)
        replacement_boundary = None
        for prior_user, prior_answer in dialogue_history[-3:]:
            _, replacement_boundary = runtime.forward(prior_user, replacement_boundary)
            _, replacement_boundary = runtime.forward(prior_answer, replacement_boundary)
        _, replacement_boundary = runtime.forward(user_tokens, replacement_boundary)
        replacement_boundary = write_cards(replacement_boundary, carried['replacement'], written['replacement'])
        replacement = _score(runtime, replacement_boundary, cards, mentions,
                             ratings['replacement'], at, budget)
        feedback = _score(runtime, feedback_state, cards, mentions, ratings['feedback'], at, budget)
        if turn == 1 and not (base == replacement == feedback):
            raise RuntimeError('first-turn controls diverged before any feedback')

        replacement_removed = set(carried['replacement']) - set(replacement['selected'])
        replacement_rebuild_items = sorted(replacement_removed | set(retired_carried['replacement']))
        feedback_removed = set(old_feedback_cards) - set(feedback['selected'])
        feedback_writes = [item for item in feedback['selected'] if item not in old_feedback_cards]
        append_order = old_feedback_cards + feedback_writes
        if feedback_removed or feedback['selected'] != append_order:
            feedback_state = write_cards(deepcopy(native), feedback['selected'], written['feedback'])
            rebuilds.append('feedback_eviction' if feedback_removed else 'feedback_reorder')
        else:
            feedback_state = write_cards(feedback_state, feedback_writes, written['feedback'])
        carried['replacement'] = replacement['selected']
        carried['feedback'] = feedback['selected']
        rounds.append(dict(
            turn=turn, read_only=base, replacement=replacement, feedback=feedback,
            preferred=preferred, new_user_mentions=list(row.get('mentions', {})),
            replacements=replacement['selected'], feedback_writes=feedback_writes,
            rebuilds=rebuilds, replacement_rebuild_items=replacement_rebuild_items,
            cards_forwarded=written,
            replacement_card_write_tokens=sum(len(cards[item][1]) for item in written['replacement']),
            feedback_card_write_tokens=sum(len(cards[item][1]) for item in written['feedback']),
            irrelevant=bool(row.get('irrelevant', False)),
            retired_by_fixture=sorted(invalidated), superseded_cards_removed=retired_carried,
        ))
        dialogue_history.append((user_tokens, answer_tokens))
        live_logits, native = runtime.forward(answer_tokens, native)
        _, feedback_state = runtime.forward(answer_tokens, feedback_state)

    # Independent unscored, unseeded path; identical token IDs and chunk boundaries.
    control, control_logits = None, None
    for tokens in live_segments:
        control_logits, control = runtime.forward(tokens, control)
    exact = runtime.equal(native, control) and runtime.equal(live_logits, control_logits)
    runtime_work = {name: getattr(runtime, name, 0) - value for name, value in work_start.items()}
    return dict(id=case['id'], rounds=rounds, live_replay_exact=exact,
                runtime_work=runtime_work,
                card_token_lengths={item: len(tokens) for item, (_, tokens) in cards.items()})


def summarize(results) -> dict[str, Any]:
    rounds = [row for case in results for row in case['rounds']]
    labeled = [row for row in rounds if row['preferred']]
    covered = {arm: [len(set(row['preferred']) & set(row[arm]['selected'])) / len(row['preferred'])
                     for row in labeled]
               for arm in ARMS}
    preservation = {arm: [len(set(row['new_user_mentions']) & set(row[arm]['selected']))
                          for row in rounds]
                   for arm in ARMS}
    obsolete = {arm: [len(row['superseded_cards_removed'][arm]) for row in rounds]
                for arm in ARMS}

    irrelevant_rounds = [
        (case, index) for case in results for index in range(1, len(case['rounds']))
        if case['rounds'][index]['irrelevant']
    ]
    def grades(selection):
        return {item: tuple(rating.get(name) for name in DIMENSIONS)
                for item, rating in selection['appraisal'].items()}

    irrelevant_stability = {}
    for arm in ARMS:
        selection_stable = 0
        compared_items = 0
        stable_items = 0
        for case, index in irrelevant_rounds:
            previous = case['rounds'][index - 1][arm]
            current = case['rounds'][index][arm]
            selection_stable += set(previous['selected']) == set(current['selected'])
            before, after = grades(previous), grades(current)
            common = before.keys() & after.keys()
            compared_items += len(common)
            stable_items += sum(before[item] == after[item] for item in common)
        irrelevant_stability[arm] = dict(
            compared_turns=len(irrelevant_rounds),
            selection_stable=selection_stable,
            appraisal_items_compared=compared_items,
            appraisal_items_stable=stable_items,
        )

    def delta(a, b):
        pairs = list(zip(a, b))
        return dict(wins=sum(x > y for x, y in pairs), ties=sum(x == y for x, y in pairs),
                    losses=sum(x < y for x, y in pairs), paired_differences=[x - y for x, y in pairs])

    arm_metrics = {}
    for arm in ARMS:
        arm_metrics[arm] = dict(
            curator_preferred_hits=sum(len(set(row['preferred']) & set(row[arm]['selected'])) for row in labeled),
            curator_preferred_coverage_mean=(sum(covered[arm]) / len(covered[arm]) if covered[arm] else None),
            newly_mentioned_retention=sum(preservation[arm]),
            superseded_cards_removed=sum(obsolete[arm]),
            pending=sum(len(row[arm]['pending']) for row in rounds),
            label_trace={dimension: [rating.get('label_probabilities', {}).get(dimension)
                                     for row in rounds for rating in row[arm]['appraisal'].values()]
                         for dimension in DIMENSIONS},
        )
    card_order_changes = {}
    score_tie_groups = {}
    for arm in ARMS:
        card_order_changes[arm] = sum(
            set(previous[arm]['selected']) == set(current[arm]['selected'])
            and previous[arm]['selected'] != current[arm]['selected']
            for case in results
            for previous, current in zip(case['rounds'], case['rounds'][1:])
        )
        tie_group_count = 0
        zero_tie_group_count = 0
        affected_turns = 0
        for row in rounds:
            by_score = Counter(score['priority_score'] for score in row[arm]['scores'])
            groups = [score for score, count in by_score.items() if count > 1]
            tie_group_count += len(groups)
            zero_tie_group_count += sum(score == 0 for score in groups)
            affected_turns += bool(groups)
        score_tie_groups[arm] = dict(groups=tie_group_count,
                                     zero_score_groups=zero_tie_group_count,
                                     affected_turns=affected_turns)
    replacement_vs_read = delta(
        [len(set(row['preferred']) & set(row['replacement']['selected'])) for row in labeled],
        [len(set(row['preferred']) & set(row['read_only']['selected'])) for row in labeled])
    feedback_vs_read = delta(
        [len(set(row['preferred']) & set(row['feedback']['selected'])) for row in labeled],
        [len(set(row['preferred']) & set(row['read_only']['selected'])) for row in labeled])
    feedback_vs_replacement = delta(
        [len(set(row['preferred']) & set(row['feedback']['selected'])) for row in labeled],
        [len(set(row['preferred']) & set(row['replacement']['selected'])) for row in labeled])
    return dict(
        primary_endpoint='human_judged_memory_use_behavior_and_strategy_preference',
        primary_human_endpoint_measured=False,
        answer_accuracy_measured=False,
        human_ratings_collected=False,
        pilot_endpoint='curator_preferred_retention_hypothesis_before_queries',
        secondary_endpoint='future_question_answer_accuracy_not_measured',
        scope='curated synthetic behavioral pilot; labels are curator hypotheses, not blinded human ratings',
        cases=len(results), rounds=len(rounds), curator_labeled_rounds=len(labeled),
        curator_preferred_item_count=sum(len(row['preferred']) for row in labeled),
        arms=arm_metrics,
        curator_hypothesis_deltas=dict(replacement_vs_read_only=replacement_vs_read,
                                        feedback_vs_read_only=feedback_vs_read,
                                        feedback_vs_replacement=feedback_vs_replacement),
        selection_changes=dict(
            replacement_vs_read_only=sum(set(row['read_only']['selected']) != set(row['replacement']['selected'])
                                         for row in rounds),
            feedback_vs_read_only=sum(set(row['read_only']['selected']) != set(row['feedback']['selected'])
                                      for row in rounds),
            feedback_vs_replacement=sum(set(row['replacement']['selected']) != set(row['feedback']['selected'])
                                        for row in rounds)),
        card_order_changes_between_turns=card_order_changes,
        score_tie_groups=score_tie_groups,
        tie_break='descending unrounded score, then lexicographic memory_id',
        rating_changes=dict(
            replacement_vs_read_only=sum(grades(row['read_only']) != grades(row['replacement']) for row in rounds),
            feedback_vs_read_only=sum(grades(row['read_only']) != grades(row['feedback']) for row in rounds),
            feedback_vs_replacement=sum(grades(row['replacement']) != grades(row['feedback']) for row in rounds)),
        superseded_card_handling={
            arm: dict(correction_events=sum(bool(row['retired_by_fixture']) for row in rounds),
                      corrections_with_prior_card=sum(bool(row['superseded_cards_removed'][arm]) for row in rounds),
                      prior_cards_removed=sum(len(row['superseded_cards_removed'][arm]) for row in rounds))
            for arm in ARMS},
        irrelevant_turn_stability=irrelevant_stability,
        explicit_card_write_tokens=dict(
            replacement=sum(row['replacement_card_write_tokens'] for row in rounds),
            feedback=sum(row['feedback_card_write_tokens'] for row in rounds)),
        explicit_card_write_token_scope='card tokens only; excludes dialogue, appraisal prompts, forwarded history, model forwards, latency, and memory',
        runtime_work={key: sum(case['runtime_work'][key] for case in results)
                      for key in ('forward_calls', 'input_tokens', 'appraisal_calls')},
        runtime_work_scope='total replay model work including appraisal prompts and unscored control replay; wall time and peak memory not measured',
        rebuilds=dict(Counter(reason for row in rounds for reason in row['rebuilds'])),
        live_replay_exact=all(case['live_replay_exact'] for case in results),
    )


def main():
    from .replay_runtime import load_runtime

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fixture', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'fixtures' / 'state-pilot-v1.json')
    parser.add_argument('--budget', type=int, default=24)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, help='write the JSON report to this file instead of stdout')
    args = parser.parse_args()
    if args.budget < 1 or args.threads < 1:
        parser.error('--budget and --threads must be positive')
    raw = args.fixture.read_bytes()
    fixture = json.loads(raw)
    case_ids = [case['id'] for case in fixture['cases']]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        parser.error('fixture must contain at least one case and no duplicate case IDs')
    with load_runtime(args.source_dir, args.checkpoint, args.threads) as runtime:
        results = [evaluate_case(runtime, case, args.budget) for case in fixture['cases']]
        root = Path(__file__).resolve().parents[1]
        implementation_files = ('probes/state_replay.py', 'probes/replay_runtime.py',
                                 'src/priority.py', 'configs/priority-policy-proposal.json')
        report = dict(scope='pretrained, curated dialogue pilot; no retrieval or generated replies',
                      fixture=fixture['version'], fixture_sha256=sha256(raw).hexdigest(),
                      label_source=fixture['label_source'],
                      candidate_generation='fixture-supplied cards and user-mention evidence; extraction, attribution, and versioning are not evaluated',
                      implementation_sha256={name: sha256((root / name).read_bytes()).hexdigest()
                                             for name in implementation_files},
                      policy=POLICY_VERSION, token_budget=args.budget, runtime=runtime.metadata,
                      summary=summarize(results), cases=results)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    else:
        print(serialized)
    return 0 if report['summary']['live_replay_exact'] else 1


if __name__ == '__main__':
    raise SystemExit(main())