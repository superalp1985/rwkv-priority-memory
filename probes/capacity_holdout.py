"""Paired RWKV pressure, state and reply probe, using frozen calibration weights."""

from argparse import ArgumentParser
from collections import Counter
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import timedelta
import json
from pathlib import Path
import re
import sqlite3
import sys
from time import perf_counter

from src.retention import update_cards

from .capacity_calibration import (
    BUDGET, DIMENSIONS, EQUAL, RESIDENTS, ROOT, SOURCE_FILES, answer_tokens, assess,
    digest, evaluate, forward_cards, implementation, load_fixture, observed,
    prepare, pressure, push_mention, runtime_arguments, save, score_evidence, user_tokens,
)
from .replay_runtime import load_runtime


ARMS = ('native', 'reselect_equal', 'capacity_equal', 'capacity_context', 'window_context')
PERSISTENT = ('reselect_equal', 'capacity_equal', 'capacity_context')
READOUT = 'ordinal_mean'
WINDOW = 3
MAX_NEW = 32
QUIET = 'Separate topic: what instrument normally has six strings?'
CORRECTION_DELAY = (
    ('Set that aside for a moment. What is the color of fresh grass?', 'Usually green.'),
    ('What object tells the time?', 'A clock.'),
    ('What is the opposite of heavy?', 'Light.'),
    ('Name a spring month.', 'April.'),
)
ROLE_BREAK = re.compile(rb'\n\n|\n(?:User|Assistant|System):')


def archive_quote(history, phrase):
    """Search one synthetic dialogue scope; return sourced user text, not a summary."""
    if not phrase.strip():
        raise ValueError('quote lookup requires a nonempty phrase')
    with closing(sqlite3.connect(':memory:')) as db:
        db.execute('CREATE VIRTUAL TABLE archive USING fts5(turn UNINDEXED, role UNINDEXED, body)')
        db.executemany('INSERT INTO archive VALUES (?, ?, ?)', (
            (turn, role, body) for turn, pair in enumerate(history, start=1)
            for role, body in zip(('user', 'assistant'), pair)
        ))
        query = '"' + phrase.replace('"', '""') + '"'
        rows = db.execute('SELECT turn, body FROM archive WHERE archive MATCH ? AND role = ? '
                          'ORDER BY bm25(archive), rowid LIMIT 3', (query, 'user')).fetchall()
    return [dict(turn=turn, role='user', text=body) for turn, body in rows]


def generate(runtime, state, evidence=()):
    branch = deepcopy(state)
    if evidence:
        lines = '\n'.join(f"User turn {e['turn']}: {e['text']}" for e in evidence)
        _, branch = runtime.forward(runtime.encode('\n\n[Original conversation excerpts]\n' + lines), branch)
    logits, branch = runtime.forward(runtime.encode('\n\nAssistant:'), branch)
    out = []
    text = b''
    stop = 'token_limit'
    for _ in range(MAX_NEW):
        token = int(logits.argmax().item())
        if token == 0:
            stop = 'eot'
            break
        if token not in runtime.tokenizer.idx2token:
            stop = 'undefined_token'
            break
        out.append(token)
        text = runtime.tokenizer.decodeBytes(out)
        delimiter = ROLE_BREAK.search(text)
        if delimiter:
            text = text[:delimiter.start()]
            stop = 'turn_boundary'
            break
        if len(out) < MAX_NEW:
            logits, branch = runtime.forward([token], branch)
    return dict(text=text.decode('utf-8', errors='replace').strip(), stop=stop, tokens=len(out))


def window_state(runtime, history, user_ids, selected, card_tokens):
    state = None
    for user, answer in history[-WINDOW:]:
        _, state = runtime.forward(user_tokens(runtime, user), state)
        _, state = runtime.forward(answer_tokens(runtime, answer), state)
    _, state = runtime.forward(user_ids, state)
    return forward_cards(runtime, state, selected, card_tokens)


def reselection(scores, lengths):
    """Eager policy control: rank all scored items and take the fitting prefix."""
    ranked = sorted(((value, name) for name, value in scores.items() if value is not None),
                    key=lambda pair: (-pair[0], pair[1]))
    chosen, used = [], 0
    for _, item in ranked:
        used += lengths[item]
        if used > BUDGET:
            break
        chosen.append(item)
    return tuple(chosen)


@dataclass
class Arm:
    state: list
    cards: tuple = RESIDENTS

    def set_cards(self, runtime, clean, selected, tokens):
        old = self.cards
        added = tuple(card for card in selected if card not in old)
        removed = tuple(card for card in old if card not in selected)
        rebuild = old + added != selected
        if rebuild:
            self.state = forward_cards(runtime, deepcopy(clean), selected, tokens)
        else:
            self.state = forward_cards(runtime, self.state, added, tokens)
        self.cards = selected
        return dict(added=added, removed=removed, rebuild=rebuild, written=selected if rebuild else added)


def run_case(runtime, case, context_weights):
    setup = prepare(runtime, case)
    arms = {name: Arm(deepcopy(setup.carried)) for name in PERSISTENT}
    clean = setup.clean
    history = list(setup.history)
    mentions = setup.mentions
    active = dict.fromkeys(RESIDENTS)
    weights = dict(capacity_equal=EQUAL, capacity_context=context_weights)
    observations = assess(runtime, setup.carried, setup.cards, mentions, setup.at, RESIDENTS)
    row = observed(case, setup, observations)
    diagnostics = {}
    checkpoints = []
    transitions = []

    def change(name, chosen, details=None):
        arm = arms[name]
        before = arm.cards
        trace = arm.set_cards(runtime, clean, chosen, setup.tokens)
        return dict(before=before, after=arm.cards, **trace, details=details)

    def scripted_answer(user, answer):
        nonlocal clean
        ids = answer_tokens(runtime, answer)
        _, clean = runtime.forward(ids, clean)
        for arm in arms.values():
            _, arm.state = runtime.forward(ids, arm.state)
        history.append((user, answer))

    def start_user(user):
        nonlocal clean
        ids = user_tokens(runtime, user)
        _, clean = runtime.forward(ids, clean)
        for arm in arms.values():
            _, arm.state = runtime.forward(ids, arm.state)
        return ids

    def eager(at, cached=None):
        grades = dict(cached or {})
        todo = [name for name in active if name not in grades]
        grades.update(assess(runtime, arms['reselect_equal'].state, setup.cards, mentions, at, todo))
        scores = score_evidence(grades, EQUAL, READOUT)
        return change('reselect_equal', reselection(scores, setup.lengths), dict(scores=scores))

    def checkpoint(kind, user, user_ids, expected=None, obsolete=None, phrase=None):
        evidence = archive_quote(history, phrase) if phrase else []
        selected = arms['capacity_context'].cards
        short = window_state(runtime, history, user_ids, selected, setup.tokens)
        states = dict(native=clean, window_context=short, **{name: arms[name].state for name in PERSISTENT})
        # Equal states get one computation; this is deterministic decoding, not extra votes.
        seen = []
        replies = {}
        for name in ARMS:
            boundary = states[name]
            result = next((res for prior, res in seen if runtime.equal(prior, boundary)), None)
            if result is None:
                result = generate(runtime, boundary, evidence)
                seen.append((boundary, result))
            replies[name] = result
        checkpoints.append(dict(kind=kind, user=user, history_turns=len(history), expected_marker=expected,
                                obsolete_marker=obsolete, lookup=phrase, evidence=evidence,
                                cards=dict(native=(), window_context=selected,
                                           **{name: arms[name].cards for name in PERSISTENT}), replies=replies))

    active['incoming'] = None
    trace = {'reselect_equal': eager(setup.at, cached=observations)}
    for name, w in weights.items():
        result = pressure(setup.lengths, observations, w, READOUT)
        trace[name] = change(name, result.cards, asdict(result))
    transitions.append(dict(kind='pressure', arms=trace))
    scripted_answer(case['incoming'][1], 'Understood.')

    user, marker = case['probe']
    ids = start_user(user)
    checkpoint('related', user, ids, expected=marker)
    scripted_answer(user, 'I will use the confirmed detail where it applies.')

    ids = start_user(QUIET)
    trace = {'reselect_equal': eager(setup.at)}
    for name in weights:
        arm = arms[name]
        result = update_cards(arm.cards, (), setup.lengths, BUDGET, None)
        trace[name] = change(name, result.cards, asdict(result))
    transitions.append(dict(kind='quiet', arms=trace))
    checkpoint('quiet', QUIET, ids)
    scripted_answer(QUIET, 'A guitar commonly has six strings.')

    if 'update' in case:
        old, _, update_text, follow, new_value, old_value = case['update']
        day = setup.at + timedelta(days=1)
        start_user(update_text)
        push_mention(mentions, 'updated', len(history) + 1, day, update_text)
        del active[old]
        active['updated'] = None
        trace = {'reselect_equal': eager(day)}
        for name, w in weights.items():
            arm = arms[name]

            def scores(items):
                valid = tuple(card for card in arm.cards if card != old)
                boundary = forward_cards(runtime, deepcopy(clean), valid, setup.tokens) if old in arm.cards else arm.state
                readings = assess(runtime, boundary, setup.cards, mentions, day, items)
                diagnostics[name + '_correction_competition'] = readings
                return score_evidence(readings, w, READOUT)

            result = update_cards(arm.cards, (), setup.lengths, BUDGET, scores, superseded={old: 'updated'})
            trace[name] = change(name, result.cards, asdict(result))
        transitions.append(dict(kind='correction', obsolete=old, active='updated', arms=trace))
        scripted_answer(update_text, 'Understood.')
        for user, answer in CORRECTION_DELAY:
            start_user(user)
            scripted_answer(user, answer)
        ids = start_user(follow)
        checkpoint('correction_followup', follow, ids, expected=new_value, obsolete=old_value)
        scripted_answer(follow, 'I will use the latest confirmed version.')

    phrase = case['quote']
    user = f'Please look up my original message mentioning "{phrase}" and quote it exactly.'
    ids = start_user(user)
    checkpoint('exact_quote', user, ids, phrase=phrase)
    return dict(row=row, cards=setup.cards, transitions=transitions, score_diagnostics=diagnostics,
                checkpoints=checkpoints, common_history=history)


def aggregate_holdout(cases, frozen):
    rows = [case['row'] for case in cases]
    metrics = {}
    for mode, fitted in frozen['fits'].items():
        policy = dict(equal=EQUAL, global_tuned=fitted['global_weights'], context_tuned=fitted['context_weights'])
        metrics[mode] = {name: evaluate(rows, w, mode) for name, w in policy.items()}
    transitions = [(transition, c['row']['id']) for c in cases for transition in c['transitions']]
    policy_events = {}
    for name in PERSISTENT:
        counts = Counter()
        for event, _ in transitions:
            t = event['arms'][name]
            kind = event['kind']
            counts[kind + '_turns'] += 1
            counts[kind + '_rebuilds'] += t['rebuild']
            counts[kind + '_changes'] += t['before'] != t['after']
            if kind == 'correction':
                counts['correction_old_was_resident'] += event['obsolete'] in t['before']
                counts['correction_old_out'] += event['obsolete'] not in t['after']
                counts['correction_new_in'] += event['active'] in t['after']
        policy_events[name] = dict(counts)
    replies = [(c['row']['id'], check) for c in cases for check in c['checkpoints']]
    by_kind = {}
    for kind in dict.fromkeys(check['kind'] for _, check in replies):
        subset = [check for _, check in replies if check['kind'] == kind]
        by_kind[kind] = dict(
            n=len(subset), quotes_found=sum(bool(check['evidence']) for check in subset),
            state_comparison_text_differences=sum(check['replies']['capacity_context']['text'] !=
                                                  check['replies']['window_context']['text'] for check in subset),
            weight_comparison_text_differences=sum(check['replies']['capacity_context']['text'] !=
                                                   check['replies']['capacity_equal']['text'] for check in subset),
            arms={name: dict(expected_marker_present=sum(bool(check['expected_marker']) and
                                                         check['expected_marker'].casefold() in check['replies'][name]['text'].casefold()
                                                         for check in subset),
                             obsolete_marker_present=sum(bool(check['obsolete_marker']) and
                                                         check['obsolete_marker'].casefold() in check['replies'][name]['text'].casefold()
                                                         for check in subset),
                             verbatim_excerpt_in_reply=sum(any(e['text'] in check['replies'][name]['text']
                                                               for e in check['evidence']) for check in subset),
                             empty=sum(not check['replies'][name]['text'] for check in subset),
                             stop=dict(Counter(check['replies'][name]['stop'] for check in subset)))
                  for name in ARMS},
        )
    return dict(retention=metrics, policy_events=policy_events, rendered_reply_checks=by_kind,
                primary_human_endpoint_measured=False, human_ratings_collected=False,
                answer_accuracy_measured=False)


def main():
    parser = ArgumentParser(description=__doc__)
    runtime_arguments(parser)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'fixtures/capacity-holdout-v1.json')
    args = parser.parse_args()
    cases, fixture = load_fixture(args.fixture)
    frozen = json.loads(args.calibration.read_bytes())
    if (frozen['budget'] != BUDGET or tuple(frozen['dimensions']) != DIMENSIONS or
            frozen['primary_readout'] != READOUT or frozen['implementation_sha256'] != implementation(SOURCE_FILES)):
        parser.error('holdout protocol must match the calibration artifact')
    if {c['id'] for c in cases} & {c['id'] for c in frozen['cases']}:
        parser.error('calibration and holdout case IDs overlap')
    start = perf_counter()
    results = []
    with load_runtime(args.source_dir, args.checkpoint, args.threads) as runtime:
        if runtime.metadata != frozen['runtime']:
            parser.error('holdout runtime must match calibration runtime')
        for case in cases:
            w = frozen['fits'][READOUT]['context_weights'][case['context']]
            results.append(run_case(runtime, case, w))
            print(f"evaluated {case['id']} ({len(results)}/{len(cases)})", file=sys.stderr, flush=True)
        report = dict(scope='synthetic fixed-history replay with generated reply forks; no human ratings',
                      fixture=fixture, calibration_sha256=digest(args.calibration), budget=BUDGET,
                      implementation_sha256=implementation((*SOURCE_FILES, 'probes/capacity_holdout.py')),
                      runtime=runtime.metadata, arms=ARMS, readout=READOUT, window=WINDOW,
                      generation=dict(strategy='greedy', token_limit=MAX_NEW),
                      history_policy='scripted shared previous assistant turns; generated replies are not replayed',
                      archive_policy='in-memory per-case original dialogue, explicit exact-quote checkpoint only',
                      eager_policy='matched readout, reselects at pressure, unrelated checkpoint and correction; no prior score fallback',
                      weight_selection='loaded unchanged from calibration, no fitting on holdout',
                      work=dict(forward_calls=runtime.forward_calls, input_tokens=runtime.input_tokens,
                                appraisal_calls=runtime.appraisal_calls, wall_seconds=round(perf_counter() - start, 2)),
                      summary=aggregate_holdout(results, frozen), cases=results)
    save(args.output, report)
    metrics = report['summary']['retention'][READOUT]
    print(json.dumps({name: {key: item[key] for key in ('hits', 'n', 'ties', 'deferred')}
                      for name, item in metrics.items()}))


if __name__ == '__main__':
    main()