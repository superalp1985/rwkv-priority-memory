"""Fit external retention weights on real RWKV appraisals of synthetic pressure cases."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import timedelta
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter

from src.retention import update_cards

from .replay_runtime import DIMENSIONS, load_runtime
from .state_replay import START


ROOT = Path(__file__).resolve().parents[1]
BUDGET = 34
EQUAL = (25, 25, 25, 25)
CONTEXTS = ('active', 'standing')
RESIDENTS = ('first', 'second')
ITEMS = (*RESIDENTS, 'incoming')
FILLER = (
    ('Separate topic: name one kind of cloud.', 'A cumulus cloud.'),
    ('How many sides has a triangle?', 'Three.'),
    ('Name a stringed instrument.', 'A violin is a stringed instrument.'),
    ('Name a fruit that grows on a tree.', 'An apple grows on a tree.'),
)
SOURCE_FILES = ('src/retention.py', 'probes/capacity_calibration.py', 'probes/replay_runtime.py')


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def implementation(files):
    return {name: digest(ROOT / name) for name in files}


def load_fixture(path):
    raw = path.read_bytes()
    fixture = json.loads(raw)
    ids = [case['id'] for case in fixture['cases']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('fixture must have nonempty, unique case IDs')
    for case in fixture['cases']:
        if case['context'] not in CONTEXTS or case['victim'] not in RESIDENTS:
            raise ValueError(f"invalid case context or victim: {case['id']}")
    info = dict(version=fixture['version'], sha256=sha256(raw).hexdigest(), labels=fixture['labels'])
    return fixture['cases'], info


def user_tokens(runtime, text):
    return runtime.encode('\n\nUser: ' + text)


def answer_tokens(runtime, text):
    return runtime.encode('\n\nAssistant: ' + text)


def forward_cards(runtime, state, items, tokens):
    for item in items:
        _, state = runtime.forward(tokens[item], state)
    return state


def push_mention(mentions, item, turn, at, text):
    mentions.setdefault(item, []).append(dict(turn=turn, at=at, quote=text))


@dataclass
class Setup:
    cards: dict
    tokens: dict
    lengths: dict
    mentions: dict
    history: list
    clean: list
    carried: list
    at: object


def prepare(runtime, case):
    cards = {item: case[item][0] for item in ITEMS}
    if 'update' in case:
        cards['updated'] = case['update'][1]
    tokens = {item: runtime.encode('\n\n[Working memory card]\n' + card)
              for item, card in cards.items()}
    lengths = {item: len(ids) for item, ids in tokens.items()}
    first, second, incoming = (lengths[item] for item in ITEMS)
    if max(first + second, first + incoming, second + incoming) > BUDGET or sum(
            lengths[item] for item in ITEMS) <= BUDGET:
        raise ValueError('need two fitting cards and a one-victim pressure decision')
    d1, d2, d3 = case['days']
    events = [(d1, 'first', case['first'][1], True), (d2, 'second', case['second'][1], True)]
    events += [(day, item, user, False) for item, day, user in case.get('confirm', ())]
    events.sort(key=lambda event: event[0])
    if not 0 <= d1 <= d2 <= d3 or any(day > d3 for day, *_ in events):
        raise ValueError('non-monotonic fixture timing')

    clean = carried = None
    mentions, history, resident = {}, [], ()
    intro = ('Please reply briefly. Use earlier confirmed details when they matter.', 'Of course.')

    def consume(user, answer, item=None, day=0, admit=False):
        nonlocal clean, carried, resident
        if item is not None:
            push_mention(mentions, item, len(history) + 1, START + timedelta(days=day), user)
        u, a = user_tokens(runtime, user), answer_tokens(runtime, answer)
        _, clean = runtime.forward(u, clean)
        _, carried = runtime.forward(u, carried)
        if admit:
            update = update_cards(resident, [item], lengths, BUDGET,
                                  lambda _: (_ for _ in ()).throw(AssertionError('setup competed')))
            if update.contests or update.deferred:
                raise AssertionError('setup should append without pressure')
            resident = update.cards
            carried = forward_cards(runtime, carried, (item,), tokens)
        _, clean = runtime.forward(a, clean)
        _, carried = runtime.forward(a, carried)
        history.append((user, answer))

    consume(*intro)
    for day, item, text, admit in events:
        consume(text, 'Noted.', item, day, admit)
    for exchange in FILLER:
        consume(*exchange)
    if resident != RESIDENTS:
        raise AssertionError('expected two admitted incumbents in chronological order')
    at = START + timedelta(days=d3)
    user = case['incoming'][1]
    push_mention(mentions, 'incoming', len(history) + 1, at, user)
    ids = user_tokens(runtime, user)
    _, clean = runtime.forward(ids, clean)
    _, carried = runtime.forward(ids, carried)
    return Setup(cards, tokens, lengths, mentions, history, clean, carried, at)


def grades_for(rating, readout):
    labels = tuple(rating[name] for name in DIMENSIONS)
    if any(label is None for label in labels):
        return None
    if readout == 'argmax':
        return labels
    if readout != 'ordinal_mean':
        raise ValueError('unknown readout')
    decoded = tuple(rating['label_probabilities'][name][:5] for name in DIMENSIONS)
    return tuple(round(sum(grade * prob for grade, prob in enumerate(row)) / sum(row), 6)
                 for row in decoded)


def assess(runtime, boundary, cards, mentions, at, items):
    return {item: runtime.appraise(deepcopy(boundary), cards[item], mentions[item], at)
            for item in items}


def score_evidence(ratings, weights, readout):
    scores = {}
    for item, rating in ratings.items():
        values = grades_for(rating, readout)
        scores[item] = None if values is None else round(sum(w * x for w, x in zip(weights, values)) / 4, 6)
    return scores


def pressure(lengths, ratings, weights, readout='ordinal_mean'):
    return update_cards(RESIDENTS, ['incoming'], lengths, BUDGET,
                        lambda _: score_evidence(ratings, weights, readout))


GRID = tuple(sorted({EQUAL, *(w for w in product(range(10, 71, 10), repeat=4) if sum(w) == 100)}))


def observed(case, setup, ratings):
    return dict(id=case['id'], context=case['context'], expected_victim=case['victim'],
                lengths=setup.lengths, ratings=ratings)


def decision(row, weights, readout):
    result = pressure(row['lengths'], row['ratings'], weights, readout)
    return {**asdict(result), 'scores': score_evidence(row['ratings'], weights, readout)}


def evaluate(rows, weights, readout):
    """Score expected eviction; cutoff ties earn half independent of tie-break."""
    predictions = []
    for row in rows:
        w = weights[row['context']] if isinstance(weights, dict) else weights
        prediction = decision(row, w, readout)
        evicted = prediction['evicted']
        tied = prediction['contests'][0]['tied']
        expected = row['expected_victim']
        predictions.append(dict(id=row['id'], context=row['context'], weights=w, expected=expected,
                                evicted=evicted, tied=tied, deferred=prediction['deferred'],
                                operational_hit=evicted == (expected,),
                                points2=1 if tied else 2 * (evicted == (expected,))))
    return dict(n=len(rows), points2=sum(p['points2'] for p in predictions),
                hits=sum(p['operational_hit'] for p in predictions),
                strict_hits=sum(p['points2'] == 2 for p in predictions),
                ties=sum(p['tied'] for p in predictions),
                deferred=sum(bool(p['deferred']) for p in predictions), predictions=predictions)


def fit(rows, readout):
    results = [(evaluate(rows, w, readout)['points2'], w) for w in GRID]
    optimum = max(points for points, _ in results)
    winners = [w for points, w in results if points == optimum]
    weights = min(winners, key=lambda w: (sum((a - b)**2 for a, b in zip(w, EQUAL)), w))
    return dict(weights=weights, best_points2=optimum, best_weight_count=len(winners),
                equal_points2=evaluate(rows, EQUAL, readout)['points2'],
                grid_results=[dict(weights=w, points2=points) for points, w in results])


def fit_variants(rows, readout):
    global_fit = fit(rows, readout)
    by_context = {context: fit([row for row in rows if row['context'] == context], readout)
                  for context in CONTEXTS}
    global_weights = global_fit['weights']
    context_weights = {context: fit_result['weights'] for context, fit_result in by_context.items()}
    return dict(global_fit=global_fit, by_context=by_context, global_weights=global_weights,
                context_weights=context_weights,
                calibration=dict(equal=evaluate(rows, EQUAL, readout),
                                 global_tuned=evaluate(rows, global_weights, readout),
                                 context_tuned=evaluate(rows, context_weights, readout)))


def save(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def runtime_arguments(parser):
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)


def main():
    parser = ArgumentParser(description=__doc__)
    runtime_arguments(parser)
    parser.add_argument('--fixture', type=Path, default=ROOT / 'fixtures/capacity-calibration-v1.json')
    args = parser.parse_args()
    cases, fixture = load_fixture(args.fixture)
    if any(not any(case['context'] == context for case in cases) for context in CONTEXTS):
        parser.error('both calibration contexts must be represented')
    start = perf_counter()
    with load_runtime(args.source_dir, args.checkpoint, args.threads) as runtime:
        rows = []
        for case in cases:
            setup = prepare(runtime, case)
            ratings = assess(runtime, setup.carried, setup.cards, setup.mentions, setup.at, RESIDENTS)
            rows.append(observed(case, setup, ratings))
            print(f"calibrated {case['id']} ({len(rows)}/{len(cases)})", file=sys.stderr, flush=True)
        report = dict(scope='synthetic calibration; labels, admission and context supplied by fixture',
                      budget=BUDGET, fixture=fixture, runtime=runtime.metadata,
                      dimensions=DIMENSIONS, implementation_sha256=implementation(SOURCE_FILES),
                      grid_size=len(GRID), equal_weights=EQUAL, primary_readout='ordinal_mean',
                      fits={mode: fit_variants(rows, mode) for mode in ('ordinal_mean', 'argmax')},
                      work=dict(forward_calls=runtime.forward_calls, input_tokens=runtime.input_tokens,
                                appraisal_calls=runtime.appraisal_calls, wall_seconds=round(perf_counter()-start, 2)),
                      cases=rows, primary_human_endpoint_measured=False)
    save(args.output, report)
    primary = report['fits'][report['primary_readout']]
    print(json.dumps({key: primary[key] for key in ('global_weights', 'context_weights')}))


if __name__ == '__main__':
    main()