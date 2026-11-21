"""Exercise native ChatRWKV x070 state on CPU with a tiny random fixture.

Run as a separate process. No pretrained checkpoint, tokenizer, or service.
The module source must match the pinned hash. Only self-generated tensors
are passed to the upstream checkpoint loader, inside a temporary directory.
"""
import argparse
import contextlib
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from src.priority import MemorySignals, rank_memories, select_seed


SOURCE_SHA256 = '75482aee89a08d2a8c8dbe628110b317fc8d0974ddffbaa52aa19190667305e0'
SEED = 2026


def fixture_weights(torch):
    """One x070 block, two 64-wide heads, 32 opaque token IDs."""
    c, rank, ff, vocab = 128, 8, 256, 32
    rng = torch.Generator(device='cpu').manual_seed(SEED)

    def rand(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float32, device='cpu')

    def matrix(rows, cols):
        return rand(rows, cols) / cols**0.5

    weights = {'emb.weight': rand(vocab, c), 'head.weight': matrix(vocab, c)}
    for name in ('blocks.0.ln0', 'blocks.0.ln1', 'blocks.0.ln2', 'ln_out', 'blocks.0.att.ln_x'):
        weights[name + '.weight'] = torch.ones(c)
        weights[name + '.bias'] = torch.zeros(c)
    att = 'blocks.0.att.'
    for name in ('x_r', 'x_w', 'x_k', 'x_v', 'x_a', 'x_g', 'k_a'):
        weights[att + name] = torch.sigmoid(rand(c))
    for name in ('w0', 'a0', 'k_k'):
        weights[att + name] = rand(c)
    weights[att + 'r_k'] = rand(2, 64)
    for name in ('w', 'a', 'g'):
        weights[att + name + '1'] = rand(c, rank) / c**0.5
        weights[att + name + '2'] = rand(rank, c) / rank**0.5
    for name in ('receptance', 'key', 'value', 'output'):
        weights[att + name + '.weight'] = matrix(c, c)
    weights.update({'blocks.0.ffn.x_k': torch.sigmoid(rand(c)),
                    'blocks.0.ffn.key.weight': matrix(ff, c),
                    'blocks.0.ffn.value.weight': matrix(c, ff)})
    return weights


def selected_card(grade):
    """Only the target's importance changes. Two one-token fixture cards."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    memories = [MemorySignals(item, item, now, now, 1, None, importance, 0.0, 0.0, 0.0)
                for item, importance in [('reference', 0.5), ('target', grade / 4)]]
    [(memory, _)] = select_seed(rank_memories(memories, now), token_budget=1)
    return {'reference': 16, 'target': 17}[memory.memory_id]


def exercise(model, torch):
    # All branches deliberately use the identical single-token forward path.
    observed = []

    def step(tokens: list[int], state: Any) -> tuple[Any, list[Any]]:
        if not tokens:
            raise ValueError('empty token sequence')
        logits, state = model.forward([tokens[0]], state)
        observed.extend([logits, *state])
        for token in tokens[1:]:
            logits, state = model.forward([token], state)
            observed.extend([logits, *state])
        return logits, state

    def equal_state(left, right):
        return len(left) == len(right) and all(torch.equal(a, b) for a, b in zip(left, right))

    def delta(a, b):
        return (a - b).abs().max().item()

    prefix_a, prefix_b, user, rubric, answer, next_turn = [2, 3], [4, 5], [6], [7, 8], [9, 10], [11, 12]
    _, live = step(prefix_a + user, None)
    frozen = copy.deepcopy(live)
    judge_input = copy.deepcopy(live)
    disjoint = all(a.data_ptr() != b.data_ptr() for a, b in zip(live, judge_input))
    score_logits, _ = step(rubric, judge_input)
    fresh_logits, _ = step(prefix_a + user + rubric, None)
    other_logits, _ = step(prefix_b + user + rubric, None)
    isolated = equal_state(live, frozen)
    reply, live = step(answer, live)
    native_reply, native_state = step(prefix_a + user + answer, None)

    # In the optional score-conditioned arm the real dialogue still uses live.
    # The disposable judge state (including its rubric) is never the commit base.
    def next_readout(card):
        suffix = ([] if card is None else [card]) + answer + next_turn
        _, next_boundary = step(suffix, copy.deepcopy(frozen))
        return step(rubric, copy.deepcopy(next_boundary))[0]

    # A binary label readout of an untrained head exercises the full loop. The
    # two forced grades below additionally isolate the causal effect of the card.
    grade = (0, 4)[int(score_logits[:2].argmax().item())]
    chosen = selected_card(grade)
    low, high = selected_card(0), selected_card(4)
    after_low, after_high = next_readout(low), next_readout(high)
    actual, same, off = next_readout(chosen), next_readout(chosen), next_readout(None)
    def margin(logits):
        return (logits[1] - logits[0]).item()

    still_native = equal_state(live, native_state)
    return {'same_history_same_logits': torch.equal(score_logits, fresh_logits),
            'independent_tensor_storage': disjoint,
            'live_matches_no_appraisal': isolated and still_native and torch.equal(reply, native_reply),
            'changed_history_same_suffix_max_logit_delta': delta(score_logits, other_logits),
            'grade_logit_margin_history_a_b': [margin(score_logits), margin(other_logits)],
            'synthetic_binary_grade': grade, 'grade_label_token_ids': [0, 1], 'selected_card_token': chosen,
            'forced_grades_0_4_card_tokens': [low, high], 'fixture_grade_flipped_selected_card': low != high,
            'next_probe_different_card_max_logit_delta': delta(after_low, after_high),
            'next_grade_logit_margin_forced_low_high': [margin(after_low), margin(after_high)],
            'selected_card_vs_no_card_max_logit_delta': delta(actual, off),
            'same_card_same_next_logits': torch.equal(actual, same),
            'all_finite': all(torch.isfinite(t).all().item() for t in observed)}


def run(source: Path) -> dict[str, Any]:
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError('unexpected ChatRWKV source hash')
    env = {'CUDA_VISIBLE_DEVICES': '', 'RWKV_V7_ON': '1', 'RWKV_CUDA_ON': '0',
           'RWKV_JIT_ON': '0', 'RWKV_DE_VERSION': '0', 'RWKV_DML_ON': '0'}
    with TemporaryDirectory() as tmp, patch.dict(os.environ, env), contextlib.redirect_stdout(io.StringIO()):
        torch = importlib.import_module('torch')
        torch.set_num_threads(1)
        module_file = Path(tmp) / 'pinned_chatrwkv_model.py'
        module_file.write_bytes(raw)
        spec = importlib.util.spec_from_file_location('pinned_chatrwkv_model', module_file)
        if spec is None or spec.loader is None:
            raise ImportError('cannot load pinned source')
        upstream = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {spec.name: upstream}):
            spec.loader.exec_module(upstream)
            stem = Path(tmp) / 'random-fixture'
            torch.save(fixture_weights(torch), str(stem) + '.pth')
            model = upstream.RWKV(str(stem), 'cpu fp32')
            result = exercise(model, torch)
    return dict(scope='unmodified ChatRWKV x070 forward, one random CPU block and opaque token IDs',
                source_sha256=SOURCE_SHA256, torch=str(torch.__version__), seed=SEED, **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source), indent=2, allow_nan=False))
