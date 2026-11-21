"""Exercise just the pinned upstream time_mixing__ function on synthetic CPU tensors."""

import argparse
import ast
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable

EXPECTED = 'a61f35716b2ef81fa1c97bfd7f67bccd78d3a8968d0570748d4631fecf885500'


def get_functions(path, torch):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError('reference SHA-256 mismatch; use the revision in the README')
    # Never import the demo module: its top level loads a CUDA model and runs sampling.
    fn, = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'time_mixing__']
    mod_fn = copy.deepcopy(fn)
    mod_fn.args.args.append(ast.arg(arg='exponent'))
    (where,) = [i for i, s in enumerate(mod_fn.body) if isinstance(s, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'vk' for t in s.targets)]
    mod_fn.body[where:where] = ast.parse('if exponent != 1.0:\n    w = w.pow(exponent)').body

    def compile_one(node) -> Callable[..., Any]:
        env: dict[str, Any] = {'torch': torch}
        unit = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
        exec(compile(unit, '<pinned-time-mix>', 'exec'), env)
        return env['time_mixing__']

    return compile_one(fn), compile_one(mod_fn)


def run(path):
    try:
        torch = importlib.import_module('torch')
    except ModuleNotFoundError as exc:
        raise SystemExit('Optional probe requires an interpreter with CPU PyTorch; see README.') from exc
    torch.set_num_threads(1)
    H, N, rank, seed = 2, 64, 8, 2026
    D = H * N
    rng = torch.Generator(device='cpu').manual_seed(seed)

    def vec(*shape):
        return torch.randn(*shape, generator=rng, dtype=torch.float32, device='cpu')

    def matrix(i, j):
        return vec(i, j) / math.sqrt(i)

    native, candidate = get_functions(path, torch)
    fixed = {k: torch.sigmoid(vec(D)) for k in ('x_r','x_w','x_k','x_v','x_a','x_g','k_a')}
    fixed.update({k:vec(D) for k in ('w0','a0','v0','k_k','r_k')})
    fixed.update({k:matrix(D, rank) for k in ('w1','a1','v1','g1')})
    fixed.update({k:matrix(rank, D) for k in ('w2','a2','v2','g2')})
    fixed.update({k:matrix(D, D) for k in ('kw','vw','rw','ow')})
    fixed.update(ln_w=torch.ones(D), ln_b=torch.zeros(D), layer_id=0, H=H, N=N)
    x1, x2 = vec(D), vec(D)
    ctx = dict(x=x1, x_prev=torch.zeros(D), v_first=torch.zeros(D),
               state=torch.zeros(H,N,N), **fixed)
    q = math.exp(-0.1)
    with torch.inference_mode():
        first = native(**ctx)
        first_on = candidate(**ctx, exponent=q)
        ctx.update(x=x2, x_prev=first[1], state=first[2], v_first=first[3])
        baseline2 = native(**ctx)
        off2 = candidate(**ctx, exponent=1.0)
        on2 = candidate(**ctx, exponent=q)
        at_empty = all(torch.equal(a,b) for a,b in zip(first, first_on))
        off = all(torch.equal(a,b) for a,b in zip(baseline2, off2))
        finite = all(torch.isfinite(x).all().item() for row in (first,first_on,baseline2,off2,on2) for x in row)
        state_diff = (baseline2[2] - on2[2]).abs().max().item()
        output_diff = (baseline2[0] - on2[0]).abs().max().item()
        if not (at_empty and off and finite and state_diff > 0 and output_diff > 0):
            raise RuntimeError('mechanism contract failed')
    return {'scope':'upstream single x070 time-mix layer, two synthetic token vectors, CPU float32; no model, logits or retrieval',
            'source_sha256':EXPECTED, 'torch':str(torch.__version__), 'seed':seed, 'heads':H,
            'head_size':N, 'output_tuple_equal_on_first_step_from_zero':at_empty,
            'off_output_tuple_bitwise_equal_on_second_step':off, 'all_finite':finite,
            'enabled_second_step_max_abs_delta_state':state_diff,
            'enabled_second_step_max_abs_delta_layer_output':output_diff}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='unaltered pinned rwkv_v7_demo_rnn.py')
    parser.add_argument('--json-out', type=Path, help='also save the machine-readable result')
    args = parser.parse_args()
    text = json.dumps(run(args.source), indent=2, allow_nan=False) + '\n'
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding='utf-8')
    print(text, end='')


if __name__ == '__main__':
    main()
