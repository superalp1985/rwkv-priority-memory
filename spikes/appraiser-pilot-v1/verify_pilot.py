"""Independently reload a saved delta and reproduce the sealed test results."""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

from pilot_data import EMOTIONS, load_brighter, load_memora
from pilot_eval import admission_metrics, emotion_metrics
from train_pilot import (BASE_SHA, Proxy, counts_and_weights, digest, encode, evaluate,
                         expectations, predict, compare_pairs)


def close(a, b):
    return bool(np.isclose(a, b, rtol=1e-5, atol=1e-7))


def matches(actual, expected):
    if isinstance(expected, dict):
        return (isinstance(actual, dict) and actual.keys() == expected.keys()
                and all(matches(actual[key], value) for key, value in expected.items()))
    if isinstance(expected, list):
        return (isinstance(actual, list) and len(actual) == len(expected)
                and all(matches(a, b) for a, b in zip(actual, expected, strict=True)))
    if isinstance(expected, float):
        return close(actual, expected)
    return actual == expected


def run(args):
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("verification expects one GPU")
    base, run_dir, task = args.data / "base", args.run, args.task
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    delta = run_dir / "delta.safetensors"
    if digest(base / "model.safetensors") != BASE_SHA or digest(delta) != result["model"]["delta_sha256"]:
        raise ValueError("base or delta digest does not match the training report")
    memora = args.data / "memora" / "a6493188.tar.gz"
    brighter = args.data / "brighter"
    sources = [memora] if task == "importance" else [brighter / f"{s}.csv" for s in ("train", "dev", "test")]
    data, _audit = load_memora(memora) if task == "importance" else load_brighter(brighter)
    actual_data_sha = {str(path.relative_to(args.data)): digest(path) for path in sources}
    if actual_data_sha != result["data_sha256"]:
        raise ValueError("source data digests do not match the training report")
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model = Proxy(base, task).to("cuda").eval()
    selected = load_file(delta, device="cpu")
    selected_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(selected) != selected_names:
        raise ValueError("saved delta parameter names differ from model trainable parameters")
    load = model.load_state_dict(selected, strict=False)
    frozen_names = set(model.state_dict()) - selected_names
    if load.unexpected_keys or set(load.missing_keys) != frozen_names:
        raise ValueError("saved delta did not restore onto exact pinned base")

    _counts, _weights, medians = counts_and_weights(data["train"], task)
    dev_rows = data["dev"]
    dev_records, _dev_clipped = encode(dev_rows, tok, task)
    actual_dev = evaluate(task, dev_rows, predict(model, dev_records, tok), medians)
    test_rows = data["test"]
    records, _clipped = encode(test_rows, tok, task)
    actual_test = evaluate(task, test_rows, predict(model, records, tok), medians)
    baseline = result["length_baseline"]
    lengths = np.asarray([len(row.target) for row in test_rows])
    labels = np.asarray([row.labels for row in test_rows])
    if task == "importance":
        actual_baseline = admission_metrics(labels, lengths)
        no_context, _ = encode(test_rows, tok, task, with_context=False)
        actual_ablation = evaluate(task, test_rows, predict(model, no_context, tok), medians)
    else:
        actual_baseline = emotion_metrics(labels, np.repeat(lengths[:, None], len(EMOTIONS), axis=1))
        actual_ablation = None
    pairs = compare_pairs(model, tok)

    metric = result["primary_metric"]
    selected_epoch = next((row for row in result["epochs"] if row["epoch"] == result["best_epoch"]), None)
    if selected_epoch is None:
        raise ValueError("reported best epoch is absent from the epoch ledger")
    best_observed = max(row["dev"][metric] for row in result["epochs"])
    checks = {
        "dev_metrics_match_selected_epoch": matches(actual_dev, selected_epoch["dev"]),
        "selected_epoch_is_dev_best": close(result["dev_metric"], best_observed)
        if "dev_metric" in result else close(selected_epoch["dev"][metric], best_observed),
        "all_test_metrics": matches(actual_test, result["test"]),
        "all_length_baseline_metrics": matches(actual_baseline, baseline),
        "diagnostic_pairs": len(pairs) == len(result["diagnostic_pairs"])
        and all(a["id"] == b["id"] and a["pass"] == b["pass"]
                and close(a["higher"], b["higher"]) and close(a["lower"], b["lower"])
                for a, b in zip(pairs, result["diagnostic_pairs"], strict=True)),
    }
    if task == "importance":
        checks["all_context_ablation_metrics"] = matches(actual_ablation, result["context_ablation_test"])
    if not all(checks.values()):
        raise AssertionError(f"fresh-process reproduction mismatch: {checks}")
    report = dict(task=task, fresh_process_reload=True, base_sha256=BASE_SHA,
                  delta_sha256=digest(delta), data_sha256=actual_data_sha, checks=checks,
                  selected_epoch=result["best_epoch"], reproduced_dev=actual_dev,
                  reproduced_test=actual_test,
                  reproduced_length_baseline=actual_baseline,
                  reproduced_context_ablation=actual_ablation,
                  reproduced_diagnostic_pairs=pairs,
                  project_grades_qualified=False,
                  caution="Source-domain results and synthetic project diagnostics do not constitute project-grade qualification.")
    if args.out.exists():
        raise FileExistsError(args.out)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps(dict(task=task, fresh_process_reload=True, checks=checks, out=str(args.out))))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("importance", "emotion"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.out = args.out or args.run / "fresh-process-verification.json"
    run(args)


if __name__ == "__main__":
    main()