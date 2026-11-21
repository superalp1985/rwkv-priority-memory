"""Fine-tune one source proxy, verify reloaded weights and run sealed diagnostics."""

from argparse import ArgumentParser
from hashlib import file_digest
import json
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F
from transformers import AutoTokenizer, XLMRobertaModel

from pilot_data import EMOTIONS, Example, load_brighter, load_memora, temporal_contract_probe
from pilot_eval import (PAIR_MARGIN, admission_metrics, contrast_uses_context,
                        emotion_metrics, pair_result)

SEED, EPOCHS, BATCH = 42, 2, 24
BASE_REV = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
BASE_SHA = "6fd4797bc397c3b8b55d6bb5740366b57e6a3ce91c04c77f22aafc0c128e6feb"
EXPECTED_UNUSED_KEYS = {
    "lm_head.bias", "lm_head.dense.weight", "lm_head.dense.bias",
    "lm_head.layer_norm.weight", "lm_head.layer_norm.bias",
    "roberta.pooler.dense.weight", "roberta.pooler.dense.bias",
}
INPUT_LIMITS = {"importance": (112, 176), "emotion": (128, 192)}
ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def store(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


class Proxy(nn.Module):
    def __init__(self, base, task):
        super().__init__()
        model, info = XLMRobertaModel.from_pretrained(
            base, local_files_only=True, use_safetensors=True, output_loading_info=True,
            add_pooling_layer=False, attn_implementation="sdpa",
        )
        if any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs", "conversion_errors")):
            raise RuntimeError(f"encoder checkpoint did not load exactly: {info}")
        if set(info.get("unexpected_keys") or ()) != EXPECTED_UNUSED_KEYS:
            raise RuntimeError(f"unexpected non-MLM weight keys: {info}")
        self.encoder = model.requires_grad_(False)
        self.encoder.encoder.layer[-2:].requires_grad_(True)
        self.head = nn.Linear(model.config.hidden_size, 1 if task == "importance" else 4 * len(EMOTIONS))
        self.task = task

    def forward(self, ids, mask):
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0, :]
        raw = self.head(hidden).float()
        return raw[:, 0] if self.task == "importance" else raw.reshape(-1, len(EMOTIONS), 4)


def encode(rows, tokenizer, task, *, with_context=True):
    target_cap, limit = INPUT_LIMITS[task]
    targets = tokenizer([row.target for row in rows], add_special_tokens=False, verbose=False)["input_ids"]
    past = ["\n".join(f"{r.capitalize()}: {t}" for r, t in row.context) if with_context else "" for row in rows]
    context = tokenizer(past, add_special_tokens=False, verbose=False)["input_ids"] if any(past) else [[]] * len(rows)
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    clipped = dict(target=0, context=0)
    records = []
    for row, target, history in zip(rows, targets, context, strict=True):
        clipped["target"] += len(target) > target_cap
        if history:
            budget = limit - min(len(target), target_cap) - 4
            if budget <= 0:
                raise ValueError(f"no context tokens fit: {row.id}")
            clipped["context"] += len(history) > budget
            ids = [bos, *history[-budget:], eos, eos, *target[:target_cap], eos]
        else:
            ids = [bos, *target[:target_cap], eos]
        if not target or len(ids) > limit:
            raise ValueError(f"bad token sequence: {row.id}")
        records.append((ids, row.labels))
    return records, clipped


def batches(records, tokenizer, *, order=None):
    if order is None:
        order = range(len(records))
    for offset in range(0, len(order), BATCH):
        group = [records[i] for i in order[offset:offset + BATCH]]
        width = ((max(len(ids) for ids, _ in group) + 7) // 8) * 8
        ids = torch.full((len(group), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, (seq, _) in enumerate(group):
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, :len(seq)] = 1
        labels = torch.tensor([y for _, y in group], dtype=torch.long)
        yield ids.to("cuda"), mask.to("cuda"), labels.to("cuda")


def counts_and_weights(train, task):
    if task == "importance":
        return None, None, None
    labels = np.asarray([row.labels for row in train])
    counts = np.stack([np.bincount(x, minlength=4) for x in labels.T])
    # Limit the influence of sparsely supported high grades without treating them as absent.
    weights = np.minimum(8, np.sqrt(counts.max(axis=1, keepdims=True) / np.maximum(counts, 1)))
    return counts, torch.as_tensor(weights, device="cuda", dtype=torch.float32), np.median(labels, axis=0)


def objective(logits, target, weights):
    if weights is None:
        return F.binary_cross_entropy_with_logits(logits, target[:, 0].float())
    per = F.cross_entropy(logits.transpose(1, 2), target, reduction="none")
    sample = weights.gather(1, target.T).T
    return (per * sample).sum() / sample.sum()


@torch.inference_mode()
def predict(model, records, tokenizer):
    model.eval()
    results = []
    for ids, mask, _ in batches(records, tokenizer):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(ids, mask)
        expected = (len(ids),) if model.task == "importance" else (len(ids), len(EMOTIONS), 4)
        if logits.shape != expected or not torch.isfinite(logits).all().item():
            raise RuntimeError("model emitted nonfinite logits or the wrong shape")
        probs = logits.sigmoid() if model.task == "importance" else logits.softmax(dim=-1)
        results.append(probs.cpu().numpy())
    return np.concatenate(results)


def expectations(probas):
    return probas @ np.arange(4)


def evaluate(task, rows, probs, medians):
    labels = np.asarray([row.labels for row in rows])
    if task == "importance":
        return admission_metrics(labels, probs)
    return emotion_metrics(labels, expectations(probs), discrete=probs.argmax(axis=-1), train_medians=medians)


def pair_examples(task, cases):
    return [Example(case["id"] + "/" + side, row["target"],
                    tuple(tuple(turn) for turn in row.get("context", [])), ())
            for case in cases for side in ("higher", "lower") for row in (case[side],)]


def compare_pairs(model, tokenizer):
    cases = json.loads((ROOT / "pairs.json").read_bytes())[model.task]
    rows = pair_examples(model.task, cases)
    records, _ = encode(rows, tokenizer, model.task)
    raw = predict(model, records, tokenizer)
    target_only, _ = encode(rows, tokenizer, model.task, with_context=False)
    unconditioned = predict(model, target_only, tokenizer)
    results = []
    for i, case in enumerate(cases):
        hi, lo = raw[2 * i:2 * i + 2]
        hi0, lo0 = unconditioned[2 * i:2 * i + 2]
        if model.task == "emotion":
            high, low = expectations(hi), expectations(lo)
            result = pair_result(case["id"], float(high.max() / 3), float(low.max() / 3))
            result.update(higher_vector=high.tolist(), lower_vector=low.tolist())
        else:
            result = pair_result(case["id"], float(hi), float(lo))
        result.update(higher_context_delta=(hi - hi0).tolist() if model.task == "emotion" else float(hi - hi0),
                      lower_context_delta=(lo - lo0).tolist() if model.task == "emotion" else float(lo - lo0),
                      contrast_uses_context=contrast_uses_context(hi, lo, hi0, lo0))
        results.append(result)
    return results


def run(args):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("expected exactly one visible Turing GPU")
    base, task = args.data / "base", args.task
    if digest(base / "model.safetensors") != BASE_SHA:
        raise ValueError("base checkpoint failed SHA-256 verification")
    memora = args.data / "memora" / "a6493188.tar.gz"
    brighter = args.data / "brighter"
    sources = [memora] if task == "importance" else [brighter / f"{s}.csv" for s in ("train", "dev", "test")]
    data, audit = load_memora(memora) if task == "importance" else load_brighter(brighter)
    tok = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    model = Proxy(base, task).to("cuda")
    counts, weights, medians = counts_and_weights(data["train"], task)
    train, clipped_train = encode(data["train"], tok, task)
    dev, clipped_dev = encode(data["dev"], tok, task)
    smoke_rows = ([next(row for row in data["dev"] if row.labels == (v,)) for v in (0, 1)]
                  if task == "importance" else data["dev"][:4])
    smoke, _ = encode(smoke_rows, tok, task)
    initial = predict(model, smoke, tok)
    evaluate(task, smoke_rows, initial, medians)
    print(f"{task}: real base + initialized head smoke OK {initial.shape} on {torch.cuda.get_device_name()}", flush=True)

    if args.smoke_only:
        dev_metrics = evaluate(task, data["dev"], predict(model, dev, tok), medians)
        ids, mask, labels = next(batches(train[:BATCH], tok))
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = objective(model(ids, mask), labels, weights)
        loss.backward()
        selected_grads = [p.grad for p in model.parameters() if p.requires_grad]
        if (not torch.isfinite(loss).item() or not selected_grads
                or any(g is not None and not torch.isfinite(g).all().item() for g in selected_grads)
                or not any(g is not None and torch.count_nonzero(g).item() for g in selected_grads)):
            raise RuntimeError("real source batch did not produce finite nonzero gradients")
        result = dict(task=task, phase="pretraining_gate", base_revision=BASE_REV, base_sha256=BASE_SHA,
                      output_schema="binary_logit" if task == "importance" else "six x four grade logits",
                      dimensions=1 if task == "importance" else len(EMOTIONS),
                      grade_count=2 if task == "importance" else 4,
                      real_checkpoint_forward_shape=list(initial.shape), source_dev_rows=len(data["dev"]),
                      source_dev_metric_pipeline=dev_metrics,
                      metric_pipeline_executed=True, training_probe=dict(rows=len(labels), loss=loss.detach().item(),
                                                                         finite_nonzero_gradients=True),
                      clipped_dev=clipped_dev, data=audit,
                      data_sha256={str(p.relative_to(args.data)): digest(p) for p in sources},
                      base_untrained=True, project_grades_qualified=False,
                      elapsed_seconds=round(perf_counter() - args.start, 2))
        store(args.out, result)
        print(json.dumps(dict(task=task, phase="pretraining_gate", dev_rows=len(data["dev"]),
                              metric_pipeline_executed=True, training_probe_passed=True,
                              artifact=str(args.out)), ensure_ascii=False), flush=True)
        return

    last_layers = list(model.encoder.encoder.layer[-2:].parameters())
    optimizer = torch.optim.AdamW([{"params": last_layers, "lr": 3e-5},
                                   {"params": model.head.parameters(), "lr": 3e-4}], weight_decay=.01)
    trainable = [p for p in model.parameters() if p.requires_grad]
    names = {k for k, v in model.named_parameters() if v.requires_grad}
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(SEED)
    primary = "average_precision" if task == "importance" else "macro_pearson"
    best, best_epoch, snapshot, epochs = -float("inf"), None, None, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = list(range(len(train)))
        rng.shuffle(order)
        total_loss, steps = 0.0, 0
        for ids, mask, labels in batches(train, tok, order=order):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = objective(model(ids, mask), labels, weights)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"nonfinite train loss in {task}, epoch {epoch}, step {steps}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(ids)
            steps += 1
        dev_metrics = evaluate(task, data["dev"], predict(model, dev, tok), medians)
        epochs.append(dict(epoch=epoch, steps=steps, train_loss=total_loss / len(train), dev=dev_metrics))
        print(f"{task}: epoch {epoch} loss={total_loss/len(train):.4f} dev {primary}={dev_metrics[primary]:.4f}", flush=True)
        if dev_metrics[primary] > best:
            best, best_epoch = dev_metrics[primary], epoch
            snapshot = {k: v.detach().cpu().contiguous() for k, v in model.named_parameters() if k in names}

    out = args.out
    out.mkdir(mode=0o700, parents=True, exist_ok=False)
    checkpoint = out / "delta.safetensors"
    assert snapshot is not None and best_epoch is not None
    save_file(snapshot, checkpoint, metadata={"base_revision": BASE_REV, "task": task, "epoch": str(best_epoch)})
    saved = load_file(checkpoint)
    restored = model.load_state_dict(saved, strict=False)
    expected_frozen = set(model.state_dict()) - names
    if (set(saved) != names or set(restored.unexpected_keys)
            or set(restored.missing_keys) != expected_frozen):
        raise RuntimeError("delta could not be restored")

    test, clipped_test = encode(data["test"], tok, task)
    probs = predict(model, test, tok)
    test_rows = data["test"]
    metrics = evaluate(task, test_rows, probs, medians)
    lengths = np.asarray([len(row.target) for row in test_rows])
    labels = np.asarray([row.labels for row in test_rows])
    if task == "importance":
        base_metrics = admission_metrics(labels, lengths)
        no_context, _ = encode(test_rows, tok, task, with_context=False)
        ablation = evaluate(task, test_rows, predict(model, no_context, tok), medians)
        passed_source = metrics[primary] > max(metrics["prevalence"], base_metrics[primary])
    else:
        base_metrics = emotion_metrics(labels, np.repeat(lengths[:, None], len(EMOTIONS), axis=1))
        ablation = None
        passed_source = metrics[primary] >= .25 and metrics[primary] > base_metrics[primary]

    # This file is read only after selecting and reloading the checkpoint.
    pairs = compare_pairs(model, tok)
    result = dict(task=task, scope="local source-domain proxy research, not project grades", seed=SEED,
                  model=dict(base_revision=BASE_REV, base_sha256=BASE_SHA, delta_sha256=digest(checkpoint),
                             delta_saved_reloaded_in_training_process=True,
                             trainable_parameters=sum(p.numel() for p in trainable)),
                  runtime=dict(torch=str(torch.__version__), transformers=__import__("transformers").__version__,
                               gpu=torch.cuda.get_device_name(), visible=os.getenv("CUDA_VISIBLE_DEVICES"),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated()),
                  data_sha256={str(p.relative_to(args.data)): digest(p) for p in sources}, data=audit,
                  code_sha256={name: digest(ROOT / name) for name in ("train_pilot.py", "pilot_eval.py", "pilot_data.py", "pairs.json")},
                  input_limits=INPUT_LIMITS[task], clipped=dict(train=clipped_train, dev=clipped_dev, test=clipped_test),
                  epochs=epochs, best_epoch=best_epoch, primary_metric=primary, test=metrics,
                  length_baseline=base_metrics, context_ablation_test=ablation, passed_source_gate=bool(passed_source),
                  unsupported_training_grades=np.argwhere(counts == 0).tolist() if counts is not None else [],
                  deterministic_temporal_probe=temporal_contract_probe(),
                  diagnostic_pairs=pairs, pair_margin=PAIR_MARGIN,
                  passed_diagnostic_gate=all(row["pass"] for row in pairs),
                  project_grades_qualified=False, rwkv_replay_authorized_by_this_run=False,
                  elapsed_seconds=round(perf_counter() - args.start, 2))
    store(out / "result.json", result)
    print(json.dumps(dict(task=task, source_test=metrics[primary], passed_source_gate=bool(passed_source),
                          diagnostic_pairs_passed=sum(row["pass"] for row in pairs),
                          diagnostic_pairs_total=len(pairs), artifact=str(out)), ensure_ascii=False), flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("importance", "emotion"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing run directory")
    os.umask(0o077)
    args.start = perf_counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()