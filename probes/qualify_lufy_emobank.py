"""Qualify the published LUFY RoBERTa checkpoint on EmoBank test data."""

from argparse import ArgumentParser
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer


DOMAIN_CASES = [
    {
        "id": "importance_calm_short_task",
        "text": "就按九点吧。",
        "dimension": "emotion_intensity",
        "expected": "low_or_ambiguous",
        "source": "docs/APPRAISAL.md example; same wording is intentionally context-dependent",
    },
    {
        "id": "importance_calm_long_term_rule",
        "text": "以后每周一的交接时间固定为早上九点。",
        "dimension": "emotion_intensity",
        "expected": "low_or_ambiguous",
        "source": "docs/APPRAISAL.md example; importance is not emotion intensity",
    },
    {
        "id": "quoted_emotional_line",
        "text": "我等这一天等了好久。",
        "dimension": "emotion_intensity",
        "expected": "do_not_treat_as_user_emotion_without_context",
        "source": "docs/APPRAISAL.md quoted-line contrast",
    },
    {
        "id": "personal_high_stakes_event",
        "text": "我刚收到录取通知，真的等这一天等了好久！",
        "dimension": "emotion_intensity",
        "expected": "higher_than_neutral_or_quoted_line",
        "source": "docs/APPRAISAL.md user-event contrast",
    },
    {
        "id": "neutral_acknowledgement",
        "text": "收到，明白了。",
        "dimension": "emotion_intensity",
        "expected": "low",
        "source": "project minimal neutral control",
    },
    {
        "id": "high_positive_arousal",
        "text": "我终于拿到梦寐以求的录取通知了，太激动了！",
        "dimension": "emotion_intensity",
        "expected": "high",
        "source": "project minimal high-arousal control",
    },
    {
        "id": "high_negative_arousal",
        "text": "我失去最重要的人，痛苦得无法呼吸。",
        "dimension": "emotion_intensity",
        "expected": "high",
        "source": "project minimal high-arousal control",
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return float(np.dot(a, b) / denominator) if denominator else float("nan")


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "n": int(len(target)),
        "pearson": correlation(target, prediction),
        "spearman": correlation(rankdata(target), rankdata(prediction)),
        "mae": float(np.mean(np.abs(target - prediction))),
        "rmse": float(np.sqrt(np.mean((target - prediction) ** 2))),
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
        "target_std": float(np.std(target)),
        "prediction_std": float(np.std(prediction)),
        "prediction_min": float(np.min(prediction)),
        "prediction_max": float(np.max(prediction)),
    }


def read_split(path: Path, split: str) -> tuple[list[str], np.ndarray]:
    texts, labels = [], []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == split:
                texts.append(row["text"])
                labels.append((float(row["V"]), float(row["A"])))
    if not texts:
        raise ValueError(f"no rows for split={split!r}")
    return texts, np.asarray(labels, dtype=np.float32)


def predict(model, tokenizer, texts: list[str], device: torch.device, batch_size: int) -> np.ndarray:
    outputs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(f"expected [batch, 2] regression output, got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all().item():
            raise RuntimeError("non-finite LUFY output")
        outputs.append(logits.float().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--emobank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", default="roberta-large")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    texts, targets = read_split(args.emobank, args.split)
    tokenizer = RobertaTokenizer.from_pretrained(args.base_model)
    model = RobertaForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(checkpoint, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"checkpoint mismatch: {incompatible}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    predictions = predict(model, tokenizer, texts, device, args.batch_size)

    domain_predictions = predict(
        model,
        tokenizer,
        [case["text"] for case in DOMAIN_CASES],
        device,
        args.batch_size,
    )
    domain = []
    for case, prediction in zip(DOMAIN_CASES, domain_predictions):
        domain.append({
            **case,
            "valence": float(prediction[0]),
            "arousal": float(prediction[1]),
        })

    result = {
        "scope": "published LUFY RoBERTa-large checkpoint qualification",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "bytes": args.checkpoint.stat().st_size,
            "load_state_dict_strict": True,
        },
        "dataset": {
            "path": str(args.emobank),
            "split": args.split,
            "rows": len(texts),
            "sha256": sha256_file(args.emobank),
            "source": "RuiSumida/LUFY official emobank.csv",
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "batch_size": args.batch_size,
            "max_length": 128,
            "base_model": args.base_model,
        },
        "metrics": {
            "valence": metrics(targets[:, 0], predictions[:, 0]),
            "arousal": metrics(targets[:, 1], predictions[:, 1]),
        },
        "domain_cases": domain,
        "provenance": {
            "checkpoint_card": "https://huggingface.co/RuiSumida/LUFY",
            "dataset_card": "https://huggingface.co/datasets/RuiSumida/LUFY",
            "implementation": "https://github.com/ryuichi-sumida/LUFY/blob/main/Code/roberta/inference_large.py",
            "project_rubric": "docs/APPRAISAL.md",
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "split": args.split,
        "rows": len(texts),
        "device": str(device),
        "valence": result["metrics"]["valence"],
        "arousal": result["metrics"]["arousal"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
