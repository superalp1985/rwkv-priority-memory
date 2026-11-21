"""Pinned source-domain data and deterministic evidence for the scorer pilot."""

import csv
import json
import random
import re
import tarfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SPLITS = ("train", "dev", "test")
EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
EXPECTED_BRIGHTER = dict(train=2642, dev=200, test=2642)
PERSONAS = dict(
    train=("academic_researcher", "creative_designer", "financial_analyst",
           "marketing_manager", "sales_manager", "software_engineer"),
    dev=("business_executive", "content_writer"),
    test=("management_consultant", "startup_founder"),
)
HORIZONS = dict(train="quarterly", dev="weekly", test="weekly")
SESSION = re.compile(r"[^/]+/data/(weekly|quarterly)/([^/]+)/conversations/session_(\d+)\.json")
ROLES = {"user_agent": "user", "ai_agent": "assistant"}


@dataclass(frozen=True)
class Example:
    id: str
    target: str
    context: tuple[tuple[str, str], ...]
    labels: tuple[int, ...]


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def disjoint_targets(splits, audits):
    owners = defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            owners[normalized(row.target)].add(split)
    duplicated = {key for key, names in owners.items() if len(names) > 1}
    result = {}
    for split, rows in splits.items():
        result[split] = [row for row in rows if normalized(row.target) not in duplicated]
        audits[split]["cross_split_targets_removed"] = len(rows) - len(result[split])
        if not result[split]:
            raise ValueError(f"no rows remain in {split} after overlap check")
    return result


def load_memora(path, *, groups=PERSONAS, train_positive_cap=960, train_negative_cap=1920, seed=42):
    owner = {persona: s for s in SPLITS for persona in groups[s]}
    if len(owner) != sum(map(len, groups.values())):
        raise ValueError("persona crosses splits")
    splits = {s: [] for s in SPLITS}
    audit: dict[str, dict[str, Any]] = {
        s: dict(horizon=HORIZONS[s], personas=list(groups[s]), sessions=0,
                delete_sessions_excluded=0) for s in SPLITS
    }
    sessions = set()
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            match = SESSION.fullmatch(member.name)
            if not match:
                continue
            horizon, persona, session = match.groups()
            split = owner.get(persona)
            if split is None or HORIZONS[split] != horizon:
                continue
            group = f"{horizon}/{persona}/{int(session)}"
            if group in sessions or not member.isfile():
                raise ValueError(f"bad session member: {group}")
            sessions.add(group)
            f = archive.extractfile(member)
            if f is None:
                raise ValueError(f"unreadable session: {group}")
            with f:
                doc = json.load(f)
            if doc["persona"] != persona or doc["session_id"] != int(session):
                raise ValueError(f"session identity mismatch: {group}")
            op = doc["operation"]
            if op not in (None, "add", "update", "delete"):
                raise ValueError(f"unknown operation: {op}")
            audit[split]["sessions"] += 1
            if op == "delete":
                audit[split]["delete_sessions_excluded"] += 1
                continue
            history = []
            last_turn = 0
            for turn in doc["conversation"]:
                n, text, flag = (turn[k] for k in ("turn", "message", "share_memory"))
                role = ROLES[turn["speaker"]]
                if type(n) is not int or n <= last_turn or not isinstance(text, str) or not text.strip():
                    raise ValueError(f"invalid turn in {group}")
                if type(flag) is not bool or (flag and role != "user"):
                    raise ValueError(f"invalid share_memory in {group}")
                if role == "user":
                    splits[split].append(Example(f"{group}/{n}", text, tuple(history[-6:]), (int(flag),)))
                history.append((role, text))
                last_turn = n
    splits = disjoint_targets(splits, audit)
    rng = random.Random(seed)
    caps = {0: train_negative_cap, 1: train_positive_cap}
    for split, rows in splits.items():
        pools = {i: sorted((row for row in rows if row.labels == (i,)), key=lambda x: x.id) for i in caps}
        if not all(pools.values()):
            raise ValueError(f"{split} missing a label")
        audit[split]["counts_before_sampling"] = {i: len(pool) for i, pool in pools.items()}
        if split == "train":
            splits[split] = sorted((row for i, pool in pools.items()
                                    for row in rng.sample(pool, min(len(pool), caps[i]))), key=lambda x: x.id)
        audit[split]["used_counts"] = dict(Counter(row.labels[0] for row in splits[split]))
    return splits, audit


def load_brighter(root, *, expected=EXPECTED_BRIGHTER):
    splits, audit = {}, {}
    ids = set()
    for split in SPLITS:
        rows = []
        with (Path(root) / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["id", "text", *EMOTIONS]:
                raise ValueError("unexpected BRIGHTER fields")
            for row in reader:
                ident, text = row["id"], row["text"]
                y = tuple(int(row[k]) for k in EMOTIONS)
                if not ident or not text or not text.strip() or ident in ids or any(v not in range(4) for v in y):
                    raise ValueError(f"invalid or duplicated row {ident}")
                ids.add(ident)
                rows.append(Example(ident, text, (), y))
        if len(rows) != expected[split]:
            raise ValueError(f"wrong {split} size: {len(rows)}")
        audit[split] = dict(original_rows=len(rows))
        splits[split] = rows
    splits = disjoint_targets(splits, audit)
    for split, rows in splits.items():
        audit[split]["used_rows"] = len(rows)
        audit[split]["used_label_histograms"] = {
            name: [sum(row.labels[i] == grade for row in rows) for grade in range(4)]
            for i, name in enumerate(EMOTIONS)
        }
    return splits, audit


def utc(at):
    if not isinstance(at, datetime) or at.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return at.astimezone(timezone.utc)


def mention_evidence(rows, item, as_of):
    now = utc(as_of)
    unique = {}
    for row in rows:
        if (row["item"], row["role"], row["origin"]) != (item, "user", "dialogue"):
            continue
        ident, at = row["turn"], utc(row["at"])
        if not isinstance(ident, str) or not ident.strip() or at > now:
            raise ValueError("invalid turn ID or future mention")
        if ident in unique and at != unique[ident]:
            raise ValueError("conflicting timestamps for the same user turn")
        unique[ident] = at
    if not unique:
        return None
    last = max(unique.values())
    return len(unique), last, (now - last).total_seconds()


def temporal_contract_probe():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    later = start + timedelta(days=3)
    first = dict(item="v1", role="user", origin="dialogue", turn="u1", at=start)
    replay = [first, dict(first), dict(first, role="assistant", turn="a1", at=later),
              dict(first, origin="retrieval", turn="r1", at=later)]
    repeated = [*replay, dict(first, turn="u2", at=later)]
    before = mention_evidence(replay, "v1", later)
    after = mention_evidence(repeated, "v1", later)
    if before is None or after is None:
        raise AssertionError("deterministic mention probe lost user evidence")
    return dict(replay_count=before[0], replay_last_mention_seconds=before[2],
                replay_does_not_reinforce=before[0] == 1,
                time_advance_does_not_refresh=before[1] == start,
                new_user_turn_increments=after[0] == 2 and after[1] == later and after[2] == 0)