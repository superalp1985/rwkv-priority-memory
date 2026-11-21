"""Create and audit a blinded, answer-free Chinese appraisal pilot packet."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "review-stimuli-zh.json"
DIMS = ("I", "E", "F", "R")
RATING_COLUMNS = {"I": "I重要性0-4或X", "E": "E情绪强度0-4或X",
                  "F": "F独立重复0-4或X", "R": "R新近度0-4或X"}
RATIONALE_COLUMNS = {"I": "I依据或X缺失说明", "E": "E依据或X缺失说明",
                     "F": "F依据或X缺失说明", "R": "R依据或X缺失说明"}
FIELDS = ("编号", "对话", "评估截止UTC", "事项", "事项版本", "逐轮对话角色时间和原文",
          "证据用户轮ID", "事项用户原文证据", "独立用户提及数", "最后用户提及UTC",
          "距最后提及秒", "评估UTC", "刺激清单SHA256",
          *RATING_COLUMNS.values(), *RATIONALE_COLUMNS.values())
IMMUTABLE_FIELDS = tuple(field for field in FIELDS
                         if field not in (*RATING_COLUMNS.values(), *RATIONALE_COLUMNS.values()))
ALLOWED = {"", "X", "0", "1", "2", "3", "4"}


def parse_time(value):
    if not isinstance(value, str):
        raise ValueError("UTC timestamp must be text")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid timestamp: {value}") from exc
    if result.utcoffset() is None:
        raise ValueError(f"timestamp needs an explicit timezone: {value}")
    return result.astimezone(timezone.utc)


def records(fixture=FIXTURE):
    fixture = Path(fixture)
    blob = fixture.read_bytes()
    dialogues = json.loads(blob)
    checksum = hashlib.sha256(blob).hexdigest()
    result = []
    seen = set()
    for dialogue in dialogues:
        as_of = parse_time(dialogue["at"])
        turns = dialogue["turns"]
        if not isinstance(turns, list) or not turns:
            raise ValueError("dialogue needs at least one turn")
        by_id = {}
        rendered = []
        previous = None
        for turn_id, timestamp, role, text in turns:
            at = parse_time(timestamp)
            if (not isinstance(turn_id, str) or not turn_id or turn_id in by_id
                    or role not in {"user", "assistant", "retrieval"}
                    or not isinstance(text, str) or not text.strip()
                    or at > as_of or (previous is not None and at < previous)):
                raise ValueError(f"invalid turn in {dialogue['dialog']}")
            previous = at
            by_id[turn_id] = (at, role, text)
            rendered.append(f"{turn_id} · {at.isoformat()} · {role}: {text}")
        for item_id, item, version, evidence in dialogue["items"]:
            if item_id in seen:
                raise ValueError(f"duplicate item id: {item_id}")
            seen.add(item_id)
            supported = {}
            for turn_id, quote in evidence:
                if turn_id not in by_id:
                    raise ValueError(f"unknown evidence turn {turn_id} in {item_id}")
                at, role, source_text = by_id[turn_id]
                if role != "user" or not quote or quote not in source_text:
                    raise ValueError(f"evidence is not a verbatim user quote: {item_id}/{turn_id}")
                supported[turn_id] = (at, quote)
            if len(supported) != len(evidence):
                raise ValueError(f"duplicate user-turn evidence: {item_id}")
            ordered = sorted(supported.items(), key=lambda pair: (pair[1][0], pair[0]))
            latest = ordered[-1][1][0] if ordered else None
            elapsed = (int((as_of - latest).total_seconds()) if latest else "无")
            result.append({
                "编号": item_id, "对话": dialogue["dialog"],
                "评估截止UTC": as_of.isoformat(), "事项": item,
                "事项版本": version, "逐轮对话角色时间和原文": "\n".join(rendered),
                "证据用户轮ID": "；".join(turn_id for turn_id, _ in ordered) or "无",
                "事项用户原文证据": "\n".join(f"{turn_id}: {quote}" for turn_id, (_, quote) in ordered) or "无",
                "独立用户提及数": str(len(supported)),
                "最后用户提及UTC": latest.isoformat() if latest else "无",
                "距最后提及秒": str(elapsed), "评估UTC": as_of.isoformat(),
                "刺激清单SHA256": checksum,
                **{column: "" for column in RATING_COLUMNS.values()},
                **{column: "" for column in RATIONALE_COLUMNS.values()},
            })
    if len(result) != 12 or len({row["对话"] for row in result}) != 10:
        raise ValueError("unexpected stimulus count; expected 12 items across 10 dialogues")
    return result


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def make_packet(outdir, fixture=FIXTURE):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    items = records(fixture)
    a = sorted(items, key=lambda row: hashlib.sha256(f"reviewer-a:{row['编号']}".encode()).digest())
    b = sorted(items, key=lambda row: hashlib.sha256(f"reviewer-b:{row['编号']}".encode()).digest())
    if [row["编号"] for row in a] == [row["编号"] for row in b]:
        b = b[1:] + b[:1]
    path_a, path_b = outdir / "blind-form-01.csv", outdir / "blind-form-02.csv"
    write_csv(path_a, a)
    write_csv(path_b, b)
    return path_a, path_b


def read_csv(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(f"unexpected review columns in {path}")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"cannot read reviewer CSV: {exc}") from exc
    if len(rows) != 12 or len({row["编号"] for row in rows}) != 12:
        raise ValueError(f"expected exactly 12 unique scored items in {path}")
    return {row["编号"]: row for row in rows}

def compare(path_a, path_b, fixture=FIXTURE):
    if Path(path_a).resolve() == Path(path_b).resolve():
        raise ValueError("reviewer A and B must use separate independent submissions")
    a, b = read_csv(path_a), read_csv(path_b)
    expected = {row["编号"]: row for row in records(fixture)}
    if a.keys() != b.keys() or a.keys() != expected.keys():
        raise ValueError("the two reviewers must assess the same fixed item set")
    allowed_numeric = {str(i) for i in range(5)}
    dims = {}
    disagreements = set()
    both_complete = 0
    complete_submissions = 0
    unfilled_items = 0
    items_missing_rationale = 0
    missing_rationale_by_dimension = {dim: 0 for dim in DIMS}
    for table in (a, b):
        submission_complete = True
        for row in table.values():
            for column in RATING_COLUMNS.values():
                grade = row[column].strip().upper()
                if grade not in ALLOWED:
                    raise ValueError(f"invalid grade for {row['编号']}/{column}; use 0-4 or X, never a decimal")
                submission_complete = submission_complete and grade in ({"X"} | allowed_numeric)
            for dim, column in RATIONALE_COLUMNS.items():
                if not row[column].strip():
                    submission_complete = False
        complete_submissions += int(submission_complete)
    for item_id in expected:
        for field in IMMUTABLE_FIELDS:
            if a[item_id][field] != expected[item_id][field] or b[item_id][field] != expected[item_id][field]:
                raise ValueError(f"stimulus/evidence was edited: {item_id}/{field}")
        missing_rationale = [dim for dim, column in RATIONALE_COLUMNS.items()
                             if not a[item_id][column].strip() or not b[item_id][column].strip()]
        if missing_rationale:
            items_missing_rationale += 1
            for dim in missing_rationale:
                missing_rationale_by_dimension[dim] += 1
        item_complete = True
        for dim, column in RATING_COLUMNS.items():
            left, right = a[item_id][column].strip().upper(), b[item_id][column].strip().upper()
            if not left or not right:
                item_complete = False
            both_x = left == right == "X"
            one_x = (left == "X") != (right == "X") and bool(left and right)
            numeric = left in allowed_numeric and right in allowed_numeric
            gap = abs(int(left) - int(right)) if numeric else 0
            entry = dims.setdefault(dim, {"n_items": 0, "unfilled_ratings": 0,
                                          "both_X": 0, "one_X": 0, "numeric_pairs": 0,
                                          "exact_agreement": 0, "within_one_grade": 0,
                                          "gap_ge_2": 0, "numeric_absolute_error_sum": 0.0})
            if not left or not right:
                entry["unfilled_ratings"] += 1
                item_complete = False
                continue
            entry["n_items"] += 1
            entry["both_X"] += int(both_x)
            entry["one_X"] += int(one_x)
            entry["exact_agreement"] += int(left == right)
            entry["numeric_pairs"] += int(numeric)
            entry["within_one_grade"] += int(numeric and gap <= 1)
            entry["numeric_absolute_error_sum"] += gap if numeric else 0
            entry["gap_ge_2"] += int(numeric and gap >= 2)
            if one_x or (numeric and gap >= 2):
                disagreements.add(item_id)
        both_complete += int(item_complete)
    for entry in dims.values():
        n = entry["numeric_pairs"]
        entry["numeric_mean_absolute_error"] = round(entry["numeric_absolute_error_sum"] / n, 4) if n else None
        del entry["numeric_absolute_error_sum"]
    return {
        "scope": "descriptive agreement check only; synthetic stimuli, not project gold labels",
        "item_count": len(expected), "dialogue_count": len({row["对话"] for row in expected.values()}),
        "complete_submissions": complete_submissions,
        "complete_on_all_four_dimensions": both_complete,
        "items_missing_rationale": items_missing_rationale,
        "missing_rationale_by_dimension": missing_rationale_by_dimension,
        "dimensions": dims, "disagreements": sorted(disagreements),
        "interpretation": "Agreement does not establish construct validity or model qualification. X means insufficient evidence, not grade zero.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("outdir", type=Path)
    audit = subparsers.add_parser("compare")
    audit.add_argument("reviewer_a", type=Path)
    audit.add_argument("reviewer_b", type=Path)
    audit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "create":
        a, b = make_packet(args.outdir)
        print(json.dumps({"reviewer_a": str(a), "reviewer_b": str(b)}, ensure_ascii=False))
        return
    result = compare(args.reviewer_a, args.reviewer_b)
    with args.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({"out": str(args.out), "complete": result["complete_on_all_four_dimensions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()