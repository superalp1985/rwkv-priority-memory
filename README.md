# RWKV Priority Memory

主目标是在持续循环更新与替换/重建评估 state 之间，找出更贴近人的记忆和使用习惯的方式。评审看卡片留存只是中间证据；最终应以完整对话中的自然沿用、更正采纳、不相关时不打扰、按需回忆，以及评审自己的整体偏好为主。回答正确率是次指标。本项目目前只做了固定权重合成对话的选卡回放，未生成回答、未查原文、未收集真人评分。

## How it works

- **State comparison.** Replay the same dialogue with a read-only fork, a replacement state rebuilt from three completed exchanges plus the current user and valid cards, or a recurrent state updated by selected cards. The checkpoint is unchanged in all three.
- **Retention.** The pilot scores fixture-supplied candidate cards on four independent forks before the next reply, with a common 24-token card budget. It does not test extracting, attributing, deduplicating, or versioning memories from raw dialogue. Counts and timestamps are evidence for the model. Synthetic target sets are curator hypotheses, not human ratings.
- **Recall (design).** Keep the authorized original transcript independently of its score. Use live state ordinarily; search for precise wording or missing historical evidence only when needed. Replies and retrieval are not exercised by the retention pilot.

| Reference | Purpose |
|---|---|
| [Research and implementation plan](docs/PLAN.md) | Score policy, state and archive boundary, related work, pilot protocol |
| [Context appraisal](docs/APPRAISAL.md) | Rubric, evidence requirements and matched examples |
| [State-conditioned appraisal and upstream PR scope](docs/STATE-INERTIA.md) | Exact turn boundary, score-driven state update, official code, duplicate survey and CPU experiment |
| [Validation](docs/VALIDATION.md) | Reproducible probes, results and untested behavior |
| [Human rating pilot](docs/HUMAN-RATING-PILOT.md) | Synthetic memory-card pilot only; human preference and complete-reply comparisons remain unmeasured |
| [Policy proposal](configs/priority-policy-proposal.json) | Pilot defaults, not a runtime service configuration |
| [GitHub name and description](docs/GITHUB.md) | Repository creation copy |
| [Historical RWKV probe](docs/LEGACY-RWKV-VALIDATION.md) | Fixed-source CPU experiment, kept separate from the new policy |

## Run locally

From the repository root, with Python 3.10+ and SQLite FTS5 for the full suite:

```bash
python3 -m unittest discover -s tests -v
python3 -m src.priority
python3 -m src.journal
```

The standard-library probes exercise ranking and SQLite score history. An optional CPU test runs the pinned upstream x070 forward with random fixture weights and opaque token IDs. A separate `python -m probes.state_replay` command runs the pinned pretrained 0.1B model on synthetic text. [Validation](docs/VALIDATION.md) gives commands and separates mechanical checks, model readouts, and unmeasured human preferences. Keep private conversations, model weights and indexes out of the repository.
