"""Build an independently randomized, blinded reply-rating packet."""

from argparse import ArgumentParser
import json
from pathlib import Path
import random
import secrets
from typing import Any

from .capacity_calibration import ROOT


ARMS = ('native', 'reselect_equal', 'capacity_equal', 'capacity_context', 'window_context')
LETTERS = tuple('ABCDE')


def build(report: dict[str, Any], rng: Any = None) -> tuple[str, dict[str, Any]]:
    rng = rng or secrets.SystemRandom()
    if tuple(report.get('arms', ())) != ARMS:
        raise ValueError('unexpected holdout arms')
    panels = []
    key_panels = []
    for case in report['cases']:
        for checkpoint in case['checkpoints']:
            replies = checkpoint['replies']
            if set(replies) != set(ARMS):
                raise ValueError('checkpoint must contain exactly five arms')
            labels = list(LETTERS)
            rng.shuffle(labels)
            group_to_arm = dict(zip(labels, ARMS))
            arm_order = list(ARMS)
            rng.shuffle(arm_order)
            history = case['common_history'][:checkpoint['history_turns']]
            if len(history) != checkpoint['history_turns']:
                raise ValueError('checkpoint history length exceeds replay history')
            panel_id = f'P{len(panels) + 1:02d}'
            panels.append((panel_id, history, checkpoint['user'],
                           [(next(label for label, arm in group_to_arm.items() if arm == name),
                             replies[name]['text']) for name in arm_order]))
            key_panels.append(dict(panel=panel_id, source_case=case['row']['id'],
                                   kind=checkpoint['kind'], user=checkpoint['user'],
                                   group_to_arm=group_to_arm,
                                   expected_marker=checkpoint['expected_marker'],
                                   obsolete_marker=checkpoint['obsolete_marker'],
                                   archive_evidence=checkpoint['evidence']))
    rng.shuffle(panels)
    key_by_id = {entry['panel']: entry for entry in key_panels}
    key = dict(version='human-reply-capacity-holdout-key-v1',
               source_report='reports/capacity-holdout-v1-cpu.json', panels=[
                   key_by_id[panel_id] for panel_id, *_ in panels
               ])
    lines = [
        '# 匿名回复盲评 · 容量策略留出集',
        '',
        '请在不知道 A–E 对应策略的情况下独立评分。每个检查点是一次独立回复分叉；候选回复没有被继续写回后续历史，因此这不是多轮互动录像，也不代表真人对话结果。所有案例均为合成英文对话，候选卡、事项和更正关系由夹具提供。',
        '',
        '对每个检查点，分别评估自然度（1–5）、是否正确使用当前有效信息、是否沿用已失效信息、无关回复是否打扰、被要求精确原文时是否给出可核验引文。选择最合用回复，可选平局或都不合适；没有足够证据时标“无法判断”，并说明原因。不要猜策略身份。',
        '',
        '## 记录表',
        '',
        '| 检查点 | 最合用组（可平局/都不合适/无法判断） | 最关键理由 |',
        '|---|---|---|',
    ]
    for panel_id, *_ in panels:
        lines.append(f'| {panel_id} |  |  |')
    lines.extend(['', '## 匿名检查点', ''])
    for panel_id, history, user, responses in panels:
        lines.extend([f'### {panel_id}', '', '**此前对话**', '', '```text'])
        for prior_user, prior_assistant in history:
            lines.extend((f'User: {prior_user}', f'Assistant: {prior_assistant}', ''))
        lines.extend(('```', '', f'**当前用户发言**  {user}', '', '**匿名助手回复**', ''))
        for letter, text in responses:
            lines.extend((f'- **{letter}:** {text}', ''))
        lines.extend((
            '自然度 1–5：_____　最合用组：_____　平局/都不合适/无法判断：_____',
            '是否使用当前有效信息：_____　是否沿用失效信息：_____　无关时是否打扰：_____',
            '精确原文请求是否有可核验引用：_____　理由：________________________________',
            '',
        ))
    return '\n'.join(lines), key


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'reports/capacity-holdout-v1-cpu.json')
    parser.add_argument('--packet', type=Path, default=ROOT / 'reports/human-reply-capacity-holdout-v1.md')
    parser.add_argument('--key', type=Path, default=ROOT / 'local/human-reply-capacity-holdout-v1-key.json')
    args = parser.parse_args()
    report = json.loads(args.input.read_bytes())
    packet, key = build(report)
    args.packet.parent.mkdir(parents=True, exist_ok=True)
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.packet.write_text(packet + '\n', encoding='utf-8')
    args.key.write_text(json.dumps(key, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(packet=str(args.packet), key=str(args.key), panels=len(key['panels']),
                          comparisons=sum(len(p['group_to_arm']) for p in key['panels'])),
                      ensure_ascii=False))


if __name__ == '__main__':
    main()