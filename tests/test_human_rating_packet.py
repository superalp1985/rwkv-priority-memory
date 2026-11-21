import json
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HumanRatingPacketTest(unittest.TestCase):
    def test_blinded_options_match_the_actual_checkpoint_card_sets(self):
        key = json.loads((ROOT / "local/human-rating-key-v1.json").read_text())
        expected_text = {
            "checklist": "装行李的清单留在桌边直到周五。",
            "handoff": "交接时间是九点。",
            "coffee": "大厅咖啡让人失望。",
            "report-nine": "周报每周一九点到期。",
            "report-ten": "周报每周一十点到期。",
            "poster": "走廊海报印成蓝色。",
        }
        packet = (ROOT / "reports/human-rating-phase2-v1.md").read_text()
        sections = re.split(r"(?m)^## (\d{2}) ", packet)
        phase2 = {int(sections[i]): sections[i + 1] for i in range(1, len(sections), 2)}
        distinguishable = 0
        for item in key["cases"]:
            report = json.loads((ROOT / "reports" / item["report"]).read_text())
            case = next(row for row in report["cases"] if row["id"] == item["source_case"])
            checkpoint = next(row for row in case["rounds"] if row["turn"] == item["turn"])
            card_sets = set()
            for label in "ABC":
                arm = item[label]
                selected = checkpoint[arm]["selected"]
                card_sets.add(tuple(selected))
                expected = "无卡片" if not selected else expected_text[selected[0]]
                match = re.search(rf"(?m)^- {label}：(.*)$", phase2[item["phase2_case"]])
                if match is None:
                    raise AssertionError((item["phase2_case"], label))
                self.assertEqual(match.group(1), expected, (item["phase2_case"], label, arm))
            distinguishable += len(card_sets) > 1
        self.assertEqual(distinguishable, 2)

    def test_phase_one_snapshots_end_before_the_current_assistant_reply(self):
        packet = (ROOT / "reports/human-rating-phase1-v1.md").read_text()
        sections = re.split(r"(?m)^## (\d{2}) ", packet)
        cases = {int(sections[i]): sections[i + 1].split("**随手记住：**", 1)[0].strip()
                 for i in range(1, len(sections), 2)}
        self.assertEqual(len(cases), 7)
        expected_last_users = {
            1: "用户：九点可以。",
            2: "用户：九点可以。",
            3: "用户：对，暂时还是周一九点。",
            4: "用户：这张图用简单字体就好。",
            5: "用户：我今天午饭吃晚了一点。",
            6: "用户：九点可以。",
            7: "用户：九点可以。",
        }
        for case_id, expected in expected_last_users.items():
            transcript = cases[case_id]
            self.assertTrue(transcript.endswith(expected), case_id)
            self.assertNotRegex(transcript, r"\n助手：[^\n]*$")


if __name__ == "__main__":
    unittest.main()
