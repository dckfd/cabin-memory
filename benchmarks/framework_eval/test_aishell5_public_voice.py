from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from .aishell5_public_voice import fuse_multichannel_rows, parse_textgrid


class Aishell5PublicVoiceTests(unittest.TestCase):
    def test_textgrid_parser_preserves_speaker_time_and_text(self):
        content = '''File type = "ooTextFile"
Object class = "TextGrid"
item []:
    item [1]:
        class = "IntervalTier"
        name = "P0247"
        xmin = 0
        xmax = 2
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 1
            text = ""
        intervals [2]:
            xmin = 1
            xmax = 2
            text = "导航去浦东机场。"
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "001" / "DX01C01.TextGrid"
            path.parent.mkdir()
            path.write_text(content, encoding="utf-8")
            rows = parse_textgrid(path)
        self.assertEqual(1, len(rows))
        self.assertEqual("001", rows[0].session)
        self.assertEqual("P0247", rows[0].speaker)
        self.assertEqual(1.0, rows[0].start)
        self.assertEqual("导航去浦东机场。", rows[0].text)

    def test_multichannel_fusion_selects_reference_free_medoid(self):
        hypotheses = {
            1: "导航去浦东机场",
            2: "导航去浦东机杨",
            3: "导航去浦东机场",
            4: "打开空调",
        }
        rows = [
            {
                "clip_id": f"sample-ch{channel}",
                "source_utterance_id": "sample",
                "source_channel": channel,
                "hypothesis": hypothesis,
                "reference": "这个字段不得用于选择",
            }
            for channel, hypothesis in hypotheses.items()
        ]
        fused = fuse_multichannel_rows(rows)
        self.assertEqual(1, len(fused))
        self.assertEqual("sample", fused[0]["clip_id"])
        self.assertEqual("导航去浦东机场", fused[0]["hypothesis"])
        self.assertEqual(1, fused[0]["selected_source_channel"])
        self.assertEqual("four-channel-transcript-medoid-v1", fused[0]["fusion_method"])


if __name__ == "__main__":
    unittest.main()
