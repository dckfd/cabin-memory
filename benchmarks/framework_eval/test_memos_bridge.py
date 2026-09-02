from __future__ import annotations

import unittest

from benchmarks.framework_eval.bridges.memos_bridge import _perspective_messages, _safe_name


class MemOSBridgeTests(unittest.TestCase):
    def test_perspective_roles_keep_time_speaker_and_source(self) -> None:
        session = {
            "timestamp": "1:00 pm on 8 May, 2023",
            "messages": [
                {
                    "message_id": "D1:1",
                    "speaker": "Alice",
                    "content": "I started a new job.",
                },
                {
                    "message_id": "D1:2",
                    "speaker": "Bob",
                    "content": "Congratulations!",
                },
            ],
        }

        messages = _perspective_messages(session, "Alice")

        self.assertEqual([row["role"] for row in messages], ["user", "assistant"])
        self.assertIn("1:00 pm on 8 May, 2023", messages[0]["content"])
        self.assertIn("source=D1:1", messages[0]["content"])
        self.assertIn("Alice: I started a new job.", messages[0]["content"])

    def test_store_name_is_stable_and_path_safe(self) -> None:
        self.assertEqual(_safe_name("conv/30"), _safe_name("conv/30"))
        self.assertNotIn("/", _safe_name("conv/30"))


if __name__ == "__main__":
    unittest.main()
